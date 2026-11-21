/* Обучение трансформера с нуля на C: прямой и обратный проход, AdamW.
 * Архитектура (как у современных LLM): RMSNorm, RoPE, GQA-внимание, SwiGLU, связанные эмбеддинги. */
#include "corax.h"
#include "gemm.h"
#include "model.h"
#include "pool.h"
#include "train.h"
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <signal.h>
#include <sys/ioctl.h>

#define CT 256 /* токенов за раз в классификаторе (экономия памяти на логитах) */

typedef struct {
    Config c;
    int B, T, BT;
    size_t np;
    float *p, *g, *m, *v;
    Weights w, gw;
    float *xs, *xn1, *xn2, *xmid, *q, *att, *k, *vv, *rms1, *rms2, *lse, *h1, *h3, *xf, *rmsf;
    float *dx, *dxf, *dxn, *datt, *dq, *dk, *dv, *G, *dG, *dH1, *logits, *rowloss;
    float *rope_cos, *rope_sin, *ascr;
    int *inp, *tgt;
} Net;

/* ---------------- память ---------------- */
static size_t net_act_bytes(const Config *c, int B, int T) {
    size_t BT = (size_t)B * T, D = c->dim, F = c->hidden_dim, KV = cfg_kv_dim(c), L = c->n_layers, V = c->vocab_size;
    size_t per_layer = BT * (5 * D + 2 * KV + 2 * F + 2) + (size_t)B * c->n_heads * T;
    size_t scratch = BT * (5 * D + 2 * KV + 3 * F) + CT * V + BT * (D + 1);
    scratch += (size_t)pool_size() * (3 * (size_t)T * T + 2 * (D / c->n_heads) * (size_t)T);
    return 4 * (L * per_layer + BT * D + scratch);
}

static void net_init(Net *n, const Config *c, int B, int T, int with_opt) {
    memset(n, 0, sizeof(*n));
    n->c = *c; n->B = B; n->T = T; n->BT = B * T;
    size_t BT = n->BT, D = c->dim, F = c->hidden_dim, KV = cfg_kv_dim(c), L = c->n_layers, V = c->vocab_size;
    n->np = params_count(c);
    n->p = xcalloc(4 * n->np);
    n->g = xcalloc(4 * n->np);
    if (with_opt) { n->m = xcalloc(4 * n->np); n->v = xcalloc(4 * n->np); }
    weights_map(&n->w, c, n->p);
    weights_map(&n->gw, c, n->g);
    n->xs = xmalloc(4 * (L + 1) * BT * D);
    n->xn1 = xmalloc(4 * L * BT * D); n->xn2 = xmalloc(4 * L * BT * D);
    n->xmid = xmalloc(4 * L * BT * D); n->q = xmalloc(4 * L * BT * D); n->att = xmalloc(4 * L * BT * D);
    n->k = xmalloc(4 * L * BT * KV); n->vv = xmalloc(4 * L * BT * KV);
    n->rms1 = xmalloc(4 * L * BT); n->rms2 = xmalloc(4 * L * BT);
    n->lse = xmalloc(4 * L * B * c->n_heads * T);
    n->h1 = xmalloc(4 * L * BT * F); n->h3 = xmalloc(4 * L * BT * F);
    n->xf = xmalloc(4 * BT * D); n->rmsf = xmalloc(4 * BT);
    n->dx = xmalloc(4 * BT * D); n->dxf = xmalloc(4 * BT * D); n->dxn = xmalloc(4 * BT * D);
    n->datt = xmalloc(4 * BT * D); n->dq = xmalloc(4 * BT * D);
    n->dk = xmalloc(4 * BT * KV); n->dv = xmalloc(4 * BT * KV);
    n->G = xmalloc(4 * BT * F); n->dG = xmalloc(4 * BT * F); n->dH1 = xmalloc(4 * BT * F);
    n->logits = xmalloc(4 * CT * V);
    n->rowloss = xmalloc(4 * BT);
    int hd = cfg_head_dim(c);
    n->rope_cos = xmalloc(4 * T * (hd / 2)); n->rope_sin = xmalloc(4 * T * (hd / 2));
    for (int p = 0; p < T; p++)
        for (int i = 0; i < hd / 2; i++) {
            double f = pow((double)c->rope_theta, -2.0 * i / hd);
            n->rope_cos[p * (hd / 2) + i] = (float)cos(p * f);
            n->rope_sin[p * (hd / 2) + i] = (float)sin(p * f);
        }
    n->inp = xmalloc(sizeof(int) * BT); n->tgt = xmalloc(sizeof(int) * BT);
    n->ascr = xmalloc(4 * (size_t)pool_size() * (3 * (size_t)T * T + 2 * (size_t)hd * T));
}

static void net_init_weights(Net *n, uint64_t seed) {
    const Config *c = &n->c;
    uint64_t rng = seed * 0x9E3779B97F4A7C15ULL + 1;
    float std = 0.02f, std_res = 0.02f / sqrtf(2.0f * c->n_layers);
    float *p = n->p;
    for (int t = 0; t < T_COUNT; t++) {
        size_t cnt = tensor_numel(c, t);
        if (!tensor_is_matrix(t)) { for (size_t i = 0; i < cnt; i++) p[i] = 1.0f; }
        else {
            float s = (t == T_WO || t == T_W2) ? std_res : std;
            for (size_t i = 0; i < cnt; i++) p[i] = rng_normal(&rng) * s;
        }
        p += cnt;
    }
}

/* ---------------- элементарные слои ---------------- */
typedef struct { float *y, *rms; const float *x, *w; int D; float eps; } NormJob;
static void norm_fwd_fn(void *ctx, int t0, int t1) {
    NormJob *j = ctx;
    for (int t = t0; t < t1; t++) {
        const float *x = j->x + (size_t)t * j->D;
        float *y = j->y + (size_t)t * j->D, ss = 0;
        for (int i = 0; i < j->D; i++) ss += x[i] * x[i];
        float r = 1.0f / sqrtf(ss / j->D + j->eps);
        j->rms[t] = r;
        for (int i = 0; i < j->D; i++) y[i] = x[i] * r * j->w[i];
    }
}
static void norm_fwd(float *y, float *rms, const float *x, const float *w, int N, int D, float eps) {
    NormJob j = {y, rms, x, w, D, eps};
    pool_run(norm_fwd_fn, &j, N, 32);
}

/* dx += d(rmsnorm)/dx · dy ;  dw += Σ dy·x·r.  Параллельно по токенам, dw — через частичные суммы потоков. */
typedef struct { float *dx, *part; const float *dy, *x, *rms, *w; int D; } NormBJob;
static void norm_bwd_fn(void *ctx, int t0, int t1) {
    NormBJob *j = ctx;
    int D = j->D;
    float *dw = j->part + (size_t)pool_tid() * D;
    for (int t = t0; t < t1; t++) {
        const float *xt = j->x + (size_t)t * D, *dyt = j->dy + (size_t)t * D;
        float *dxt = j->dx + (size_t)t * D, r = j->rms[t], dot = 0;
        for (int i = 0; i < D; i++) { dot += dyt[i] * j->w[i] * xt[i]; dw[i] += dyt[i] * xt[i] * r; }
        float c = r * r * r * dot / D;
        for (int i = 0; i < D; i++) dxt[i] += dyt[i] * j->w[i] * r - c * xt[i];
    }
}
static float *g_normpart = NULL;
static void norm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *rms, const float *w, int N, int D) {
    int P = pool_size();
    if (!g_normpart) g_normpart = xmalloc(sizeof(float) * 64 * 8192); /* до 64 потоков */
    if (D > 8192) die("dim > 8192");
    memset(g_normpart, 0, sizeof(float) * (size_t)P * D);
    NormBJob j = {dx, g_normpart, dy, x, rms, w, D};
    pool_run(norm_bwd_fn, &j, N, 64);
    for (int p = 0; p < P; p++)
        for (int i = 0; i < D; i++) dw[i] += g_normpart[(size_t)p * D + i];
}

static void rope_apply(float *v, int N, int T, int nh, int hd, const float *cs, const float *sn, int inverse) {
    for (int t = 0; t < N; t++) {
        int pos = t % T;
        const float *c = cs + pos * (hd / 2), *s = sn + pos * (hd / 2);
        for (int h = 0; h < nh; h++) {
            float *p = v + ((size_t)t * nh + h) * hd;
            for (int i = 0; i < hd / 2; i++) {
                float a = p[2 * i], b = p[2 * i + 1], si = inverse ? -s[i] : s[i];
                p[2 * i] = a * c[i] - b * si;
                p[2 * i + 1] = a * si + b * c[i];
            }
        }
    }
}

typedef struct { Net *n; int l; } AttnJob;

/* Рабочая память внимания одного потока: 3 матрицы TxT + 2 матрицы hd x T. */
static float *attn_scratch(Net *n) {
    size_t T = n->T, hd = cfg_head_dim(&n->c);
    return n->ascr + (size_t)pool_tid() * (3 * T * T + 2 * hd * T);
}

/* P = softmax(Q K^T * scale) по строкам с причинной маской; хвосты блоков обнуляются. */
static void attn_probs(float *P, int T, float scale, const float *lse_in, float *lse_out) {
    for (int t = 0; t < T; t++) {
        float *row = P + (size_t)t * T;
        int ub = (t & ~3) + 4 < T ? (t & ~3) + 4 : T;
        if (lse_in) {
            float l = lse_in[t];
            for (int u = 0; u <= t; u++) row[u] = expf(row[u] * scale - l);
        } else {
            float mx = -1e30f;
            for (int u = 0; u <= t; u++) { row[u] *= scale; if (row[u] > mx) mx = row[u]; }
            float sum = 0;
            for (int u = 0; u <= t; u++) { row[u] = expf(row[u] - mx); sum += row[u]; }
            float inv = 1.0f / sum;
            for (int u = 0; u <= t; u++) row[u] *= inv;
            lse_out[t] = mx + logf(sum);
        }
        for (int u = t + 1; u < ub; u++) row[u] = 0;
    }
}

static void attn_fwd_fn(void *ctx, int i0, int i1) {
    AttnJob *j = ctx;
    Net *n = j->n;
    const Config *c = &n->c;
    int T = n->T, H = c->n_heads, hd = cfg_head_dim(c), D = c->dim, KV = cfg_kv_dim(c);
    int grp = H / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    float *P = attn_scratch(n), *VT = P + 3 * (size_t)T * T;
    for (int bh = i0; bh < i1; bh++) {
        int b = bh / H, h = bh % H, kvh = h / grp;
        size_t r0 = (size_t)j->l * n->BT + (size_t)b * T;
        const float *Q = n->q + r0 * D + h * hd, *K = n->k + r0 * KV + kvh * hd, *V = n->vv + r0 * KV + kvh * hd;
        gemm_nt_st(P, T, Q, D, K, KV, T, T, hd, 0, GM_LOWER);
        attn_probs(P, T, scale, NULL, n->lse + (((size_t)j->l * n->B + b) * H + h) * T);
        transpose_st(VT, T, V, KV, T, hd);
        gemm_nt_st(n->att + r0 * D + h * hd, D, P, T, VT, T, T, hd, T, 0, GM_KPRE);
    }
}

/* Обратный проход внимания (как во FlashAttention: вероятности пересчитываются, TxT не хранится между слоями). */
static void attn_bwd_fn(void *ctx, int i0, int i1) {
    AttnJob *j = ctx;
    Net *n = j->n;
    const Config *c = &n->c;
    int T = n->T, H = c->n_heads, KVH = c->n_kv_heads, hd = cfg_head_dim(c), D = c->dim, KV = cfg_kv_dim(c);
    int grp = H / KVH;
    float scale = 1.0f / sqrtf((float)hd);
    float *P = attn_scratch(n), *dS = P + (size_t)T * T, *TR = dS + (size_t)T * T;
    float *KT = TR + (size_t)T * T, *XT = KT + (size_t)hd * T;
    for (int bk = i0; bk < i1; bk++) {
        int b = bk / KVH, kvh = bk % KVH;
        size_t r0 = (size_t)j->l * n->BT + (size_t)b * T, g0 = (size_t)b * T;
        const float *K = n->k + r0 * KV + kvh * hd, *V = n->vv + r0 * KV + kvh * hd;
        float *dK = n->dk + g0 * KV + kvh * hd, *dV = n->dv + g0 * KV + kvh * hd;
        for (int t = 0; t < T; t++)
            for (int i = 0; i < hd; i++) dK[(size_t)t * KV + i] = dV[(size_t)t * KV + i] = 0;
        transpose_st(KT, T, K, KV, T, hd);
        for (int h = kvh * grp; h < (kvh + 1) * grp; h++) {
            const float *Q = n->q + r0 * D + h * hd, *O = n->att + r0 * D + h * hd;
            const float *dO = n->datt + g0 * D + h * hd;
            float *dQ = n->dq + g0 * D + h * hd;
            gemm_nt_st(P, T, Q, D, K, KV, T, T, hd, 0, GM_LOWER);
            attn_probs(P, T, scale, n->lse + (((size_t)j->l * n->B + b) * H + h) * T, NULL);
            gemm_nt_st(dS, T, dO, D, V, KV, T, T, hd, 0, GM_LOWER); /* dP */
            for (int t = 0; t < T; t++) {
                const float *o = O + (size_t)t * D, *d = dO + (size_t)t * D;
                float Dt = 0;
                for (int i = 0; i < hd; i++) Dt += d[i] * o[i];
                float *ds = dS + (size_t)t * T, *p = P + (size_t)t * T;
                int ub = (t & ~3) + 4 < T ? (t & ~3) + 4 : T;
                for (int u = 0; u <= t; u++) ds[u] = p[u] * (ds[u] - Dt) * scale;
                for (int u = t + 1; u < ub; u++) ds[u] = 0;
            }
            gemm_nt_st(dQ, D, dS, T, KT, T, T, hd, T, 0, GM_KPRE);       /* dQ = dS K */
            transpose_st(TR, T, dS, T, T, T);
            transpose_st(XT, T, Q, D, T, hd);
            gemm_nt_st(dK, KV, TR, T, XT, T, T, hd, T, 1, GM_KSUF);      /* dK += dS^T Q */
            transpose_st(TR, T, P, T, T, T);
            transpose_st(XT, T, dO, D, T, hd);
            gemm_nt_st(dV, KV, TR, T, XT, T, T, hd, T, 1, GM_KSUF);      /* dV += P^T dO */
        }
    }
}

typedef struct { float *G; const float *h1, *h3; float *dG, *dH1; } SwiTJob;
static void swi_fwd_fn(void *ctx, int i0, int i1) {
    SwiTJob *j = ctx;
    for (int i = i0; i < i1; i++) { float a = j->h1[i]; j->G[i] = a / (1.0f + expf(-a)) * j->h3[i]; }
}
static void swi_bwd_fn(void *ctx, int i0, int i1) {
    SwiTJob *j = ctx;
    for (int i = i0; i < i1; i++) {
        float a = j->h1[i], sg = 1.0f / (1.0f + expf(-a)), silu = a * sg, dg = j->dG[i];
        j->dH1[i] = dg * j->h3[i] * sg * (1.0f + a * (1.0f - sg));
        j->dG[i] = dg * silu; /* теперь это dH3 */
    }
}

/* ---------------- прямой проход ---------------- */
static void net_forward(Net *n) {
    const Config *c = &n->c;
    Weights *w = &n->w;
    int BT = n->BT, D = c->dim, F = c->hidden_dim, KV = cfg_kv_dim(c), hd = cfg_head_dim(c), L = c->n_layers;
    for (int t = 0; t < BT; t++) memcpy(n->xs + (size_t)t * D, w->emb + (size_t)n->inp[t] * D, 4 * D);
    for (int l = 0; l < L; l++) {
        size_t oD = (size_t)l * BT * D, oK = (size_t)l * BT * KV, oF = (size_t)l * BT * F;
        float *xin = n->xs + oD, *xout = n->xs + oD + (size_t)BT * D;
        norm_fwd(n->xn1 + oD, n->rms1 + (size_t)l * BT, xin, w->attn_norm + l * D, BT, D, c->norm_eps);
        gemm_nt(n->q + oD, n->xn1 + oD, w->wq + (size_t)l * D * D, BT, D, D, 0);
        gemm_nt(n->k + oK, n->xn1 + oD, w->wk + (size_t)l * KV * D, BT, KV, D, 0);
        gemm_nt(n->vv + oK, n->xn1 + oD, w->wv + (size_t)l * KV * D, BT, KV, D, 0);
        rope_apply(n->q + oD, BT, n->T, c->n_heads, hd, n->rope_cos, n->rope_sin, 0);
        rope_apply(n->k + oK, BT, n->T, c->n_kv_heads, hd, n->rope_cos, n->rope_sin, 0);
        AttnJob aj = {n, l};
        pool_run(attn_fwd_fn, &aj, n->B * c->n_heads, 1);
        memcpy(n->xmid + oD, xin, 4 * (size_t)BT * D);
        gemm_nt(n->xmid + oD, n->att + oD, w->wo + (size_t)l * D * D, BT, D, D, 1);
        norm_fwd(n->xn2 + oD, n->rms2 + (size_t)l * BT, n->xmid + oD, w->ffn_norm + l * D, BT, D, c->norm_eps);
        gemm_nt(n->h1 + oF, n->xn2 + oD, w->w1 + (size_t)l * F * D, BT, F, D, 0);
        gemm_nt(n->h3 + oF, n->xn2 + oD, w->w3 + (size_t)l * F * D, BT, F, D, 0);
        SwiTJob sj = {n->G, n->h1 + oF, n->h3 + oF, NULL, NULL};
        pool_run(swi_fwd_fn, &sj, BT * F, 4096);
        memcpy(xout, n->xmid + oD, 4 * (size_t)BT * D);
        gemm_nt(xout, n->G, w->w2 + (size_t)l * D * F, BT, D, F, 1);
    }
    norm_fwd(n->xf, n->rmsf, n->xs + (size_t)L * BT * D, w->final_norm, BT, D, c->norm_eps);
}

/* ---------------- классификатор + loss ---------------- */
typedef struct { Net *n; int c0; float scale; int backward; } LossJob;
static void loss_fn(void *ctx, int r0, int r1) {
    LossJob *j = ctx;
    Net *n = j->n;
    int V = n->c.vocab_size;
    for (int r = r0; r < r1; r++) {
        float *lg = n->logits + (size_t)r * V;
        int tg = n->tgt[j->c0 + r];
        if (tg < 0) {
            n->rowloss[j->c0 + r] = 0;
            if (j->backward) memset(lg, 0, 4 * V);
            continue;
        }
        float mx = lg[0];
        for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
        float sum = 0, ltg = lg[tg];
        if (j->backward) { /* exp считаем один раз и сразу оставляем в буфере */
            for (int i = 0; i < V; i++) { float e = expf(lg[i] - mx); lg[i] = e; sum += e; }
            float k = j->scale / sum;
            for (int i = 0; i < V; i++) lg[i] *= k;
            lg[tg] -= j->scale;
        } else {
            for (int i = 0; i < V; i++) sum += expf(lg[i] - mx);
        }
        n->rowloss[j->c0 + r] = mx + logf(sum) - ltg;
    }
}

/* Возвращает сумму loss по валидным целям. При backward пишет dxf и копит dEmb. */
static double net_classifier(Net *n, int backward, float scale) {
    const Config *c = &n->c;
    int BT = n->BT, D = c->dim, V = c->vocab_size;
    for (int c0 = 0; c0 < BT; c0 += CT) {
        int ct = BT - c0 < CT ? BT - c0 : CT;
        gemm_nt(n->logits, n->xf + (size_t)c0 * D, n->w.emb, ct, V, D, 0);
        LossJob lj = {n, c0, scale, backward};
        pool_run(loss_fn, &lj, ct, 4);
        if (backward) {
            gemm(n->dxf + (size_t)c0 * D, n->logits, n->w.emb, ct, D, V, 0, 0, 1);       /* dxf = dlogits · E */
            gemm(n->gw.emb, n->logits, n->xf + (size_t)c0 * D, V, D, ct, 1, 1, 1);       /* dE += dlogits^T · xf */
        }
    }
    double s = 0;
    for (int t = 0; t < BT; t++) s += n->rowloss[t];
    return s;
}

/* dW[out,in] += dy^T · x   (транспонирование встроено в упаковку gemm) */
static void linear_bwd_w(Net *n, float *dW, const float *dy, const float *x, int out, int in) {
    gemm(dW, dy, x, out, in, n->BT, 1, 1, 1);
}
/* dx[BT,in] (+)= dy[BT,out] · W[out,in] */
static void linear_bwd_x(Net *n, float *dx, const float *dy, const float *W, int out, int in, int acc) {
    gemm(dx, dy, W, n->BT, in, out, acc, 0, 1);
}

/* ---------------- обратный проход ---------------- */
static void net_backward(Net *n) {
    const Config *c = &n->c;
    Weights *w = &n->w, *gw = &n->gw;
    int BT = n->BT, D = c->dim, F = c->hidden_dim, KV = cfg_kv_dim(c), hd = cfg_head_dim(c), L = c->n_layers;
    memset(n->dx, 0, 4 * (size_t)BT * D);
    norm_bwd(n->dx, gw->final_norm, n->dxf, n->xs + (size_t)L * BT * D, n->rmsf, w->final_norm, BT, D);
    for (int l = L - 1; l >= 0; l--) {
        size_t oD = (size_t)l * BT * D, oK = (size_t)l * BT * KV, oF = (size_t)l * BT * F;
        float *xin = n->xs + oD;
        /* --- FFN --- */
        SwiTJob sj = {n->G, n->h1 + oF, n->h3 + oF, n->dG, n->dH1};
        pool_run(swi_fwd_fn, &sj, BT * F, 4096);
        linear_bwd_x(n, n->dG, n->dx, w->w2 + (size_t)l * D * F, D, F, 0);
        linear_bwd_w(n, gw->w2 + (size_t)l * D * F, n->dx, n->G, D, F);
        pool_run(swi_bwd_fn, &sj, BT * F, 4096);
        linear_bwd_x(n, n->dxn, n->dH1, w->w1 + (size_t)l * F * D, F, D, 0);
        linear_bwd_x(n, n->dxn, n->dG, w->w3 + (size_t)l * F * D, F, D, 1);
        linear_bwd_w(n, gw->w1 + (size_t)l * F * D, n->dH1, n->xn2 + oD, F, D);
        linear_bwd_w(n, gw->w3 + (size_t)l * F * D, n->dG, n->xn2 + oD, F, D);
        norm_bwd(n->dx, gw->ffn_norm + l * D, n->dxn, n->xmid + oD, n->rms2 + (size_t)l * BT, w->ffn_norm + l * D, BT, D);
        /* --- внимание --- */
        linear_bwd_x(n, n->datt, n->dx, w->wo + (size_t)l * D * D, D, D, 0);
        linear_bwd_w(n, gw->wo + (size_t)l * D * D, n->dx, n->att + oD, D, D);
        AttnJob aj = {n, l};
        pool_run(attn_bwd_fn, &aj, n->B * c->n_kv_heads, 1);
        rope_apply(n->dq, BT, n->T, c->n_heads, hd, n->rope_cos, n->rope_sin, 1);
        rope_apply(n->dk, BT, n->T, c->n_kv_heads, hd, n->rope_cos, n->rope_sin, 1);
        linear_bwd_x(n, n->dxn, n->dq, w->wq + (size_t)l * D * D, D, D, 0);
        linear_bwd_x(n, n->dxn, n->dk, w->wk + (size_t)l * KV * D, KV, D, 1);
        linear_bwd_x(n, n->dxn, n->dv, w->wv + (size_t)l * KV * D, KV, D, 1);
        linear_bwd_w(n, gw->wq + (size_t)l * D * D, n->dq, n->xn1 + oD, D, D);
        linear_bwd_w(n, gw->wk + (size_t)l * KV * D, n->dk, n->xn1 + oD, KV, D);
        linear_bwd_w(n, gw->wv + (size_t)l * KV * D, n->dv, n->xn1 + oD, KV, D);
        norm_bwd(n->dx, gw->attn_norm + l * D, n->dxn, xin, n->rms1 + (size_t)l * BT, w->attn_norm + l * D, BT, D);
        (void)oK;
    }
    for (int t = 0; t < BT; t++) {
        float *ge = gw->emb + (size_t)n->inp[t] * D;
        const float *d = n->dx + (size_t)t * D;
        for (int i = 0; i < D; i++) ge[i] += d[i];
    }
}

/* ---------------- AdamW ---------------- */
typedef struct { float *p, *g, *m, *v; float lr, b1, b2, eps, wd, bc1, bc2, gscale; } AdamJob;
static void adam_fn(void *ctx, int i0, int i1) {
    AdamJob *j = ctx;
    for (int i = i0; i < i1; i++) {
        float g = j->g[i] * j->gscale;
        float m = j->m[i] = j->b1 * j->m[i] + (1 - j->b1) * g;
        float v = j->v[i] = j->b2 * j->v[i] + (1 - j->b2) * g * g;
        float mh = m / j->bc1, vh = v / j->bc2;
        j->p[i] -= j->lr * (mh / (sqrtf(vh) + j->eps) + j->wd * j->p[i]);
    }
}

typedef struct { const float *g; double *part; int nparts; size_t n; } NormSqJob;
static void normsq_fn(void *ctx, int i0, int i1) {
    NormSqJob *j = ctx;
    for (int b = i0; b < i1; b++) {
        size_t s = j->n * b / j->nparts, e = j->n * (b + 1) / j->nparts;
        double acc = 0;
        for (size_t i = s; i < e; i++) acc += (double)j->g[i] * j->g[i];
        j->part[b] = acc;
    }
}

static float grad_norm(Net *n) {
    double part[256];
    NormSqJob j = {n->g, part, 256, n->np};
    pool_run(normsq_fn, &j, 256, 4);
    double s = 0;
    for (int i = 0; i < 256; i++) s += part[i];
    return (float)sqrt(s);
}

static void adam_step(Net *n, int step, float lr, const TrainOpts *o, float gscale) {
    float bc1 = 1 - powf(o->beta1, step), bc2 = 1 - powf(o->beta2, step);
    size_t off = 0;
    for (int t = 0; t < T_COUNT; t++) {
        size_t cnt = tensor_numel(&n->c, t);
        AdamJob j = {n->p + off, n->g + off, n->m + off, n->v + off, lr, o->beta1, o->beta2, 1e-8f,
                     tensor_is_matrix(t) ? o->weight_decay : 0.0f, bc1, bc2, gscale};
        pool_run(adam_fn, &j, (int)cnt, 8192);
        off += cnt;
    }
}

/* ---------------- данные ---------------- */
typedef struct {
    const uint16_t *tok; size_t n;
    const uint8_t *mask;
    const uint32_t *idx; size_t nidx;
    void *maps[3]; size_t sizes[3];
} Data;

static const void *map_file(const char *path, size_t *size, int required) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) { if (required) die("не найден файл данных %s", path); return NULL; }
    struct stat st;
    fstat(fd, &st);
    *size = (size_t)st.st_size;
    if (*size == 0) { close(fd); if (required) die("пустой файл %s", path); return NULL; }
    void *p = mmap(NULL, *size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) die("mmap %s", path);
    return p;
}

static void data_open(Data *d, const char *path) {
    memset(d, 0, sizeof(*d));
    char buf[1100];
    size_t len = strlen(path);
    int is_bin = len > 4 && strcmp(path + len - 4, ".bin") == 0;
    if (is_bin) snprintf(buf, sizeof buf, "%s", path);
    else snprintf(buf, sizeof buf, "%s.bin", path);
    size_t sz;
    d->tok = map_file(buf, &sz, 1);
    d->n = sz / 2;
    char pre[1024];
    snprintf(pre, sizeof pre, "%.*s", is_bin ? (int)len - 4 : (int)len, path);
    snprintf(buf, sizeof buf, "%s.mask", pre);
    size_t msz = 0, isz = 0;
    d->mask = map_file(buf, &msz, 0);
    snprintf(buf, sizeof buf, "%s.idx", pre);
    d->idx = map_file(buf, &isz, 0);
    d->nidx = isz / 4;
    if (d->mask && msz != d->n) die("%s: размер маски не совпадает с токенами", pre);
}

static void data_batch(const Data *d, int B, int T, int V, int *inp, int *tgt, uint64_t *rng) {
    if (d->n < (size_t)T + 2) die("данных слишком мало: %zu токенов, нужно больше %d", d->n, T + 1);
    for (int b = 0; b < B; b++) {
        size_t start;
        if (d->idx && d->nidx) {
            start = d->idx[rng_u64(rng) % d->nidx];
            if (start + T + 1 > d->n) start = d->n - T - 1;
        } else {
            start = rng_u64(rng) % (d->n - T - 1);
        }
        for (int t = 0; t < T; t++) {
            int a = d->tok[start + t], y = d->tok[start + t + 1];
            inp[b * T + t] = a < V ? a : 0;
            tgt[b * T + t] = (y < V && (!d->mask || d->mask[start + t + 1])) ? y : -1;
        }
    }
}

/* ---------------- файлы ---------------- */
static uint8_t *read_all(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *b = xmalloc(n > 0 ? n : 1);
    if (n > 0 && fread(b, 1, n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
    fclose(f);
    *size = (size_t)n;
    return b;
}

static void write_or_die(FILE *f, const void *p, size_t n, const char *path) {
    if (n && fwrite(p, 1, n, f) != n) die("запись %s (кончилось место?)", path);
}

void save_model_f32(const char *path, const Config *c, const float *params, const uint8_t *tok, size_t tok_size,
                    int64_t steps, int64_t tokens) {
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    ensure_parent_dir(path);
    FILE *f = fopen(tmp, "wb");
    if (!f) die("не создать %s", tmp);
    FileHeader h;
    config_to_header(c, &h);
    h.qtype = QT_F32;
    h.train_steps = steps; h.train_tokens = tokens;
    size_t wbytes = params_count(c) * 4;
    h.tok_offset = tok ? (int64_t)(sizeof h + wbytes) : 0;
    h.tok_size = tok ? (int64_t)tok_size : 0;
    write_or_die(f, &h, sizeof h, tmp);
    write_or_die(f, params, wbytes, tmp);
    if (tok) write_or_die(f, tok, tok_size, tmp);
    if (fclose(f) != 0) die("запись %s", tmp);
    if (rename(tmp, path) != 0) die("rename %s", path);
}

static void save_opt(const char *path, Net *n, int64_t step, int64_t tokens, uint64_t rng) {
    char p[1100], tmp[1110];
    snprintf(p, sizeof p, "%s.opt", path);
    snprintf(tmp, sizeof tmp, "%s.tmp", p);
    FILE *f = fopen(tmp, "wb");
    if (!f) die("не создать %s", tmp);
    uint32_t magic = CRX_OPT_MAGIC;
    uint64_t np = n->np;
    write_or_die(f, &magic, 4, tmp);
    write_or_die(f, &np, 8, tmp);
    write_or_die(f, &step, 8, tmp);
    write_or_die(f, &tokens, 8, tmp);
    write_or_die(f, &rng, 8, tmp);
    write_or_die(f, n->m, 4 * n->np, tmp);
    write_or_die(f, n->v, 4 * n->np, tmp);
    if (fclose(f) != 0) die("запись %s", tmp);
    if (rename(tmp, p) != 0) die("rename %s", p);
}

static int load_opt(const char *path, Net *n, int64_t *step, int64_t *tokens, uint64_t *rng) {
    char p[1100];
    snprintf(p, sizeof p, "%s.opt", path);
    FILE *f = fopen(p, "rb");
    if (!f) return 0;
    uint32_t magic; uint64_t np;
    int ok = fread(&magic, 4, 1, f) == 1 && magic == CRX_OPT_MAGIC && fread(&np, 8, 1, f) == 1 && np == n->np &&
             fread(step, 8, 1, f) == 1 && fread(tokens, 8, 1, f) == 1 && fread(rng, 8, 1, f) == 1 &&
             fread(n->m, 4, n->np, f) == n->np && fread(n->v, 4, n->np, f) == n->np;
    fclose(f);
    return ok;
}

/* ---------------- генерация примера во время обучения ---------------- */
static void emit_flat(const uint8_t *p, int n, void *ud) {
    (void)ud;
    for (int i = 0; i < n; i++) putchar(p[i] == '\n' ? ' ' : p[i]);
}

static void print_sample(Net *n, const Tokenizer *tok, const char *prompt, int ntok, uint64_t *rng) {
    Model m;
    model_from_f32(&m, &n->c, n->p);
    State s;
    state_init(&s, &n->c, 0);
    int ids[512];
    int np = 0;
    ids[np++] = tok->special[SP_BOS];
    if (prompt && *prompt) np += tok_encode(tok, prompt, ids + np, 256);
    Sampler sp;
    sampler_default(&sp);
    sp.temperature = 0.8f;
    cprintf("\033[2m  ┌ пример: \033[0m");
    for (int i = 1; i < np; i++) { int l; const uint8_t *p = tok_piece(tok, ids[i], &l); fwrite(p, 1, l, stdout); }
    int pos = 0, cur = ids[0];
    Utf8Stream us = {0};
    for (; pos < s.seq_len - 1 && pos < np + ntok; pos++) {
        float *lg = forward(&m, &s, cur, pos);
        if (pos + 1 < np) { cur = ids[pos + 1]; continue; }
        cur = sample_token(&sp, rng, lg, n->c.vocab_size, NULL, 0);
        if (cur == tok->special[SP_EOS]) break;
        if (tok_is_special(tok, cur)) continue;
        int l;
        const uint8_t *p = tok_piece(tok, cur, &l);
        utf8_put(&us, p, l, emit_flat, NULL);
    }
    cprintf("\033[0m\n");
    fflush(stdout);
    state_free(&s);
    model_free(&m);
}

/* ---------------- главный цикл ---------------- */
static volatile sig_atomic_t g_stop = 0;
static void on_stop(int sig) {
    (void)sig;
    if (g_stop) _exit(130); /* второй Ctrl+C — выйти сразу */
    g_stop = 1;
    const char msg[] = "\n  ⏸ сохраняю после текущего шага… (ещё раз Ctrl+C — выйти без сохранения)\n";
    ssize_t r = write(STDOUT_FILENO, msg, sizeof msg - 1);
    (void)r;
}

static int term_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 10) return ws.ws_col;
    return 80;
}

/* Обрезает строку с ANSI-кодами до max видимых символов (UTF-8), чтобы она не переносилась. */
static void fit_line(char *s, int max) {
    int vis = 0;
    for (char *p = s; *p;) {
        if (*p == 0x1b && p[1] == '[') {
            p += 2;
            while (*p && !((*p >= 'A' && *p <= 'Z') || (*p >= 'a' && *p <= 'z'))) p++;
            if (*p) p++;
            continue;
        }
        if (vis >= max) { strcpy(p, "\033[0m"); return; }
        unsigned char ch = (unsigned char)*p;
        p += ch < 0x80 ? 1 : (ch & 0xE0) == 0xC0 ? 2 : (ch & 0xF0) == 0xE0 ? 3 : 4;
        vis++;
    }
}

static void live_clear(int live) {
    if (live) { cprintf("\r\033[K"); fflush(stdout); }
}
void train_opts_default(TrainOpts *o) {
    memset(o, 0, sizeof(*o));
    o->preset = "nano";
    o->batch = 0; o->seq = 0; o->accum = 1;
    o->steps = 2000; o->warmup = -1;
    o->lr = -1; o->min_lr_frac = 0.1f;
    o->weight_decay = 0.1f; o->beta1 = 0.9f; o->beta2 = 0.95f; o->clip = 1.0f;
    o->eval_every = 200; o->eval_iters = 20;
    o->save_every = 200; o->sample_every = 200; o->log_every = 10;
    o->seed = 1337;
    o->out = "models/base.bin";
}

static void fmt_time(double s, char *buf, size_t n) {
    if (s < 0 || s > 1e8) { snprintf(buf, n, "--"); return; }
    int h = (int)(s / 3600), m = (int)(s / 60) % 60, sec = (int)s % 60;
    if (h > 0) snprintf(buf, n, "%dч%02dм", h, m);
    else snprintf(buf, n, "%dм%02dс", m, sec);
}

static double eval_loss(Net *n, const Data *d, int iters, int V) {
    uint64_t rng = 0xC0FFEE;
    double s = 0;
    long cnt = 0;
    for (int i = 0; i < iters; i++) {
        data_batch(d, n->B, n->T, V, n->inp, n->tgt, &rng);
        net_forward(n);
        s += net_classifier(n, 0, 0);
        for (int t = 0; t < n->BT; t++) cnt += n->tgt[t] >= 0;
    }
    return cnt ? s / cnt : 0;
}

int train_run(TrainOpts *o) {
    Config c;
    uint8_t *tokblob = NULL;
    size_t toksize = 0;
    float *init = NULL;
    int64_t start_step = 0, tokens_seen = 0, base_steps = 0;
    uint64_t rng = o->seed * 0x9E3779B97F4A7C15ULL + 7;
    const char *src = o->resume ? o->out : o->init;
    ModelFile mf;
    int have_mf = 0;

    if (src) {
        model_open(&mf, src);
        have_mf = 1;
        if (mf.h.qtype != QT_F32) die("%s квантована — для обучения нужен f32-файл (до quantize)", src);
        c = mf.c;
        init = (float *)(mf.data + sizeof(FileHeader));
        if (mf.h.tok_offset > 0) {
            toksize = (size_t)mf.h.tok_size;
            tokblob = xmalloc(toksize);
            memcpy(tokblob, mf.data + mf.h.tok_offset, toksize);
        }
        base_steps = mf.h.train_steps;
        tokens_seen = mf.h.train_tokens;
    } else {
        if (!preset_get(o->preset, &c)) die("неизвестный пресет '%s'", o->preset);
    }
    if (o->tokenizer) {
        free(tokblob);
        tokblob = read_all(o->tokenizer, &toksize);
        if (!tokblob) die("не прочитать токенизатор %s", o->tokenizer);
    }
    if (!tokblob) die("нужен токенизатор: --tokenizer data/tokenizer.bin");
    Tokenizer tok;
    if (!tok_load(&tok, tokblob, toksize)) die("токенизатор повреждён");
    if (!src) c.vocab_size = tok.vocab_size;
    else if (c.vocab_size != tok.vocab_size) die("словарь модели (%d) и токенизатора (%d) не совпадают", c.vocab_size, tok.vocab_size);
    if (o->seq > 0 && !src) c.seq_len = o->seq;
    char err[128];
    if (!config_validate(&c, err, sizeof err)) die("конфиг: %s", err);

    int T = o->seq > 0 ? o->seq : c.seq_len;
    if (T > c.seq_len) T = c.seq_len;
    int B = o->batch > 0 ? o->batch : (2048 / T > 0 ? 2048 / T : 1);
    if (o->lr <= 0) o->lr = o->init && !o->resume ? 2e-4f : (c.dim <= 384 ? 1e-3f : 6e-4f);
    if (o->warmup < 0) o->warmup = o->steps / 20 < 200 ? o->steps / 20 : 200;

    Data train, val;
    data_open(&train, o->data);
    int have_val = 0;
    if (o->val) { data_open(&val, o->val); have_val = 1; }

    double act_mb = net_act_bytes(&c, B, T) / 1048576.0, par_mb = params_count(&c) * 16.0 / 1048576.0;
    cprintf("\033[1;35m◆ Corax LM\033[0m — %s\n", o->init && !o->resume ? "дообучение" : o->resume ? "продолжение обучения" : "обучение с нуля");
    config_print(&c, stdout);
    cprintf("  данные: %s (%.2f млн токенов%s)\n", o->data, train.n / 1e6, train.mask ? ", диалоги с маской" : "");
    cprintf("  батч %d x %d токенов x накопление %d = %d токенов/шаг, шагов %d, lr %.2g\n",
           B, T, o->accum, B * T * o->accum, o->steps, o->lr);
    cprintf("  память: ~%.0f МБ (веса+оптимизатор %.0f, активации %.0f), потоков %d (%s)\n",
           act_mb + par_mb, par_mb, act_mb, pool_size(), pool_cpu_info());
    fflush(stdout);

    Net n;
    net_init(&n, &c, B, T, 1);
    if (init) memcpy(n.p, init, 4 * n.np);
    else net_init_weights(&n, o->seed);
    if (have_mf) model_close(&mf);
    if (o->resume) {
        if (load_opt(o->out, &n, &start_step, &tokens_seen, &rng)) {
            cprintf("  возобновлено с шага %lld\n", (long long)start_step);
            base_steps -= start_step;
        }
        else cprintf("  состояние оптимизатора не найдено — начинаю оптимизатор заново\n");
    }
    if (start_step >= o->steps) { cprintf("уже обучено %lld шагов (>= --steps)\n", (long long)start_step); return 0; }

    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_stop;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);
    int live = g_color && isatty(STDOUT_FILENO);
    cprintf("  \033[2mCtrl+C — пауза с сохранением; продолжить: та же команда с --resume (или тот же скрипт)\033[0m\n");
    fflush(stdout);

    double t_start = now_sec(), t_win = t_start, t_draw = 0;
    int64_t tok_win = 0, done_tokens = 0;
    double loss_ema = -1, tps = 0;
    int64_t s1 = start_step;
    for (int64_t step = start_step; step < o->steps && !g_stop; step++) {
        float lr;
        if (step < o->warmup) lr = o->lr * (step + 1) / o->warmup;
        else {
            float pr = (float)(step - o->warmup) / (float)(o->steps - o->warmup > 0 ? o->steps - o->warmup : 1);
            lr = o->lr * (o->min_lr_frac + (1 - o->min_lr_frac) * 0.5f * (1 + cosf(3.14159265f * pr)));
        }
        memset(n.g, 0, 4 * n.np);
        double loss_sum = 0;
        long valid = 0;
        for (int a = 0; a < o->accum; a++) {
            data_batch(&train, B, T, c.vocab_size, n.inp, n.tgt, &rng);
            int nv = 0;
            for (int t = 0; t < n.BT; t++) nv += n.tgt[t] >= 0;
            if (nv == 0) continue;
            net_forward(&n);
            loss_sum += net_classifier(&n, 1, 1.0f / (nv * o->accum));
            valid += nv;
            net_backward(&n);
        }
        float gn = grad_norm(&n);
        float gscale = (o->clip > 0 && gn > o->clip) ? o->clip / gn : 1.0f;
        if (isfinite(gn)) adam_step(&n, (int)(step + 1), lr, o, gscale);
        else { live_clear(live); cprintf("\033[33m  шаг %lld: градиент NaN/Inf — пропущен\033[0m\n", (long long)step + 1); }
        double loss = valid ? loss_sum / valid : 0;
        loss_ema = loss_ema < 0 ? loss : 0.95 * loss_ema + 0.05 * loss;
        tokens_seen += (int64_t)B * T * o->accum;
        done_tokens += (int64_t)B * T * o->accum;
        s1 = step + 1;
        double now = now_sec();
        if (now - t_win > 2.0 || s1 == start_step + 1) { /* скорость по окну ~2 с */
            tps = (done_tokens - tok_win) / (now - t_win + 1e-9);
            t_win = now; tok_win = done_tokens;
        }
        double per_step = (now - t_start) / (double)(s1 - start_step);
        char eta[32];
        fmt_time(per_step * (o->steps - s1), eta, sizeof eta);
        if (live) {
            if (now - t_draw > 0.2 || s1 == o->steps) {
                t_draw = now;
                int cols = term_cols();
                int bw = cols >= 80 ? 16 : cols >= 56 ? 10 : 6;
                char bar[96] = "", line[512];
                int fill = (int)((double)bw * s1 / o->steps + 0.5), bl = 0;
                for (int i = 0; i < bw; i++) bl += snprintf(bar + bl, sizeof bar - bl, "%s", i < fill ? "━" : "─");
                if (cols >= 64)
                    snprintf(line, sizeof line, "  \033[38;5;141m%s\033[0m %lld/%d · loss \033[1;36m%.3f\033[0m · \033[32m%.0f ток/с\033[0m · осталось %s",
                             bar, (long long)s1, o->steps, loss_ema, tps, eta);
                else
                    snprintf(line, sizeof line, " \033[38;5;141m%s\033[0m %lld/%d \033[1;36m%.2f\033[0m \033[32m%.0fт/с\033[0m %s",
                             bar, (long long)s1, o->steps, loss_ema, tps, eta);
                fit_line(line, cols - 1);
                cprintf("\r\033[K%s", line);
                fflush(stdout);
            }
        } else if (s1 % o->log_every == 0 || s1 == o->steps || s1 == start_step + 1) {
            cprintf("  шаг %lld/%d  loss %.4f (ср %.3f)  lr %.1e  |g| %.2f  %.0f ток/с  осталось %s\n",
                    (long long)s1, o->steps, loss, loss_ema, lr, gn, tps, eta);
            fflush(stdout);
        }
        if (have_val && (s1 % o->eval_every == 0 || s1 == o->steps)) {
            double vl = eval_loss(&n, &val, o->eval_iters, c.vocab_size);
            live_clear(live);
            cprintf("  \033[1;33m▸\033[0m шаг %lld · loss %.3f · \033[1;33mвалидация %.3f, перплексия %.1f\033[0m · %.0f ток/с\n",
                    (long long)s1, loss_ema, vl, exp(vl), tps);
        }
        if (o->sample_every > 0 && (s1 % o->sample_every == 0 || s1 == o->steps)) {
            live_clear(live);
            print_sample(&n, &tok, o->sample_prompt, 48, &rng);
        }
        if (s1 % o->save_every == 0 || s1 == o->steps) {
            save_model_f32(o->out, &c, n.p, tokblob, toksize, base_steps + s1, tokens_seen);
            save_opt(o->out, &n, s1, tokens_seen, rng);
            if (!live) { cprintf("  \033[2m💾 сохранено: %s\033[0m\n", o->out); fflush(stdout); }
        }
    }
    live_clear(live);
    if (g_stop) {
        if (s1 > start_step) {
            save_model_f32(o->out, &c, n.p, tokblob, toksize, base_steps + s1, tokens_seen);
            save_opt(o->out, &n, s1, tokens_seen, rng);
        }
        cprintf("\033[1;33m⏸ пауза на шаге %lld/%d — всё сохранено в %s\033[0m\n", (long long)s1, o->steps, o->out);
        cprintf("  продолжить: та же команда с --resume (quickstart.sh продолжит сам)\n");
        cprintf("  попробовать уже сейчас: ./corax gen -m %s -p \"Однажды\"\n", o->out);
        fflush(stdout);
        tok_free(&tok);
        free(tokblob);
        return 130;
    }
    char el[32];
    fmt_time(now_sec() - t_start, el, sizeof el);
    cprintf("\033[1;32m✔ готово за %s. Модель: %s\033[0m\n", el, o->out);
    tok_free(&tok);
    free(tokblob);
    return 0;
}

/* ---------------- тест градиентов против PyTorch ---------------- */
int train_gradtest(const char *model_path, const char *batch_path, const char *out_path) {
    ModelFile mf;
    model_open(&mf, model_path);
    size_t bsz;
    uint8_t *bb = read_all(batch_path, &bsz);
    if (!bb) die("батч %s", batch_path);
    int32_t B, T;
    memcpy(&B, bb, 4); memcpy(&T, bb + 4, 4);
    Net n;
    net_init(&n, &mf.c, B, T, 0);
    memcpy(n.p, mf.data + sizeof(FileHeader), 4 * n.np);
    memcpy(n.inp, bb + 8, 4 * (size_t)B * T);
    memcpy(n.tgt, bb + 8 + 4 * (size_t)B * T, 4 * (size_t)B * T);
    int nv = 0;
    for (int t = 0; t < n.BT; t++) nv += n.tgt[t] >= 0;
    memset(n.g, 0, 4 * n.np);
    net_forward(&n);
    float loss = (float)(net_classifier(&n, 1, 1.0f / nv) / nv);
    net_backward(&n);
    FILE *f = fopen(out_path, "wb");
    if (!f) die("%s", out_path);
    fwrite(&loss, 4, 1, f);
    fwrite(n.g, 4, n.np, f);
    fclose(f);
    cprintf("loss %.6f\n", loss);
    return 0;
}
