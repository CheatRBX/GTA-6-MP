/* Движок инференса: квантованные матрицы (q8/q4) с NEON dotprod, многопоточность, KV-кеш. */
#include "model.h"
#include "pool.h"
#include <math.h>
#include <string.h>
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ================= квантование ================= */
void quantize_q8(const float *x, int8_t *q, float *s, int n) {
    for (int g = 0; g < n / QGS; g++) {
        const float *xg = x + g * QGS;
        float amax = 0;
        for (int k = 0; k < QGS; k++) { float a = fabsf(xg[k]); if (a > amax) amax = a; }
        float d = amax / 127.0f, id = d > 0 ? 1.0f / d : 0.0f;
        s[g] = d;
        for (int k = 0; k < QGS; k++) q[g * QGS + k] = (int8_t)lrintf(xg[k] * id);
    }
}

/* q4: байт j группы хранит w[j] (младшие 4 бита) и w[j+16] (старшие). Значения 0..15, смещение 8. */
void quantize_q4(const float *x, uint8_t *q, float *s, int n) {
    for (int g = 0; g < n / QGS; g++) {
        const float *xg = x + g * QGS;
        float amax = 0, vmax = 0;
        for (int k = 0; k < QGS; k++) { float a = fabsf(xg[k]); if (a > amax) { amax = a; vmax = xg[k]; } }
        float d = vmax / -8.0f, id = d != 0 ? 1.0f / d : 0.0f;
        s[g] = d;
        uint8_t *qg = q + g * (QGS / 2);
        for (int j = 0; j < QGS / 2; j++) {
            int a = (int)lrintf(xg[j] * id) + 8, b = (int)lrintf(xg[j + 16] * id) + 8;
            a = a < 0 ? 0 : a > 15 ? 15 : a;
            b = b < 0 ? 0 : b > 15 ? 15 : b;
            qg[j] = (uint8_t)(a | (b << 4));
        }
    }
}

void dequantize_row(const Mat *m, int row, float *out) {
    int n = m->cols;
    if (m->qt == QT_F32) { memcpy(out, (const float *)m->w + (size_t)row * n, sizeof(float) * n); return; }
    const float *s = m->s + (size_t)row * (n / QGS);
    if (m->qt == QT_Q8) {
        const int8_t *q = (const int8_t *)m->w + (size_t)row * n;
        for (int i = 0; i < n; i++) out[i] = q[i] * s[i / QGS];
    } else {
        const uint8_t *q = (const uint8_t *)m->w + (size_t)row * n / 2;
        for (int g = 0; g < n / QGS; g++)
            for (int j = 0; j < QGS / 2; j++) {
                uint8_t b = q[g * 16 + j];
                out[g * QGS + j] = ((b & 15) - 8) * s[g];
                out[g * QGS + j + 16] = ((b >> 4) - 8) * s[g];
            }
    }
}

/* ================= скалярные произведения ================= */
static inline float dot_f32(const float *a, const float *b, int n) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int i = 0;
    for (; i + 4 <= n; i += 4) { s0 += a[i] * b[i]; s1 += a[i + 1] * b[i + 1]; s2 += a[i + 2] * b[i + 2]; s3 += a[i + 3] * b[i + 3]; }
    for (; i < n; i++) s0 += a[i] * b[i];
    return (s0 + s1) + (s2 + s3);
}

static inline float dot_q8(const int8_t *w, const float *ws, const int8_t *x, const float *xs, int ng) {
#if defined(__ARM_FEATURE_DOTPROD)
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    const int32x4_t z = vdupq_n_s32(0);
    int g = 0;
    for (; g + 2 <= ng; g += 2) {
        int32x4_t p0 = vdotq_s32(vdotq_s32(z, vld1q_s8(w), vld1q_s8(x)), vld1q_s8(w + 16), vld1q_s8(x + 16));
        int32x4_t p1 = vdotq_s32(vdotq_s32(z, vld1q_s8(w + 32), vld1q_s8(x + 32)), vld1q_s8(w + 48), vld1q_s8(x + 48));
        acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[g] * xs[g]);
        acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), ws[g + 1] * xs[g + 1]);
        w += 64; x += 64;
    }
    if (g < ng) {
        int32x4_t p0 = vdotq_s32(vdotq_s32(z, vld1q_s8(w), vld1q_s8(x)), vld1q_s8(w + 16), vld1q_s8(x + 16));
        acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[g] * xs[g]);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(__ARM_NEON) && defined(__aarch64__)
    float32x4_t acc = vdupq_n_f32(0);
    for (int g = 0; g < ng; g++) {
        int8x16_t w0 = vld1q_s8(w), w1 = vld1q_s8(w + 16), x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
        int16x8_t a = vmull_s8(vget_low_s8(w0), vget_low_s8(x0));
        a = vmlal_s8(a, vget_high_s8(w0), vget_high_s8(x0));
        int16x8_t b = vmull_s8(vget_low_s8(w1), vget_low_s8(x1));
        b = vmlal_s8(b, vget_high_s8(w1), vget_high_s8(x1));
        int32x4_t p = vpaddlq_s16(a);
        p = vpadalq_s16(p, b);
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ws[g] * xs[g]);
        w += 32; x += 32;
    }
    return vaddvq_f32(acc);
#else
    float acc = 0;
    for (int g = 0; g < ng; g++) {
        int32_t s = 0;
        for (int k = 0; k < QGS; k++) s += (int32_t)w[k] * (int32_t)x[k];
        acc += (float)s * ws[g] * xs[g];
        w += QGS; x += QGS;
    }
    return acc;
#endif
}

static inline float dot_q4(const uint8_t *w, const float *ws, const int8_t *x, const float *xs, int ng) {
#if defined(__ARM_FEATURE_DOTPROD)
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    const int32x4_t z = vdupq_n_s32(0);
    float32x4_t acc0 = vdupq_n_f32(0), acc1 = vdupq_n_f32(0);
    int g = 0;
    for (; g + 2 <= ng; g += 2) {
        uint8x16_t v0 = vld1q_u8(w), v1 = vld1q_u8(w + 16);
        int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
        int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
        int8x16_t l1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v1, m4)), s8);
        int8x16_t h1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v1, 4)), s8);
        int32x4_t p0 = vdotq_s32(vdotq_s32(z, l0, vld1q_s8(x)), h0, vld1q_s8(x + 16));
        int32x4_t p1 = vdotq_s32(vdotq_s32(z, l1, vld1q_s8(x + 32)), h1, vld1q_s8(x + 48));
        acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[g] * xs[g]);
        acc1 = vmlaq_n_f32(acc1, vcvtq_f32_s32(p1), ws[g + 1] * xs[g + 1]);
        w += 32; x += 64;
    }
    if (g < ng) {
        uint8x16_t v0 = vld1q_u8(w);
        int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
        int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
        int32x4_t p0 = vdotq_s32(vdotq_s32(z, l0, vld1q_s8(x)), h0, vld1q_s8(x + 16));
        acc0 = vmlaq_n_f32(acc0, vcvtq_f32_s32(p0), ws[g] * xs[g]);
    }
    return vaddvq_f32(vaddq_f32(acc0, acc1));
#elif defined(__ARM_NEON) && defined(__aarch64__)
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8x16_t s8 = vdupq_n_s8(8);
    float32x4_t acc = vdupq_n_f32(0);
    for (int g = 0; g < ng; g++) {
        uint8x16_t v0 = vld1q_u8(w);
        int8x16_t l0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0, m4)), s8);
        int8x16_t h0 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0, 4)), s8);
        int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
        int16x8_t a = vmull_s8(vget_low_s8(l0), vget_low_s8(x0));
        a = vmlal_s8(a, vget_high_s8(l0), vget_high_s8(x0));
        int16x8_t b = vmull_s8(vget_low_s8(h0), vget_low_s8(x1));
        b = vmlal_s8(b, vget_high_s8(h0), vget_high_s8(x1));
        int32x4_t p = vpaddlq_s16(a);
        p = vpadalq_s16(p, b);
        acc = vmlaq_n_f32(acc, vcvtq_f32_s32(p), ws[g] * xs[g]);
        w += 16; x += 32;
    }
    return vaddvq_f32(acc);
#else
    float acc = 0;
    for (int g = 0; g < ng; g++) {
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            s += ((int32_t)(w[j] & 15) - 8) * x[j];
            s += ((int32_t)(w[j] >> 4) - 8) * x[j + 16];
        }
        acc += (float)s * ws[g] * xs[g];
        w += 16; x += 32;
    }
    return acc;
#endif
}

static inline float row_dot(const Mat *m, int row, const float *xf, const int8_t *xq, const float *xs) {
    int n = m->cols, ng = n / QGS;
    switch (m->qt) {
    case QT_Q8: return dot_q8((const int8_t *)m->w + (size_t)row * n, m->s + (size_t)row * ng, xq, xs, ng);
    case QT_Q4: return dot_q4((const uint8_t *)m->w + (size_t)row * n / 2, m->s + (size_t)row * ng, xq, xs, ng);
    default: return dot_f32((const float *)m->w + (size_t)row * n, xf, n);
    }
}

/* ================= параллельные задания ================= */
typedef struct {
    int nmat;
    const Mat *m[3];
    float *out[3];
    int off[4];
    const float *xf;
    const int8_t *xq;
    const float *xs;
} MatJob;

static void matjob_fn(void *ctx, int i0, int i1) {
    MatJob *j = ctx;
    int k = 0;
    for (int r = i0; r < i1; r++) {
        while (r >= j->off[k + 1]) k++;
        j->out[k][r - j->off[k]] = row_dot(j->m[k], r - j->off[k], j->xf, j->xq, j->xs);
    }
}

static int chunk_for(int rows) {
    int c = rows / (pool_size() * 12);
    if (c < 4) c = 4;
    if (c > 64) c = 64;
    return c;
}

static void matmul_n(int n, const Mat **ms, float **outs, const float *xf, const int8_t *xq, const float *xs) {
    MatJob j = {.nmat = n, .xf = xf, .xq = xq, .xs = xs};
    j.off[0] = 0;
    for (int i = 0; i < n; i++) { j.m[i] = ms[i]; j.out[i] = outs[i]; j.off[i + 1] = j.off[i] + ms[i]->rows; }
    for (int i = n + 1; i < 4; i++) j.off[i] = j.off[n];
    pool_run(matjob_fn, &j, j.off[n], chunk_for(j.off[n]));
}

typedef struct { const Mat *w1, *w3; float *out; const float *xf; const int8_t *xq; const float *xs; } SwiJob;
static void swi_fn(void *ctx, int i0, int i1) {
    SwiJob *j = ctx;
    for (int i = i0; i < i1; i++) {
        float a = row_dot(j->w1, i, j->xf, j->xq, j->xs);
        float b = row_dot(j->w3, i, j->xf, j->xq, j->xs);
        j->out[i] = a / (1.0f + expf(-a)) * b;
    }
}

typedef struct { const Model *m; State *s; int layer, pos; } AttJob;
static void att_fn(void *ctx, int h0, int h1) {
    AttJob *j = ctx;
    const Config *c = &j->m->c;
    State *s = j->s;
    int hd = cfg_head_dim(c), kvd = cfg_kv_dim(c), grp = c->n_heads / c->n_kv_heads;
    float scale = 1.0f / sqrtf((float)hd);
    size_t loff = (size_t)j->layer * s->seq_len * kvd;
    for (int h = h0; h < h1; h++) {
        const float *q = s->q + h * hd;
        float *att = s->att + (size_t)h * s->seq_len;
        int kvh = h / grp;
        float mx = -1e30f;
        for (int t = 0; t <= j->pos; t++) {
            const kv_t *k = s->kc + loff + (size_t)t * kvd + kvh * hd;
            float d = 0;
            for (int i = 0; i < hd; i++) d += q[i] * (float)k[i];
            d *= scale;
            att[t] = d;
            if (d > mx) mx = d;
        }
        float sum = 0;
        for (int t = 0; t <= j->pos; t++) { att[t] = expf(att[t] - mx); sum += att[t]; }
        float inv = 1.0f / sum;
        float *out = s->xb2 + h * hd;
        for (int i = 0; i < hd; i++) out[i] = 0;
        for (int t = 0; t <= j->pos; t++) {
            const kv_t *v = s->vc + loff + (size_t)t * kvd + kvh * hd;
            float a = att[t] * inv;
            for (int i = 0; i < hd; i++) out[i] += a * (float)v[i];
        }
    }
}

/* ================= модель ================= */
static void rmsnorm(float *o, const float *x, const float *w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
}

static void rope(float *v, int nheads, int hd, const float *cs, const float *sn) {
    for (int h = 0; h < nheads; h++) {
        float *p = v + h * hd;
        for (int i = 0; i < hd / 2; i++) {
            float a = p[2 * i], b = p[2 * i + 1], c = cs[i], s = sn[i];
            p[2 * i] = a * c - b * s;
            p[2 * i + 1] = a * s + b * c;
        }
    }
}

static size_t mat_bytes(int qt, size_t numel) {
    return qt == QT_F32 ? numel * 4 : qt == QT_Q8 ? numel : numel / 2;
}

/* Разметка весов: f32/q8/q4. Для q8/q4 у каждого тензора сначала данные всех слоёв, потом масштабы. */
static void map_weights(Model *m, const uint8_t *base, size_t avail, int qt) {
    const Config *c = &m->c;
    int L = c->n_layers;
    m->wq = xcalloc(sizeof(Mat) * L); m->wk = xcalloc(sizeof(Mat) * L); m->wv = xcalloc(sizeof(Mat) * L);
    m->wo = xcalloc(sizeof(Mat) * L); m->w1 = xcalloc(sizeof(Mat) * L); m->w2 = xcalloc(sizeof(Mat) * L);
    m->w3 = xcalloc(sizeof(Mat) * L);
    Mat *dst[T_COUNT] = {&m->emb, NULL, m->wq, m->wk, m->wv, m->wo, NULL, m->w1, m->w2, m->w3, NULL};
    size_t off = 0;
    for (int t = 0; t < T_COUNT; t++) {
        int cnt, rows, cols;
        tensor_shape(c, t, &cnt, &rows, &cols);
        size_t numel = (size_t)rows * cols;
        if (!tensor_is_matrix(t)) {
            if (off + numel * cnt * 4 > avail) die("файл модели обрезан");
            const float *p = (const float *)(base + off);
            if (t == T_ATTN_NORM) m->attn_norm = p;
            else if (t == T_FFN_NORM) m->ffn_norm = p;
            else m->final_norm = p;
            off += numel * cnt * 4;
            continue;
        }
        size_t dbytes = mat_bytes(qt, numel);
        size_t sbytes = qt == QT_F32 ? 0 : numel / QGS * 4;
        if (off + (dbytes + sbytes) * cnt > avail) die("файл модели обрезан");
        const uint8_t *data = base + off;
        const float *scales = (const float *)(base + off + dbytes * cnt);
        for (int l = 0; l < cnt; l++) {
            Mat *mt = &dst[t][l];
            mt->qt = qt; mt->rows = rows; mt->cols = cols;
            mt->w = data + dbytes * l;
            mt->s = qt == QT_F32 ? NULL : scales + numel / QGS * l;
        }
        off += (dbytes + sbytes) * cnt;
    }
}

void model_load(Model *m, const char *path) {
    memset(m, 0, sizeof(*m));
    model_open(&m->mf, path);
    m->from_file = 1;
    m->c = m->mf.c;
    m->qtype = m->mf.h.qtype;
    if (m->qtype < QT_F32 || m->qtype > QT_Q4) die("неизвестный тип квантования %d", m->qtype);
    m->train_steps = m->mf.h.train_steps;
    m->train_tokens = m->mf.h.train_tokens;
    size_t end = m->mf.h.tok_offset > 0 ? (size_t)m->mf.h.tok_offset : m->mf.size;
    map_weights(m, m->mf.data + sizeof(FileHeader), end - sizeof(FileHeader), m->qtype);
    if (m->mf.h.tok_offset > 0) {
        if ((size_t)(m->mf.h.tok_offset + m->mf.h.tok_size) > m->mf.size) die("токенизатор в файле обрезан");
        if (!tok_load(&m->tok, m->mf.data + m->mf.h.tok_offset, (size_t)m->mf.h.tok_size)) die("токенизатор в файле повреждён");
        if (m->tok.vocab_size != m->c.vocab_size) die("размер словаря модели и токенизатора не совпадает");
        m->has_tok = 1;
    }
}

void model_from_f32(Model *m, const Config *c, float *params) {
    memset(m, 0, sizeof(*m));
    m->c = *c;
    m->qtype = QT_F32;
    map_weights(m, (const uint8_t *)params, params_count(c) * 4, QT_F32);
}

static void quantize_mat(Mat *mt, int qt) {
    if (mt->qt != QT_F32 || qt == QT_F32) return;
    size_t rows = mt->rows, cols = mt->cols;
    const float *src = mt->w;
    float *s = xmalloc(sizeof(float) * rows * (cols / QGS));
    void *q = xmalloc(qt == QT_Q8 ? rows * cols : rows * cols / 2);
    for (size_t r = 0; r < rows; r++) {
        if (qt == QT_Q8) quantize_q8(src + r * cols, (int8_t *)q + r * cols, s + r * (cols / QGS), (int)cols);
        else quantize_q4(src + r * cols, (uint8_t *)q + r * cols / 2, s + r * (cols / QGS), (int)cols);
    }
    mt->qt = qt; mt->w = q; mt->s = s;
}

void model_quantize_mem(Model *m, int qt) {
    quantize_mat(&m->emb, qt);
    for (int l = 0; l < m->c.n_layers; l++) {
        quantize_mat(&m->wq[l], qt); quantize_mat(&m->wk[l], qt); quantize_mat(&m->wv[l], qt);
        quantize_mat(&m->wo[l], qt); quantize_mat(&m->w1[l], qt); quantize_mat(&m->w2[l], qt);
        quantize_mat(&m->w3[l], qt);
    }
    m->qtype = qt;
}

void model_free(Model *m) {
    free(m->wq); free(m->wk); free(m->wv); free(m->wo); free(m->w1); free(m->w2); free(m->w3);
    if (m->has_tok) tok_free(&m->tok);
    if (m->from_file) model_close(&m->mf);
    memset(m, 0, sizeof(*m));
}

size_t model_weight_bytes(const Model *m) {
    size_t s = 0;
    for (int t = 0; t < T_COUNT; t++) {
        size_t n = tensor_numel(&m->c, t);
        if (!tensor_is_matrix(t)) s += n * 4;
        else s += mat_bytes(m->qtype, n) + (m->qtype == QT_F32 ? 0 : n / QGS * 4);
    }
    return s;
}

void state_init(State *s, const Config *c, int seq_len) {
    memset(s, 0, sizeof(*s));
    if (seq_len <= 0 || seq_len > c->seq_len) seq_len = c->seq_len;
    s->seq_len = seq_len;
    int D = c->dim, F = c->hidden_dim, kvd = cfg_kv_dim(c), hd = cfg_head_dim(c);
    int mx = D > F ? D : F;
    s->x = xcalloc(sizeof(float) * D);
    s->xb = xcalloc(sizeof(float) * mx);
    s->xb2 = xcalloc(sizeof(float) * D);
    s->hb = xcalloc(sizeof(float) * F);
    s->q = xcalloc(sizeof(float) * D);
    s->k = xcalloc(sizeof(float) * kvd);
    s->v = xcalloc(sizeof(float) * kvd);
    s->att = xcalloc(sizeof(float) * c->n_heads * seq_len);
    s->logits = xcalloc(sizeof(float) * c->vocab_size);
    s->xq = xcalloc(mx);
    s->xs = xcalloc(sizeof(float) * (mx / QGS));
    s->kc = xcalloc(sizeof(kv_t) * (size_t)c->n_layers * seq_len * kvd);
    s->vc = xcalloc(sizeof(kv_t) * (size_t)c->n_layers * seq_len * kvd);
    s->rope_cos = xmalloc(sizeof(float) * seq_len * (hd / 2));
    s->rope_sin = xmalloc(sizeof(float) * seq_len * (hd / 2));
    for (int p = 0; p < seq_len; p++)
        for (int i = 0; i < hd / 2; i++) {
            double f = pow((double)c->rope_theta, -2.0 * i / hd);
            s->rope_cos[p * (hd / 2) + i] = (float)cos(p * f);
            s->rope_sin[p * (hd / 2) + i] = (float)sin(p * f);
        }
}

void state_free(State *s) {
    free(s->x); free(s->xb); free(s->xb2); free(s->hb); free(s->q); free(s->k); free(s->v);
    free(s->att); free(s->logits); free(s->xq); free(s->xs); free(s->kc); free(s->vc);
    free(s->rope_cos); free(s->rope_sin);
    memset(s, 0, sizeof(*s));
}

float *forward(const Model *m, State *s, int token, int pos) {
    const Config *c = &m->c;
    int D = c->dim, F = c->hidden_dim, kvd = cfg_kv_dim(c), hd = cfg_head_dim(c);
    int quant = m->qtype != QT_F32;
    float eps = c->norm_eps;
    if (pos >= s->seq_len) die("контекст переполнен");
    if (token < 0 || token >= c->vocab_size) token = 0;
    dequantize_row(&m->emb, token, s->x);
    const float *cs = s->rope_cos + pos * (hd / 2), *sn = s->rope_sin + pos * (hd / 2);

    for (int l = 0; l < c->n_layers; l++) {
        rmsnorm(s->xb, s->x, m->attn_norm + l * D, D, eps);
        if (quant) quantize_q8(s->xb, s->xq, s->xs, D);
        const Mat *qkv[3] = {&m->wq[l], &m->wk[l], &m->wv[l]};
        float *outs[3] = {s->q, s->k, s->v};
        matmul_n(3, qkv, outs, s->xb, s->xq, s->xs);
        rope(s->q, c->n_heads, hd, cs, sn);
        rope(s->k, c->n_kv_heads, hd, cs, sn);
        size_t o = ((size_t)l * s->seq_len + pos) * kvd;
        for (int i = 0; i < kvd; i++) { s->kc[o + i] = (kv_t)s->k[i]; s->vc[o + i] = (kv_t)s->v[i]; }

        AttJob aj = {m, s, l, pos};
        pool_run(att_fn, &aj, c->n_heads, 1);

        if (quant) quantize_q8(s->xb2, s->xq, s->xs, D);
        const Mat *wo[1] = {&m->wo[l]};
        float *ob[1] = {s->xb};
        matmul_n(1, wo, ob, s->xb2, s->xq, s->xs);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];

        rmsnorm(s->xb, s->x, m->ffn_norm + l * D, D, eps);
        if (quant) quantize_q8(s->xb, s->xq, s->xs, D);
        SwiJob sj = {&m->w1[l], &m->w3[l], s->hb, s->xb, s->xq, s->xs};
        pool_run(swi_fn, &sj, F, chunk_for(F));

        if (quant) quantize_q8(s->hb, s->xq, s->xs, F);
        const Mat *w2[1] = {&m->w2[l]};
        matmul_n(1, w2, ob, s->hb, s->xq, s->xs);
        for (int i = 0; i < D; i++) s->x[i] += s->xb[i];
    }
    rmsnorm(s->x, s->x, m->final_norm, D, eps);
    if (quant) quantize_q8(s->x, s->xq, s->xs, D);
    const Mat *em[1] = {&m->emb};
    float *lo[1] = {s->logits};
    matmul_n(1, em, lo, s->x, s->xq, s->xs);
    return s->logits;
}

/* ================= сэмплирование ================= */
void sampler_default(Sampler *sp) {
    sp->temperature = 0.7f;
    sp->top_p = 0.9f;
    sp->top_k = 40;
    sp->rep_penalty = 1.1f;
    sp->rep_window = 64;
    sp->rng = 0;
}

typedef struct { float p; int i; } PI;
static int cmp_pi(const void *a, const void *b) {
    float x = ((const PI *)a)->p, y = ((const PI *)b)->p;
    return x < y ? 1 : x > y ? -1 : 0;
}

int sample_token(const Sampler *sp, uint64_t *rng, float *lg, int V, const int *hist, int nhist) {
    if (sp->rep_penalty > 1.0f && hist) {
        int from = nhist - sp->rep_window;
        if (from < 0) from = 0;
        for (int i = from; i < nhist; i++) {
            int t = hist[i], later = 0;
            if (t < 0 || t >= V) continue;
            for (int j = i + 1; j < nhist; j++) if (hist[j] == t) { later = 1; break; }
            if (later) continue; /* штрафуем токен один раз */
            lg[t] = lg[t] > 0 ? lg[t] / sp->rep_penalty : lg[t] * sp->rep_penalty;
        }
    }
    if (sp->temperature <= 0.0f) {
        int best = 0;
        for (int i = 1; i < V; i++) if (lg[i] > lg[best]) best = i;
        return best;
    }
    static PI *buf = NULL;
    static int cap = 0;
    if (cap < V) { free(buf); buf = xmalloc(sizeof(PI) * V); cap = V; }
    float inv_t = 1.0f / sp->temperature;
    float top_p = sp->top_p > 0 && sp->top_p < 1.0f ? sp->top_p : 1.0f;
    int n = 0;
    if (sp->top_k > 0 && sp->top_k < V) {
        /* Быстро: один проход, держим k лучших в отсортированном массиве (без сортировки всего словаря). */
        int k = sp->top_k;
        for (int i = 0; i < V; i++) {
            float v = lg[i];
            if (n == k && v <= buf[k - 1].p) continue;
            int j = n < k ? n++ : k - 1;
            while (j > 0 && buf[j - 1].p < v) { buf[j] = buf[j - 1]; j--; }
            buf[j].p = v; buf[j].i = i;
        }
        float mx = buf[0].p;
        for (int j = 0; j < n; j++) buf[j].p = expf((buf[j].p - mx) * inv_t);
    } else {
        float mx = lg[0];
        for (int i = 1; i < V; i++) if (lg[i] > mx) mx = lg[i];
        float sum = 0;
        for (int i = 0; i < V; i++) { buf[i].p = expf((lg[i] - mx) * inv_t); buf[i].i = i; sum += buf[i].p; }
        float cutoff = top_p < 1.0f ? (1.0f - top_p) / (V - 1) * sum : 0.0f;
        for (int i = 0; i < V; i++) if (buf[i].p >= cutoff) buf[n++] = buf[i];
        qsort(buf, n, sizeof(PI), cmp_pi);
    }
    float sum = 0;
    for (int j = 0; j < n; j++) sum += buf[j].p;
    float cum = 0;
    int last = n - 1;
    for (int j = 0; j < n; j++) {
        cum += buf[j].p;
        if (cum >= top_p * sum) { last = j; break; }
    }
    float total = 0;
    for (int j = 0; j <= last; j++) total += buf[j].p;
    float r = rng_f32(rng) * total, acc = 0;
    for (int j = 0; j <= last; j++) { acc += buf[j].p; if (r < acc) return buf[j].i; }
    return buf[last].i;
}
