/* Умножение матриц для обучения: микроядро 4x4 на векторных расширениях GCC/Clang
 * (на ARM это NEON, на x86 — SSE/AVX), блокировка по кешу, пул потоков. */
#include "corax.h"
#include "gemm.h"
#include "pool.h"
#include <string.h>

typedef float v4f __attribute__((vector_size(16)));
static inline v4f ldv(const float *p) { v4f v; memcpy(&v, p, 16); return v; }
static inline float hsum(v4f v) { return (v[0] + v[1]) + (v[2] + v[3]); }

#define MB 16   /* строк C на задачу */
#define NB 64   /* столбцов C на задачу */
#define KB 512  /* блок по K */

static inline float dotv(const float *a, const float *b, int K) {
    v4f s0 = {0}, s1 = {0};
    int k = 0;
    for (; k + 8 <= K; k += 8) { s0 += ldv(a + k) * ldv(b + k); s1 += ldv(a + k + 4) * ldv(b + k + 4); }
    float s = hsum(s0 + s1);
    for (; k < K; k++) s += a[k] * b[k];
    return s;
}

static void kern4x4(const float *A, const float *B, float *C, int lda, int ldb, int ldc, int K, int acc) {
    const float *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda;
    const float *b0 = B, *b1 = B + ldb, *b2 = B + 2 * ldb, *b3 = B + 3 * ldb;
    v4f c00 = {0}, c01 = {0}, c02 = {0}, c03 = {0};
    v4f c10 = {0}, c11 = {0}, c12 = {0}, c13 = {0};
    v4f c20 = {0}, c21 = {0}, c22 = {0}, c23 = {0};
    v4f c30 = {0}, c31 = {0}, c32 = {0}, c33 = {0};
    int k = 0;
    for (; k + 4 <= K; k += 4) {
        v4f x0 = ldv(a0 + k), x1 = ldv(a1 + k), x2 = ldv(a2 + k), x3 = ldv(a3 + k);
        v4f y0 = ldv(b0 + k), y1 = ldv(b1 + k), y2 = ldv(b2 + k), y3 = ldv(b3 + k);
        c00 += x0 * y0; c01 += x0 * y1; c02 += x0 * y2; c03 += x0 * y3;
        c10 += x1 * y0; c11 += x1 * y1; c12 += x1 * y2; c13 += x1 * y3;
        c20 += x2 * y0; c21 += x2 * y1; c22 += x2 * y2; c23 += x2 * y3;
        c30 += x3 * y0; c31 += x3 * y1; c32 += x3 * y2; c33 += x3 * y3;
    }
    float r[4][4] = {
        {hsum(c00), hsum(c01), hsum(c02), hsum(c03)},
        {hsum(c10), hsum(c11), hsum(c12), hsum(c13)},
        {hsum(c20), hsum(c21), hsum(c22), hsum(c23)},
        {hsum(c30), hsum(c31), hsum(c32), hsum(c33)},
    };
    for (; k < K; k++) {
        float x[4] = {a0[k], a1[k], a2[k], a3[k]}, y[4] = {b0[k], b1[k], b2[k], b3[k]};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) r[i][j] += x[i] * y[j];
    }
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) {
            float *c = C + i * ldc + j;
            *c = acc ? *c + r[i][j] : r[i][j];
        }
}

typedef struct { float *C; const float *A, *B; int M, N, K, acc, nbn; } GemmJob;

static void gemm_task(void *ctx, int t0, int t1) {
    GemmJob *j = ctx;
    for (int t = t0; t < t1; t++) {
        int m0 = (t / j->nbn) * MB, n0 = (t % j->nbn) * NB;
        int m1 = m0 + MB < j->M ? m0 + MB : j->M;
        int n1 = n0 + NB < j->N ? n0 + NB : j->N;
        for (int k0 = 0; k0 < j->K; k0 += KB) {
            int kl = j->K - k0 < KB ? j->K - k0 : KB;
            int acc = j->acc || k0 > 0;
            for (int m = m0; m < m1; m += 4) {
                for (int n = n0; n < n1; n += 4) {
                    const float *A = j->A + (size_t)m * j->K + k0;
                    const float *B = j->B + (size_t)n * j->K + k0;
                    float *C = j->C + (size_t)m * j->N + n;
                    if (m + 4 <= m1 && n + 4 <= n1) {
                        kern4x4(A, B, C, j->K, j->K, j->N, kl, acc);
                    } else {
                        int mr = m1 - m < 4 ? m1 - m : 4, nr = n1 - n < 4 ? n1 - n : 4;
                        for (int i = 0; i < mr; i++)
                            for (int q = 0; q < nr; q++) {
                                float v = dotv(A + (size_t)i * j->K, B + (size_t)q * j->K, kl);
                                float *c = C + (size_t)i * j->N + q;
                                *c = acc ? *c + v : v;
                            }
                    }
                }
            }
        }
    }
}

void gemm_nt(float *C, const float *A, const float *B, int M, int N, int K, int acc) {
    if (M <= 0 || N <= 0) return;
    if (K <= 0) { if (!acc) memset(C, 0, sizeof(float) * (size_t)M * N); return; }
    GemmJob j = {C, A, B, M, N, K, acc, (N + NB - 1) / NB};
    int tasks = ((M + MB - 1) / MB) * j.nbn;
    pool_run(gemm_task, &j, tasks, 1);
}

typedef struct { float *dst; const float *src; int R, C; } TrJob;
static void tr_task(void *ctx, int b0, int b1) {
    TrJob *j = ctx;
    for (int b = b0; b < b1; b++) {
        int r0 = b * 32, r1 = r0 + 32 < j->R ? r0 + 32 : j->R;
        for (int c0 = 0; c0 < j->C; c0 += 32) {
            int c1 = c0 + 32 < j->C ? c0 + 32 : j->C;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) j->dst[(size_t)c * j->R + r] = j->src[(size_t)r * j->C + c];
        }
    }
}

void transpose(float *dst, const float *src, int R, int C) {
    TrJob j = {dst, src, R, C};
    pool_run(tr_task, &j, (R + 31) / 32, 1);
}

void gemm_nt_st(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M, int N, int K, int acc, int mode) {
    for (int i0 = 0; i0 < M; i0 += 4) {
        int mr = M - i0 < 4 ? M - i0 : 4;
        int nend = N, k0 = 0, k1 = K;
        if (mode == GM_LOWER) nend = i0 + 4 < N ? i0 + 4 : N;
        else if (mode == GM_KPRE) k1 = i0 + 4 < K ? i0 + 4 : K;
        else if (mode == GM_KSUF) k0 = i0 < K ? i0 : K;
        int kl = k1 - k0;
        const float *Ai = A + (size_t)i0 * lda + k0;
        for (int j0 = 0; j0 < nend; j0 += 4) {
            int nr = nend - j0 < 4 ? nend - j0 : 4;
            const float *Bj = B + (size_t)j0 * ldb + k0;
            float *Cij = C + (size_t)i0 * ldc + j0;
            if (mr == 4 && nr == 4 && kl > 0) {
                kern4x4(Ai, Bj, Cij, lda, ldb, ldc, kl, acc);
            } else {
                for (int i = 0; i < mr; i++)
                    for (int q = 0; q < nr; q++) {
                        float v = kl > 0 ? dotv(Ai + (size_t)i * lda, Bj + (size_t)q * ldb, kl) : 0.0f;
                        float *c = Cij + (size_t)i * ldc + q;
                        *c = acc ? *c + v : v;
                    }
            }
        }
    }
}

void transpose_st(float *dst, int ldd, const float *src, int lds, int R, int C) {
    for (int r0 = 0; r0 < R; r0 += 16)
        for (int c0 = 0; c0 < C; c0 += 16) {
            int r1 = r0 + 16 < R ? r0 + 16 : R, c1 = c0 + 16 < C ? c0 + 16 : C;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++) dst[(size_t)c * ldd + r] = src[(size_t)r * lds + c];
        }
}
