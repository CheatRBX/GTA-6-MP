#pragma once
/* Плотная математика для обучения (f32, многопоточно). */

/* C[M,N] = (acc ? C : 0) + Σ_k A(m,k)·B(n,k)
 *   ta=0: A хранится [M][K], ta=1: A хранится [K][M]
 *   tb=0: B хранится [N][K], tb=1: B хранится [K][N]   (транспонирование — бесплатно, при упаковке) */
void gemm(float *C, const float *A, const float *B, int M, int N, int K, int acc, int ta, int tb);
void gemm_nt(float *C, const float *A, const float *B, int M, int N, int K, int acc); /* ta=tb=0 */
/* dst[C,R] = src[R,C]^T */
void transpose(float *dst, const float *src, int R, int C);

/* Однопоточные варианты (для вызова внутри задач пула), строки со своими шагами.
 * mode: GM_FULL — всё; GM_LOWER — только j <= i (с точностью до блока 4);
 * GM_KPRE — по K только [0, i0+4); GM_KSUF — по K только [i0, K)  (i0 — начало блока из 4 строк). */
enum { GM_FULL, GM_LOWER, GM_KPRE, GM_KSUF };
void gemm_nt_st(float *C, int ldc, const float *A, int lda, const float *B, int ldb, int M, int N, int K, int acc, int mode);
void transpose_st(float *dst, int ldd, const float *src, int lds, int R, int C);
