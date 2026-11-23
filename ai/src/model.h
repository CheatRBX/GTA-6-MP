#pragma once
#include "corax.h"
#include "tokenizer.h"

/* Матрица весов: f32, q8 (int8 + масштаб на группу 32) или q4 (4 бита + масштаб). */
typedef struct {
    int qt, rows, cols;
    const void *w;     /* f32: float[rows*cols]; q8: int8[rows*cols]; q4: uint8[rows*cols/2] */
    const float *s;    /* масштабы: float[rows*cols/32] (q8/q4) */
} Mat;

typedef struct {
    Config c;
    int qtype;
    Mat emb;
    Mat *wq, *wk, *wv, *wo, *w1, *w2, *w3; /* по слою */
    const float *attn_norm, *ffn_norm, *final_norm;
    Tokenizer tok;
    int has_tok;
    ModelFile mf;
    int from_file;
    int64_t train_steps, train_tokens;
} Model;

/* KV-кеш в половинной точности на ARM (меньше трафика памяти на длинном контексте). */
#if defined(__aarch64__) && !defined(CORAX_KV_F32)
typedef __fp16 kv_t;
#else
typedef float kv_t;
#endif

typedef struct {
    int seq_len;
    float *x, *xb, *xb2, *hb, *q, *k, *v, *att, *logits;
    int8_t *xq;
    float *xs;
    kv_t *kc, *vc;            /* [L][seq][kv_dim] */
    float *rope_cos, *rope_sin; /* [seq][head_dim/2] */
} State;

void model_load(Model *m, const char *path);
void model_from_f32(Model *m, const Config *c, float *params); /* без копирования */
void model_free(Model *m);
void model_quantize_mem(Model *m, int qt); /* квантовать f32-модель в памяти (для замеров) */
size_t model_weight_bytes(const Model *m);

void state_init(State *s, const Config *c, int seq_len);
void state_free(State *s);
float *forward(const Model *m, State *s, int token, int pos);

typedef struct {
    float temperature, top_p, rep_penalty;
    int top_k, rep_window;
    uint64_t rng;
} Sampler;
void sampler_default(Sampler *sp);
int sample_token(const Sampler *sp, uint64_t *rng, float *logits, int vocab, const int *hist, int nhist);

/* Квантование строк (общие для quantize и тестов). */
void quantize_q8(const float *x, int8_t *q, float *s, int n);
void quantize_q4(const float *x, uint8_t *q, float *s, int n);
void dequantize_row(const Mat *m, int row, float *out);
