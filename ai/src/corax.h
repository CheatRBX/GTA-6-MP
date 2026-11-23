/*
 * Corax LM — собственная языковая модель с нуля.
 * Общие определения: конфиг модели, формат файла, утилиты.
 */
#pragma once
#define _GNU_SOURCE
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define CRX_MAGIC     0x4D585243u /* "CRXM" — файл модели */
#define CRX_VERSION   1
#define CRX_TOK_MAGIC 0x54585243u /* "CRXT" — токенизатор */
#define CRX_OPT_MAGIC 0x4F585243u /* "CRXO" — состояние оптимизатора */
#define QGS 32                    /* размер группы квантования */

enum { QT_F32 = 0, QT_Q8 = 1, QT_Q4 = 2 };

typedef struct {
    int dim;        /* ширина модели */
    int n_layers;   /* число слоёв-трансформеров */
    int n_heads;    /* головы внимания (query) */
    int n_kv_heads; /* головы key/value (GQA) */
    int hidden_dim; /* ширина SwiGLU */
    int vocab_size; /* размер словаря */
    int seq_len;    /* максимальный контекст */
    float rope_theta;
    float norm_eps;
} Config;

/* Заголовок файла модели — ровно 256 байт, little-endian. */
typedef struct {
    uint32_t magic, version;
    int64_t train_steps, train_tokens;
    int64_t tok_offset, tok_size; /* токенизатор вшит в конец файла */
    int32_t dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size, seq_len;
    int32_t qtype, group_size;
    float rope_theta, norm_eps;
    uint8_t reserved[172];
} FileHeader;

_Static_assert(sizeof(FileHeader) == 256, "header must be 256 bytes");

/* Порядок тензоров в файле (тензор за тензором, все слои подряд). */
enum {
    T_EMB, T_ATTN_NORM, T_WQ, T_WK, T_WV, T_WO,
    T_FFN_NORM, T_W1, T_W2, T_W3, T_FINAL_NORM, T_COUNT
};

static inline int cfg_head_dim(const Config *c) { return c->dim / c->n_heads; }
static inline int cfg_kv_dim(const Config *c) { return cfg_head_dim(c) * c->n_kv_heads; }

/* count — сколько копий (слоёв), rows x cols — форма одной копии. */
static inline void tensor_shape(const Config *c, int t, int *count, int *rows, int *cols) {
    int L = c->n_layers, D = c->dim, F = c->hidden_dim, KV = cfg_kv_dim(c);
    switch (t) {
    case T_EMB:        *count = 1; *rows = c->vocab_size; *cols = D; break;
    case T_ATTN_NORM:  *count = L; *rows = 1;  *cols = D; break;
    case T_WQ:         *count = L; *rows = D;  *cols = D; break;
    case T_WK:         *count = L; *rows = KV; *cols = D; break;
    case T_WV:         *count = L; *rows = KV; *cols = D; break;
    case T_WO:         *count = L; *rows = D;  *cols = D; break;
    case T_FFN_NORM:   *count = L; *rows = 1;  *cols = D; break;
    case T_W1:         *count = L; *rows = F;  *cols = D; break;
    case T_W2:         *count = L; *rows = D;  *cols = F; break;
    case T_W3:         *count = L; *rows = F;  *cols = D; break;
    default:           *count = 1; *rows = 1;  *cols = D; break; /* T_FINAL_NORM */
    }
}

static inline int tensor_is_matrix(int t) {
    return !(t == T_ATTN_NORM || t == T_FFN_NORM || t == T_FINAL_NORM);
}

static inline size_t tensor_numel(const Config *c, int t) {
    int n, r, k;
    tensor_shape(c, t, &n, &r, &k);
    return (size_t)n * r * k;
}

static inline size_t params_count(const Config *c) {
    size_t s = 0;
    for (int t = 0; t < T_COUNT; t++) s += tensor_numel(c, t);
    return s;
}

/* f32 веса: указатели в один плоский буфер (тот же порядок, что в файле). */
typedef struct {
    float *emb, *attn_norm, *wq, *wk, *wv, *wo, *ffn_norm, *w1, *w2, *w3, *final_norm;
} Weights;

static inline void weights_map(Weights *w, const Config *c, float *p) {
    float **slots[T_COUNT] = {&w->emb, &w->attn_norm, &w->wq, &w->wk, &w->wv, &w->wo,
                              &w->ffn_norm, &w->w1, &w->w2, &w->w3, &w->final_norm};
    for (int t = 0; t < T_COUNT; t++) { *slots[t] = p; p += tensor_numel(c, t); }
}

/* ---------- утилиты ---------- */
void die(const char *fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
void *xmalloc(size_t n);         /* выровненный malloc с проверкой */
void *xcalloc(size_t n);
double now_sec(void);
void ensure_parent_dir(const char *path); /* mkdir -p для папки файла */
const char *qtype_name(int qt);
extern int g_color;                 /* 0 — без ANSI-цветов (не терминал / NO_COLOR) */
void cprintf(const char *fmt, ...) __attribute__((format(printf, 1, 2))); /* printf с вырезанием цветов */

/* Чтение файла модели: mmap. */
typedef struct {
    FileHeader h;
    Config c;
    uint8_t *data;   /* весь файл */
    size_t size;
    int fd;
} ModelFile;
void model_open(ModelFile *mf, const char *path);
void model_close(ModelFile *mf);
void header_to_config(const FileHeader *h, Config *c);
void config_to_header(const Config *c, FileHeader *h);
int config_validate(const Config *c, char *err, size_t errlen);
void config_print(const Config *c, FILE *f);

/* Пресеты размеров модели. */
int preset_get(const char *name, Config *c);
void preset_list(FILE *f);

/* ГПСЧ */
static inline uint64_t rng_u64(uint64_t *s) {
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}
static inline float rng_f32(uint64_t *s) { return (rng_u64(s) >> 40) * (1.0f / 16777216.0f); }
float rng_normal(uint64_t *s);
