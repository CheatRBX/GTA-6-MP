#include "corax.h"
#include "tokenizer.h"
#include <string.h>

/* ---- хеш-таблица слияний ---- */
static inline uint32_t hash64(uint64_t k) {
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33;
    return (uint32_t)k;
}
static void hput(Tokenizer *t, uint32_t a, uint32_t b, int32_t rank) {
    uint64_t key = ((uint64_t)a << 32) | b;
    uint32_t i = hash64(key) & t->hmask;
    while (t->hvals[i] >= 0) {
        if (t->hkeys[i] == key) return; /* дубликат — оставляем первый ранг */
        i = (i + 1) & t->hmask;
    }
    t->hkeys[i] = key; t->hvals[i] = rank;
}
static inline int32_t hget(const Tokenizer *t, uint32_t a, uint32_t b) {
    uint64_t key = ((uint64_t)a << 32) | b;
    uint32_t i = hash64(key) & t->hmask;
    while (t->hvals[i] >= 0) {
        if (t->hkeys[i] == key) return t->hvals[i];
        i = (i + 1) & t->hmask;
    }
    return -1;
}

static int rd32(const uint8_t *d, size_t size, size_t *off, int32_t *v) {
    if (*off + 4 > size) return 0;
    memcpy(v, d + *off, 4); *off += 4; return 1;
}

int tok_load(Tokenizer *t, const uint8_t *d, size_t size) {
    memset(t, 0, sizeof(*t));
    size_t off = 0;
    int32_t magic, ver, vs, nm, ns;
    if (!rd32(d, size, &off, &magic) || (uint32_t)magic != CRX_TOK_MAGIC) return 0;
    if (!rd32(d, size, &off, &ver) || ver != 1) return 0;
    if (!rd32(d, size, &off, &vs) || !rd32(d, size, &off, &nm) || !rd32(d, size, &off, &ns)) return 0;
    if (vs != 256 + nm + ns || nm < 0 || ns < SP_COUNT) return 0;
    t->vocab_size = vs; t->n_merges = nm; t->n_special = ns;
    t->piece = xcalloc(sizeof(uint8_t *) * vs);
    t->piece_len = xcalloc(sizeof(int) * vs);
    uint32_t cap = 1;
    while (cap < (uint32_t)(nm * 2 + 16)) cap <<= 1;
    t->hmask = cap - 1;
    t->hkeys = xmalloc(sizeof(uint64_t) * cap);
    t->hvals = xmalloc(sizeof(int32_t) * cap);
    for (uint32_t i = 0; i < cap; i++) t->hvals[i] = -1;
    for (int i = 0; i < 256; i++) {
        t->piece[i] = xmalloc(1); t->piece[i][0] = (uint8_t)i; t->piece_len[i] = 1;
    }
    for (int i = 0; i < nm; i++) {
        int32_t a, b;
        if (!rd32(d, size, &off, &a) || !rd32(d, size, &off, &b)) return 0;
        int id = 256 + i;
        if (a < 0 || b < 0 || a >= id || b >= id) return 0;
        int la = t->piece_len[a], lb = t->piece_len[b];
        t->piece[id] = xmalloc(la + lb);
        memcpy(t->piece[id], t->piece[a], la);
        memcpy(t->piece[id] + la, t->piece[b], lb);
        t->piece_len[id] = la + lb;
        hput(t, a, b, i);
    }
    for (int i = 0; i < ns; i++) {
        int32_t len;
        if (!rd32(d, size, &off, &len) || len < 0 || off + len > size) return 0;
        int id = 256 + nm + i;
        t->piece[id] = xmalloc(len + 1);
        memcpy(t->piece[id], d + off, len);
        t->piece_len[id] = len;
        off += len;
    }
    for (int i = 0; i < SP_COUNT; i++) t->special[i] = 256 + nm + i;
    return 1;
}

void tok_free(Tokenizer *t) {
    if (!t->piece) return;
    for (int i = 0; i < t->vocab_size; i++) free(t->piece[i]);
    free(t->piece); free(t->piece_len); free(t->hkeys); free(t->hvals);
    memset(t, 0, sizeof(*t));
}

const uint8_t *tok_piece(const Tokenizer *t, int id, int *len) {
    if (id < 0 || id >= t->vocab_size) { *len = 0; return (const uint8_t *)""; }
    *len = t->piece_len[id];
    return t->piece[id];
}

int tok_is_special(const Tokenizer *t, int id) { return id >= 256 + t->n_merges; }

/* ---- пред-токенизация: должна в точности совпадать с PAT в tools/tokenizer.py ---- */
enum { CL_S, CL_W, CL_P };

static int cp_class(uint32_t c) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x0b || c == 0x0c) return CL_S;
    if (c < 128) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_') return CL_W;
        return CL_P;
    }
    if (c >= 0xC0 && c <= 0x1FFF && c != 0xD7 && c != 0xF7) return CL_W;
    if (c >= 0x2070 && c <= 0x209F) return CL_W;
    if (c >= 0x20D0 && c <= 0x2FFF) return CL_W;
    if (c >= 0x3040) return CL_W;
    return CL_P;
}

/* Декодирует один символ UTF-8. Некорректный байт — класс P, длина 1. */
static int utf8_next(const uint8_t *s, int n, int i, int *cls) {
    uint8_t b = s[i];
    uint32_t cp; int len;
    if (b < 0x80) { cp = b; len = 1; }
    else if ((b & 0xE0) == 0xC0) { cp = b & 0x1F; len = 2; }
    else if ((b & 0xF0) == 0xE0) { cp = b & 0x0F; len = 3; }
    else if ((b & 0xF8) == 0xF0) { cp = b & 0x07; len = 4; }
    else { *cls = CL_P; return 1; }
    if (i + len > n) { *cls = CL_P; return 1; }
    for (int k = 1; k < len; k++) {
        if ((s[i + k] & 0xC0) != 0x80) { *cls = CL_P; return 1; }
        cp = (cp << 6) | (s[i + k] & 0x3F);
    }
    *cls = cp_class(cp);
    return len;
}

/* BPE внутри одного куска: каждый раз сливаем пару с наименьшим рангом (все вхождения). */
static int bpe_word(const Tokenizer *t, const uint8_t *s, int n, int *out, int max_out) {
    int stack_ids[256];
    int *ids = n <= 256 ? stack_ids : xmalloc(sizeof(int) * n);
    int m = n;
    for (int i = 0; i < n; i++) ids[i] = s[i];
    while (m >= 2) {
        int32_t best = -1;
        for (int i = 0; i + 1 < m; i++) {
            int32_t r = hget(t, ids[i], ids[i + 1]);
            if (r >= 0 && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;
        int a = 0, b = 0;
        /* восстановить пару по рангу */
        for (int i = 0; i + 1 < m; i++)
            if (hget(t, ids[i], ids[i + 1]) == best) { a = ids[i]; b = ids[i + 1]; break; }
        int w = 0;
        for (int i = 0; i < m;) {
            if (i + 1 < m && ids[i] == a && ids[i + 1] == b) { ids[w++] = 256 + best; i += 2; }
            else ids[w++] = ids[i++];
        }
        m = w;
    }
    int k = m < max_out ? m : max_out;
    memcpy(out, ids, sizeof(int) * k);
    if (ids != stack_ids) free(ids);
    return k;
}

int tok_encode(const Tokenizer *t, const char *text, int *out, int max_out) {
    const uint8_t *s = (const uint8_t *)text;
    int n = (int)strlen(text), nout = 0, i = 0;
    while (i < n && nout < max_out) {
        int c0, l0 = utf8_next(s, n, i, &c0);
        int c1 = -1, l1 = 0;
        if (i + l0 < n) l1 = utf8_next(s, n, i + l0, &c1);
        (void)l1;
        int j, cls;
        if (s[i] == ' ' && (c1 == CL_W || c1 == CL_P)) { cls = c1; j = i + 1; }
        else if (c0 == CL_W || c0 == CL_P) { cls = c0; j = i; }
        else cls = CL_S, j = i;
        if (cls != CL_S) {
            while (j < n) {
                int c, l = utf8_next(s, n, j, &c);
                if (c != cls) break;
                j += l;
            }
        } else {
            int start = i, cnt = 0, last = i;
            j = i;
            while (j < n) {
                int c, l = utf8_next(s, n, j, &c);
                if (c != CL_S) break;
                last = j; j += l; cnt++;
            }
            if (j < n && cnt >= 2) j = last; /* последний пробел отдаём следующему слову */
            (void)start;
        }
        nout += bpe_word(t, s + i, j - i, out + nout, max_out - nout);
        i = j;
    }
    return nout;
}

/* ---- потоковый вывод UTF-8 ---- */
static int utf8_need(uint8_t b) {
    if (b < 0x80) return 1;
    if ((b & 0xE0) == 0xC0) return 2;
    if ((b & 0xF0) == 0xE0) return 3;
    if ((b & 0xF8) == 0xF0) return 4;
    return 1; /* мусор — выводим как есть */
}

void utf8_put(Utf8Stream *st, const uint8_t *p, int n, void (*emit)(const uint8_t *, int, void *), void *ud) {
    for (int i = 0; i < n; i++) {
        st->buf[st->n++] = p[i];
        int need = utf8_need(st->buf[0]);
        if (st->n >= need) { emit(st->buf, st->n, ud); st->n = 0; }
        else if (st->n > 1 && (p[i] & 0xC0) != 0x80) { /* оборванная последовательность */
            emit(st->buf, st->n, ud); st->n = 0;
        }
    }
}
