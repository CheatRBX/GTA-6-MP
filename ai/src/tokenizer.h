#pragma once
#include <stdint.h>
#include <stddef.h>

/* Байтовый BPE-токенизатор (обучается с нуля скриптом tools/tokenizer.py).
 * id 0..255 — байты, 256.. — слияния, в конце — спецтокены. */
enum { SP_BOS = 0, SP_EOS, SP_USER, SP_ASSISTANT, SP_SYSTEM, SP_COUNT };

typedef struct {
    int vocab_size, n_merges, n_special;
    uint8_t **piece;     /* байты каждого токена */
    int *piece_len;
    uint64_t *hkeys;     /* хеш (a,b) -> ранг слияния */
    int32_t *hvals;
    uint32_t hmask;
    int special[SP_COUNT];
} Tokenizer;

int  tok_load(Tokenizer *t, const uint8_t *data, size_t size); /* 1 = ок */
void tok_free(Tokenizer *t);
/* Кодирует текст без спецтокенов. Возвращает число токенов (<= max_out). */
int  tok_encode(const Tokenizer *t, const char *text, int *out, int max_out);
const uint8_t *tok_piece(const Tokenizer *t, int id, int *len);
int  tok_is_special(const Tokenizer *t, int id);

/* Вывод UTF-8 по кусочкам: токен может оборвать многобайтовый символ. */
typedef struct { uint8_t buf[8]; int n; } Utf8Stream;
void utf8_put(Utf8Stream *s, const uint8_t *p, int n, void (*emit)(const uint8_t *, int, void *), void *ud);
