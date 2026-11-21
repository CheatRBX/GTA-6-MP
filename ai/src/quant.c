/* Квантование: f32 -> q8 (int8, ~4x меньше) или q4 (4 бита, ~7x меньше). */
#include "model.h"
#include "train.h"
#include <math.h>
#include <string.h>

static void wr(FILE *f, const void *p, size_t n, const char *path) {
    if (n && fwrite(p, 1, n, f) != n) die("запись %s (кончилось место?)", path);
}

int quantize_run(const char *in, const char *out, const char *type) {
    int qt = strcmp(type, "q8") == 0 ? QT_Q8 : strcmp(type, "q4") == 0 ? QT_Q4 : strcmp(type, "f32") == 0 ? QT_F32 : -1;
    if (qt < 0) die("тип должен быть q8 или q4");
    ModelFile mf;
    model_open(&mf, in);
    if (mf.h.qtype != QT_F32) die("%s уже квантована (%s); квантуй исходный f32-файл", in, qtype_name(mf.h.qtype));
    const Config *c = &mf.c;
    char tmp[1100];
    snprintf(tmp, sizeof tmp, "%s.tmp", out);
    ensure_parent_dir(out);
    FILE *f = fopen(tmp, "wb");
    if (!f) die("не создать %s", tmp);
    FileHeader h = mf.h;
    h.qtype = qt;
    h.tok_offset = 0;
    wr(f, &h, sizeof h, tmp);
    const float *p = (const float *)(mf.data + sizeof(FileHeader));
    double err2 = 0, sig2 = 0;
    int mx = c->dim > c->hidden_dim ? c->dim : c->hidden_dim;
    if (c->vocab_size > mx) mx = c->vocab_size;
    float *deq = xmalloc(sizeof(float) * mx);
    printf("квантование %s -> %s (%s)\n", in, out, type);
    for (int t = 0; t < T_COUNT; t++) {
        int cnt, rows, cols;
        tensor_shape(c, t, &cnt, &rows, &cols);
        size_t numel = (size_t)cnt * rows * cols;
        if (!tensor_is_matrix(t) || qt == QT_F32) { wr(f, p, numel * 4, tmp); p += numel; continue; }
        size_t ngroups = numel / QGS;
        size_t dbytes = qt == QT_Q8 ? numel : numel / 2;
        uint8_t *data = xmalloc(dbytes);
        float *sc = xmalloc(sizeof(float) * ngroups);
        size_t nrows = (size_t)cnt * rows;
        for (size_t r = 0; r < nrows; r++) {
            const float *x = p + r * cols;
            float *s = sc + r * (cols / QGS);
            Mat m = {qt, 1, cols, NULL, s};
            if (qt == QT_Q8) { quantize_q8(x, (int8_t *)data + r * cols, s, cols); m.w = (int8_t *)data + r * cols; }
            else { quantize_q4(x, data + r * cols / 2, s, cols); m.w = data + r * cols / 2; }
            if (r % 7 == 0) { /* оценка ошибки на части строк */
                dequantize_row(&m, 0, deq);
                for (int i = 0; i < cols; i++) { double d = deq[i] - x[i]; err2 += d * d; sig2 += (double)x[i] * x[i]; }
            }
        }
        wr(f, data, dbytes, tmp);
        wr(f, sc, ngroups * 4, tmp);
        free(data); free(sc);
        p += numel;
    }
    if (mf.h.tok_offset > 0) {
        long pos = ftell(f);
        h.tok_offset = pos;
        wr(f, mf.data + mf.h.tok_offset, (size_t)mf.h.tok_size, tmp);
        fseek(f, 0, SEEK_SET);
        wr(f, &h, sizeof h, tmp);
    }
    long fsz;
    fseek(f, 0, SEEK_END);
    fsz = ftell(f);
    if (fclose(f) != 0) die("запись %s", tmp);
    if (rename(tmp, out) != 0) die("rename %s", out);
    printf("готово: %.1f МБ -> %.1f МБ (в %.1f раза меньше), относительная ошибка весов %.2f%%\n",
           mf.size / 1048576.0, fsz / 1048576.0, (double)mf.size / fsz, sig2 > 0 ? 100.0 * sqrt(err2 / sig2) : 0.0);
    free(deq);
    model_close(&mf);
    return 0;
}
