#include "corax.h"
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>

void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "ошибка: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

void *xmalloc(size_t n) {
    void *p = NULL;
    if (n == 0) n = 64;
    if (posix_memalign(&p, 64, (n + 63) & ~(size_t)63) != 0 || !p)
        die("не хватает памяти (%.1f МБ)", n / 1048576.0);
    return p;
}

void *xcalloc(size_t n) {
    void *p = xmalloc(n);
    memset(p, 0, n);
    return p;
}

void ensure_parent_dir(const char *path) {
    char buf[1024];
    snprintf(buf, sizeof buf, "%s", path);
    for (char *p = buf + 1; *p; p++) {
        if (*p != '/') continue;
        *p = 0;
        mkdir(buf, 0755); /* уже есть — не страшно */
        *p = '/';
    }
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int g_color = 1;

void cprintf(const char *fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) return;
    if (n >= (int)sizeof buf) n = sizeof buf - 1;
    if (!g_color) { /* вырезаем ESC[...m */
        int w = 0;
        for (int i = 0; i < n; i++) {
            if (buf[i] == 0x1b && i + 1 < n && buf[i + 1] == '[') {
                int j = i + 2;
                while (j < n && !((buf[j] >= 'A' && buf[j] <= 'Z') || (buf[j] >= 'a' && buf[j] <= 'z'))) j++;
                i = j;
                continue;
            }
            buf[w++] = buf[i];
        }
        n = w;
    }
    fwrite(buf, 1, n, stdout);
}

const char *qtype_name(int qt) {
    return qt == QT_F32 ? "f32" : qt == QT_Q8 ? "q8" : qt == QT_Q4 ? "q4" : "?";
}

float rng_normal(uint64_t *s) {
    float u1 = rng_f32(s), u2 = rng_f32(s);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

void header_to_config(const FileHeader *h, Config *c) {
    c->dim = h->dim; c->n_layers = h->n_layers; c->n_heads = h->n_heads;
    c->n_kv_heads = h->n_kv_heads; c->hidden_dim = h->hidden_dim;
    c->vocab_size = h->vocab_size; c->seq_len = h->seq_len;
    c->rope_theta = h->rope_theta; c->norm_eps = h->norm_eps;
}

void config_to_header(const Config *c, FileHeader *h) {
    memset(h, 0, sizeof(*h));
    h->magic = CRX_MAGIC; h->version = CRX_VERSION;
    h->dim = c->dim; h->n_layers = c->n_layers; h->n_heads = c->n_heads;
    h->n_kv_heads = c->n_kv_heads; h->hidden_dim = c->hidden_dim;
    h->vocab_size = c->vocab_size; h->seq_len = c->seq_len;
    h->rope_theta = c->rope_theta; h->norm_eps = c->norm_eps;
    h->group_size = QGS;
}

int config_validate(const Config *c, char *err, size_t n) {
    if (c->dim <= 0 || c->n_layers <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0 ||
        c->hidden_dim <= 0 || c->vocab_size <= 0 || c->seq_len <= 0) {
        snprintf(err, n, "размеры должны быть положительными"); return 0;
    }
    if (c->dim % c->n_heads) { snprintf(err, n, "dim должен делиться на n_heads"); return 0; }
    if (c->n_heads % c->n_kv_heads) { snprintf(err, n, "n_heads должен делиться на n_kv_heads"); return 0; }
    if (cfg_head_dim(c) % 2) { snprintf(err, n, "head_dim должен быть чётным"); return 0; }
    if (c->dim % QGS || c->hidden_dim % QGS) {
        snprintf(err, n, "dim и hidden_dim должны делиться на %d (квантование)", QGS); return 0;
    }
    if (c->vocab_size > 65535) { snprintf(err, n, "словарь больше 65535"); return 0; }
    return 1;
}

void config_print(const Config *c, FILE *f) {
    fprintf(f, "  dim=%d слоёв=%d голов=%d kv_голов=%d hidden=%d словарь=%d контекст=%d\n",
            c->dim, c->n_layers, c->n_heads, c->n_kv_heads, c->hidden_dim, c->vocab_size, c->seq_len);
    fprintf(f, "  параметров: %.2f млн\n", params_count(c) / 1e6);
}

/* Понятное сообщение, когда модели нет: что лежит рядом и что делать. */
static void model_missing(const char *path) {
    char dir[1024];
    snprintf(dir, sizeof dir, "%s", path);
    char *slash = strrchr(dir, '/');
    if (slash) *slash = 0; else snprintf(dir, sizeof dir, ".");
    fprintf(stderr, "\nМодели «%s» пока нет.\n", path);
    DIR *d = opendir(dir);
    int found = 0, has_base = 0;
    if (d) {
        struct dirent *e;
        while ((e = readdir(d))) {
            size_t l = strlen(e->d_name);
            if (l > 4 && strcmp(e->d_name + l - 4, ".bin") == 0) {
                if (!found) fprintf(stderr, "  есть в %s/:", dir);
                fprintf(stderr, "%s %s", found ? "," : "", e->d_name);
                found++;
                if (!strcmp(e->d_name, "base.bin")) has_base = 1;
            }
        }
        closedir(d);
    }
    fprintf(stderr, found ? "\n" : "  в папке %s моделей ещё нет.\n", dir);
    fprintf(stderr, "\nЧат-модель появляется в самом конце quickstart.sh (обучение → дообучение → сжатие).\n");
    fprintf(stderr, "Если обучение прервалось — просто запусти снова: bash scripts/quickstart.sh (продолжит с сохранения).\n");
    if (has_base) fprintf(stderr, "Уже сейчас можно попробовать базовую модель: ./corax gen -m %s/base.bin -p \"Однажды\"\n", dir);
    exit(1);
}

void model_open(ModelFile *mf, const char *path) {
    memset(mf, 0, sizeof(*mf));
    mf->fd = open(path, O_RDONLY);
    if (mf->fd < 0 && errno == ENOENT) model_missing(path);
    if (mf->fd < 0) die("не открыть модель %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(mf->fd, &st) != 0) die("stat %s", path);
    mf->size = (size_t)st.st_size;
    if (mf->size < sizeof(FileHeader)) die("%s: файл слишком мал", path);
    mf->data = mmap(NULL, mf->size, PROT_READ, MAP_PRIVATE, mf->fd, 0);
    if (mf->data == MAP_FAILED) die("mmap %s", path);
    memcpy(&mf->h, mf->data, sizeof(FileHeader));
    if (mf->h.magic != CRX_MAGIC) die("%s: это не модель Corax", path);
    if (mf->h.version != CRX_VERSION) die("%s: неизвестная версия %u", path, mf->h.version);
    if (mf->h.group_size != QGS) die("%s: группа квантования %d не поддерживается", path, mf->h.group_size);
    header_to_config(&mf->h, &mf->c);
    char err[128];
    if (!config_validate(&mf->c, err, sizeof err)) die("%s: %s", path, err);
}

void model_close(ModelFile *mf) {
    if (mf->data && mf->data != MAP_FAILED) munmap(mf->data, mf->size);
    if (mf->fd > 0) close(mf->fd);
    memset(mf, 0, sizeof(*mf));
}

/* ---------- пресеты ---------- */
typedef struct { const char *name; int D, L, H, KV, F, T; const char *note; } Preset;
static const Preset PRESETS[] = {
    {"nano",   256,  6,  8, 4,  704,  256, "~6.5 млн, телефон: 1–2 часа обучения"},
    {"mini",   384,  8,  6, 2, 1024,  512, "~16 млн, телефон: несколько часов"},
    {"small",  512, 10,  8, 4, 1376,  512, "~33 млн, телефон: ночь и больше"},
    {"medium", 768, 12, 12, 4, 2048, 1024, "~85 млн, умнее; учить на ПК/Colab, ~70+ ток/с"},
    {"large", 1024, 16, 16, 4, 2816, 1024, "~190 млн, только ПК/GPU, на телефоне q4"},
};

int preset_get(const char *name, Config *c) {
    for (size_t i = 0; i < sizeof PRESETS / sizeof *PRESETS; i++) {
        const Preset *p = &PRESETS[i];
        if (strcmp(p->name, name) == 0) {
            memset(c, 0, sizeof(*c));
            c->dim = p->D; c->n_layers = p->L; c->n_heads = p->H; c->n_kv_heads = p->KV;
            c->hidden_dim = p->F; c->seq_len = p->T;
            c->rope_theta = 10000.0f; c->norm_eps = 1e-5f;
            return 1;
        }
    }
    return 0;
}

void preset_list(FILE *f) {
    for (size_t i = 0; i < sizeof PRESETS / sizeof *PRESETS; i++) {
        const Preset *p = &PRESETS[i];
        fprintf(f, "  %-7s dim=%-4d слоёв=%-2d голов=%-2d контекст=%-4d  %s\n",
                p->name, p->D, p->L, p->H, p->T, p->note);
    }
}
