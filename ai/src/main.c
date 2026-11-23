/* Corax LM — CLI: chat / gen / bench / info / train / finetune / quantize. */
#include "corax.h"
#include "model.h"
#include "pool.h"
#include "train.h"
#include "gemm.h"
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* ================= терминал ================= */
static int g_bar = 0, g_rows = 0, g_cols = 80;
static volatile sig_atomic_t g_interrupt = 0, g_winch = 0;

#define C_RESET  (g_color ? "\033[0m" : "")
#define C_BOLD   (g_color ? "\033[1m" : "")
#define C_DIM    (g_color ? "\033[2m" : "")
#define C_ACCENT (g_color ? "\033[38;5;141m" : "") /* фиолетовый Corax */
#define C_CYAN   (g_color ? "\033[38;5;80m" : "")
#define C_GREEN  (g_color ? "\033[32m" : "")
#define C_YELLOW (g_color ? "\033[33m" : "")
#define C_RED    (g_color ? "\033[31m" : "")

static void on_sigint(int s) { (void)s; g_interrupt = 1; }
static void on_winch(int s) { (void)s; g_winch = 1; }

static void term_size(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_row > 4) { g_rows = ws.ws_row; g_cols = ws.ws_col; }
    else g_rows = 0;
}

/* Нижняя строка состояния через область прокрутки (как в терминальных ассистентах). */
static void bar_setup(void) {
    term_size();
    if (g_rows < 6) { g_bar = 0; return; }
    printf("\n\033[1A\0337\033[1;%dr\0338", g_rows - 1);
    fflush(stdout);
}

static void bar_reset(void) {
    if (!g_bar) return;
    term_size();
    printf("\0337\033[r\033[%d;1H\033[2K\0338", g_rows > 0 ? g_rows : 1);
    fflush(stdout);
}

static int utf8_cols(const char *s, int maxcols, int *bytes) {
    int cols = 0, i = 0;
    while (s[i] && cols < maxcols) {
        unsigned char ch = (unsigned char)s[i];
        int len = ch < 0x80 ? 1 : (ch & 0xE0) == 0xC0 ? 2 : (ch & 0xF0) == 0xE0 ? 3 : 4;
        i += len;
        cols++;
    }
    *bytes = i;
    return cols;
}

static void bar_draw(const char *text) {
    if (!g_bar) return;
    if (g_winch) { g_winch = 0; bar_setup(); }
    int nb, w = utf8_cols(text, g_cols - 1, &nb);
    printf("\0337\033[%d;1H\033[2K%s\033[48;5;236m\033[38;5;252m %.*s%*s%s\0338", g_rows, C_RESET, nb, text,
           g_cols - 2 - w > 0 ? g_cols - 2 - w : 0, "", C_RESET);
    fflush(stdout);
}

static void emit_stdout(const uint8_t *p, int n, void *ud) { (void)ud; fwrite(p, 1, n, stdout); }

/* ================= аргументы ================= */
typedef struct { int argc; char **argv; } Args;

static const char *arg_str(Args *a, const char *name, const char *def) {
    for (int i = 0; i < a->argc - 1; i++)
        if (strcmp(a->argv[i], name) == 0) return a->argv[i + 1];
    return def;
}
static int arg_flag(Args *a, const char *name) {
    for (int i = 0; i < a->argc; i++)
        if (strcmp(a->argv[i], name) == 0) return 1;
    return 0;
}
static int arg_int(Args *a, const char *name, int def) { const char *s = arg_str(a, name, NULL); return s ? atoi(s) : def; }
static float arg_float(Args *a, const char *name, float def) { const char *s = arg_str(a, name, NULL); return s ? (float)atof(s) : def; }
static const char *arg_model(Args *a) {
    const char *m = arg_str(a, "-m", NULL);
    if (!m) m = arg_str(a, "--model", NULL);
    if (!m) die("укажи модель: -m models/model-q8.bin");
    return m;
}

static void setup_threads(Args *a) {
    int t = arg_int(a, "-t", arg_int(a, "--threads", 0));
    pool_init(t > 0 ? t : pool_default_threads());
}

static void setup_sampler(Args *a, Sampler *sp) {
    sampler_default(sp);
    sp->temperature = arg_float(a, "--temp", sp->temperature);
    sp->top_p = arg_float(a, "--top-p", sp->top_p);
    sp->top_k = arg_int(a, "--top-k", sp->top_k);
    sp->rep_penalty = arg_float(a, "--rep", sp->rep_penalty);
    const char *seed = arg_str(a, "--seed", NULL);
    sp->rng = seed ? strtoull(seed, NULL, 10) : (uint64_t)(now_sec() * 1e6);
    if (sp->rng == 0) sp->rng = 1;
}

/* ================= info ================= */
static void print_model_card(const Model *m, const char *path) {
    const char *slash = strrchr(path, '/');
    if (slash) path = slash + 1;
    double mb = model_weight_bytes(m) / 1048576.0;
    printf("%s◆ Corax LM%s  %s%s%s\n", C_ACCENT, C_RESET, C_BOLD, path, C_RESET);
    printf("  %s%.1f млн параметров%s · %s · веса %.1f МБ · контекст %d\n", C_BOLD, params_count(&m->c) / 1e6, C_RESET,
           qtype_name(m->qtype), mb, m->c.seq_len);
    printf("  %sслоёв %d · dim %d · голов %d/%d · словарь %d · обучено шагов %lld (%.1f млн токенов)%s\n", C_DIM,
           m->c.n_layers, m->c.dim, m->c.n_heads, m->c.n_kv_heads, m->c.vocab_size, (long long)m->train_steps,
           m->train_tokens / 1e6, C_RESET);
}

static int cmd_info(Args *a) {
    Model m;
    model_load(&m, arg_model(a));
    print_model_card(&m, arg_model(a));
    printf("  токенизатор: %s\n", m.has_tok ? "вшит" : "нет");
    model_free(&m);
    return 0;
}

/* ================= bench ================= */
static double bench_once(const Model *m, int ntok, double *prefill_tps) {
    State s;
    state_init(&s, &m->c, 0);
    uint64_t rng = 42;
    int n = ntok < s.seq_len - 1 ? ntok : s.seq_len - 1;
    int npre = n / 2 < 64 ? n / 2 : 64;
    double t0 = now_sec();
    int tok = 1;
    for (int p = 0; p < npre; p++) forward(m, &s, (int)(rng_u64(&rng) % m->c.vocab_size), p);
    double t1 = now_sec();
    for (int p = npre; p < n; p++) {
        float *lg = forward(m, &s, tok, p);
        int best = 0;
        for (int i = 1; i < m->c.vocab_size; i++) if (lg[i] > lg[best]) best = i;
        tok = best;
    }
    double t2 = now_sec();
    if (prefill_tps) *prefill_tps = npre / (t1 - t0);
    state_free(&s);
    return (n - npre) / (t2 - t1);
}

/* Модель из пресета со случайными весами — чтобы узнать скорость ДО обучения. */
static void model_random(Model *m, const char *preset, int vocab, const char *quant) {
    Config c;
    if (!preset_get(preset, &c)) { preset_list(stderr); die("неизвестный пресет '%s'", preset); }
    c.vocab_size = vocab;
    char err[128];
    if (!config_validate(&c, err, sizeof err)) die("конфиг: %s", err);
    size_t np = params_count(&c);
    float *p = xmalloc(4 * np);
    uint64_t r = 7;
    for (size_t i = 0; i < np; i++) p[i] = (rng_f32(&r) - 0.5f) * 0.05f;
    model_from_f32(m, &c, p);
    for (int i = 0; i < c.n_layers * c.dim; i++) { ((float *)m->attn_norm)[i] = 1; ((float *)m->ffn_norm)[i] = 1; }
    for (int i = 0; i < c.dim; i++) ((float *)m->final_norm)[i] = 1;
    int qt = !strcmp(quant, "q4") ? QT_Q4 : !strcmp(quant, "f32") ? QT_F32 : QT_Q8;
    model_quantize_mem(m, qt);
}

static int cmd_bench(Args *a) {
    Model m;
    const char *preset = arg_str(a, "--preset", NULL);
    char label[256];
    if (preset) {
        snprintf(label, sizeof label, "пресет %s (случайные веса — только замер скорости)", preset);
        model_random(&m, preset, arg_int(a, "--vocab", 16384), arg_str(a, "--quant", "q8"));
    } else {
        snprintf(label, sizeof label, "%s", arg_model(a));
        model_load(&m, label);
    }
    const char *path = label;
    print_model_card(&m, path);
    int n = arg_int(a, "-n", 256);
    printf("  процессор: %s\n", pool_cpu_info());
    if (arg_flag(a, "--sweep")) {
        int best_t = 1;
        double best = 0;
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        for (int t = 1; t <= ncpu && t <= 16; t++) {
            pool_init(t);
            bench_once(&m, 32, NULL);
            double tps = bench_once(&m, n, NULL);
            printf("  потоков %2d: %s%7.1f ток/с%s\n", t, C_GREEN, tps, C_RESET);
            if (tps > best) { best = tps; best_t = t; }
        }
        printf("%s  лучше всего: -t %d (%.1f ток/с)%s\n", C_BOLD, best_t, best, C_RESET);
    } else {
        setup_threads(a);
        bench_once(&m, 32, NULL); /* прогрев */
        double pre, tps = bench_once(&m, n, &pre);
        printf("  потоков %d\n", pool_size());
        printf("  чтение промпта: %s%.1f ток/с%s\n", C_CYAN, pre, C_RESET);
        printf("  генерация:      %s%s%.1f ток/с%s (%.2f мс/токен)\n", C_BOLD, C_GREEN, tps, C_RESET, 1000.0 / tps);
    }
    model_free(&m);
    return 0;
}

/* ================= генерация ================= */
typedef struct {
    Model m;
    State s;
    Sampler sp;
    int *hist;
    int nhist, pos;
    char system[4096];
    int stats;
    int max_new;
    double last_tps;
} Session;

static void sess_feed(Session *ss, int tok) {
    forward(&ss->m, &ss->s, tok, ss->pos++);
    ss->hist[ss->nhist++] = tok;
}

static void status(Session *ss, const char *what, double tps) {
    if (!g_bar) return;
    char buf[512];
    if (g_cols >= 64)
        snprintf(buf, sizeof buf, "◆ Corax · %.0fM · %s · потоков %d │ %s │ %.1f ток/с │ контекст %d/%d",
                 params_count(&ss->m.c) / 1e6, qtype_name(ss->m.qtype), pool_size(), what, tps, ss->pos, ss->s.seq_len);
    else
        snprintf(buf, sizeof buf, "◆ %s │ %.0f ток/с │ %d/%d", what, tps, ss->pos, ss->s.seq_len);
    bar_draw(buf);
}

static const char *SPIN[] = {"◐", "◓", "◑", "◒"};

/* Подаёт токены с индикатором «читаю запрос». Возвращает время. */
static double feed_tokens(Session *ss, const int *ids, int n, int show) {
    double t0 = now_sec(), last = 0;
    show = show && g_color;
    for (int i = 0; i < n; i++) {
        sess_feed(ss, ids[i]);
        double now = now_sec();
        if (show && now - last > 0.08) {
            last = now;
            printf("\r\033[K%s%s Читаю запрос… %d/%d токенов%s", C_ACCENT, SPIN[(int)(now * 10) % 4], i + 1, n, C_RESET);
            fflush(stdout);
            status(ss, "читаю запрос", (i + 1) / (now - t0 + 1e-9));
        }
    }
    if (show) { printf("\r\033[K"); fflush(stdout); }
    return now_sec() - t0;
}

/* Генерирует ответ до EOS. Возвращает число токенов. */
static int generate(Session *ss, int stop_on_eos, double *gen_time) {
    const Tokenizer *tk = &ss->m.tok;
    int eos = tk->special[SP_EOS], V = ss->m.c.vocab_size, n = 0;
    Utf8Stream us = {0};
    double t0 = now_sec(), last = 0;
    float *lg = ss->s.logits;
    g_interrupt = 0;
    while (n < ss->max_new) {
        if (ss->pos >= ss->s.seq_len) { printf("%s [контекст кончился]%s", C_DIM, C_RESET); break; }
        if (g_interrupt) { printf("%s [прервано]%s", C_YELLOW, C_RESET); g_interrupt = 0; break; }
        int tok = sample_token(&ss->sp, &ss->sp.rng, lg, V, ss->hist, ss->nhist);
        if (tok == eos && stop_on_eos) { if (ss->pos < ss->s.seq_len) sess_feed(ss, tok); break; }
        if (!tok_is_special(tk, tok)) {
            int l;
            const uint8_t *p = tok_piece(tk, tok, &l);
            utf8_put(&us, p, l, emit_stdout, NULL);
            fflush(stdout);
        }
        n++;
        lg = forward(&ss->m, &ss->s, tok, ss->pos);
        ss->hist[ss->nhist++] = tok;
        ss->pos++;
        double now = now_sec();
        if (now - last > 0.1) { last = now; status(ss, "генерирую", n / (now - t0 + 1e-9)); }
    }
    *gen_time = now_sec() - t0;
    return n;
}

static void print_stats(Session *ss, int ntok, double gen_t, int nprompt, double pre_t) {
    double tps = ntok / (gen_t + 1e-9);
    ss->last_tps = tps;
    status(ss, "жду ввода", tps);
    if (!ss->stats) return;
    if (g_cols >= 64)
        printf("%s  └ %d ток · %s%.1f ток/с%s%s · чтение %d ток за %.2fс · контекст %d/%d%s\n", C_DIM, ntok, C_GREEN, tps,
               C_RESET, C_DIM, nprompt, pre_t, ss->pos, ss->s.seq_len, C_RESET);
    else
        printf("%s  └ %d ток · %s%.0f ток/с%s%s · %d/%d%s\n", C_DIM, ntok, C_GREEN, tps, C_RESET, C_DIM, ss->pos,
               ss->s.seq_len, C_RESET);
}

static void sess_open(Session *ss, Args *a) {
    memset(ss, 0, sizeof(*ss));
    const char *path = arg_model(a);
    model_load(&ss->m, path);
    if (!ss->m.has_tok) die("в модели нет токенизатора");
    setup_threads(a);
    setup_sampler(a, &ss->sp);
    state_init(&ss->s, &ss->m.c, arg_int(a, "--ctx", 0));
    ss->hist = xmalloc(sizeof(int) * (ss->s.seq_len + 8));
    ss->stats = !arg_flag(a, "--no-stats");
    ss->max_new = arg_int(a, "-n", 100000);
}

static int cmd_gen(Args *a) {
    Session ss;
    sess_open(&ss, a);
    const char *prompt = arg_str(a, "-p", arg_str(a, "--prompt", ""));
    int ids[4096];
    ids[0] = ss.m.tok.special[SP_BOS];
    int n = 1 + tok_encode(&ss.m.tok, prompt, ids + 1, ss.s.seq_len - 2);
    if (ss.max_new == 100000) ss.max_new = 256;
    double pre = feed_tokens(&ss, ids, n, 0);
    printf("%s%s%s", C_BOLD, prompt, C_RESET);
    fflush(stdout);
    double gt;
    int ng = generate(&ss, 1, &gt);
    printf("\n");
    print_stats(&ss, ng, gt, n, pre);
    return 0;
}

/* ================= чат ================= */
static void chat_reset(Session *ss, int show) {
    ss->pos = 0; ss->nhist = 0;
    const Tokenizer *tk = &ss->m.tok;
    int ids[4096], n = 0;
    ids[n++] = tk->special[SP_BOS];
    if (ss->system[0]) {
        ids[n++] = tk->special[SP_SYSTEM];
        n += tok_encode(tk, ss->system, ids + n, ss->s.seq_len / 2);
        ids[n++] = tk->special[SP_EOS];
    }
    feed_tokens(ss, ids, n, show);
}

static void chat_help(void) {
    printf("%s  Команды:%s\n", C_BOLD, C_RESET);
    printf("    /reset          очистить контекст (забыть разговор)\n");
    printf("    /system ТЕКСТ   задать системную подсказку (характер ассистента)\n");
    printf("    /temp 0.7       температура (0 = всегда самый вероятный ответ)\n");
    printf("    /topp 0.9       top-p,   /topk 40   top-k,   /rep 1.1   штраф за повторы\n");
    printf("    /stats          вкл/выкл статистику после ответа\n");
    printf("    /info           сведения о модели\n");
    printf("    /exit           выход (или Ctrl+D)\n");
    printf("  %sCtrl+C во время ответа — остановить генерацию. Строка с \\ в конце — перенос.%s\n", C_DIM, C_RESET);
}

static void restore_terminal(void) { bar_reset(); }

static int cmd_chat(Args *a) {
    Session ss;
    term_size();
    sess_open(&ss, a);
    snprintf(ss.system, sizeof ss.system, "%s", arg_str(a, "--system", ""));
    const char *path = arg_model(a);
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_handler = on_sigint;
    sigaction(SIGINT, &sa, NULL); /* без SA_RESTART: Ctrl+C прерывает ввод */
    sa.sa_handler = on_winch;
    sa.sa_flags = SA_RESTART;
    sigaction(SIGWINCH, &sa, NULL);
    g_bar = g_color && isatty(STDOUT_FILENO) && isatty(STDIN_FILENO) && !arg_flag(a, "--no-bar");
    if (g_bar) { bar_setup(); atexit(restore_terminal); }

    printf("%s╭────────────────────────────────────╮%s\n", C_ACCENT, C_RESET);
    printf("%s│%s %s◆ Corax LM%s                         %s│%s\n", C_ACCENT, C_RESET, C_BOLD, C_RESET, C_ACCENT, C_RESET);
    printf("%s│%s %sсвоя нейросеть · офлайн · с нуля%s   %s│%s\n", C_ACCENT, C_RESET, C_DIM, C_RESET, C_ACCENT, C_RESET);
    printf("%s╰────────────────────────────────────╯%s\n", C_ACCENT, C_RESET);
    print_model_card(&ss.m, path);
    printf("  %sпотоков %d · %s%s\n", C_DIM, pool_size(), pool_cpu_info(), C_RESET);
    printf("  %s/help — команды · Ctrl+C — стоп ответа%s\n\n", C_DIM, C_RESET);
    chat_reset(&ss, 1);
    status(&ss, "жду ввода", 0);

    const Tokenizer *tk = &ss.m.tok;
    char line[8192];
    int *ids = xmalloc(sizeof(int) * (ss.s.seq_len + 8));
    for (;;) {
        printf("%s%s❯%s ", C_BOLD, C_ACCENT, C_RESET);
        fflush(stdout);
        size_t len = 0;
        line[0] = 0;
        int eof = 0;
        for (;;) { /* поддержка переноса строки через '\' */
            if (!fgets(line + len, (int)(sizeof line - len), stdin)) {
                if (errno == EINTR && g_interrupt) { g_interrupt = 0; clearerr(stdin); errno = 0; printf("\n"); eof = 2; }
                else eof = 1;
                break;
            }
            len = strlen(line);
            while (len && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = 0;
            if (len && line[len - 1] == '\\' && len < sizeof line - 2) { line[len - 1] = '\n'; printf("%s…%s ", C_DIM, C_RESET); fflush(stdout); continue; }
            break;
        }
        if (eof == 1) { printf("\n"); break; }
        if (eof == 2) continue;
        if (!line[0]) continue;
        if (line[0] == '/') {
            char *arg = strchr(line, ' ');
            if (arg) { *arg++ = 0; while (*arg == ' ') arg++; }
            if (!strcmp(line, "/exit") || !strcmp(line, "/quit") || !strcmp(line, "/q")) break;
            else if (!strcmp(line, "/help")) chat_help();
            else if (!strcmp(line, "/reset")) { chat_reset(&ss, 1); printf("%s  └ контекст очищен%s\n", C_DIM, C_RESET); }
            else if (!strcmp(line, "/system")) {
                snprintf(ss.system, sizeof ss.system, "%s", arg ? arg : "");
                chat_reset(&ss, 1);
                printf("%s  └ системная подсказка %s, контекст очищен%s\n", C_DIM, ss.system[0] ? "задана" : "убрана", C_RESET);
            }
            else if (!strcmp(line, "/temp") && arg) { ss.sp.temperature = (float)atof(arg); printf("%s  └ температура %.2f%s\n", C_DIM, ss.sp.temperature, C_RESET); }
            else if (!strcmp(line, "/topp") && arg) { ss.sp.top_p = (float)atof(arg); printf("%s  └ top-p %.2f%s\n", C_DIM, ss.sp.top_p, C_RESET); }
            else if (!strcmp(line, "/topk") && arg) { ss.sp.top_k = atoi(arg); printf("%s  └ top-k %d%s\n", C_DIM, ss.sp.top_k, C_RESET); }
            else if (!strcmp(line, "/rep") && arg) { ss.sp.rep_penalty = (float)atof(arg); printf("%s  └ штраф повторов %.2f%s\n", C_DIM, ss.sp.rep_penalty, C_RESET); }
            else if (!strcmp(line, "/stats")) { ss.stats = !ss.stats; printf("%s  └ статистика %s%s\n", C_DIM, ss.stats ? "вкл" : "выкл", C_RESET); }
            else if (!strcmp(line, "/info")) print_model_card(&ss.m, path);
            else printf("%s  └ неизвестная команда, /help — список%s\n", C_YELLOW, C_RESET);
            continue;
        }
        int nu = tok_encode(tk, line, ids + 1, ss.s.seq_len);
        ids[0] = tk->special[SP_USER];
        int n = nu + 1;
        ids[n++] = tk->special[SP_EOS];
        ids[n++] = tk->special[SP_ASSISTANT];
        int reserve = ss.s.seq_len / 4 > 32 ? ss.s.seq_len / 4 : 32;
        if (ss.pos + n + reserve > ss.s.seq_len) {
            printf("%s  └ контекст заполнен — начинаю новый%s%s\n", C_DIM, ss.system[0] ? " (системная подсказка сохранена)" : "", C_RESET);
            chat_reset(&ss, 0);
            if (ss.pos + n + reserve > ss.s.seq_len) { /* слишком длинное сообщение: оставляем хвост */
                int keep = ss.s.seq_len - ss.pos - reserve - 3;
                if (keep < 1) keep = 1;
                memmove(ids + 1, ids + 1 + (nu - keep), sizeof(int) * keep);
                n = keep + 1;
                ids[n++] = tk->special[SP_EOS];
                ids[n++] = tk->special[SP_ASSISTANT];
                printf("%s  └ сообщение обрезано до %d токенов%s\n", C_YELLOW, keep, C_RESET);
            }
        }
        double pre = feed_tokens(&ss, ids, n, 1);
        printf("%s●%s ", C_ACCENT, C_RESET);
        fflush(stdout);
        double gt;
        int ng = generate(&ss, 1, &gt);
        printf("\n");
        print_stats(&ss, ng, gt, n, pre);
        printf("\n");
    }
    printf("%s  пока!%s\n", C_DIM, C_RESET);
    return 0;
}

/* ================= обучение ================= */
static void fill_train_opts(Args *a, TrainOpts *o, int finetune) {
    train_opts_default(o);
    if (finetune) {
        o->init = arg_str(a, "--model", arg_str(a, "-m", NULL));
        if (!o->init) die("укажи базовую модель: --model models/base.bin");
        o->steps = 600; o->eval_every = 100; o->save_every = 100; o->sample_every = 0;
        o->out = "models/chat.bin";
    }
    o->preset = arg_str(a, "--preset", o->preset);
    o->data = arg_str(a, "--data", finetune ? "data/sft" : "data/train.bin");
    o->val = arg_str(a, "--val", NULL);
    o->tokenizer = arg_str(a, "--tokenizer", finetune ? NULL : "data/tokenizer.bin");
    o->out = arg_str(a, "--out", o->out);
    o->sample_prompt = arg_str(a, "--sample-prompt", NULL);
    o->resume = arg_flag(a, "--resume");
    o->batch = arg_int(a, "--batch", o->batch);
    o->seq = arg_int(a, "--seq", o->seq);
    o->accum = arg_int(a, "--accum", o->accum);
    o->steps = arg_int(a, "--steps", o->steps);
    o->warmup = arg_int(a, "--warmup", o->warmup);
    o->lr = arg_float(a, "--lr", o->lr);
    o->weight_decay = arg_float(a, "--wd", o->weight_decay);
    o->eval_every = arg_int(a, "--eval-every", o->eval_every);
    o->eval_iters = arg_int(a, "--eval-iters", o->eval_iters);
    o->save_every = arg_int(a, "--save-every", o->save_every);
    o->sample_every = arg_int(a, "--sample-every", o->sample_every);
    o->log_every = arg_int(a, "--log-every", o->log_every);
    const char *seed = arg_str(a, "--seed", NULL);
    if (seed) o->seed = strtoull(seed, NULL, 10);
    if (o->accum < 1) o->accum = 1;
    if (o->save_every < 1) o->save_every = o->steps;
    if (o->eval_every < 1) o->eval_every = o->steps;
    if (o->log_every < 1) o->log_every = 1;
}

/* Обучение: замеряем умножение матриц на «только больших ядрах» и на «всех ядрах», берём быстрее. */
static double probe_gflops(int nt, float *A, float *B, float *C) {
    pool_init(nt);
    gemm_nt(C, A, B, 512, 512, 512, 0); /* прогрев */
    double best = 1e30;
    for (int r = 0; r < 3; r++) {
        double t0 = now_sec();
        gemm_nt(C, A, B, 512, 512, 512, 0);
        gemm_nt(C, A, B, 512, 512, 512, 1);
        double dt = now_sec() - t0;
        if (dt < best) best = dt;
    }
    return 2.0 * 2 * 512.0 * 512 * 512 / best / 1e9;
}

static int pick_train_threads(void) {
    int big = pool_default_threads();
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    int all = online > 16 ? 16 : (online < 1 ? 1 : (int)online);
    if (all <= big) return all;
    float *A = xmalloc(4 * 512 * 512), *B = xmalloc(4 * 512 * 512), *C = xmalloc(4 * 512 * 512);
    uint64_t r = 1;
    for (int i = 0; i < 512 * 512; i++) { A[i] = rng_f32(&r) - 0.5f; B[i] = rng_f32(&r) - 0.5f; }
    double g_big = probe_gflops(big, A, B, C), g_all = probe_gflops(all, A, B, C);
    free(A); free(B); free(C);
    int pick = g_all > g_big * 1.05 ? all : big;
    cprintf("  \033[2mподбор потоков: %d больших ядер — %.0f GFLOPS, все %d — %.0f GFLOPS → беру %d\033[0m\n",
            big, g_big, all, g_all, pick);
    return pick;
}

static void usage(void) {
    printf("%s◆ Corax LM%s — своя нейросеть с нуля\n\n", C_ACCENT, C_RESET);
    printf("  corax chat     -m МОДЕЛЬ [-t потоки] [--temp 0.8] [--system \"...\"]   чат\n");
    printf("  corax gen      -m МОДЕЛЬ -p \"начало текста\" [-n 200]                 продолжение текста\n");
    printf("  corax bench    -m МОДЕЛЬ [--sweep]                                   замер скорости\n");
    printf("  corax bench    --preset m70 [--quant q8|q4]                          скорость модели ДО обучения\n");
    printf("  corax info     -m МОДЕЛЬ                                             сведения о модели\n");
    printf("  corax train    --preset small --data data/train.bin --val data/val.bin --out models/base.bin\n");
    printf("  corax finetune --model models/base.bin --data data/sft --out models/chat.bin\n");
    printf("  corax quantize models/chat.bin models/chat-q8.bin q8|q4\n");
    printf("  corax presets                                                        размеры моделей\n\n");
    printf("  Подробности — в README.md\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);
    if (getenv("NO_COLOR") || !isatty(STDOUT_FILENO)) g_color = 0;
    Args a = {argc, argv};
    if (arg_flag(&a, "--no-color")) g_color = 0;
    if (argc < 2 || !strcmp(argv[1], "help") || !strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) { usage(); return 0; }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "chat")) return cmd_chat(&a);
    if (!strcmp(cmd, "gen")) return cmd_gen(&a);
    if (!strcmp(cmd, "bench")) return cmd_bench(&a);
    if (!strcmp(cmd, "info")) return cmd_info(&a);
    if (!strcmp(cmd, "presets")) { preset_list(stdout); return 0; }
    if (!strcmp(cmd, "quantize")) {
        if (argc < 5) die("использование: corax quantize ВХОД.bin ВЫХОД.bin q8|q4");
        return quantize_run(argv[2], argv[3], argv[4]);
    }
    if (!strcmp(cmd, "train") || !strcmp(cmd, "finetune")) {
        TrainOpts o;
        fill_train_opts(&a, &o, !strcmp(cmd, "finetune"));
        int t = arg_int(&a, "-t", arg_int(&a, "--threads", 0));
        pool_init(t > 0 ? t : pick_train_threads());
        return train_run(&o);
    }
    if (!strcmp(cmd, "_fwdtest")) { /* логиты на каждой позиции — для тестов */
        if (argc < 5) die("_fwdtest model tokens.i32 out.f32");
        pool_init(arg_int(&a, "-t", 3));
        Model m;
        model_load(&m, argv[2]);
        FILE *f = fopen(argv[3], "rb");
        if (!f) die("%s", argv[3]);
        int ids[4096], n = (int)fread(ids, 4, 4096, f);
        fclose(f);
        State s;
        state_init(&s, &m.c, 0);
        FILE *o = fopen(argv[4], "wb");
        for (int i = 0; i < n; i++) fwrite(forward(&m, &s, ids[i], i), 4, m.c.vocab_size, o);
        fclose(o);
        return 0;
    }
    if (!strcmp(cmd, "_tokenize")) { /* stdin -> id через пробел */
        if (argc < 3) die("_tokenize model");
        Model m;
        model_load(&m, argv[2]);
        static char buf[1 << 22];
        size_t n = fread(buf, 1, sizeof buf - 1, stdin);
        buf[n] = 0;
        int *ids = xmalloc(sizeof(int) * (n + 1));
        int k = tok_encode(&m.tok, buf, ids, (int)n + 1);
        for (int i = 0; i < k; i++) printf("%d%c", ids[i], i + 1 < k ? ' ' : '\n');
        return 0;
    }
    if (!strcmp(cmd, "_gradtest")) {
        if (argc < 5) die("_gradtest model batch out");
        pool_init(arg_int(&a, "-t", 2));
        return train_gradtest(argv[2], argv[3], argv[4]);
    }
    usage();
    return 1;
}
