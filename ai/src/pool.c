#include "corax.h"
#include "pool.h"
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>

#if defined(__aarch64__) || defined(__arm__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#elif defined(__x86_64__) || defined(__i386__)
#define cpu_relax() __builtin_ia32_pause()
#else
#define cpu_relax() ((void)0)
#endif

#define MAX_CPUS 64
#define SPIN_LIMIT 200000

static struct {
    int n;                 /* потоков всего, включая главный */
    pthread_t th[MAX_CPUS];
    atomic_int gen, sleepers, done, next, quit;
    pool_fn fn;
    void *ctx;
    int n_items, chunk, n_chunks;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int order[MAX_CPUS];   /* ядра от самого быстрого к медленному */
    long freq[MAX_CPUS];
    int ncpu, have_freq;
    int start_gen;
    char info[256];
} P = { .n = 1, .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER };

static long read_long(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    if (fscanf(f, "%ld", &v) != 1) v = -1;
    fclose(f);
    return v;
}

static void detect_cpus(void) {
    static int done = 0;
    if (done) return;
    done = 1;
    long n = sysconf(_SC_NPROCESSORS_CONF);
    if (n < 1) n = 1;
    if (n > MAX_CPUS) n = MAX_CPUS;
    P.ncpu = (int)n;
    P.have_freq = 1;
    for (int i = 0; i < P.ncpu; i++) {
        char path[128];
        snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        P.freq[i] = read_long(path);
        if (P.freq[i] <= 0) P.have_freq = 0;
        P.order[i] = i;
    }
    if (P.have_freq) { /* сортировка по убыванию частоты (стабильная) */
        for (int i = 1; i < P.ncpu; i++)
            for (int j = i; j > 0 && P.freq[P.order[j]] > P.freq[P.order[j - 1]]; j--) {
                int t = P.order[j]; P.order[j] = P.order[j - 1]; P.order[j - 1] = t;
            }
    }
}

int pool_default_threads(void) {
    detect_cpus();
    long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online < 1) online = 1;
    if (!P.have_freq) return online > 8 ? 8 : (int)online;
    long top = P.freq[P.order[0]];
    int big = 0;
    for (int i = 0; i < P.ncpu; i++)
        if (P.freq[i] >= top * 3 / 4) big++;
    if (big < 1) big = 1;
    return big;
}

static const char *cores_word(int n) {
    int a = n % 100, b = n % 10;
    if (a >= 11 && a <= 14) return "ядер";
    if (b == 1) return "ядро";
    if (b >= 2 && b <= 4) return "ядра";
    return "ядер";
}

const char *pool_cpu_info(void) {
    detect_cpus();
    if (!P.have_freq) {
        snprintf(P.info, sizeof P.info, "%d %s", P.ncpu, cores_word(P.ncpu));
        return P.info;
    }
    /* сгруппировать по частоте: "1x2.63 + 3x2.40 + 4x1.80 ГГц" */
    int pos = snprintf(P.info, sizeof P.info, "%d %s: ", P.ncpu, cores_word(P.ncpu));
    int i = 0;
    while (i < P.ncpu && pos < (int)sizeof P.info - 24) {
        long f = P.freq[P.order[i]];
        int cnt = 0;
        while (i < P.ncpu && P.freq[P.order[i]] == f) { cnt++; i++; }
        pos += snprintf(P.info + pos, sizeof P.info - pos, "%s%d×%.2f", P.info[pos - 2] == ':' ? "" : " + ", cnt, f / 1e6);
    }
    snprintf(P.info + pos, sizeof P.info - pos, " ГГц");
    return P.info;
}

static __thread int g_tid = 0;
int pool_tid(void) { return g_tid; }

static void pin_self(int id) {
#ifdef __linux__
    if (!P.have_freq || P.n > P.ncpu) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(P.order[id % P.ncpu], &set);
    sched_setaffinity(0, sizeof set, &set); /* ошибки не критичны */
#else
    (void)id;
#endif
}

static void do_work(void) {
    for (;;) {
        int c = atomic_fetch_add(&P.next, 1);
        if (c >= P.n_chunks) break;
        int i0 = c * P.chunk, i1 = i0 + P.chunk;
        if (i1 > P.n_items) i1 = P.n_items;
        P.fn(P.ctx, i0, i1);
    }
}

static void *worker(void *arg) {
    int id = (int)(intptr_t)arg;
    g_tid = id;
    pin_self(id);
    int seen = P.start_gen;
    for (;;) {
        int g, spins = 0;
        while ((g = atomic_load(&P.gen)) == seen) {
            if (++spins < SPIN_LIMIT) { cpu_relax(); continue; }
            pthread_mutex_lock(&P.mu);
            atomic_fetch_add(&P.sleepers, 1);
            while ((g = atomic_load(&P.gen)) == seen) pthread_cond_wait(&P.cv, &P.mu);
            atomic_fetch_sub(&P.sleepers, 1);
            pthread_mutex_unlock(&P.mu);
            break;
        }
        seen = g;
        if (atomic_load(&P.quit)) return NULL;
        do_work();
        atomic_fetch_add(&P.done, 1);
    }
}

void pool_init(int nthreads) {
    detect_cpus();
    if (nthreads < 1) nthreads = pool_default_threads();
    if (nthreads > MAX_CPUS) nthreads = MAX_CPUS;
    pool_shutdown();
    P.n = nthreads;
    atomic_store(&P.quit, 0);
    P.start_gen = atomic_load(&P.gen);
    pin_self(0);
    for (int i = 1; i < P.n; i++) {
        pthread_attr_t a;
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, 4 << 20);
        if (pthread_create(&P.th[i], &a, worker, (void *)(intptr_t)i) != 0) die("pthread_create");
        pthread_attr_destroy(&a);
    }
}

int pool_size(void) { return P.n; }

static void wake_all(void) {
    if (atomic_load(&P.sleepers) > 0) {
        pthread_mutex_lock(&P.mu);
        pthread_cond_broadcast(&P.cv);
        pthread_mutex_unlock(&P.mu);
    }
}

void pool_run(pool_fn fn, void *ctx, int n, int chunk) {
    if (n <= 0) return;
    if (chunk < 1) chunk = 1;
    int nchunks = (n + chunk - 1) / chunk;
    if (P.n <= 1 || nchunks == 1) { fn(ctx, 0, n); return; }
    P.fn = fn; P.ctx = ctx; P.n_items = n; P.chunk = chunk; P.n_chunks = nchunks;
    atomic_store(&P.next, 0);
    atomic_store(&P.done, 0);
    atomic_fetch_add(&P.gen, 1);
    wake_all();
    do_work();
    while (atomic_load(&P.done) < P.n - 1) cpu_relax();
}

void pool_shutdown(void) {
    if (P.n <= 1) return;
    atomic_store(&P.quit, 1);
    atomic_fetch_add(&P.gen, 1);
    pthread_mutex_lock(&P.mu);
    pthread_cond_broadcast(&P.cv);
    pthread_mutex_unlock(&P.mu);
    for (int i = 1; i < P.n; i++) pthread_join(P.th[i], NULL);
    P.n = 1;
}
