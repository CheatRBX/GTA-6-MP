#pragma once
/* Пул потоков: параллельный for с динамической раздачей кусков.
 * Динамическая раздача важна для big.LITTLE (Snapdragon): быстрые ядра
 * сами забирают больше работы, медленные не тормозят остальных. */

typedef void (*pool_fn)(void *ctx, int i0, int i1);

int  pool_default_threads(void);       /* число «больших» ядер */
void pool_init(int nthreads);           /* nthreads включая главный поток */
int  pool_size(void);
int  pool_tid(void);                  /* номер текущего потока: 0..pool_size()-1 */
void pool_run(pool_fn fn, void *ctx, int n, int chunk); /* fn(ctx,i0,i1) по кускам [0,n) */
void pool_shutdown(void);
const char *pool_cpu_info(void);
