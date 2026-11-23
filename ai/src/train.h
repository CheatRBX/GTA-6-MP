#pragma once
#include "corax.h"

typedef struct {
    const char *preset, *data, *val, *tokenizer, *out, *init, *sample_prompt;
    int resume;
    int batch, seq, accum, steps, warmup;
    float lr, min_lr_frac, weight_decay, beta1, beta2, clip;
    int eval_every, eval_iters, save_every, sample_every, log_every;
    uint64_t seed;
} TrainOpts;

void train_opts_default(TrainOpts *o);
int  train_run(TrainOpts *o);
int  train_gradtest(const char *model, const char *batch, const char *out);
void save_model_f32(const char *path, const Config *c, const float *params, const uint8_t *tok, size_t tok_size,
                    int64_t steps, int64_t tokens);
int  quantize_run(const char *in, const char *out, const char *type);
