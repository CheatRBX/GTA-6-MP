# Corax LM — сборка. В Termux: pkg install clang make && make
CC ?= cc
UNAME_M := $(shell uname -m)
SRC := src/util.c src/pool.c src/tokenizer.c src/infer.c src/gemm.c src/train.c src/quant.c src/main.c
HDR := $(wildcard src/*.h)

CFLAGS ?= -O3 -ffast-math -fno-finite-math-only -funroll-loops
CFLAGS += -std=c11 -Wall -Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-missing-field-initializers
LDLIBS := -lm -lpthread

# Архитектура: на aarch64 (Snapdragon) включаем dotprod — int8-умножение в NEON.
ifeq ($(UNAME_M),aarch64)
  ifeq ($(NODOTPROD),1)
    ARCHFLAGS ?= -march=armv8-a
  else
    ARCHFLAGS ?= -march=armv8.2-a+dotprod+fp16
  endif
else ifeq ($(UNAME_M),x86_64)
  ARCHFLAGS ?= -march=native
endif
CFLAGS += $(ARCHFLAGS)

corax: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

# Для старых ARM без dotprod (если corax падает с "Illegal instruction")
generic:
	$(MAKE) clean
	$(MAKE) NODOTPROD=1

debug:
	$(CC) -O1 -g -fsanitize=address,undefined -std=c11 -Wall -o corax-debug $(SRC) $(LDLIBS)

clean:
	rm -f corax corax-debug

.PHONY: generic debug clean
