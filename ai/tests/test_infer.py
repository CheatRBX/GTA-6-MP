#!/usr/bin/env python3
"""Сверка инференса C с PyTorch (f32 / q8 / q4) и токенизатора C с Python.
Запуск: python tests/test_infer.py data/tokenizer.bin"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from tokenizer import Tokenizer  # noqa: E402
from train_torch import Cfg, Corax, save_corax  # noqa: E402

CORAX = os.environ.get("CORAX_CMD", os.path.join(ROOT, "corax")).split()
TEXTS = [
    "Привет, мир! Как дела?",
    "  два пробела,\n\nновая строка\tи таб   в конце   ",
    "English words, numbers 12345 и 3.14159; «кавычки» — тире… эмодзи 🙂👍🏽",
    "def f(x):\n    return x**2  # код\n",
    "Ёлка, ёжик, ЁЖ. Ü ß ç — ünïcödé. 中文 日本語 한국어",
    "\n\n\n   \n",
    " a  b   c    d",
    "Однажды в студёную зимнюю пору я из лесу вышел; был сильный мороз.",
]


def check_tokenizer(model_path, tok):
    for t in TEXTS:
        py = tok.encode(t)
        out = subprocess.run([*CORAX, "_tokenize", model_path], input=t.encode(), capture_output=True, check=True)
        c = [int(v) for v in out.stdout.split()]
        assert c == py, f"токенизатор расходится на {t!r}:\n C ={c}\n py={py}"
        assert tok.decode(py) == t, f"decode не обратим: {t!r}"
    print(f"  токенизатор: C == Python на {len(TEXTS)} текстах, decode обратим")


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "tokenizer.bin")
    tok = Tokenizer.load(tok_path)
    tok_bytes = open(tok_path, "rb").read()
    torch.manual_seed(0)
    cfg = Cfg(128, 3, 4, 2, 352, tok.vocab_size, 64)
    model = Corax(cfg).eval()
    ids = [tok.bos] + tok.encode("Однажды маленький кот нашёл в лесу красный мяч и")[:40]
    with tempfile.TemporaryDirectory() as d:
        f32 = os.path.join(d, "m.bin")
        save_corax(f32, model, tok_bytes)
        check_tokenizer(f32, tok)
        ip = os.path.join(d, "ids.bin")
        np.array(ids, dtype=np.int32).tofile(ip)
        with torch.no_grad():
            ref = model(torch.tensor([ids]))[0].numpy()
        for qt, tol in (("f32", 2e-3), ("q8", 0.03), ("q4", 0.15)):  # на ARM KV-кеш в fp16
            path = f32
            if qt != "f32":
                path = os.path.join(d, f"m-{qt}.bin")
                subprocess.run([*CORAX, "quantize", f32, path, qt], check=True, stdout=subprocess.DEVNULL)
            op = os.path.join(d, "out.bin")
            subprocess.run([*CORAX, "_fwdtest", path, ip, op], check=True)
            got = np.fromfile(op, dtype=np.float32).reshape(len(ids), -1)
            rel = np.abs(got - ref).max() / np.abs(ref).max()
            agree = (got.argmax(-1) == ref.argmax(-1)).mean()
            print(f"  {qt}: max относительная ошибка логитов {rel:.2e}, совпадение top-1 {agree * 100:.0f}%")
            assert rel < tol, f"{qt}: слишком большая ошибка"
    print("OK: инференс C совпадает с PyTorch")


if __name__ == "__main__":
    main()
