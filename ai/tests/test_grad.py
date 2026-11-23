#!/usr/bin/env python3
"""Сверка C-обучения с PyTorch: loss и все градиенты должны совпасть.
Запуск: python tests/test_grad.py  (нужны torch и numpy, собранный ./corax)"""
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
from train_torch import Cfg, Corax, save_corax  # noqa: E402


def run_case(cfg, B, T, seed):
    torch.manual_seed(seed)
    model = Corax(cfg)
    with torch.no_grad():  # норм-веса не единицы, чтобы проверить и их градиенты
        for L in model.layers:
            L.attn_norm.uniform_(0.5, 1.5)
            L.ffn_norm.uniform_(0.5, 1.5)
        model.final_norm.uniform_(0.5, 1.5)
        for p in model.parameters():
            if p.dim() == 2:
                p.mul_(5)  # покрупнее веса — заметнее ошибки
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    y = torch.randint(0, cfg.vocab_size, (B, T), generator=g)
    y[0, :3] = -1  # маска
    with tempfile.TemporaryDirectory() as d:
        mp, bp, gp = (os.path.join(d, n) for n in ("m.bin", "b.bin", "g.bin"))
        save_corax(mp, model, None)
        with open(bp, "wb") as f:
            f.write(struct.pack("<ii", B, T))
            f.write(x.numpy().astype(np.int32).tobytes())
            f.write(y.numpy().astype(np.int32).tobytes())
        subprocess.run([*os.environ.get("CORAX_CMD", os.path.join(ROOT, "corax")).split(), "_gradtest", mp, bp, gp, "-t", "3"], check=True,
                       stdout=subprocess.DEVNULL)
        out = np.fromfile(gp, dtype=np.float32)
    loss = model(x, y)
    loss.backward()
    c_loss, c_grads = out[0], out[1:]
    ref = torch.cat([t.grad.flatten() for t in model.ordered_tensors()]).numpy()
    assert c_grads.shape == ref.shape, (c_grads.shape, ref.shape)
    names = ["emb", "attn_norm", "wq", "wk", "wv", "wo", "ffn_norm", "w1", "w2", "w3", "final_norm"]
    off = 0
    worst = 0.0
    tensors = model.ordered_tensors()
    sizes = [tensors[0].numel()] + [sum(t.numel() for t in tensors[1 + i * cfg.n_layers:1 + (i + 1) * cfg.n_layers])
                                    for i in range(9)] + [tensors[-1].numel()]
    for name, n in zip(names, sizes):
        a, b = c_grads[off:off + n], ref[off:off + n]
        rel = np.abs(a - b).max() / (np.abs(b).max() + 1e-12)
        worst = max(worst, rel)
        print(f"  {name:11s} max|Δ|/max|g| = {rel:.2e}")
        off += n
    dl = abs(c_loss - loss.item())
    print(f"  loss C={c_loss:.6f} torch={loss.item():.6f} Δ={dl:.2e}")
    assert dl < 1e-4 and worst < 2e-3, "градиенты не совпали"


def main():
    print("случай 1: GQA (8 голов / 2 kv), чётные размеры")
    run_case(Cfg(64, 2, 8, 2, 96, 97, 32), B=3, T=19, seed=1)
    print("случай 2: без GQA, хвосты матриц")
    run_case(Cfg(96, 3, 3, 3, 160, 131, 16), B=2, T=13, seed=2)
    print("OK: C-обучение совпадает с PyTorch")


if __name__ == "__main__":
    main()
