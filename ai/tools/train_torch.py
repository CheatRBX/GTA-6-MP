#!/usr/bin/env python3
"""Обучение Corax LM на ПК / видеокарте / Google Colab (PyTorch).

Та же архитектура и тот же формат файла, что у `corax train` на C:
модель, обученную здесь, можно квантовать и запускать на телефоне,
и наоборот — дообучать здесь модель, начатую на телефоне.

  pip install torch numpy
  python tools/train_torch.py --preset medium --data data/train.bin --val data/val.bin \
      --tokenizer data/tokenizer.bin --out models/base.bin --steps 20000
  python tools/train_torch.py --init models/base.bin --data data/sft --out models/chat.bin --steps 1000 --lr 1e-4
"""
import argparse
import math
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MAGIC = 0x4D585243
HEADER_FMT = "<IIqqqq7iiiff"  # 84 байта + 172 резерва = 256
PRESETS = {  # D, L, H, KV, F, T — как в src/util.c
    "nano": (256, 6, 8, 4, 704, 256),
    "mini": (384, 8, 6, 2, 1024, 512),
    "small": (512, 10, 8, 4, 1376, 512),
    "medium": (768, 12, 12, 4, 2048, 1024),
    "large": (1024, 16, 16, 4, 2816, 1024),
}


class Cfg:
    def __init__(self, dim, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size, seq_len,
                 rope_theta=10000.0, norm_eps=1e-5):
        self.dim, self.n_layers, self.n_heads, self.n_kv_heads = dim, n_layers, n_heads, n_kv_heads
        self.hidden_dim, self.vocab_size, self.seq_len = hidden_dim, vocab_size, seq_len
        self.rope_theta, self.norm_eps = rope_theta, norm_eps

    @property
    def head_dim(self):
        return self.dim // self.n_heads


def rmsnorm(x, w, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def rope_tables(cfg, T, device):
    hd = cfg.head_dim
    i = torch.arange(hd // 2, dtype=torch.float64)
    freq = cfg.rope_theta ** (-2.0 * i / hd)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * freq[None, :]
    return torch.cos(ang).float().to(device), torch.sin(ang).float().to(device)


def apply_rope(x, cos, sin):  # x: [B,T,H,hd], пары (2i, 2i+1)
    B, T, H, hd = x.shape
    x2 = x.float().view(B, T, H, hd // 2, 2)
    a, b = x2[..., 0], x2[..., 1]
    c, s = cos[:T, None, :], sin[:T, None, :]
    out = torch.stack((a * c - b * s, a * s + b * c), dim=-1)
    return out.view(B, T, H, hd).type_as(x)


class Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D, KV = cfg.dim, cfg.n_kv_heads * cfg.head_dim
        self.attn_norm = nn.Parameter(torch.ones(D))
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, KV, bias=False)
        self.wv = nn.Linear(D, KV, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.ffn_norm = nn.Parameter(torch.ones(D))
        self.w1 = nn.Linear(D, cfg.hidden_dim, bias=False)
        self.w2 = nn.Linear(cfg.hidden_dim, D, bias=False)
        self.w3 = nn.Linear(D, cfg.hidden_dim, bias=False)


class Corax(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.Parameter(torch.ones(cfg.dim))
        std_res = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, 0.0, std_res if name.endswith(("wo.weight", "w2.weight")) else 0.02)
        self._rope = None

    def forward(self, idx, targets=None):
        cfg = self.cfg
        B, T = idx.shape
        if self._rope is None or self._rope[0].shape[0] < T or self._rope[0].device != idx.device:
            self._rope = rope_tables(cfg, max(T, cfg.seq_len), idx.device)
        cos, sin = self._rope
        H, KVH, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        x = self.emb(idx)
        for L in self.layers:
            h = rmsnorm(x, L.attn_norm, cfg.norm_eps)
            q = apply_rope(L.wq(h).view(B, T, H, hd), cos, sin)
            k = apply_rope(L.wk(h).view(B, T, KVH, hd), cos, sin)
            v = L.wv(h).view(B, T, KVH, hd)
            if KVH != H:
                k = k.repeat_interleave(H // KVH, dim=2)
                v = v.repeat_interleave(H // KVH, dim=2)
            o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
            x = x + L.wo(o.transpose(1, 2).reshape(B, T, cfg.dim))
            h = rmsnorm(x, L.ffn_norm, cfg.norm_eps)
            x = x + L.w2(F.silu(L.w1(h)) * L.w3(h))
        x = rmsnorm(x, self.final_norm, cfg.norm_eps)
        logits = x @ self.emb.weight.t()
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

    # порядок тензоров как в файле: тензор за тензором, все слои подряд
    def ordered_tensors(self):
        ls = self.layers
        out = [self.emb.weight]
        for get in (lambda l: l.attn_norm, lambda l: l.wq.weight, lambda l: l.wk.weight, lambda l: l.wv.weight,
                    lambda l: l.wo.weight, lambda l: l.ffn_norm, lambda l: l.w1.weight, lambda l: l.w2.weight,
                    lambda l: l.w3.weight):
            out += [get(l) for l in ls]
        out.append(self.final_norm)
        return out


def save_corax(path, model, tok_bytes, steps=0, tokens=0):
    cfg = model.cfg
    params = b"".join(t.detach().float().cpu().contiguous().numpy().tobytes() for t in model.ordered_tensors())
    tok_off = 256 + len(params) if tok_bytes else 0
    hdr = struct.pack(HEADER_FMT, MAGIC, 1, steps, tokens, tok_off, len(tok_bytes or b""),
                      cfg.dim, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.hidden_dim, cfg.vocab_size, cfg.seq_len,
                      0, 32, cfg.rope_theta, cfg.norm_eps)
    hdr += b"\0" * (256 - len(hdr))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr + params + (tok_bytes or b""))
    os.replace(tmp, path)


def load_corax(path):
    with open(path, "rb") as f:
        data = f.read()
    (magic, ver, steps, tokens, tok_off, tok_size, D, L, H, KV, Fd, V, T, qt, gs, theta, eps) = \
        struct.unpack_from(HEADER_FMT, data, 0)
    if magic != MAGIC:
        raise SystemExit(f"{path}: это не модель Corax")
    if qt != 0:
        raise SystemExit(f"{path}: квантованная модель — нужен f32-файл")
    cfg = Cfg(D, L, H, KV, Fd, V, T, theta, eps)
    model = Corax(cfg)
    off = 256
    with torch.no_grad():
        for t in model.ordered_tensors():
            n = t.numel()
            t.copy_(torch.from_numpy(np.frombuffer(data, dtype=np.float32, count=n, offset=off).copy()).view_as(t))
            off += 4 * n
    tok = data[tok_off:tok_off + tok_size] if tok_off else None
    return model, tok, steps, tokens


class Data:
    def __init__(self, path):
        pre = path[:-4] if path.endswith(".bin") else path
        self.tok = np.memmap(pre + ".bin", dtype=np.uint16, mode="r")
        self.mask = np.memmap(pre + ".mask", dtype=np.uint8, mode="r") if os.path.exists(pre + ".mask") else None
        self.idx = np.fromfile(pre + ".idx", dtype=np.uint32) if os.path.exists(pre + ".idx") else None

    def batch(self, B, T, rng):
        n = len(self.tok)
        if self.idx is not None and len(self.idx):
            starts = np.minimum(self.idx[rng.integers(0, len(self.idx), B)].astype(np.int64), n - T - 1)
        else:
            starts = rng.integers(0, n - T - 1, B)
        x = np.stack([self.tok[s:s + T] for s in starts]).astype(np.int64)
        y = np.stack([self.tok[s + 1:s + T + 1] for s in starts]).astype(np.int64)
        if self.mask is not None:
            m = np.stack([self.mask[s + 1:s + T + 1] for s in starts])
            y[m == 0] = -1
        return torch.from_numpy(x), torch.from_numpy(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small", choices=PRESETS)
    ap.add_argument("--init", help="начать с модели Corax (дообучение)")
    ap.add_argument("--data", default="data/train.bin")
    ap.add_argument("--val")
    ap.add_argument("--tokenizer", default="data/tokenizer.bin")
    ap.add_argument("--out", default="models/base.bin")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seq", type=int, default=0)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0)
    ap.add_argument("--warmup", type=int, default=-1)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--compile", action="store_true", help="torch.compile (быстрее на GPU)")
    ap.add_argument("--resume", action="store_true", help="продолжить с --out и <out>.pt (оптимизатор, шаг)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok_bytes = open(a.tokenizer, "rb").read() if a.tokenizer and os.path.exists(a.tokenizer) else None
    steps0 = tokens0 = 0
    state_path = a.out + ".pt"
    if a.resume:
        if not os.path.exists(a.out):
            raise SystemExit(f"нечего продолжать: нет {a.out}")
        a.init = a.out
    if a.init:
        model, tb, steps0, tokens0 = load_corax(a.init)
        tok_bytes = tok_bytes if (tok_bytes and not tb) else (tb or tok_bytes)
    else:
        if not tok_bytes:
            raise SystemExit("нужен токенизатор: --tokenizer data/tokenizer.bin")
        vocab = struct.unpack_from("<i", tok_bytes, 8)[0]
        D, L, H, KV, Fd, T = PRESETS[a.preset]
        model = Corax(Cfg(D, L, H, KV, Fd, vocab, a.seq or T))
    cfg = model.cfg
    T = min(a.seq or cfg.seq_len, cfg.seq_len)
    lr = a.lr or (2e-4 if (a.init and not a.resume) else (1e-3 if cfg.dim <= 384 else 6e-4))
    warmup = a.warmup if a.warmup >= 0 else min(a.steps // 20, 500)
    model.to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"Corax LM: {nparams / 1e6:.1f} млн параметров, устройство {dev}, батч {a.batch}x{T}x{a.accum}, lr {lr}")
    decay = [p for p in model.parameters() if p.dim() == 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=(dev == "cuda"))
    start = 1
    if a.resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=dev)
        opt.load_state_dict(st["opt"])
        start = st["step"] + 1
        steps0 = st["steps0"]
        print(f"продолжаю с шага {start}")
    fwd = torch.compile(model) if a.compile else model
    if dev == "cuda":  # bf16 на новых GPU; fp16 + GradScaler на T4 и старше
        half = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        amp = torch.autocast("cuda", dtype=half)
    else:
        half = None
        amp = torch.autocast("cpu", enabled=False)
    scaler = torch.amp.GradScaler("cuda", enabled=(half == torch.float16))
    train, val = Data(a.data), (Data(a.val) if a.val else None)
    rng = np.random.default_rng(1337)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    t0 = time.time()
    tokens = tokens0
    for step in range(start, a.steps + 1):
        cur = lr * step / max(warmup, 1) if step <= warmup else \
            lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(a.steps - warmup, 1))))
        for g in opt.param_groups:
            g["lr"] = cur
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for _ in range(a.accum):
            x, y = train.batch(a.batch, T, rng)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with amp:
                loss = fwd(x, y) / a.accum
            scaler.scale(loss).backward()
            tot += loss.item()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        tokens += a.batch * T * a.accum
        if step % 10 == 0 or step == 1:
            el = time.time() - t0
            print(f"шаг {step}/{a.steps} loss {tot:.4f} lr {cur:.2e} |g| {gn:.2f} "
                  f"{((step - start + 1) * a.batch * T * a.accum) / el:.0f} ток/с, "
                  f"осталось {el / (step - start + 1) * (a.steps - step) / 60:.1f} мин",
                  flush=True)
        if val and (step % a.eval_every == 0 or step == a.steps):
            model.eval()
            vrng = np.random.default_rng(7)
            with torch.no_grad(), amp:
                vl = sum(fwd(*[t.to(dev) for t in val.batch(a.batch, T, vrng)]).item() for _ in range(20)) / 20
            model.train()
            print(f"  валидация: loss {vl:.4f}, перплексия {math.exp(vl):.1f}", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_corax(a.out, model, tok_bytes, steps0 + step, tokens)
            torch.save({"opt": opt.state_dict(), "step": step, "steps0": steps0}, state_path)
            print(f"  сохранено {a.out}", flush=True)
    print(f"готово. Дальше на телефоне: ./corax quantize {a.out} models/model-q8.bin q8")


if __name__ == "__main__":
    sys.exit(main())
