#!/usr/bin/env python3
"""Corax BPE-токенизатор: байтовый BPE, обучается с нуля. Только стандартная библиотека.

Формат tokenizer.bin (little-endian):
  u32 magic 'CRXT', u32 version=1, i32 vocab_size, i32 n_merges, i32 n_special,
  i32[n_merges][2] пары слияний (токен 256+i = a+b),
  для каждого спецтокена: i32 len + байты.

Пред-токенизация (PAT) один-в-один повторена в src/tokenizer.c.
"""
import argparse
import heapq
import os
import re
import struct
import sys
import time
from collections import Counter, defaultdict

MAGIC = 0x54585243
SPECIALS = ["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|system|>"]

_S = " \\t\\n\\r\\x0b\\x0c"
_W = ("A-Za-z0-9_\\u00C0-\\u00D6\\u00D8-\\u00F6\\u00F8-\\u1FFF"
      "\\u2070-\\u209F\\u20D0-\\u2FFF\\u3040-\\U0010FFFF")
PAT = re.compile(f" ?[{_W}]+| ?[^{_S}{_W}]+|[{_S}]+(?![^{_S}])|[{_S}]+")


def pretokenize(text):
    return PAT.findall(text)


class Tokenizer:
    def __init__(self, merges, specials=SPECIALS):
        self.merges = [tuple(m) for m in merges]
        self.specials = list(specials)
        self.ranks = {p: i for i, p in enumerate(self.merges)}
        self.vocab = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self.vocab.append(self.vocab[a] + self.vocab[b])
        base = 256 + len(self.merges)
        self.special_ids = {s: base + i for i, s in enumerate(self.specials)}
        for s in self.specials:
            self.vocab.append(s.encode())
        self.bos = self.special_ids["<|bos|>"]
        self.eos = self.special_ids["<|eos|>"]
        self.user = self.special_ids["<|user|>"]
        self.assistant = self.special_ids["<|assistant|>"]
        self.system = self.special_ids["<|system|>"]
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---------- сохранение / загрузка ----------
    def save(self, path):
        with open(path, "wb") as f:
            f.write(struct.pack("<IIiii", MAGIC, 1, self.vocab_size, len(self.merges), len(self.specials)))
            for a, b in self.merges:
                f.write(struct.pack("<ii", a, b))
            for s in self.specials:
                bs = s.encode()
                f.write(struct.pack("<i", len(bs)) + bs)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            data = f.read()
        return cls.from_bytes(data)

    @classmethod
    def from_bytes(cls, data):
        magic, ver, vs, nm, ns = struct.unpack_from("<IIiii", data, 0)
        if magic != MAGIC or ver != 1:
            raise ValueError("это не токенизатор Corax")
        off = 20
        merges = [struct.unpack_from("<ii", data, off + 8 * i) for i in range(nm)]
        off += 8 * nm
        specials = []
        for _ in range(ns):
            (ln,) = struct.unpack_from("<i", data, off)
            off += 4
            specials.append(data[off:off + ln].decode())
            off += ln
        return cls(merges, specials)

    # ---------- кодирование ----------
    def _bpe(self, wb):
        ids = list(wb)
        ranks = self.ranks
        while len(ids) >= 2:
            best = None
            for i in range(len(ids) - 1):
                r = ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best is None or r < best):
                    best = r
            if best is None:
                break
            a, b = self.merges[best]
            new_id = 256 + best
            out, i, n = [], 0, len(ids)
            while i < n:
                if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
        return ids

    def encode(self, text):
        out = []
        cache = self._cache
        for w in PAT.findall(text):
            ids = cache.get(w)
            if ids is None:
                ids = self._bpe(w.encode("utf-8"))
                if len(cache) < 2_000_000:
                    cache[w] = ids
            out.extend(ids)
        return out

    def decode(self, ids, skip_special=True):
        base = 256 + len(self.merges)
        bs = b"".join(self.vocab[i] for i in ids if not (skip_special and i >= base))
        return bs.decode("utf-8", errors="replace")


def train_bpe(word_counts, vocab_size, min_freq=2, verbose=True):
    """Обучение BPE по словарю частот кусков текста. Инкрементальный пересчёт пар + ленивая куча."""
    n_merges = vocab_size - 256 - len(SPECIALS)
    if n_merges < 0:
        raise ValueError("vocab_size слишком мал (минимум %d)" % (256 + len(SPECIALS)))
    words, freqs = [], []
    for w, c in word_counts.items():
        b = w.encode("utf-8")
        if len(b) >= 2:
            words.append(list(b))
            freqs.append(c)
    pair_counts = defaultdict(int)
    where = defaultdict(set)
    for wi, w in enumerate(words):
        f = freqs[wi]
        for p in zip(w, w[1:]):
            pair_counts[p] += f
            where[p].add(wi)
    heap = [(-c, p) for p, c in pair_counts.items()]
    heapq.heapify(heap)
    merges = []
    t0 = time.time()
    while len(merges) < n_merges and heap:
        negc, p = heapq.heappop(heap)
        c = pair_counts.get(p, 0)
        if c <= 0:
            continue
        if -negc != c:  # устаревшая запись
            heapq.heappush(heap, (-c, p))
            continue
        if c < min_freq:
            break
        new_id = 256 + len(merges)
        merges.append(p)
        a, b = p
        changed = set()
        for wi in where.pop(p, ()):
            w = words[wi]
            n = len(w)
            out, i = [], 0
            while i < n:
                if i < n - 1 and w[i] == a and w[i + 1] == b:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            if len(out) == n:
                continue
            f = freqs[wi]
            for q in zip(w, w[1:]):
                pair_counts[q] -= f
                changed.add(q)
            for q in zip(out, out[1:]):
                pair_counts[q] += f
                where[q].add(wi)
                changed.add(q)
            words[wi] = out
        pair_counts.pop(p, None)
        for q in changed:
            cq = pair_counts.get(q, 0)
            if q != p and cq > 0:
                heapq.heappush(heap, (-cq, q))
            elif cq <= 0:
                pair_counts.pop(q, None)
        if verbose and (len(merges) % 500 == 0 or len(merges) == n_merges):
            print(f"  слияний {len(merges)}/{n_merges}  ({time.time() - t0:.0f} с)", flush=True)
    if len(merges) < n_merges and verbose:
        print(f"  данных мало: получилось только {len(merges)} слияний", flush=True)
    return Tokenizer(merges)


def iter_text_files(paths):
    for p in paths:
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for fn in sorted(files):
                    if fn.endswith((".txt", ".jsonl", ".md")):
                        yield os.path.join(root, fn)
        else:
            yield p


def read_texts(path, max_bytes=None):
    """Отдаёт куски текста из .txt (кусками) или .jsonl (поле text/content)."""
    import json
    got = 0
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text") or obj.get("content") or ""
                if not isinstance(t, str) or not t:
                    continue
                yield t
                got += len(t)
                if max_bytes and got >= max_bytes:
                    return
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                chunk = f.read(1 << 22)
                if not chunk:
                    return
                yield chunk
                got += len(chunk)
                if max_bytes and got >= max_bytes:
                    return


def count_words(paths, sample_mb):
    budget = int(sample_mb * 1024 * 1024)
    counts = Counter()
    used = 0
    files = list(iter_text_files(paths))
    if not files:
        raise SystemExit("не найдено текстовых файлов")
    per_file = max(budget // len(files), 1 << 20)
    for fp in files:
        for chunk in read_texts(fp, per_file):
            counts.update(PAT.findall(chunk))
            used += len(chunk)
        if used >= budget:
            break
    return counts, used


def main():
    ap = argparse.ArgumentParser(description="Обучить BPE-токенизатор Corax с нуля")
    ap.add_argument("inputs", nargs="+", help=".txt/.jsonl файлы или папки")
    ap.add_argument("--vocab", type=int, default=8192, help="размер словаря (по умолчанию 8192)")
    ap.add_argument("--out", default="data/tokenizer.bin")
    ap.add_argument("--sample-mb", type=float, default=20, help="сколько МБ текста взять для обучения")
    ap.add_argument("--min-freq", type=int, default=2)
    a = ap.parse_args()
    t0 = time.time()
    counts, used = count_words(a.inputs, a.sample_mb)
    print(f"текст: {used / 1e6:.1f} млн символов, уникальных кусков: {len(counts)}")
    tok = train_bpe(counts, a.vocab, a.min_freq)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tok.save(a.out)
    print(f"готово: {a.out}  словарь={tok.vocab_size}  за {time.time() - t0:.0f} с")
    sample = "Привет! Это проверка токенизатора Corax."
    ids = tok.encode(sample)
    print(f"пример: {len(sample)} символов -> {len(ids)} токенов")


if __name__ == "__main__":
    sys.exit(main())
