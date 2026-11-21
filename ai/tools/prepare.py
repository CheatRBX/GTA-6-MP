#!/usr/bin/env python3
"""Подготовка данных для Corax LM. Только стандартная библиотека Python.

  pretrain: тексты (.txt / .jsonl) -> data/train.bin, data/val.bin (uint16 токены)
  sft:      диалоги (.jsonl)        -> data/sft.bin + sft.mask + sft.idx (для дообучения)

Если токенизатора ещё нет — pretrain обучит его сам.
"""
import argparse
import json
import os
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import Tokenizer, count_words, iter_text_files, read_texts, train_bpe  # noqa: E402

DOC_SEP = "<|endoftext|>"


def iter_docs(paths, doc_sep):
    """Документы: элементы .jsonl, куски .txt между разделителями, иначе файл целиком (потоком)."""
    for fp in iter_text_files(paths):
        if fp.endswith(".jsonl"):
            for t in read_texts(fp):
                yield t, True
            continue
        buf = ""
        for chunk in read_texts(fp):
            buf += chunk
            if doc_sep and doc_sep in buf:
                parts = buf.split(doc_sep)
                buf = parts.pop()
                for p in parts:
                    if p.strip():
                        yield p.strip(), True
            elif len(buf) > (1 << 22):
                # длинный текст без разделителей: режем по последнему переводу строки
                cut = buf.rfind("\n", 0, len(buf) - 1)
                if cut <= 0:
                    cut = len(buf)
                yield buf[:cut + 1], False
                buf = buf[cut + 1:]
        if buf.strip():
            yield buf, True


def cyr_ratio(text):
    letters = cyr = 0
    for ch in text[:4000]:
        if ch.isalpha():
            letters += 1
            if "\u0400" <= ch <= "\u04ff":
                cyr += 1
    return cyr / letters if letters else 0.0


def cmd_pretrain(a):
    os.makedirs(a.out_dir, exist_ok=True)
    tok_path = a.tokenizer or os.path.join(a.out_dir, "tokenizer.bin")
    if not os.path.exists(tok_path):
        print(f"токенизатор не найден — обучаю новый ({a.vocab} токенов)...")
        if a.min_cyr > 0:  # словарь — только по документам, прошедшим фильтр языка
            from collections import Counter
            from tokenizer import PAT
            counts, used, budget = Counter(), 0, int(a.tok_sample_mb * 1024 * 1024)
            for text, _ in iter_docs(a.inputs, a.doc_sep):
                if cyr_ratio(text) >= a.min_cyr:
                    counts.update(PAT.findall(text))
                    used += len(text)
                    if used >= budget:
                        break
        else:
            counts, used = count_words(a.inputs, a.tok_sample_mb)
        print(f"  текст для словаря: {used / 1e6:.1f} млн символов")
        tok = train_bpe(counts, a.vocab)
        tok.save(tok_path)
        print(f"  сохранён {tok_path}")
    tok = Tokenizer.load(tok_path)
    print(f"токенизатор: {tok_path}, словарь {tok.vocab_size}")

    train = array("H")
    val = array("H")
    every = max(int(round(1 / a.val_frac)), 2) if a.val_frac > 0 else 0
    n_docs = 0
    n_chars = 0
    t0 = time.time()
    last = t0
    new_doc = True
    skipped = 0
    for text, ends in iter_docs(a.inputs, a.doc_sep):
        if a.min_cyr > 0 and cyr_ratio(text) < a.min_cyr:
            skipped += 1
            continue
        ids = tok.encode(text)
        if new_doc:
            ids = [tok.bos] + ids
        if ends:
            ids.append(tok.eos)
        target = val if (every and n_docs % every == every - 1) else train
        target.extend(ids)
        if ends:
            n_docs += 1
        new_doc = ends
        n_chars += len(text)
        if time.time() - last > 5:
            last = time.time()
            print(f"  {n_chars / 1e6:.1f} млн символов, {len(train) / 1e6:.2f} млн токенов, "
                  f"{n_chars / 1e6 / (last - t0):.2f} млн симв/с", flush=True)
        if a.max_tokens and len(train) >= a.max_tokens:
            break
    if len(val) < 4096 and len(train) > 8192:
        # мало документов: валидация = хвост потока
        k = max(int(len(train) * max(a.val_frac, 0.01)), 4096)
        val = array("H", train[-k:]) + val
        del train[-k:]
    with open(os.path.join(a.out_dir, "train.bin"), "wb") as f:
        train.tofile(f)
    with open(os.path.join(a.out_dir, "val.bin"), "wb") as f:
        val.tofile(f)
    if skipped:
        print(f"  пропущено документов не на русском: {skipped}")
    ratio = n_chars / max(len(train) + len(val), 1)
    print(f"готово за {time.time() - t0:.0f} с: train={len(train) / 1e6:.2f} млн токенов, "
          f"val={len(val) / 1e6:.3f} млн, документов {n_docs}, ~{ratio:.2f} символа/токен")


# ---------------- SFT (дообучение на диалогах) ----------------
def to_messages(obj):
    """Поддерживаемые форматы строки JSONL -> список (role, content)."""
    if "messages" in obj or "conversations" in obj:
        msgs = []
        for m in obj.get("messages") or obj.get("conversations"):
            role = m.get("role") or m.get("from") or ""
            text = m.get("content") if "content" in m else m.get("value", "")
            role = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant"}.get(role, role)
            msgs.append((role, text or ""))
        return msgs
    msgs = []
    if obj.get("system"):
        msgs.append(("system", obj["system"]))
    if "instruction" in obj:
        q = obj["instruction"] + (("\n\n" + obj["input"]) if obj.get("input") else "")
        return msgs + [("user", q), ("assistant", obj.get("output", ""))]
    for qk, ak in (("prompt", "response"), ("question", "answer"), ("user", "assistant"), ("q", "a")):
        if qk in obj and ak in obj:
            return msgs + [("user", obj[qk]), ("assistant", obj[ak])]
    return None


def encode_dialog(tok, msgs, default_system):
    """Шаблон: <bos> [<system> текст <eos>] (<user> текст <eos> <assistant> ответ <eos>)+
    Маска = 1 только на токенах ответа ассистента и его <eos> (их модель учится предсказывать)."""
    ids, mask = [tok.bos], [0]
    if default_system and not any(r == "system" for r, _ in msgs):
        msgs = [("system", default_system)] + msgs
    for role, text in msgs:
        t = tok.encode(text.strip())
        if role == "system":
            ids += [tok.system] + t + [tok.eos]
            mask += [0] * (len(t) + 2)
        elif role == "user":
            ids += [tok.user] + t + [tok.eos, tok.assistant]
            mask += [0] * (len(t) + 3)
        elif role == "assistant":
            ids += t + [tok.eos]
            mask += [1] * (len(t) + 1)
    return ids, mask


def cmd_sft(a):
    tok = Tokenizer.load(a.tokenizer)
    toks, masks, idx = array("H"), array("B"), array("I")
    n_ok = n_bad = 0
    total_target = 0
    for fp in a.inputs:
        with open(fp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    msgs = to_messages(json.loads(line))
                except (json.JSONDecodeError, AttributeError, TypeError):
                    msgs = None
                if not msgs or not any(r == "assistant" for r, _ in msgs):
                    n_bad += 1
                    print(f"  пропуск {fp}:{ln} — непонятный формат", file=sys.stderr)
                    continue
                ids, mask = encode_dialog(tok, msgs, a.system)
                if len(ids) > a.max_len:
                    ids, mask = ids[:a.max_len], mask[:a.max_len]
                for _ in range(a.repeat):
                    idx.append(len(toks))
                    toks.extend(ids)
                    masks.extend(mask)
                total_target += sum(mask)
                n_ok += 1
    if n_ok == 0:
        raise SystemExit("нет ни одного диалога")
    pre = a.out
    os.makedirs(os.path.dirname(os.path.abspath(pre)), exist_ok=True)
    with open(pre + ".bin", "wb") as f:
        toks.tofile(f)
    with open(pre + ".mask", "wb") as f:
        masks.tofile(f)
    with open(pre + ".idx", "wb") as f:
        idx.tofile(f)
    print(f"готово: {n_ok} диалогов (x{a.repeat}), пропущено {n_bad}, токенов {len(toks)}, "
          f"из них обучаемых {total_target * a.repeat}")
    print(f"файлы: {pre}.bin / .mask / .idx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pretrain", help="тексты -> train.bin/val.bin")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--tokenizer", default=None, help="путь к tokenizer.bin (по умолчанию <out-dir>/tokenizer.bin)")
    p.add_argument("--vocab", type=int, default=8192, help="размер словаря, если токенизатор обучается")
    p.add_argument("--tok-sample-mb", type=float, default=20)
    p.add_argument("--val-frac", type=float, default=0.01)
    p.add_argument("--doc-sep", default=DOC_SEP, help="разделитель документов в .txt")
    p.add_argument("--max-tokens", type=int, default=0, help="ограничить объём (0 = всё)")
    p.add_argument("--min-cyr", type=float, default=0.0,
                   help="оставить только документы, где доля кириллицы среди букв >= X (0.5 = только русские)")
    p.set_defaults(fn=cmd_pretrain)

    s = sub.add_parser("sft", help="диалоги .jsonl -> sft.bin/.mask/.idx")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--tokenizer", default="data/tokenizer.bin")
    s.add_argument("--out", default="data/sft")
    s.add_argument("--system", default="", help="системная подсказка по умолчанию")
    s.add_argument("--max-len", type=int, default=1024)
    s.add_argument("--repeat", type=int, default=1, help="повторить каждый диалог N раз")
    s.set_defaults(fn=cmd_sft)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
