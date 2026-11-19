#!/usr/bin/env python3
"""Проверка bootanimation.zip на то, что Android его примет.

Проверяет: архив без сжатия, корректный desc.txt, у каждой части есть кадры,
все кадры — PNG нужного размера, имена идут по порядку.
Код возврата 0 — всё в порядке, 1 — есть ошибки (они печатаются).

    python3 brand/bootanim/check.py brand/out/bootanimation.zip
"""
from __future__ import annotations

import io
import re
import sys
import zipfile
from pathlib import PurePosixPath

from PIL import Image

LINE = re.compile(r"^(?P<type>[pcf])\s+(?P<count>\d+)\s+(?P<pause>\d+)\s+(?P<path>\S+)"
                  r"(?:\s+(?P<fade>\d+))?(?:\s+(?P<color>#[0-9A-Fa-f]{6}))?\s*$")


def check(path: str) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED:
                errors.append(f"{info.filename}: сжат (метод {info.compress_type}), нужен ZIP_STORED")
        names = {i.filename for i in infos if not i.is_dir()}
        if "desc.txt" not in names:
            return errors + ["нет desc.txt в корне архива"]
        lines = [ln for ln in zf.read("desc.txt").decode().splitlines() if ln.strip()]
        header = lines[0].split()
        if len(header) < 3 or not all(x.isdigit() for x in header[:3]):
            return errors + [f"первая строка desc.txt должна быть «ШИРИНА ВЫСОТА FPS», а не {lines[0]!r}"]
        width, height, fps = map(int, header[:3])
        if not 1 <= fps <= 120:
            errors.append(f"странный FPS: {fps}")
        has_loop = False
        for ln in lines[1:]:
            m = LINE.match(ln)
            if not m:
                errors.append(f"строка desc.txt не разобрана: {ln!r}")
                continue
            if m["type"] in "pf" and m["count"] == "0":
                has_loop = True
            part = m["path"]
            frames = sorted(n for n in names if PurePosixPath(n).parent == PurePosixPath(part)
                            and n.lower().endswith(".png"))
            if not frames:
                errors.append(f"{part}: нет PNG-кадров")
                continue
            expected = [f"{part}/{i:05d}.png" for i in range(len(frames))]
            if frames != expected:
                errors.append(f"{part}: кадры должны называться 00000.png, 00001.png… подряд")
            for name in frames:
                img = Image.open(io.BytesIO(zf.read(name)))
                if img.size != (width, height):
                    errors.append(f"{name}: размер {img.size}, а в desc.txt {width}×{height}")
                    break
        if not has_loop:
            errors.append("нет бесконечной части (p 0 …): анимация закончится раньше загрузки")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    errors = check(sys.argv[1])
    for e in errors:
        print("ОШИБКА:", e)
    if not errors:
        print(f"OK: {sys.argv[1]}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
