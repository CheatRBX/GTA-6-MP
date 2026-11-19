#!/usr/bin/env python3
"""Положить файлы из папки-оверлея в ext4-образ без монтирования (через debugfs).

Оверлей повторяет корень образа: overlay/system/etc/foo → /system/etc/foo.
Существующие файлы заменяются. Каждому файлу и папке ставится владелец root:root,
права (0755 для папок и исполняемых, иначе 0644) и SELinux-метка из contexts.txt.

    python3 image/inject.py system.img overlay/ --contexts image/contexts.txt

Образ должен быть без shared_blocks (repack.sh снимает их заранее): иначе удаление
заменяемого файла испортит чужие блоки.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

DEFAULT_CONTEXT = "u:object_r:system_file:s0"


def load_contexts(path: Path | None) -> list[tuple[str, str]]:
    rules: list[tuple[str, str]] = []
    if path is None:
        return rules
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        pattern, context = line.split()
        rules.append((pattern, context))
    return rules


def context_for(path: str, rules: list[tuple[str, str]]) -> str:
    for pattern, context in rules:
        if fnmatch.fnmatchcase(path, pattern):
            return context
    return DEFAULT_CONTEXT


def debugfs(image: Path, request: str) -> str:
    out = subprocess.run(["debugfs", "-R", request, str(image)], capture_output=True, text=True)
    return out.stdout


def exists(image: Path, path: str) -> bool:
    return "Inode:" in debugfs(image, f'stat "{path}"')


def has_shared_blocks(image: Path) -> bool:
    out = subprocess.run(["dumpe2fs", "-h", str(image)], capture_output=True, text=True).stdout
    return "shared_blocks" in out


def build_commands(image: Path, overlay: Path, rules: list[tuple[str, str]], ctx_dir: Path) -> list[str]:
    cmds: list[str] = []
    ctx_files: dict[str, Path] = {}

    def label(path: str) -> None:
        context = context_for(path, rules)
        if context not in ctx_files:
            f = ctx_dir / f"ctx{len(ctx_files)}"
            f.write_bytes(context.encode() + b"\0")  # как у Android: строка с завершающим нулём
            ctx_files[context] = f
        cmds.append(f'ea_set -f "{ctx_files[context]}" "{path}" security.selinux')

    def owner(path: str, mode: int) -> None:
        cmds.append(f'sif "{path}" mode 0{mode:o}')
        cmds.append(f'sif "{path}" uid 0')
        cmds.append(f'sif "{path}" gid 0')

    for root, dirs, files in os.walk(overlay):
        dirs.sort()
        rel_root = PurePosixPath("/") / Path(root).relative_to(overlay).as_posix()
        for d in dirs:
            path = str(rel_root / d)
            if not exists(image, path):
                cmds.append(f'mkdir "{path}"')
                owner(path, 0o040755)
                label(path)
        for name in sorted(files):
            local = Path(root) / name
            path = str(rel_root / name)
            if exists(image, path):
                cmds.append(f'rm "{path}"')
            cmds.append(f'write "{local}" "{path}"')
            executable = os.access(local, os.X_OK)
            owner(path, 0o100755 if executable else 0o100644)
            label(path)
    return cmds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path)
    ap.add_argument("overlay", type=Path)
    ap.add_argument("--contexts", type=Path)
    args = ap.parse_args()

    if has_shared_blocks(args.image):
        print("ОШИБКА: в образе shared_blocks — сначала e2fsck -E unshare_blocks (это делает repack.sh)",
              file=sys.stderr)
        return 1
    rules = load_contexts(args.contexts)
    with tempfile.TemporaryDirectory() as tmp:
        cmds = build_commands(args.image, args.overlay, rules, Path(tmp))
        script = Path(tmp) / "cmds"
        script.write_text("\n".join(cmds) + "\n")
        res = subprocess.run(["debugfs", "-w", "-f", str(script), str(args.image)],
                             capture_output=True, text=True)
    errors = [ln for ln in res.stderr.splitlines()
              if ln.strip() and not ln.startswith("debugfs ")]
    if res.returncode != 0 or errors:
        print("\n".join(errors) or res.stderr, file=sys.stderr)
        return 1
    files = sum(1 for c in cmds if c.startswith("write "))
    print(f"inject: {files} файлов → {args.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
