#!/usr/bin/env python3
"""Генератор загрузочной анимации Android (bootanimation.zip).

Три части, как их понимает Android (frameworks/base/cmds/bootanimation/FORMAT.md):
  part0  c 1 0  вступление: логотип «прорисовывается» по кругу, вспышка, появляется надпись
  part1  p 0 0  цикл, пока система грузится: блик пробегает по логотипу, свечение дышит
  part2  c 1 0  финал после загрузки: логотип чуть увеличивается и гаснет

Архив пишется БЕЗ сжатия (ZIP_STORED): со сжатым архивом Android анимацию не покажет.

    python3 brand/bootanim/build.py                         # → brand/out/bootanimation.zip
    python3 brand/bootanim/build.py --preview brand/out/bootanimation-preview.mp4 \
        --gif docs/img/bootanimation-preview.gif
"""
from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import logo  # noqa: E402

OUT = logo.BRAND / "out"


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.array([int(value[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32) / 255.0


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def ease_out_cubic(x: float) -> float:
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in_out(x: float) -> float:
    x = clamp01(x)
    return 3 * x * x - 2 * x * x * x


class Scene:
    """Готовые слои (символ, свечение, надпись) и их композиция в кадр."""

    def __init__(self, brand: dict, variant: str, size: int):
        self.size = size
        colors = brand["colors"]
        self.bg = hex_rgb(colors["background"])
        self.glow_rgb = hex_rgb(colors["primary"]) * 0.55 + hex_rgb(colors["accent"]) * 0.45
        self.sym_px = int(size * 0.36)
        sym = logo.render_png(logo.svg(variant, colors), self.sym_px)
        self.sym = np.asarray(sym, dtype=np.float32) / 255.0
        yy, xx = np.mgrid[0:self.sym_px, 0:self.sym_px].astype(np.float32)
        c = (self.sym_px - 1) / 2
        # Угол от «12 часов» по часовой стрелке, 0…360.
        self.angle = (np.degrees(np.arctan2(yy - c, xx - c)) + 90.0) % 360.0
        # Диагональная координата 0…1 для блика.
        self.diag = (xx + (self.sym_px - yy)) / (2.0 * self.sym_px)
        word = logo.wordmark(brand["name"].upper(), int(size * 0.046), colors["foreground"])
        self.word = np.asarray(word, dtype=np.float32) / 255.0
        self.word_gap = int(size * 0.07)

    def _scaled(self, rgba: np.ndarray, scale: float) -> np.ndarray:
        if abs(scale - 1.0) < 1e-3:
            return rgba
        img = Image.fromarray((rgba * 255).astype(np.uint8), "RGBA")
        px = max(1, int(round(img.width * scale)))
        return np.asarray(img.resize((px, px), Image.LANCZOS), dtype=np.float32) / 255.0

    @staticmethod
    def _over(canvas: np.ndarray, layer: np.ndarray, x: int, y: int, opacity: float = 1.0) -> None:
        """Наложить RGBA-слой на RGB-холст с обрезкой по краям."""
        h, w = layer.shape[:2]
        H, W = canvas.shape[:2]
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
        if x0 >= x1 or y0 >= y1:
            return
        part = layer[y0 - y:y1 - y, x0 - x:x1 - x]
        a = part[..., 3:4] * opacity
        canvas[y0:y1, x0:x1] = canvas[y0:y1, x0:x1] * (1 - a) + part[..., :3] * a

    def _glow(self, sym: np.ndarray, strength: float) -> np.ndarray:
        alpha = Image.fromarray((sym[..., 3] * 255).astype(np.uint8), "L")
        pad = int(self.sym_px * 0.25)
        big = Image.new("L", (alpha.width + 2 * pad, alpha.height + 2 * pad), 0)
        big.paste(alpha, (pad, pad))
        blurred = np.asarray(big.filter(ImageFilter.GaussianBlur(self.sym_px * 0.07)), dtype=np.float32) / 255.0
        out = np.zeros(blurred.shape + (4,), dtype=np.float32)
        out[..., :3] = self.glow_rgb
        out[..., 3] = np.clip(blurred * strength, 0, 1)
        return out

    def frame(self, *, reveal: float = 360.0, scale: float = 1.0, glow: float = 0.35,
              shimmer: float | None = None, word_alpha: float = 1.0, word_shift: float = 0.0,
              opacity: float = 1.0) -> Image.Image:
        sym = self.sym.copy()
        if reveal < 360.0:
            feather = 14.0
            mask = np.clip((reveal - self.angle) / feather + 0.5, 0, 1)
            sym[..., 3] *= mask
        if shimmer is not None:
            band = np.exp(-((self.diag - shimmer) / 0.07) ** 2) * 0.6
            sym[..., :3] = np.clip(sym[..., :3] + band[..., None] * (1 - sym[..., :3]), 0, 1)
        sym = self._scaled(sym, scale)
        canvas = np.empty((self.size, self.size, 3), dtype=np.float32)
        canvas[:] = self.bg
        word_h = self.word.shape[0]
        block_h = self.sym_px + self.word_gap + word_h
        top = (self.size - block_h) // 2
        cx, cy = self.size // 2, top + self.sym_px // 2
        if glow > 0:
            g = self._glow(sym, glow)
            self._over(canvas, g, cx - g.shape[1] // 2, cy - g.shape[0] // 2, opacity)
        self._over(canvas, sym, cx - sym.shape[1] // 2, cy - sym.shape[0] // 2, opacity)
        if word_alpha > 0:
            wy = top + self.sym_px + self.word_gap + int(word_shift)
            self._over(canvas, self.word, cx - self.word.shape[1] // 2, wy, word_alpha * opacity)
        return Image.fromarray((np.clip(canvas, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")


def build_parts(scene: Scene, fps: int, intro_s: float, loop_s: float, outro_s: float):
    n0, n1, n2 = (max(2, round(s * fps)) for s in (intro_s, loop_s, outro_s))
    intro = []
    for i in range(n0):
        t = i / (n0 - 1)
        flash = math.exp(-((t - 0.72) / 0.12) ** 2) * 0.9
        intro.append(scene.frame(
            reveal=388.0 * ease_in_out(t / 0.7) - 14.0 if t < 0.7 else 360.0,
            scale=0.86 + 0.14 * ease_out_cubic(t / 0.8),
            glow=0.35 * ease_out_cubic(t / 0.7) + flash,
            word_alpha=ease_out_cubic((t - 0.55) / 0.4),
            word_shift=24 * (1 - ease_out_cubic((t - 0.55) / 0.4)),
        ))
    loop = []
    for i in range(n1):
        t = i / n1  # без последнего кадра: цикл бесшовный
        loop.append(scene.frame(
            shimmer=-0.4 + 1.8 * t,
            glow=0.35 + 0.12 * math.sin(2 * math.pi * t),
        ))
    outro = []
    for i in range(n2):
        t = (i + 1) / n2
        outro.append(scene.frame(scale=1.0 + 0.12 * ease_in_out(t), opacity=1.0 - ease_in_out(t),
                                 glow=0.35 * (1 - t)))
    return {"part0": intro, "part1": loop, "part2": outro}


def write_zip(parts: dict, size: int, fps: int, bg: str, path: Path) -> None:
    desc = f"{size} {size} {fps}\nc 1 0 part0 {bg}\np 0 0 part1 {bg}\nc 1 0 part2 {bg}\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("desc.txt", desc)
        for name, frames in parts.items():
            for i, img in enumerate(frames):
                with tempfile.SpooledTemporaryFile() as buf:
                    img.save(buf, "PNG", optimize=True)
                    buf.seek(0)
                    zf.writestr(f"{name}/{i:05d}.png", buf.read())


def write_preview(parts: dict, fps: int, mp4: Path | None, gif: Path | None) -> None:
    if not shutil.which("ffmpeg"):
        print("ffmpeg не найден — превью пропущено", file=sys.stderr)
        return
    sequence = parts["part0"] + parts["part1"] * 2 + parts["part2"]
    with tempfile.TemporaryDirectory() as tmp:
        for i, img in enumerate(sequence):
            img.save(Path(tmp) / f"{i:05d}.png")
        src = ["-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(Path(tmp) / "%05d.png")]
        if mp4:
            mp4.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", *src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                            str(mp4)], check=True)
        if gif:
            gif.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", *src, "-vf",
                            "fps=20,scale=360:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];"
                            "[b][p]paletteuse=dither=bayer:bayer_scale=4", str(gif)], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=logo.VARIANTS, help="вариант логотипа (по умолчанию из brand.json)")
    ap.add_argument("--out", default=str(OUT / "bootanimation.zip"))
    ap.add_argument("--size", type=int, help="сторона кадра в px (по умолчанию из brand.json)")
    ap.add_argument("--preview", help="сохранить превью MP4")
    ap.add_argument("--gif", help="сохранить превью GIF (уменьшенное)")
    args = ap.parse_args()

    brand = logo.load_brand()
    cfg = brand["bootanimation"]
    size = args.size or cfg["size"]
    variant = args.variant or brand["logo_variant"]
    scene = Scene(brand, variant, size)
    parts = build_parts(scene, cfg["fps"], cfg["intro_seconds"], cfg["loop_seconds"], cfg["outro_seconds"])
    out = Path(args.out)
    write_zip(parts, size, cfg["fps"], brand["colors"]["background"], out)
    write_preview(parts, cfg["fps"], Path(args.preview) if args.preview else None,
                  Path(args.gif) if args.gif else None)
    counts = ", ".join(f"{k}: {len(v)}" for k, v in parts.items())
    print(f"bootanimation: {out} ({out.stat().st_size / 1e6:.1f} МБ; кадров {counts}; вариант {variant})")


if __name__ == "__main__":
    main()
