#!/usr/bin/env python3
"""Генератор логотипа: три варианта в SVG, иконки, PNG всех размеров и лист сравнения.

Вся геометрия описана здесь, а цвета и имя берутся из brand/brand.json,
поэтому SVG, PNG и загрузочная анимация всегда совпадают.

    python3 brand/tools/logo.py            # SVG в brand/logo/, PNG в brand/out/logo/
    python3 brand/tools/logo.py --sheet docs/img/logo-variants.png
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BRAND = ROOT / "brand"
VIEW = 512
C = VIEW / 2
VARIANTS = ("orbit", "shard", "wing")
PNG_SIZES = (48, 72, 96, 144, 192, 512, 1024)


def load_brand() -> dict:
    return json.loads((BRAND / "brand.json").read_text(encoding="utf-8"))


def _pt(angle_deg: float, radius: float) -> tuple[float, float]:
    a = math.radians(angle_deg)
    return C + radius * math.cos(a), C + radius * math.sin(a)


def _fmt(p: tuple[float, float]) -> str:
    return f"{p[0]:.2f},{p[1]:.2f}"


def _gradient(colors: dict, gid: str = "g") -> str:
    return (
        f'<linearGradient id="{gid}" x1="80" y1="440" x2="440" y2="72" gradientUnits="userSpaceOnUse">'
        f'<stop offset="0" stop-color="{colors["primary"]}"/>'
        f'<stop offset="1" stop-color="{colors["accent"]}"/>'
        "</linearGradient>"
    )


def symbol_orbit(colors: dict) -> tuple[str, str]:
    """Незамкнутое кольцо «C» и точка-спутник в разрыве."""
    radius, width, half_gap = 170, 60, 42
    start, end = _pt(half_gap, radius), _pt(360 - half_gap, radius)
    dot = _pt(0, radius)
    body = (
        f'<path d="M{_fmt(start)} A{radius},{radius} 0 1 1 {_fmt(end)}" fill="none" '
        f'stroke="url(#g)" stroke-width="{width}" stroke-linecap="round"/>'
        f'<circle cx="{dot[0]:.2f}" cy="{dot[1]:.2f}" r="34" fill="{colors["accent"]}"/>'
    )
    return _gradient(colors), body


def symbol_shard(colors: dict) -> tuple[str, str]:
    """Шестиугольник без правой грани и ромб в разрыве."""
    radius, width = 175, 54
    pts = [_pt(a, radius) for a in (30, 90, 150, 210, 270, 330)]
    d = "M" + " L".join(_fmt(p) for p in pts)
    mid_x = C + radius * math.cos(math.radians(30))
    s = 40
    diamond = f"M{mid_x:.2f},{C - s} L{mid_x + s:.2f},{C} L{mid_x:.2f},{C + s} L{mid_x - s:.2f},{C} Z"
    body = (
        f'<path d="{d}" fill="none" stroke="url(#g)" stroke-width="{width}" '
        'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<path d="{diamond}" fill="{colors["accent"]}"/>'
    )
    return _gradient(colors), body


def symbol_wing(colors: dict) -> tuple[str, str]:
    """Голова ворона из граней; глаз — настоящее отверстие (evenodd), без масок."""
    dx, dy = -59, -13  # центрируем фигуру в квадрате 512×512
    head = [(x + dx, y + dy) for x, y in (
        (170, 420), (160, 300), (185, 205), (240, 140), (315, 118), (370, 135),
        (470, 200), (372, 205), (345, 250), (335, 330), (355, 420))]
    eye_x, eye_y, eye_r = 318 + dx, 172 + dy, 17
    d = "M" + " L".join(f"{x},{y}" for x, y in head) + " Z"
    d += (f" M{eye_x - eye_r},{eye_y} a{eye_r},{eye_r} 0 1 0 {2 * eye_r},0"
          f" a{eye_r},{eye_r} 0 1 0 {-2 * eye_r},0 Z")
    body = (f'<path d="{d}" fill="url(#g)" fill-rule="evenodd" stroke="url(#g)" '
            'stroke-width="12" stroke-linejoin="round"/>')
    return _gradient(colors), body


SYMBOLS = {"orbit": symbol_orbit, "shard": symbol_shard, "wing": symbol_wing}


def svg(variant: str, colors: dict, icon: bool = False) -> str:
    defs, body = SYMBOLS[variant](colors)
    if icon:
        # Иконка: скруглённый квадрат фона, символ уменьшен до 62 %.
        scale = 0.62
        off = C * (1 - scale)
        bg = (
            f'<rect width="{VIEW}" height="{VIEW}" rx="112" fill="{colors["surface"]}"/>'
            f'<rect x="8" y="8" width="{VIEW - 16}" height="{VIEW - 16}" rx="106" fill="none" '
            f'stroke="{colors["primary"]}" stroke-opacity="0.18" stroke-width="4"/>'
        )
        body = bg + f'<g transform="translate({off:.2f},{off:.2f}) scale({scale})">{body}</g>'
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {VIEW} {VIEW}" width="{VIEW}" height="{VIEW}">'
        f"<defs>{defs}</defs>{body}</svg>\n"
    )


def render_png(svg_text: str, size: int):
    """SVG → PIL.Image (RGBA)."""
    import cairosvg
    from PIL import Image

    data = cairosvg.svg2png(bytestring=svg_text.encode(), output_width=size, output_height=size)
    return Image.open(io.BytesIO(data)).convert("RGBA")


def wordmark(text: str, height: int, color: str, tracking: float = 0.18):
    """Надпись шрифтом Inter Display Bold с разрядкой; возвращает RGBA."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(str(BRAND / "fonts" / "InterDisplay-Bold.otf"), height)
    gap = int(height * tracking)
    widths = [font.getbbox(ch)[2] - font.getbbox(ch)[0] for ch in text]
    total = sum(widths) + gap * (len(text) - 1)
    asc, desc = font.getmetrics()
    img = Image.new("RGBA", (total + 4, asc + desc), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x = 0
    for ch, w in zip(text, widths):
        draw.text((x - font.getbbox(ch)[0], 0), ch, font=font, fill=color)
        x += w + gap
    return img.crop(img.getbbox())


def sheet(path: Path, brand: dict) -> None:
    """Лист сравнения трёх вариантов: символ, иконка, надпись."""
    from PIL import Image, ImageDraw, ImageFont

    colors = brand["colors"]
    cell, pad = 520, 40
    out = Image.new("RGB", (cell * 3, cell + 170), colors["background"])
    label_font = ImageFont.truetype(str(BRAND / "fonts" / "InterDisplay-Bold.otf"), 30)
    draw = ImageDraw.Draw(out)
    word = wordmark(brand["name"].upper(), 54, colors["foreground"])
    for i, variant in enumerate(VARIANTS):
        x0 = i * cell
        sym_size = cell - 2 * pad - 40
        sym = render_png(svg(variant, colors), sym_size)
        out.paste(sym, (x0 + (cell - sym_size) // 2, pad), sym)
        icon = render_png(svg(variant, colors, icon=True), 120)
        out.paste(icon, (x0 + pad, cell - 10), icon)
        out.paste(word, (x0 + pad + 150, cell + 20), word)
        draw.text((x0 + pad + 150, cell + 100), f"{chr(65 + i)} · {variant}", font=label_font,
                  fill=colors["accent"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, optimize=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(BRAND / "out" / "logo"), help="куда класть PNG")
    ap.add_argument("--sheet", help="сохранить лист сравнения вариантов в этот PNG")
    args = ap.parse_args()

    brand = load_brand()
    colors = brand["colors"]
    out = Path(args.out)
    for variant in VARIANTS:
        for icon in (False, True):
            name = f"{variant}-icon" if icon else variant
            text = svg(variant, colors, icon=icon)
            (BRAND / "logo" / f"{name}.svg").write_text(text, encoding="utf-8")
            target = out / name
            target.mkdir(parents=True, exist_ok=True)
            for size in PNG_SIZES:
                render_png(text, size).save(target / f"{size}.png", optimize=True)
    if args.sheet:
        sheet(Path(args.sheet), brand)
    print(f"logo: {len(VARIANTS)} вариантов → {BRAND / 'logo'}, PNG → {out}")


if __name__ == "__main__":
    main()
