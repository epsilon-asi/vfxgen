#!/usr/bin/env python3
"""Fidelity sheet: legacy renders (out/<name>/) vs the new library scenes, side by side.

For each effect: top row = legacy, bottom row = new, sampled at the same normalised times,
composited on a dark background. Writes docs/fidelity/<name>.png and docs/fidelity/all.png.

    python scripts/compare_legacy.py [names...] [--draft] [--samples 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import vfxgen as vfx

ROOT = Path(__file__).resolve().parent.parent
LEGACY = ROOT / "out"
OUT = ROOT / "docs" / "fidelity"
BG = np.array((18, 20, 28), np.float32)


def on_dark(f: np.ndarray) -> np.ndarray:
    a = f[..., 3:4].astype(np.float32) / 255
    return (f[..., :3] * a + BG * (1 - a)).astype(np.uint8)


def fit(img: np.ndarray, h: int) -> Image.Image:
    im = Image.fromarray(img)
    w = max(1, round(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def sample(frames: list[np.ndarray], k: int) -> list[np.ndarray]:
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return [frames[i] for i in idx]


def compare(name: str, k: int, draft: bool, cell_h: int = 160) -> Image.Image | None:
    lf = sorted((LEGACY / name).glob(f"{name}_*.png"))
    if not lf:
        print(f"  {name}: no legacy frames, skipped")
        return None
    legacy = [np.asarray(Image.open(p).convert("RGBA")) for p in lf]
    new = vfx.render(name, draft=draft).frames
    rows = []
    for label, frames in (("legacy", legacy), ("new", new)):
        cells = [fit(on_dark(f), cell_h) for f in sample(frames, k)]
        w = sum(c.width for c in cells) + 4 * (len(cells) - 1)
        row = Image.new("RGB", (w + 70, cell_h), tuple(int(v) for v in BG))
        ImageDraw.Draw(row).text((6, cell_h // 2 - 6), label, fill=(200, 200, 210))
        x = 70
        for c in cells:
            row.paste(c, (x, 0))
            x += c.width + 4
        rows.append(row)
    W = max(r.width for r in rows)
    img = Image.new("RGB", (W, cell_h * 2 + 26), tuple(int(v) for v in BG))
    ImageDraw.Draw(img).text((6, 6), name, fill=(235, 235, 245))
    img.paste(rows[0], (0, 22))
    img.paste(rows[1], (0, 24 + cell_h))
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--samples", type=int, default=6)
    a = ap.parse_args()
    names = a.names or [e["name"] for e in vfx.list_effects()]
    OUT.mkdir(parents=True, exist_ok=True)
    sheets = []
    for n in names:
        img = compare(n, a.samples, a.draft)
        if img is None:
            continue
        img.save(OUT / f"{n}.png")
        sheets.append(img)
        print(OUT / f"{n}.png")
    if sheets:
        W = max(s.width for s in sheets)
        H = sum(s.height + 6 for s in sheets)
        allimg = Image.new("RGB", (W, H), tuple(int(v) for v in BG))
        y = 0
        for s in sheets:
            allimg.paste(s, (0, y))
            y += s.height + 6
        allimg.save(OUT / "all.png")
        print(OUT / "all.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
