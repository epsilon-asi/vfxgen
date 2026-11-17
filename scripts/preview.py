#!/usr/bin/env python3
"""Developer preview: render a scene / effect / atom and write contact sheets over dark and
checkerboard backgrounds for quick visual inspection (not a sprite output).

    python scripts/preview.py explosion [--set radius=1.2] [--draft] [-o /tmp/out]
Prints the paths of `<name>_dark.png` and `<name>_checker.png`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import vfxgen as vfx
from vfxgen.cli import split_sets


def composite(sheet: np.ndarray, bg: np.ndarray) -> np.ndarray:
    a = sheet[..., 3:4].astype(np.float32) / 255
    return (sheet[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--colors")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--draft", action="store_true")
    ap.add_argument("--columns", type=int, default=8)
    ap.add_argument("--every", type=int, default=1, help="only every Nth frame")
    ap.add_argument("-o", "--out", default="/tmp/vfx_preview")
    a = ap.parse_args()
    vars_, over = split_sets(a.target, a.set)
    r = vfx.render(a.target, vars_, overrides=over, seed=a.seed, colors=a.colors, draft=a.draft)
    frames = r.frames[::a.every]
    from vfxgen.output.atlas import build_sheet
    sheet, _ = build_sheet(frames, r.fps, r.loop, columns=min(a.columns, len(frames)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(a.target).stem
    dark = composite(sheet, np.array((18, 20, 28), np.float32))
    yy, xx = np.mgrid[0:sheet.shape[0], 0:sheet.shape[1]]
    v = np.where(((xx // 16) + (yy // 16)) % 2 == 0, 95, 125).astype(np.float32)
    chk = composite(sheet, np.dstack([v, v, v]))
    pd, pc = out / f"{name}_dark.png", out / f"{name}_checker.png"
    Image.fromarray(dark).save(pd)
    Image.fromarray(chk).save(pc)
    print(f"{len(r.frames)} frames {r.width}x{r.height} @ {r.fps:g} fps, {r.elapsed:.1f}s")
    print(pd)
    print(pc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
