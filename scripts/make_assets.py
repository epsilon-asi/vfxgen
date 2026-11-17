#!/usr/bin/env python3
"""Generate the built-in sample image assets in ``src/vfxgen/library/assets/``.

Deterministic and procedural (no randomness, no external files): every shape is drawn at 4x
with PIL, outlined by dilating the silhouette, then box-downsampled in premultiplied alpha so
edges are clean.

    uv run python scripts/make_assets.py [--out DIR]

Assets:
* ``hero.png``  (96x128) small readable character (sword, red tunic), feet at the bottom.
* ``coin.png``  (32x32) gold coin with rim, emblem and shine.
* ``shard.png`` (24x40) two-faceted ice / crystal shard.
* ``leaf.png``  (32x32) green leaf with midrib and stem.
* ``star.png``  (32x32) chunky five-point star.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SS = 4                                   # supersampling factor
OUT = Path(__file__).resolve().parent.parent / "src" / "vfxgen" / "library" / "assets"


class Canvas:
    """RGBA drawing surface at SS x the target size, coordinates in target pixels."""

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.im = Image.new("RGBA", (w * SS, h * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.im)

    def _p(self, pts):
        return [(x * SS, y * SS) for x, y in pts]

    def poly(self, pts, col):
        self.d.polygon(self._p(pts), fill=col)

    def ellipse(self, cx, cy, rx, ry, col):
        self.d.ellipse([(cx - rx) * SS, (cy - ry) * SS, (cx + rx) * SS, (cy + ry) * SS], fill=col)

    def line(self, pts, width, col):
        self.d.line(self._p(pts), fill=col, width=max(1, int(round(width * SS))), joint="curve")
        r = width / 2
        for x, y in (pts[0], pts[-1]):
            self.ellipse(x, y, r, r, col)

    def rect(self, x0, y0, x1, y1, col, radius=0.0):
        self.d.rounded_rectangle([x0 * SS, y0 * SS, x1 * SS, y1 * SS], radius=radius * SS, fill=col)

    def capsule(self, p0, p1, r, col):
        self.line([p0, p1], 2 * r, col)


def outlined(fill: Canvas, outline_px: float, col) -> Canvas:
    """Put a solid outline of ``outline_px`` target pixels under the drawing."""
    a = fill.im.getchannel("A")
    k = int(round(outline_px * SS)) * 2 + 1
    dil = a.filter(ImageFilter.MaxFilter(k))
    base = Image.new("RGBA", fill.im.size, col)
    base.putalpha(dil)
    base.alpha_composite(fill.im)
    out = Canvas(fill.w, fill.h)
    out.im = base
    out.d = ImageDraw.Draw(out.im)
    return out


def finish(c: Canvas) -> Image.Image:
    """Box-downsample in premultiplied alpha."""
    arr = np.asarray(c.im, np.float64) / 255.0
    arr[..., :3] *= arr[..., 3:4]
    h, w = c.h, c.w
    arr = arr.reshape(h, SS, w, SS, 4).mean(axis=(1, 3))
    a = arr[..., 3:4]
    rgb = np.where(a > 1e-6, arr[..., :3] / np.maximum(a, 1e-6), 0)
    out = np.dstack([rgb, a])
    out = np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
    out[out[..., 3] == 0] = 0
    return Image.fromarray(out, "RGBA")


def star_pts(cx, cy, r_out, r_in, n=5, rot=-90.0):
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot + i * 180.0 / n)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


# --- assets ----------------------------------------------------------------------------------
def hero() -> Image.Image:
    c = Canvas(96, 128)
    skin, skin_d = (242, 201, 160, 255), (214, 160, 120, 255)
    tunic, tunic_d = (200, 56, 46, 255), (150, 36, 34, 255)
    pants = (58, 74, 122, 255)
    boot, boot_d = (96, 60, 36, 255), (70, 42, 26, 255)
    hair, hair_d = (110, 64, 30, 255), (80, 44, 20, 255)
    steel, steel_d = (214, 222, 232, 255), (150, 164, 182, 255)
    gold = (242, 190, 50, 255)
    dark = (30, 24, 36, 255)
    # sword (behind the right hand)
    c.poly([(73, 92), (79, 34), (83, 28), (85, 35), (79, 93)], steel)
    c.poly([(79, 34), (83, 28), (85, 35), (79, 93), (76, 92)], steel)
    c.poly([(81, 33), (83, 29), (84, 35), (79, 92), (78, 92)], steel_d)
    c.rect(69, 89, 85, 93, gold, 1.5)                         # cross guard
    c.capsule((76, 93), (75, 101), 2.2, (110, 70, 40, 255))  # grip
    c.ellipse(75, 103, 2.6, 2.6, gold)                       # pommel
    # legs + boots
    c.rect(36, 92, 47, 116, pants, 3)
    c.rect(50, 92, 61, 116, pants, 3)
    c.rect(32, 110, 47, 125, boot, 3)
    c.rect(50, 110, 65, 125, boot, 3)
    c.rect(32, 121, 47, 125, boot_d, 2)
    c.rect(50, 121, 65, 125, boot_d, 2)
    # arms (sleeves + hands)
    c.capsule((30, 62), (22, 88), 5.5, tunic_d)
    c.capsule((66, 62), (74, 88), 5.5, tunic_d)
    c.ellipse(21, 91, 5, 5, skin)
    c.ellipse(75, 92, 5, 5, skin)
    # torso tunic
    c.poly([(31, 56), (65, 56), (71, 100), (25, 100)], tunic)
    c.poly([(48, 56), (65, 56), (71, 100), (48, 100)], tunic_d)
    c.poly([(40, 56), (56, 56), (48, 68)], skin_d)            # collar V
    c.rect(27, 80, 69, 87, (64, 44, 30, 255), 1)             # belt
    c.rect(44, 79, 52, 88, gold, 1.5)                         # buckle
    c.rect(46.5, 81.5, 49.5, 85.5, (64, 44, 30, 255), 0.5)
    # head
    c.ellipse(48, 38, 17, 18, skin)
    c.poly([(48, 22), (65, 30), (65, 42), (48, 40)], skin_d)  # shaded side (covered by hair top)
    c.ellipse(48, 38, 15.5, 16.5, skin)
    c.poly([(48, 38), (63, 34), (64, 46), (54, 54)], skin_d)
    # hair: cap + fringe spikes
    c.ellipse(48, 30, 18, 12, hair)
    c.poly([(30, 30), (36, 44), (40, 32), (46, 40), (50, 31), (56, 40), (60, 31), (63, 43), (67, 30)], hair)
    c.poly([(48, 19), (66, 28), (67, 32), (60, 31)], hair_d)
    # face
    c.rect(40, 39, 43.5, 45, dark, 1.2)
    c.rect(53, 39, 56.5, 45, dark, 1.2)
    c.rect(41, 39.5, 42.3, 41, (255, 255, 255, 255), 0.4)
    c.rect(54, 39.5, 55.3, 41, (255, 255, 255, 255), 0.4)
    c.line([(45, 50), (51, 50)], 1.2, (150, 70, 60, 255))
    return finish(outlined(c, 1.6, dark))


def coin() -> Image.Image:
    c = Canvas(32, 32)
    c.ellipse(16, 16, 14.5, 14.5, (184, 120, 20, 255))       # rim shadow
    c.ellipse(15.4, 15.4, 13.6, 13.6, (250, 196, 60, 255))   # rim lit
    c.ellipse(16, 16, 10.6, 10.6, (214, 150, 30, 255))       # face groove
    c.ellipse(16.3, 16.3, 9.6, 9.6, (246, 186, 50, 255))     # face
    c.poly(star_pts(16.3, 16.6, 6.4, 2.8), (255, 226, 120, 255))
    c.poly(star_pts(16.8, 17.2, 5.2, 2.2), (226, 160, 36, 255))
    c.poly(star_pts(16.3, 16.6, 5.2, 2.2), (255, 220, 110, 255))
    c.line([(8.5, 11.5), (11.5, 7.5)], 2.0, (255, 250, 220, 255))    # shine
    c.ellipse(7.6, 14.4, 1.0, 1.0, (255, 250, 220, 255))
    return finish(outlined(c, 1.0, (96, 56, 10, 255)))


def shard() -> Image.Image:
    c = Canvas(24, 40)
    tip, left, right, base = (13, 2), (4, 17), (20, 21), (10, 38)
    mid = (12, 20)
    c.poly([tip, left, base, mid], (168, 228, 255, 255))
    c.poly([tip, mid, base, right], (70, 160, 220, 255))
    c.poly([left, mid, base], (120, 200, 245, 255))
    c.line([(12.5, 5), (6.5, 17)], 1.0, (240, 252, 255, 255))
    return finish(outlined(c, 1.0, (24, 70, 110, 255)))


def leaf() -> Image.Image:
    c = Canvas(32, 32)
    # leaf body: lens shape along the diagonal
    pts = []
    for i in range(41):
        s = i / 40
        w = math.sin(math.pi * s) ** 0.8 * 7.5
        pts.append((s, w))
    a = math.radians(-45)
    ca, sa = math.cos(a), math.sin(a)

    def P(s, off):
        x, y = -11 + 22 * s, off
        return (16 + ca * x - sa * y - 1, 16 + sa * x + ca * y + 1)
    upper = [P(s, -w) for s, w in pts]
    lower = [P(s, w) for s, w in pts][::-1]
    c.poly(upper + lower, (92, 176, 60, 255))
    c.poly([P(s, 0) for s, _ in pts] + lower, (60, 138, 44, 255))
    c.line([P(-0.18, 0), P(0.9, 0)], 1.2, (200, 236, 140, 255))
    for s in (0.3, 0.5, 0.7):
        c.line([P(s, 0), P(s + 0.12, -5)], 0.8, (170, 220, 120, 255))
    return finish(outlined(c, 1.0, (30, 70, 24, 255)))


def star() -> Image.Image:
    c = Canvas(32, 32)
    c.poly(star_pts(16, 17, 14.5, 6.6), (226, 150, 20, 255))
    c.poly(star_pts(15.5, 16.4, 13.2, 6.0), (255, 214, 60, 255))
    c.poly(star_pts(14.8, 15.6, 6.0, 2.6), (255, 246, 190, 255))
    return finish(outlined(c, 1.2, (110, 60, 10, 255)))


ASSETS = {"hero.png": hero, "coin.png": coin, "shard.png": shard, "leaf.png": leaf, "star.png": star}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in ASSETS.items():
        im = fn()
        im.save(out / name, optimize=False, compress_level=9)
        print(out / name, im.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
