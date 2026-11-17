"""Pixel-art render path.

The scene is rendered natively at low resolution (one output pixel per art pixel, no
supersampling), then every frame is snapped to a palette with ordered (Bayer) dithering, alpha is
made hard (dithered threshold), and an optional 1 px outline is added. All frames share one
palette, so an animation doesn't flicker between colours.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .color import _hex

_BAYER2 = np.array([[0, 2], [3, 1]], np.float32)


def bayer(n: int) -> np.ndarray:
    """Normalised Bayer threshold matrix (n = 2, 4, 8) in [-0.5, 0.5)."""
    m = _BAYER2
    while m.shape[0] < n:
        k = m.shape[0]
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size - 0.5


def parse_palette(spec) -> np.ndarray:
    """Hex list ("#000,#fff,...", or a list of hex strings) -> (k, 3) float RGB 0..255."""
    items = spec.split(",") if isinstance(spec, str) else list(spec)
    cols = [_hex(c) for c in items if str(c).strip()]
    if len(cols) < 2:
        raise ValueError("a pixel palette needs at least 2 colours")
    return np.array(cols, np.float32)


def auto_palette(frames: list[np.ndarray], colors: int) -> np.ndarray:
    """A shared palette for all frames (median cut over every opaque pixel)."""
    px = np.concatenate([f[f[..., 3] > 32][:, :3] for f in frames] or [np.zeros((0, 3), np.uint8)])
    if len(px) == 0:
        return np.array([[0, 0, 0], [255, 255, 255]], np.float32)
    if len(px) > 200_000:
        px = px[:: len(px) // 200_000 + 1]
    side = int(np.ceil(np.sqrt(len(px))))
    pad = np.zeros((side * side, 3), np.uint8)
    pad[: len(px)] = px
    pad[len(px):] = px[0]
    im = Image.fromarray(pad.reshape(side, side, 3), "RGB").quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    pal = np.array(im.getpalette()[: 3 * colors], np.float32).reshape(-1, 3)
    used = np.unique(np.asarray(im))
    return pal[used[used < len(pal)]]


def quantize_frame(frame: np.ndarray, palette: np.ndarray, dither: str = "bayer4", strength: float = 0.5,
                   alpha_threshold: float = 0.5, outline=None) -> np.ndarray:
    h, w = frame.shape[:2]
    f = frame.astype(np.float32)
    if dither != "none":
        n = int(dither.replace("bayer", ""))
        M = np.tile(bayer(n), (h // n + 1, w // n + 1))[:h, :w]
    else:
        M = np.zeros((h, w), np.float32)
    a = f[..., 3] / 255.0
    solid = (a + M * strength) > alpha_threshold
    spread = 255.0 / max(2.0, np.cbrt(len(palette)) * 2)
    rgb = f[..., :3] + (M * strength * spread)[..., None]
    d = ((rgb[:, :, None, :] - palette[None, None, :, :]) ** 2).sum(-1)
    q = palette[np.argmin(d, axis=-1)]
    out = np.zeros((h, w, 4), np.uint8)
    out[..., :3] = np.where(solid[..., None], q, 0).astype(np.uint8)
    out[..., 3] = np.where(solid, 255, 0).astype(np.uint8)
    if outline is not None:
        col = np.array(_hex(outline) if isinstance(outline, str) else outline, np.uint8)
        nb = np.zeros_like(solid)
        nb[1:] |= solid[:-1]
        nb[:-1] |= solid[1:]
        nb[:, 1:] |= solid[:, :-1]
        nb[:, :-1] |= solid[:, 1:]
        ring = nb & ~solid
        out[ring, :3] = col
        out[ring, 3] = 255
    return out
