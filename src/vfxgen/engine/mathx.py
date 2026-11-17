"""Small numeric helpers shared by atoms: easing, smoothstep, seeds, blur,
midpoint displacement and closed-form ballistics."""
from __future__ import annotations

import hashlib
import math
from typing import Callable

import numpy as np

TAU = 2 * math.pi


# --- seeds -------------------------------------------------------------------
def derive_seed(*parts) -> int:
    """Deterministic 63-bit seed from any parts (ints/strings). Stable across runs
    and Python versions (does not use ``hash()``)."""
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(repr(p).encode())
        h.update(b"\x1f")
    return int.from_bytes(h.digest(), "little") & ((1 << 63) - 1)


# --- shaping -------------------------------------------------------------------
def smoothstep(e0, e1, x):
    """Hermite step from 0 at e0 to 1 at e1 (works for e0 > e1 too)."""
    t = np.clip((np.asarray(x, np.float32) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def clamp(x, lo, hi):
    return np.clip(x, lo, hi)


EASES: dict[str, Callable[[float], float]] = {
    "linear": lambda x: x,
    "in": lambda x: x * x * x,
    "out": lambda x: 1 - (1 - x) ** 3,
    "in_out": lambda x: 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2,
    "step": lambda x: 0.0 if x < 1 else 1.0,
    "sine": lambda x: 0.5 - 0.5 * math.cos(math.pi * x),
    "back": lambda x: 1 + 2.70158 * (x - 1) ** 3 + 1.70158 * (x - 1) ** 2,
    "bounce": lambda x: _bounce(x),
}


def _bounce(x: float) -> float:
    n, d = 7.5625, 2.75
    if x < 1 / d:
        return n * x * x
    if x < 2 / d:
        x -= 1.5 / d
        return n * x * x + 0.75
    if x < 2.5 / d:
        x -= 2.25 / d
        return n * x * x + 0.9375
    x -= 2.625 / d
    return n * x * x + 0.984375


def ease(name: str, x: float) -> float:
    """Apply a named easing curve to x in [0, 1]. ``power:N`` = x**N."""
    x = min(max(float(x), 0.0), 1.0)
    if name.startswith("power:"):
        return x ** float(name.split(":", 1)[1])
    try:
        return EASES[name](x)
    except KeyError:
        raise ValueError(f"unknown ease {name!r}; use {', '.join(EASES)} or power:N") from None


def ease_out(x, p: float = 3.0):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** p


# --- blur / shift (float32 images) ---------------------------------------------
def _box1d(a, r, axis, wrap=False):
    k = 2 * r + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    if wrap and r + 1 > a.shape[axis]:     # radius larger than the tile: average whole axis
        m = a.mean(axis=axis, keepdims=True) * k
        return np.broadcast_to(m, a.shape).astype(np.float64)
    c = np.cumsum(np.pad(a, pad, mode="wrap" if wrap else "constant"), axis=axis, dtype=np.float64)
    if axis == 1:
        return c[:, k:] - c[:, :-k]
    return c[k:, :] - c[:-k, :]


def blur(img, sigma, wrap: bool = False):
    """Gaussian blur (sigma in pixels of ``img``), three box passes. Zero outside, or periodic
    with ``wrap`` (tileable output)."""
    if sigma <= 0:
        return np.asarray(img, np.float32)
    r = max(1, int(round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    k = 2 * r + 1
    out = img
    for _ in range(3):
        out = _box1d(_box1d(out, r, 1, wrap), r, 0, wrap) / (k * k)
    return out.astype(np.float32)


def gradient(img, wrap: bool = False):
    """(gy, gx) like np.gradient, periodic when ``wrap``."""
    if not wrap:
        return np.gradient(img)
    gy = (np.roll(img, -1, axis=0) - np.roll(img, 1, axis=0)) * 0.5
    gx = (np.roll(img, -1, axis=1) - np.roll(img, 1, axis=1)) * 0.5
    return gy, gx


def shift(img, dx, dy, wrap: bool = False):
    """Sample img at (x + dx, y + dy) with integer pixels; zero outside (or periodic)."""
    dx, dy = int(round(dx)), int(round(dy))
    if wrap:
        return np.roll(np.roll(img, -dy, axis=0), -dx, axis=1)
    out = np.zeros_like(img)
    h, w = img.shape[:2]
    xs0, xs1 = max(dx, 0), min(w + dx, w)
    ys0, ys1 = max(dy, 0), min(h + dy, h)
    if xs0 < xs1 and ys0 < ys1:
        out[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = img[ys0:ys1, xs0:xs1]
    return out


# --- geometry -------------------------------------------------------------------
def refine(rng: np.random.Generator, pts, rough: float, detail: float) -> np.ndarray:
    """Midpoint displacement: subdivide until segments <= ``detail``, offsetting each
    new midpoint sideways by ~rough * segment length. Existing points are kept, so a
    coarse path keeps its shape (lightning, cracks)."""
    pts = np.asarray(pts, np.float64)
    while True:
        d = pts[1:] - pts[:-1]
        seglen = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        if seglen.max() <= detail:
            return pts
        nrm = np.hstack([-d[:, 1:2], d[:, 0:1]]) / seglen
        off = np.clip(rng.normal(0, rough, (len(d), 1)), -2, 2) * seglen
        mid = (pts[:-1] + pts[1:]) / 2 + nrm * off
        out = np.empty((len(pts) + len(mid), 2))
        out[0::2] = pts
        out[1::2] = mid
        pts = out


def ballistic(p0, v0, t, drag: float = 0.0, gravity: float = 0.0):
    """Closed-form position at time t of a particle with linear drag and gravity (+y down).

    Works with scalars or numpy arrays (vectorised over particles)."""
    t = np.asarray(t, np.float64)
    if drag > 1e-9:
        k = (1 - np.exp(-drag * t)) / drag
        gk = gravity / drag * (t - k)
    else:
        k = t
        gk = 0.5 * gravity * t * t
    p0 = np.asarray(p0, np.float64)
    v0 = np.asarray(v0, np.float64)
    x = p0[..., 0] + v0[..., 0] * k
    y = p0[..., 1] + v0[..., 1] * k + gk
    return np.stack([x, y], axis=-1)
