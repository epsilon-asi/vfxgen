"""Look-regression snapshots: three draft frames (25 / 50 / 75 % through) of every library effect,
stored side by side as one PNG in tests/regression/snapshots/<name>.png."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import vfxgen as vfx

SNAP_DIR = Path(__file__).resolve().parent / "snapshots"
FRACTIONS = (0.25, 0.5, 0.75)


def snapshot(name: str) -> np.ndarray:
    info = vfx.validate(name, draft=True)
    n = info.frame_count
    idx = sorted({min(n - 1, int(round(f * (n - 1)))) for f in FRACTIONS})
    r = vfx.render(name, draft=True, workers=1, cache=False, frames=idx)
    return np.concatenate(r.frames, axis=1)


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """(mean abs difference of premultiplied RGBA in 0..255, max abs difference)."""
    if a.shape != b.shape:
        return float("inf"), 255
    fa = a.astype(np.float32)
    fb = b.astype(np.float32)
    pa = np.dstack([fa[..., :3] * fa[..., 3:] / 255, fa[..., 3:]])
    pb = np.dstack([fb[..., :3] * fb[..., 3:] / 255, fb[..., 3:]])
    d = np.abs(pa - pb)
    return float(d.mean()), int(d.max())


def path(name: str) -> Path:
    return SNAP_DIR / f"{name}.png"


def save(name: str, img: np.ndarray) -> Path:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    p = path(name)
    Image.fromarray(img, "RGBA").save(p, optimize=True)
    return p


def load(name: str) -> np.ndarray | None:
    p = path(name)
    return np.asarray(Image.open(p).convert("RGBA")) if p.is_file() else None
