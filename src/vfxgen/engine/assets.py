"""Image assets (your own sprites / textures) used by atoms.

Paths resolve relative to the scene file's folder, then each ``VFXGEN_PATH`` folder, then the
built-in ``library/assets`` folder. Images load as float32 **premultiplied** RGBA (h, w, 4) in 0..1
and are cached per process. Every loaded file's content hash is recorded so render caching
invalidates when an asset changes.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..errors import ResolveError

ASSET_DIR = Path(__file__).resolve().parent.parent / "library" / "assets"
_cache: dict[str, np.ndarray] = {}
LOADED: dict[str, str] = {}          # resolved path -> sha256 prefix (cache keys / explicit meta)
_track: list | None = None


def begin_tracking() -> None:
    """Start recording which assets a build loads (for the render cache key)."""
    global _track
    _track = []


def end_tracking() -> dict[str, str]:
    global _track
    used = {k: LOADED.get(k, "") for k in (_track or [])}
    _track = None
    return used


def find_asset(path: str, base_dir: Optional[Path] = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        if p.is_file():
            return p
        raise ResolveError(f"asset not found: {path}")
    dirs = [base_dir] if base_dir else []
    dirs += [Path(d).expanduser() for d in os.environ.get("VFXGEN_PATH", "").split(os.pathsep) if d.strip()]
    dirs.append(ASSET_DIR)
    for d in dirs:
        if d is not None and (d / p).is_file():
            return (d / p).resolve()
    raise ResolveError(f"asset {path!r} not found (looked in: {', '.join(str(d) for d in dirs if d)})")


def load_image(path: str, base_dir: Optional[Path] = None) -> np.ndarray:
    """Premultiplied float RGBA (h, w, 4) in 0..1."""
    f = find_asset(path, base_dir)
    key = str(f)
    if key not in _cache:
        data = f.read_bytes()
        LOADED[key] = hashlib.sha256(data).hexdigest()[:16]
        im = np.asarray(Image.open(f).convert("RGBA"), np.float32) / 255.0
        im[..., :3] *= im[..., 3:4]
        _cache[key] = im
    if _track is not None:
        _track.append(key)
    return _cache[key]


def bilinear(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Sample ``img`` (h, w[, c]) at pixel coordinates (pixel centres at i + 0.5); zero outside."""
    h, w = img.shape[:2]
    x = np.asarray(px, np.float32) - 0.5
    y = np.asarray(py, np.float32) - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    out = 0.0
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            xi, yi = x0 + dx, y0 + dy
            ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            v = img[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)]
            wgt = (wx * wy * ok).astype(np.float32)
            out = out + v * (wgt[..., None] if img.ndim == 3 else wgt)
    return np.asarray(out, np.float32)


# --- sprite analysis helpers (emitters, outlines, dissolves) -----------------------------------
def edge_mask(alpha: np.ndarray, width: float = 1.5) -> np.ndarray:
    """Soft band along the silhouette edge (0..1)."""
    from .mathx import blur
    b = blur(alpha, max(width, 0.5))
    return np.clip(4 * b * (1 - b), 0, 1) * (b > 0.02)


def glow_mask(alpha: np.ndarray, radius: float) -> np.ndarray:
    """Soft outward halo of the silhouette (outline glow), 0..1, excluding the interior."""
    from .mathx import blur
    b = blur(alpha, max(radius, 0.5))
    return np.clip(b * 2.2, 0, 1) * (1 - np.clip(alpha * 1.5, 0, 1))


def sample_points(weight: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """n points (x, y in image pixels) distributed proportionally to ``weight``."""
    w = np.clip(weight, 0, None).ravel().astype(np.float64)
    if n <= 0 or w.sum() <= 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(w.size, size=n, p=w / w.sum())
    ys, xs = np.divmod(idx, weight.shape[1])
    j = rng.random((n, 2))
    return np.stack([xs + j[:, 0], ys + j[:, 1]], axis=1).astype(np.float32)
