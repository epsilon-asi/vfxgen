"""On-disk render cache.

Key = hash of (fully explicit scene, draft flag, vfxgen version, source of every atom module).
An identical re-render is a disk read; editing the scene, any var, any seed or any atom's code
changes the key. Entries are compressed .npz files in ``$VFXGEN_CACHE`` (default
``~/.cache/vfxgen``), evicted least-recently-used beyond ``$VFXGEN_CACHE_MB`` (default 2048).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_code_hash: Optional[str] = None


def cache_dir() -> Path:
    d = Path(os.environ.get("VFXGEN_CACHE", "~/.cache/vfxgen")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def code_hash() -> str:
    """Hash of every loaded vfxgen / user-atom source file (invalidates on code edits)."""
    global _code_hash
    if _code_hash is not None:
        return _code_hash
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent.parent
    files = sorted(pkg.rglob("*.py"))
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and getattr(m, "__name__", "").startswith("vfxgen_user_atoms"):
            files.append(Path(f))
    for f in files:
        try:
            h.update(f.read_bytes())
        except OSError:
            pass
    _code_hash = h.hexdigest()[:16]
    return _code_hash


def key(explicit: dict, draft: bool, extra: dict | None = None) -> str:
    """``extra``: render options and asset hashes that also change the pixels."""
    from .. import __version__
    body = {k: v for k, v in explicit.items() if k != "meta"}
    s = json.dumps(body, sort_keys=True, default=str)
    x = json.dumps(extra or {}, sort_keys=True, default=str)
    return hashlib.sha256(f"{__version__}|{code_hash()}|{int(draft)}|{s}|{x}".encode()).hexdigest()[:32]


def get(k: str) -> Optional[list[np.ndarray]]:
    p = cache_dir() / f"{k}.npz"
    if not p.is_file():
        return None
    try:
        with np.load(p) as z:
            frames = [z[f"f{i}"] for i in range(int(z["n"]))]
        os.utime(p)
        return frames
    except Exception:
        p.unlink(missing_ok=True)
        return None


def put(k: str, frames: list[np.ndarray]) -> None:
    d = cache_dir()
    tmp = d / f".{k}.tmp.npz"
    np.savez_compressed(tmp, n=np.array(len(frames)), **{f"f{i}": f for i, f in enumerate(frames)})
    tmp.replace(d / f"{k}.npz")
    _evict()


def _evict() -> None:
    limit = float(os.environ.get("VFXGEN_CACHE_MB", "2048")) * 1024 * 1024
    files = sorted(cache_dir().glob("*.npz"), key=lambda p: p.stat().st_mtime)
    total = sum(p.stat().st_size for p in files)
    while files and total > limit:
        p = files.pop(0)
        total -= p.stat().st_size
        p.unlink(missing_ok=True)


def clear() -> int:
    n = 0
    for p in cache_dir().glob("*.npz"):
        p.unlink(missing_ok=True)
        n += 1
    return n


def info() -> dict:
    files = list(cache_dir().glob("*.npz"))
    return {"dir": str(cache_dir()), "entries": len(files),
            "size_mb": round(sum(p.stat().st_size for p in files) / 1e6, 1)}
