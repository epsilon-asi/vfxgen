"""``vfxgen export``: re-export an existing generation (trim, alpha mode, fps, size, packing)
without re-rendering."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import VfxError
from ..spec.loader import read_yaml
from .atlas import apply_alpha, build_sheet


def _resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    """High-quality resize in premultiplied space (no dark fringes)."""
    f = frame.astype(np.float32) / 255
    a = f[..., 3:4]
    prem = np.dstack([f[..., :3] * a, a])
    chans = [np.asarray(Image.fromarray(prem[..., c]).resize((w, h), Image.LANCZOS)) for c in range(4)]
    out = np.clip(np.dstack(chans), 0, 1)
    al = out[..., 3:4]
    rgb = np.where(al > 1e-6, out[..., :3] / np.maximum(al, 1e-6), 0)
    return (np.dstack([rgb, al]) * 255 + 0.5).clip(0, 255).astype(np.uint8)


def export(generation: str, *, out: str | None = None, trim: bool = False, alpha: str = "straight",
           fps: float | None = None, size: str | None = None, columns: int | None = None, pad: int = 0,
           pot: bool = False) -> Path:
    gdir = Path(generation).resolve()
    frames_dir = gdir / "frames"
    if not frames_dir.is_dir():
        raise VfxError(f"{gdir} has no frames/ folder (render with the 'frames' format)")
    ymls = list(gdir.glob("*.yaml"))
    meta = read_yaml(ymls[0].read_text()) if ymls else {}
    name = (meta or {}).get("name") or gdir.parent.name
    src_fps = float((meta or {}).get("meta", {}).get("fps") or (meta or {}).get("fps") or 24)
    loop = bool((meta or {}).get("loop"))
    frames = [np.asarray(Image.open(p).convert("RGBA")) for p in sorted(frames_dir.glob("*.png"))]
    out_fps = src_fps
    if fps and fps != src_fps:
        n_src = len(frames)
        dur = n_src / src_fps if loop else (n_src - 1) / src_fps
        n = max(1, int(round(dur * fps))) if loop else int(dur * fps + 1e-6) + 1
        frames = [frames[min(n_src - 1, int(round(i / fps * src_fps)))] for i in range(n)]
        out_fps = fps
    if size:
        w, _, h = size.lower().partition("x")
        frames = [_resize(f, int(w), int(h or w)) for f in frames]
    odir = Path(out) if out else gdir / "export"
    odir.mkdir(parents=True, exist_ok=True)
    sheet, atlas = build_sheet(frames, out_fps, loop, columns=columns, trim=trim, pad=pad, pot=pot, alpha=alpha)
    Image.fromarray(sheet, "RGBA").save(odir / f"{name}_sheet.png", optimize=True)
    atlas["image"] = f"{name}_sheet.png"
    (odir / f"{name}_sheet.json").write_text(json.dumps(atlas, indent=1))
    fd = odir / "frames"
    fd.mkdir(exist_ok=True)
    x0, y0 = atlas["trim_offset"]
    fw, fh = atlas["frame_size"]
    for i, f in enumerate(frames):
        Image.fromarray(apply_alpha(f[y0:y0 + fh, x0:x0 + fw], alpha), "RGBA").save(fd / f"{name}_{i:03d}.png")
    return odir
