"""Sprite sheets, atlases, trimming and alpha modes."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def union_bbox(frames: list[np.ndarray]) -> Optional[tuple[int, int, int, int]]:
    """(x0, y0, x1, y1) covering every non-transparent pixel of every frame, or None."""
    any_a = np.zeros(frames[0].shape[:2], bool)
    for f in frames:
        any_a |= f[..., 3] > 0
    if not any_a.any():
        return None
    rows = np.flatnonzero(any_a.any(axis=1))
    cols = np.flatnonzero(any_a.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def apply_alpha(frame: np.ndarray, mode: str) -> np.ndarray:
    """straight (default) | premultiplied | additive (glow on black, opaque)."""
    if mode == "straight":
        return frame
    f = frame.astype(np.float32)
    a = f[..., 3:4] / 255.0
    if mode == "premultiplied":
        return np.dstack([np.clip(f[..., :3] * a + 0.5, 0, 255), f[..., 3:4]]).astype(np.uint8)
    if mode == "additive":
        rgb = np.clip(f[..., :3] * a + 0.5, 0, 255)
        return np.dstack([rgb, np.full(a.shape, 255.0)]).astype(np.uint8)
    raise ValueError(f"unknown alpha mode {mode!r}; use straight, premultiplied or additive")


def build_sheet(frames: list[np.ndarray], fps: float, loop: bool, *, columns: int | None = None,
                trim: bool = False, pad: int = 0, pot: bool = False, alpha: str = "straight",
                times: list[float] | None = None, events: list | None = None, tracks: dict | None = None,
                pivot: tuple | None = None, blend: str = "alpha"):
    """Pack frames into a grid -> (sheet RGBA uint8, atlas dict)."""
    h0, w0 = frames[0].shape[:2]
    box = union_bbox(frames) if trim else None
    if trim and box is None:
        box = (0, 0, 1, 1)
    if box is None:
        box = (0, 0, w0, h0)
    x0, y0, x1, y1 = box
    fw, fh = x1 - x0, y1 - y0
    n = len(frames)
    cols = columns or min(n, 8)
    rows = math.ceil(n / cols)
    W = cols * fw + (cols + 1) * pad
    H = rows * fh + (rows + 1) * pad
    if pot:
        W, H = 1 << (W - 1).bit_length(), 1 << (H - 1).bit_length()
    sheet = np.zeros((H, W, 4), np.uint8)
    rects = []
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        x = pad + c * (fw + pad)
        y = pad + r * (fh + pad)
        sheet[y:y + fh, x:x + fw] = apply_alpha(f[y0:y1, x0:x1], alpha)
        rects.append({"index": i, "x": x, "y": y, "w": fw, "h": fh,
                      "t": round(times[i], 6) if times else round(i / fps, 6)})
    atlas = {
        "frames": rects, "frame_count": n, "fps": fps, "loop": loop, "columns": cols, "rows": rows,
        "frame_size": [fw, fh], "source_size": [w0, h0],
        "trim_offset": [x0, y0],                       # where the trimmed frame sits in the source canvas
        "pivot": [((pivot or (0.5, 0.5))[0] * w0 - x0) / fw, ((pivot or (0.5, 0.5))[1] * h0 - y0) / fh],
        "alpha": alpha, "blend": blend, "sheet_size": [W, H],
        "events": list(events or []),
        "tracks": {k: [None if p is None else [p[0] - x0, p[1] - y0] for p in v] for k, v in (tracks or {}).items()},
    }
    return sheet, atlas
