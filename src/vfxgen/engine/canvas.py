"""Channel buffers and the views atoms draw through.

A :class:`Target` is one channel's scene-space buffer at supersampled (SS)
resolution. A :class:`View` is an atom's window onto a target: the atom draws in
its own LOCAL coordinates and the view maps every scene pixel back into local
space (inverse affine), so translation, rotation, scale and flips are exact.

Lengths passed to primitives (radii, widths, positions) are in local units. The
anti-aliasing softness of hard-edged shapes (`soft`) is in output pixels so edges
stay crisp at any scale.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .color import ColorSpec, color_at
from .mathx import smoothstep
from .transform import Affine

MaskFn = Callable[[np.ndarray, np.ndarray], np.ndarray]   # scene x, y (output px) -> factor


class Target:
    """A channel's accumulation buffer in scene space.

    * light / smoke / volume: one float32 plane (`data`), additive.
    * solid: premultiplied RGB (`rgb`, 0..1) + alpha (`a`), painter's order.
    """

    def __init__(self, kind: str, width: int, height: int, ss: int, wrap: bool = False):
        self.kind = kind
        self.wrap = bool(wrap)          # tileable scene: drawing wraps around the edges
        self.w, self.h, self.ss = int(width), int(height), int(ss)
        shape = (self.h * self.ss, self.w * self.ss)
        if kind == "solid":
            self.rgb = np.zeros(shape + (3,), np.float32)
            self.a = np.zeros(shape, np.float32)
            self.data = self.a
        else:
            self.data = np.zeros(shape, np.float32)
        self.dirty = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    # -- region access (handles wrap-around for tileable scenes) ---------------------
    def _ix(self, ys: slice, xs: slice):
        H, W = self.data.shape
        return np.ix_(np.arange(ys.start, ys.stop) % H, np.arange(xs.start, xs.stop) % W)

    def get(self, arr: np.ndarray, ys: slice, xs: slice) -> np.ndarray:
        return arr[self._ix(ys, xs)] if self.wrap else arr[ys, xs]

    def set(self, arr: np.ndarray, ys: slice, xs: slice, val: np.ndarray) -> None:
        if self.wrap:
            arr[self._ix(ys, xs)] = val
        else:
            arr[ys, xs] = val

    def add(self, arr: np.ndarray, ys: slice, xs: slice, val: np.ndarray) -> None:
        if self.wrap:
            np.add.at(arr, self._ix(ys, xs), val)
        else:
            arr[ys, xs] += val

    def downsample(self, arr: Optional[np.ndarray] = None) -> np.ndarray:
        a = self.data if arr is None else arr
        s = self.ss
        if s == 1:
            return a.astype(np.float32, copy=True)
        if a.ndim == 3:
            return a.reshape(self.h, s, self.w, s, a.shape[2]).mean(axis=(1, 3))
        return a.reshape(self.h, s, self.w, s).mean(axis=(1, 3))


class View:
    """An atom instance's drawing surface for one output role."""

    def __init__(self, target: Target, xform: Affine, *, gain: float = 1.0,
                 mask: Optional[MaskFn] = None, palette: ColorSpec | None = None,
                 clip: Optional[tuple[float, float, float, float]] = None):
        self.target = target
        self.xf = xform
        self.inv = xform.inverse()
        self.gain = float(gain)
        self.mask = mask
        self.palette = palette
        self.ss = target.ss
        self.px = xform.linear_scale          # output px per local unit
        self.clip = clip                      # local-space bounds (x0, y0, x1, y1) or None

    # -- coordinate windows ---------------------------------------------------
    def window(self, lx0: float, ly0: float, lx1: float, ly1: float, step: int = 1):
        """Buffer slice covering the local box, plus LOCAL coordinates of its pixel centres.

        Returns ``(ys, xs, X, Y)`` or ``None`` when empty. ``step`` > 1 samples a coarser grid
        (used by low-res fields)."""
        if self.clip is not None:
            cx0, cy0, cx1, cy1 = self.clip
            lx0, ly0, lx1, ly1 = max(lx0, cx0), max(ly0, cy0), min(lx1, cx1), min(ly1, cy1)
            if lx0 >= lx1 or ly0 >= ly1:
                return None
        sx0, sy0, sx1, sy1 = self.xf.bbox(lx0, ly0, lx1, ly1)
        ss = self.ss
        H, W = self.target.shape
        if self.target.wrap:            # no clamping: indices wrap (limited to one tile)
            x0 = int(np.floor(sx0 * ss)) - 1
            y0 = int(np.floor(sy0 * ss)) - 1
            x1 = min(int(np.ceil(sx1 * ss)) + 2, x0 + W)
            y1 = min(int(np.ceil(sy1 * ss)) + 2, y0 + H)
        else:
            x0 = max(int(np.floor(sx0 * ss)) - 1, 0)
            y0 = max(int(np.floor(sy0 * ss)) - 1, 0)
            x1 = min(int(np.ceil(sx1 * ss)) + 2, W)
            y1 = min(int(np.ceil(sy1 * ss)) + 2, H)
        if x0 >= x1 or y0 >= y1:
            return None
        if step > 1:   # align to step grid so upsampling tiles exactly
            x0 -= x0 % step
            y0 -= y0 % step
        if self.target.wrap:           # never cover more than one tile (no double deposits)
            x1 = min(x1, x0 + W)
            y1 = min(y1, y0 + H)
        xs = (np.arange(x0, x1, step, dtype=np.float32) + 0.5 * step) / ss
        ys = (np.arange(y0, y1, step, dtype=np.float32) + 0.5 * step) / ss
        SX, SY = np.meshgrid(xs, ys)
        X, Y = self.inv.apply(SX, SY)
        return slice(y0, y1), slice(x0, x1), X.astype(np.float32), Y.astype(np.float32)

    def scene_coords(self, ys: slice, xs: slice):
        ss = self.ss
        sx = (np.arange(xs.start, xs.stop, dtype=np.float32) + 0.5) / ss
        sy = (np.arange(ys.start, ys.stop, dtype=np.float32) + 0.5) / ss
        return np.meshgrid(sx, sy)

    def _deposit(self, ys: slice, xs: slice, val: np.ndarray) -> None:
        if self.gain != 1.0:
            val = val * self.gain
        if self.mask is not None:
            SX, SY = self.scene_coords(ys, xs)
            val = val * self.mask(SX, SY)
        self.target.add(self.target.data, ys, xs, val)
        self.target.dirty = True

    def _soft(self, soft_px: float) -> float:
        return max(soft_px / max(self.px, 1e-6), 1e-6)

    # -- additive primitives (light / smoke / volume) ---------------------------
    def dot(self, x: float, y: float, r: float, a: float) -> None:
        """Gaussian blob, sigma r, peak a."""
        if a == 0 or r <= 0:
            return
        R = 3 * r
        w = self.window(x - R, y - R, x + R, y + R)
        if w is None:
            return
        ys, xs, X, Y = w
        dx, dy = X - x, Y - y
        self._deposit(ys, xs, a * np.exp(-(dx * dx + dy * dy) / (2 * r * r)))

    def ring(self, cx: float, cy: float, R: float, width: float, a: float, squash: float = 1.0,
             dashes: int = 0, duty: float = 0.5, phase: float = 0.0) -> None:
        """Gaussian-profile ring (ellipse when squash < 1). Optional n-fold dashes."""
        if a == 0 or width <= 0:
            return
        m = 3 * width
        sq = max(squash, 1e-3)
        w = self.window(cx - R - m, cy - R * sq - m, cx + R + m, cy + R * sq + m)
        if w is None:
            return
        ys, xs, X, Y = w
        dx = X - cx
        dy = (Y - cy) / sq
        d = np.hypot(dx, dy) - R
        val = a * np.exp(-(d * d) / (2 * width * width))
        if dashes > 0:
            ang = np.arctan2(dy, dx) / (2 * np.pi) * dashes + phase
            f = ang - np.floor(ang)
            val = val * smoothstep(0.0, 0.08, f) * smoothstep(duty, duty - 0.08, f)
        self._deposit(ys, xs, val)

    def polyline(self, pts, r: float, a: float, r_end: float | None = None,
                 a_end: float | None = None) -> None:
        """Gaussian-profile stroke tapering start -> end. Segments merge with max()."""
        pts = np.asarray(pts, np.float32)
        n = len(pts)
        if n < 2:
            return
        rs = np.linspace(r, r if r_end is None else r_end, n).astype(np.float32)
        As = np.linspace(a, a if a_end is None else a_end, n).astype(np.float32)
        if not np.any(As):
            return
        rs = np.maximum(rs, 1e-4)
        R = 3 * float(rs.max())
        w = self.window(float(pts[:, 0].min()) - R, float(pts[:, 1].min()) - R,
                        float(pts[:, 0].max()) + R, float(pts[:, 1].max()) + R)
        if w is None:
            return
        ys, xs, X, Y = w
        acc = np.zeros(X.shape, np.float32)
        ss = self.ss
        hh, ww = X.shape
        for i in range(n - 1):
            xa, ya = pts[i]
            xb, yb = pts[i + 1]
            Ri = 3 * max(rs[i], rs[i + 1])
            # segment's local box -> scene box -> index range inside this window
            bx0, by0, bx1, by1 = self.xf.bbox(min(xa, xb) - Ri, min(ya, yb) - Ri,
                                              max(xa, xb) + Ri, max(ya, yb) + Ri)
            c0 = max(int(np.floor(bx0 * ss)) - 1 - xs.start, 0)
            r0 = max(int(np.floor(by0 * ss)) - 1 - ys.start, 0)
            c1 = min(int(np.ceil(bx1 * ss)) + 2 - xs.start, ww)
            r1 = min(int(np.ceil(by1 * ss)) + 2 - ys.start, hh)
            if c0 >= c1 or r0 >= r1:
                continue
            px = X[r0:r1, c0:c1] - xa
            py = Y[r0:r1, c0:c1] - ya
            vx, vy = xb - xa, yb - ya
            t = np.clip((px * vx + py * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
            dx, dy = px - t * vx, py - t * vy
            rr = rs[i] + (rs[i + 1] - rs[i]) * t
            amp = As[i] + (As[i + 1] - As[i]) * t
            val = amp * np.exp(-(dx * dx + dy * dy) / (2 * rr * rr))
            sub = acc[r0:r1, c0:c1]
            np.maximum(sub, val, out=sub)
        self._deposit(ys, xs, acc)

    def polygon(self, pts, a: float, soft: float = 0.7) -> None:
        """Filled polygon (even-odd), edge softened by ``soft`` output px."""
        pts = np.asarray(pts, np.float32)
        s = self._soft(soft)
        m = 3 * s + 1e-3
        w = self.window(float(pts[:, 0].min()) - m, float(pts[:, 1].min()) - m,
                        float(pts[:, 0].max()) + m, float(pts[:, 1].max()) + m)
        if w is None:
            return
        ys, xs, X, Y = w
        sd = poly_sd(X, Y, pts)
        self._deposit(ys, xs, a * smoothstep(s, -s, sd))

    def capsule(self, p0, p1, r0: float, r1: float, a: float, soft: float = 0.6) -> None:
        """Hard-edged (anti-aliased) tapered capsule."""
        s = self._soft(soft)
        m = max(r0, r1) + 2 * s
        w = self.window(min(p0[0], p1[0]) - m, min(p0[1], p1[1]) - m,
                        max(p0[0], p1[0]) + m, max(p0[1], p1[1]) + m)
        if w is None:
            return
        ys, xs, X, Y = w
        self._deposit(ys, xs, a * smoothstep(s, -s, capsule_sd(X, Y, p0, p1, r0, r1)))

    def field(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
              bounds: Optional[Sequence[float]] = None, lowres: str | bool = False) -> None:
        """Add ``fn(X, Y)`` (local coordinates) over ``bounds`` (local box; default: atom clip).

        ``lowres``: evaluate at output-pixel resolution and upsample to SS
        (``"nearest"`` or ``"bilinear"``; ``True`` = nearest). Big speed-up for smooth noise
        fields; no effect at SS=1."""
        b = bounds if bounds is not None else self.clip
        if b is None:
            raise ValueError("field() needs bounds (or an atom clip)")
        step = self.ss if (lowres and self.ss > 1) else 1
        w = self.window(*b, step=step)
        if w is None:
            return
        ys, xs, X, Y = w
        val = np.asarray(fn(X, Y), np.float32)
        if step > 1:
            val = upsample(val, step, "bilinear" if lowres == "bilinear" else "nearest")
            H, W = self.target.shape
            hh = (ys.stop - ys.start) if self.target.wrap else min(val.shape[0], H - ys.start)
            ww = (xs.stop - xs.start) if self.target.wrap else min(val.shape[1], W - xs.start)
            val = val[:hh, :ww]
            ys = slice(ys.start, ys.start + hh)
            xs = slice(xs.start, xs.start + ww)
        self._deposit(ys, xs, val)

    def stamp(self, img, x: float, y: float, w: float | None = None, h: float | None = None,
              rot: float = 0.0, a: float = 1.0, anchor=(0.5, 0.5), channel: int | str = "alpha") -> None:
        """Deposit a greyscale image (e.g. a sprite's alpha, an edge mask, a texture's luminance)
        scaled by ``a``. ``img``: 2-D array, or RGBA premultiplied array with ``channel`` =
        "alpha" | "luma" | index. Size (w, h) in local units (default: image pixels)."""
        from .assets import bilinear
        arr = np.asarray(img, np.float32)
        if arr.ndim == 3:
            arr = arr[..., 3] if channel == "alpha" else (
                arr[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32) if channel == "luma" else arr[..., int(channel)])
        ih, iw = arr.shape
        w = float(iw if w is None else w)
        h = float(ih if h is None else h)
        bx0, by0, bx1, by1 = _image_box(x, y, w, h, rot, anchor)
        win = self.window(bx0, by0, bx1, by1)
        if win is None or a == 0:
            return
        ys, xs, X, Y = win
        px, py, _, _ = _image_uv(X, Y, x, y, w, h, rot, anchor, iw, ih)
        self._deposit(ys, xs, a * bilinear(arr, px, py))

    def private(self) -> "View":
        """A scratch view with its own empty buffer (same transform). Draw into it, post-process
        with :meth:`map`, then :meth:`add` it to this view. Used when an atom must
        modify only its OWN deposits (e.g. erode its dust with noise)."""
        t = Target(self.target.kind, self.target.w, self.target.h, self.ss, wrap=self.target.wrap)
        cls = SolidView if self.target.kind == "solid" else View
        return cls(t, self.xf, palette=self.palette, clip=self.clip)

    def multiply(self, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 bounds: Optional[Sequence[float]] = None, lowres: str | bool = False) -> None:
        """Multiply this view's buffer by ``fn(X, Y)`` (local coordinates) over bounds; e.g. erode
        a private view's density with noise. ``lowres`` as in :meth:`field`."""
        b = bounds if bounds is not None else self.clip
        step = self.ss if (lowres and self.ss > 1) else 1
        w = self.window(*b, step=step)
        if w is None:
            return
        ys, xs, X, Y = w
        val = np.asarray(fn(X, Y), np.float32)
        if step > 1:
            val = upsample(val, step, "bilinear" if lowres == "bilinear" else "nearest")
            H, W = self.target.shape
            hh = (ys.stop - ys.start) if self.target.wrap else min(val.shape[0], H - ys.start)
            ww = (xs.stop - xs.start) if self.target.wrap else min(val.shape[1], W - xs.start)
            val = val[:hh, :ww]
            ys, xs = slice(ys.start, ys.start + hh), slice(xs.start, xs.start + ww)
        tg = self.target
        tg.set(tg.data, ys, xs, tg.get(tg.data, ys, xs) * val)

    def map(self, fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
            bounds: Optional[Sequence[float]] = None) -> None:
        """Replace this view's buffer values ``v`` by ``fn(X, Y, v)`` over bounds (local), at full
        resolution. For a cheap multiplicative factor use :meth:`multiply` with ``lowres``."""
        b = bounds if bounds is not None else self.clip
        w = self.window(*b)
        if w is None:
            return
        ys, xs, X, Y = w
        tg = self.target
        tg.set(tg.data, ys, xs, np.asarray(fn(X, Y, tg.get(tg.data, ys, xs)), np.float32))

    def add(self, other: "View") -> None:
        """Deposit another (private) view's buffer into this one (gain and mask applied)."""
        d = other.target.data
        if not d.any():
            return
        rows = np.flatnonzero(d.any(axis=1))
        cols = np.flatnonzero(d.any(axis=0))
        ys = slice(int(rows[0]), int(rows[-1]) + 1)
        xs = slice(int(cols[0]), int(cols[-1]) + 1)
        self._deposit(ys, xs, d[ys, xs])


class SolidView(View):
    """Painter's-order drawing of opaque matter. Colours are given as a ramp position ``k``
    into the channel palette (facet lighting / material), or explicit ``rgb`` (0..255)."""

    def _color(self, k: float | None, rgb) -> np.ndarray:
        if rgb is None:
            rgb = color_at(self.palette or "rock", 0.5 if k is None else k)
        return np.asarray(rgb, np.float32) / 255.0

    def _paint(self, ys: slice, xs: slice, cov: np.ndarray, rgb: np.ndarray, a: float) -> None:
        k = cov * a * self.gain
        if self.mask is not None:
            SX, SY = self.scene_coords(ys, xs)
            k = k * self.mask(SX, SY)
        k = np.clip(k, 0, 1)
        t = self.target
        sub = t.get(t.rgb, ys, xs) * (1 - k)[..., None] + k[..., None] * rgb
        t.set(t.rgb, ys, xs, sub)
        t.set(t.a, ys, xs, t.get(t.a, ys, xs) * (1 - k) + k)
        t.dirty = True

    def polygon(self, pts, k: float | None = None, a: float = 1.0, soft: float = 0.6, rgb=None) -> None:  # type: ignore[override]
        pts = np.asarray(pts, np.float32)
        s = self._soft(soft)
        m = 2 * s + 1e-3
        w = self.window(float(pts[:, 0].min()) - m, float(pts[:, 1].min()) - m,
                        float(pts[:, 0].max()) + m, float(pts[:, 1].max()) + m)
        if w is None:
            return
        ys, xs, X, Y = w
        self._paint(ys, xs, smoothstep(s, -s, poly_sd(X, Y, pts)), self._color(k, rgb), a)

    def capsule(self, p0, p1, r0, r1, k: float | None = None, a: float = 1.0, soft: float = 0.6,  # type: ignore[override]
                rgb=None) -> None:
        s = self._soft(soft)
        m = max(r0, r1) + 2 * s
        w = self.window(min(p0[0], p1[0]) - m, min(p0[1], p1[1]) - m,
                        max(p0[0], p1[0]) + m, max(p0[1], p1[1]) + m)
        if w is None:
            return
        ys, xs, X, Y = w
        self._paint(ys, xs, smoothstep(s, -s, capsule_sd(X, Y, p0, p1, r0, r1)), self._color(k, rgb), a)

    def stroke(self, pts, r: float, k: float | None = None, a: float = 1.0, soft: float = 0.6,
               r_end: float | None = None, rgb=None) -> None:
        """Open polyline of capsules (cracks)."""
        pts = np.asarray(pts, np.float32)
        rs = np.linspace(r, r if r_end is None else r_end, len(pts))
        s = self._soft(soft)
        m = float(rs.max()) + 2 * s
        w = self.window(float(pts[:, 0].min()) - m, float(pts[:, 1].min()) - m,
                        float(pts[:, 0].max()) + m, float(pts[:, 1].max()) + m)
        if w is None:
            return
        ys, xs, X, Y = w
        sd = np.full(X.shape, np.inf, np.float32)
        for i in range(len(pts) - 1):
            np.minimum(sd, capsule_sd(X, Y, pts[i], pts[i + 1], rs[i], rs[i + 1]), out=sd)
        self._paint(ys, xs, smoothstep(s, -s, sd), self._color(k, rgb), a)

    def paint_field(self, fn, bounds=None, a: float = 1.0) -> None:
        """Paint with per-pixel colour: ``fn(X, Y) -> (coverage 0..1 (h, w), rgb 0..255 (h, w, 3))``
        over ``bounds`` (local box; default: atom clip). Gradients, embossing, textured fills."""
        b = bounds if bounds is not None else self.clip
        win = self.window(*b)
        if win is None:
            return
        ys, xs, X, Y = win
        cov, rgb = fn(X, Y)
        self._paint(ys, xs, np.asarray(cov, np.float32), np.asarray(rgb, np.float32) / 255.0, a)

    def add(self, other: "View", opacity: float = 1.0) -> None:  # type: ignore[override]
        """Composite a private solid view onto this one as a GROUP (one opacity for the whole
        drawing, so an outline never shows through its own fill when faded)."""
        src = other.target
        if not src.a.any():
            return
        rows = np.flatnonzero(src.a.any(axis=1))
        cols = np.flatnonzero(src.a.any(axis=0))
        ys = slice(int(rows[0]), int(rows[-1]) + 1)
        xs = slice(int(cols[0]), int(cols[-1]) + 1)
        k = src.a[ys, xs] * opacity * self.gain
        prem = src.rgb[ys, xs] * (opacity * self.gain)
        if self.mask is not None:
            SX, SY = self.scene_coords(ys, xs)
            m = self.mask(SX, SY)
            k = k * m
            prem = prem * m[..., None]
        t = self.target
        t.set(t.rgb, ys, xs, prem + t.get(t.rgb, ys, xs) * (1 - k)[..., None])
        t.set(t.a, ys, xs, k + t.get(t.a, ys, xs) * (1 - k))
        t.dirty = True

    def image(self, img, x: float, y: float, w: float | None = None, h: float | None = None,
              rot: float = 0.0, a: float = 1.0, anchor=(0.5, 0.5), tint=None, tint_amount: float = 0.0,
              flash: float = 0.0, mask=None, tint_mode: str = "mix") -> None:
        """Paint a premultiplied RGBA image (your sprite / texture) in painter's order.

        tint: (r, g, b) 0..255 mixed in by ``tint_amount`` (``tint_mode="mix"``: flat mix;
        ``"luma"``: keeps the image's shading, recolours its hue); flash: 0..1 mix toward white (hit
        flash); mask: optional fn(u, v) -> 0..1 in image UV (dissolves, wipes)."""
        from .assets import bilinear
        arr = np.asarray(img, np.float32)
        ih, iw = arr.shape[:2]
        w = float(iw if w is None else w)
        h = float(ih if h is None else h)
        bx0, by0, bx1, by1 = _image_box(x, y, w, h, rot, anchor)
        win = self.window(bx0, by0, bx1, by1)
        if win is None or a == 0:
            return
        ys, xs, X, Y = win
        px, py, u, v = _image_uv(X, Y, x, y, w, h, rot, anchor, iw, ih)
        smp = bilinear(arr, px, py)
        al = smp[..., 3] * a * self.gain
        if mask is not None:
            al = al * np.clip(mask(u, v), 0, 1)
        rgb = np.where(smp[..., 3:4] > 1e-6, smp[..., :3] / np.maximum(smp[..., 3:4], 1e-6), 0)
        if tint is not None and tint_amount > 0:
            tc = np.asarray(tint, np.float32) / 255.0
            if tint_mode == "luma":           # colourize: keep the image's light/dark shading
                lum = (rgb @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
                target = np.where(lum < 0.5, tc * (lum * 2), tc + (1 - tc) * (lum * 2 - 1))
            else:
                target = tc
            rgb = rgb + (target - rgb) * tint_amount
        if flash > 0:
            rgb = rgb + (1 - rgb) * flash
        if self.mask is not None:
            SX, SY = self.scene_coords(ys, xs)
            al = al * self.mask(SX, SY)
        al = np.clip(al, 0, 1)
        t = self.target
        t.set(t.rgb, ys, xs, t.get(t.rgb, ys, xs) * (1 - al)[..., None] + al[..., None] * rgb)
        t.set(t.a, ys, xs, t.get(t.a, ys, xs) * (1 - al) + al)
        t.dirty = True

    def dot(self, *a, **k):  # pragma: no cover - guard
        raise TypeError("solid channels take polygon / capsule / stroke, not light primitives")

    ring = polyline = field = dot


def _image_uv(X, Y, x, y, w, h, rot, anchor, iw, ih):
    """Local coords -> image pixel coords for an image of size (w, h) local units placed with its
    ``anchor`` (u, v) at (x, y) and rotated ``rot`` degrees clockwise about that point."""
    import math as _m
    r = _m.radians(rot)
    cs, sn = _m.cos(r), _m.sin(r)
    dx, dy = X - x, Y - y
    lx = cs * dx + sn * dy
    ly = -sn * dx + cs * dy
    u = lx / w + anchor[0]
    v = ly / h + anchor[1]
    return u * iw, v * ih, u, v


def _image_box(x, y, w, h, rot, anchor):
    import math as _m
    corners = []
    r = _m.radians(rot)
    cs, sn = _m.cos(r), _m.sin(r)
    for u in (0.0, 1.0):
        for v in (0.0, 1.0):
            lx, ly = (u - anchor[0]) * w, (v - anchor[1]) * h
            corners.append((x + cs * lx - sn * ly, y + sn * lx + cs * ly))
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


# --- signed distance helpers -----------------------------------------------------
def poly_sd(X: np.ndarray, Y: np.ndarray, pts) -> np.ndarray:
    """Signed distance to a polygon (negative inside, even-odd rule)."""
    pts = np.asarray(pts, np.float32)
    inside = np.zeros(X.shape, bool)
    dmin = np.full(X.shape, np.inf, np.float32)
    n = len(pts)
    for i in range(n):
        xa, ya = pts[i]
        xb, yb = pts[(i + 1) % n]
        cond = (ya > Y) != (yb > Y)
        xcross = xa + (Y - ya) * (xb - xa) / (yb - ya + 1e-12)
        inside ^= cond & (X < xcross)
        vx, vy = xb - xa, yb - ya
        t = np.clip(((X - xa) * vx + (Y - ya) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        np.minimum(dmin, np.hypot(X - xa - t * vx, Y - ya - t * vy), out=dmin)
    return np.where(inside, -dmin, dmin)


def capsule_sd(X, Y, p0, p1, r0, r1):
    """Signed distance to a tapered capsule (round ends)."""
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((X - p0[0]) * vx + (Y - p0[1]) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
    return np.hypot(X - p0[0] - t * vx, Y - p0[1] - t * vy) - (r0 + (r1 - r0) * t)


def upsample(a: np.ndarray, k: int, mode: str = "nearest") -> np.ndarray:
    if k == 1:
        return a
    if mode == "nearest":
        return np.repeat(np.repeat(a, k, axis=0), k, axis=1)
    # bilinear (cell-centred)
    h, w = a.shape
    ys = (np.arange(h * k) + 0.5) / k - 0.5
    xs = (np.arange(w * k) + 0.5) / k - 0.5
    y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    fy = np.clip(ys - y0, 0, 1)[:, None]
    fx = np.clip(xs - x0, 0, 1)[None, :]
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy) + bot * fy).astype(np.float32)
