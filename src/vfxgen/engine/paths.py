"""Paths for ``move``, ``repeat.along`` and ``spawn.at``.

Every path is converted to a dense polyline parameterised by arc length, so
``point(u)`` / ``tangent(u)`` with u in [0, 1] move at constant speed.
Angles are degrees, clockwise from +x (screen y points down).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from ..errors import SpecError
from ..spec.models import PathSpec

_DENSE = 720


def _bezier(ctrl: np.ndarray, n: int) -> np.ndarray:
    """Chained cubic Béziers when len = 3k+1, else one Bézier of any degree."""
    if len(ctrl) >= 4 and (len(ctrl) - 1) % 3 == 0:
        segs = [ctrl[i:i + 4] for i in range(0, len(ctrl) - 1, 3)]
    else:
        segs = [ctrl]
    out = []
    per = max(2, n // len(segs))
    for s in segs:
        ts = np.linspace(0, 1, per)[:, None]
        pts = np.repeat(s[None], len(ts), axis=0).astype(np.float64)
        k = len(s)
        for r in range(1, k):            # de Casteljau
            pts = pts[:, :-1] * (1 - ts[..., None]) + pts[:, 1:] * ts[..., None]
        out.append(pts[:, 0])
    return np.concatenate(out)


def _ellipse(center, rx, ry, a0, a1, rot, n):
    th = np.radians(np.linspace(a0, a1, n))
    x, y = rx * np.cos(th), ry * np.sin(th)
    r = math.radians(rot)
    cs, sn = math.cos(r), math.sin(r)
    return np.stack([center[0] + cs * x - sn * y, center[1] + sn * x + cs * y], axis=1)


def _radius(d: dict) -> tuple[float, float]:
    r = d.get("radius", 50)
    if isinstance(r, (list, tuple)):
        return float(r[0]), float(r[1])
    return float(r), float(r)


class Path:
    def __init__(self, spec: PathSpec, origin: tuple[float, float] = (0.0, 0.0)):
        self.spec = spec
        shape = spec.shape
        val = getattr(spec, shape)
        ox, oy = origin
        self.discrete: Optional[np.ndarray] = None
        closed = spec.closed
        if shape in ("line", "polyline", "points"):
            pts = np.asarray(val, np.float64)
            if len(pts) < 1 or (shape != "points" and len(pts) < 2):
                raise SpecError(f"path `{shape}` needs at least 2 points")
            if shape == "points":
                self.discrete = pts + (ox, oy)
                if len(pts) == 1:
                    pts = np.vstack([pts, pts])
            if closed:
                pts = np.vstack([pts, pts[:1]])
        elif shape == "bezier":
            pts = _bezier(np.asarray(val, np.float64), _DENSE)
        elif shape in ("arc", "circle", "ellipse"):
            c = val.get("center", [0, 0])
            rx, ry = _radius(val)
            if shape == "arc":
                a0, a1 = float(val.get("from", 0)), float(val.get("to", 180))
            else:
                a0 = float(val.get("from", 0))
                a1 = a0 + 360.0
                closed = True
            pts = _ellipse(c, rx, ry, a0, a1, float(val.get("rotate", 0)), _DENSE)
        elif shape == "spiral":
            c = val.get("center", [0, 0])
            r0, r1 = float(val.get("r0", 0)), float(val.get("r1", 100))
            turns = float(val.get("turns", 2))
            th = np.radians(float(val.get("from", 0)) + np.linspace(0, 360 * turns, _DENSE))
            rr = np.linspace(r0, r1, _DENSE)
            pts = np.stack([c[0] + rr * np.cos(th), c[1] + rr * np.sin(th)], axis=1)
        else:  # pragma: no cover
            raise SpecError(f"unknown path {shape}")
        self.closed = closed
        self.pts = pts + (ox, oy)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def point(self, u: float) -> tuple[float, float]:
        u = float(u) % 1.0 if self.closed else min(max(float(u), 0.0), 1.0)
        if self.length <= 0:
            return float(self.pts[0, 0]), float(self.pts[0, 1])
        s = u * self.length
        return float(np.interp(s, self.cum, self.pts[:, 0])), float(np.interp(s, self.cum, self.pts[:, 1]))

    def tangent(self, u: float) -> float:
        """Direction of travel at u, degrees clockwise from +x."""
        e = 1e-3
        a, b = self.point(max(u - e, 0) if not self.closed else u - e), self.point(min(u + e, 1) if not self.closed else u + e)
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    def sample(self, n: int, spacing: str = "even", rng: np.random.Generator | None = None):
        """n points along the path -> list of ((x, y), tangent_deg)."""
        if self.discrete is not None:
            out = []
            for i in range(n):
                p = self.discrete[i % len(self.discrete)]
                out.append(((float(p[0]), float(p[1])), 0.0))
            return out
        if spacing == "random":
            rng = rng or np.random.default_rng(0)
            us = np.sort(rng.random(n))
        elif n == 1:
            us = np.array([0.0])
        elif self.closed:
            us = np.arange(n) / n
        else:
            us = np.linspace(0, 1, n)
        return [(self.point(u), self.tangent(u)) for u in us]
