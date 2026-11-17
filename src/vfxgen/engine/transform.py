"""2D affine transforms (local -> parent coordinates, y down, degrees clockwise)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Affine:
    """Maps (x, y) -> (a*x + b*y + c, d*x + e*y + f)."""

    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    e: float = 1.0
    f: float = 0.0

    # -- constructors ---------------------------------------------------------
    @staticmethod
    def identity() -> "Affine":
        return Affine()

    @staticmethod
    def translate(tx: float, ty: float) -> "Affine":
        return Affine(1, 0, float(tx), 0, 1, float(ty))

    @staticmethod
    def rotate(deg: float) -> "Affine":
        """Clockwise on screen (y points down)."""
        r = math.radians(deg)
        cs, sn = math.cos(r), math.sin(r)
        return Affine(cs, -sn, 0, sn, cs, 0)

    @staticmethod
    def scale(sx: float, sy: float | None = None) -> "Affine":
        return Affine(float(sx), 0, 0, 0, float(sx if sy is None else sy), 0)

    # -- algebra --------------------------------------------------------------
    def __matmul__(self, o: "Affine") -> "Affine":
        """(self @ o)(p) == self(o(p))."""
        return Affine(
            self.a * o.a + self.b * o.d, self.a * o.b + self.b * o.e, self.a * o.c + self.b * o.f + self.c,
            self.d * o.a + self.e * o.d, self.d * o.b + self.e * o.e, self.d * o.c + self.e * o.f + self.f,
        )

    def inverse(self) -> "Affine":
        det = self.a * self.e - self.b * self.d
        if abs(det) < 1e-12:
            raise ZeroDivisionError("singular transform (scale 0?)")
        ia, ib = self.e / det, -self.b / det
        id_, ie = -self.d / det, self.a / det
        return Affine(ia, ib, -(ia * self.c + ib * self.f), id_, ie, -(id_ * self.c + ie * self.f))

    # -- application ----------------------------------------------------------
    def apply(self, x, y):
        return self.a * x + self.b * y + self.c, self.d * x + self.e * y + self.f

    def apply_pts(self, pts) -> np.ndarray:
        p = np.asarray(pts, np.float64)
        x, y = self.apply(p[..., 0], p[..., 1])
        return np.stack([x, y], axis=-1)

    def apply_vec(self, x, y):
        """Apply the linear part only (directions)."""
        return self.a * x + self.b * y, self.d * x + self.e * y

    @property
    def det(self) -> float:
        return self.a * self.e - self.b * self.d

    @property
    def linear_scale(self) -> float:
        """Geometric-mean scale factor (scene px per local unit)."""
        return math.sqrt(abs(self.det))

    @property
    def angle(self) -> float:
        """Rotation of the local x axis, degrees clockwise."""
        return math.degrees(math.atan2(self.d, self.a))

    def bbox(self, x0: float, y0: float, x1: float, y1: float) -> tuple[float, float, float, float]:
        """Axis-aligned bounds of the transformed local box."""
        xs, ys = self.apply(np.array([x0, x1, x0, x1], np.float64), np.array([y0, y0, y1, y1], np.float64))
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
