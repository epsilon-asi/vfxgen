"""Periodic 3D gradient (Perlin) noise.

Seamless loops: sample ``z = t_norm * period_z`` with t_norm in [0, 1). Every
fbm octave uses an integer lacunarity, so it is periodic too and frame N wraps
to frame 0 exactly.
"""
from __future__ import annotations

import numpy as np


class Noise:
    def __init__(self, rng: np.random.Generator, period: tuple[int, int, int] = (32, 32, 8)):
        self.px, self.py, self.pz = (int(v) for v in period)
        g = rng.normal(size=(self.pz, self.py, self.px, 3)).astype(np.float32)
        self.g = g / np.linalg.norm(g, axis=-1, keepdims=True)

    def __call__(self, x, y, z=0.0):
        """Single octave, roughly in [-1, 1]. x, y, z broadcastable arrays."""
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        x, y = np.broadcast_arrays(x, y)
        z = np.broadcast_to(np.asarray(z, np.float32), x.shape)
        xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
        fx, fy, fz = x - xi, y - yi, z - zi
        xi = xi.astype(np.int64)
        yi = yi.astype(np.int64)
        zi = zi.astype(np.int64)

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v, w = fade(fx), fade(fy), fade(fz)
        res = 0.0
        for dz in (0, 1):
            wz = w if dz else 1 - w
            for dy in (0, 1):
                wy = v if dy else 1 - v
                for dx in (0, 1):
                    wx = u if dx else 1 - u
                    gi = self.g[(zi + dz) % self.pz, (yi + dy) % self.py, (xi + dx) % self.px]
                    dot = gi[..., 0] * (fx - dx) + gi[..., 1] * (fy - dy) + gi[..., 2] * (fz - dz)
                    res = res + wx * wy * wz * dot
        return (res * 1.6).astype(np.float32)

    def fbm(self, x, y, z=0.0, octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5):
        """Fractal sum, normalised to roughly [-1, 1]. Keep lacunarity an integer
        when the time axis (z) must loop."""
        total, amp, norm, f = 0.0, 1.0, 0.0, 1.0
        for o in range(octaves):
            total = total + amp * self(np.asarray(x) * f + o * 17.3, np.asarray(y) * f + o * 9.1,
                                       np.asarray(z) * f)
            norm += amp
            amp *= gain
            f *= lacunarity
        return (total / norm).astype(np.float32)
