"""Swirling flow: a divergence-free curl-noise velocity field.

Particles (or smoke puff centres) are advected through it by integrating from their spawn time to
the current time with a fixed step on every evaluation, so frames stay independent of each other
(time-pure) at a modest cost. Loops stay seamless when the field's time axis is periodic
(``period``) and particles respawn on whole laps.
"""
from __future__ import annotations

import numpy as np

from .noise import Noise


class CurlField:
    def __init__(self, rng: np.random.Generator, scale: float = 80.0, strength: float = 60.0,
                 speed: float = 0.5, period: float | None = None, octaves: int = 2):
        """scale: eddy size (px). strength: peak swirl speed (px/s). speed: how fast the field
        evolves (noise cells per second). period: loop length (s) for a seamless time axis."""
        self.noise = Noise(rng, period=(32, 32, 8))
        self.scale = float(scale)
        self.strength = float(strength)
        self.speed = float(speed)
        self.period = period
        self.octaves = octaves

    def _z(self, t):
        if self.period:
            return (np.asarray(t) / self.period) % 1.0 * self.noise.pz
        return np.asarray(t) * self.speed

    def velocity(self, x, y, t):
        """Curl of a noise potential: (dpsi/dy, -dpsi/dx) scaled to ``strength`` px/s."""
        e = 0.35
        X = np.asarray(x) / self.scale
        Y = np.asarray(y) / self.scale
        z = self._z(t)
        f = lambda a, b: self.noise.fbm(a, b, z, octaves=self.octaves)  # noqa: E731
        dpdy = (f(X, Y + e) - f(X, Y - e)) / (2 * e)
        dpdx = (f(X + e, Y) - f(X - e, Y)) / (2 * e)
        return dpdy * self.strength, -dpdx * self.strength

    def displacement(self, base_fn, t_spawn, age, dt: float = 1 / 60):
        """Integrated swirl offset for particles whose un-swirled position at age ``a`` is
        ``base_fn(a)`` (array (n, 2)); spawned at scene times ``t_spawn``; current ``age`` (n,)."""
        age = np.maximum(np.asarray(age, np.float64), 0)
        n = age.shape[0]
        off = np.zeros((n, 2))
        if n == 0:
            return off
        steps = int(np.ceil(age.max() / dt)) if age.max() > 0 else 0
        for k in range(steps):
            a = k * dt
            active = age > a
            if not active.any():
                break
            h = np.minimum(dt, age - a)
            p = base_fn(np.minimum(a, age)) + off
            vx, vy = self.velocity(p[:, 0], p[:, 1], np.asarray(t_spawn) + a)
            off[:, 0] += np.where(active, vx * h, 0)
            off[:, 1] += np.where(active, vy * h, 0)
        return off
