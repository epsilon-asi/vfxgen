"""Closed-form particle system.

Particles are defined by their spawn state and evaluated at any time without
stepping a simulation (drag + gravity in closed form, swirl as a rotation about
the origin, sway as a sine offset). That keeps every frame independent.

* :class:`Emitter` holds per-particle arrays drawn once from a seeded Generator.
* :meth:`Emitter.state` returns positions, velocities, normalised age and alive mask
  at a time ``t`` - either continuous looping emission (``loop=True``, ages wrap
  with the period) or a one-shot burst.
* :func:`draw_particles` deposits them as dots, streaks, glints, plus-signs or puffs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mathx import TAU, ballistic, smoothstep

SHAPES = ("point", "circle", "disk", "line", "box", "ellipse")
DIRECTIONS = ("angle", "radial", "inward", "tangent")


@dataclass
class EmitterConfig:
    count: int = 40
    shape: str = "point"
    radius: float = 0.0            # circle / disk / ellipse x radius; line half-length; box half-width
    radius_y: Optional[float] = None   # ellipse y radius / box half-height
    line_angle: float = 0.0
    direction: str = "angle"
    angle: float = -90.0           # degrees, clockwise from +x (-90 = up)
    spread: float = 360.0          # full cone width in degrees
    speed: tuple[float, float] = (60.0, 160.0)
    life: tuple[float, float] = (0.4, 0.8)
    gravity: float = 0.0           # px/s^2, +y down
    drag: float = 0.0
    swirl: float = 0.0             # degrees per second around the origin
    sway: float = 0.0              # px
    sway_freq: float = 1.0         # Hz (loops: per lap, whole numbers stay seamless)
    stagger: float = 0.0           # one-shot: spawn times spread over [0, stagger]
    loop: bool = False
    period: float = 1.0
    laps: tuple[int, int] = (1, 1)  # loop: lifetimes per period (whole numbers)
    curl: float = 0.0              # swirling flow strength (px/s); 0 = off
    curl_scale: float = 80.0       # eddy size (px)
    curl_speed: float = 0.5        # how fast the flow evolves (one-shots)


class Emitter:
    def __init__(self, rng: np.random.Generator, cfg: EmitterConfig):
        self.cfg = c = cfg
        n = c.count
        self.n = n
        u = rng.random(n)
        # spawn positions
        if c.shape == "point":
            px = np.zeros(n)
            py = np.zeros(n)
        elif c.shape in ("circle", "ellipse"):
            th = TAU * u
            ry = c.radius if c.radius_y is None else c.radius_y
            px, py = c.radius * np.cos(th), ry * np.sin(th)
        elif c.shape == "disk":
            th = TAU * rng.random(n)
            rr = c.radius * np.sqrt(u)
            px, py = rr * np.cos(th), rr * np.sin(th)
        elif c.shape == "line":
            s = (u * 2 - 1) * c.radius
            a = math.radians(c.line_angle)
            px, py = s * math.cos(a), s * math.sin(a)
        elif c.shape == "box":
            ry = c.radius if c.radius_y is None else c.radius_y
            px = (u * 2 - 1) * c.radius
            py = (rng.random(n) * 2 - 1) * ry
        else:
            raise ValueError(f"unknown emitter shape {c.shape!r}; use {', '.join(SHAPES)}")
        self.p0 = np.stack([px, py], axis=1)
        # directions
        base = np.radians(c.angle) + np.radians(c.spread) * (rng.random(n) - 0.5)
        if c.direction == "radial":
            ang = np.where(np.hypot(px, py) > 1e-6, np.arctan2(py, px), TAU * rng.random(n))
            ang = ang + np.radians(c.spread) * (rng.random(n) - 0.5) * (c.spread < 360)
        elif c.direction == "inward":
            ang = np.arctan2(-py, -px)
        elif c.direction == "tangent":
            ang = np.arctan2(py, px) + math.pi / 2
        elif c.direction == "angle":
            ang = base
        else:
            raise ValueError(f"unknown direction {c.direction!r}; use {', '.join(DIRECTIONS)}")
        spd = c.speed[0] + (c.speed[1] - c.speed[0]) * rng.random(n)
        self.v0 = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1)
        self.life = c.life[0] + (c.life[1] - c.life[0]) * rng.random(n)
        self.t0 = c.stagger * rng.random(n)
        self.phase = rng.random(n)
        self.laps = rng.integers(c.laps[0], c.laps[1] + 1, n) if c.laps[1] > c.laps[0] else np.full(n, c.laps[0])
        self.sway_ph = rng.random(n)
        self.rand = rng.random((n, 4))   # free per-particle randoms for styling (size, brightness, twinkle)
        self.flow = None
        if c.curl > 0:                   # drawn last so enabling flow never reshuffles the rest
            from .flow import CurlField
            self.flow = CurlField(rng, scale=c.curl_scale, strength=c.curl, speed=c.curl_speed,
                                  period=c.period if c.loop else None)

    @classmethod
    def from_points(cls, rng: np.random.Generator, cfg: EmitterConfig, points, velocities=None) -> "Emitter":
        """Emitter whose particles spawn at given ``points`` (n, 2) (e.g. sampled from a sprite's
        silhouette with ``assets.sample_points``); ``cfg.count`` is set to len(points). Optional
        ``velocities`` (n, 2) replace the configured directions/speeds."""
        pts = np.asarray(points, np.float64).reshape(-1, 2)
        cfg = EmitterConfig(**{**cfg.__dict__, "count": len(pts), "shape": "point"})
        em = cls(rng, cfg)
        em.p0 = pts.copy()
        if velocities is not None:
            em.v0 = np.asarray(velocities, np.float64).reshape(-1, 2).copy()
        return em

    def duration(self) -> float:
        return float(self.cfg.stagger + self.life.max()) if self.n else 0.0

    def age(self, t: float):
        """(age seconds, normalised age q in [0,1), alive mask)."""
        c = self.cfg
        if c.loop:
            lap_len = c.period / np.maximum(self.laps, 1)
            q = (t / c.period * self.laps + self.phase) % 1.0
            return q * lap_len, q, np.ones(self.n, bool)
        a = t - self.t0
        q = a / self.life
        return a, q, (a >= 0) & (q < 1)

    def positions(self, age: np.ndarray, q: np.ndarray) -> np.ndarray:
        c = self.cfg
        pos = ballistic(self.p0, self.v0, np.maximum(age, 0), c.drag, c.gravity)
        if c.swirl:
            a = np.radians(c.swirl) * age
            cs, sn = np.cos(a), np.sin(a)
            x, y = pos[:, 0].copy(), pos[:, 1].copy()
            pos[:, 0] = cs * x - sn * y
            pos[:, 1] = sn * x + cs * y
        if c.sway:
            f = c.sway_freq
            ph = (q * f if c.loop else age * f) + self.sway_ph
            pos[:, 0] += c.sway * np.sin(TAU * ph)
        return pos

    def state(self, t: float):
        age, q, alive = self.age(t)
        pos = self.positions(age, q)
        if self.flow is not None and alive.any():
            idx = np.flatnonzero(alive)
            a = age[idx]
            t_spawn = t - a
            base = lambda aa, idx=idx: self._positions_at(idx, aa)  # noqa: E731
            pos[idx] += self.flow.displacement(base, t_spawn, a)
        return pos, age, q, alive

    def _positions_at(self, idx, ages):
        """Un-swirled positions of particles ``idx`` at ages ``ages`` (no sway)."""
        c = self.cfg
        p = ballistic(self.p0[idx], self.v0[idx], np.maximum(ages, 0), c.drag, c.gravity)
        if c.swirl:
            ang = np.radians(c.swirl) * ages
            cs, sn = np.cos(ang), np.sin(ang)
            x, y = p[:, 0].copy(), p[:, 1].copy()
            p[:, 0] = cs * x - sn * y
            p[:, 1] = sn * x + cs * y
        return p


def life_alpha(q, fade_in: float = 0.1, fade_out: float = 1.5):
    """Opacity over normalised life: smooth fade in, power fade out."""
    q = np.clip(q, 0, 1)
    a = smoothstep(0, max(fade_in, 1e-6), q) if fade_in > 0 else np.ones_like(q)
    return a * (1 - q) ** fade_out


def draw_particles(view, em: Emitter, t: float, *, style: str = "dot", size: float = 1.2,
                   size_end: float | None = None, brightness: float = 1.0, brightness_var: float = 0.4,
                   tail: float = 0.05, fade_in: float = 0.1, fade_out: float = 1.5, twinkle: float = 0.0,
                   twinkle_rate: float = 4.0, halo: float = 0.0, extent: float | None = None,
                   bounds: tuple | None = None) -> None:
    """Deposit an emitter's particles into a view.

    style: dot | streak | glint | plus | puff. ``tail`` (s) is the streak length in time.
    ``extent``: fade particles out as they approach |x| or |y| = extent (keeps the atom inside
    its bounds)."""
    pos, age, q, alive = em.state(t)
    if not alive.any():
        return
    a = life_alpha(q, fade_in, fade_out) * brightness * (1 - brightness_var + brightness_var * 2 * em.rand[:, 0])
    if twinkle > 0:
        a = a * (1 - twinkle + twinkle * (0.5 + 0.5 * np.sin(TAU * (twinkle_rate * (q if em.cfg.loop else age)
                                                                  + em.rand[:, 1]))) ** 2)
    if extent is not None:
        d = np.maximum(np.abs(pos[:, 0]), np.abs(pos[:, 1]))
        a = a * smoothstep(extent, extent * 0.82, d)
    if bounds is not None:                 # fade near a rectangle (x0, y0, x1, y1)
        bx0, by0, bx1, by1 = bounds
        m = min(bx1 - bx0, by1 - by0) * 0.09 + 1e-6
        d = np.minimum.reduce([pos[:, 0] - bx0, bx1 - pos[:, 0], pos[:, 1] - by0, by1 - pos[:, 1]])
        a = a * smoothstep(0, m, d)
    sz = size * (0.7 + 0.6 * em.rand[:, 2])
    if size_end is not None:
        sz = sz * (1 + (size_end / max(size, 1e-6) - 1) * np.clip(q, 0, 1))
    tail_pos = None
    if style == "streak":
        ta = np.maximum(age - tail, 0)
        tq = ta / np.maximum(age, 1e-9) * q
        tail_pos = em.positions(ta, tq)
    for i in np.flatnonzero(alive & (a > 1e-3)):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        ai, si = float(a[i]), float(sz[i])
        if style == "dot":
            view.dot(x, y, si, ai)
        elif style == "puff":
            view.dot(x, y, si, ai)
        elif style == "streak":
            tx, ty = float(tail_pos[i, 0]), float(tail_pos[i, 1])
            mx, my = (x + tx) / 2, (y + ty) / 2
            view.polyline([(tx, ty), (mx, my), (x, y)], si * 0.45, ai * 0.15, r_end=si, a_end=ai)
        elif style in ("glint", "plus"):
            L = si * (5 if style == "glint" else 3.5)
            w = si * (0.45 if style == "glint" else 0.6)
            rot = float(em.rand[i, 3]) * math.pi if style == "glint" else 0.0
            for k in range(4):
                ang = rot + k * math.pi / 2
                view.polyline([(x, y), (x + L * math.cos(ang), y + L * math.sin(ang))], w, ai, r_end=w * 0.3,
                              a_end=0.0)
            view.dot(x, y, si * 0.9, ai)
        else:
            raise ValueError(f"unknown particle style {style!r}")
        if halo > 0:
            view.dot(x, y, si * 4, ai * halo)
