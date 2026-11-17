"""Smoke, gas and dust atoms: noise-eroded puff clouds, poison bubbles and thrown grains.

* ``smoke_puffs``  looping puff cloud (modes column / roll / burst / ring)
* ``smoke_burst``  one-shot puff burst (modes burst / ring) - same model, ``duration``
* ``bubbles``      poison bubbles that grow, then pop into a fading ring
* ``pebbles``      small grains thrown up that fall back to the ground (dust impacts)

The puff helpers (:class:`PuffTexture`, :func:`deposit_puff`) are shared with the fire
atoms (explosion cloud).
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Volume, atom
from ..engine.flow import CurlField
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise

__all__ = ["PuffTexture", "deposit_puff", "puff_eval"]


# =============================================================================
# shared puff helpers
# =============================================================================
class PuffTexture:
    """Tileable 2D fbm texture (unit variance) with bilinear wrap-around lookup. Sampled in
    each puff's own rotated/scaled frame, so billow detail travels and grows with the puff."""

    def __init__(self, rng: np.random.Generator, n: int = 256, cells: int = 8, octaves: int = 5,
                 gain: float = 0.5):
        nz = Noise(rng, period=(cells, cells, 2))
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32) * (cells / n)
        t = nz.fbm(xs, ys, 0.0, octaves=octaves, gain=gain)
        self.t = ((t - t.mean()) / (t.std() + 1e-6)).astype(np.float32)
        self.n = n

    def __call__(self, u, v):
        n = self.n
        u = np.asarray(u, np.float32) * n
        v = np.asarray(v, np.float32) * n
        i0, j0 = np.floor(u), np.floor(v)
        fu, fv = u - i0, v - j0
        i0 = i0.astype(np.int64) % n
        j0 = j0.astype(np.int64) % n
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        T = self.t
        a = T[j0, i0] * (1 - fu) + T[j0, i1] * fu
        b = T[j1, i0] * (1 - fu) + T[j1, i1] * fu
        return a * (1 - fv) + b * fv


def puff_eval(tex: PuffTexture, X, Y, x, y, r, ox=0.0, oy=0.0, rot=0.0, detail=0.6):
    """Gaussian body and unit-variance texture of one puff at local coords X, Y."""
    dx = (X - x) / r
    dy = (Y - y) / r
    cr, sr = math.cos(rot), math.sin(rot)
    u = cr * dx - sr * dy
    v = sr * dx + cr * dy
    body = np.exp(-(dx * dx + dy * dy) * 0.75)
    n = tex(u * detail * 0.25 + ox, v * detail * 0.25 + oy)
    return body, n


PUFF_REACH = 2.2    # puff support in radii (body * (1 + amp * n) < threshold beyond this)


def deposit_puff(view, tex: PuffTexture, x, y, r, a, ox=0.0, oy=0.0, rot=0.0, detail=0.6, amp=0.55,
                 thr=0.18, lowres="bilinear") -> None:
    """Paint one noise-eroded puff: ``a * max(body * (1 + amp * n) - thr, 0)``. Raise ``thr``
    to break the puff into wisps as it dissipates."""
    if a <= 1e-4 or r <= 0:
        return
    R = PUFF_REACH * r

    def f(X, Y):
        body, n = puff_eval(tex, X, Y, x, y, r, ox, oy, rot, detail)
        return a * np.clip(body * (1 + amp * n) - thr, 0, None)
    view.field(f, (x - R, y - R, x + R, y + R), lowres=lowres)


# =============================================================================
# puff clouds
# =============================================================================
_MODES = ["column", "roll", "burst", "ring"]


class _PuffParams(Atom.Params):
    mode: str = P("column", doc="motion: column (rise + sway), roll (well up and roll outward), "
                                "burst (thrown sideways with drag), ring (thrown radially with drag)",
                  choices=_MODES)
    count: int = P(34, ge=0, doc="number of puffs")
    laps: int = P(1, ge=1, doc="column/roll: lifetimes per period (whole number keeps loops seamless)")
    rise: float = P(252.0, doc="px risen over a puff's life (column: total column height)")
    spread: float = P(20.0, ge=0, doc="column: sideways spread added over life (px, sigma); roll: outward "
                                      "roll distance (px); burst/ring: launch speed (px/s)")
    spread_start: float = P(5.0, ge=0, doc="column: sideways spread at birth (px, sigma); roll: max start "
                                           "distance from the centre (px); burst: launch offset scaled by the "
                                           "puff's lateral speed (px); ring: spawn radius (px)")
    squash: float = P(1.0, gt=0, doc="roll/ring: vertical scale of the outward motion (ground ellipse < 1)")
    sway: float = P(14.0, ge=0, doc="column: sideways sway amplitude at the top (px)")
    drift: float = P(10.0, doc="column: sideways drift by the end of life, curving the column (px)")
    drag: float = P(4.5, ge=0, doc="burst/ring: linear drag (1/s)")
    lateral_min: float = P(0.03, ge=0, le=1, doc="burst: minimum lateral speed fraction (rest go up/stay)")
    forward: float = P(0.5, ge=0, le=1, doc="burst: fraction of puffs thrown to +x (0.5 = symmetric)")
    lift: float = P(0.0, ge=0, doc="burst: upward kick of the slow (non-lateral) puffs (px/s scale)")
    ground: bool = P(False, doc="burst/ring: lift each puff by half its radius so it sits on the ground line")
    spawn_y: float = P(0.0, ge=0, doc="random vertical spawn offset (px, sigma)")
    extent: float = P(0.0, ge=0, doc="burst/ring: puffs fade out by this distance from the origin (px; 0 = off)")
    radius_start: float = P(14.0, gt=0, doc="puff radius at birth (px)")
    radius_end: float = P(54.0, gt=0, doc="puff radius at end of life (px)")
    radius_var: float = P(0.2, ge=0, le=0.9, doc="random puff size variation (fraction)")
    life_min: float = P(0.7, gt=0, doc="burst/ring: minimum puff life (s)")
    life_max: float = P(1.0, gt=0, doc="burst/ring: maximum puff life (s)")
    stagger: float = P(0.0, ge=0, doc="burst/ring: spawn times spread over this many seconds")
    fade_in: float = P(0.12, gt=0, le=1, doc="fade-in fraction of life")
    fade_out: float = P(0.5, ge=0, lt=1, doc="life fraction where the fade-out starts")
    density: float = P(0.2, ge=0, doc="density deposited per puff")
    billow: float = P(0.85, ge=0, doc="noise amplitude of the billows")
    wisp_start: float = P(0.15, ge=0, doc="erosion threshold at birth")
    wispiness: float = P(0.45, ge=0, doc="extra erosion over life: puffs break into wisps")
    spin: float = P(1.0, ge=0, doc="random texture rotation over life (radians, sigma)")
    grain: int = P(8, ge=2, le=32, doc="billow texture cells (fewer = coarser billows)")
    output: str = P("smoke", doc="which output role to draw into", choices=["smoke", "volume"])
    curl: float = P(0.0, ge=0, doc="swirling flow strength (px/s): puffs are carried along curling eddies; "
                                   "0 = off (straight paths)")
    curl_scale: float = P(60.0, gt=0, doc="size of the swirling eddies (px)")


class _PuffCloud(Atom):
    outputs = {"smoke": Smoke(palette="smoke", density=5.0, shadow=1.0, relief=1.0, size_ref=384),
               "volume": Volume(palette="dust")}
    streams = ("puffs", "texture")
    anchors = {"origin": (0.0, 0.0)}

    def _span(self) -> float:
        return self.length()

    def setup(self):
        p = self.p
        rng = self.rng("puffs")
        n = p.count
        self.n = n
        L = self._span()
        self.R = dict(
            ph=(np.arange(n) + rng.uniform(0, 0.8, n)) / max(n, 1), ox=rng.random(n), oy=rng.random(n),
            rot=rng.uniform(0, TAU, n), spin=rng.normal(0, 1, n), side=rng.normal(0, 1, n),
            rr=1 + p.radius_var * (2 * rng.random(n) - 1), sw=rng.uniform(0, TAU, n),
            ang=rng.uniform(0, TAU, n), d0=rng.random(n), dy=np.clip(rng.normal(0, 1, n), -2.5, 2.5),
            lat=rng.uniform(p.lateral_min, 1.0, n) ** 0.8,
            sign=np.where(rng.random(n) < p.forward, 1.0, -1.0), up=rng.uniform(0.27, 1.0, n),
            t0=p.stagger * rng.random(n))
        lmin, lmax = min(p.life_min, p.life_max), max(p.life_min, p.life_max)
        life = rng.uniform(lmin, lmax, n)
        self.R["life"] = np.minimum(life, np.maximum(L - self.R["t0"], 1e-3)) if p.mode in ("burst", "ring") \
            else life
        # swirl: drawn last from the puffs stream so enabling it never reshuffles the rest
        self.flow = CurlField(rng, scale=p.curl_scale, strength=p.curl, speed=0.5,
                              period=self.length() if self.kind == "loop" else None) if p.curl > 0 else None
        self.tex = PuffTexture(self.rng("texture"), n=256, cells=p.grain, octaves=5)
        self._bounds = self._sample_bounds()

    # -- state ------------------------------------------------------------------
    def states(self, t: float):
        """Per-puff (x, y, r, alpha, threshold, tex ox, tex oy, rot) at local time t."""
        p, R = self.p, self.R
        if self.n == 0:
            z = np.zeros(0)
            return z, z, z, z, z, z, z, z
        if p.mode in ("column", "roll"):
            s = p.laps * t / p.period
            q = (R["ph"] + s) % 1.0
            if p.mode == "column":
                y = -p.rise * q ** 0.9
                x = R["side"] * (p.spread_start + p.spread * q) \
                    + p.sway * q * np.sin(TAU * (s % 1.0) + R["sw"] + 2.5 * q) + p.drift * q * q
            else:
                dist = R["d0"] * p.spread_start + p.spread * q
                x = np.cos(R["ang"]) * dist
                y = np.sin(R["ang"]) * dist * p.squash - p.rise * q
            r = (p.radius_start + (p.radius_end - p.radius_start) * q) * R["rr"]
            a = smoothstep(0.0, p.fade_in, q) * (1 - smoothstep(p.fade_out, 1.0, q))
            ox, oy = R["ox"] + 0.25 * q, R["oy"] - 0.35 * q
        else:
            tt = t - R["t0"]
            q = np.clip(tt / R["life"], 0, 1)
            alive = (tt > 0) & (tt < R["life"])
            tt = np.maximum(tt, 0)
            k = (1 - np.exp(-p.drag * tt)) / p.drag if p.drag > 1e-9 else tt
            lat = R["lat"]
            r = (p.radius_start + (p.radius_end - p.radius_start) * q ** 0.6 * (0.7 + 0.6 * lat)) * R["rr"]
            if p.mode == "burst":
                x = R["sign"] * lat * (p.spread_start + p.spread * k)
                y = -(1 - lat) ** 1.5 * R["up"] * p.lift * k * 2.2
                d = np.abs(x)
            else:
                dist = p.spread_start + p.spread * k
                x = np.cos(R["ang"]) * dist
                y = np.sin(R["ang"]) * dist * p.squash
                d = dist
            y = y - p.rise * q
            if p.ground:
                y = y - 0.55 * r
            a = smoothstep(0.0, p.fade_in, q) * (1 - smoothstep(p.fade_out, 1.0, q)) * alive
            if p.extent > 0:
                a = a * smoothstep(p.extent, 0.35 * p.extent, d)
            ox, oy = R["ox"] + 0.2 * q, R["oy"] - 0.2 * q
        y = y + R["dy"] * p.spawn_y
        if self.flow is not None:
            x, y = self._swirl(t, x, y, a)
        thr = p.wisp_start + p.wispiness * q ** 1.5
        rot = R["rot"] + p.spin * R["spin"] * q
        return x, y, r, a * p.density, thr, ox, oy, rot

    # -- swirl (curl flow) -----------------------------------------------------------
    def _ages(self, t: float) -> np.ndarray:
        p, R = self.p, self.R
        if p.mode in ("column", "roll"):
            lap = p.period / p.laps
            return ((R["ph"] + p.laps * t / p.period) % 1.0) * lap
        return np.clip(t - R["t0"], 0, R["life"])

    def _path(self, idx: np.ndarray, age: np.ndarray) -> np.ndarray:
        """Un-swirled puff centres (n, 2) of puffs ``idx`` at ages ``age`` (s), without sway."""
        p, R = self.p, self.R
        if p.mode in ("column", "roll"):
            q = np.clip(age / (p.period / p.laps), 0, 1)
            if p.mode == "column":
                x = R["side"][idx] * (p.spread_start + p.spread * q) + p.drift * q * q
                y = -p.rise * q ** 0.9
            else:
                dist = R["d0"][idx] * p.spread_start + p.spread * q
                x = np.cos(R["ang"][idx]) * dist
                y = np.sin(R["ang"][idx]) * dist * p.squash - p.rise * q
        else:
            q = np.clip(age / R["life"][idx], 0, 1)
            k = (1 - np.exp(-p.drag * age)) / p.drag if p.drag > 1e-9 else age
            lat = R["lat"][idx]
            if p.mode == "burst":
                x = R["sign"][idx] * lat * (p.spread_start + p.spread * k)
                y = -(1 - lat) ** 1.5 * R["up"][idx] * p.lift * k * 2.2
            else:
                dist = p.spread_start + p.spread * k
                x = np.cos(R["ang"][idx]) * dist
                y = np.sin(R["ang"][idx]) * dist * p.squash
            y = y - p.rise * q
        return np.stack([x, y], axis=1)

    def _swirl(self, t, x, y, a):
        """Add the curl-flow displacement integrated over each visible puff's age."""
        idx = np.flatnonzero(a > 1e-4)
        if idx.size == 0:
            return x, y
        age = self._ages(t)[idx]
        off = self.flow.displacement(lambda aa: self._path(idx, aa), t - age, age,
                                     dt=getattr(self, "_swirl_dt", 1 / 30))
        x = np.array(x, dtype=np.float64, copy=True)
        y = np.array(y, dtype=np.float64, copy=True)
        x[idx] += off[:, 0]
        y[idx] += off[:, 1]
        return x, y

    def _sample_bounds(self):
        if self.flow is None:
            return self._sample_box(96)
        # swirl: exact un-swirled box (cheap) united with a coarse swirled sampling (the curl is
        # integrated per sample, so use few samples and a coarse step), plus a safety pad
        flow, self.flow = self.flow, None
        b0 = self._sample_box(96)
        self.flow = flow
        self._swirl_dt = 1 / 8
        b1 = self._sample_box(12, pad=0.15 * self.p.curl_scale)
        self._swirl_dt = 1 / 30
        return (min(b0[0], b1[0]), min(b0[1], b1[1]), max(b0[2], b1[2]), max(b0[3], b1[3]))

    def _sample_box(self, ns: int, pad: float = 0.0):
        L = self._span()
        x0 = y0 = 1e9
        x1 = y1 = -1e9
        for k in range(ns + 1):
            x, y, r, a, *_ = self.states(L * k / ns)
            m = a > 1e-5
            if not m.any():
                continue
            R = PUFF_REACH * r[m] + pad
            x0, x1 = min(x0, float((x[m] - R).min())), max(x1, float((x[m] + R).max()))
            y0, y1 = min(y0, float((y[m] - R).min())), max(y1, float((y[m] + R).max()))
        if x0 > x1:
            return (-16.0, -16.0, 16.0, 16.0)
        return (x0 - 2, y0 - 2, x1 + 2, y1 + 2)

    def get_bounds(self):
        return self._bounds

    def draw(self, ctx, t):
        p = self.p
        if self.kind == "oneshot" and t >= p.duration:
            return
        view = getattr(ctx, p.output)
        x, y, r, a, thr, ox, oy, rot = self.states(t)
        for i in np.flatnonzero(a > 1e-4):
            deposit_puff(view, self.tex, float(x[i]), float(y[i]), float(r[i]), float(a[i]), float(ox[i]),
                         float(oy[i]), float(rot[i]), detail=0.6, amp=p.billow, thr=float(thr[i]))


@atom("smoke_puffs", tags=["smoke", "puffs", "loop"])
class SmokePuffs(_PuffCloud):
    """Looping cloud of noise-eroded puffs (smoke column, poison gas, mist, chimneys).

    Each puff is a soft ball eroded by a billow texture that travels and grows with it; it is
    born small, moves, swells and erodes into wisps. Modes: ``column`` (rise with sway and
    drift), ``roll`` (well up from the centre and roll outward), ``burst`` / ``ring`` (one
    burst per period, empty at both ends; use ``smoke_burst`` for one-shots). Draws density
    into ``smoke`` (relief-shaded by the channel) or ``volume``."""
    kind = "loop"

    class Params(_PuffParams):
        period: float = P(2.0, gt=0, doc="loop length (s)")


@atom("smoke_burst", tags=["smoke", "puffs", "oneshot", "dust"])
class SmokeBurst(_PuffCloud):
    """One-shot burst of noise-eroded puffs (dust puff, muzzle wisp, cold mist ring).

    Same puff model as ``smoke_puffs``. ``burst``: puffs are thrown along +-x with drag
    (``forward`` = share going +x), slow ones lift; ``ring``: thrown radially from a ring.
    Physics are in absolute seconds; puffs live ``life_min..life_max`` seconds."""
    kind = "oneshot"

    class Params(_PuffParams):
        duration: float = P(0.9, gt=0, doc="total length (s); puffs are cut at this time")
        mode: str = P("burst", doc="burst (thrown sideways with drag) or ring (thrown radially with drag)",
                      choices=["burst", "ring"])
        count: int = P(34, ge=0, doc="number of puffs")
        rise: float = P(10.0, doc="px risen over a puff's life")
        spread: float = P(400.0, ge=0, doc="launch speed (px/s)")
        spread_start: float = P(0.0, ge=0, doc="burst: launch offset scaled by lateral speed (px); "
                                               "ring: spawn radius (px)")
        lift: float = P(110.0, ge=0, doc="burst: upward kick of the slow (non-lateral) puffs (px/s scale)")
        ground: bool = P(True, doc="lift each puff by half its radius so it sits on the ground line")
        extent: float = P(108.0, ge=0, doc="puffs fade out by this distance from the origin (px; 0 = off)")
        radius_start: float = P(13.0, gt=0, doc="puff radius at birth (px)")
        radius_end: float = P(31.0, gt=0, doc="puff radius at end of life (px)")
        radius_var: float = P(0.23, ge=0, le=0.9, doc="random puff size variation (fraction)")
        life_min: float = P(0.62, gt=0, doc="minimum puff life (s)")
        life_max: float = P(0.78, gt=0, doc="maximum puff life (s)")
        stagger: float = P(0.04, ge=0, doc="spawn times spread over this many seconds")
        fade_in: float = P(0.03, gt=0, le=1, doc="fade-in fraction of life")
        fade_out: float = P(0.35, ge=0, lt=1, doc="life fraction where the fade-out starts")
        density: float = P(0.24, ge=0, doc="density deposited per puff")
        billow: float = P(0.8, ge=0, doc="noise amplitude of the billows")
        wispiness: float = P(0.5, ge=0, doc="extra erosion over life: puffs break into wisps")
        spin: float = P(1.5, ge=0, doc="random texture rotation over life (radians, sigma)")

    outputs = {"smoke": Smoke(palette="dust", density=6.0, shadow=0.9, ambient=0.25, relief=1.6,
                              height_gain=0.3, size_ref=256),
               "volume": Volume(palette="dust")}


# =============================================================================
# bubbles
# =============================================================================
@atom("bubbles", tags=["light", "bubbles", "loop", "poison"])
class Bubbles(Atom):
    """Rising bubbles that grow, then pop into a fading flash ring (poison, swamp, potion).

    Each bubble loops once per lap: a thin ring with a highlight grows and rises, then pops at
    ``pop_at`` of its life into an expanding ring that fades to nothing."""
    kind = "loop"
    outputs = {"light": Light(palette="toxic", bloom=[[1.5, 0.4], [5, 0.35]], alpha_gamma=0.9)}
    streams = ("bubbles",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        count: int = P(6, ge=0, doc="number of bubbles")
        laps: int = P(1, ge=1, doc="bubble lifetimes per period (whole number)")
        spread_x: float = P(50.0, ge=0, doc="horizontal scatter (px, sigma)")
        spread_y: float = P(14.0, ge=0, doc="vertical scatter (px, sigma)")
        rise: float = P(30.0, doc="px risen over a bubble's life")
        radius_min: float = P(4.0, gt=0, doc="minimum full bubble radius (px)")
        radius_max: float = P(9.0, gt=0, doc="maximum full bubble radius (px)")
        width: float = P(0.5, gt=0, doc="ring thickness (px)")
        brightness: float = P(0.32, ge=0, doc="bubble ring intensity")
        pop_brightness: float = P(0.45, ge=0, doc="pop ring intensity")
        pop_at: float = P(0.82, gt=0, lt=1, doc="life fraction where the bubble pops")

    def setup(self):
        p = self.p
        rng = self.rng("bubbles")
        n = p.count
        self.ph = rng.random(n)
        self.x = np.clip(rng.normal(0, 1, n), -2.5, 2.5) * p.spread_x
        self.y = np.clip(rng.normal(0, 1, n), -2.5, 2.5) * p.spread_y
        self.R = rng.uniform(min(p.radius_min, p.radius_max), max(p.radius_min, p.radius_max), n)

    def get_bounds(self):
        p = self.p
        m = max(p.radius_min, p.radius_max) * 2.2 + 3 * p.width + 2
        return (-2.5 * p.spread_x - m, -2.5 * p.spread_y - max(p.rise, 0) - m,
                2.5 * p.spread_x + m, 2.5 * p.spread_y + max(-p.rise, 0) + m)

    def draw(self, ctx, t):
        p = self.p
        s = p.laps * t / p.period
        for i in range(len(self.ph)):
            q = (self.ph[i] + s) % 1.0
            x, y = float(self.x[i]), float(self.y[i] - p.rise * q)
            if q < p.pop_at:
                R = float(self.R[i]) * (0.3 + 0.7 * q / p.pop_at)
                fi = float(smoothstep(0, 0.15, q))
                ctx.light.ring(x, y, R, p.width, p.brightness * fi)
                ctx.light.dot(x - R * 0.35, y - R * 0.35, 0.8, p.brightness * 1.1 * fi)
            else:
                k = (q - p.pop_at) / (1 - p.pop_at)
                ctx.light.ring(x, y, float(self.R[i]) * (1 + 1.2 * k), p.width, p.pop_brightness * (1 - k) ** 2)


# =============================================================================
# pebbles
# =============================================================================
@atom("pebbles", tags=["dust", "particles", "oneshot"])
class Pebbles(Atom):
    """Small grains thrown up from an impact that fall back and vanish when they land.

    Drawn into the same density as the dust (``smoke`` or ``volume`` role) so they share its
    shading. Origin = impact point on the ground line (grains disappear below y = 0)."""
    kind = "oneshot"
    outputs = {"smoke": Smoke(palette="dust", density=6.0, shadow=0.9, ambient=0.25, relief=1.6,
                              height_gain=0.3, size_ref=256),
               "volume": Volume(palette="dust")}
    streams = ("pebbles",)

    class Params(Atom.Params):
        duration: float = P(0.9, gt=0, doc="total length (s)")
        count: int = P(14, ge=0, doc="number of grains")
        speed_x: float = P(110.0, ge=0, doc="sideways speed (px/s, sigma)")
        speed_min: float = P(120.0, ge=0, doc="minimum upward speed (px/s)")
        speed_max: float = P(420.0, ge=0, doc="maximum upward speed (px/s)")
        gravity: float = P(1500.0, gt=0, doc="gravity (px/s², down)")
        start_y: float = P(-4.0, doc="launch height relative to the ground line (px, negative = above)")
        size_min: float = P(0.7, gt=0, doc="minimum grain radius (px)")
        size_max: float = P(1.5, gt=0, doc="maximum grain radius (px)")
        brightness: float = P(3.0, ge=0, doc="density per grain")
        visible: float = P(0.63, gt=0, doc="grains fade out linearly and are gone by this time (s)")
        output: str = P("smoke", doc="which output role to draw into", choices=["smoke", "volume"])

    def setup(self):
        p = self.p
        rng = self.rng("pebbles")
        n = p.count
        self.vx = np.clip(rng.normal(0, 1, n), -2.2, 2.2) * p.speed_x
        self.vy = -rng.uniform(min(p.speed_min, p.speed_max), max(p.speed_min, p.speed_max), n)
        self.r = rng.uniform(min(p.size_min, p.size_max), max(p.size_min, p.size_max), n)

    def get_bounds(self):
        p = self.p
        tl = min(p.visible, p.duration)
        top = max(p.speed_min, p.speed_max) ** 2 / (2 * p.gravity) - p.start_y
        xm = (float(np.abs(self.vx).max()) if len(self.vx) else 0.0) * tl
        m = 3 * max(p.size_min, p.size_max) + 2
        return (-xm - m, -top - m, xm + m, max(0.0, p.start_y) + m)

    def draw(self, ctx, t):
        p = self.p
        tl = min(p.visible, p.duration)
        if t >= tl:
            return
        view = getattr(ctx, p.output)
        x = self.vx * t
        y = p.start_y + self.vy * t + 0.5 * p.gravity * t * t
        a = p.brightness * (1 - t / tl)
        for i in np.flatnonzero(y <= 0):
            view.dot(float(x[i]), float(y[i]), float(self.r[i]), a)
