"""Environment / weather atoms: rain, snow, fireflies, falling leaves, god rays, dust motes and a
waterfall.

* ``rainfall``        rain streaks over an area, splashes on a ground line (light)
* ``snowfall``        soft flakes drifting down with sway and swirling (curl) flow (light)
* ``firefly_swarm``   glowing motes wandering on a curl flow and blinking (light)
* ``leaf_fall``       tumbling leaves: flat-shaded polygons that sway, spin and flip (solid)
* ``god_rays``        angled light shafts with noise flicker and dust drifting in the beams (light)
* ``floating_motes``  slow floating specks that catch the light (light)
* ``water_cascade``   falling water sheet with foam at the base (volume + smoke + light)

All are loops. Things that fall or wander use **per-lap respawn**: every particle lives a whole
number of laps per period, is re-placed at random each lap (a table indexed by lap number modulo
its lap count) and is invisible at its lap boundary (it has left the area or faded out). So the
motion itself (speed, sway, spin, swirl) needs no whole-number tuning and every loop is seamless.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Solid, Volume, atom
from ..engine.color import register_palette
from ..engine.flow import CurlField
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise

__all__ = ["lap_clock", "pick", "soft_box", "lap_spawn_y"]

register_palette("rain", "#4e6276,#93a9be,#d6e3ee,#ffffff")
register_palette("snow", "#8a9cb8,#d4deec,#f6f9fd,#ffffff")
register_palette("firefly", "#2f5c0c,#8fd832,#eeff96,#ffffff")
register_palette("autumn", "#4a1606,#9c3412,#d4691e,#eab040")
register_palette("sunbeam", "#6e5226,#d8b67e,#fff0cc,#ffffff")
register_palette("mote", "#6e5a3a,#d6c29a,#fff6e0,#ffffff")


# =============================================================================
# shared helpers
# =============================================================================
def lap_clock(t: float, period: float, laps: np.ndarray, phase: np.ndarray):
    """Per-particle lap index ``j`` (0..laps-1), lap progress ``q`` (0..1) and age (s)."""
    s = t / period * laps + phase
    fl = np.floor(s)
    q = s - fl
    j = fl.astype(np.int64) % laps
    return j, q, q * period / laps


def pick(table: np.ndarray, j: np.ndarray) -> np.ndarray:
    """``table[i, j[i], ...]`` for per-particle, per-lap random tables of shape (n, laps_max, ...)."""
    return table[np.arange(table.shape[0]), j]


def _ss(e0, e1, x):
    if np.isscalar(e0) and np.isscalar(e1) and abs(e1 - e0) < 1e-9:
        return (np.asarray(x) >= e1).astype(np.float32)
    e0 = np.asarray(e0, np.float32)
    e1 = np.asarray(e1, np.float32)
    d = np.where(np.abs(e1 - e0) < 1e-6, 1e-6, e1 - e0)
    k = np.clip((np.asarray(x, np.float32) - e0) / d, 0, 1)
    return k * k * (3 - 2 * k)


def soft_box(x, y, x0, y0, x1, y1, edge: float):
    """1 inside the box, fading to 0 at its edges over ``edge`` px."""
    return _ss(x0, x0 + edge, x) * _ss(x1, x1 - edge, x) * _ss(y0, y0 + edge, y) * _ss(y1, y1 - edge, y)


def lap_spawn_y(travel_lap: np.ndarray, span: float, u: np.ndarray) -> np.ndarray:
    """Start offset (relative to the top of the travel span) for a particle that falls
    ``travel_lap`` px per lap through a span of ``span`` px. Long laps enter from above after a
    random delay; short laps start somewhere inside the span (they fade in / out)."""
    extra = travel_lap - span
    return np.where(extra >= 0, -u * extra, u * (-extra))


def _flow(rng, curl, scale, period):
    return CurlField(rng, scale=scale, strength=curl, period=period) if curl > 0 else None


def _curl_offsets(flow, base_fn, t, age, dt=1 / 20):
    """Swirl displacement for particles with un-swirled path ``base_fn(ages)``."""
    if flow is None or len(age) == 0:
        return np.zeros((len(age), 2))
    return flow.displacement(base_fn, t - age, age, dt=dt)


# =============================================================================
# rain
# =============================================================================
@atom("rainfall", tags=["environment", "weather", "rain", "loop"])
class Rainfall(Atom):
    """Rain: slanted streaks falling over an area, splashing on a ground line.

    Local origin = centre of the ground line; drops fall from ``y = -height`` and land in a band
    ``ground_depth`` px deep above ``y = 0`` (far drops land higher, are slower, thinner, dimmer).
    Each drop lands, plays a splash (squashed ring, a hot impact dot and a few droplets) and
    respawns at a new x on its next lap. Streaks fade out at the left / right edges."""
    kind = "loop"
    outputs = {"light": Light(palette="rain", bloom=[[1.0, 0.3], [3.0, 0.18]], exposure=1.0, alpha_gamma=0.9)}
    streams = ("drops", "splashes")
    anchors = {"ground": (0.0, 0.0)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        width: float = P(240.0, gt=0, doc="width of the rain area (px)")
        height: float = P(240.0, gt=0, doc="fall height from the top of the area to the ground line (px)")
        count: int = P(80, ge=0, doc="number of drops (each falls one or more times per period)")
        speed: float = P(850.0, gt=0, doc="fall speed of the nearest drops (px/s)")
        speed_var: float = P(0.25, ge=0, le=1, doc="random speed variation (fraction)")
        depth: float = P(0.6, ge=0, le=1, doc="depth spread: far drops are slower, thinner and dimmer")
        wind: float = P(0.15, ge=-2, le=2, doc="slant: sideways px per px of fall (+ = blows to +x)")
        length: float = P(0.03, ge=0, doc="streak length (s of travel)")
        thickness: float = P(0.6, gt=0, doc="streak thickness of the nearest drops (px)")
        brightness: float = P(0.9, ge=0, doc="streak intensity")
        ground_depth: float = P(24.0, ge=0, doc="depth of the landing band above the ground line (px)")
        splashes: float = P(1.0, ge=0, doc="splash intensity (0 = no splashes)")
        splash_size: float = P(6.0, gt=0, doc="splash ring radius of the nearest drops (px)")
        splash_time: float = P(0.2, gt=0, doc="splash duration (s)")
        droplets: int = P(3, ge=0, le=8, doc="droplets thrown up by each splash")
        edge: float = P(8.0, ge=0, doc="soft fade at the left / right / top edges (px)")

    def setup(self):
        p = self.p
        rng = self.rng("drops")
        n = p.count
        d = rng.random(n)
        self.d = d
        spd = p.speed * (1 - 0.55 * p.depth * (1 - d)) * (1 + p.speed_var * (rng.random(n) - 0.5))
        self.spd = np.maximum(spd, 1.0)
        self.yg = -p.ground_depth * (1 - d)
        self.tl = (p.height + self.yg) / self.spd                       # landing age
        need = self.tl + p.length + (p.splash_time if p.splashes > 0 else 0.0)
        self.laps = np.maximum(1, np.floor(p.period / np.maximum(need, 1e-6))).astype(np.int64)
        L = int(self.laps.max()) if n else 1
        drift = p.wind * p.height
        lo, hi = -p.width / 2 + min(0.0, drift), p.width / 2 + max(0.0, drift)
        self.xl = rng.uniform(lo, hi, (n, L))
        self.phase = rng.random(n)
        self.bri = (0.3 + 0.7 * d) * (0.75 + 0.5 * rng.random(n))
        self.th = p.thickness * (0.55 + 0.65 * d)
        rs = self.rng("splashes")
        k = max(p.droplets, 1)
        self.dvx = rs.uniform(25, 65, (n, L, k)) * np.where(rs.random((n, L, k)) < 0.5, -1, 1)
        self.dvy = -rs.uniform(45, 110, (n, L, k))
        self.sb = rs.uniform(0.6, 1.2, (n, L))

    def get_bounds(self):
        p = self.p
        pad = p.splash_size * 1.3 + 3 * p.thickness + 4
        return (-p.width / 2 - pad, -p.height - 3 * p.thickness - 3, p.width / 2 + pad,
                0.4 * p.splash_size * 1.3 + 4)

    def _mask(self, x, y):
        p = self.p
        return soft_box(x, y, -p.width / 2, -p.height, p.width / 2, 1e6, p.edge)

    def draw(self, ctx, t):
        p = self.p
        if self.d.size == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        xl = pick(self.xl, j)
        top = -p.height
        for i in range(len(self.d)):
            a, tl, spd = float(age[i]), float(self.tl[i]), float(self.spd[i])
            yg, x_land = float(self.yg[i]), float(xl[i])
            ah, at = min(a, tl), min(max(a - p.length, 0.0), tl)
            if at < tl and ah > at:
                yh, yt = top + spd * ah, top + spd * at
                xh, xt = x_land - p.wind * (yg - yh), x_land - p.wind * (yg - yt)
                b = p.brightness * float(self.bri[i])
                mh, mt = float(self._mask(xh, yh)), float(self._mask(xt, yt))
                if mh + mt > 1e-3:
                    th = float(self.th[i])
                    ctx.light.polyline([(xt, yt), (xh, yh)], th * 0.45, b * 0.12 * mt, r_end=th, a_end=b * mh)
            s = a - tl
            if p.splashes > 0 and 0 <= s < p.splash_time and abs(x_land) <= p.width / 2:
                k = s / p.splash_time
                di = float(self.d[i])
                sz = p.splash_size * (0.55 + 0.6 * di)
                b = p.splashes * p.brightness * float(self.sb[i, j[i]]) * (0.4 + 0.6 * di)
                fade = float(smoothstep(p.width / 2, p.width / 2 - p.edge, abs(x_land))) if p.edge > 0 else 1.0
                b *= fade
                R = sz * (0.25 + 0.75 * (1 - (1 - k) ** 2))
                ctx.light.ring(x_land, yg, R, 0.4 + 0.35 * di, 0.9 * b * (1 - k) ** 1.5, squash=0.3)
                if k < 0.3:
                    ctx.light.dot(x_land, yg - 0.5, 0.7 + 0.5 * di, 1.6 * b * (1 - k / 0.3))
                for m in range(p.droplets):
                    vx, vy = float(self.dvx[i, j[i], m]), float(self.dvy[i, j[i], m])
                    sc = 0.5 + 0.6 * di
                    dx = vx * sc * s
                    dy = vy * sc * s + 0.5 * 900.0 * s * s
                    if dy <= 0:
                        ctx.light.dot(x_land + dx, yg + dy, 0.45 + 0.35 * di, 1.1 * b * (1 - k))


# =============================================================================
# snow
# =============================================================================
@atom("snowfall", tags=["environment", "weather", "snow", "loop"])
class Snowfall(Atom):
    """Snow: soft flakes drifting down with sway, wind and swirling (curl) flow.

    Local origin = centre of the area (``width`` x ``height``). Near flakes are bigger, brighter
    and faster; far ones small and slow (``depth``). Flakes enter from the top (or fade in when
    they are too slow to cross the area in one lap) and leave at the bottom; they fade at the
    area's edges (``edge``)."""
    kind = "loop"
    outputs = {"light": Light(palette="snow", bloom=[[1.2, 0.3], [4.0, 0.15]], exposure=1.0, alpha_gamma=0.8)}
    streams = ("flakes", "flow")

    class Params(Atom.Params):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        width: float = P(240.0, gt=0, doc="area width (px)")
        height: float = P(240.0, gt=0, doc="area height (px)")
        count: int = P(110, ge=0, doc="number of flakes")
        speed: float = P(55.0, gt=0, doc="fall speed of the nearest flakes (px/s)")
        speed_var: float = P(0.35, ge=0, le=1, doc="random speed variation (fraction)")
        depth: float = P(0.6, ge=0, le=1, doc="depth spread: far flakes are smaller, slower and dimmer")
        wind: float = P(8.0, doc="sideways drift (px/s, + = to +x)")
        sway: float = P(8.0, ge=0, doc="sideways sway amplitude (px)")
        sway_rate: float = P(0.45, ge=0, doc="sway frequency (Hz)")
        curl: float = P(16.0, ge=0, doc="swirling flow strength (px/s); 0 = straight paths")
        curl_scale: float = P(55.0, gt=0, doc="size of the swirling eddies (px)")
        size: float = P(1.3, gt=0, doc="radius of the nearest flakes (px, gaussian sigma)")
        size_var: float = P(0.5, ge=0, le=1, doc="random size variation (fraction)")
        brightness: float = P(1.6, ge=0, doc="flake intensity")
        softness: float = P(0.35, ge=0, doc="soft halo around each flake (intensity)")
        twinkle: float = P(0.0, ge=0, le=1, doc="random twinkle depth")
        edge: float = P(14.0, ge=0, doc="soft fade at the area's edges (px)")

    def setup(self):
        p = self.p
        rng = self.rng("flakes")
        n = p.count
        d = rng.random(n)
        self.d = d
        spd = p.speed * (1 - 0.6 * p.depth * (1 - d)) * (1 + p.speed_var * (rng.random(n) - 0.5))
        self.spd = np.maximum(spd, 1.0)
        m = self._margin()
        span = p.height + 2 * m
        self.span = span
        self.laps = np.maximum(1, np.round(self.spd * p.period / span)).astype(np.int64)
        self.travel = self.spd * p.period / self.laps
        L = int(self.laps.max()) if n else 1
        u = rng.random((n, L))
        self.y0 = -p.height / 2 - m + lap_spawn_y(self.travel[:, None], span, u)
        drift = p.wind * (p.period / self.laps)
        lo = -p.width / 2 - np.maximum(drift, 0) - 0.3 * p.sway
        hi = p.width / 2 - np.minimum(drift, 0) + 0.3 * p.sway
        self.x0 = lo[:, None] + rng.random((n, L)) * (hi - lo)[:, None]
        self.phase = rng.random(n)
        self.sw_ph = rng.random((n, L))
        self.sw_k = rng.uniform(0.6, 1.3, n)
        self.sz = p.size * (0.45 + 0.75 * d) * (1 + p.size_var * (rng.random(n) - 0.5))
        self.bri = p.brightness * (0.35 + 0.65 * d) * (0.8 + 0.4 * rng.random(n))
        self.tw = rng.random(n)
        self.flow = _flow(self.rng("flow"), p.curl, p.curl_scale, p.period)

    def _margin(self):
        return 3 * self.p.size * 1.5 + 4

    def get_bounds(self):
        p = self.p
        m = self._margin() + 2
        return (-p.width / 2 - m, -p.height / 2 - m, p.width / 2 + m, p.height / 2 + m)

    def draw(self, ctx, t):
        p = self.p
        if self.d.size == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        x0, y0 = pick(self.x0, j), pick(self.y0, j)
        x = x0 + p.wind * age
        y = y0 + self.spd * age
        env = _ss(0, 0.08, q) * _ss(1, 0.92, q)
        vis = (env > 1e-3) & (y > -p.height / 2 - self._margin() - 4) & (y < p.height / 2 + self._margin() + 4)
        idx = np.flatnonzero(vis)
        if idx.size == 0:
            return

        def base(aa, idx=idx):
            return np.stack([x0[idx] + p.wind * aa, y0[idx] + self.spd[idx] * aa], axis=1)
        off = _curl_offsets(self.flow, base, t, age[idx])
        xs = x[idx] + off[:, 0] + p.sway * self.sw_k[idx] * np.sin(
            TAU * (p.sway_rate * self.sw_k[idx] * age[idx] + pick(self.sw_ph, j)[idx]))
        ys = y[idx] + off[:, 1]
        a = self.bri[idx] * env[idx] * soft_box(xs, ys, -p.width / 2, -p.height / 2, p.width / 2, p.height / 2, p.edge)
        if p.twinkle > 0:
            a = a * (1 - p.twinkle + p.twinkle * (0.5 + 0.5 * np.sin(TAU * (1.5 * age[idx] + self.tw[idx]))) ** 2)
        for k, i in enumerate(idx):
            ak = float(a[k])
            if ak < 1e-3:
                continue
            s = float(self.sz[i])
            ctx.light.dot(float(xs[k]), float(ys[k]), s, ak)
            if p.softness > 0:
                ctx.light.dot(float(xs[k]), float(ys[k]), s * 2.4, ak * p.softness * 0.35)


# =============================================================================
# fireflies
# =============================================================================
@atom("firefly_swarm", tags=["environment", "fireflies", "motes", "loop"])
class FireflySwarm(Atom):
    """Fireflies: glowing motes wandering on a swirling (curl) flow, bobbing and blinking.

    Local origin = centre of the area. Each firefly lives ``laps_min..laps_max`` lifetimes per
    period: it fades in somewhere in the area, drifts and swirls, blinks ``blink_rate`` times a
    second, then fades out and reappears elsewhere."""
    kind = "loop"
    outputs = {"light": Light(palette="firefly", bloom=[[1.5, 0.5], [5.0, 0.45], [14.0, 0.3]], exposure=1.0,
                              alpha_gamma=0.8)}
    streams = ("flies", "flow")

    class Params(Atom.Params):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        width: float = P(220.0, gt=0, doc="area width (px)")
        height: float = P(160.0, gt=0, doc="area height (px)")
        count: int = P(22, ge=0, doc="number of fireflies")
        laps_min: int = P(1, ge=1, doc="minimum lifetimes per period (whole number)")
        laps_max: int = P(2, ge=1, doc="maximum lifetimes per period (whole number)")
        drift: float = P(10.0, ge=0, doc="random wandering speed (px/s)")
        rise: float = P(3.0, doc="upward drift (px/s; negative = sinks)")
        curl: float = P(22.0, ge=0, doc="swirling flow strength (px/s); 0 = straight drift")
        curl_scale: float = P(50.0, gt=0, doc="size of the swirling eddies (px)")
        bob: float = P(4.0, ge=0, doc="vertical bobbing amplitude (px)")
        blink_rate: float = P(0.9, ge=0, doc="blinks per second (randomised per firefly)")
        blink_sharpness: float = P(3.0, ge=1, doc="how short and sharp the blinks are")
        dim: float = P(0.12, ge=0, le=1, doc="brightness between blinks (fraction)")
        fade: float = P(0.25, gt=0, le=0.5, doc="fade-in / fade-out fraction of each life")
        size: float = P(1.2, gt=0, doc="core radius (px, gaussian sigma)")
        halo: float = P(0.3, ge=0, doc="soft halo brightness around each firefly")
        halo_size: float = P(5.0, gt=0, doc="halo radius (px)")
        brightness: float = P(2.4, ge=0, doc="peak intensity")
        edge: float = P(16.0, ge=0, doc="fade out near the area's edges (px)")

    def setup(self):
        p = self.p
        rng = self.rng("flies")
        n = p.count
        lo, hi = min(p.laps_min, p.laps_max), max(p.laps_min, p.laps_max)
        self.laps = rng.integers(lo, hi + 1, n) if n else np.zeros(0, np.int64)
        L = hi
        self.phase = rng.random(n)
        self.hx = rng.uniform(-p.width / 2 + p.edge, p.width / 2 - p.edge, (n, L))
        self.hy = rng.uniform(-p.height / 2 + p.edge, p.height / 2 - p.edge, (n, L))
        ang = rng.uniform(0, TAU, (n, L))
        sp = p.drift * rng.uniform(0.3, 1.0, (n, L))
        self.vx, self.vy = np.cos(ang) * sp, np.sin(ang) * sp - p.rise
        self.bk = p.blink_rate * rng.uniform(0.6, 1.4, n)
        self.bph = rng.random((n, L))
        self.bob_k = rng.uniform(0.5, 1.2, n)
        self.bob_ph = rng.random(n)
        self.bri = rng.uniform(0.6, 1.0, n)
        self.sz = p.size * rng.uniform(0.8, 1.2, n)
        self.flow = _flow(self.rng("flow"), p.curl, p.curl_scale, p.period)

    def get_bounds(self):
        p = self.p
        m = 3 * max(p.halo_size, p.size) + 2
        return (-p.width / 2 - m, -p.height / 2 - m, p.width / 2 + m, p.height / 2 + m)

    def draw(self, ctx, t):
        p = self.p
        n = len(self.laps)
        if n == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        env = _ss(0, p.fade, q) * _ss(1, 1 - p.fade, q)
        hx, hy, vx, vy = pick(self.hx, j), pick(self.hy, j), pick(self.vx, j), pick(self.vy, j)
        idx = np.flatnonzero(env > 1e-3)
        if idx.size == 0:
            return

        def base(aa, idx=idx):
            return np.stack([hx[idx] + vx[idx] * aa, hy[idx] + vy[idx] * aa], axis=1)
        off = _curl_offsets(self.flow, base, t, age[idx])
        a_ = age[idx]
        x = hx[idx] + vx[idx] * a_ + off[:, 0]
        y = hy[idx] + vy[idx] * a_ + off[:, 1] + p.bob * np.sin(TAU * (0.7 * self.bob_k[idx] * a_ + self.bob_ph[idx]))
        bl = (0.5 + 0.5 * np.cos(TAU * (self.bk[idx] * a_ + pick(self.bph, j)[idx]))) ** p.blink_sharpness
        b = p.brightness * self.bri[idx] * env[idx] * (p.dim + (1 - p.dim) * bl)
        b = b * soft_box(x, y, -p.width / 2, -p.height / 2, p.width / 2, p.height / 2, p.edge)
        for k, i in enumerate(idx):
            bk = float(b[k])
            if bk < 1e-3:
                continue
            ctx.light.dot(float(x[k]), float(y[k]), float(self.sz[i]), bk)
            if p.halo > 0:
                ctx.light.dot(float(x[k]), float(y[k]), p.halo_size, bk * p.halo)


# =============================================================================
# falling leaves
# =============================================================================
def _leaf_outline(kind: str, rng: np.random.Generator) -> np.ndarray:
    """Unit leaf outline (length 1 along +u, tip at u = +0.5), shape (m, 2)."""
    if kind == "maple":
        th = np.linspace(0, TAU, 44, endpoint=False)
        r = 0.5 * (0.58 + 0.42 * np.abs(np.cos(2.5 * th)) ** 1.6 + 0.05 * np.abs(np.cos(7.5 * th)))
        r = r * (1 - 0.18 * np.clip(-np.cos(th), 0, 1))               # smaller lobes toward the stem
        r = r * (1 + 0.06 * rng.normal(size=th.size))
        return np.stack([np.cos(th) * r, np.sin(th) * r * 0.95], axis=1)
    w = rng.uniform(0.46, 0.62)
    asym = rng.uniform(-0.08, 0.08)
    us = np.linspace(-0.5, 0.5, 9)
    prof = np.sin(np.pi * (us + 0.5)) ** 0.8 * (1 - 0.25 * (us + 0.5))  # widest a bit below the middle
    top = np.stack([us, w * 0.5 * prof * (1 + asym)], axis=1)
    bot = np.stack([us[::-1], -w * 0.5 * prof[::-1] * (1 - asym)], axis=1)
    return np.concatenate([top, bot[1:-1]], axis=0)


@atom("leaf_fall", tags=["environment", "leaves", "autumn", "solid", "loop"])
class LeafFall(Atom):
    """Falling leaves: small flat-shaded leaf polygons that drift down while swaying like a
    pendulum, tilting with the swing, spinning and flipping over (the narrowing leaf shows its
    darker back side).

    Local origin = centre of the area. Drawn into a solid role (painter's order, far leaves
    first); colour = channel palette at a random position per leaf (``color_var``)."""
    kind = "loop"
    outputs = {"solid": Solid(palette="autumn")}
    streams = ("leaves", "flow")

    class Params(Atom.Params):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        width: float = P(240.0, gt=0, doc="area width (px)")
        height: float = P(240.0, gt=0, doc="area height (px)")
        count: int = P(14, ge=0, doc="number of leaves")
        shape: str = P("mixed", doc="leaf outline", choices=["oval", "maple", "mixed"])
        size: float = P(18.0, gt=0, doc="length of the nearest leaves (px)")
        size_var: float = P(0.3, ge=0, le=1, doc="random size variation (fraction)")
        depth: float = P(0.5, ge=0, le=1, doc="depth spread: far leaves are smaller and slower")
        speed: float = P(60.0, gt=0, doc="fall speed of the nearest leaves (px/s)")
        speed_var: float = P(0.3, ge=0, le=1, doc="random speed variation (fraction)")
        wind: float = P(12.0, doc="sideways drift (px/s, + = to +x)")
        sway: float = P(16.0, ge=0, doc="pendulum sway amplitude (px)")
        sway_rate: float = P(0.5, ge=0, doc="pendulum swings per second")
        tilt: float = P(35.0, ge=0, doc="tilt with the swing (deg)")
        spin: float = P(60.0, ge=0, doc="random spin speed (deg/s, sigma)")
        flip_rate: float = P(0.45, ge=0, doc="flips per second (random per leaf, around this)")
        curl: float = P(0.0, ge=0, doc="swirling flow strength (px/s); 0 = off")
        curl_scale: float = P(60.0, gt=0, doc="size of the swirling eddies (px)")
        color_var: float = P(0.6, ge=0, le=1, doc="spread of leaf colours along the palette")
        veins: bool = P(True, doc="draw a midrib and stem on bigger leaves")
        edge: float = P(10.0, ge=0, doc="fade in / out at the area's edges (px)")

    def setup(self):
        p = self.p
        rng = self.rng("leaves")
        n = p.count
        d = rng.random(n)
        self.d = d
        spd = p.speed * (1 - 0.5 * p.depth * (1 - d)) * (1 + p.speed_var * (rng.random(n) - 0.5))
        self.spd = np.maximum(spd, 1.0)
        self.L = p.size * (0.55 + 0.45 * d) * (1 + p.size_var * (rng.random(n) - 0.5))
        m = float(self.L.max()) + p.sway * 0.2 + 2 if n else p.size
        span = p.height + 2 * m
        self.m = m
        self.laps = np.maximum(1, np.round(self.spd * p.period / span)).astype(np.int64)
        travel = self.spd * p.period / self.laps
        Lm = int(self.laps.max()) if n else 1
        self.y0 = -p.height / 2 - m + lap_spawn_y(travel[:, None], span, rng.random((n, Lm)))
        drift = p.wind * (p.period / self.laps)
        lo = -p.width / 2 - np.maximum(drift, 0)
        hi = p.width / 2 - np.minimum(drift, 0)
        self.x0 = lo[:, None] + rng.random((n, Lm)) * (hi - lo)[:, None]
        self.phase = rng.random(n)
        self.sw_ph = rng.random((n, Lm))
        self.sw_k = rng.uniform(0.7, 1.3, n)
        self.rot0 = rng.uniform(0, TAU, (n, Lm))
        self.spin = np.radians(p.spin) * rng.normal(0, 1, n)
        self.fk = p.flip_rate * rng.uniform(0.5, 1.5, n)
        self.fph = rng.random((n, Lm))
        c0 = 0.5 - 0.5 * p.color_var
        self.k = c0 + p.color_var * rng.random(n) * 0.9 + 0.08
        kinds = {"oval": ["oval"], "maple": ["maple"], "mixed": ["oval", "maple"]}[p.shape]
        self.kind_i = [kinds[int(v)] for v in rng.integers(0, len(kinds), n)] if n else []
        self.shapes = [_leaf_outline(kd, rng) for kd in self.kind_i]
        self.flow = _flow(self.rng("flow"), p.curl, p.curl_scale, p.period)

    def get_bounds(self):
        p = self.p
        m = self.m + 2
        return (-p.width / 2 - m, -p.height / 2 - m, p.width / 2 + m, p.height / 2 + m)

    def draw(self, ctx, t):
        p = self.p
        n = len(self.d)
        if n == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        x0, y0 = pick(self.x0, j), pick(self.y0, j)
        env = _ss(0, 0.06, q) * _ss(1, 0.94, q)
        y = y0 + self.spd * age
        H2 = p.height / 2 + self.m
        idx = np.flatnonzero((env > 1e-3) & (y > -H2 - 2) & (y < H2 + 2))
        if idx.size == 0:
            return

        def base(aa, idx=idx):
            return np.stack([x0[idx] + p.wind * aa, y0[idx] + self.spd[idx] * aa], axis=1)
        off = _curl_offsets(self.flow, base, t, age[idx])
        for k in np.argsort(self.d[idx], kind="stable"):
            i = int(idx[k])
            a = float(age[i])
            ph = TAU * (p.sway_rate * float(self.sw_k[i]) * a + float(self.sw_ph[i, j[i]]))
            x = float(x0[i]) + p.wind * a + p.sway * math.sin(ph) + float(off[k, 0])
            yy = float(y[i]) + float(off[k, 1]) - 0.15 * p.sway * math.cos(2 * ph)   # floats up at swing ends
            rot = float(self.rot0[i, j[i]]) + float(self.spin[i]) * a + math.radians(p.tilt) * math.cos(ph)
            flip = math.cos(TAU * (float(self.fk[i]) * a + float(self.fph[i, j[i]])))
            Ls = float(self.L[i])
            w2, h2 = p.width / 2 - 0.5 * Ls, p.height / 2 - 0.5 * Ls       # whole leaf inside the fade
            al = float(env[i]) * float(soft_box(x, yy, -w2, -h2, w2, h2, p.edge))
            if al < 1e-3:
                continue
            pts = self.shapes[i] * Ls
            fs = max(abs(flip) ** 0.6, 0.1)
            u, v = pts[:, 0], pts[:, 1] * fs
            c, s = math.cos(rot), math.sin(rot)
            poly = np.stack([x + c * u - s * v, yy + s * u + c * v], axis=1)
            kk = float(self.k[i])
            kk = kk if flip >= 0 else kk * 0.72
            kk = float(np.clip(kk + 0.1 * (fs - 0.5), 0, 1))
            ctx.solid.polygon(poly, k=kk, a=al)
            if self.kind_i[i] == "oval":          # lit half of the leaf (folded along the midrib)
                half = poly[: len(poly) // 2 + 1]
                ctx.solid.polygon(half, k=float(min(kk + 0.12, 1.0)), a=al * 0.8)
            if p.veins and Ls * fs >= 5:
                tip = 0.42 if self.kind_i[i] == "oval" else 0.3
                rib = [(x - c * 0.62 * Ls, yy - s * 0.62 * Ls), (x + c * tip * Ls, yy + s * tip * Ls)]
                ctx.solid.stroke(rib, max(0.04 * Ls, 0.35), k=max(kk - 0.22, 0.0), a=al * 0.85,
                                 r_end=max(0.015 * Ls, 0.2))


# =============================================================================
# god rays
# =============================================================================
@atom("god_rays", tags=["environment", "light", "rays", "loop"])
class GodRays(Atom):
    """Light shafts (god rays): soft angled beams falling from a horizontal source line, each
    widening and fading with distance, flickering with slow noise, streaked along their length,
    with dust motes drifting through them that glow only inside the light.

    Local origin = centre of the source line (top); beams lean ``angle`` degrees toward +x."""
    kind = "loop"
    outputs = {"light": Light(palette="sunbeam", bloom=[[2.0, 0.2], [8.0, 0.15]], exposure=1.0, alpha_gamma=0.9)}
    streams = ("beams", "noise", "dust", "flow")

    class Params(Atom.Params):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        width: float = P(150.0, gt=0, doc="span of the beam origins along the source line (px)")
        length: float = P(230.0, gt=0, doc="beam length (px, measured vertically)")
        angle: float = P(22.0, ge=-60, le=60, doc="lean of the beams from vertical (deg, + = toward +x)")
        beams: int = P(6, ge=1, le=24, doc="number of beams")
        beam_width: float = P(9.0, gt=0, doc="beam half-width at the source (px, sigma)")
        width_var: float = P(0.5, ge=0, le=1, doc="random beam width variation")
        spread: float = P(0.7, ge=0, doc="extra relative width at the far end")
        brightness: float = P(0.45, ge=0, doc="beam intensity")
        falloff: float = P(1.2, ge=0, doc="fade along the beam (higher = shorter beams)")
        flicker: float = P(0.4, ge=0, le=1, doc="noise flicker depth of each beam")
        flicker_speed: int = P(2, ge=1, doc="flicker evolution cycles per period (whole number)")
        sway: float = P(3.0, ge=0, doc="slow sideways sway of the beams (px)")
        streaks: float = P(0.5, ge=0, le=1, doc="noise streaks running along the beams")
        haze: float = P(0.12, ge=0, doc="soft glow filling the whole shaft area")
        dust: int = P(40, ge=0, doc="number of dust motes drifting in the beams")
        dust_brightness: float = P(2.2, ge=0, doc="mote intensity (scaled by the light they are in)")
        dust_size: float = P(0.8, gt=0, doc="mote radius (px)")
        dust_drift: float = P(7.0, ge=0, doc="mote drift speed (px/s)")
        curl: float = P(10.0, ge=0, doc="swirling flow on the motes (px/s)")
        curl_scale: float = P(40.0, gt=0, doc="size of the motes' swirling eddies (px)")

    def setup(self):
        p = self.p
        rng = self.rng("beams")
        nb = p.beams
        xs = (np.arange(nb) + 0.5) / nb - 0.5
        self.bx = (xs + rng.uniform(-0.35, 0.35, nb) / nb) * p.width
        self.bw = p.beam_width * (1 + p.width_var * (rng.random(nb) - 0.5) * 1.6)
        self.bb = rng.uniform(0.55, 1.0, nb)
        self.bph = rng.random(nb)
        self.noise = Noise(self.rng("noise"), period=(32, 16, 4))
        rd = self.rng("dust")
        n = p.dust
        self.dlaps = rd.integers(1, 3, n) if n else np.zeros(0, np.int64)
        L = 2
        self.dphase = rd.random(n)
        self.dbeam = rd.integers(0, nb, (n, L))
        self.dv = rd.uniform(0.08, 0.85, (n, L))
        self.du = rd.normal(0, 0.8, (n, L))
        ang = rd.uniform(0, TAU, (n, L))
        sp = p.dust_drift * rd.uniform(0.3, 1.0, (n, L))
        self.dvx, self.dvy = np.cos(ang) * sp, np.sin(ang) * sp
        self.dtw = rd.random(n)
        self.dsz = p.dust_size * rd.uniform(0.6, 1.3, n)
        self.flow = _flow(self.rng("flow"), p.curl, p.curl_scale, p.period)
        ta = math.tan(math.radians(p.angle))
        self.ta, self.ca = ta, math.cos(math.radians(p.angle))

    def get_bounds(self):
        p = self.p
        m = 3 * self.p.beam_width * (1 + p.spread) * (1 + p.width_var) + p.sway + 4
        dx = p.length * math.tan(math.radians(p.angle))
        x0 = min(-p.width / 2, -p.width / 2 + dx) - m
        x1 = max(p.width / 2, p.width / 2 + dx) + m
        return (x0, -4.0, x1, p.length + 4)

    def _flicker(self, t):
        p = self.p
        z = (t / p.period) * p.flicker_speed * self.noise.pz
        i = np.arange(len(self.bx), dtype=np.float32)
        n = self.noise.fbm(i * 3.7 + 0.3, np.full_like(i, 5.5), z, octaves=2)
        return self.bb * np.clip(1 + p.flicker * 1.8 * n, 0.05, None)

    def intensity(self, X, Y, t, flick=None):
        """Beam light at local points (arrays), without haze / streaks."""
        p = self.p
        flick = self._flicker(t) if flick is None else flick
        yn = np.clip(Y / p.length, 0, 1)
        env = _ss(-2, 0.1 * p.length, Y) * (1 - yn) ** p.falloff * _ss(p.length, 0.85 * p.length, Y)
        acc = 0.0
        s_t = t / p.period
        for i in range(len(self.bx)):
            cx = self.bx[i] + p.sway * math.sin(TAU * (s_t + self.bph[i])) + Y * self.ta
            s = self.bw[i] * (1 + p.spread * yn)
            du = (X - cx) * self.ca
            acc = acc + float(flick[i]) * np.exp(-du * du / (2 * s * s)) * np.sqrt(self.bw[i] / s)
        return acc * env

    def draw(self, ctx, t):
        p = self.p
        flick = self._flicker(t)
        z = (t / p.period) * self.noise.pz
        b = self.get_bounds()

        def f(X, Y):
            v = self.intensity(X, Y, t, flick)
            if p.streaks > 0:
                u = (X - Y * self.ta) / 9.0
                n = self.noise.fbm(u, Y / p.length * 2.0, z, octaves=2)
                v = v * np.clip(1 - p.streaks + p.streaks * (1 + 2.2 * n), 0, None)
            if p.haze > 0:
                yn = np.clip(Y / p.length, 0, 1)
                cx = Y * self.ta
                hw = p.width / 2 + p.beam_width * 2
                hz = _ss(hw + 5 * p.beam_width, hw - 4 * p.beam_width, np.abs(X - cx)) * (1 - yn) ** (p.falloff + 0.5)
                v = v + p.haze * hz * _ss(-2, 0.15 * p.length, Y)
            return v * p.brightness
        ctx.light.field(f, b, lowres="bilinear")
        n = len(self.dlaps)
        if n == 0 or p.dust_brightness <= 0:
            return
        j, q, age = lap_clock(t, p.period, self.dlaps, self.dphase)
        bi = pick(self.dbeam, j)
        Y0 = pick(self.dv, j) * p.length
        yn0 = Y0 / p.length
        X0 = self.bx[bi] + Y0 * self.ta + pick(self.du, j) * self.bw[bi] * (1 + p.spread * yn0) / self.ca
        vx, vy = pick(self.dvx, j), pick(self.dvy, j)
        env = _ss(0, 0.25, q) * _ss(1, 0.75, q)

        def base(aa):
            return np.stack([X0 + vx * aa, Y0 + vy * aa], axis=1)
        off = _curl_offsets(self.flow, base, t, age)
        x = X0 + vx * age + off[:, 0]
        y = Y0 + vy * age + off[:, 1]
        lit = self.intensity(x, y, t, flick)
        tw = 0.55 + 0.45 * np.sin(TAU * (0.8 * age + self.dtw)) ** 2
        a = p.dust_brightness * p.brightness * lit * env * tw
        for k in np.flatnonzero(a > 2e-3):
            ctx.light.dot(float(x[k]), float(y[k]), float(self.dsz[k]), float(a[k]))


# =============================================================================
# dust motes
# =============================================================================
@atom("floating_motes", tags=["environment", "dust", "motes", "loop"])
class FloatingMotes(Atom):
    """Dust motes: slow floating specks on a swirling (curl) flow that catch the light, flashing
    briefly as they turn (optional 4-point glint on the brightest flashes).

    Local origin = centre of the area. ``band`` > 0 concentrates the light in a diagonal band
    (a sunbeam through the motes): motes outside it stay dim."""
    kind = "loop"
    outputs = {"light": Light(palette="mote", bloom=[[1.2, 0.35], [4.0, 0.2]], exposure=1.0, alpha_gamma=0.8)}
    streams = ("motes", "flow")

    class Params(Atom.Params):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        width: float = P(220.0, gt=0, doc="area width (px)")
        height: float = P(220.0, gt=0, doc="area height (px)")
        count: int = P(50, ge=0, doc="number of motes")
        laps_min: int = P(1, ge=1, doc="minimum lifetimes per period (whole number)")
        laps_max: int = P(1, ge=1, doc="maximum lifetimes per period (whole number)")
        drift: float = P(5.0, ge=0, doc="random drift speed (px/s)")
        rise: float = P(-2.0, doc="upward drift (px/s; negative = slowly settles)")
        curl: float = P(9.0, ge=0, doc="swirling flow strength (px/s)")
        curl_scale: float = P(45.0, gt=0, doc="size of the swirling eddies (px)")
        size: float = P(0.75, gt=0, doc="mote radius (px)")
        size_var: float = P(0.6, ge=0, le=1, doc="random size variation (fraction)")
        brightness: float = P(1.2, ge=0, doc="base intensity")
        glint: float = P(1.6, ge=0, doc="extra intensity of the catch-light flashes")
        glint_rate: float = P(0.35, ge=0, doc="flashes per second (random per mote)")
        glint_sharpness: float = P(10.0, ge=1, doc="how short the flashes are")
        star: float = P(0.5, ge=0, doc="4-point star brightness on strong flashes (0 = off)")
        halo: float = P(0.15, ge=0, doc="soft halo brightness")
        band: float = P(0.0, ge=0, le=1, doc="light only a diagonal band (0 = everywhere lit)")
        band_angle: float = P(-60.0, doc="direction of the light band (deg)")
        band_width: float = P(70.0, gt=0, doc="half-width of the light band (px)")
        fade: float = P(0.25, gt=0, le=0.5, doc="fade-in / fade-out fraction of each life")
        edge: float = P(16.0, ge=0, doc="fade near the area's edges (px)")

    def setup(self):
        p = self.p
        rng = self.rng("motes")
        n = p.count
        lo, hi = min(p.laps_min, p.laps_max), max(p.laps_min, p.laps_max)
        self.laps = rng.integers(lo, hi + 1, n) if n else np.zeros(0, np.int64)
        L = hi
        self.phase = rng.random(n)
        self.hx = rng.uniform(-p.width / 2, p.width / 2, (n, L))
        self.hy = rng.uniform(-p.height / 2, p.height / 2, (n, L))
        ang = rng.uniform(0, TAU, (n, L))
        sp = p.drift * rng.uniform(0.2, 1.0, (n, L))
        self.vx, self.vy = np.cos(ang) * sp, np.sin(ang) * sp - p.rise
        self.gk = p.glint_rate * rng.uniform(0.5, 1.5, n)
        self.gph = rng.random((n, L))
        self.sz = p.size * (1 + p.size_var * (rng.random(n) - 0.5) * 1.6)
        self.bri = rng.uniform(0.4, 1.0, n)
        self.flow = _flow(self.rng("flow"), p.curl, p.curl_scale, p.period)

    def get_bounds(self):
        p = self.p
        m = 3 * p.size * (1 + p.size_var) * 6 + 4
        return (-p.width / 2 - m, -p.height / 2 - m, p.width / 2 + m, p.height / 2 + m)

    def draw(self, ctx, t):
        p = self.p
        n = len(self.laps)
        if n == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        env = _ss(0, p.fade, q) * _ss(1, 1 - p.fade, q)
        hx, hy, vx, vy = pick(self.hx, j), pick(self.hy, j), pick(self.vx, j), pick(self.vy, j)
        idx = np.flatnonzero(env > 1e-3)
        if idx.size == 0:
            return

        def base(aa, idx=idx):
            return np.stack([hx[idx] + vx[idx] * aa, hy[idx] + vy[idx] * aa], axis=1)
        a_ = age[idx]
        off = _curl_offsets(self.flow, base, t, a_)
        x = hx[idx] + vx[idx] * a_ + off[:, 0]
        y = hy[idx] + vy[idx] * a_ + off[:, 1]
        light = np.ones_like(x)
        if p.band > 0:
            ba = math.radians(p.band_angle)
            d = -x * math.sin(ba) + y * math.cos(ba)
            light = 1 - p.band + p.band * np.exp(-d * d / (2 * p.band_width ** 2))
        fl = (0.5 + 0.5 * np.cos(TAU * (self.gk[idx] * a_ + pick(self.gph, j)[idx]))) ** p.glint_sharpness
        b = env[idx] * light * soft_box(x, y, -p.width / 2, -p.height / 2, p.width / 2, p.height / 2, p.edge)
        base_b = p.brightness * self.bri[idx] * b
        flash = p.glint * fl * b
        for k, i in enumerate(idx):
            bk = float(base_b[k] + flash[k])
            if bk < 1e-3:
                continue
            xx, yy, s = float(x[k]), float(y[k]), float(self.sz[i])
            ctx.light.dot(xx, yy, s, bk)
            if p.halo > 0:
                ctx.light.dot(xx, yy, s * 4, bk * p.halo)
            if p.star > 0 and flash[k] > 0.25:
                L = s * 6 * min(float(flash[k]), 1.5)
                a = float(flash[k]) * p.star
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ctx.light.polyline([(xx, yy), (xx + dx * L, yy + dy * L)], s * 0.35, a, r_end=s * 0.12, a_end=0.0)


# =============================================================================
# waterfall
# =============================================================================
@atom("water_cascade", tags=["environment", "water", "waterfall", "loop"])
class WaterCascade(Atom):
    """Waterfall: a falling water sheet with accelerating streaks, a bright lip at the top, glossy
    highlights and churning foam where it hits the pool.

    Local origin = base centre (where the sheet meets the pool); the lip is at ``y = -height``.
    Water body -> ``volume`` (glossy shading), highlights and lip -> ``light``, foam -> ``smoke``.
    Streak texture moves with the falling speed ``sqrt(speed² + 2·gravity·fallen)`` and scrolls a
    whole number of noise cells per period, so the loop is seamless."""
    kind = "loop"
    outputs = {"volume": Volume(palette="water", density=1.6, bump=5, soft=1.0, shadow=0.1, wrap=0.7, spec=0.9,
                                spec_power=26, rim=0.4, size_ref=256),
               "smoke": Smoke(palette="steam", density=2.2, shadow=0.5, relief=0.8, ambient=0.45, height_gain=0.6,
                              size_ref=256),
               "light": Light(palette="glint", bloom=[[1.0, 0.35], [3.0, 0.2]], exposure=1.0, alpha_gamma=0.85)}
    streams = ("noise", "foam")
    anchors = {"base": (0.0, 0.0), "lip": lambda self, t: (0.0, -self.p.height)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        width_top: float = P(64.0, gt=0, doc="sheet width at the lip (px)")
        width_bottom: float = P(84.0, gt=0, doc="sheet width at the base (px)")
        height: float = P(190.0, gt=0, doc="fall height (px)")
        speed: float = P(90.0, ge=0, doc="water speed at the lip (px/s)")
        gravity: float = P(700.0, ge=0, doc="acceleration down the fall (px/s²)")
        flow_cycles: int = P(16, ge=1, doc="noise cells scrolled per period (whole number; more = faster texture)")
        streak_scale: float = P(4.0, gt=0, doc="width of the vertical streaks (px)")
        streaks: float = P(0.7, ge=0, doc="streak contrast in the water body")
        density: float = P(0.9, ge=0, doc="water body density")
        gaps: float = P(0.35, ge=0, le=1, doc="how much the sheet tears into gaps toward the bottom")
        edge_wobble: float = P(4.0, ge=0, doc="ragged edge amplitude (px)")
        highlights: float = P(0.9, ge=0, doc="intensity of the glossy highlight streaks (light)")
        lip: float = P(1.0, ge=0, doc="intensity of the bright lip at the top (light)")
        foam: float = P(1.0, ge=0, doc="foam density at the base (smoke; 0 = none)")
        foam_width: float = P(1.6, gt=0, doc="foam width relative to the sheet's base width")
        foam_height: float = P(18.0, gt=0, doc="foam height (px)")
        churn: int = P(1, ge=0, doc="foam churn cycles per period (whole number)")

    def setup(self):
        p = self.p
        self.noise = Noise(self.rng("noise"), period=(64, p.flow_cycles, 2))
        self.fnoise = Noise(self.rng("foam"), period=(32, 32, 4))

    def get_bounds(self):
        p = self.p
        w = max(p.width_top, p.width_bottom, p.width_bottom * p.foam_width) / 2 + p.edge_wobble + 10
        return (-w, -p.height - 8, w, p.foam_height * 0.9 + 6)

    def _tau(self, Y):
        p = self.p
        yr = np.clip(Y + p.height, 0, None)
        if p.gravity <= 1e-6:
            return yr / max(p.speed, 1e-3)
        v = np.sqrt(p.speed ** 2 + 2 * p.gravity * yr)
        return (v - p.speed) / p.gravity

    def draw(self, ctx, t):
        p = self.p
        s = t / p.period
        rate = p.flow_cycles / p.period
        z = s * self.noise.pz
        H = p.height

        def sheet(X, Y):
            yn = np.clip((Y + H) / H, 0, 1)
            fy = (self._tau(Y) - t) * rate
            wob = p.edge_wobble * self.noise.fbm(np.sign(X) * 5.3 + 11.0, fy, z, octaves=2)
            half = (p.width_top + (p.width_bottom - p.width_top) * yn ** 0.8) / 2
            m = _ss(half + 1.5, half - 2.5, np.abs(X) - wob) * _ss(-H - 1.5, -H + 2.5, Y) * _ss(4, -4, Y)
            st = self.noise.fbm(X / p.streak_scale, fy, z, octaves=3)
            return m, st, fy, yn

        def vol(X, Y):
            m, st, fy, yn = sheet(X, Y)
            body = 0.8 + p.streaks * st * (0.4 + 0.8 * yn)
            tear = p.gaps * _ss(0.45, 1.0, yn) * _ss(0.1, -0.35, st)
            return p.density * m * np.clip(body - tear, 0, None)
        ctx.volume.field(vol, self.get_bounds(), lowres="bilinear")

        def light(X, Y):
            m, st, fy, yn = sheet(X, Y)
            st2 = self.noise.fbm(X / (p.streak_scale * 0.6) + 7.7, fy * 1.0 + 3.1, z, octaves=2)
            hl = p.highlights * m * np.clip((st2 - 0.18) * 3.5, 0, None) ** 1.6 * (0.35 + 0.65 * yn)
            half_t = p.width_top / 2
            lipv = p.lip * np.exp(-((Y + H + 1.0) / 2.5) ** 2) * _ss(half_t + 1, half_t - 4, np.abs(X))
            lipv = lipv * (0.7 + 0.5 * self.noise(X / 6.0 + 2.2, 4.4, z))
            return hl + lipv
        ctx.light.field(light, (-p.width_bottom, -H - 6, p.width_bottom, 4), lowres="bilinear")

        if p.foam > 0:
            rx = p.width_bottom * p.foam_width / 2
            ry = p.foam_height
            zf = s * p.churn * self.fnoise.pz

            def foam(X, Y):
                r = np.sqrt((X / rx) ** 2 + ((Y + 0.35 * ry) / ry) ** 2)
                body = np.exp(-r * r * 1.6) * _ss(ry * 0.9, 0, Y)
                n = self.fnoise.fbm(X / 11.0, Y / 7.0 + 2.0, zf, octaves=3)
                return p.foam * np.clip(body * (1 + 0.9 * n) - 0.18, 0, None)
            ctx.smoke.field(foam, (-rx * 1.6, -ry * 1.6 - 0.35 * ry, rx * 1.6, ry * 0.9), lowres="bilinear")
