"""Tileable surface atoms (for scenes with ``tile: true``): periodic in space and time.

* ``fog_field``      soft drifting fog / mist banks (smoke)
* ``water_surface``  water body with moving ripples (volume) and caustic light network (light)
* ``energy_field``   flowing plasma veins over a soft glow (light)
* ``rain_sheet``     slanted rain streaks falling through the tile (light)

Tile-only: place them with ``at: [0, 0]``, ``anchor: {px: [0, 0]}`` and identity transform, and
set ``width`` / ``height`` to the scene size. Every noise lookup uses a lattice period that
divides the tile, drifts move whole tiles per period, and the time axis loops, so the output
tiles seamlessly in space and as a loop. Fields are evaluated at output-pixel centres
(``lowres="nearest"``), which is exact and seam-free on the wrap.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Volume, atom
from ..engine.color import register_palette
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise
from .environment import lap_clock, pick

register_palette("fog", "#535d69,#98a2ae,#e2e8ee")
register_palette("caustics", "#2a6a8c,#7cc8e6,#d8f6ff,#ffffff")

_TILE_DOC = " (px; set to the scene size)"


def _cells(n: int, W: float, H: float) -> tuple[int, int]:
    """Lattice cells across x / y for ``n`` cells across the tile width (square-ish cells)."""
    return max(1, int(n)), max(1, int(round(n * H / W)))


class _TileParams(Atom.Params):
    width: float = P(256.0, gt=0, doc="tile width" + _TILE_DOC)
    height: float = P(256.0, gt=0, doc="tile height" + _TILE_DOC)


class _TileField(Atom):
    """Common bits: bounds = the tile, full-tile field helper."""
    kind = "loop"

    def tile_box(self, view):
        """Full-tile field box. Starts one SS pixel in: works around View.window() in wrap mode
        with lowres (step > 1), which aligns the start down and covers one extra output pixel,
        depositing the wrap column / row twice (a seam at full quality)."""
        e = 1.0 / (view.ss * max(view.px, 1e-6))
        return (e, e, float(self.p.width), float(self.p.height))

    def get_bounds(self):
        # margin: room for bloom / shading spill when the atom is rendered on its own (non-tile)
        m = 40.0
        return (-m, -m, float(self.p.width) + m, float(self.p.height) + m)

    def _uv(self, X, Y, cx, cy, sx=0.0, sy=0.0):
        """Lattice coordinates (cells) shifted by (sx, sy) whole tiles."""
        p = self.p
        return X / p.width * cx - sx * cx, Y / p.height * cy - sy * cy


# =============================================================================
# fog
# =============================================================================
@atom("fog_field", tags=["tile", "fog", "mist", "smoke", "loop", "environment"])
class FogField(_TileField):
    """Tileable fog: soft domain-warped banks of mist drifting across the tile, with a finer,
    faster wisp layer. Tile-only (see module doc)."""
    outputs = {"smoke": Smoke(palette="fog", density=1.6, shadow=0.25, relief=0.25, ambient=0.65,
                              height_gain=0.5, size_ref=256)}
    streams = ("banks", "wisps")

    class Params(_TileParams):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        cells: int = P(3, ge=1, le=16, doc="fog banks across the tile (whole number)")
        octaves: int = P(3, ge=1, le=6, doc="detail levels")
        density: float = P(0.7, ge=0, doc="fog density")
        coverage: float = P(0.55, ge=0, le=1, doc="how much of the tile is foggy")
        softness: float = P(0.8, gt=0, doc="softness of the bank edges")
        drift: int = P(1, doc="tiles drifted along x per period (whole number; negative = to -x)")
        drift_y: int = P(0, doc="tiles drifted along y per period (whole number)")
        evolve: int = P(1, ge=0, doc="noise evolution cycles per period (whole number)")
        warp: float = P(0.35, ge=0, doc="domain warp (cells): curls the banks")
        haze: float = P(0.15, ge=0, le=1, doc="thin haze everywhere, between the banks (fraction of density)")
        wisps: float = P(0.45, ge=0, doc="amount of the finer wisp layer")
        wisp_drift: int = P(2, doc="tiles the wisp layer drifts along x per period (whole number)")

    def setup(self):
        p = self.p
        cx, cy = _cells(p.cells, p.width, p.height)
        self.c1 = (cx, cy)
        self.n1 = Noise(self.rng("banks"), period=(cx, cy, 4))
        self.c2 = (2 * cx, 2 * cy)
        self.n2 = Noise(self.rng("wisps"), period=(2 * cx, 2 * cy, 4))

    def draw(self, ctx, t):
        p = self.p
        s = t / p.period
        z = s * p.evolve * 4

        def f(X, Y):
            u, v = self._uv(X, Y, *self.c1, p.drift * s, p.drift_y * s)
            if p.warp > 0:
                wu = self.n1.fbm(u + 3.1, v + 7.7, z, octaves=2)
                wv = self.n1.fbm(u + 11.3, v + 1.9, z, octaves=2)
                u, v = u + p.warp * wu, v + p.warp * wv
            n = self.n1.fbm(u, v, z, octaves=p.octaves)
            thr = (0.5 - p.coverage) * 0.6
            d = smoothstep(thr - 0.35 * p.softness, thr + 0.35 * p.softness, n)
            if p.wisps > 0:
                u2, v2 = self._uv(X, Y, *self.c2, p.wisp_drift * s, p.drift_y * s)
                w = self.n2.fbm(u2, v2, z * 2, octaves=3)
                d = d * (1 - 0.35 * p.wisps + 0.7 * p.wisps * smoothstep(-0.3, 0.4, w))
            return p.density * (p.haze + (1 - p.haze) * d)
        ctx.smoke.field(f, self.tile_box(ctx.smoke), lowres="nearest")


# =============================================================================
# water
# =============================================================================
def _worley_edges(u, v, fx, fy):
    """Periodic Worley F2 - F1 for feature points ``(fx, fy)`` (cell offsets, shape (ny, nx))."""
    ny, nx = fx.shape
    iu, iv = np.floor(u), np.floor(v)
    du, dv = u - iu, v - iv
    iu = iu.astype(np.int64)
    iv = iv.astype(np.int64)
    F1 = np.full(u.shape, 9.0, np.float32)
    F2 = np.full(u.shape, 9.0, np.float32)
    for dj in (-1, 0, 1):
        cj = (iv + dj) % ny
        for di in (-1, 0, 1):
            ci = (iu + di) % nx
            px = di + fx[cj, ci] - du
            py = dj + fy[cj, ci] - dv
            d = np.sqrt(px * px + py * py)
            F2 = np.where(d < F1, F1, np.minimum(F2, d))
            F1 = np.minimum(F1, d)
    return F2 - F1


@atom("water_surface", tags=["tile", "water", "caustics", "loop", "environment"])
class WaterSurface(_TileField):
    """Tileable water: a water body whose surface ripples (``volume`` role: glossy shading from the
    ripple heights) and a moving caustic light network (``light`` role: animated, domain-warped
    Worley cell edges in two layers). Tile-only (see module doc)."""
    outputs = {"volume": Volume(palette="water", density=1.8, bump=9, soft=1.2, shadow=0.0, wrap=0.6, spec=0.7,
                                spec_power=28, rim=0.0, size_ref=256),
               "light": Light(palette="caustics", blend="add", bloom=[[1.0, 0.3], [4.0, 0.2]], exposure=1.0,
                              alpha_gamma=0.9)}
    streams = ("cells", "ripples")

    class Params(_TileParams):
        period: float = P(4.0, gt=0, doc="loop length (s)")
        cells: int = P(5, ge=1, le=24, doc="caustic cells across the tile (whole number)")
        caustics: float = P(1.0, ge=0, doc="caustic light intensity (0 = none)")
        sharpness: float = P(0.09, gt=0, doc="caustic line width (cell units; smaller = thinner)")
        wobble: float = P(0.22, ge=0, le=0.5, doc="how far each caustic cell's centre circles (cell units)")
        warp: float = P(0.25, ge=0, doc="domain warp of the caustic network (cells)")
        flow: int = P(1, doc="tiles drifted along x per period (whole number)")
        flow_y: int = P(0, doc="tiles drifted along y per period (whole number)")
        depth: float = P(1.3, ge=0, doc="water body density (0 = caustics only)")
        ripple: float = P(0.35, ge=0, doc="ripple height variation of the water body")
        ripple_cells: int = P(4, ge=1, le=24, doc="ripples across the tile (whole number)")
        evolve: int = P(1, ge=0, doc="ripple evolution cycles per period (whole number)")

    def setup(self):
        p = self.p
        rng = self.rng("cells")
        self.layers = []
        for k, mult in enumerate((1.0, 1.5)):
            nx, ny = _cells(int(round(p.cells * mult)), p.width, p.height)
            base_x = 0.5 + rng.uniform(-0.25, 0.25, (ny, nx))
            base_y = 0.5 + rng.uniform(-0.25, 0.25, (ny, nx))
            ph = rng.random((ny, nx))
            cyc = rng.choice(np.array([-1, 1]), (ny, nx)) * (1 + (rng.random((ny, nx)) < 0.3 * k))
            self.layers.append((nx, ny, base_x, base_y, ph, cyc))
        cx, cy = _cells(p.ripple_cells, p.width, p.height)
        self.rc = (cx, cy)
        rr = self.rng("ripples")
        self.rn = Noise(rr, period=(cx, cy, 4))
        self.wn = Noise(rr, period=(cx, cy, 4))

    def _caustics(self, X, Y, s):
        p = self.p
        z = s * 4
        cs = []
        for k, (nx, ny, bx, by, ph, cyc) in enumerate(self.layers):
            sgn = 1 if k == 0 else -1
            u = X / p.width * nx - sgn * p.flow * s * nx
            v = Y / p.height * ny - sgn * p.flow_y * s * ny
            if p.warp > 0:
                wu, wv = self._uv(X, Y, *self.rc)
                u = u + p.warp * self.wn.fbm(wu + 1.7, wv, z, octaves=2) * (1.4 if k else 1)
                v = v + p.warp * self.wn.fbm(wu, wv + 4.2, z, octaves=2) * (1.4 if k else 1)
            ang = TAU * (cyc * s + ph)
            fx = bx + p.wobble * np.cos(ang)
            fy = by + p.wobble * np.sin(ang)
            e = _worley_edges(u, v, fx.astype(np.float32), fy.astype(np.float32))
            w = p.sharpness * (1.0 if k == 0 else 0.8)
            cs.append(np.exp(-e / w))
        c1, c2 = cs
        return 0.3 * (c1 + c2) + 1.1 * c1 * c2

    def draw(self, ctx, t):
        p = self.p
        s = t / p.period
        if p.depth > 0:
            z = s * p.evolve * 4

            def vol(X, Y):
                u, v = self._uv(X, Y, *self.rc)
                h = self.rn.fbm(u - p.flow * s * self.rc[0], v - p.flow_y * s * self.rc[1], z, octaves=3)
                h2 = self.rn.fbm(u * 2 + 3.3 + p.flow * s * self.rc[0] * 2, v * 2 + 1.1, z * 2, octaves=2)
                return p.depth * np.clip(1 + p.ripple * (h * 1.2 + 0.6 * h2), 0.05, None)
            ctx.volume.field(vol, self.tile_box(ctx.volume), lowres="nearest")
        if p.caustics > 0:
            ctx.light.field(lambda X, Y: p.caustics * self._caustics(X, Y, s), self.tile_box(ctx.light), lowres="nearest")


# =============================================================================
# energy
# =============================================================================
@atom("energy_field", tags=["tile", "energy", "plasma", "light", "loop"])
class EnergyField(_TileField):
    """Tileable energy / plasma: ridged, domain-warped noise veins flowing across the tile over a
    soft glow, with an optional pulse. Tile-only (see module doc)."""
    outputs = {"light": Light(palette="arcane", bloom=[[1.5, 0.45], [5.0, 0.35], [14.0, 0.2]], exposure=1.0,
                              alpha_gamma=0.85)}
    streams = ("noise", "warp")

    class Params(_TileParams):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        cells: int = P(3, ge=1, le=16, doc="noise cells across the tile (whole number)")
        octaves: int = P(4, ge=1, le=6, doc="detail levels")
        flow: int = P(1, doc="tiles flowed along x per period (whole number)")
        flow_y: int = P(0, doc="tiles flowed along y per period (whole number)")
        evolve: int = P(1, ge=0, doc="noise evolution cycles per period (whole number)")
        warp: float = P(0.9, ge=0, doc="domain warp (cells): how much the veins twist")
        sharpness: float = P(6.0, ge=1, doc="vein sharpness (higher = thinner veins)")
        veins: float = P(1.4, ge=0, doc="vein intensity")
        glow: float = P(0.25, ge=0, doc="soft background glow intensity")
        pulse: float = P(0.15, ge=0, le=1, doc="brightness pulse depth")
        pulse_cycles: int = P(1, ge=0, doc="pulses per period (whole number)")

    def setup(self):
        p = self.p
        cx, cy = _cells(p.cells, p.width, p.height)
        self.c = (cx, cy)
        self.n = Noise(self.rng("noise"), period=(cx, cy, 4))
        self.w = Noise(self.rng("warp"), period=(cx, cy, 4))

    def draw(self, ctx, t):
        p = self.p
        s = t / p.period
        z = s * p.evolve * 4
        k = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * s) if p.pulse else 1.0

        def f(X, Y):
            u, v = self._uv(X, Y, *self.c, p.flow * s, p.flow_y * s)
            u0, v0 = self._uv(X, Y, *self.c)
            wu = self.w.fbm(u0 + 0.5, v0 + 3.3, z, octaves=2)
            wv = self.w.fbm(u0 + 6.1, v0 + 0.7, z, octaves=2)
            n1 = self.n.fbm(u + p.warp * wu, v + p.warp * wv, z, octaves=p.octaves)
            r1 = (1 - np.clip(np.abs(n1) * 2.2, 0, 1)) ** p.sharpness
            n2 = self.n.fbm(u0 * 2 - p.flow * s * self.c[0] * 2 + p.warp * wv, v0 * 2 + 5.5 + p.warp * wu, z * 2,
                            octaves=3)
            r2 = (1 - np.clip(np.abs(n2) * 2.2, 0, 1)) ** (p.sharpness * 1.4)
            g = np.clip(0.5 + 1.2 * self.w.fbm(u0 + 2.0, v0 + 2.0, z, octaves=2), 0, 1)
            return k * (p.veins * (r1 + 0.5 * r2) * (0.6 + 0.8 * g) + p.glow * g)
        ctx.light.field(f, self.tile_box(ctx.light), lowres="nearest")


# =============================================================================
# rain sheet
# =============================================================================
@atom("rain_sheet", tags=["tile", "rain", "weather", "light", "loop", "environment"])
class RainSheet(Atom):
    """Tileable rain: slanted streaks falling through the tile (wrapping around its edges). Each
    drop lives whole laps per period, fades in, falls, fades out and reappears at a new random
    place, so any speed and slant loop seamlessly. Tile-only (see module doc)."""
    kind = "loop"
    outputs = {"light": Light(palette="rain", bloom=[[1.0, 0.25], [3.0, 0.12]], exposure=1.0, alpha_gamma=0.9)}
    streams = ("drops",)

    class Params(_TileParams):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        count: int = P(110, ge=0, doc="number of drops")
        speed: float = P(700.0, gt=0, doc="fall speed of the nearest drops (px/s)")
        speed_var: float = P(0.3, ge=0, le=1, doc="random speed variation (fraction)")
        depth: float = P(0.6, ge=0, le=1, doc="depth spread: far drops are slower, thinner and dimmer")
        wind: float = P(0.15, ge=-2, le=2, doc="slant: sideways px per px of fall")
        length: float = P(0.035, gt=0, doc="streak length (s of travel)")
        thickness: float = P(0.55, gt=0, doc="streak thickness of the nearest drops (px)")
        brightness: float = P(0.85, ge=0, doc="streak intensity")
        travel: float = P(0.8, gt=0, le=4, doc="distance each drop falls per life (fraction of the tile height)")
        fade: float = P(0.2, gt=0, le=0.5, doc="fade-in / fade-out fraction of each life")

    def setup(self):
        p = self.p
        rng = self.rng("drops")
        n = p.count
        d = rng.random(n)
        self.d = d
        spd = p.speed * (1 - 0.55 * p.depth * (1 - d)) * (1 + p.speed_var * (rng.random(n) - 0.5))
        self.spd = np.maximum(spd, 1.0)
        self.laps = np.maximum(1, np.round(self.spd * p.period / (p.travel * p.height))).astype(np.int64)
        L = int(self.laps.max()) if n else 1
        self.x0 = rng.uniform(0, p.width, (n, L))
        self.y0 = rng.uniform(0, p.height, (n, L))
        self.phase = rng.random(n)
        self.bri = (0.3 + 0.7 * d) * (0.75 + 0.5 * rng.random(n))
        self.th = p.thickness * (0.55 + 0.65 * d)
        k = 1 / math.hypot(p.wind, 1)
        self.dir = (p.wind * k, k)

    def _margin(self):
        p = self.p
        return p.speed * 1.2 * p.length + 3 * p.thickness * 1.2 + 4

    def get_bounds(self):
        m = self._margin()
        return (-m, -m, float(self.p.width) + m, float(self.p.height) + m)

    def draw(self, ctx, t):
        p = self.p
        if self.d.size == 0:
            return
        j, q, age = lap_clock(t, p.period, self.laps, self.phase)
        env = smoothstep(0, p.fade, q) * smoothstep(1, 1 - p.fade, q)
        dist = self.spd * age
        hx = (pick(self.x0, j) + self.dir[0] * dist) % p.width
        hy = (pick(self.y0, j) + self.dir[1] * dist) % p.height
        ln = self.spd * p.length
        for i in np.flatnonzero(env > 1e-3):
            b = p.brightness * float(self.bri[i] * env[i])
            x, y, L = float(hx[i]), float(hy[i]), float(ln[i])
            th = float(self.th[i])
            ctx.light.polyline([(x - self.dir[0] * L, y - self.dir[1] * L), (x, y)], th * 0.45, b * 0.1,
                               r_end=th, a_end=b)
