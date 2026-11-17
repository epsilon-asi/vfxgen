"""Status-effect helper atoms: stroke / filled symbols that drift (sleep Zs, buff arrows, blood
drops) and icons orbiting over a head (stun stars).

Symbols are painted as solid matter (fill colour + darker outline, so they read on light and
dark backgrounds) with an optional soft light glow. Colours come from ``glyph_palette``: the
fill is the ramp colour at 0.4 (a single hex colour is used as-is), the outline a darkened
ramp start.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Solid, atom
from ..engine.color import color_at
from ..engine.mathx import TAU, smoothstep

SYMBOLS = ["z", "star", "drop", "skull", "plus", "arrow_up", "arrow_down", "heart", "cross", "spark"]


def _star(n=5, ro=0.5, ri=0.22, rot=-90.0):
    return [(ro if i % 2 == 0 else ri) * np.array([math.cos(math.radians(rot + i * 180 / n)),
                                                   math.sin(math.radians(rot + i * 180 / n))]) for i in range(2 * n)]


def _circle(cx, cy, r, n=20, ry=None):
    ry = r if ry is None else ry
    return [(cx + r * math.cos(TAU * i / n), cy + ry * math.sin(TAU * i / n)) for i in range(n)]


def _drop():
    pts = [(0.0, -0.5)]
    for a in np.linspace(-25, 205, 18):
        r = math.radians(a)
        pts.append((0.3 * math.cos(r), 0.17 + 0.3 * math.sin(r)))
    return pts


def _heart():
    pts = []
    for s in np.linspace(0, TAU, 40, endpoint=False):
        x = 16 * math.sin(s) ** 3
        y = 13 * math.cos(s) - 5 * math.cos(2 * s) - 2 * math.cos(3 * s) - math.cos(4 * s)
        pts.append((x / 34, -y / 34 + 0.04))
    return pts


_ARROW = [(0, -0.5), (0.38, -0.08), (0.15, -0.08), (0.15, 0.46), (-0.15, 0.46), (-0.15, -0.08), (-0.38, -0.08)]

# symbol -> list of (kind, points, part): kind "poly" (filled) | "line" (stroke, width in units);
# part "fill" | "line" (outline colour, painted on top: eyes, details)
SHAPES: dict[str, list] = {
    "z": [("line", [(-0.3, -0.36), (0.3, -0.36), (-0.3, 0.36), (0.3, 0.36)], "fill", 0.15)],
    "plus": [("poly", [(-0.13, -0.45), (0.13, -0.45), (0.13, -0.13), (0.45, -0.13), (0.45, 0.13), (0.13, 0.13),
                       (0.13, 0.45), (-0.13, 0.45), (-0.13, 0.13), (-0.45, 0.13), (-0.45, -0.13), (-0.13, -0.13)], "fill", 0)],
    "cross": [("line", [(-0.32, -0.32), (0.32, 0.32)], "fill", 0.17), ("line", [(0.32, -0.32), (-0.32, 0.32)], "fill", 0.17)],
    "arrow_up": [("poly", _ARROW, "fill", 0)],
    "arrow_down": [("poly", [(x, -y) for x, y in _ARROW], "fill", 0)],
    "star": [("poly", _star(), "fill", 0)],
    "spark": [("poly", _star(4, 0.5, 0.12, -90), "fill", 0)],
    "drop": [("poly", _drop(), "fill", 0), ("line", [(-0.13, 0.12), (-0.1, 0.27)], "hi", 0.07)],
    "heart": [("poly", _heart(), "fill", 0), ("line", [(-0.25, -0.15), (-0.17, -0.25)], "hi", 0.07)],
    "skull": [("poly", _circle(0, -0.1, 0.4, 24, 0.36), "fill", 0),
              ("poly", [(-0.22, 0.12), (0.22, 0.12), (0.2, 0.44), (-0.2, 0.44)], "fill", 0),
              ("poly", _circle(-0.15, -0.06, 0.1, 12), "line", 0),
              ("poly", _circle(0.15, -0.06, 0.1, 12), "line", 0),
              ("poly", [(0, 0.08), (0.05, 0.18), (-0.05, 0.18)], "line", 0),
              ("line", [(-0.08, 0.3), (-0.08, 0.42)], "line", 0.04),
              ("line", [(0.08, 0.3), (0.08, 0.42)], "line", 0.04)],
}


def _colors(palette):
    fill = color_at(palette, 0.4)
    deep = np.asarray(color_at(palette, 0.0)) * 0.42
    hi = color_at(palette, 0.9)
    return fill, tuple(float(v) for v in deep), hi


def draw_symbol(ctx, symbol: str, x: float, y: float, size: float, rot: float, a: float, cols,
                outline: float = 1.2, glow: float = 0.0, solid: bool = True) -> None:
    """Paint ``symbol`` centred at (x, y), ``size`` px tall, rotated ``rot`` degrees."""
    if a <= 2e-3 or size <= 0.3:
        return
    fill, deep, hi = cols
    r = math.radians(rot)
    cs, sn = math.cos(r), math.sin(r)

    def tf(pts):
        P = np.asarray(pts, np.float64) * size
        return np.stack([x + cs * P[:, 0] - sn * P[:, 1], y + sn * P[:, 0] + cs * P[:, 1]], 1)
    parts = SHAPES[symbol]
    if glow > 0 and ctx.has("light"):
        ctx.light.dot(x, y, size * 0.42, glow * a * 0.45)
        for kind, pts, part, w in parts:
            if part != "fill":
                continue
            q = tf(pts)
            if kind == "poly":
                q = np.vstack([q, q[:1]])
            ctx.light.polyline(q, max(size * 0.07, 0.6), glow * a)
    if not solid:
        return
    sol = ctx.solid
    # outline pass (all fill parts), then fills, then details
    for kind, pts, part, w in parts:
        if part != "fill":
            continue
        q = tf(pts)
        if kind == "poly":
            sol.stroke(np.vstack([q, q[:1]]), outline, rgb=deep, a=a)
            sol.polygon(q, rgb=deep, a=a)
        else:
            sol.stroke(q, w * size / 2 + outline, rgb=deep, a=a)
    for kind, pts, part, w in parts:
        q = tf(pts)
        col = fill if part == "fill" else (deep if part == "line" else hi)
        if kind == "poly":
            sol.polygon(q, rgb=col, a=a, soft=0.5)
        else:
            sol.stroke(q, max(w * size / 2, 0.45), rgb=col, a=a)
    if symbol in ("star", "spark", "plus", "arrow_up", "arrow_down", "z", "cross"):
        # small highlight toward the light (upper left)
        sol.capsule((x - size * 0.12, y - size * 0.18), (x - size * 0.05, y - size * 0.26),
                    max(size * 0.045, 0.4), max(size * 0.03, 0.3), rgb=hi, a=a * 0.85)


@atom("glyph", tags=["status", "symbol", "solid", "light", "loop"])
class Glyph(Atom):
    """Status symbols that spawn, drift, grow and fade in a seamless loop: sleep ``z``,
    stun ``star``, blood / poison ``drop``, ``skull``, ``plus`` / ``arrow_up`` buffs,
    ``arrow_down`` debuffs, ``heart``, ``cross``, ``spark``.

    Each symbol lives ``laps`` times per period. It spawns inside a ``spread_x`` x ``spread_y``
    box around the origin, moves ``rise`` px up (negative = falls) along a curve with
    ``accel`` (1 = steady, 2 = accelerating like a falling drop), drifts sideways, sways and
    wobbles, grows from ``size_start`` to ``size_end``. Painted solid (fill + dark outline)
    with an optional light glow."""
    kind = "loop"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="#9fd0ff", bloom=[[2, 0.45], [6, 0.3]], alpha_gamma=0.85)}
    streams = ("glyphs",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        symbol: str = P("z", doc="symbol", choices=SYMBOLS)
        count: int = P(3, ge=0, le=200, doc="symbols alive per period")
        laps: int = P(1, ge=1, doc="lifetimes per period (whole number)")
        phases: str = P("even", doc="spawn timing: evenly staggered or random", choices=["even", "random"])
        spread_x: float = P(6.0, ge=0, doc="spawn box half-width (px)")
        spread_y: float = P(4.0, ge=0, doc="spawn box half-height (px)")
        rise: float = P(60.0, doc="px travelled upward over a life (negative = downward)")
        accel: float = P(1.0, gt=0, doc="motion curve exponent (1 = steady, 2 = accelerating)")
        drift: float = P(22.0, doc="sideways drift over a life (px, + = right)")
        drift_var: float = P(0.3, ge=0, doc="random drift variation (fraction)")
        sway: float = P(4.0, ge=0, doc="sideways sway amplitude (px)")
        sway_cycles: int = P(1, ge=0, doc="sway cycles per life (whole number)")
        size_start: float = P(8.0, ge=0, doc="symbol height at birth (px)")
        size_end: float = P(18.0, ge=0, doc="symbol height at end of life (px)")
        size_var: float = P(0.15, ge=0, le=1, doc="random size variation")
        tilt: float = P(-10.0, doc="base rotation (deg)")
        wobble: float = P(10.0, ge=0, doc="rotation wobble amplitude (deg)")
        spin: int = P(0, doc="whole turns per life (stars)")
        fade_in: float = P(0.15, ge=0, le=1, doc="fade-in fraction of life")
        fade_out: float = P(0.35, ge=0, le=1, doc="fade-out fraction at the end of life")
        outline: float = P(1.2, ge=0, doc="dark outline width (px)")
        glow: float = P(0.6, ge=0, doc="light glow brightness (0 = off)")
        solid: bool = P(True, doc="paint the symbol (false = glow only)")
        glyph_palette: str = P("#dff1ff", doc="symbol colour (fill at 0.4, outline from the dark end)")

    def setup(self):
        p = self.p
        rng = self.rng("glyphs")
        n = int(p.count)
        L = int(p.laps)
        r0 = rng.random(n)
        self.ph = np.arange(n) / max(n, 1) if p.phases == "even" else r0
        self.x0 = (2 * rng.random((n, L)) - 1) * p.spread_x
        self.y0 = (2 * rng.random((n, L)) - 1) * p.spread_y
        self.dr = 1 + p.drift_var * (2 * rng.random((n, L)) - 1)
        self.sv = 1 + p.size_var * (2 * rng.random(n) - 1)
        self.r = rng.random((n, 2))
        self.cols = _colors(p.glyph_palette)

    def get_bounds(self):
        p = self.p
        s = max(p.size_start, p.size_end) * (1 + p.size_var) + p.outline + 6
        dx = abs(p.drift) * (1 + p.drift_var) + p.sway + p.spread_x + s
        up = max(p.rise, 0) + p.spread_y + s
        dn = max(-p.rise, 0) + p.spread_y + s
        return (-dx, -up, dx, dn)

    def draw(self, ctx, t):
        p = self.p
        L = int(p.laps)
        ph = t / p.period
        for i in range(len(self.ph)):
            x_ = ph * L + self.ph[i]
            q = x_ % 1.0
            lap = int(math.floor(x_)) % L
            a = float(smoothstep(0, max(p.fade_in, 1e-4), q) * smoothstep(1, 1 - max(p.fade_out, 1e-4), q))
            if a <= 2e-3:
                continue
            x = self.x0[i, lap] + p.drift * self.dr[i, lap] * q + p.sway * math.sin(TAU * (p.sway_cycles * q + self.r[i, 0]))
            y = self.y0[i, lap] - p.rise * q ** p.accel
            size = (p.size_start + (p.size_end - p.size_start) * q) * self.sv[i]
            rot = p.tilt + p.wobble * math.sin(TAU * (q + self.r[i, 1])) + 360.0 * p.spin * q
            draw_symbol(ctx, p.symbol, float(x), float(y), float(size), rot, a, self.cols, p.outline, p.glow, p.solid)


@atom("orbit_icons", tags=["status", "stun", "orbit", "solid", "light", "loop"])
class OrbitIcons(Atom):
    """Icons circling on a tilted ellipse (stun stars / birds over a head, orbiting charms).

    The origin is the orbit centre. Icons on the near side (lower half of the ellipse) are
    bigger and brighter and are painted in front; each leaves a short light trail. Icons are
    ``symbol`` glyphs, or an image asset when ``image`` is set."""
    kind = "loop"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="holy", bloom=[[2, 0.45], [6, 0.3]], alpha_gamma=0.85)}
    streams = ("icons",)

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        symbol: str = P("star", doc="symbol (when no image)", choices=SYMBOLS)
        image: str = P("", doc="image asset to draw instead of the symbol ('' = symbol)")
        count: int = P(3, ge=1, le=24, doc="number of icons")
        radius_x: float = P(26.0, ge=0, doc="orbit half-width (px)")
        radius_y: float = P(8.0, ge=0, doc="orbit half-height (px)")
        tilt: float = P(0.0, doc="orbit tilt (deg)")
        laps: int = P(1, doc="revolutions per period (whole number; negative = reverse)")
        size: float = P(11.0, gt=0, doc="icon height (px)")
        depth: float = P(0.3, ge=0, le=0.9, doc="near / far size and brightness difference")
        spin: int = P(1, doc="icon self-rotations per period (whole number)")
        bob: float = P(2.0, ge=0, doc="vertical bob (px)")
        trail: float = P(0.8, ge=0, doc="light trail brightness (0 = off)")
        trail_length: float = P(0.9, ge=0, le=3, doc="trail length (radians of orbit)")
        outline: float = P(1.1, ge=0, doc="dark outline width (px)")
        glow: float = P(0.6, ge=0, doc="glow brightness around each icon")
        glyph_palette: str = P("#ffd84a", doc="icon colour")

    def setup(self):
        p = self.p
        rng = self.rng("icons")
        self.img = self.asset(p.image) if p.image else None
        self.cols = _colors(p.glyph_palette)
        self.ph = np.arange(p.count) / p.count + rng.uniform(-0.04, 0.04, p.count)
        self.r = rng.random(p.count)

    def get_bounds(self):
        p = self.p
        s = p.size * (1 + p.depth) + p.outline + 8
        R = max(p.radius_x, p.radius_y)
        return (-R - s, -R - s - p.bob, R + s, R + s + p.bob)

    def _pos(self, ang):
        p = self.p
        x, y = p.radius_x * math.cos(ang), p.radius_y * math.sin(ang)
        tr = math.radians(p.tilt)
        return x * math.cos(tr) - y * math.sin(tr), x * math.sin(tr) + y * math.cos(tr)

    def draw(self, ctx, t):
        p = self.p
        ph = t / p.period
        items = []
        for i in range(p.count):
            ang = TAU * (p.laps * ph + self.ph[i])
            near = math.sin(ang)                      # +1 = front (lower on screen)
            items.append((near, i, ang))
        items.sort()
        sgn = 1 if p.laps >= 0 else -1
        for near, i, ang in items:
            k = 1 + p.depth * near
            x, y = self._pos(ang)
            y += p.bob * math.sin(TAU * (2 * ph + self.r[i]))
            a = float(min(1.0, 0.55 + 0.45 * k))
            if p.trail > 0:
                pts = [self._pos(ang - sgn * p.trail_length * s) for s in np.linspace(0, 1, 8)]
                pts = [(px, py + p.bob * math.sin(TAU * (2 * ph + self.r[i]))) for px, py in pts]
                ctx.light.polyline(pts, 1.4 * k, p.trail * k, r_end=0.4, a_end=0.0)
            size = p.size * k
            rot = 360.0 * p.spin * ph + 360 * self.r[i]
            if self.img is not None:
                ih, iw = self.img.shape[:2]
                if p.glow > 0:
                    ctx.light.dot(x, y, size * 0.45, p.glow * 0.5 * a)
                ctx.solid.image(self.img, x, y, w=size * iw / ih, h=size, rot=rot, a=a)
            else:
                draw_symbol(ctx, p.symbol, x, y, size, rot, a, self.cols, p.outline, p.glow)


@atom("ice_crystals", tags=["status", "ice", "freeze", "solid", "light", "loop"])
class IceCrystals(Atom):
    """A crust of solid ice spikes around a character's feet (freeze / root status): spikes
    stand on a ground ellipse around the origin, leaning outward, big ones at the back, with
    twinkling tips. Optional ``float_count`` small shards hovering around the body that bob and
    slowly turn."""
    kind = "loop"
    outputs = {"solid": Solid(palette="ice"),
               "light": Light(palette="frost", bloom=[[2, 0.45], [6, 0.3]], alpha_gamma=0.85)}
    streams = ("crystals",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        count: int = P(16, ge=0, le=200, doc="ground spikes")
        radius_x: float = P(30.0, ge=0, doc="ground ellipse half-width (px)")
        radius_y: float = P(8.0, ge=0, doc="ground ellipse half-height (px)")
        size_min: float = P(10.0, gt=0, doc="smallest spike length (px)")
        size_max: float = P(26.0, gt=0, doc="largest spike length (px)")
        lean: float = P(30.0, ge=0, le=80, doc="outward lean of the outer spikes (deg)")
        width: float = P(0.26, gt=0, le=0.6, doc="spike half-width (fraction of length)")
        float_count: int = P(0, ge=0, le=60, doc="small shards hovering around the body")
        float_height: float = P(100.0, ge=0, doc="height of the hovering band above the feet (px)")
        twinkle: float = P(1.0, ge=0, doc="tip twinkle brightness")
        ice_palette: str = P("ice", doc="ice colours (shadow -> lit)")

    def setup(self):
        p = self.p
        rng = self.rng("crystals")
        n = int(p.count)
        u = (np.arange(n) + rng.uniform(-0.35, 0.35, n)) / max(n, 1)
        th = TAU * u
        rr = 0.55 + 0.45 * np.sqrt(rng.random(n))
        x = np.cos(th) * p.radius_x * rr
        y = np.sin(th) * p.radius_y * rr
        L = p.size_min + (p.size_max - p.size_min) * rng.random(n) ** 1.3
        L = L * (1.15 - 0.3 * (np.sin(th) + 1) / 2)              # front spikes shorter
        side = np.cos(th)
        ang = np.radians(-90 + p.lean * side + rng.normal(0, 8, n))
        self.sp = sorted([(float(y[i]), float(x[i]), float(ang[i]), float(L[i]), float(rng.random())) for i in range(n)])
        m = int(p.float_count)
        self.fl = rng.random((m, 6))
        self.cols = (tuple(float(v) for v in np.asarray(color_at(p.ice_palette, 0.0)) * 0.55),
                     color_at(p.ice_palette, 0.55), color_at(p.ice_palette, 1.0))

    def get_bounds(self):
        p = self.p
        s = p.size_max * 1.2 + 6
        return (-p.radius_x - s - 30, -max(p.size_max * 1.2, p.float_height + 30) - 10,
                p.radius_x + s + 30, p.radius_y + 12)

    def draw(self, ctx, t):
        from .sprites import ice_spike
        p = self.p
        ph = t / p.period
        for (y, x, ang, L, r) in self.sp:
            tw = p.twinkle * max(0.0, math.sin(TAU * (ph * (1 + int(r * 3)) + r))) ** 8
            ice_spike(ctx, (x, y), ang, L, p.width, self.cols, tw)
        for i, r in enumerate(self.fl):
            a0 = TAU * (ph + r[0])
            x = math.cos(a0) * (p.radius_x + 6 + 10 * r[1])
            y = -p.float_height * (0.2 + 0.75 * r[2]) + 4 * math.sin(TAU * (2 * ph + r[3]))
            ang = TAU * (ph * (1 if r[4] > 0.5 else -1) + r[5])
            front = math.sin(a0)
            L = (5 + 5 * r[1]) * (1 + 0.2 * front)
            tw = p.twinkle * max(0.0, math.sin(TAU * (2 * ph + r[3]))) ** 10
            ice_spike(ctx, (x, y), ang, L, 0.3, self.cols, tw)
