"""Arcane atoms: orb shells, orbiting motes, rune bands, tick bands, star polygons,
star glints, the portal vortex and its noisy rim.

Ported from the legacy ``fx_energy`` effects (orb, magic_circle, portal, sparkle,
pickup_burst). Lengths are legacy pixels at scale 1; every rotation / speed that
must loop is a whole number of symmetry steps (or cycles) per period.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, atom
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise


def _ss(e0: float, e1: float, x: float) -> float:
    """Scalar smoothstep."""
    return float(smoothstep(e0, e1, x))


def _ease_out(x: float, p: float = 3.0) -> float:
    x = min(max(x, 0.0), 1.0)
    return 1 - (1 - x) ** p


def _int_list(s: str, what: str) -> list[int]:
    try:
        vals = [int(v) for v in str(s).replace(" ", "").split(",") if v]
    except ValueError:
        raise ValueError(f"{what}: expected comma-separated integers, got {s!r}") from None
    return vals or [1]


def draw_glint(view, x: float, y: float, L: float, a: float, rot: float = 0.0,
               w: float | None = None, diag: float = 0.4) -> None:
    """4-point star glint: tapered cross streaks, shorter diagonals, hot core (legacy ``_glint``)."""
    if a <= 1e-5 or L <= 0:
        return
    w = w or max(L * 0.07, 0.45)
    for k in range(4):
        ang = rot + k * TAU / 4
        view.polyline([(x, y), (x + L * math.cos(ang), y + L * math.sin(ang))], w, a, r_end=w * 0.3, a_end=0.0)
        if diag:
            ang += TAU / 8
            Ld = L * diag
            view.polyline([(x, y), (x + Ld * math.cos(ang), y + Ld * math.sin(ang))],
                          w * 0.8, a * 0.55, r_end=w * 0.3, a_end=0.0)
    view.dot(x, y, max(L * 0.09, 0.6), a * 1.3)
    view.dot(x, y, L * 0.3, a * 0.18)


def _sweep(sweep: float, ang: float, head: float) -> float:
    """Legacy magic-circle per-element sweep brightness."""
    return 0.72 + 0.45 * sweep * max(0.0, math.cos(ang - head)) ** 2


# =============================================================================
# orb
# =============================================================================
@atom("wobble_shells", tags=["light", "orb", "loop"])
class WobbleShells(Atom):
    """Thin energy shells around a centre whose radius and brightness wobble with random
    angular harmonics that spin a whole number of cycles per period (orb shells)."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane", bloom=[[3, .6], [8, .6], [22, .55]])}
    streams = ("shells",)

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        shells: int = P(3, ge=0, le=12, doc="number of shells")
        radius: float = P(52.0, gt=0, doc="radius of the main shell (px); the next two are 1.31x and 0.73x")
        wobble: float = P(1.0, ge=0, doc="shape distortion multiplier (1 = 12 px)")
        width: float = P(2.2, gt=0, doc="shell thickness (px)")
        brightness: float = P(1.0, ge=0, doc="shell brightness multiplier")
        pulse: float = P(0.07, ge=0, le=0.5, doc="radius pulse amplitude (one pulse per period)")
        spin: int = P(1, doc="harmonic spin (cycles per period; 0 = frozen, negative reverses)")

    def setup(self):
        rng = self.rng("shells")
        s = self.p.radius / 52.0

        def harmonics(n):
            return [(int(rng.integers(2, 8)), int(rng.choice([-2, -1, 1, 2])),
                     float(rng.uniform(0, TAU)), float(rng.uniform(0.4, 1.0))) for _ in range(n)]
        base = [(52 * s, harmonics(4), harmonics(2), 1.0), (68 * s, harmonics(4), harmonics(2), 0.65),
                (38 * s, harmonics(3), harmonics(2), 0.5)]
        for _ in range(max(0, self.p.shells - 3)):
            base.append((float(rng.uniform(32, 76)) * s, harmonics(4), harmonics(2), float(rng.uniform(0.4, 0.7))))
        self.shell_list = base[:max(0, self.p.shells)]

    def get_bounds(self):
        rmax = max([r for r, *_ in self.shell_list], default=self.p.radius)
        r = rmax * (1 + self.p.pulse) + 12 * self.p.wobble + 3 * self.p.width + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        if not self.shell_list:
            return
        tn = t / p.period
        ts = tn * p.spin
        pl = 1 + p.pulse * math.sin(TAU * tn)
        wob = 12.0 * p.wobble

        def f(X, Y):
            r = np.hypot(X, Y)
            th = np.arctan2(Y, X)
            out = np.zeros_like(X)
            for R0, shape, bright, strength in self.shell_list:
                n = sum(a * np.cos(m * th + TAU * k * ts + ph) for m, k, ph, a in shape) / sum(a for *_, a in shape)
                b = sum(a * np.cos(m * th - TAU * k * ts + ph) for m, k, ph, a in bright)
                b = np.clip(0.55 + 0.6 * b / sum(a for *_, a in bright), 0, None)
                R = R0 * pl + wob * n
                out += strength * 1.6 * p.brightness * b * b * np.exp(-((r - R) / p.width) ** 2)
            return out
        ctx.light.field(f, self.get_bounds())


@atom("orbiters", tags=["light", "orb", "loop", "particles"])
class Orbiters(Atom):
    """Motes orbiting a centre on (optionally tilted, squashed) ellipses, each with a tapered
    trail. Every mote laps a whole number of times per period (orb motes, magic-circle sparks)."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane")}
    streams = ("motes",)

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        count: int = P(14, ge=0, doc="number of motes")
        radius_min: float = P(80.0, ge=0, doc="smallest orbit radius (px)")
        radius_max: float = P(104.0, ge=0, doc="largest orbit radius (px)")
        squash_min: float = P(0.2, gt=0, le=1, doc="flattest orbit (minor/major axis ratio)")
        squash_max: float = P(0.9, gt=0, le=1, doc="roundest orbit (1 = circle)")
        speeds: str = P("1,1,-1,2", doc="laps per period, picked at random per mote (comma-separated ints)")
        spin: int = P(1, doc="speed multiplier (whole number; 0 = frozen)")
        trail: float = P(0.55, ge=0, doc="trail length (radians of orbit)")
        trail_points: int = P(10, ge=2, doc="points along each trail")
        trail_width: float = P(0.6, gt=0, doc="trail width at its tail (px)")
        head_width: float = P(2.0, gt=0, doc="trail width at the mote (px)")
        head_size: float = P(2.6, gt=0, doc="mote dot radius (px)")
        brightness_min: float = P(0.8, ge=0, doc="dimmest mote")
        brightness_max: float = P(1.6, ge=0, doc="brightest mote")
        trail_brightness: float = P(1.4, ge=0, doc="trail intensity at the mote (x mote brightness)")
        head_brightness: float = P(1.0, ge=0, doc="dot intensity (x mote brightness)")
        depth: float = P(0.6, ge=0, le=1, doc="front/back brightness modulation along the orbit")

    def setup(self):
        rng = self.rng("motes")
        p = self.p
        speeds = _int_list(p.speeds, "speeds")
        lo, hi = min(p.radius_min, p.radius_max), max(p.radius_min, p.radius_max)
        slo, shi = min(p.squash_min, p.squash_max), max(p.squash_min, p.squash_max)
        self.motes = [dict(rho=float(rng.uniform(lo, hi)), phi=float(rng.uniform(0, TAU)),
                           k=int(rng.choice(speeds)), squash=float(rng.uniform(slo, shi)),
                           tilt=float(rng.uniform(0, math.pi)),
                           a=float(rng.uniform(p.brightness_min, max(p.brightness_min, p.brightness_max))))
                      for _ in range(p.count)]

    def get_bounds(self):
        r = max(self.p.radius_min, self.p.radius_max) + 3 * max(self.p.head_size, self.p.head_width) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        ts = t / p.period * p.spin
        for m in self.motes:
            ct, st = math.cos(m["tilt"]), math.sin(m["tilt"])
            head = m["phi"] + TAU * m["k"] * ts
            phis = head - np.sign(m["k"]) * np.linspace(p.trail, 0, p.trail_points)
            ex, ey = m["rho"] * np.cos(phis), m["rho"] * m["squash"] * np.sin(phis)
            pts = np.stack([ct * ex - st * ey, st * ex + ct * ey], 1)
            dep = (1 - p.depth) + p.depth * (0.5 + 0.5 * math.sin(head))
            a = m["a"] * dep
            if m["k"] != 0 and p.trail > 0:
                ctx.light.polyline(pts, p.trail_width, 0.0, r_end=p.head_width, a_end=a * p.trail_brightness)
            ctx.light.dot(float(pts[-1, 0]), float(pts[-1, 1]), p.head_size, a * p.head_brightness)


# =============================================================================
# magic circle pieces
# =============================================================================
def _make_glyph(rng) -> list:
    """A procedural rune: strokes in a unit cell (x in [-.3, .3], y in [-.5, .5])."""
    cols = [-0.3, 0.0, 0.3]
    rows = [-0.5, -1 / 6, 1 / 6, 0.5]
    strokes = []
    if rng.random() < 0.7:
        ci = int(rng.integers(0, 3))
        r0, r1 = (0, 3) if rng.random() < 0.7 else (0, 2)
        strokes.append([(cols[ci], rows[r0]), (cols[ci], rows[r1])])
    for _ in range(int(rng.integers(2, 4))):
        ci, ri = int(rng.integers(0, 3)), int(rng.integers(0, 4))
        pts = [(cols[ci], rows[ri])]
        for _ in range(int(rng.integers(1, 3))):
            for _try in range(8):
                dc, dr = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
                if dr == 0 and dc == 0:
                    continue
                nc, nr = ci + dc * int(rng.integers(1, 3)), ri + dr
                if 0 <= nc < 3 and 0 <= nr < 4:
                    ci, ri = nc, nr
                    pts.append((cols[ci], rows[ri]))
                    break
        if len(pts) > 1:
            strokes.append(pts)
    extra = rng.random()
    if extra < 0.3:
        ox, oy = cols[int(rng.integers(0, 3))], rows[int(rng.integers(1, 3))]
        a = np.linspace(0, TAU, 13)
        strokes.append(list(zip(ox + 0.13 * np.cos(a), oy + 0.13 * np.sin(a))))
    elif extra < 0.55:
        strokes.append([(cols[int(rng.integers(0, 3))], rows[int(rng.integers(0, 4))])])
    return strokes


@atom("rune_ring", tags=["light", "arcane", "loop", "glyphs"])
class RuneRing(Atom):
    """A band of procedural glyphs around a circle (upright = outward). ``glyphs`` unique runes
    repeat ``repeat`` times, so the band turns whole symmetry steps per period. An optional
    highlight sweeps around once per ``sweep_turns``."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane")}
    streams = ("glyphs",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        glyphs: int = P(9, ge=1, doc="unique runes in the band")
        repeat: int = P(4, ge=1, doc="times the rune set repeats around the band")
        radius: float = P(105.5, gt=0, doc="band radius (px)")
        height: float = P(11.0, gt=0, doc="glyph height (px)")
        line_width: float = P(0.62, gt=0, doc="stroke width (px)")
        brightness: float = P(1.15, ge=0, doc="stroke intensity")
        rotation: int = P(-1, doc="symmetry steps (360/repeat deg) turned per period; negative = counter-clockwise")
        sweep: float = P(1.0, ge=0, doc="strength of the per-glyph highlight sweep (0 = off)")
        sweep_turns: int = P(1, doc="highlight laps per period")

    def setup(self):
        rng = self.rng("glyphs")
        self.uniq = []
        H = self.p.height
        for _ in range(self.p.glyphs):
            strokes = []
            for st in _make_glyph(rng):
                st = np.asarray(st, np.float64)
                if len(st) == 1:
                    strokes.append(st)
                    continue
                dense = []
                for a, b in zip(st[:-1], st[1:]):
                    k = max(2, int(np.linalg.norm(b - a) * H / 2.5))
                    dense.append(a + (b - a) * np.linspace(0, 1, k, endpoint=False)[:, None])
                dense.append(st[-1:])
                strokes.append(np.vstack(dense))
            self.uniq.append(strokes)

    def get_bounds(self):
        r = self.p.radius + 0.5 * self.p.height + 3 * 1.5 * self.p.line_width + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        n = p.glyphs * p.repeat
        rot = p.rotation * TAU / p.repeat * tn
        head = TAU * p.sweep_turns * tn
        R, H, lw = p.radius, p.height, p.line_width
        for i in range(n):
            ang0 = rot + i * TAU / n
            aa = p.brightness * (_sweep(p.sweep, ang0, head) if p.sweep > 0 else 1.0)
            for st in self.uniq[i % p.glyphs]:
                ang = ang0 + st[:, 0] * H / R
                rad = R - st[:, 1] * H
                xs, ys = rad * np.cos(ang), rad * np.sin(ang)
                if len(st) == 1:
                    ctx.light.dot(float(xs[0]), float(ys[0]), lw * 1.5, aa * 0.9)
                else:
                    ctx.light.polyline(np.stack([xs, ys], 1), lw, aa)


@atom("ticks", tags=["light", "arcane", "loop", "ring"])
class Ticks(Atom):
    """Radial tick band: ``count`` evenly spaced radial strokes between two radii (dial / magic
    circle scale). Turns ``rotation`` ticks per period."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane")}

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        count: int = P(24, ge=1, doc="number of ticks")
        radius_inner: float = P(118.0, ge=0, doc="inner end of the ticks (px)")
        radius_outer: float = P(130.0, gt=0, doc="outer end of the ticks (px)")
        width: float = P(0.47, gt=0, doc="tick thickness (px)")
        brightness: float = P(1.0, ge=0, doc="intensity")
        rotation: int = P(1, doc="tick steps turned per period (negative reverses)")
        offset: float = P(0.0, doc="angular offset in tick steps (0.5 = halfway between ticks)")

    def get_bounds(self):
        r = self.p.radius_outer + 3 * self.p.width + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        r0, r1 = min(p.radius_inner, p.radius_outer), max(p.radius_inner, p.radius_outer)
        shift = p.rotation * t / p.period + p.offset

        def f(X, Y):
            r = np.hypot(X, Y)
            th = np.arctan2(Y, X)
            q = th * p.count / TAU - shift
            q = q - np.round(q)
            d = q * TAU / p.count * r
            win = smoothstep(r0 - 0.8, r0 + 0.4, r) * smoothstep(r1 + 0.8, r1 - 0.4, r)
            return p.brightness * np.exp(-0.5 * (d / p.width) ** 2) * win
        b = r1 + 2
        ctx.light.field(f, (-b, -b, b, b))


@atom("star_shape", tags=["light", "arcane", "loop", "geometry"])
class StarShape(Atom):
    """Regular polygon (``step`` 1) or star polygon {points/step} drawn as glowing lines, with
    optional rings and dots on the tips. ``step`` sharing a factor with ``points`` gives a
    compound (hexagram = 6/2, 8-point star of two squares = 8/2)."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane")}

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        points: int = P(6, ge=3, doc="number of vertices")
        step: int = P(2, ge=1, doc="vertex step (1 = polygon, 2 = hexagram for 6 points)")
        radius: float = P(92.0, gt=0, doc="tip radius (px)")
        angle: float = P(-90.0, doc="angle of the first tip (deg, clockwise from +x)")
        line_width: float = P(0.65, gt=0, doc="line thickness (px)")
        brightness: float = P(1.15, ge=0, doc="line intensity")
        rotation: int = P(1, doc="symmetry steps (360/points deg) turned per period")
        tip_ring_radius: float = P(6.5, ge=0, doc="radius of the ring on each tip (px; 0 = none)")
        tip_ring_width: float = P(0.62, gt=0, doc="tip ring thickness (px)")
        tip_ring_brightness: float = P(1.0, ge=0, doc="tip ring intensity")
        tip_dot_size: float = P(1.6, ge=0, doc="dot on each tip (px; 0 = none)")
        tip_dot_brightness: float = P(1.0, ge=0, doc="tip dot intensity")
        sweep: float = P(0.0, ge=0, doc="highlight sweep on the tip rings (0 = off)")
        sweep_turns: int = P(1, doc="highlight laps per period")

    def get_bounds(self):
        p = self.p
        r = p.radius + max(p.tip_ring_radius + 3 * p.tip_ring_width, 3 * p.tip_dot_size,
                           3 * p.line_width) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        n = p.points
        rot = math.radians(p.angle) + p.rotation * TAU / n * tn
        angs = [rot + k * TAU / n for k in range(n)]
        tips = [(p.radius * math.cos(a), p.radius * math.sin(a)) for a in angs]
        k = p.step % n or 1
        if p.brightness > 0:
            for i in range(n):          # each edge i -> i+step (legacy draws edges separately)
                a, b = tips[i], tips[(i + k) % n]
                ctx.light.polyline([a, b], p.line_width, p.brightness)
        head = TAU * p.sweep_turns * tn
        for a, (x, y) in zip(angs, tips):
            if p.tip_ring_radius > 0 and p.tip_ring_brightness > 0:
                g = _sweep(p.sweep, a, head) if p.sweep > 0 else 1.0
                ctx.light.ring(x, y, p.tip_ring_radius, p.tip_ring_width, p.tip_ring_brightness * g)
            if p.tip_dot_size > 0:
                ctx.light.dot(x, y, p.tip_dot_size, p.tip_dot_brightness)


# =============================================================================
# glints
# =============================================================================
@atom("glints", tags=["light", "sparkle", "glint"])
class Glints(Atom):
    """4-point star glints (tapered cross streaks, short diagonals, hot core).

    Modes:
    * ``field``: a cluster that twinkles out of sync; each glint is lit on its own window of
      a (k * t + phase) cycle, so the loop is exact (sparkle).
    * ``single``: one glint at the origin that flashes and shrinks over one period.
    * ``burst``: glints fly out on ease-out radial paths while shrinking and spinning, over
      one period (pickup). Use ``time_mode: once`` on the layer for the one-shot modes."""
    kind = "loop"
    outputs = {"light": Light(palette="frost", bloom=[[1.2, .5], [4, .4], [10, .3]])}
    streams = ("glints",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (field) or animation length (single / burst) (s)")
        mode: str = P("field", doc="field | single | burst", choices=["field", "single", "burst"])
        count: int = P(14, ge=0, doc="number of glints (field / burst)")
        big: int = P(4, ge=0, doc="field: how many are large hero glints")
        spread: float = P(82.0, ge=0, doc="field: cluster radius; burst: furthest travel (px)")
        size: float = P(1.0, gt=0, doc="glint size multiplier")
        lit: float = P(1.0, gt=0, doc="field: how long each glint stays lit (multiplier)")
        spin: float = P(1.0, doc="how much glints rotate while lit / flying")
        diagonals: bool = P(True, doc="add the short diagonal rays")
        brightness: float = P(1.0, ge=0, doc="intensity multiplier")
        line_width: float = P(-1.0, doc="ray width (px); -1 = automatic")
        flash: float = P(0.0, ge=0, doc="single / burst: initial flash brightness (decays with `decay`)")
        decay: float = P(9.4, ge=0, doc="single / burst: flash decay rate (per period)")
        sustain: float = P(1.5, ge=0, doc="single / burst: steady brightness, faded out over the end")
        fade_from: float = P(0.55, ge=0, lt=1, doc="single / burst: fraction of the period where the fade-out starts")
        shrink: float = P(0.55, ge=0, le=1, doc="single / burst: fraction of the length lost by the end")
        travel: float = P(0.8, gt=0, doc="burst: fraction of the period spent travelling (ease-out)")

    def setup(self):
        rng = self.rng("glints")
        p = self.p
        if p.mode == "field":
            self.gl = []
            for i in range(p.count):
                rad = math.sqrt(rng.uniform(0, 1)) * p.spread
                ang = rng.uniform(0, TAU)
                is_big = i < p.big
                self.gl.append(dict(x=rad * math.cos(ang), y=rad * math.sin(ang),
                                    L=(rng.uniform(36, 50) if is_big else rng.uniform(12, 24)) * p.size,
                                    ph=(i * 0.618) % 1.0, dur=min(rng.uniform(0.3, 0.5) * p.lit, 1.0),
                                    k=1 if is_big else int(rng.choice([1, 2])),
                                    rot=rng.uniform(-0.25, 0.25), spin=rng.uniform(-0.6, 0.6) * p.spin,
                                    br=rng.uniform(0.9, 1.4)))
        elif p.mode == "burst":
            n = max(p.count, 1)
            self.gl = [dict(ang=k * TAU / n + rng.uniform(-0.25, 0.25),
                            D=rng.uniform(62 / 90, 1.0) * p.spread,
                            L=rng.uniform(15, 22) * p.size, spin=rng.uniform(-1.5, 1.5) * p.spin)
                       for k in range(p.count)]
        else:
            self.gl = []

    def get_bounds(self):
        p = self.p
        if p.mode == "field":
            r = p.spread + 50 * p.size + 4
        elif p.mode == "burst":
            r = p.spread + 22 * p.size + 4
        else:
            r = 46 * p.size + 4
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        diag = 0.4 if p.diagonals else 0.0
        if p.mode == "field":
            tn = t / p.period
            for g in self.gl:
                s = ((g["k"] * tn + g["ph"]) % 1.0) / g["dur"]
                if s >= 1:
                    continue
                e = math.sin(math.pi * s) ** 2
                w = p.line_width if p.line_width > 0 else max(0.55, g["L"] * 0.045)
                draw_glint(ctx.light, g["x"], g["y"], g["L"] * (0.35 + 0.65 * e), p.brightness * g["br"] * e * 1.3,
                           rot=g["rot"] + g["spin"] * (s - 0.5), w=w, diag=diag)
            return
        u = t / p.period
        if u >= 1:
            return
        fade = _ss(1.0, p.fade_from, u)
        a = p.brightness * (p.flash * math.exp(-u * p.decay) + p.sustain * fade)
        if p.mode == "single":
            w = p.line_width if p.line_width > 0 else 1.6 * p.size
            draw_glint(ctx.light, 0.0, 0.0, 46 * p.size * (1 - p.shrink * u), a, rot=0.0, w=w, diag=diag)
            return
        e = _ease_out(u / p.travel, 3)
        w = p.line_width if p.line_width > 0 else 0.9 * p.size
        for s in self.gl:
            d = s["D"] * e
            draw_glint(ctx.light, d * math.cos(s["ang"]), d * math.sin(s["ang"]), s["L"] * (1 - p.shrink * u), a,
                       rot=s["spin"] * u, w=w, diag=diag)


# =============================================================================
# portal
# =============================================================================
@atom("vortex", tags=["light", "smoke", "portal", "loop"])
class Vortex(Atom):
    """Portal body: domain-warped fbm in log-polar coordinates with spiral arms that flow
    inward, ragged licks just outside the rim, particles sucked in along the same log-spiral,
    and the dark see-through eye (smoke role). Ellipse radii ``radius_x`` / ``radius_y``."""
    kind = "loop"
    outputs = {"light": Light(palette="void", bloom=[[1.5, .45], [5, .4], [14, .35]], exposure=0.9),
               "smoke": Smoke(palette="abyss", density=0.55, shadow=0.3)}
    streams = ("noise", "particles")

    class Params(Atom.Params):
        period: float = P(1.6, gt=0, doc="loop length (s)")
        radius_x: float = P(116.0, gt=0, doc="horizontal radius (px)")
        radius_y: float = P(148.0, gt=0, doc="vertical radius (px)")
        arms: int = P(3, ge=0, doc="number of spiral arms")
        twist: float = P(1.15, doc="how tightly the arms wind")
        inflow_speed: int = P(1, doc="inward flow (noise periods per loop; 0 = none)")
        rotation_speed: int = P(0, doc="extra spin (turns per loop; negative reverses)")
        warp: float = P(1.0, ge=0, doc="domain-warp turbulence")
        brightness: float = P(1.0, ge=0, doc="body brightness")
        licks: float = P(1.0, ge=0, doc="brightness of the ragged energy licking outside the rim")
        particles: int = P(46, ge=0, doc="particles sucked in")
        particle_brightness: float = P(1.0, ge=0, doc="particle intensity multiplier")
        eye: float = P(1.6, ge=0, doc="peak density of the dark eye (smoke role; 0 = none)")

    NX, NY, NZ = 12, 8, 4
    C = 3.6

    def setup(self):
        rng = self.rng("noise")
        per = (self.NX, self.NY, self.NZ)
        self.noise = Noise(rng, period=per)
        self.wnoise = Noise(rng, period=per)
        pr = self.rng("particles")
        self.parts = [dict(L0=math.log(pr.uniform(0.98, 1.16)), th0=pr.uniform(0, TAU), ph=pr.uniform(0, 1),
                           k=int(pr.choice([1, 1, 2])), K=pr.uniform(2.0, 2.6), a=pr.uniform(0.6, 1.4))
                      for _ in range(self.p.particles)]

    def get_bounds(self):
        rx, ry = self.p.radius_x * 1.2 + 3, self.p.radius_y * 1.2 + 3
        return (-rx, -ry, rx, ry)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        NX, NY, NZ, C, TW = self.NX, self.NY, self.NZ, self.C, p.twist
        z = tn * NZ
        rx, ry = p.radius_x, p.radius_y

        def body(X, Y):
            ex, ey = X / rx, Y / ry
            rho = np.hypot(ex, ey)
            th = np.arctan2(ey, ex)
            L = np.log(rho + 0.02)
            bx = th / TAU * NX + TW * C * L - NX * p.rotation_speed * tn
            by = C * L + NY * p.inflow_speed * tn
            env = smoothstep(0.06, 0.6, rho) * smoothstep(1.13, 0.94, rho) * (0.25 + 0.75 * np.clip(rho, 0, 1) ** 1.6)
            wx = self.wnoise.fbm(bx, by, z, octaves=2)
            wy = self.wnoise.fbm(bx + 3.0, by + 5.0, z, octaves=2)
            n = self.noise.fbm(bx + 0.9 * p.warp * wx, by + 0.7 * p.warp * wy, z, octaves=4)
            ridge = (1 - np.abs(n)) ** 6
            bd = smoothstep(0.0, 0.75, n)
            arm = 0.5 + 0.5 * np.cos(TAU * (p.arms * bx / NX + by / NY) + 2.2 * p.warp * wx)
            I = p.brightness * env * (0.3 + 0.7 * arm ** 1.5) * (1.7 * ridge + 1.1 * bd * bd + 0.12)
            I = I + p.licks * np.exp(-np.maximum(rho - 1, 0) / 0.035) * smoothstep(0.93, 1.0, rho) \
                * (0.15 + 0.9 * bd) * (0.5 + 0.5 * arm)
            return I
        ctx.light.field(body, self.get_bounds(), lowres="bilinear")

        if p.eye > 0 and ctx.has("smoke"):
            def eye(X, Y):
                rho = np.hypot(X / rx, Y / ry)
                return p.eye * smoothstep(0.85, 0.05, rho) ** 1.5
            ctx.smoke.field(eye, (-0.86 * rx, -0.86 * ry, 0.86 * rx, 0.86 * ry))

        dth_dL = -TAU * TW * C / NX
        spin = TAU * p.rotation_speed * tn
        for q in self.parts:
            s = (q["k"] * tn + q["ph"]) % 1.0
            ss = np.linspace(max(s - 0.07, 0), s, 9)
            Ls = q["L0"] - q["K"] * ss
            rr = np.exp(Ls)
            ths = q["th0"] + dth_dL * (Ls - q["L0"]) + spin
            fade = _ss(0.0, 0.12, s) * _ss(0.12, 0.4, float(rr[-1]))
            if fade <= 0.01:
                continue
            pts = np.stack([rr * rx * np.cos(ths), rr * ry * np.sin(ths)], 1)
            a = q["a"] * fade * p.particle_brightness
            ctx.light.polyline(pts, 0.25, 0.0, r_end=0.95, a_end=a * 1.3)
            ctx.light.dot(float(pts[-1, 0]), float(pts[-1, 1]), 1.2, a)


@atom("noisy_rim", tags=["light", "portal", "loop", "ring"])
class NoisyRim(Atom):
    """Thin bright elliptical rim plus a soft inner halo, both modulated by angular fbm noise
    that boils a whole number of noise periods per loop (portal rim)."""
    kind = "loop"
    outputs = {"light": Light(palette="void")}
    streams = ("noise",)

    class Params(Atom.Params):
        period: float = P(1.6, gt=0, doc="loop length (s)")
        radius_x: float = P(116.0, gt=0, doc="horizontal radius (px)")
        radius_y: float = P(148.0, gt=0, doc="vertical radius (px)")
        brightness: float = P(1.0, ge=0, doc="overall rim brightness")
        line: float = P(1.5, ge=0, doc="thin line intensity")
        line_noise: float = P(1.2, ge=0, doc="noise modulation of the line")
        line_width: float = P(0.011, gt=0, doc="line thickness (fraction of the radius)")
        halo: float = P(0.45, ge=0, doc="soft halo intensity")
        halo_noise: float = P(0.35, ge=0, doc="noise modulation of the halo")
        halo_width: float = P(0.045, gt=0, doc="halo thickness (fraction of the radius)")
        halo_offset: float = P(-0.015, doc="halo centre offset from the rim (fraction of the radius)")
        noise_scale: int = P(24, ge=1, doc="noise cells around the rim")
        boil: int = P(4, ge=0, doc="noise time cycles per period")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(self.p.noise_scale, 4, max(self.p.boil, 1)))
        self.ang = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)

    def get_bounds(self):
        p = self.p
        k = 1 + max(4 * p.line_width, p.halo_offset + 4 * p.halo_width) + 0.01
        return (-p.radius_x * k - 2, -p.radius_y * k - 2, p.radius_x * k + 2, p.radius_y * k + 2)

    def draw(self, ctx, t):
        p = self.p
        z = t / p.period * p.boil
        rn1 = self.noise.fbm(self.ang / TAU * p.noise_scale, 0.0, z, octaves=3)

        def f(X, Y):
            ex, ey = X / p.radius_x, Y / p.radius_y
            rho = np.hypot(ex, ey)
            rn = np.interp(np.arctan2(ey, ex), self.ang, rn1)
            v = (p.line + p.line_noise * rn) * np.exp(-0.5 * ((rho - 1.0) / p.line_width) ** 2) \
                + (p.halo + p.halo_noise * rn) * np.exp(-0.5 * ((rho - 1 - p.halo_offset) / p.halo_width) ** 2)
            return p.brightness * np.clip(v, 0, None)
        ctx.light.field(f, self.get_bounds())
