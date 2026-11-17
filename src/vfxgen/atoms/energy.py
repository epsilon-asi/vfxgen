"""Energy atoms: lightning bolt, electric crackle, beam (line, emitter flare, impact splash),
shield bubble, light columns, anamorphic streak and charge-up inflow.

Ported from the legacy ``fx_energy`` effects (lightning, electric_crackle, beam, shield,
heal_aura, holy_pillar, charge_up). Lengths are legacy pixels at scale 1. Frame-indexed
legacy logic is expressed in seconds; per-frame randomness uses ``tick_rng`` at
``tick_rate`` Hz.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, atom
from ..engine.mathx import TAU, refine, smoothstep
from ..engine.noise import Noise


def _ss(e0: float, e1: float, x: float) -> float:
    return float(smoothstep(e0, e1, x))


def _ease_out(x, p: float = 3.0):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** p


def _end_fade(t: float, duration: float) -> float:
    return _ss(duration, duration * 0.8, t)


def _beam_flicker(amount: float, tn: float) -> float:
    """Deterministic beam flicker shared by beam_line and splash_fan (tn = t / period)."""
    return 1 + amount * (0.06 * math.sin(TAU * 3 * tn) + 0.04 * math.sin(TAU * 5 * tn + 1.3))


# =============================================================================
# lightning
# =============================================================================
_RESTRIKE = [0.8, 1.0, 0.45, 0.95, 0.7, 0.5, 0.35, 0.22, 0.12, 0.05]
_SMOOTH = [1.0, 0.85, 0.7, 0.55, 0.42, 0.3, 0.2, 0.12, 0.06, 0.02]


@atom("bolt", tags=["light", "lightning", "electric", "oneshot"])
class Bolt(Atom):
    """Lightning strike from ``top`` (local origin) down to ``impact``: a midpoint-displaced
    coarse skeleton with recursive branches rooted on it, re-jagged ``tick_rate`` times per
    second, riding a strike / flicker / re-strike envelope, plus an impact flare."""
    kind = "oneshot"
    outputs = {"light": Light(palette="electric", bloom=[[3, .7], [10, .9], [32, .9]])}
    streams = ("skeleton", "branches", "flicker")
    anchors = {"top": lambda self, t: (float(self.main[0, 0]), float(self.main[0, 1])),
               "impact": lambda self, t: (float(self.main[-1, 0]), float(self.main[-1, 1]))}

    class Params(Atom.Params):
        duration: float = P(0.625, gt=0, doc="seconds until gone")
        length: float = P(448.0, gt=0, doc="vertical distance from top to impact (px)")
        lateral: float = P(60.0, ge=0, doc="max sideways wander of the main channel (px)")
        width: float = P(1.0, gt=0, doc="stroke width multiplier")
        brightness: float = P(1.0, ge=0, doc="stroke brightness multiplier")
        roughness: float = P(0.24, ge=0, doc="jaggedness of the bolt's overall shape")
        jitter: float = P(0.21, ge=0, doc="per-tick crackle of the path")
        branches: int = P(-1, ge=-1, doc="number of side branches (-1 = random 4-6)")
        sub_branches: bool = P(True, doc="allow branches to fork again")
        sway: float = P(0.08, ge=0, le=0.5, doc="random sideways offset of top/impact (fraction of 256 px)")
        impact: float = P(1.0, ge=0, doc="brightness of the impact flare (0 = off)")
        restrike: bool = P(True, doc="flicker + re-strike envelope (false = single smooth fade)")
        tick_rate: float = P(16.0, gt=0, doc="re-jag / flicker changes per second")

    def setup(self):
        p = self.p
        rng = self.rng("skeleton")
        s = 2.0
        H = p.length + 64
        sw = p.sway * 256
        start = np.array([rng.uniform(-sw, sw) if sw else 0.0, 0.0])
        end = np.array([rng.uniform(-sw, sw) if sw else 0.0, p.length])
        main = refine(rng, [start, end], rough=p.roughness, detail=H / 28)
        span = np.abs(main[:, 0]).max()
        if span > p.lateral > 0:
            main[:, 0] *= p.lateral / span
        self.main = main
        bl = [(main, -1, 1.0 * s * p.width, 3.4)]
        br = self.rng("branches")
        b_rough = 0.26 * p.roughness / 0.24
        xl = p.lateral + 8

        def grow(parent_idx, path, depth, count):
            n = len(path)
            for _ in range(count):
                i = int(br.integers(max(1, int(n * 0.12)), max(2, int(n * 0.85))))
                q0 = path[i]
                d = path[min(i + 2, n - 1)] - path[max(i - 1, 0)]
                d = d / (np.linalg.norm(d) + 1e-9)
                ang = br.choice([-1, 1]) * br.uniform(0.4, 1.0)
                ca, sa = math.cos(ang), math.sin(ang)
                dirv = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
                L = br.uniform(0.18, 0.38) * H * (0.5 if depth else 1.0) * (0.4 + 0.6 * (1 - i / n))
                q = q0 + dirv * L
                q[0] = np.clip(q[0], -xl, xl)
                q[1] = np.clip(q[1], 4, p.length - 4)
                bp = refine(br, [q0, q], rough=b_rough, detail=max(L / 6, 5 * s))
                bl.append((bp, parent_idx, (0.72 if depth == 0 else 0.55) * s * p.width, 1.7 if depth == 0 else 0.9))
                if depth == 0:
                    k = int(br.integers(0, 3))
                    grow(len(bl) - 1, bp, 1, k if p.sub_branches else 0)

        n_br = int(br.integers(4, 7))
        grow(0, main, 0, n_br if p.branches < 0 else p.branches)
        self.branch_list = bl
        env = _RESTRIKE if p.restrike else _SMOOTH
        self.env_t = [k * p.duration / len(env) for k in range(len(env))] + [p.duration]
        self.env_v = list(env) + [0.0]

    def get_bounds(self):
        p = self.p
        w = max(p.lateral + 36, 136 if p.impact > 0 else 0)
        return (-w, -16, w, p.length + (110 if p.impact > 0 else 16))

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        e = float(np.interp(t, self.env_t, self.env_v))
        if e <= 1e-4:
            return
        s = 2.0
        fr = self.tick_rng("flicker", t, p.tick_rate)
        xl = p.lateral + 28
        visible = []
        for pts, parent, r, a in self.branch_list:
            ok = parent == -1 or (visible[parent] and fr.random() > 0.2 + 0.3 * (1 - e))
            visible.append(ok)
            if not ok:
                continue
            path = refine(fr, pts, rough=p.jitter, detail=(1.6 if parent == -1 else 1.4) * s)
            path[:, 0] = np.clip(path[:, 0], -xl, xl)
            aa = a * e * p.brightness
            if parent == -1:
                ctx.light.polyline(path, r, aa)
            else:
                ctx.light.polyline(path, r, aa, r_end=r * 0.5, a_end=aa * 0.12)
        if p.impact > 0:
            ex, ey = float(self.main[-1, 0]), float(self.main[-1, 1])
            ctx.light.dot(ex, ey, 7 * s, 1.1 * e * p.impact)
            ctx.light.dot(ex, ey, 18 * s, 0.25 * e * p.impact)


@atom("arc_field", tags=["light", "electric", "loop"])
class ArcField(Atom):
    """Small erratic electric arcs jumping around a point: arc events scheduled on a looping
    timeline (in ticks of ``tick_rate`` Hz), each re-jagged every tick, some forking; plus
    tiny micro-arcs hugging the core, fresh every tick."""
    kind = "loop"
    outputs = {"light": Light(palette="electric", bloom=[[1.2, .6], [4, .7], [12, .6]])}
    streams = ("arcs", "jitter", "micro")

    class Params(Atom.Params):
        period: float = P(1.5, gt=0, doc="loop length (s)")
        tick_rate: float = P(16.0, gt=0, doc="arc re-jag / scheduling steps per second")
        arcs: int = P(40, ge=0, doc="arc events per loop (density)")
        reach: float = P(1.0, gt=0, le=1.25, doc="arc length / radius multiplier")
        roughness: float = P(0.22, ge=0, doc="jaggedness of the arcs")
        hop_chance: float = P(0.4, ge=0, le=1, doc="fraction of arcs hopping around the core instead of shooting out")
        branch_chance: float = P(0.7, ge=0, le=1, doc="chance an arc grows a small fork")
        micro_arcs: int = P(3, ge=0, doc="tiny arcs hugging the core every tick")
        width: float = P(1.0, gt=0, doc="stroke width multiplier")
        brightness: float = P(1.0, ge=0, doc="arc brightness multiplier")

    def setup(self):
        p = self.p
        rng = self.rng("arcs")
        self.n_ticks = max(1, int(round(p.period * p.tick_rate)))
        rch = p.reach
        self.events = []
        for _ in range(p.arcs):
            start = int(rng.integers(0, self.n_ticks))
            dur = int(rng.choice([1, 2, 2, 3]))
            if rng.random() < 1 - p.hop_chance:
                a = rng.uniform(0, TAU)
                L = rng.uniform(45, 92) * rch
                bend = rng.uniform(-0.5, 0.5)
                pts = [(5 * math.cos(a), 5 * math.sin(a)),
                       (L * 0.5 * math.cos(a + bend), L * 0.5 * math.sin(a + bend)),
                       (L * math.cos(a + bend * 0.4), L * math.sin(a + bend * 0.4))]
            else:
                a0 = rng.uniform(0, TAU)
                span = rng.uniform(0.7, 1.7) * rng.choice([-1, 1])
                r0, r1 = rng.uniform(25, 60) * rch, rng.uniform(25, 70) * rch
                aa = np.linspace(a0, a0 + span, 5)
                rr = np.linspace(r0, r1, 5) + np.sin(np.linspace(0, math.pi, 5)) * rng.uniform(5, 18) * rch
                pts = list(zip(rr * np.cos(aa), rr * np.sin(aa)))
            self.events.append(dict(start=start, dur=dur, pts=np.array(pts), br=rng.uniform(0.8, 1.3)))

    def get_bounds(self):
        r = 92 * self.p.reach + 44
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        rate = p.tick_rate
        k = int(math.floor(t * rate + 1e-9)) % self.n_ticks
        wd = p.width
        fr = self.tick_rng("micro", t, rate)
        for _ in range(p.micro_arcs):
            a = fr.uniform(0, TAU)
            L = fr.uniform(14, 30) * p.reach
            b = a + fr.uniform(-0.9, 0.9)
            pts = [(4 * math.cos(a), 4 * math.sin(a)), (L * math.cos(b), L * math.sin(b))]
            ctx.light.polyline(refine(fr, pts, rough=p.roughness * 0.3 / 0.22, detail=1.8), 0.6 * wd,
                               1.6 * p.brightness, r_end=0.35 * wd, a_end=0.4 * p.brightness)
        seeds = self.tick_rng("jitter", t, rate).integers(0, 1 << 62, len(self.events))
        for ev, sd in zip(self.events, seeds):
            age = (k - ev["start"]) % self.n_ticks
            if age >= ev["dur"]:
                continue
            g = np.random.default_rng(int(sd))
            e = ev["br"] * (1.0 if age == 0 else 0.6) * p.brightness
            path = refine(g, ev["pts"], rough=p.roughness, detail=2.2)
            ctx.light.polyline(path, 0.9 * wd, 2.6 * e, r_end=0.55 * wd, a_end=1.6 * e)
            ctx.light.dot(float(path[-1, 0]), float(path[-1, 1]), 2.5, 0.9 * e)
            if len(path) > 8 and g.random() < p.branch_chance:
                i = int(g.integers(len(path) // 4, 3 * len(path) // 4))
                d = path[min(i + 2, len(path) - 1)] - path[i - 1]
                d = d / (np.linalg.norm(d) + 1e-9)
                ang = g.choice([-1, 1]) * g.uniform(0.5, 1.0)
                ca, sa = math.cos(ang), math.sin(ang)
                q = path[i] + np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]) * g.uniform(12, 26)
                bp = refine(g, [path[i], q], rough=p.roughness * 0.25 / 0.22, detail=2.0)
                ctx.light.polyline(bp, 0.5 * wd, 1.3 * e, r_end=0.3 * wd, a_end=0.2 * e)


# =============================================================================
# beam
# =============================================================================
@atom("beam_line", tags=["light", "beam", "laser", "loop"])
class BeamLine(Atom):
    """Horizontal energy beam from ``emitter`` (local origin) to ``impact`` (+x): blinding core
    with surges racing along it, turbulent fbm sheath and twisting filaments scrolling whole
    noise periods per loop, all under the shared beam flicker."""
    kind = "loop"
    outputs = {"light": Light(palette="electric", bloom=[[1.5, .5], [5, .4], [15, .35]])}
    streams = ("noise",)
    anchors = {"emitter": (0.0, 0.0), "impact": lambda self, t: (float(self.p.length), 0.0)}

    class Params(Atom.Params):
        period: float = P(1.2, gt=0, doc="loop length (s)")
        length: float = P(412.0, gt=0, doc="emitter-to-impact distance (px)")
        core_width: float = P(1.0, gt=0, doc="width of the blinding core")
        core: float = P(1.0, ge=0, doc="core brightness multiplier")
        sheath_width: float = P(1.0, gt=0, doc="width of the turbulent sheath")
        turbulence: float = P(1.0, ge=0, doc="brightness of the sheath plasma")
        filaments: bool = P(True, doc="bright twisting filaments in the sheath")
        scroll_speed: int = P(1, doc="flow speed along the beam (noise periods per loop)")
        surge: float = P(0.22, ge=0, doc="amplitude of energy pulses racing along the core")
        surge_cycles: int = P(4, doc="surge pulses passing a point per loop")
        flicker: float = P(1.0, ge=0, doc="brightness flicker amount")

    def setup(self):
        rng = self.rng("noise")
        self.nA = Noise(rng, period=(16, 16, 4))
        self.nB = Noise(rng, period=(16, 16, 4))
        self.nW = Noise(rng, period=(16, 4, 4))

    def _band(self):
        return 58 * max(1.0, self.p.sheath_width)

    def get_bounds(self):
        b = self._band()
        return (-12, -b, self.p.length + 12, b)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        z = tn * 4
        L = p.length
        k = p.scroll_speed
        sw = p.sheath_width
        flick = _beam_flicker(p.flicker, tn)
        b = self._band()

        def sheath(X, Y):
            along = smoothstep(-6, 18, X) * smoothstep(L + 10, L - 6, X)
            wob = self.nW.fbm(X / 68 - 16 * k * tn, 0.0, z, octaves=2)
            W = 17 * sw * (1 + 0.35 * wob)
            a_ = self.nA.fbm(X / 34 - 16 * k * tn, Y / (11 * sw), z, octaves=3)
            v = np.exp(-(Y / W) ** 2) * (0.08 + 0.42 * smoothstep(-0.1, 0.7, a_)) \
                + 0.07 * np.exp(-(Y / (W * 2.0)) ** 2) * (0.6 + 0.4 * a_)
            if p.filaments:
                b_ = self.nB.fbm(X / 64 - 32 * k * tn, Y / (9 * sw), z, octaves=3)
                v = v + 0.75 * (1 - np.abs(b_)) ** 8 * np.exp(-(Y / (W * 1.3)) ** 2)
            return v * p.turbulence * along * flick
        if p.turbulence > 0:
            ctx.light.field(sheath, (-8, -b, L + 12, b), lowres="bilinear")

        wob_hi = 1 + 0.12 * math.sin(TAU * 2 * tn)
        cw = p.core_width

        def core(X, Y):
            along = smoothstep(-3, 8, X) * smoothstep(L + 6, L - 2, X)
            sg = 1 + p.surge * np.sin(TAU * (X / 70 - p.surge_cycles * tn))
            return along * flick * p.core * (2.8 * np.exp(-(Y / (2.0 * cw * wob_hi)) ** 2)
                                              + 0.6 * sg * np.exp(-(Y / (5.5 * cw)) ** 2))
        if p.core > 0:
            cb = min(b, 5.5 * cw * 3.2 + 2)
            ctx.light.field(core, (-6, -cb, L + 8, cb))


@atom("spike_flare", tags=["light", "beam", "flare", "loop"])
class SpikeFlare(Atom):
    """Emitter flare: pulsing hot core + halo, a vertical cross spike, a backward spike and
    ``spikes`` rotating spikes of alternating length. Legacy seam bug fixed: when the spikes
    turn an odd number of steps per loop, their lengths morph into their neighbours' so
    the loop is seamless."""
    kind = "loop"
    outputs = {"light": Light(palette="electric")}

    class Params(Atom.Params):
        period: float = P(1.2, gt=0, doc="loop length (s)")
        core_radius: float = P(6.0, gt=0, doc="hot core radius (px)")
        core_brightness: float = P(2.4, ge=0, doc="hot core intensity")
        halo_radius: float = P(16.0, ge=0, doc="halo radius (px)")
        halo_brightness: float = P(0.55, ge=0, doc="halo intensity")
        pulse: float = P(0.15, ge=0, le=0.9, doc="size pulse amplitude")
        pulse_cycles: int = P(2, ge=0, doc="pulses per period")
        spikes: int = P(6, ge=0, doc="rotating spikes")
        spike_length: float = P(14.0, ge=0, doc="length of the short rotating spikes (px)")
        alternate: float = P(6.0, ge=0, doc="extra length of every other rotating spike (px)")
        spike_width: float = P(0.9, gt=0, doc="rotating spike width (px)")
        spike_brightness: float = P(0.9, ge=0, doc="rotating spike intensity")
        rotation: int = P(1, doc="rotating-spike steps (360/spikes deg) per period")
        angle: float = P(17.2, doc="angle offset of the rotating spikes (deg)")
        cross: float = P(34.0, ge=0, doc="length of the vertical cross spikes (px; 0 = none)")
        cross_width: float = P(1.4, gt=0, doc="cross spike width (px)")
        cross_brightness: float = P(1.6, ge=0, doc="cross spike intensity")
        back: float = P(30.0, ge=0, doc="length of the spike pointing back along -x (px; 0 = none)")
        back_width: float = P(1.6, gt=0, doc="back spike width (px)")
        back_brightness: float = P(1.2, ge=0, doc="back spike intensity")

    def get_bounds(self):
        p = self.p
        r = max(p.cross + 3 * p.cross_width, p.back + 3 * p.back_width,
                p.spike_length + p.alternate + 3 * p.spike_width,
                3 * p.halo_radius, 3 * p.core_radius) * (1 + p.pulse) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        ep = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * tn)
        v = ctx.light
        v.dot(0, 0, p.core_radius * ep, p.core_brightness)
        if p.halo_radius > 0:
            v.dot(0, 0, p.halo_radius * ep, p.halo_brightness)
        if p.cross > 0:
            for sgn in (-1, 1):
                v.polyline([(0, 0), (0, sgn * p.cross * ep)], p.cross_width, p.cross_brightness, r_end=0.3, a_end=0)
        if p.back > 0:
            v.polyline([(0, 0), (-p.back * ep, 0)], p.back_width, p.back_brightness, r_end=0.3, a_end=0)
        n = p.spikes
        for k in range(n):
            ang = TAU / n * (p.rotation * tn + k) + math.radians(p.angle)
            # alternate long/short; morphs into the neighbour's length as it rotates (seamless)
            L = (p.spike_length + p.alternate * (0.5 - 0.5 * math.cos(math.pi * (k + p.rotation * tn)))) * ep
            v.polyline([(0, 0), (L * math.cos(ang), L * math.sin(ang))], p.spike_width, p.spike_brightness,
                       r_end=0.2, a_end=0)


@atom("splash_fan", tags=["light", "beam", "impact", "loop"])
class SplashFan(Atom):
    """Beam impact: a noisy splash fan pointing back along -x, a hot impact point, a vertical
    flare on the wall and pulsing squashed shock rings. Shares the beam flicker."""
    kind = "loop"
    outputs = {"light": Light(palette="electric")}
    streams = ("noise",)

    class Params(Atom.Params):
        period: float = P(1.2, gt=0, doc="loop length (s)")
        flicker: float = P(1.0, ge=0, doc="flicker amount (same curve as beam_line)")
        fan: float = P(0.9, ge=0, doc="splash fan brightness")
        reach: float = P(30.0, ge=0, doc="splash fan length toward -x (px)")
        base: float = P(6.0, ge=0, doc="splash size in every direction (px)")
        cutoff: float = P(70.0, gt=0, doc="splash fan cut-off radius (px)")
        noise_scale: int = P(16, ge=1, doc="noise cells around the fan")
        boil: int = P(8, ge=0, doc="noise time cycles per period")
        core_radius: float = P(7.0, gt=0, doc="impact point radius (px)")
        core_brightness: float = P(2.8, ge=0, doc="impact point intensity")
        halo_radius: float = P(20.0, ge=0, doc="impact halo radius (px)")
        halo_brightness: float = P(0.35, ge=0, doc="impact halo intensity")
        wall: float = P(0.5, ge=0, doc="vertical wall flare brightness")
        wall_width: float = P(5.0, gt=0, doc="wall flare half-width (px)")
        wall_height: float = P(34.0, gt=0, doc="wall flare half-height (px)")
        rings: int = P(2, ge=0, doc="shock rings in flight at once")
        ring_cycles: int = P(2, ge=0, doc="ring launches per period (per ring)")
        ring_radius: float = P(26.0, ge=0, doc="distance a ring travels (px)")
        ring_width: float = P(1.1, gt=0, doc="ring thickness (px)")
        ring_brightness: float = P(0.9, ge=0, doc="ring intensity")
        ring_squash: float = P(0.45, gt=0, le=1, doc="ring x/y ratio (rings are flattened along the beam)")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(self.p.noise_scale, 4, max(self.p.boil // 2, 1)))
        self.ang = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)

    def get_bounds(self):
        p = self.p
        r = max(p.cutoff, 3 * p.halo_radius, 2.2 * p.wall_height, 4 + p.ring_radius + 4 * p.ring_width) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        flick = _beam_flicker(p.flicker, tn)
        sn1 = self.noise.fbm(self.ang / TAU * p.noise_scale, 0.0, tn * p.boil / 2, octaves=3)

        def f(X, Y):
            R = np.hypot(X, Y)
            th = np.arctan2(Y, X)
            sn = np.interp(th, self.ang, sn1)
            back = 0.25 + 0.75 * np.maximum(-np.cos(th), 0)
            Lr = p.base + p.reach * back * np.clip(0.45 + 0.8 * sn, 0.05, None)
            v = p.fan * np.exp(-R / np.maximum(Lr, 1e-3) * 2.5) * smoothstep(p.cutoff, p.cutoff - 2, R)
            v = v + p.wall * np.exp(-(X / p.wall_width) ** 2 - (Y / p.wall_height) ** 2)
            for k in range(p.rings):
                ph = (p.ring_cycles * tn + k / max(p.rings, 1)) % 1.0
                a = p.ring_brightness * (1 - ph) ** 2
                if a > 0.01:
                    Rr = 4 + p.ring_radius * float(_ease_out(ph, 2))
                    d = np.hypot(X / p.ring_squash, Y) - Rr
                    v = v + a * np.exp(-0.5 * (d / p.ring_width) ** 2)
            return v
        ctx.light.field(f, self.get_bounds())
        ctx.light.dot(0, 0, p.core_radius, p.core_brightness * flick)
        if p.halo_radius > 0:
            ctx.light.dot(0, 0, p.halo_radius, p.halo_brightness)


# =============================================================================
# shield
# =============================================================================
@atom("sphere_shell", tags=["light", "shield", "bubble", "loop"])
class SphereShell(Atom):
    """Protective bubble: fresnel rim, hexagon grid on the sphere (stereographic, so cells
    foreshorten toward the rim), a shimmer wave sweeping across with a trailing glow, random
    twinkling cells, noisy rim line and halo. Legacy trail seam fixed (the trail fades out
    before the wave wraps)."""
    kind = "loop"
    outputs = {"light": Light(palette="frost", bloom=[[1.2, .45], [4, .4], [12, .35]], alpha_gamma=0.9)}
    streams = ("noise",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        radius: float = P(128.0, gt=0, doc="bubble radius (px)")
        hex_density: float = P(7.0, gt=0, doc="hexagon cells across the sphere")
        grid_brightness: float = P(1.0, ge=0, doc="resting brightness of the grid lines")
        wave: float = P(1.0, ge=0, doc="strength of the shimmer wave")
        waves: int = P(1, ge=0, doc="shimmer sweeps per period (0 = none)")
        twinkle: float = P(0.3, ge=0, le=1, doc="fraction of cells that glint")
        rim: float = P(1.0, ge=0, doc="fresnel rim brightness")
        fill: float = P(1.0, ge=0, doc="faint interior tint")
        breathe: float = P(0.06, ge=0, doc="brightness breathing amplitude")
        highlight: bool = P(True, doc="small specular highlight top-left")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(16, 4, 4))
        self.ang = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)
        d3 = np.array([0.62, -0.48, 0.62])
        self.d3 = d3 / np.linalg.norm(d3)

    def get_bounds(self):
        r = self.p.radius + 36
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        R = p.radius
        K = p.hex_density
        sq3 = math.sqrt(3)
        pxs = ctx.px
        rn1 = self.noise.fbm(self.ang / TAU * 16, 0.0, tn * 4, octaves=3)
        if p.waves > 0:
            frac = (p.waves * tn) % 1.0
            ph = -1.45 + 2.85 * frac
            trail_w = _ss(1.45, 0.95, ph)     # seam fix: trail gone before the wave wraps
        else:
            ph, trail_w = None, 0.0
        breathe = 1 + p.breathe * math.sin(TAU * tn)
        d3 = self.d3

        def f(X, Y):
            nx, ny = X / R, Y / R
            rr = np.hypot(nx, ny)
            inside = (rr < 1).astype(np.float32)
            nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
            th = np.arctan2(ny, nx)
            gx0, gy0 = nx / (1 + nz) * K, ny / (1 + nz) * K
            ax, ay = np.mod(gx0, 1.0) - 0.5, np.mod(gy0, sq3) - sq3 / 2
            bx, by = np.mod(gx0 - 0.5, 1.0) - 0.5, np.mod(gy0 - sq3 / 2, sq3) - sq3 / 2
            use_a = ax * ax + ay * ay < bx * bx + by * by
            gx, gy = np.where(use_a, ax, bx), np.where(use_a, ay, by)
            hexd = np.maximum(np.abs(gx) * 0.5 + np.abs(gy) * sq3 / 2, np.abs(gx))
            edge = 0.5 - hexd
            # grid units per output pixel (constant-width anti-aliased lines)
            if X.shape[0] > 1 and X.shape[1] > 1:
                sp = math.hypot(float(X[0, 1] - X[0, 0]), float(Y[0, 1] - Y[0, 0]))
                g0, g1 = np.gradient(gx0)
                dpx = np.hypot(g0, g1) / max(sp, 1e-9) / max(pxs, 1e-9)
            else:
                dpx = np.full(X.shape, K / R / max(pxs, 1e-9), np.float32)
            lines = np.exp(-0.5 * (edge / (0.7 * dpx + 1e-4)) ** 2) * smoothstep(0.08, 0.4, nz) * inside
            fillm = smoothstep(0.0, 0.12, edge) * smoothstep(0.05, 0.35, nz) * inside
            idx = np.round((gx0 - gx) * 2).astype(np.int64)
            idy = np.round((gy0 - gy) / (sq3 / 2)).astype(np.int64)
            hsh = ((idx * 73856093) ^ (idy * 19349663)) % 1009 / 1009.0
            hk = 1 + (hsh * 7).astype(np.int64) % 2
            lucky = ((((idx * 83492791) ^ (idy * 2654435761)) % 997 / 997.0) < p.twinkle).astype(np.float32)
            fres = (1 - nz) ** 3 * inside
            rim_line = np.exp(-0.5 * ((rr - 1) * R / 1.3) ** 2)
            halo = np.exp(-np.maximum(rr - 1, 0) * R / 7) * (rr > 1) * 0.35
            if ph is not None:
                sdir = nx * d3[0] + ny * d3[1] + nz * d3[2]
                wv = p.wave * np.exp(-0.5 * ((sdir - ph) / 0.12) ** 2) * inside
                trail = trail_w * p.wave * np.exp(-0.5 * ((sdir - ph + 0.25) / 0.25) ** 2) * inside
            else:
                wv = trail = 0.0
            tw = np.maximum(0, np.sin(TAU * (hk * tn + hsh))) ** 10 * lucky
            rn = np.interp(th, self.ang, rn1)
            I = (0.05 * p.fill * inside + p.rim * 0.75 * fres * (0.85 + 0.3 * rn)
                 + p.rim * ((1.25 + 0.5 * rn) * rim_line + halo)
                 + lines * (0.16 * p.grid_brightness + 1.5 * wv + 0.35 * trail + 0.5 * tw)
                 + fillm * (0.12 * wv + 0.06 * trail + 0.09 * tw)
                 + 0.6 * wv * fres)
            return I * breathe
        ctx.light.field(f, self.get_bounds())
        if p.highlight:
            ctx.light.dot(-0.45 * R, -0.5 * R, 9, 0.12)


# =============================================================================
# columns of light
# =============================================================================
@atom("light_column", tags=["light", "column", "pillar", "heal", "loop"])
class LightColumn(Atom):
    """Vertical column of light standing on the local origin (ground centre), filled with
    streak noise scrolling ``rise`` noise periods per loop (negative = downward).

    * ``aura`` style: edge-brightened hollow profile, brightening toward the ground, bottom
      clipped to the front arc of a ground ellipse (heal column).
    * ``pillar`` style: gaussian body + hot core line, flat bottom at the head, which slams
      down to the ground over ``slam_time``; width eases from ``width_start`` over
      ``grow_time`` and an initial ``flash`` decays (holy pillar). With ``grow_time``,
      ``slam_time`` or ``flash`` set the column is a one-shot over its first period: use
      ``time_mode: once``."""
    kind = "loop"
    outputs = {"light": Light(palette="heal")}
    streams = ("noise",)

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        style: str = P("aura", doc="aura | pillar", choices=["aura", "pillar"])
        width: float = P(84.0, gt=0, doc="aura: column half-width; pillar: body width (px)")
        width_start: float = P(-1.0, doc="width at t=0 (px); -1 = same as width")
        grow_time: float = P(0.0, ge=0, doc="seconds to ease from width_start to width (0 = constant)")
        height: float = P(272.0, gt=0, doc="column height above the ground (px)")
        brightness: float = P(0.65, ge=0, doc="body brightness")
        core: float = P(0.0, ge=0, doc="pillar: hot core line brightness")
        core_halo: float = P(0.0, ge=0, doc="pillar: wide soft halo brightness")
        top_fade: float = P(-1.0, doc="length of the fade at the top (px); -1 = whole height")
        fade_power: float = P(1.6, gt=0, doc="exponent of the top fade")
        ground_squash: float = P(0.3, ge=0, le=1, doc="ground ellipse squash for the bottom clip (0 = flat)")
        slam_time: float = P(0.0, ge=0, doc="seconds for the head to reach the ground (0 = already there)")
        head_start: float = P(0.55, ge=0, le=1, doc="fraction of the height the head has covered at t=0")
        rise: int = P(1, doc="streak scroll (noise periods per loop; negative = downward)")
        streak_x: float = P(14.0, gt=0, doc="streak noise scale across (px)")
        streak_y: float = P(60.0, gt=0, doc="streak noise scale along (px)")
        pulse: float = P(0.1, ge=0, le=1, doc="brightness pulse amplitude")
        pulse_cycles: int = P(1, ge=0, doc="pulses per period")
        flash: float = P(0.0, ge=0, doc="pillar: extra core brightness at t=0, decaying")
        flash_time: float = P(0.11, gt=0, doc="flash time constant (s)")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(16, 8, 4))

    def _w(self, t):
        p = self.p
        if p.grow_time <= 0:
            return p.width
        ws = p.width if p.width_start < 0 else p.width_start
        return ws + (p.width - ws) * float(_ease_out(t / p.grow_time, 3))

    def get_bounds(self):
        p = self.p
        wmax = max(p.width, p.width_start)
        if p.style == "aura":
            return (-wmax - 2, -p.height - 2, wmax + 2, p.ground_squash * wmax + 8)
        x = max(2.2 * wmax * (2.6 if p.core_halo > 0 else 1.2), 2.6 * wmax) + 2
        return (-x, -p.height - 2, x, p.ground_squash * wmax + 8)

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.period
        H = p.height
        wd = self._w(t)
        pl = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * tn) if p.pulse else 1.0
        head = 0.0
        if p.slam_time > 0:
            head = -H * (1 - min(1.0, p.head_start + (1 - p.head_start) * t / p.slam_time))
        fl = 1 + p.flash * math.exp(-t / p.flash_time) if p.flash else 1.0
        z = tn * 4
        scroll = 8 * p.rise * tn
        tf = H if p.top_fade < 0 else p.top_fade
        gsq = p.ground_squash
        aura = p.style == "aura"
        bx0, by0, bx1, by1 = self.get_bounds()
        if not aura:
            body_x = 2.6 * wd + 2
            bx0, bx1 = -body_x, body_x
        by0 = max(by0, -H - 2)

        def vmask(X, Y):
            v = smoothstep(-H, -H + tf, Y) ** p.fade_power
            yb = head + gsq * wd * np.sqrt(np.clip(1 - (X / wd) ** 2, 0, 1)) if gsq > 0 else head
            return v * smoothstep(yb + 5, yb - 5, Y)

        def body(X, Y):
            n = self.noise.fbm(X / p.streak_x, Y / p.streak_y + scroll, z, octaves=3)
            if aura:
                dxl = np.abs(X) / wd
                prof = smoothstep(1.0, 0.72, dxl) * (0.35 + 0.65 * dxl ** 2)
                v = prof * (0.45 + 0.75 * smoothstep(-0.2, 0.7, n))
            else:
                v = np.exp(-(X / wd) ** 2) * (0.2 + 0.9 * smoothstep(-0.1, 0.6, n))
            return v * vmask(X, Y) * p.brightness * pl
        if p.brightness > 0:
            ctx.light.field(body, (bx0, by0, bx1, by1), lowres="bilinear")
        if not aura and (p.core > 0 or p.core_halo > 0):
            cw = wd * 0.32 + 1.5
            hb = 2.2 * wd * (2.6 if p.core_halo > 0 else 0) + 3 * cw

            def core(X, Y):
                v = p.core * np.exp(-(X / cw) ** 2) + p.core_halo * np.exp(-(X / (wd * 2.2)) ** 2)
                return v * vmask(X, Y) * fl * pl
            ctx.light.field(core, (-hb, by0, hb, by1))


@atom("anamorphic_streak", tags=["light", "flare", "streak", "oneshot"])
class AnamorphicStreak(Atom):
    """Horizontal lens streak through the origin: thick and hot at the centre, tapering to
    hairlines at both ends. Flashes and decays to a sustained level, fading out at the end."""
    kind = "oneshot"
    outputs = {"light": Light(palette="holy")}

    class Params(Atom.Params):
        duration: float = P(1.1, gt=0, doc="seconds until gone")
        length: float = P(104.0, gt=0, doc="half-length (px)")
        width: float = P(2.2, gt=0, doc="thickness at the centre (px)")
        thin: float = P(0.4, gt=0, doc="thickness at the ends (px)")
        brightness: float = P(1.2, ge=0, doc="intensity multiplier")
        flash: float = P(1.3, ge=0, doc="initial flash level (decays)")
        decay: float = P(0.11, gt=0, doc="flash time constant (s)")
        sustain: float = P(0.5, ge=0, doc="steady level after the flash")

    def get_bounds(self):
        p = self.p
        return (-p.length - 3 * p.thin - 2, -3 * p.width - 2, p.length + 3 * p.thin + 2, 3 * p.width + 2)

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        a = p.brightness * (p.flash * math.exp(-t / p.decay) + p.sustain) * _end_fade(t, p.duration)
        if a <= 1e-4:
            return
        ctx.light.polyline([(-p.length, 0), (0, 0)], p.thin, 0.0, r_end=p.width, a_end=a)
        ctx.light.polyline([(0, 0), (p.length, 0)], p.width, a, r_end=p.thin, a_end=0.0)


# =============================================================================
# charge up
# =============================================================================
@atom("inflow", tags=["light", "charge", "particles", "oneshot"])
class Inflow(Atom):
    """Charge-up: particles streak inward on accelerating spiral paths with tapered trails,
    faint rings converge, and the core brightens until ``pop_at``. With ``pop`` the atom stops
    there (put flash / shockwave / burst layers at that time); without, the core fades out."""
    kind = "oneshot"
    outputs = {"light": Light(palette="arcane", bloom=[[1.2, .5], [4, .5], [12, .45]])}
    streams = ("particles",)

    class Params(Atom.Params):
        duration: float = P(1.35, gt=0, doc="total length (s)")
        count: int = P(64, ge=0, doc="inward-streaking particles")
        radius_min: float = P(70.0, ge=0, doc="closest particle start (px)")
        radius_max: float = P(104.0, ge=0, doc="furthest particle start (px)")
        swirl: float = P(1.0, doc="how much the particles spiral on the way in (radians)")
        trail: float = P(0.45, ge=0, le=1, doc="trail length (fraction of each path)")
        life_min: float = P(0.22, gt=0, doc="shortest flight (fraction of duration)")
        life_max: float = P(0.34, gt=0, doc="longest flight (fraction of duration)")
        brightness: float = P(1.0, ge=0, doc="particle brightness multiplier")
        core: float = P(1.0, ge=0, doc="charging core brightness")
        rings: int = P(3, ge=0, doc="faint rings contracting inward")
        ring_radius: float = P(84.0, ge=0, doc="converging ring start radius (px)")
        pop_at: float = P(0.72, ge=0.3, le=0.92, doc="fraction of the duration when the charge is full")
        pop: bool = P(True, doc="stop at pop_at (a pop follows); false = the core fades out instead")

    def setup(self):
        p = self.p
        rng = self.rng("particles")
        lo, hi = min(p.radius_min, p.radius_max), max(p.radius_min, p.radius_max)
        self.parts = [dict(t0=rng.uniform(-0.3, p.pop_at - 0.28), D=rng.uniform(p.life_min, max(p.life_min, p.life_max)),
                           r0=rng.uniform(lo, hi), a0=rng.uniform(0, TAU), sw=rng.uniform(0.6, 1.4) * p.swirl,
                           br=rng.uniform(0.7, 1.3)) for _ in range(p.count)]

    def get_bounds(self):
        p = self.p
        r = max(p.radius_min, p.radius_max, p.ring_radius, 3 * 24) + 6
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        u = t / p.duration
        v = ctx.light
        if u < p.pop_at:
            c = u / p.pop_at
            v.dot(0, 0, 2.5 + 4.5 * c, (0.5 + 3.0 * c * c) * p.core)
            v.dot(0, 0, 10 + 14 * c, (0.12 + 0.45 * c * c) * p.core)
            for q in self.parts:
                s = (u - q["t0"]) / q["D"]
                if not 0 <= s < 1:
                    continue
                ss = np.linspace(max(s - p.trail, 0), s, 12)
                rr = q["r0"] * (1 - ss ** 2)
                aa = q["a0"] + q["sw"] * ss ** 2
                pts = np.stack([rr * np.cos(aa), rr * np.sin(aa)], 1)
                a = q["br"] * _ss(0, 0.15, s) * (0.6 + 0.6 * s) * p.brightness
                v.polyline(pts, 0.25, 0.0, r_end=0.9, a_end=a * 1.5)
            for k in range(p.rings):
                rt = c * 3 - k * 0.8
                if 0 <= rt < 1:
                    v.ring(0, 0, max(p.ring_radius * (1 - rt) + 10 * rt, 0.5), 1.0, 0.6 * math.sin(math.pi * rt) ** 2)
        elif not p.pop:
            pt = (u - p.pop_at) / (1 - p.pop_at)
            fade = (1 - pt) ** 1.5
            v.dot(0, 0, 7 * (1 - 0.6 * pt), 3.5 * p.core * fade)
            v.dot(0, 0, 24, 0.57 * p.core * fade)
