"""Material impact atoms: wood splinters, stone chips + dust, blood droplets, metal sparks and
an energy hit (rings, glints, crackle).

Every atom's local origin is the impact point. Debris is thrown in a cone around ``angle``
(degrees, clockwise from +x; -90 = straight up = the surface normal of a floor). Rotate the
layer so -90 points away from the struck surface. Matter is painted into ``solid`` roles
(two-tone facets: palette position = lighting), blood is ``volume`` density (glossy surface
shading), dust is ``smoke``, sparks and energy are light. Lengths are pixels at scale 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Solid, Volume, atom
from ..engine.color import register_palette
from ..engine.mathx import TAU, ballistic, refine, smoothstep
from ..engine.particles import Emitter, EmitterConfig, draw_particles
from .combat import glint
from .smoke import PuffTexture, deposit_puff

register_palette("wood_splinter", "0:#3a1d0c,0.3:#7a4620,0.62:#c08a4e,0.85:#e8c48c,1:#f7e4bc")
register_palette("stone_chip", "0:#2e2c2d,0.2:#4d4a4a,0.6:#9a958d,1:#d6d0c4")
register_palette("stone_dust", "0:#5a5450,0.5:#9c958c,1:#d6d0c4")
register_palette("blood_gloss", "0:#2a0003,0.45:#7a0610,0.8:#b81a22,1:#ffb4b4")
register_palette("spark_hot", "0:#b83c00,0.4:#ff9a1e,0.75:#ffe68c,1:#fffff5")

LIGHT = np.array([-0.55, -0.85]) / math.hypot(-0.55, -0.85)     # toward the light (screen, y down)


def _ss(e0, e1, x) -> float:
    return float(smoothstep(e0, e1, x))


def _end_fade(t: float, duration: float, start: float = 0.6) -> float:
    return _ss(duration, duration * start, t)


def _dirs(rng, n, angle, spread):
    a = np.radians(angle) + np.radians(spread) * (rng.random(n) - 0.5)
    return a


def _rot(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    return pts @ np.array([[c, s], [-s, c]])


class _ConeParams(Atom.Params):
    angle: float = P(-90.0, doc="spray direction (deg, clockwise from +x; -90 = up / away from a floor)")
    spread: float = P(150.0, ge=0, le=360, doc="cone width of the spray (deg)")
    speed: float = P(1.0, ge=0, doc="launch speed multiplier")
    gravity: float = P(900.0, doc="gravity (px/s², +down)")
    drag: float = P(1.5, ge=0, doc="air drag (1/s)")
    extent: float = P(120.0, gt=0, doc="debris fades out before this distance from the impact (px); also bounds")


# =============================================================================
# wood
# =============================================================================
def _splinter_shape(rng, L, W):
    """Long thin wood shard along +x, centred: jagged broken ends, slightly bent."""
    n = 5
    xs = np.linspace(-L / 2, L / 2, n)
    bend = rng.uniform(-0.12, 0.12) * W
    top = [(x, -W / 2 * (1 - (2 * x / L) ** 2 * 0.6) + bend * (1 - (2 * x / L) ** 2) + rng.uniform(-0.15, 0.15) * W)
           for x in xs[1:-1]]
    bot = [(x, W / 2 * (1 - (2 * x / L) ** 2 * 0.6) + bend * (1 - (2 * x / L) ** 2) + rng.uniform(-0.15, 0.15) * W)
           for x in xs[1:-1]]
    tip_r = [(L / 2, rng.uniform(-0.25, 0.25) * W)]
    tip_l = [(-L / 2 + rng.uniform(0, 0.12) * L, rng.uniform(-0.3, 0.3) * W),
             (-L / 2 - rng.uniform(0, 0.08) * L, rng.uniform(-0.2, 0.2) * W)]
    upper = np.array(tip_l[1:] + top + tip_r)
    lower = np.array(tip_r + bot[::-1] + tip_l[:1])
    spine = np.array([tip_l[1]] + [(x, bend * (1 - (2 * x / L) ** 2)) for x in xs[1:-1]] + tip_r)
    return upper, lower, spine


@atom("splinters", tags=["impact", "wood", "solid", "debris"])
class Splinters(Atom):
    """Wood splinters bursting from a hit: long two-tone shards (dark bark side, pale broken
    grain side, a light fibre line along the spine) that tumble end over end (they also flip,
    narrowing as they turn edge-on), fly out under drag and gravity and fade; plus small
    sawdust flecks. Solid matter (palette: dark -> pale wood)."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="wood_splinter")}
    streams = ("shards", "flecks")

    class Params(_ConeParams):
        duration: float = P(0.9, gt=0, doc="seconds until every splinter has faded")
        count: int = P(12, ge=0, doc="number of splinters")
        size: float = P(1.0, gt=0, doc="splinter length multiplier")
        flecks: int = P(18, ge=0, doc="small sawdust flecks")
        spin: float = P(1.0, ge=0, doc="tumble speed multiplier")
        fade_start: float = P(0.55, ge=0, lt=1, doc="fade-out starts (fraction of duration)")

    def setup(self):
        p = self.p
        rng = self.rng("shards")
        n = p.count
        self.L = rng.uniform(11, 28, n) * p.size
        self.L[: max(1, n // 4)] *= 1.35            # a few long ones
        self.W = self.L * rng.uniform(0.16, 0.26, n)
        self.shapes = [_splinter_shape(rng, self.L[i], self.W[i]) for i in range(n)]
        ang = _dirs(rng, n, p.angle, p.spread)
        spd = rng.uniform(220, 520, n) * p.speed / np.sqrt(self.L / (18 * p.size))
        self.v0 = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        self.p0 = np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(0, 5, (n, 1))
        self.rot0 = ang + rng.uniform(-0.5, 0.5, n)
        self.w = rng.uniform(5, 16, n) * rng.choice([-1, 1], n) * p.spin
        self.flip = rng.uniform(4, 12, n) * p.spin
        self.flip_ph = rng.uniform(0, TAU, n)
        self.life = rng.uniform(0.75, 1.0, n)
        rng = self.rng("flecks")
        m = p.flecks
        a2 = _dirs(rng, m, p.angle, p.spread * 1.1)
        s2 = rng.uniform(120, 420, m) * p.speed
        self.fv = np.stack([np.cos(a2) * s2, np.sin(a2) * s2], 1)
        self.fr = rng.uniform(0.7, 1.5, m)
        self.fk = rng.uniform(0.25, 0.7, m)
        self.flife = rng.uniform(0.35, 0.8, m)

    def get_bounds(self):
        e = self.p.extent + 26 * self.p.size * 1.35
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if u >= 1:
            return
        sol = ctx.solid
        fadeall = _ss(1.0, p.fade_start, u)
        pos = ballistic(self.p0, self.v0, t, p.drag, p.gravity)
        for i in range(p.count):
            fade = fadeall * _ss(self.life[i], self.life[i] * p.fade_start, u)
            d = float(np.hypot(*pos[i]))
            fade *= _ss(p.extent, p.extent * 0.8, d)
            if fade <= 0.01:
                continue
            rot = self.rot0[i] + self.w[i] * t * math.exp(-t * 1.2)
            wk = 0.3 + 0.7 * abs(math.cos(self.flip_ph[i] + self.flip[i] * t))
            up, lo, spine = self.shapes[i]
            sc = np.array([1.0, wk])
            c = pos[i]
            P_up = _rot(up * sc, rot) + c
            P_lo = _rot(lo * sc, rot) + c
            P_sp = _rot(spine * sc, rot) + c
            # which half faces the light
            nrm = np.array([-math.sin(rot), math.cos(rot)])  # local +y in world
            lit_lo = float(nrm @ LIGHT) > 0
            k_up, k_lo = (0.28, 0.72) if lit_lo else (0.72, 0.28)
            whole = np.vstack([P_up, P_lo[1:-1]])
            sol.polygon(whole, k=0.12, a=fade, soft=0.6)
            sol.polygon(np.vstack([P_up, P_sp[::-1]]), k=k_up, a=fade, soft=0.5)
            sol.polygon(np.vstack([P_lo, P_sp]), k=k_lo, a=fade, soft=0.5)
            sol.stroke(P_sp[1:-1], max(0.35, self.W[i] * wk * 0.08), k=0.95, a=fade * 0.8)
        fp = ballistic(np.zeros_like(self.fv), self.fv, t, p.drag * 2, p.gravity)
        fq = ballistic(np.zeros_like(self.fv), self.fv, max(t - 0.02, 0), p.drag * 2, p.gravity)
        for j in range(len(self.fv)):
            q = u / self.flife[j]
            if q >= 1:
                continue
            a = (1 - q) ** 0.7 * _ss(p.extent, p.extent * 0.8, float(np.hypot(*fp[j])))
            sol.capsule(tuple(fq[j]), tuple(fp[j]), self.fr[j] * 0.6, self.fr[j], k=self.fk[j], a=a)


# =============================================================================
# stone
# =============================================================================
def _chip_shape(rng, r):
    n = int(rng.integers(4, 7))
    th = np.sort(rng.uniform(0, TAU, n))
    th = np.linspace(0, TAU, n, endpoint=False) + rng.uniform(-0.35, 0.35, n)
    rad = r * rng.uniform(0.55, 1.0, n)
    return np.stack([np.cos(th) * rad, np.sin(th) * rad * rng.uniform(0.6, 1.0)], 1)


@atom("chips", tags=["impact", "stone", "solid", "debris", "dust"])
class Chips(Atom):
    """Stone chips knocked off a hit: angular two-tone chunks (dark silhouette + lit facet
    toward the light) and grit tumbling out under gravity, plus a dust puff (smoke role) that
    billows out along the spray and settles. Solid palette dark -> light stone."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="stone_chip"),
               "smoke": Smoke(palette="stone_dust", density=3.0, shadow=0.8, ambient=0.3, relief=1.2,
                              height_gain=0.3, size_ref=256)}
    streams = ("chips", "grit", "dust", "texture")

    class Params(_ConeParams):
        duration: float = P(1.0, gt=0, doc="seconds until everything has faded")
        count: int = P(14, ge=0, doc="number of stone chips")
        size: float = P(1.0, gt=0, doc="chip size multiplier")
        grit: int = P(24, ge=0, doc="small grit specks")
        dust: float = P(1.0, ge=0, doc="dust puff density (0 = none)")
        dust_size: float = P(1.0, gt=0, doc="dust puff size multiplier")
        puffs: int = P(9, ge=0, doc="number of dust puffs")
        fade_start: float = P(0.55, ge=0, lt=1, doc="chip fade-out starts (fraction of duration)")

    def setup(self):
        p = self.p
        rng = self.rng("chips")
        n = p.count
        r = rng.uniform(2.2, 6.5, n) * p.size
        self.r = r
        self.shapes = [_chip_shape(rng, r[i]) for i in range(n)]
        ang = _dirs(rng, n, p.angle, p.spread)
        spd = rng.uniform(150, 430, n) * p.speed * np.sqrt(4.0 / (r / p.size))
        self.v0 = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        self.rot0 = rng.uniform(0, TAU, n)
        self.w = rng.uniform(-14, 14, n)
        self.life = rng.uniform(0.7, 1.0, n)
        rng = self.rng("grit")
        m = p.grit
        a2 = _dirs(rng, m, p.angle, p.spread * 1.1)
        s2 = rng.uniform(160, 520, m) * p.speed
        self.gv = np.stack([np.cos(a2) * s2, np.sin(a2) * s2], 1)
        self.gr = rng.uniform(0.6, 1.3, m) * min(p.size, 2)
        self.glife = rng.uniform(0.3, 0.75, m)
        rng = self.rng("dust")
        k = p.puffs
        a3 = _dirs(rng, k, p.angle, min(p.spread, 200) * 0.8)
        s3 = rng.uniform(70, 280, k) * p.speed
        self.dv = np.stack([np.cos(a3) * s3, np.sin(a3) * s3], 1)
        self.dr = rng.uniform(5, 10, k) * p.dust_size
        self.dd = rng.uniform(0.7, 1.0, k)
        self.dt0 = rng.uniform(0, 0.06, k)
        self.doff = rng.uniform(0, 1, (k, 2))
        self.drot = rng.uniform(0, TAU, k)
        self.tex = PuffTexture(self.rng("texture"), n=128, cells=6)

    def get_bounds(self):
        e = self.p.extent + 8 * self.p.size
        e = max(e, 60 * self.p.dust_size + 50)
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if u >= 1:
            return
        if p.dust > 0 and ctx.has("smoke"):
            sm = ctx.smoke
            for i in range(p.puffs):
                tt = t - self.dt0[i]
                if tt <= 0:
                    continue
                pos = ballistic((0.0, 0.0), self.dv[i], tt, 6.0, -30.0)
                rad = self.dr[i] * (1 + 1.8 * (1 - math.exp(-tt / 0.3)))
                a = p.dust * 0.55 * self.dd[i] * _ss(0, 0.04, tt) * _ss(1.0, 0.3, u)
                if a <= 1e-3:
                    continue
                deposit_puff(sm, self.tex, float(pos[0]), float(pos[1]), rad, a, float(self.doff[i, 0]),
                             float(self.doff[i, 1]), float(self.drot[i]) + 0.6 * tt, detail=0.8, amp=0.7,
                             thr=0.12 + 0.4 * u)
        sol = ctx.solid
        gp = ballistic(np.zeros_like(self.gv), self.gv, t, p.drag * 1.5, p.gravity)
        gq = ballistic(np.zeros_like(self.gv), self.gv, max(t - 0.018, 0), p.drag * 1.5, p.gravity)
        for j in range(len(self.gv)):
            q = u / self.glife[j]
            if q >= 1:
                continue
            a = (1 - q) ** 0.6 * _ss(p.extent, p.extent * 0.8, float(np.hypot(*gp[j])))
            sol.capsule(tuple(gq[j]), tuple(gp[j]), self.gr[j] * 0.6, self.gr[j], k=0.3, a=a)
        pos = ballistic(np.zeros_like(self.v0), self.v0, t, p.drag, p.gravity)
        fadeall = _ss(1.0, p.fade_start, u)
        for i in range(p.count):
            fade = fadeall * _ss(self.life[i], self.life[i] * p.fade_start, u)
            fade *= _ss(p.extent, p.extent * 0.8, float(np.hypot(*pos[i])))
            if fade <= 0.01:
                continue
            rot = self.rot0[i] + self.w[i] * t * math.exp(-t)
            pts = _rot(self.shapes[i], rot) + pos[i]
            sol.polygon(pts, k=0.22, a=fade, soft=0.5)
            facet = (pts - pos[i]) * 0.55 + pos[i] + LIGHT * self.r[i] * 0.3
            sol.polygon(facet, k=0.62, a=fade * 0.95, soft=0.5)
            sol.polygon((pts - pos[i]) * 0.22 + pos[i] + LIGHT * self.r[i] * 0.45, k=0.9, a=fade * 0.7, soft=0.5)


# =============================================================================
# blood
# =============================================================================
@atom("blood_splatter", tags=["impact", "blood", "volume", "liquid"])
class BloodSplatter(Atom):
    """Blood hit: a quick central splash blob, droplets flung out in a cone (each a glossy
    capsule stretched along its velocity, shrinking as it flies) and a few heavier drips that
    sag and fall. Deposits density into a ``volume`` role: use a glossy volume channel (spec +
    rim) with a dark-red ramp. Game-appropriate: small, quick, no gore."""
    kind = "oneshot"
    outputs = {"volume": Volume(palette="blood_gloss", density=1.4, bump=5, soft=0.8, shadow=0.35, wrap=0.6,
                                spec=0.9, spec_power=36, rim=0.35, size_ref=256)}
    streams = ("drops", "drips")

    class Params(_ConeParams):
        duration: float = P(0.85, gt=0, doc="seconds until everything has faded")
        count: int = P(26, ge=0, doc="number of droplets")
        size: float = P(1.0, gt=0, doc="droplet size multiplier")
        drips: int = P(3, ge=0, doc="heavy drips that sag and fall")
        splash: float = P(1.0, ge=0, doc="central splash blob amount (0 = none)")
        stretch: float = P(0.028, ge=0, doc="droplet stretch along velocity (s of travel)")
        density: float = P(1.0, ge=0, doc="density multiplier")
        fade_start: float = P(0.5, ge=0, lt=1, doc="fade-out starts (fraction of duration)")
        gravity: float = P(1100.0, doc="gravity (px/s², +down)")
        drag: float = P(2.2, ge=0, doc="air drag (1/s)")

    def setup(self):
        p = self.p
        rng = self.rng("drops")
        n = p.count
        ang = _dirs(rng, n, p.angle, p.spread)
        big = rng.random(n) < 0.25
        self.r = np.where(big, rng.uniform(2.6, 4.2, n), rng.uniform(1.0, 2.4, n)) * p.size
        spd = rng.uniform(150, 460, n) * p.speed * np.where(big, 0.7, 1.0)
        self.v0 = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], 1)
        self.p0 = np.stack([np.cos(ang), np.sin(ang)], 1) * rng.uniform(1, 6, (n, 1))
        self.life = rng.uniform(0.55, 1.0, n)
        rng = self.rng("drips")
        m = p.drips
        a2 = _dirs(rng, m, p.angle, p.spread * 0.5)
        s2 = rng.uniform(50, 110, m) * p.speed
        self.dv = np.stack([np.cos(a2) * s2, np.sin(a2) * s2], 1)
        self.dr = rng.uniform(2.8, 4.2, m) * p.size
        self.dt0 = rng.uniform(0.02, 0.1, m)
        self.splash_a = rng.uniform(0, TAU, 5)
        self.splash_k = rng.uniform(0.5, 1.0, 5)

    def get_bounds(self):
        e = self.p.extent + 6 * self.p.size
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if u >= 1:
            return
        V = ctx.volume
        fadeall = _ss(1.0, p.fade_start, u)
        D = p.density
        if p.splash > 0:
            q = t / 0.22
            if q < 1:
                d = np.array([math.cos(math.radians(p.angle)), math.sin(math.radians(p.angle))])
                a = p.splash * D * 0.9 * (1 - q) ** 1.5 * _ss(0, 0.15, q)
                R = (3 + 7 * q ** 0.5) * p.size
                V.dot(float(d[0] * R * 0.5), float(d[1] * R * 0.5), R, a)
                for j in range(5):
                    ang = math.radians(p.angle) + (self.splash_a[j] - math.pi) * min(p.spread, 300) / 360
                    L = R * (1.2 + 1.6 * q) * self.splash_k[j]
                    V.capsule((d[0] * R * 0.3, d[1] * R * 0.3), (math.cos(ang) * L, math.sin(ang) * L),
                              R * 0.45, R * 0.15, a * 0.8, soft=1.0)
        pos = ballistic(self.p0, self.v0, t, p.drag, p.gravity)
        for i in range(p.count):
            fade = fadeall * _ss(self.life[i], self.life[i] * 0.6, u)
            fade *= _ss(p.extent, p.extent * 0.8, float(np.hypot(*pos[i])))
            if fade <= 0.01:
                continue
            r = self.r[i] * (0.25 + 0.75 * fade)
            tail = ballistic(self.p0[i], self.v0[i], max(t - p.stretch * (0.25 + 0.75 * fade), 0), p.drag, p.gravity)
            V.capsule(tuple(tail), tuple(pos[i]), r * 0.35, r, D * min(1.0, 2.5 * fade), soft=1.0)
        for j in range(len(self.dv)):
            tt = t - self.dt0[j]
            if tt <= 0:
                continue
            # a drip: slow blob sagging then falling, its neck stretching back to where it left
            pos = ballistic((0.0, 0.0), self.dv[j], tt, 3.0, p.gravity * 0.55)
            neck = ballistic((0.0, 0.0), self.dv[j], tt * 0.45, 3.0, p.gravity * 0.55)
            fd = fadeall * _ss(p.extent, p.extent * 0.8, float(np.hypot(*pos)))
            if fd <= 0.01:
                continue
            a = D * min(1.0, 2.5 * fd)
            r = self.dr[j] * (0.3 + 0.7 * fd)
            neck_a = a * _ss(0.3, 0.12, tt)
            if neck_a > 0.01:
                V.capsule(tuple(neck), tuple(pos), r * 0.2, r * 0.6, neck_a, soft=1.0)
            V.capsule((float(pos[0]), float(pos[1]) - r * 0.5), (float(pos[0]), float(pos[1]) + r * 0.15),
                      r * 0.35, r * 0.75, a, soft=1.0)


# =============================================================================
# metal
# =============================================================================
@atom("metal_sparks", tags=["impact", "metal", "light", "sparks"])
class MetalSparks(Atom):
    """Metal strike: a white-hot contact flash with a short cross glint, fast thin streak sparks
    sprayed in a cone with drag and gravity, and a few slower, heavier glowing bits that arc
    down. Light role (hot ramp: orange -> white)."""
    kind = "oneshot"
    outputs = {"light": Light(palette="spark_hot", bloom=[[1.2, 0.5], [4, 0.45], [12, 0.3]], alpha_gamma=0.85)}
    streams = ("sparks", "bits", "glint")

    class Params(_ConeParams):
        duration: float = P(0.55, gt=0, doc="seconds until everything has faded")
        count: int = P(36, ge=0, doc="number of fast streak sparks")
        bits: int = P(6, ge=0, doc="slower heavy glowing bits")
        flash: float = P(1.0, ge=0, doc="contact flash brightness (0 = none)")
        flash_size: float = P(1.0, gt=0, doc="contact flash size multiplier")
        spark_width: float = P(1.0, gt=0, doc="spark thickness multiplier")
        brightness: float = P(1.0, ge=0, doc="spark brightness multiplier")
        spread: float = P(120.0, ge=0, le=360, doc="cone width of the spray (deg)")
        drag: float = P(4.0, ge=0, doc="air drag (1/s)")
        gravity: float = P(650.0, doc="gravity (px/s², +down)")

    def setup(self):
        p = self.p
        self.em = Emitter(self.rng("sparks"), EmitterConfig(
            count=p.count, angle=p.angle, spread=p.spread, speed=(320 * p.speed, 820 * p.speed),
            life=(0.18, min(0.45, p.duration)), gravity=p.gravity, drag=p.drag, stagger=0.03))
        self.bits = Emitter(self.rng("bits"), EmitterConfig(
            count=p.bits, angle=p.angle, spread=p.spread * 0.8, speed=(120 * p.speed, 300 * p.speed),
            life=(0.3, max(0.31, p.duration * 0.95)), gravity=p.gravity * 1.6, drag=p.drag * 0.4))
        self.g_ang = float(self.rng("glint").uniform(0, math.pi / 2))

    def get_bounds(self):
        e = self.p.extent + 6
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        L = ctx.light
        end = _end_fade(t, p.duration, 0.8)
        if p.flash > 0:
            fs = p.flash_size
            a = p.flash * end
            L.dot(0, 0, (4 + 40 * t) * fs, 3.2 * a * math.exp(-t / 0.035))
            L.dot(0, 0, 14 * fs, 0.5 * a * math.exp(-t / 0.08))
            ga = 2.4 * a * math.exp(-t / 0.05)
            if ga > 0.02:
                glint(L, 0.0, 0.0, (14 + 160 * t) * fs, ga, ang=self.g_ang, r=1.1 * fs)
        draw_particles(L, self.em, t, style="streak", size=0.95 * p.spark_width, brightness=2.4 * p.brightness * end,
                       brightness_var=0.35, tail=0.06, fade_in=0.0, fade_out=1.2, extent=p.extent)
        draw_particles(L, self.bits, t, style="dot", size=1.5 * p.spark_width, brightness=1.8 * p.brightness * end,
                       brightness_var=0.3, fade_in=0.0, fade_out=0.8, halo=0.25, extent=p.extent)


# =============================================================================
# energy
# =============================================================================
@atom("energy_hit", tags=["impact", "energy", "light", "electric"])
class EnergyHit(Atom):
    """Energy / magic hit: hot core flash, a fast thin ring and a slower dashed ring that
    spins, short jagged crackle arcs jumping out from the core (re-jagged ``crackle_rate`` times
    per second) and glints twinkling around the hit. Light role."""
    kind = "oneshot"
    outputs = {"light": Light(palette="electric", bloom=[[1.2, 0.6], [4, 0.6], [12, 0.45]], alpha_gamma=0.85)}
    streams = ("glints", "crackle")

    class Params(Atom.Params):
        duration: float = P(0.55, gt=0, doc="seconds until faded")
        radius: float = P(48.0, gt=0, doc="final radius of the outer ring (px)")
        core: float = P(1.0, ge=0, doc="core flash brightness")
        rings: int = P(2, ge=0, le=3, doc="number of rings (0-3)")
        dashes: int = P(14, ge=0, doc="dashes on the slower ring (0 = solid)")
        glints: int = P(6, ge=0, doc="twinkling glints")
        arcs: int = P(5, ge=0, doc="crackle arcs per tick")
        crackle_rate: float = P(30.0, gt=0, doc="crackle re-jags per second")
        crackle_time: float = P(0.6, ge=0, le=1, doc="crackle stops at this fraction of the duration")
        brightness: float = P(1.0, ge=0, doc="overall intensity")

    def setup(self):
        p = self.p
        rng = self.rng("glints")
        n = p.glints
        rad = np.sqrt(rng.uniform(0.15, 1.0, n)) * p.radius * 0.9
        ang = rng.uniform(0, TAU, n)
        self.gx, self.gy = rad * np.cos(ang), rad * np.sin(ang)
        self.gt = rng.uniform(0.02, 0.45, n) * p.duration
        self.gd = rng.uniform(0.12, 0.22, n) * p.duration / 0.55
        self.gl = rng.uniform(9, 17, n)
        self.gr = rng.uniform(0, math.pi / 2, n)

    def get_bounds(self):
        r = self.p.radius * 1.15 + 16
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if u >= 1:
            return
        L = ctx.light
        B = p.brightness
        end = _ss(1.0, 0.75, u)
        if p.core > 0:
            L.dot(0, 0, 5 + 30 * t, 3.0 * p.core * B * math.exp(-t / 0.04) * end)
            L.dot(0, 0, 12, 0.6 * p.core * B * math.exp(-t / 0.12) * end)
        R = p.radius
        specs = [(1.0, 0.0, 0.05, 2.4, 1.6, 0), (0.72, 0.04, 0.09, 1.6, 1.2, p.dashes), (0.45, 0.08, 0.07, 1.4, 1.0, 0)]
        for k in range(p.rings):
            rk, t0, tau, w0, br, dash = specs[k]
            tt = t - t0 * p.duration / 0.55
            if tt < 0:
                continue
            rr = R * rk * (1 - math.exp(-tt / (tau * p.duration / 0.55)))
            a = br * B * math.exp(-tt / (0.14 * p.duration / 0.55)) * end
            if a < 1e-3:
                continue
            L.ring(0, 0, max(rr, 0.5), w0 * (1 - 0.5 * u), a, dashes=dash, duty=0.55, phase=1.5 * u)
        for i in range(p.glints):
            q = (t - self.gt[i]) / self.gd[i]
            if 0 <= q < 1:
                e = math.sin(math.pi * q) ** 2
                glint(L, float(self.gx[i]), float(self.gy[i]), self.gl[i] * (0.6 + 0.4 * e), 2.0 * B * e,
                      ang=float(self.gr[i]) + q * 0.6, r=0.7)
        if p.arcs > 0 and u < p.crackle_time:
            fr = self.tick_rng("crackle", t, p.crackle_rate)
            ca = B * _ss(p.crackle_time, p.crackle_time * 0.4, u)
            for _ in range(p.arcs):
                a0 = fr.uniform(0, TAU)
                r0 = fr.uniform(2, 6)
                L1 = fr.uniform(0.35, 0.85) * R * (0.5 + 0.5 * min(u / 0.15, 1))
                b = a0 + fr.uniform(-0.6, 0.6)
                pts = [(r0 * math.cos(a0), r0 * math.sin(a0)),
                       (L1 * 0.5 * math.cos((a0 + b) / 2), L1 * 0.5 * math.sin((a0 + b) / 2)),
                       (L1 * math.cos(b), L1 * math.sin(b))]
                path = refine(fr, pts, rough=0.28, detail=2.0)
                L.polyline(path, 0.75, 2.2 * ca, r_end=0.4, a_end=0.6 * ca)
                L.dot(float(path[-1, 0]), float(path[-1, 1]), 1.8, 0.8 * ca)
