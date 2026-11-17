"""Projectile atoms: ``projectile_core`` (the flying head: plasma orb / teardrop or faceted
crystal shard, facing +x) and ``muzzle_ring`` (launch rings shooting forward along +x).

Everything faces +x with the head / muzzle at the local origin, so a layer can be rotated to aim
it and a scene pivot can sit exactly on the head. The rest of a projectile family (trail, sparks,
flashes, impact debris) is composed in YAML from ``ribbon``, ``flame_trail``, ``particles``,
``burst``, ``flash``, ``shockwave``, ``ice_shards`` ...
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Solid, atom
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise

_LIGHT_DIR = np.array([-0.55, -0.83])          # light from the upper left (matches ice_shards)


@atom("projectile_core", tags=["projectile", "orb", "shard", "light", "solid", "loop"])
class ProjectileCore(Atom):
    """The head of a projectile at the local origin, facing +x.

    * ``orb``: plasma ball: white-hot core, body with a noisy spinning rim, soft halo. ``stretch``
      > 1 elongates the back into a teardrop (speed look).
    * ``shard``: faceted crystal dart pointing +x: four flat-shaded solid facets that roll
      ``spin`` times per period (the ridge line sweeps across), glowing edges, tip glint and an
      inner glow (``solid`` role = facet ramp, shadow -> lit; ``light`` role = glow).

    Pulse / flicker / spin are whole cycles per period, so the loop is seamless. Use
    ``time_mode: once`` + ``length`` to show it briefly in launch / impact scenes."""
    kind = "loop"
    outputs = {"light": Light(palette="arcane", bloom=[[1.5, 0.5], [5, 0.45], [16, 0.35]], exposure=1.0,
                              alpha_gamma=0.8),
               "solid": Solid(palette="ice")}
    streams = ("noise", "flicker")
    anchors = {"head": (0.0, 0.0)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        style: str = P("orb", doc="orb (plasma ball / teardrop) or shard (faceted crystal)", choices=["orb", "shard"])
        radius: float = P(12.0, gt=0, doc="head radius (orb) / half length (shard) (px)")
        stretch: float = P(1.6, ge=1, doc="orb: how far the back is drawn out (x radius behind the head)")
        thickness: float = P(0.42, gt=0, le=1, doc="shard: width / length ratio")
        brightness: float = P(1.0, ge=0, doc="body intensity")
        core: float = P(1.6, ge=0, doc="white-hot centre intensity")
        rim: float = P(0.6, ge=0, doc="orb: bright noisy rim intensity; shard: edge glow")
        turbulence: float = P(0.5, ge=0, doc="orb: rim noise amount")
        halo: float = P(0.35, ge=0, doc="soft halo intensity")
        halo_radius: float = P(2.6, gt=0, doc="halo radius (x radius)")
        spin: int = P(2, doc="rim noise / shard roll turns per period (whole number)")
        pulse: float = P(0.12, ge=0, le=1, doc="size / brightness pulse depth")
        pulse_cycles: int = P(4, ge=0, doc="pulses per period (whole number)")
        flicker: float = P(0.0, ge=0, le=1, doc="random per-tick brightness flicker")
        flicker_rate: float = P(20.0, gt=0, doc="flicker changes per second")

    def setup(self):
        self.noise = Noise(self.rng("noise"), period=(8, 8, 4))

    def get_bounds(self):
        p = self.p
        R = p.radius * (1 + p.pulse * 0.3)
        h = max(p.halo_radius * R * 3.0, R * 1.6) + 2
        back = max(h, R * (p.stretch if p.style == "orb" else 0.6) * 1.5 + 2)
        return (-back, -h, h, h)

    def _k(self, t, tn):
        p = self.p
        k = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * tn) if p.pulse_cycles else 1.0
        if p.flicker:
            k *= 1 - p.flicker * float(self.tick_rng("flicker", t, p.flicker_rate).random())
        return k

    def draw(self, ctx, t):
        p = self.p
        tn = (t % p.period) / p.period
        k = self._k(t, tn)
        if p.style == "orb":
            self._orb(ctx.light, tn, k)
        else:
            self._shard(ctx, tn, k)

    def _orb(self, v, tn, k):
        p = self.p
        R = p.radius * (1 + 0.3 * (k - 1))
        st = p.stretch
        N = self.noise
        hr = p.halo_radius * R

        def f(X, Y):
            Xs = np.where(X < 0, X / st, X)
            r = np.hypot(Xs, Y)
            ph = np.arctan2(Y, Xs) + TAU * p.spin * tn
            n = N.fbm(np.cos(ph) * 1.5 + 4, np.sin(ph) * 1.5 + 4, tn * 4, octaves=2)
            Rn = R * (1 + p.turbulence * 0.35 * n)
            body = smoothstep(Rn * 1.05, Rn * 0.35, r)
            rim = np.exp(-0.5 * ((r - Rn * 0.9) / (0.16 * R)) ** 2) * (0.6 + 0.8 * np.clip(n + 0.5, 0, 1))
            core = np.exp(-0.5 * (np.hypot(X * 0.8, Y) / (0.38 * R)) ** 2)
            halo = np.exp(-0.5 * (np.hypot(np.where(X < 0, X / (0.5 + 0.5 * st), X), Y) / hr) ** 2)
            return k * (p.brightness * body + p.rim * rim + p.core * core) + p.halo * k * halo
        v.field(f, self.get_bounds())

    def _shard(self, ctx, tn, k):
        p = self.p
        L = 2 * p.radius
        W = L * p.thickness
        tip = np.array([L * 0.62, 0.0])
        back = np.array([-L * 0.38, 0.0])
        top = np.array([-L * 0.02, -W / 2])
        bot = np.array([-L * 0.02, W / 2])
        roll = TAU * p.spin * tn
        ridge = np.array([-L * 0.02, 0.42 * W * math.sin(roll)])
        tip_r = tip + np.array([0, 0])
        # facet normals of a fake 3D diamond rolling about x; shade = palette position
        facets = []
        for side, sgn in ((top, -1), (bot, 1)):
            lit = 0.5 + 0.4 * sgn * -math.cos(roll) * 0.8 + 0.1 * (-sgn)
            facets.append(([tip_r, side, ridge], min(max(lit + 0.18, 0.05), 1.0)))
            facets.append(([back, side, ridge], min(max(lit - 0.12, 0.02), 1.0)))
        sol = ctx.solid
        for poly, kk in facets:
            sol.polygon(poly, k=kk, a=0.96, soft=0.5)
        v = ctx.light
        hr = p.halo_radius * p.radius
        v.dot(0, 0, hr * 0.55, p.halo * k)
        v.dot(L * 0.05, 0, p.radius * 0.45, p.core * 0.35 * k)
        e = p.rim * k
        if e > 0:
            r = max(0.45, W * 0.04)
            v.polyline([back, top, tip], r, e * 0.5, r_end=r * 0.8, a_end=e)
            v.polyline([back, bot, tip], r, e * 0.25, r_end=r * 0.8, a_end=e * 0.6)
            v.polyline([back, ridge, tip], r * 0.8, e * 0.2, a_end=e * 0.8)
            v.dot(float(tip[0]), float(tip[1]), r * 2.6, e * 1.2)


@atom("muzzle_ring", tags=["projectile", "launch", "ring", "light", "oneshot"])
class MuzzleRing(Atom):
    """Launch rings: ellipses seen edge-on (thin in x, tall in y) that pop out of the muzzle at
    the origin and shoot forward along +x while they widen and fade, one after another
    (``count`` rings ``stagger`` s apart). The near side (+x) is brighter."""
    kind = "oneshot"
    outputs = {"light": Light(palette="arcane", bloom=[[1.5, 0.5], [5, 0.4], [14, 0.3]], exposure=1.0,
                              alpha_gamma=0.85)}
    streams = ()

    class Params(Atom.Params):
        duration: float = P(0.4, gt=0, doc="seconds until the last ring is gone")
        count: int = P(2, ge=1, le=8, doc="number of rings")
        stagger: float = P(0.06, ge=0, doc="seconds between rings")
        radius_start: float = P(6.0, ge=0, doc="ring radius at birth (px)")
        radius_end: float = P(30.0, ge=0, doc="ring radius at the end (px)")
        shrink: float = P(0.7, ge=0, le=1, doc="each later ring is this much smaller")
        squash: float = P(0.35, gt=0, le=1, doc="x / y radius ratio (perspective)")
        travel: float = P(60.0, doc="forward travel along +x over a ring's life (px)")
        offset: float = P(4.0, doc="x where the rings are born (px)")
        width: float = P(2.0, gt=0, doc="ring thickness (px)")
        brightness: float = P(1.2, ge=0, doc="peak intensity")
        fade: float = P(1.6, ge=0, doc="fade-out exponent over a ring's life")

    def _life(self):
        p = self.p
        return max(p.duration - p.stagger * (p.count - 1), 1e-3)

    def get_bounds(self):
        p = self.p
        R = max(p.radius_start, p.radius_end) + 3 * p.width + 2
        return (min(0.0, p.offset) - R * p.squash - 4, -R, max(p.offset + p.travel, 0) + R * p.squash + 4, R)

    def draw(self, ctx, t):
        p = self.p
        life = self._life()
        for i in range(p.count):
            q = (t - i * p.stagger) / life
            if not 0 <= q < 1:
                continue
            e = 1 - (1 - q) ** 3
            sc = (1 - p.shrink) ** i if p.shrink < 1 else 1.0
            sc = 1 - (1 - sc) * 0.6
            R = (p.radius_start + (p.radius_end - p.radius_start) * e) * sc
            cx = p.offset + p.travel * e * (1 - 0.15 * i)
            a = p.brightness * (1 - q) ** p.fade * smoothstep(0, 0.08, q) * sc
            if a < 1e-4 or R <= 0:
                continue
            rx, ry, w = max(R * p.squash, 1e-3), max(R, 1e-3), p.width

            def f(X, Y, cx=cx, rx=rx, ry=ry, a=a, w=w):
                dx, dy = (X - cx) / rx, Y / ry
                rr = np.hypot(dx, dy)
                d = (rr - 1) * np.sqrt((rx * dx) ** 2 + (ry * dy) ** 2) / np.maximum(rr, 1e-3)
                side = 0.7 + 0.3 * np.clip(dx / np.maximum(rr, 1e-3), -1, 1)
                return a * side * np.exp(-0.5 * (d / w) ** 2)
            m = 3 * w + 1
            ctx.light.field(f, (cx - rx - m, -ry - m, cx + rx + m, ry + m))
