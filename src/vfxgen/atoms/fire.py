"""Fire atoms: noise flames, fireball trail, explosion cloud, burning debris, muzzle burst.

* ``flame``         teardrop noise flame (bonfire / torch styles), base at the origin
* ``flame_trail``   fireball head + streaming trail, head at the origin, facing +x
* ``fire_cloud``    explosion fireball cooling into smoke (light + smoke from the same puffs)
* ``debris_trails`` burning debris heads with smoke trails (light + smoke)
* ``muzzle_burst``  muzzle flash petals + side spikes + core, facing +x
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, atom
from ..engine.mathx import TAU, smoothstep
from ..engine.noise import Noise
from .smoke import PuffTexture, puff_eval

_FIRE_BLOOM = [[1.5, 0.2], [6, 0.18], [20, 0.14]]


def _end_fade(t: float, duration: float) -> float:
    return float(smoothstep(duration, duration * 0.85, t))


# =============================================================================
# flame
# =============================================================================
_STYLE = {
    # noise periods (x, y, z) of the main / warp noise; whether the vertical streak detail is used
    "fire": dict(N=(64, 8, 2), W=(64, 4, 1)),
    "torch": dict(N=(64, 6, 2), W=(64, 3, 1)),
}


@atom("flame", tags=["fire", "light", "loop"])
class Flame(Atom):
    """Teardrop noise flame: upward-scrolling, sideways-warped fbm added to a teardrop
    envelope that tapers with height, so noise peaks break off into licking tongues and wisps.

    ``style: fire`` = bonfire (vertical streak detail, cooler flanks, white-hot core near the
    base); ``style: torch`` = compact candle/torch flame with a slightly darker root. The origin
    is the centre of the flame base; the flame grows toward -y. Integer scroll cycles per period
    keep the loop seamless."""
    kind = "loop"
    outputs = {"light": Light(palette="fire", bloom=_FIRE_BLOOM, exposure=0.9, alpha_gamma=0.65)}
    streams = ("noise", "warp", "streaks")
    anchors = {"base": (0.0, 0.0)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        style: str = P("fire", doc="fire (bonfire) or torch (compact flame)", choices=["fire", "torch"])
        height: float = P(240.0, gt=0, doc="nominal flame height (px)")
        half_width: float = P(68.0, gt=0, doc="half width of the flame base (px)")
        cell_x: float = P(15.0, gt=0, doc="horizontal noise cell size (px)")
        cell_y: float = P(46.0, gt=0, doc="vertical noise cell size (px)")
        speed: int = P(1, ge=1, doc="rise speed: noise periods scrolled per loop (whole number)")
        noise_base: float = P(0.5, ge=0, doc="noise strength at the base (turbulence)")
        noise_tip: float = P(1.45, ge=0, doc="noise strength at the tip (tongues, wisps)")
        octaves: int = P(5, ge=1, le=7, doc="noise octaves")
        top: float = P(1.15, ge=0, doc="how fast the envelope closes toward the tip")
        sway: float = P(1.0, ge=0, doc="sideways lick / domain-warp amount")
        streaks: float = P(0.7, ge=0, doc="fast vertical streak detail (fire style; 0 = off)")
        core: float = P(1.0, ge=0, doc="extra heat of the white-hot core near the base (fire style)")
        intensity: float = P(1.0, ge=0, doc="brightness multiplier")
        tip_fade: float = P(1.275, gt=0, doc="height (in flame heights) where wisps are fully faded")

    def setup(self):
        st = _STYLE[self.p.style]
        self.N = Noise(self.rng("noise"), period=st["N"])
        self.W = Noise(self.rng("warp"), period=st["W"])
        self.D = Noise(self.rng("streaks"), period=(64, 16, 3))

    def get_bounds(self):
        p = self.p
        w = p.half_width * (1.75 + 0.35 * p.sway) + 6
        return (-w, -p.height * p.tip_fade - 6, w, p.height * 0.3 + 6)

    def field(self, X, Y, tn):
        p = self.p
        H, hw = p.height, p.half_width
        N, W = self.N, self.W
        v = -Y / H                                   # 0 at base, 1 at nominal tip
        vp = np.clip(v, 0, None)
        sway = W.fbm(X / (p.cell_x * 3), Y / (p.cell_y * 3) + tn * W.py, tn * W.pz, octaves=2) * p.sway
        u = X / hw + sway * 0.55 * vp ** 1.2
        n = N.fbm(X / p.cell_x, Y / p.cell_y + tn * N.py * p.speed, tn * N.pz, octaves=p.octaves)
        wv = np.clip(1.0 - 0.75 * vp, 0.08, None) ** 0.9 * (0.9 + 0.1 * smoothstep(-0.06, 0.16, v))
        shape = 1.0 - (np.abs(u) / wv) ** 1.8 - p.top * vp ** 1.1
        vb = v - 0.05 * u * u                        # rounded, bowl-shaped bottom
        shape = shape - np.clip(-vb, 0, None) ** 2 * 50
        amp = p.noise_base + (p.noise_tip - p.noise_base) * np.clip(v, 0, 1.2)
        F = shape + amp * n
        if p.style == "fire":
            m = smoothstep(0.0, 0.1, F)
            heat = 1.5 * np.exp(-1.6 * vp)
            I = m * (0.3 + 1.2 * np.clip(F, 0, 1.6) ** 1.3) * heat
            if p.streaks > 0:
                d = self.D.fbm(X / 10, Y / 30 + tn * 16 * p.speed, tn * 3, octaves=3)
                I = I * np.clip(0.75 + p.streaks * d, 0.15, None)
            I = I * (0.55 + 0.45 * np.exp(-(X / (0.676 * hw)) ** 2))          # cooler flanks
            hot = np.exp(-(X / (0.441 * hw)) ** 2) * np.exp(-((Y + 0.125 * H) / (0.1667 * H)) ** 2)
            I = I * (1 + 0.9 * p.core * hot)
        else:
            m = smoothstep(0.0, 0.12, F)
            heat = 1.6 * np.exp(-1.5 * vp)
            I = m * (0.3 + 1.3 * np.clip(F, 0, 1.4) ** 1.2) * heat
            I = I * (0.55 + 0.45 * smoothstep(-0.04, 0.12, v))                # darker root
        return I * smoothstep(p.tip_fade, p.tip_fade * 0.82, v) * p.intensity

    def draw(self, ctx, t):
        tn = (t / self.p.period) % 1.0
        ctx.light.field(lambda X, Y: self.field(X, Y, tn), self.get_bounds(), lowres="nearest")


# =============================================================================
# flame trail (fireball)
# =============================================================================
@atom("flame_trail", tags=["fire", "light", "loop", "projectile"])
class FlameTrail(Atom):
    """Fireball head + flame trail in head-relative coordinates, facing +x (rotate the layer to
    aim it). One flame field: a round hot head plus a widening tail, both modulated by fbm that
    scrolls backwards (whole noise periods per loop), so the trail streams and tears into wisps.
    Origin = head centre."""
    kind = "loop"
    outputs = {"light": Light(palette="fire", bloom=[[1.5, 0.25], [6, 0.25], [18, 0.2]], exposure=0.95,
                              alpha_gamma=0.6)}
    streams = ("noise", "warp")
    anchors = {"head": (0.0, 0.0), "tail": lambda self, t: (-self.p.trail_length, 0.0)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        head_radius: float = P(26.0, gt=0, doc="radius of the hot head (px)")
        trail_length: float = P(210.0, gt=0, doc="flame trail length scale (px)")
        trail_width: float = P(1.0, gt=0, doc="trail thickness multiplier")
        turbulence: float = P(1.0, ge=0, doc="noise strength: wobble, tongues and wisp break-up")
        speed: int = P(1, ge=1, doc="trail streaming speed in noise periods per loop (whole number)")
        cell: float = P(28.0, gt=0, doc="noise cell size (px)")
        intensity: float = P(1.0, ge=0, doc="brightness multiplier")

    def setup(self):
        self.N = Noise(self.rng("noise"), period=(12, 64, 3))
        self.Wp = Noise(self.rng("warp"), period=(6, 64, 2))

    def get_bounds(self):
        p = self.p
        h = 2.0 * p.head_radius * p.trail_width + 45 * p.turbulence + 6
        return (-1.22 * p.trail_length - 4, -h, 1.7 * p.head_radius + 6, h)

    def draw(self, ctx, t):
        p = self.p
        tn = (t / p.period) % 1.0
        RH, L, C = p.head_radius, p.trail_length, p.cell

        def f(X, Y):
            dx = -X                                                # distance behind the head
            warp = self.Wp.fbm(X / (C * 2) + tn * 6 * p.speed, Y / (C * 2), tn * 2, octaves=2)
            dyw = Y + warp * 38 * p.turbulence * np.clip(dx / L, 0, 1.5) ** 1.2
            n = self.N.fbm(X / (C * 1.5) + tn * 12 * p.speed, Y / (C * 0.55), tn * 3, octaves=5)
            a = np.clip(dx, 0, None) / L                           # 0 at head, 1 at tail
            width = RH * p.trail_width * (1.0 + 0.9 * a) * np.clip(1 - 0.5 * a, 0.2, None)
            trail = 1.0 - np.abs(dyw / width) ** 1.7 - 0.75 * a ** 1.3
            dfront = np.hypot(np.clip(-dx, 0, None), Y) / RH
            trail = np.where(dx < 0, 1.0 - dfront ** 2, trail)   # round front
            amp = (0.55 + 0.8 * np.clip(a, 0, 1.3)) * p.turbulence
            F = trail + amp * n
            m = smoothstep(0.0, 0.12, F)
            heat = 1.7 * np.exp(-3.0 * a) + 0.12
            I = m * (0.25 + np.clip(F, 0, 1.6) ** 1.6) * heat
            return I * smoothstep(1.213 * L, 0.994 * L, dx) * p.intensity
        ctx.light.field(f, self.get_bounds(), lowres="nearest")


# =============================================================================
# fire cloud (explosion)
# =============================================================================
@atom("fire_cloud", tags=["fire", "smoke", "explosion", "oneshot"])
class FireCloud(Atom):
    """Explosion fireball cooling into billowing smoke.

    A cluster of noise-eroded puffs expands with a decelerating radius and rises. Each puff
    carries its own heat that decays (inner puffs stay hot longest): the SAME puff deposits
    fire into ``light`` (eroded harder as it cools, shrinking into hot pockets) and smoke into
    ``smoke`` (dark fringe early, then eroded away). Origin = blast centre. For the legacy
    front-loaded timing put ``time_warp: {curve: power, exp: 1.35}`` on the layer."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire", bloom=[[2, 0.25], [8, 0.2], [24, 0.15]], alpha_gamma=0.75),
               "smoke": Smoke(palette="explosion_smoke", density=1.2, shadow=1.0, size_ref=384,
                              lit_by={"channels": ["light"], "amount": 0.8})}
    streams = ("puffs", "texture")
    anchors = {"blast": (0.0, 0.0)}

    class Params(Atom.Params):
        duration: float = P(1.5, gt=0, doc="total length (s)")
        radius: float = P(84.0, gt=0, doc="fireball radius reached by the fast blast (px)")
        creep: float = P(30.0, ge=0, doc="slow radius growth after the blast (px/s)")
        expand_time: float = P(0.09, gt=0, doc="blast expansion time constant (s)")
        count: int = P(22, ge=1, doc="number of billow puffs forming the cloud")
        rise: float = P(36.0, doc="buoyant rise of the whole cloud (px at t = 1 s, grows as t^1.4)")
        puff_rise: float = P(36.0, ge=0, doc="extra per-puff rise speed, random up to this (px/s)")
        intensity: float = P(1.0, ge=0, doc="fire brightness multiplier")
        burn_time: float = P(1.0, gt=0, doc="how long the fire keeps burning inside the smoke (multiplier)")
        smoke: float = P(1.0, ge=0, doc="smoke density multiplier (0 = pure fireball)")
        turbulence: float = P(1.0, ge=0, doc="billow / break-up noise strength")
        smoke_fade: float = P(0.51, ge=0, doc="time the smoke starts dissolving (s)")
        erode_start: float = P(0.36, ge=0, doc="time the smoke starts eroding into wisps (s)")
        smoke_gone: float = P(1.4, gt=0, doc="time the smoke has fully dissolved (s; capped at `duration`)")

    def setup(self):
        p = self.p
        rng = self.rng("puffs")
        n = p.count
        self.P = P_ = dict(th=rng.uniform(0, TAU, n), d=np.sqrt(rng.uniform(0.0, 1.0, n)) * 0.72,
                           r=rng.uniform(0.3, 0.58, n), rise=rng.uniform(0, 1, n) * p.puff_rise,
                           spin=rng.normal(0, 1.2, n), rot=rng.uniform(0, TAU, n),
                           ox=rng.random(n), oy=rng.random(n))
        P_["d"][0], P_["r"][0] = 0.0, 0.72                     # core puff
        P_["tau"] = (0.10 + 0.24 * (1 - P_["d"] / 0.72) + rng.uniform(-0.03, 0.05, n)) * p.burn_time
        self.tex = PuffTexture(self.rng("texture"), n=256, cells=8, octaves=5)
        self._stash = None
        b = [1e9, 1e9, -1e9, -1e9]
        for k in range(61):
            for x, y, r, *_ in self._puffs(p.duration * k / 60):
                R = 2.0 * r
                b = [min(b[0], x - R), min(b[1], y - R), max(b[2], x + R), max(b[3], y + R)]
        self._bounds = tuple(float(v) for v in (b[0] - 2, b[1] - 2, b[2] + 2, b[3] + 2))

    def get_bounds(self):
        return self._bounds

    def _puffs(self, t):
        p, P_ = self.p, self.P
        R = p.radius * (1 - math.exp(-t / p.expand_time)) + p.creep * t
        cy = -p.rise * t ** 1.4
        out = []
        for i in range(p.count):
            th = P_["th"][i]
            dist = P_["d"][i] * R
            x = math.cos(th) * dist
            y = cy + math.sin(th) * dist * 0.85 - P_["rise"][i] * t * (1 - 0.5 * math.sin(th))
            r = P_["r"][i] * R * (1 + 0.25 * t)
            rot = P_["rot"][i] + P_["spin"][i] * t
            heat = math.exp(-t / P_["tau"][i])
            out.append((x, y, r, rot, heat, P_["ox"][i], P_["oy"][i]))
        return out

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        gone = min(p.smoke_gone, p.duration)
        fade = 1 - float(smoothstep(min(p.smoke_fade, gone * 0.999), gone, t))
        erode = 0.18 + 0.75 * float(smoothstep(min(p.erode_start, gone * 0.999), gone, t))
        end = _end_fade(t, p.duration)
        for x, y, r, rot, heat, ox, oy in self._puffs(t):
            if r < 0.3:
                continue
            Rb = 2.0 * r
            box = (x - Rb, y - Rb, x + Rb, y + Rb)
            cache = {}

            def body_n(X, Y, x=x, y=y, r=r, ox=ox, oy=oy, rot=rot):
                key = X.shape
                if key not in cache:
                    cache[key] = puff_eval(self.tex, X, Y, x, y, r, ox, oy, rot, detail=0.6)
                return cache[key]
            sa = (0.5 + 0.9 * float(smoothstep(0.1, 0.8, 1 - heat))) * fade * p.smoke
            if sa > 1e-4:
                def fs(X, Y, sa=sa):
                    body, n = body_n(X, Y)
                    return sa * 1.6 * np.clip(body * (1 + 0.8 * p.turbulence * n) - erode, 0, None)
                ctx.smoke.field(fs, box, lowres="bilinear")
            fa = 3.4 * heat ** 1.4 * p.intensity * end
            if fa > 0.06:
                def fl(X, Y, fa=fa, heat=heat):
                    body, n = body_n(X, Y)
                    hot = body * (1 + 1.1 * p.turbulence * n) - (0.2 + 0.95 * (1 - heat))
                    return fa * np.clip(hot, 0, None) ** 0.9
                ctx.light.field(fl, box, lowres="bilinear")


# =============================================================================
# debris trails
# =============================================================================
@atom("debris_trails", tags=["fire", "smoke", "explosion", "oneshot"])
class DebrisTrails(Atom):
    """Burning debris chunks thrown from a blast, each a hot head trailing smoke.

    Heads (``light``) and their smoke trails (``smoke``) follow the same ballistic path
    (drag on the launch velocity, plain gravity). Origin = blast centre."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire", bloom=[[2, 0.25], [8, 0.2], [24, 0.15]], alpha_gamma=0.75),
               "smoke": Smoke(palette="explosion_smoke", density=1.2, shadow=1.0, size_ref=384)}
    streams = ("debris",)

    class Params(Atom.Params):
        duration: float = P(1.5, gt=0, doc="total length (s)")
        count: int = P(8, ge=0, doc="number of burning debris chunks")
        angle: float = P(-90.0, doc="mean launch direction (deg, clockwise from +x; -90 = up)")
        spread: float = P(160.0, ge=0, le=360, doc="launch cone width (deg)")
        speed_min: float = P(260.0, ge=0, doc="minimum launch speed (px/s)")
        speed_max: float = P(500.0, ge=0, doc="maximum launch speed (px/s)")
        drag: float = P(0.9, ge=0, doc="drag on the launch velocity (1/s)")
        gravity: float = P(260.0, doc="gravity (px/s², +down)")
        life_min: float = P(0.55, gt=0, doc="minimum burn time (s)")
        life_max: float = P(1.15, gt=0, doc="maximum burn time (s)")
        stagger: float = P(0.16, ge=0, doc="launch times spread over this many seconds")
        trail_time: float = P(0.6, gt=0, doc="smoke trail length (s of travel)")
        head_size: float = P(2.6, gt=0, doc="burning head radius (px)")
        brightness: float = P(2.4, ge=0, doc="head intensity")
        smoke: float = P(1.0, ge=0, doc="smoke trail density multiplier")
        smoke_fade: float = P(0.51, ge=0, doc="time the trail smoke starts dissolving (s)")
        smoke_gone: float = P(1.4, gt=0, doc="time the trail smoke has fully dissolved (s; capped at `duration`)")
        extent: float = P(175.0, gt=0, doc="debris fades out before reaching this distance (px)")

    def setup(self):
        p = self.p
        rng = self.rng("debris")
        n = p.count
        self.ang = np.radians(p.angle) + np.radians(p.spread) * (rng.random(n) - 0.5)
        self.spd = rng.uniform(min(p.speed_min, p.speed_max), max(p.speed_min, p.speed_max), n)
        self.life = rng.uniform(min(p.life_min, p.life_max), max(p.life_min, p.life_max), n)
        self.t0 = rng.uniform(0, p.stagger, n) if p.stagger > 0 else np.zeros(n)

    def get_bounds(self):
        e = self.p.extent + 6 * 3 + 2
        return (-e, -e, e, e)

    def _pos(self, i, q):
        p = self.p
        q = np.asarray(q, np.float64)
        k = (1 - np.exp(-p.drag * q)) / p.drag if p.drag > 1e-9 else q
        x = math.cos(self.ang[i]) * self.spd[i] * k
        y = math.sin(self.ang[i]) * self.spd[i] * k + 0.5 * p.gravity * q * q
        return np.stack([x, y], axis=-1)

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        gone = min(p.smoke_gone, p.duration)
        fade = (1 - float(smoothstep(min(p.smoke_fade, gone * 0.999), gone, t))) * p.smoke
        for i in range(len(self.ang)):
            life, t0 = float(self.life[i]), float(self.t0[i])
            tt = min(t, life) - t0
            if tt <= 0:
                continue
            pts = self._pos(i, np.linspace(max(0.0, tt - p.trail_time), tt, 12))
            hx, hy = float(pts[-1, 0]), float(pts[-1, 1])
            df = (1 - float(smoothstep(0.7, 1.0, (tt + t0) / life))) \
                * float(smoothstep(p.extent, 0.77 * p.extent, max(abs(hx), abs(hy))))
            if df <= 1e-4:
                continue
            if fade > 1e-4:
                ctx.smoke.polyline(pts, 6.0, 0.35 * df * fade, r_end=2.0, a_end=1.3 * df * fade)
            if t < life:
                ctx.light.dot(hx, hy, p.head_size, p.brightness * df)
                ctx.light.polyline(pts[-5:], 0.8, 0.0, r_end=2.2 * p.head_size / 2.6, a_end=1.2 * df)


# =============================================================================
# muzzle burst
# =============================================================================
@atom("muzzle_burst", tags=["fire", "light", "oneshot", "muzzle"])
class MuzzleBurst(Atom):
    """Gun muzzle flash facing +x: a jagged forward cone of tapered flame petals, a short side
    star and a white-hot core, collapsing within a few hundredths of a second. Petals re-jitter
    ``tick_rate`` times per second. Origin = muzzle point."""
    kind = "oneshot"
    outputs = {"light": Light(palette="fire", bloom=[[1.5, 0.5], [5, 0.4], [14, 0.3]], alpha_gamma=0.7)}
    streams = ("petals", "jitter")
    anchors = {"muzzle": (0.0, 0.0)}

    class Params(Atom.Params):
        duration: float = P(0.25, gt=0, doc="total length (s)")
        length: float = P(150.0, gt=0, doc="forward cone length (px)")
        petals: int = P(9, ge=1, doc="number of flame petals in the cone")
        spread: float = P(9.2, ge=0, doc="petal angle scatter (deg, sigma)")
        petal_width: float = P(6.0, gt=0, doc="petal half width (px); the centre petal is 1.5x")
        spikes: int = P(5, ge=0, doc="number of side star spikes")
        spike_length: float = P(34.0, ge=0, doc="side spike length (px)")
        intensity: float = P(1.0, ge=0, doc="flash brightness multiplier")
        decay: float = P(0.05, gt=0, doc="flash envelope time constant (s)")
        tick_rate: float = P(24.0, gt=0, doc="petal re-jitter rate (Hz)")
        jitter: float = P(1.7, ge=0, doc="per-tick petal angle jitter (deg, sigma)")

    def setup(self):
        p = self.p
        rng = self.rng("petals")
        self.pet_ang = np.radians(rng.normal(0, 1, p.petals) * p.spread)
        self.pet_ang[0] = 0.0
        self.pet_len = rng.uniform(0.45, 1.0, p.petals)
        self.pet_len[0] = 1.0
        self.side = np.linspace(0, TAU, p.spikes, endpoint=False) + rng.uniform(0, TAU)
        self.side_len = rng.uniform(0.5, 1.0, p.spikes)

    def get_bounds(self):
        p = self.p
        r = max(p.spike_length + 8, 92.0)
        return (-r, -r, max(1.12 * p.length + 12, 124.0), r)

    def draw(self, ctx, t):
        p = self.p
        e = math.exp(-t / p.decay) * p.intensity * _end_fade(t, p.duration)
        if e <= 0.01 or t >= p.duration:
            return
        fr = self.tick_rng("jitter", t, p.tick_rate)
        L = p.length * (0.65 + 0.35 * e)
        for i in range(p.petals):
            a = float(self.pet_ang[i]) + fr.normal(0, math.radians(p.jitter))
            ln = L * float(self.pet_len[i]) * fr.uniform(0.8, 1.1)
            wid = p.petal_width * (1.5 if i == 0 else 1.0) * (0.6 + 0.4 * e)
            tip = (math.cos(a) * ln, math.sin(a) * ln)
            mid = (math.cos(a) * ln * 0.35, math.sin(a) * ln * 0.35)
            nx, ny = -math.sin(a), math.cos(a)
            poly = [(0.0, -wid * 0.4 * ny), (mid[0] + nx * wid, mid[1] + ny * wid), tip,
                    (mid[0] - nx * wid, mid[1] - ny * wid), (0.0, wid * 0.4 * ny)]
            ctx.light.polygon(poly, 0.9 * e * (1.0 if i == 0 else 0.7), soft=2.5 * ctx.px)
        for j in range(p.spikes):
            a = float(self.side[j])
            ln = p.spike_length * float(self.side_len[j]) * (0.5 + 0.5 * e)
            ctx.light.polyline([(0.0, 0.0), (math.cos(a) * ln, math.sin(a) * ln)], 2.6, 1.2 * e,
                               r_end=0.4, a_end=0.1 * e)
        ctx.light.dot(6, 0, 9, 3.0 * e)
        ctx.light.dot(30, 0, 30, 0.35 * e)
