"""Elemental matter atoms: ice (frost nova shards, crystal shatter, cold mist, ring sparkles),
earth (ground cracks, dust cloud, rocks) and water (crown splash, jet, ripples).

Ported from the legacy ``fx_physical`` effects. Matter is non-emissive: solids paint into a
``solid`` role (colour = channel palette position ``k``), dust / water deposit density into a
``volume`` role, mist into a ``smoke`` role; light is used only for edges, glints and glows.
Lengths are legacy pixels at scale 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Solid, Volume, atom
from ..engine.mathx import TAU, refine, smoothstep
from ..engine.noise import Noise
from .combat import glint

LIGHT = np.array([-0.55, -0.85]) / math.hypot(-0.55, -0.85)     # toward the light (screen, y down)


def _ss(e0, e1, x) -> float:
    return float(smoothstep(e0, e1, x))


def _end_fade(t: float, duration: float) -> float:
    return _ss(duration, duration * 0.8, t)


def _erode(view, bounds, factor):
    """Run ``draw(private_view)`` deposits through ``factor(X, Y)`` (noise evaluated on an
    output-resolution grid) and add them to ``view``. Returns the private view to draw into
    plus a finisher."""
    pv = view.private()

    def finish():
        s = max(int(pv.ss), 1)

        def fn(X, Y, v):
            if not v.any():
                return v
            Xc, Yc = X[::s, ::s], Y[::s, ::s]
            f = np.asarray(factor(Xc, Yc), np.float32)
            if s > 1:
                f = np.repeat(np.repeat(f, s, axis=0), s, axis=1)[:v.shape[0], :v.shape[1]]
            return v * f
        pv.map(fn, bounds)
        view.add(pv)
    return pv, finish


# =============================================================================
# ice
# =============================================================================
def _ice_kite(sol, light, p, ang, L, W, alpha, edge, skew=0.0):
    """Kite-shaped ice shard: two flat-shaded facets (palette position = facet lighting)
    plus a bright outline, spine and tip in light."""
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    tip = p + d * L * 0.62
    back = p - d * L * 0.38
    mid = p + d * L * (0.02 + skew)
    sl = mid + n * W * 0.5
    sr = mid - n * W * 0.62
    for side, nn in ((sl, n), (sr, -n)):
        k = 0.5 + 0.5 * float(nn @ LIGHT)
        sol.polygon([tip, side, back], k=k ** 0.9, a=alpha * (0.72 + 0.2 * k), soft=0.55)
    if edge > 0 and light is not None:
        r = max(0.35, W * 0.035)
        lit = sl if float(n @ LIGHT) > 0 else sr
        light.polyline([back, lit, tip], r, edge * 0.5, r_end=r * 0.7, a_end=edge)
        light.polyline([back, tip], r * 0.8, edge * 0.25, a_end=edge * 0.7)
        light.dot(float(tip[0]), float(tip[1]), r * 2.2, edge * 0.6)


@atom("ice_shards", tags=["ice", "solid", "burst"])
class IceShards(Atom):
    """Frost-nova shards: big kites spread evenly around the burst plus small random ones fly
    out (exponential ease), tumble a little, shrink and fade. Facets are flat-shaded solid
    matter (channel palette = ice ramp, shadow -> lit); edges glow in light."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="ice"),
               "light": Light(palette="frost", bloom=[[1.2, 0.45], [4.5, 0.4], [14, 0.3]], exposure=1.1,
                              alpha_gamma=0.9)}
    streams = ("shards",)

    class Params(Atom.Params):
        duration: float = P(1.0, gt=0, doc="seconds until every shard has faded")
        shard_count: int = P(14, ge=0, doc="big shards (spread evenly around the burst)")
        small_shards: int = P(30, ge=0, doc="small shards (random directions)")
        shard_size: float = P(1.0, gt=0, doc="shard length multiplier")
        spread: float = P(1.0, ge=0, doc="how far shards travel (multiplier)")
        max_distance: float = P(174.0, gt=0, doc="shard tips never pass this radius (px)")
        edge: float = P(1.0, ge=0, doc="brightness of the glowing shard edges")

    def setup(self):
        p = self.p
        rng = self.rng("shards")
        NB, NS = p.shard_count, p.small_shards
        N = NB + NS
        self.n = N
        self.ang = np.concatenate([(np.arange(NB) + rng.uniform(-0.3, 0.3, NB)) / max(NB, 1) * TAU,
                                   rng.uniform(0, TAU, NS)])
        self.big = np.arange(N) < NB
        self.L = np.where(self.big, rng.uniform(36, 56, N), rng.uniform(10, 20, N)) * p.shard_size
        self.W = self.L * rng.uniform(0.28, 0.4, N)
        dist = np.where(self.big, rng.uniform(70, 112, N), rng.uniform(60, 140, N)) * p.spread
        self.dist = np.maximum(np.minimum(dist, p.max_distance - 4 - self.L * 0.62), 0)
        self.tau = rng.uniform(0.09, 0.16, N)
        self.spin = rng.uniform(-1.2, 1.2, N)
        self.tilt = rng.uniform(-0.25, 0.25, N)
        self.skew = rng.uniform(-0.12, 0.12, N)
        self.f0 = rng.uniform(0.45, 0.7, N)
        self.f1 = np.minimum(self.f0 + rng.uniform(0.2, 0.3, N), 0.94)
        self.order = np.argsort(self.L)

    def get_bounds(self):
        r = max(self.p.max_distance, float(self.L.max() * 0.62 + 8) if self.n else 0) + 6
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        tn = t / self.p.duration
        light = ctx.light if ctx.has("light") else None
        for i in self.order:
            fade = 1 - _ss(self.f0[i], self.f1[i], tn)
            if fade <= 0:
                continue
            rad = 4 + self.dist[i] * (1 - math.exp(-tn / self.tau[i]))
            pt = np.array([math.cos(self.ang[i]) * rad, math.sin(self.ang[i]) * rad])
            k = min(1.0, 0.55 + tn / 0.08) * (0.6 + 0.4 * fade)
            _ice_kite(ctx.solid, light, pt, self.ang[i] + self.tilt[i] + self.spin[i] * tn,
                      self.L[i] * k, self.W[i] * k, fade ** 0.7,
                      (1.6 if self.big[i] else 1.1) * fade * self.p.edge, self.skew[i])


@atom("ring_glints", tags=["glint", "light", "sparkle"])
class RingGlints(Atom):
    """Four-point sparkles that twinkle on and off while riding an expanding ring
    (radius r0 + (r1 - r0) * (1 - exp(-t / expand_time)), each at a random fraction of it).
    Share the ring curve with a ``shockwave`` / ``frost_mist`` layer via vars."""
    kind = "oneshot"
    outputs = {"light": Light(palette="frost")}
    streams = ("glints",)

    class Params(Atom.Params):
        duration: float = P(1.0, gt=0, doc="seconds")
        count: int = P(40, ge=0, doc="number of sparkles")
        radius_start: float = P(12.0, ge=0, doc="ring radius at t = 0 (px)")
        radius_end: float = P(152.0, ge=0, doc="ring radius it approaches (px)")
        expand_time: float = P(0.17, gt=0, doc="ring expansion time constant (s)")
        k_min: float = P(0.2, ge=0, doc="minimum fraction of the ring radius")
        k_max: float = P(1.0, ge=0, doc="maximum fraction of the ring radius")
        start_min: float = P(0.05, ge=0, doc="earliest sparkle birth (s)")
        start_max: float = P(0.6, ge=0, doc="latest sparkle birth (s)")
        life_min: float = P(0.18, gt=0, doc="minimum sparkle life (s)")
        life_max: float = P(0.32, gt=0, doc="maximum sparkle life (s)")
        size_min: float = P(4.0, ge=0, doc="minimum ray length (px)")
        size_max: float = P(9.0, ge=0, doc="maximum ray length (px)")
        brightness: float = P(2.2, ge=0, doc="peak intensity")
        angle: float = P(17.2, doc="rotation of the crosses (deg)")
        width: float = P(0.55, gt=0, doc="ray thickness (px)")

    def setup(self):
        p = self.p
        rng = self.rng("glints")
        n = p.count
        self.ang = rng.uniform(0, TAU, n)
        self.k = rng.uniform(p.k_min, max(p.k_min, p.k_max), n)
        self.t0 = rng.uniform(p.start_min, max(p.start_min, p.start_max), n)
        self.life = rng.uniform(p.life_min, max(p.life_min, p.life_max), n)
        self.Ls = rng.uniform(p.size_min, max(p.size_min, p.size_max), n)

    def get_bounds(self):
        p = self.p
        r = max(p.radius_start, p.radius_end) * max(p.k_min, p.k_max, 1e-3) + max(p.size_min, p.size_max) \
            + 5 * p.width + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        R = p.radius_start + (p.radius_end - p.radius_start) * (1 - math.exp(-t / p.expand_time))
        ef = _end_fade(t, p.duration)
        for i in range(p.count):
            u = (t - self.t0[i]) / self.life[i]
            if 0 < u < 1:
                s = math.sin(math.pi * u)
                rad = R * self.k[i]
                glint(ctx.light, math.cos(self.ang[i]) * rad, math.sin(self.ang[i]) * rad,
                      self.Ls[i] * (0.6 + 0.4 * s), s * s * p.brightness * ef, ang=math.radians(p.angle),
                      r=p.width)


@atom("frost_mist", tags=["ice", "smoke", "mist"])
class FrostMist(Atom):
    """Cold mist: soft puffs riding a ring that expands (exponentially or linearly), drifting
    up, eroded by evolving fbm noise (only its own density). Frost-nova mist, shatter puff,
    splash spray haze."""
    kind = "oneshot"
    outputs = {"smoke": Smoke(palette="steam", density=0.7, shadow=0.35)}
    streams = ("puffs", "noise")

    class Params(Atom.Params):
        duration: float = P(1.0, gt=0, doc="seconds")
        amount: float = P(1.0, ge=0, doc="density multiplier (0 = off)")
        count: int = P(90, ge=0, doc="number of puffs")
        layout: str = P("random", doc="puff angles: random, or even around the ring", choices=["random", "even"])
        radius_start: float = P(12.0, ge=0, doc="ring radius at t = 0 (px)")
        radius_end: float = P(152.0, ge=0, doc="exp: radius it approaches; linear: radius at the end (px)")
        expand: str = P("exp", doc="ring growth curve", choices=["exp", "linear"])
        expand_time: float = P(0.17, gt=0, doc="exp growth time constant (s)")
        k_max: float = P(0.92, ge=0, doc="outermost puff position (fraction of the ring radius)")
        k_spread: float = P(0.6, ge=0, doc="how far puffs scatter inward (fraction; squared distribution)")
        squash: float = P(1.0, gt=0, le=1, doc="vertical squash of the ring")
        rise: float = P(10.0, doc="upward drift over the duration (px)")
        puff_min: float = P(5.0, gt=0, doc="minimum puff radius (px)")
        puff_max: float = P(11.0, gt=0, doc="maximum puff radius (px)")
        grow_start: float = P(0.8, ge=0, doc="puff size factor at t = 0")
        grow: float = P(1.1, ge=0, doc="puff size factor added over the duration")
        density_min: float = P(0.2, ge=0, doc="minimum puff density")
        density_max: float = P(0.45, ge=0, doc="maximum puff density")
        fade_in: float = P(0.12, ge=0, le=1, doc="fade-in (fraction of duration)")
        fade_out_start: float = P(0.4, ge=0, le=1, doc="fade-out start (fraction of duration)")
        fade_out_end: float = P(0.98, ge=0, le=1, doc="fully gone (fraction of duration)")
        noise_scale: float = P(18.0, gt=0, doc="erosion noise cell size (px)")
        noise_grow: float = P(1.5, ge=0, doc="noise cells grow by this factor over the duration")
        noise_speed: float = P(1.5, ge=0, doc="noise evolution over the duration (cells)")
        noise_drift: float = P(0.0, doc="noise scroll upward over the duration (cells)")
        erosion_floor: float = P(0.0, ge=0, doc="density kept where the noise is lowest")
        erosion_gain: float = P(1.6, ge=0, doc="density gain where the noise is highest")
        erosion_lo: float = P(-0.3, doc="noise value where erosion is total")
        erosion_hi: float = P(0.5, doc="noise value where erosion stops")

    def setup(self):
        p = self.p
        rng = self.rng("puffs")
        n = p.count
        self.ang = rng.uniform(0, TAU, n) if p.layout == "random" else np.arange(n) / max(n, 1) * TAU
        self.k = p.k_max - rng.uniform(0, 1, n) ** 2 * p.k_spread if p.layout == "random" else np.full(n, p.k_max)
        self.r = rng.uniform(p.puff_min, max(p.puff_min, p.puff_max), n)
        self.a = rng.uniform(p.density_min, max(p.density_min, p.density_max), n)
        self.noise = Noise(self.rng("noise"), period=(16, 16, 8))

    def _R(self, t):
        p = self.p
        if p.expand == "exp":
            return p.radius_start + (p.radius_end - p.radius_start) * (1 - math.exp(-t / p.expand_time))
        return p.radius_start + (p.radius_end - p.radius_start) * min(t / p.duration, 1.0)

    def get_bounds(self):
        p = self.p
        R = max(p.radius_start, p.radius_end) * max(p.k_max, 1e-3)
        m = 3 * max(p.puff_min, p.puff_max) * (p.grow_start + p.grow) + 2
        return (-R - m, -R * p.squash - m - max(p.rise, 0), R + m, R * p.squash + m + max(-p.rise, 0))

    def draw(self, ctx, t):
        p = self.p
        tn = t / p.duration
        mf = p.amount * _ss(0, max(p.fade_in, 1e-4), tn) * (1 - _ss(p.fade_out_start, max(p.fade_out_end, p.fade_out_start + 1e-4), tn))
        if mf <= 1e-4 or p.count == 0:
            return
        b = self.get_bounds()
        E = 1 + p.noise_grow * tn
        sc = p.noise_scale * E
        z = 3 + p.noise_speed * tn
        dz = p.noise_drift * tn

        def factor(X, Y):
            n = self.noise.fbm(X / sc, Y / sc + dz, z, octaves=4)
            return p.erosion_floor + p.erosion_gain * smoothstep(p.erosion_lo, p.erosion_hi, n)
        pv, finish = _erode(ctx.smoke, b, factor)
        R = self._R(t)
        size = p.grow_start + p.grow * tn
        for i in range(p.count):
            rad = R * self.k[i]
            pv.dot(math.cos(self.ang[i]) * rad, math.sin(self.ang[i]) * rad * p.squash - p.rise * tn,
                   self.r[i] * size, self.a[i] * mf)
        finish()


def _crystal_pieces(rng, w, h, k):
    """Split a hexagonal crystal (front view of a pointed hex prism) into 2k fragments
    around an off-centre core. Returns list of (pts, face)."""
    outline = np.array([(0, -h * 0.5), (w * 0.5, -h * 0.28), (w * 0.5, h * 0.42), (0, h * 0.5),
                        (-w * 0.5, h * 0.42), (-w * 0.5, -h * 0.28)], np.float64)
    per = []
    for i in range(6):
        a, b = outline[i], outline[(i + 1) % 6]
        m = max(2, int(np.linalg.norm(b - a) / (h / 12)))
        for j in range(m):
            per.append(a + (b - a) * j / m)
    per = np.array(per)
    core = np.array([rng.uniform(-0.1, 0.1) * w, rng.uniform(-0.05, 0.15) * h])
    k = min(k, len(per))
    # stratified cuts (one per perimeter sector) so every seed breaks all the way round
    cuts = np.unique(((np.arange(k) + rng.uniform(0.15, 0.85, k)) * len(per) / k).astype(int) % len(per))
    k = len(cuts)
    pieces = []
    for i in range(k):
        a, b = cuts[i], cuts[(i + 1) % k]
        arc = per[a:b + 1] if b > a else np.vstack([per[a:], per[:b + 1]])
        f1, f2 = rng.uniform(0.35, 0.6, 2)
        m1 = core + (arc[0] - core) * f1
        m2 = core + (arc[-1] - core) * f2
        pieces.append(np.vstack([[m1], arc, [m2]]))
        pieces.append(np.array([core, m1, m2]))
    out = []
    for pc in pieces:
        cxp = pc[:, 0].mean()
        face = 1.0 if cxp < -w * 0.17 else (0.0 if cxp > w * 0.17 else 0.55)
        out.append((pc, face))
    return out


@atom("crystal_shatter", tags=["ice", "solid", "burst"])
class CrystalShatter(Atom):
    """Hex ice crystal that shows glowing cracks (the fragment outlines) for ``crack_time``
    seconds, then bursts: flat-shaded fragments fly ballistically with spin, outlines glow and
    fade. Pair with ``flash``, ``burst`` (glitter) and ``frost_mist`` layers starting at
    ``crack_time``. Local origin: crystal centre."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="ice"),
               "light": Light(palette="frost", bloom=[[1, 0.45], [4, 0.4], [11, 0.25]], exposure=1.1,
                              alpha_gamma=0.9)}
    streams = ("pieces", "motion")

    class Params(Atom.Params):
        duration: float = P(0.683, gt=0, doc="total seconds (crack phase + flight)")
        crack_time: float = P(0.083, ge=0, doc="seconds the intact crystal shows cracks before breaking")
        pieces: int = P(9, ge=3, doc="wedges the crystal splits into (each also splits in two)")
        width: float = P(54.0, gt=0, doc="crystal width (px)")
        height: float = P(108.0, gt=0, doc="crystal height (px)")
        burst: float = P(1.0, ge=0, doc="fragment launch speed multiplier")
        gravity: float = P(380.0, doc="gravity (px/s², +down)")
        extent: float = P(126.0, gt=0, doc="fragments fade out before reaching this distance (px)")
        edge: float = P(1.0, ge=0, doc="crack / outline glow multiplier")
        core_glow: float = P(0.5, ge=0, doc="glow inside the cracking crystal")

    def setup(self):
        p = self.p
        rng = self.rng("pieces")
        self.frags = _crystal_pieces(rng, p.width, p.height, p.pieces)
        rng = self.rng("motion")
        mv = []
        for pts, face in self.frags:
            cen = pts.mean(axis=0)
            d = cen + rng.normal(0, 2, 2)
            d = d / (np.linalg.norm(d) + 1e-6)
            sp = rng.uniform(90, 200) * p.burst
            mv.append(dict(cen=cen, rel=pts - cen, face=face,
                           v=d * sp + np.array([0, -rng.uniform(40, 120)]) * p.burst,
                           w=rng.uniform(-7, 7), f0=rng.uniform(0.45, 0.75)))
        self.mv = mv

    def get_bounds(self):
        r = max(self.p.extent, self.p.height * 0.5, self.p.width * 0.5) + 4
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        sol, light = ctx.solid, ctx.light
        ct = p.crack_time
        if t < ct:
            k = min((t + ct / 2) / (1.5 * ct), 1.0)
            for m in self.mv:
                sol.polygon(m["cen"] + m["rel"], k=0.25 + 0.6 * m["face"], a=0.92, soft=0.5)
            for m in self.mv:
                pts = m["cen"] + m["rel"]
                light.polyline(np.vstack([pts, pts[:1]]), 0.5, 1.4 * k * p.edge)
            light.dot(0, 0, 20, p.core_glow * k)
            return
        flight = max(p.duration - ct, 1e-3)
        tf = t - ct
        tn = tf / flight
        if tn >= 1:
            return
        for m in self.mv:
            pos = m["cen"] + m["v"] * tf + np.array([0, 0.5 * p.gravity * tf * tf])
            edge_d = p.extent - max(abs(pos[0]), abs(pos[1]))
            fade = (1 - _ss(m["f0"], 0.97, tn)) * _ss(26, 56, edge_d)
            if fade <= 0:
                continue
            a = m["w"] * tf
            cr, sr = math.cos(a), math.sin(a)
            rel = m["rel"] @ np.array([[cr, sr], [-sr, cr]]) * (0.75 + 0.25 * fade)
            pts = pos + rel
            kf = float(np.clip(0.25 + 0.6 * m["face"] + 0.25 * math.sin(a + m["face"] * 3), 0, 1))
            sol.polygon(pts, k=kf, a=0.9 * fade ** 0.7, soft=0.5)
            light.polyline(np.vstack([pts, pts[:1]]), 0.45, 0.9 * fade * p.edge)
            light.dot(float(pts[0][0]), float(pts[0][1]), 1.2, 0.8 * fade * p.edge)


# =============================================================================
# earth
# =============================================================================
@atom("ground_cracks", tags=["earth", "solid", "impact"])
class GroundCracks(Atom):
    """Ground-plane impact decal: a dark crater and radial jagged cracks (with lit lips) that
    race out from the impact, plus a hot glow along the cracks that cools fast. Cracks are
    solid matter (palette positions: 0 crack, 0.1 crater, 1 lip). Local origin: impact point
    on the ground; the ground plane is squashed by ``squash``."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="impact", bloom=[[1.5, 0.5], [6, 0.4]], exposure=1.0)}
    streams = ("cracks",)

    class Params(Atom.Params):
        duration: float = P(1.25, gt=0, doc="seconds (cracks fade out over the last third)")
        crack_count: int = P(9, ge=0, doc="radial cracks (some get a side branch)")
        crack_length: float = P(1.0, gt=0, doc="crack length multiplier")
        squash: float = P(0.26, gt=0, le=1, doc="ground-plane squash (perspective)")
        crater_radius: float = P(20.0, ge=0, doc="crater radius on the ground (px, before squash)")
        grow_time: float = P(0.035, gt=0, doc="crack growth time constant (s)")
        glow: float = P(0.9, ge=0, doc="hot glow along the cracks (0 = none)")
        glow_time: float = P(0.08, gt=0, doc="glow cooling time constant (s)")

    def setup(self):
        p = self.p
        rng = self.rng("cracks")
        SQ = p.squash
        cracks = []
        NC = p.crack_count
        for i in range(NC):
            a = (i + rng.uniform(-0.35, 0.35)) / NC * TAU
            Lc = min(rng.uniform(70, 150) * p.crack_length, 160) * (0.55 + 0.45 * abs(math.cos(a)))
            end = (math.cos(a) * Lc, math.sin(a) * Lc * SQ)
            pts = refine(rng, [(0.0, 0.0), end], 0.14, 8)
            cracks.append((pts, rng.uniform(3.0, 4.4), 0.0))
            for _ in range(int(rng.integers(0, 2))):
                j = int(rng.integers(len(pts) // 4, len(pts) * 3 // 4))
                b = a + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
                Lb = Lc * rng.uniform(0.25, 0.45)
                q = pts[j] + (math.cos(b) * Lb, math.sin(b) * Lb * SQ)
                cracks.append((refine(rng, [pts[j], q], 0.18, 6), 1.8, j / len(pts)))
        self.cracks = cracks
        allp = np.vstack([c[0] for c in cracks] + [np.array([[-p.crater_radius, -p.crater_radius * SQ],
                                                             [p.crater_radius, p.crater_radius * SQ]])])
        m = 12.0
        self._b = (float(allp[:, 0].min() - m), float(allp[:, 1].min() - m),
                   float(allp[:, 0].max() + m), float(allp[:, 1].max() + m))

    def get_bounds(self):
        return self._b

    def draw(self, ctx, t):
        p = self.p
        endfade = 1 - _ss(0.62, 0.97, t / p.duration)
        if endfade <= 0:
            return
        sol = ctx.solid
        th = np.linspace(0, TAU, 24, endpoint=False)
        if p.crater_radius > 0:
            sol.polygon(np.stack([np.cos(th) * p.crater_radius, np.sin(th) * p.crater_radius * p.squash], 1),
                        k=0.1, a=0.75 * endfade, soft=3)
        grow = 1 - math.exp(-t / p.grow_time)
        g = p.glow * math.exp(-t / p.glow_time)
        for pts, wd, start in self.cracks:
            k = (grow - start) / (1 - start)
            if k <= 0:
                continue
            Pp = pts[:max(2, int(len(pts) * k))]
            sol.stroke(Pp + (0, 0.9), wd * 0.55, k=1.0, a=0.35 * endfade, r_end=0.3)
            sol.stroke(Pp, wd * 0.55, k=0.0, a=0.95 * endfade, r_end=0.3)
            if g > 1e-3 and ctx.has("light"):
                ctx.light.polyline(Pp, wd * 0.25, g, r_end=0.3, a_end=0.0)


@atom("dust_cloud", tags=["earth", "volume", "dust"])
class DustCloud(Atom):
    """Ground-slam dust: billowing puffs (each with sub-billows on its upper rim) rolling out
    sideways along the ground plus a rising core, and an optional squashed dust ring riding
    the shockwave. Its own density is eroded by expanding fbm and kept off the ground's
    underside and the region edges. Render into a ``volume`` channel (dust palette).
    Local origin: impact point on the ground."""
    kind = "oneshot"
    outputs = {"volume": Volume(palette="dust", density=1.1, bump=40, soft=1.5, shadow=0.3, wrap=0.6,
                                size_ref=384)}
    streams = ("puffs", "noise")

    class Params(Atom.Params):
        duration: float = P(1.25, gt=0, doc="seconds")
        amount: float = P(1.0, ge=0, doc="dust density multiplier (0 = no cloud)")
        puffs: int = P(64, ge=0, doc="number of billow puffs")
        spread: float = P(1.0, ge=0, doc="how far the dust rolls out sideways")
        rise: float = P(1.0, ge=0, doc="how high the central dust column climbs")
        squash: float = P(0.26, gt=0, le=1, doc="ground-plane squash")
        ring: float = P(0.55, ge=0, doc="dust ring density riding the shockwave (0 = none)")
        ring_radius_start: float = P(10.0, ge=0, doc="dust ring start radius (px)")
        ring_radius_end: float = P(175.0, ge=0, doc="dust ring radius it approaches (px)")
        ring_expand_time: float = P(0.13, gt=0, doc="dust ring expansion time constant (s)")
        half_width: float = P(192.0, gt=0, doc="dust thins out before this sideways distance (px)")
        ceiling: float = P(204.0, gt=0, doc="dust thins out before this height above the ground (px)")
        under: float = P(34.0, ge=0, doc="dust fades out this far below the ground line (px)")
        noise_scale: float = P(24.0, gt=0, doc="erosion noise cell width (px; height is 5/6 of it)")

    def setup(self):
        p = self.p
        rng = self.rng("puffs")
        ND = p.puffs
        side = rng.choice([-1, 1], ND)
        self.core = rng.random(ND) < 0.14
        self.vx = np.where(self.core, rng.uniform(-60, 60, ND), side * rng.uniform(100, 360, ND) * p.spread)
        self.vy = np.where(self.core, -rng.uniform(120, 240, ND) * p.rise, -rng.uniform(5, 80, ND))
        self.r0 = rng.uniform(7, 13, ND)
        self.da = rng.uniform(0.3, 0.55, ND)
        self.dz = rng.uniform(-14, 14, ND)
        self.drag = rng.uniform(3.5, 5.0, ND)
        self.kid_a = -math.pi / 2 + rng.uniform(-1.6, 1.6, (ND, 3))
        self.kid_k = rng.uniform(0.45, 0.65, (ND, 3))
        self.noise = Noise(self.rng("noise"), period=(16, 16, 8))

    def get_bounds(self):
        p = self.p
        return (-p.half_width, -p.ceiling, p.half_width, p.under + 2)

    def draw(self, ctx, t):
        p = self.p
        if p.amount <= 0:
            return
        SQ = p.squash
        b = self.get_bounds()
        E = 1 + 2.0 * t
        sx, sy = p.noise_scale * E, p.noise_scale * 5 / 6 * E
        z = 2 + 2 * t

        def factor(X, Y):
            n = self.noise.fbm(X / sx, Y / sy, z, octaves=4)
            keep = (smoothstep(p.under, p.under * 8 / 34, Y) *
                    smoothstep(6, 52, p.half_width - np.abs(X)) * smoothstep(6, 40, Y + p.ceiling))
            return (0.2 + 1.1 * smoothstep(-0.55, 0.55, n)) * keep
        pv, finish = _erode(ctx.volume, b, factor)
        drew = False
        if p.ring > 0:
            sw = math.exp(-t / 0.3) * _ss(0, 0.03, t) * min(p.amount, 1.5)
            if sw > 0.02:
                Rw = p.ring_radius_start + (p.ring_radius_end - p.ring_radius_start) * (1 - math.exp(-t / p.ring_expand_time))
                pv.ring(0, -3, Rw * 0.97, 3 + 14 * t, p.ring * sw, squash=SQ * 1.15)
                drew = True
        env = _ss(0, 0.04, t) * (1 - _ss(0.35, 1.0, t / p.duration)) / (1 + 1.5 * t) * p.amount
        if env > 1e-4:
            for i in range(p.puffs):
                k = (1 - math.exp(-self.drag[i] * t)) / self.drag[i]
                x = self.vx[i] * k
                y = self.dz[i] * SQ + self.vy[i] * k - (0 if self.core[i] else 18 * t)
                rad = self.r0[i] * (1 + 3.0 * (1 - math.exp(-t / 0.35)))
                y = min(y, self.dz[i] * SQ - rad * 0.9)
                a = self.da[i] * env
                pv.dot(x, y, rad, a)
                for j in range(3):
                    pv.dot(x + math.cos(self.kid_a[i, j]) * rad * 0.9, y + math.sin(self.kid_a[i, j]) * rad * 0.9,
                           rad * self.kid_k[i, j], a * 0.8)
            drew = True
        if drew:
            finish()


def _rock_shape(rng, r):
    n = int(rng.integers(5, 8))
    th = np.linspace(0, TAU, n, endpoint=False) + rng.uniform(-0.25, 0.25, n)
    rad = r * rng.uniform(0.7, 1.0, n)
    return np.stack([np.cos(th) * rad, np.sin(th) * rad], 1)


@atom("rocks", tags=["earth", "solid", "debris"])
class Rocks(Atom):
    """Rock chunks (two-tone: shadow silhouette + lit facet) and pebble streaks thrown up from
    an impact, falling under gravity onto a squashed ground plane with one damped bounce and
    a slide (closed form). Solid matter (palette positions: 0.15 rock shadow, 0.35 pebble,
    0.6 rock lit face). Local origin: impact point on the ground."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="rock")}
    streams = ("rocks", "pebbles")

    class Params(Atom.Params):
        duration: float = P(1.25, gt=0, doc="seconds (debris fades over the last third)")
        rocks: int = P(20, ge=0, doc="rock chunks")
        pebbles: int = P(40, ge=0, doc="small dirt specks")
        rock_size: float = P(1.0, gt=0, doc="rock size multiplier")
        launch: float = P(1.0, ge=0, doc="launch speed multiplier")
        gravity: float = P(1100.0, gt=0, doc="gravity (px/s²)")
        squash: float = P(0.26, gt=0, le=1, doc="ground-plane squash (landing depth spread)")
        x_limit: float = P(162.0, gt=0, doc="debris never travels further sideways than this (px)")
        ceiling: float = P(190.0, gt=0, doc="debris never flies higher than this above the ground (px)")
        bounce: float = P(0.25, ge=0, le=1, doc="bounce restitution")

    def setup(self):
        p = self.p
        rng = self.rng("rocks")
        NR = p.rocks
        self.rk = [_rock_shape(rng, rng.uniform(3.5, 9.0) * p.rock_size) for _ in range(NR)]
        self.r_vx = rng.uniform(-230, 230, NR) * p.launch
        self.r_vy = -rng.uniform(230, 430, NR) * p.launch
        self.r_z = rng.uniform(-18, 18, NR)
        self.r_w = rng.uniform(-14, 14, NR)
        rng = self.rng("pebbles")
        NP = p.pebbles
        self.p_vx = rng.uniform(-260, 260, NP) * p.launch
        self.p_vy = -rng.uniform(150, 380, NP) * p.launch
        self.p_z = rng.uniform(-18, 18, NP)

    def get_bounds(self):
        p = self.p
        m = 9.0 * p.rock_size + 4
        return (-p.x_limit - m, -p.ceiling - m, p.x_limit + m, 18 * p.squash + m)

    def _ballistic(self, x0, y0, vx, vy, zl, t):
        p = self.p
        G = p.gravity
        yl = zl * p.squash
        tl = (-vy + math.sqrt(vy * vy + 2 * G * max(yl - y0, 0))) / G
        if t < tl:
            x, y = x0 + vx * t, y0 + vy * t + 0.5 * G * t * t
        else:
            xl = x0 + vx * tl
            bvy = -p.bounce * (vy + G * tl)
            tb = t - tl
            y = yl + bvy * tb + 0.5 * G * tb * tb
            if y > yl:
                x, y = xl + 0.3 * vx * min(tb, -2 * bvy / G), yl
            else:
                x = xl + 0.3 * vx * tb
        return float(np.clip(x, -p.x_limit, p.x_limit)), max(y, -p.ceiling)

    def draw(self, ctx, t):
        p = self.p
        endfade = 1 - _ss(0.62, 0.97, t / p.duration)
        if endfade <= 0:
            return
        sol = ctx.solid
        for i in range(p.rocks):
            x, y = self._ballistic(self.r_vx[i] * 0.05, -6, self.r_vx[i], self.r_vy[i], self.r_z[i], t)
            shape = self.rk[i]
            r = float(np.abs(shape).max())
            y -= r * 0.4
            rot = self.r_w[i] * min(t, 0.6)
            cr, sr = math.cos(rot), math.sin(rot)
            pts = shape @ np.array([[cr, sr], [-sr, cr]]) + (x, y)
            sol.polygon(pts, k=0.15, a=endfade, soft=0.5)
            facet = (pts - (x, y)) * 0.55 + (x, y) + LIGHT * r * 0.36
            sol.polygon(facet, k=0.6, a=endfade * 0.9, soft=0.6)
        for i in range(p.pebbles):
            x, y = self._ballistic(0, -4, self.p_vx[i], self.p_vy[i], self.p_z[i], t)
            x2, y2 = self._ballistic(0, -4, self.p_vx[i], self.p_vy[i], self.p_z[i], max(t - 0.025, 0))
            sol.capsule((x2, y2), (x, y), 0.7, 1.1, k=0.35, a=endfade)


# =============================================================================
# water
# =============================================================================
@atom("ripple", tags=["water", "volume", "ring"])
class Ripple(Atom):
    """Expanding density ring on a ground/water plane (water ripples, dust rings). Deposits
    into a ``volume`` role (route it to the water body or a dust channel)."""
    kind = "oneshot"
    outputs = {"volume": Volume(palette="water", bump=6, soft=1, shadow=0.15, wrap=0.8, spec=1.0,
                                spec_power=30, rim=0.8, size_ref=320)}

    class Params(Atom.Params):
        duration: float = P(1.1, gt=0, doc="seconds")
        radius_start: float = P(20.0, ge=0, doc="radius at t = 0 (px)")
        radius_end: float = P(160.0, ge=0, doc="radius it approaches (px)")
        expand_time: float = P(0.35, gt=0, doc="expansion time constant (s)")
        width: float = P(1.5, gt=0, doc="ring width at t = 0 (px, gaussian sigma)")
        width_grow: float = P(3.0, ge=0, doc="width added per second (px/s)")
        density: float = P(0.45, ge=0, doc="peak density")
        fade_time: float = P(0.45, gt=0, doc="density time constant (s)")
        fade_start: float = P(0.55, ge=0, le=1, doc="final fade-out starts (fraction of duration)")
        squash: float = P(0.3, gt=0, le=1, doc="vertical squash (ground ellipse)")

    def get_bounds(self):
        p = self.p
        r = max(p.radius_start, p.radius_end) + 3 * (p.width + p.width_grow * p.duration) + 2
        return (-r, -r * p.squash - 3 * (p.width + p.width_grow * p.duration) - 2, r,
                r * p.squash + 3 * (p.width + p.width_grow * p.duration) + 2)

    def draw(self, ctx, t):
        p = self.p
        surf = 1 - _ss(p.fade_start * p.duration, p.duration, t)
        a = p.density * math.exp(-t / p.fade_time) * surf
        if a < 1e-4:
            return
        R = p.radius_start + (p.radius_end - p.radius_start) * (1 - math.exp(-t / p.expand_time))
        ctx.volume.ring(0, 0, R, p.width + p.width_grow * t, a, squash=p.squash)


def _cap(view, p0, p1, r0, r1, a, soft=0.6):
    view.capsule((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1])), r0, r1, a, soft=soft)


@atom("water_crown", tags=["water", "volume", "splash"])
class WaterCrown(Atom):
    """Side-view crown splash: a continuous water sheet rising around the impact, breaking
    into tapered fingers with rim beads; drops pinch off the finger tips (and fast spray
    drops leave the impact), fly ballistically as stretched capsules with a light glint each,
    and vanish where they land. Water density goes to the ``volume`` role (render as one
    glossy water body), glints to ``light``, the spray haze to ``smoke``. Local origin:
    impact point on the water surface."""
    kind = "oneshot"
    outputs = {"volume": Volume(palette="water", bump=6, soft=1, shadow=0.15, wrap=0.8, spec=1.0,
                                spec_power=30, rim=0.8, size_ref=320),
               "light": Light(palette="glint", bloom=[[1, 0.4], [3, 0.25]], exposure=1.2),
               "smoke": Smoke(palette="steam", density=1.0, shadow=0.0, size_ref=320)}
    streams = ("fingers", "drops", "spray", "noise")

    class Params(Atom.Params):
        duration: float = P(1.1, gt=0, doc="seconds")
        fingers: int = P(40, ge=6, doc="fingers around the crown rim (each sheds 1-2 drops)")
        height: float = P(92.0, ge=0, doc="crown height (px)")
        width: float = P(1.0, gt=0, le=1.6, doc="crown radius multiplier")
        squash: float = P(0.3, gt=0, le=1, doc="ground-plane squash of the crown ring")
        drop_amount: float = P(1.0, ge=0, doc="multiplier on drops shed per finger (0 = none)")
        spray_drops: int = P(26, ge=0, doc="fast small droplets thrown at the impact")
        gravity: float = P(1100.0, gt=0, doc="gravity for drops (px/s²)")
        gloss: float = P(1.0, ge=0, doc="drop glint brightness")
        mist_amount: float = P(1.0, ge=0, doc="spray mist density (0 = off)")
        half_width: float = P(160.0, gt=0, doc="sideways extent of the splash region (px)")
        ceiling: float = P(236.0, gt=0, doc="drops vanish above this height (px above the surface); top of the region")
        floor: float = P(83.0, gt=0, doc="extent below the surface line (px)")

    def setup(self):
        p = self.p
        rng = self.rng("fingers")
        NF = p.fingers
        self.NF = NF
        self.phi = (np.arange(NF) + rng.uniform(-0.3, 0.3, NF)) / NF * TAU
        self.fh = rng.uniform(0.75, 1.15, NF)
        self.fd = rng.uniform(0.14, 0.26, NF)
        self.fr = rng.uniform(2.0, 3.6, NF)
        self.order = np.argsort(np.sin(self.phi))
        SQ = p.squash
        rng = self.rng("drops")
        drops = []
        for j in range(NF):
            for k in range(int(round(rng.integers(1, 3) * p.drop_amount))):
                td = self.fd[j] + k * rng.uniform(0.03, 0.08)
                x0, y0 = self._tip(j, td)
                out_v = rng.uniform(25, 70)
                drops.append(dict(t0=td, x0=x0, y0=y0, vx=math.cos(self.phi[j]) * out_v + rng.normal(0, 10),
                                  vy=-rng.uniform(120, 300), r=self.fr[j] * (0.9 if k == 0 else 0.6),
                                  yl=math.sin(self.phi[j]) * 50 * SQ, front=math.sin(self.phi[j]) > 0))
        rng = self.rng("spray")
        for _ in range(p.spray_drops):
            a = rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
            v = rng.uniform(180, 380)
            drops.append(dict(t0=0.0, x0=0.0, y0=-4.0, vx=math.cos(a) * v * 0.45, vy=math.sin(a) * v,
                              r=rng.uniform(1.0, 2.0), yl=rng.uniform(-6, 10), front=True))
        self.drops = drops
        self.noise = Noise(self.rng("noise"), period=(16, 16, 8))

    def _crown(self, t):
        p = self.p
        Rb = 12 + 46 * p.width * (1 - math.exp(-t / 0.12))
        H = p.height * (1 - math.exp(-t / 0.06)) * (1 - _ss(0.22, 0.62, t)) ** 1.2
        return Rb, Rb * (1.25 + 0.5 * min(t / 0.3, 1)), H

    def _tip(self, j, t):
        Rb, Rr, H = self._crown(t)
        return (Rr * math.cos(self.phi[j]), Rr * math.sin(self.phi[j]) * self.p.squash - H * self.fh[j])

    def get_bounds(self):
        p = self.p
        return (-p.half_width, -p.ceiling, p.half_width, p.floor)

    def draw(self, ctx, t):
        p = self.p
        SQ = p.squash
        W = ctx.volume
        Rb, Rr, H = self._crown(t)
        surf = 1 - _ss(0.55, 1.0, t / p.duration)
        if surf > 0:
            W.ring(0, 0, Rb * 1.1, 4 + 4 * t, 0.9 * surf, squash=SQ)
        NF = self.NF
        if H > 1:
            hk_ = _ss(0.1, 0.45, H / max(p.height, 1e-6))
            tips = [self._tip(j, t) for j in range(NF)]
            base = [(Rb * math.cos(ph), Rb * math.sin(ph) * SQ) for ph in self.phi]
            for j in self.order:
                dens = 0.55 if math.sin(self.phi[j]) > 0 else 0.3
                j2 = (j + 1) % NF
                quad = []
                for jj, hk in ((j, 0.0), (j2, 0.0), (j2, 0.68), (j, 0.68)):
                    tx, ty = tips[jj]
                    bx, by = base[jj]
                    quad.append((bx + (tx - bx) * hk, by + (ty - by) * hk))
                W.polygon(quad, dens * (0.4 + 0.6 * hk_), soft=1.0)
            for j in self.order:
                front = math.sin(self.phi[j])
                bx, by = base[j]
                tx, ty = tips[j]
                mx, my = bx + (tx - bx) * 0.45, by + (ty - by) * 0.45
                wt = max(self.fr[j] * (1 - _ss(self.fd[j] - 0.05, self.fd[j], t)) * 0.8, 0.8)
                W.capsule((mx, my), (tx, ty), TAU * Rb / NF * 0.45 + 1.0, wt * 0.7,
                          (0.6 if front > 0 else 0.35) * hk_, soft=0.8)
                if t < self.fd[j]:
                    W.dot(tx, ty, self.fr[j] * 0.6, 1.2)
        G = p.gravity
        light = ctx.light if ctx.has("light") else None
        top = -p.ceiling + 12
        for d in self.drops:
            tt = t - d["t0"]
            if tt <= 0:
                continue
            x = d["x0"] + d["vx"] * tt
            y = d["y0"] + d["vy"] * tt + 0.5 * G * tt * tt
            vy = d["vy"] + G * tt
            if y > d["yl"] or y < top or abs(x) > p.half_width - 4:
                continue
            sp = math.hypot(d["vx"], vy) + 1e-6
            st = min(sp * 0.018, 14)
            px, py = x - d["vx"] / sp * st, y - vy / sp * st
            W.capsule((px, py), (x, y), d["r"] * 0.45, d["r"], 1.3 if d["front"] else 0.8, soft=0.6)
            if light is not None and p.gloss > 0:
                light.dot(x - d["r"] * 0.35, y - d["r"] * 0.4, d["r"] * 0.32, 1.3 * p.gloss)
        if ctx.has("smoke") and p.mist_amount > 0:
            mf = p.mist_amount * _ss(0.02, 0.15, t) * (1 - _ss(0.35, 0.95, t / p.duration))
            if mf > 1e-4:
                def factor(X, Y):
                    n = self.noise.fbm(X / 22, Y / 22 + t * 2, t * 2, octaves=4)
                    return 0.3 + 1.0 * smoothstep(-0.5, 0.6, n)
                pv, finish = _erode(ctx.smoke, self.get_bounds(), factor)
                for k in range(14):
                    a = k / 14 * TAU
                    pv.dot(math.cos(a) * Rr * 1.1, math.sin(a) * Rr * SQ - H * 0.8 - 12 * t, 10 + 22 * t, 0.08 * mf)
                finish()


@atom("water_jet", tags=["water", "volume", "splash"])
class WaterJet(Atom):
    """Worthington jet: a central column of water shoots up and sinks back, launching a bead
    off its top that flies up and falls. Water density into ``volume``, a glint on the bead
    into ``light``. Local origin: the jet's base on the water surface."""
    kind = "oneshot"
    outputs = {"volume": Volume(palette="water", bump=6, soft=1, shadow=0.15, wrap=0.8, spec=1.0,
                                spec_power=30, rim=0.8, size_ref=320),
               "light": Light(palette="glint", bloom=[[1, 0.4], [3, 0.25]], exposure=1.2)}

    class Params(Atom.Params):
        duration: float = P(0.55, gt=0, doc="seconds the jet lives")
        height: float = P(105.0, gt=0, doc="jet height (px)")
        width: float = P(6.0, gt=0, doc="jet radius at the base (px)")
        bead_speed: float = P(230.0, ge=0, doc="launch speed of the top bead (px/s)")
        bead_launch: float = P(0.35, ge=0, le=1, doc="bead launches at this fraction of the duration")
        gravity: float = P(1100.0, gt=0, doc="gravity on the bead (px/s²)")
        gloss: float = P(1.0, ge=0, doc="bead glint brightness")

    def get_bounds(self):
        p = self.p
        top = max(p.height, p.height * 0.9 + p.bead_speed ** 2 / (2 * p.gravity)) + 12
        w = max(p.width, 4.5) + 4
        return (-w, -top, w, 12)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if not 0 <= u < 1:
            return
        W = ctx.volume
        hj = p.height * math.sin(math.pi * min(u * 1.3, 1)) ** 0.8
        wj = p.width * (1 - 0.4 * u)
        if hj > 2:
            W.capsule((0.0, 2.0), (0.0, -hj), wj, wj * 0.55, 0.9, soft=0.8)
        if u > p.bead_launch:
            tb = (u - p.bead_launch) * p.duration
            yb = -p.height * 0.9 - p.bead_speed * tb + 0.5 * p.gravity * tb * tb
            e = _end_fade(t, p.duration)
            if yb < 0 and e > 0:
                W.capsule((0.0, yb + 3), (0.0, yb), 4.5, 4.0, 1.4 * e, soft=0.6)
                if ctx.has("light") and p.gloss > 0:
                    ctx.light.dot(-1.6, yb - 1.8, 1.4, 1.3 * p.gloss * e)
