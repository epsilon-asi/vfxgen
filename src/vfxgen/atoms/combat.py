"""Combat / wind atoms: whirlwind funnel, sword-slash crescent, radial speed lines,
hit-impact star and the four-point cross flash.

Ported from the legacy ``fx_physical`` effects (whirlwind, slash, slash_combo,
hit_impact). Lengths are legacy pixels at scale 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, atom
from ..engine.mathx import TAU, smoothstep


def _end_fade(t: float, duration: float) -> float:
    return float(smoothstep(duration, duration * 0.8, t))


def glint(view, x: float, y: float, L: float, a: float, ang: float = 0.0, r: float = 0.5) -> None:
    """Four-point star sparkle: two crossed tapered strokes and a hot core (legacy ``_glint``)."""
    if a <= 1e-4 or L <= 0:
        return
    for k in range(2):
        th = ang + k * math.pi / 2
        dx, dy = math.cos(th) * L, math.sin(th) * L
        view.polyline([(x - dx, y - dy), (x, y)], r * 0.3, 0.0, r_end=r, a_end=a)
        view.polyline([(x, y), (x + dx, y + dy)], r, a, r_end=r * 0.3, a_end=0.0)
    view.dot(x, y, r * 1.6, a * 0.8)


# =============================================================================
# whirlwind
# =============================================================================
@atom("wind_funnel", tags=["wind", "light", "loop", "vortex"])
class WindFunnel(Atom):
    """Tornado funnel: soft wide bands and thin streaks spiral up a swaying funnel, with
    specks circling the base. Seamless loop: every band/streak rises once and turns a whole
    number of times per period.

    Local origin: the funnel base (ground) centre; the funnel rises toward -y."""
    kind = "loop"
    outputs = {"light": Light(palette="wind", bloom=[[2.4, 0.35], [8, 0.25]], exposure=1.1, alpha_gamma=1.0)}
    streams = ("streaks", "bands", "specks")

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s): one rise and whole turns per period")
        height: float = P(324.0, gt=0, doc="funnel height (px)")
        base_radius: float = P(10.0, ge=0, doc="funnel radius at the base (px)")
        top_radius: float = P(92.0, ge=0, doc="funnel radius at the top (px)")
        flare: float = P(1.35, gt=0, doc="radius profile exponent (higher = narrow longer, flares late)")
        sway: float = P(14.0, ge=0, doc="sideways snaking amplitude at the top (px)")
        streaks: int = P(340, ge=0, doc="number of thin wind streaks")
        bands: int = P(50, ge=0, doc="number of soft wide bands behind the streaks")
        dust: int = P(70, ge=0, doc="number of specks circling the base")
        spin: int = P(1, ge=1, doc="whole-number spin multiplier (turns per period x 2..3)")
        streak_width: float = P(1.4, gt=0, doc="streak thickness (px)")
        band_width: float = P(7.0, gt=0, doc="band thickness (px)")
        brightness: float = P(0.9, ge=0, doc="streak intensity")
        band_brightness: float = P(0.25, ge=0, doc="band intensity")
        dust_radius: float = P(80.0, ge=0, doc="outer radius of the base speck ring (px)")
        dust_size: float = P(1.4, gt=0, doc="speck size (px)")
        dust_brightness: float = P(1.0, ge=0, doc="speck intensity multiplier")

    def setup(self):
        p = self.p
        rng = self.rng("streaks")
        N = p.streaks
        self.S = dict(h0=rng.random(N), th0=rng.uniform(0, TAU, N), k=rng.choice([2, 3], N) * p.spin,
                      rj=rng.uniform(0.75, 1.06, N), a=rng.uniform(0.5, 1.0, N), arc=rng.uniform(0.7, 1.6, N))
        rng = self.rng("bands")
        NB = p.bands
        self.B = dict(h0=rng.random(NB), th0=rng.uniform(0, TAU, NB), k=rng.choice([2, 3], NB) * p.spin,
                      rj=rng.uniform(0.8, 1.05, NB), arc=rng.uniform(1.6, 2.6, NB))
        rng = self.rng("specks")
        ND = p.dust
        self.D = dict(th0=rng.uniform(0, TAU, ND), k=rng.choice([1, 2], ND) * p.spin,
                      rad=rng.uniform(0.2, 1.0, ND) * p.dust_radius, lift=rng.uniform(0, 20, ND),
                      a=rng.uniform(0.2, 0.55, ND))

    def get_bounds(self):
        p = self.p
        wmax = 3 * max(p.band_width, p.streak_width)
        x = max(max(p.top_radius, p.base_radius) * 1.06 + p.sway + wmax, p.dust_radius + 3 * p.dust_size) + 2
        y0 = -p.height - 0.21 * max(p.top_radius, p.base_radius) * 1.06 - wmax - 2
        y1 = max(8 + 0.22 * p.dust_radius + 3 * p.dust_size, 0.21 * p.base_radius * 1.06 + wmax) + 2
        return (-x, y0, x, y1)

    def _radius(self, hh):
        p = self.p
        return p.base_radius + (p.top_radius - p.base_radius) * hh ** p.flare

    def _streak(self, view, u, hh, th, k, rj, arc, width, a):
        fade = math.sin(math.pi * hh) ** 1.3
        depth = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(th))
        amp = a * depth * fade
        if amp < 1e-3:
            return
        ths = th - np.sign(k) * np.linspace(arc, 0, 9)          # tail -> head
        rad = self._radius(hh) * rj
        cx = self.p.sway * hh * math.sin(TAU * (0.9 * hh - u))
        xs = cx + rad * np.cos(ths)
        ys = -hh * self.p.height + 0.2 * rad * np.sin(ths)
        view.polyline(np.stack([xs, ys], 1), width * 0.6, amp * 0.15, r_end=width, a_end=amp)

    def draw(self, ctx, t):
        p = self.p
        v = ctx.light
        u = (t / p.period) % 1.0
        B, S, D = self.B, self.S, self.D
        for i in range(p.bands):
            hh = (B["h0"][i] + u) % 1.0
            th = B["th0"][i] + TAU * B["k"][i] * u
            self._streak(v, u, hh, th, B["k"][i], B["rj"][i], B["arc"][i], p.band_width, p.band_brightness)
        for i in range(p.streaks):
            hh = (S["h0"][i] + u) % 1.0
            th = S["th0"][i] + TAU * S["k"][i] * u
            self._streak(v, u, hh, th, S["k"][i], S["rj"][i], S["arc"][i], p.streak_width,
                         p.brightness * S["a"][i])
        for i in range(p.dust):
            th = D["th0"][i] + TAU * D["k"][i] * u
            x = D["rad"][i] * math.cos(th)
            y = 8 - D["lift"][i] + 0.22 * D["rad"][i] * math.sin(th)
            depth = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(th))
            v.dot(x, y, p.dust_size, D["a"][i] * depth * p.dust_brightness)


# =============================================================================
# slash
# =============================================================================
def _sweep_curve(tt: float, hit: float = 0.3):
    """Head/tail positions (0..1 along the arc) and brightness at normalised slash time."""
    u_h = 0.35 + 0.65 * (1 - (1 - min(tt / hit, 1)) ** 2.2)
    u_t = 0.98 * float(smoothstep(0.08, 1.0, tt)) ** 1.15
    amp = 1.0 if tt < hit else (1 - float(smoothstep(hit, 1.0, tt))) ** 0.8
    return u_h, min(u_t, u_h), amp


def _mid_arc(self, t):
    return tuple(float(v) for v in self.arc_point(0.5, self.p.radius - self.p.thickness * 0.2)[0])


def _head(self, t):
    u_h = _sweep_curve(min(max(t / self.p.duration, 0.0), 1.0))[0]
    return tuple(float(v) for v in self.arc_point(u_h)[0])


@atom("crescent_slash", tags=["slash", "light", "combat"])
class CrescentSlash(Atom):
    """Sword-slash crescent: the arc whips in, the tail chases the head out. Hot cutting
    edge + glow body, thin speed lines along the arc and (optional) sparks shed off the edge.

    Local origin: the crescent's bounding-box centre (``through`` > 0 pulls it toward the
    blade's midpoint so several slashes cross there). Anchors: ``mid_arc`` (blade midpoint,
    for cross flashes) and ``head`` (leading tip at time t). The natural length is the sweep
    ``duration`` or the end of the last spark, whichever is later."""
    kind = "oneshot"
    outputs = {"light": Light(palette="steel", bloom=[[1.2, 0.5], [4, 0.45], [12, 0.3]], exposure=1.25,
                              alpha_gamma=0.9)}
    streams = ("shape", "sparks")
    anchors = {"mid_arc": _mid_arc, "head": _head}

    class Params(Atom.Params):
        duration: float = P(0.333, gt=0, doc="sweep length (s): head whips in, tail chases it out")
        radius: float = P(100.0, gt=0, doc="arc radius (px)")
        thickness: float = P(46.0, gt=0, doc="crescent thickness at its widest (px)")
        arc_angle: float = P(165.0, ge=30, le=300, doc="degrees of arc the blade sweeps")
        squash: float = P(0.6, gt=0, le=1, doc="vertical squash of the arc (1 = circle)")
        angle: float = P(0.0, doc="rotation of the slash (deg, clockwise)")
        angle_jitter: float = P(26.0, ge=0, doc="random extra rotation, +/- degrees")
        flip: int = P(-1, ge=-1, le=1, doc="-1 random, 0 normal, 1 upside-down (mirrored swing)")
        through: float = P(0.0, ge=0, le=1, doc="0 = origin at the crescent's box centre, 1 = at the blade midpoint")
        edge: float = P(1.0, ge=0, doc="brightness of the hot cutting edge")
        brightness: float = P(1.0, ge=0, doc="overall intensity (crescent + speed lines)")
        speed_lines: int = P(7, ge=0, doc="thin speed lines along the cut")
        line_brightness: float = P(0.5, ge=0, doc="speed line intensity")
        sparks: int = P(0, ge=0, doc="sparks shed off the cutting edge")
        spark_delay: float = P(0.18, ge=0, doc="sparks spawn over this fraction of the sweep, following the head")
        spark_speed_min: float = P(150.0, ge=0, doc="minimum spark speed (px/s)")
        spark_speed_max: float = P(320.0, ge=0, doc="maximum spark speed (px/s)")
        spark_life_min: float = P(0.15, gt=0, doc="minimum spark life (s)")
        spark_life_max: float = P(0.25, gt=0, doc="maximum spark life (s)")
        spark_gravity: float = P(300.0, doc="spark gravity (px/s², +down)")
        spark_drag: float = P(5.0, gt=0, doc="spark air drag (1/s)")
        spark_width: float = P(0.7, gt=0, doc="spark thickness (px)")
        spark_brightness: float = P(1.8, ge=0, doc="spark intensity")

    def setup(self):
        p = self.p
        rng = self.rng("shape")
        fl = int(rng.integers(0, 2)) if p.flip < 0 else int(p.flip)
        self.rot = math.radians(p.angle + rng.uniform(-1, 1) * p.angle_jitter) + math.pi * fl
        self.sweep = math.radians(p.arc_angle)
        self.a0 = math.radians(272.5) - self.sweep / 2
        n = p.speed_lines
        self.lu = rng.uniform(0.25, 0.95, n)
        self.lr = p.radius + rng.uniform(-0.3, 0.12, n) * p.radius
        self.ll = rng.uniform(0.16, 0.34, n) * p.radius
        self.c = np.zeros(2)
        pts = np.array([self.arc_point(u, p.radius - p.thickness * 0.3)[0] for u in np.linspace(0, 1, 33)])
        ref = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2])
        ref = ref * (1 - p.through) + pts[16] * p.through
        self.c = -ref                                      # arc centre in local coords
        rng = self.rng("sparks")
        sp = []
        lo, hi = min(p.spark_life_min, p.spark_life_max), max(p.spark_life_min, p.spark_life_max)
        for _ in range(p.sparks):
            u = rng.uniform(0.3, 0.95)
            pt, tg = self.arc_point(u)
            ang = math.atan2(tg[1], tg[0]) + rng.normal(0, 0.45)
            v = np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(p.spark_speed_min, max(p.spark_speed_min, p.spark_speed_max))
            sp.append((p.duration * p.spark_delay * (u - 0.3) / 0.65, pt, v, rng.uniform(lo, hi),
                       rng.uniform(0.67, 1.33) * p.spark_brightness, rng.uniform(0.71, 1.29) * p.spark_width))
        self.spk = sp
        self._bounds = self._compute_bounds()

    def arc_point(self, u, rad=None):
        """Local point (and unit tangent) at fraction u along the arc."""
        ph = self.a0 + self.sweep * u
        rad = self.p.radius if rad is None else rad
        sq = self.p.squash
        lx, ly = rad * math.cos(ph), rad * math.sin(ph) * sq
        tx, ty = -math.sin(ph), math.cos(ph) * sq
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        tn = math.hypot(tx, ty)
        return (np.array([self.c[0] + lx * cr - ly * sr, self.c[1] + lx * sr + ly * cr]),
                np.array([tx * cr - ty * sr, tx * sr + ty * cr]) / tn)

    def length(self) -> float:
        end = max((t0 + life for t0, _, _, life, _, _ in self.spk), default=0.0)
        return float(max(self.p.duration, end + 1e-3))

    def _compute_bounds(self):
        p = self.p
        pts = []
        for u in np.linspace(0, 1, 49):
            for rad in (p.radius - p.thickness, p.radius * 1.13):
                pt, tg = self.arc_point(u, rad)
                pts += [pt, pt + tg * 0.45 * p.radius, pt - tg * 0.45 * p.radius]
        pts = np.array(pts)
        m = 4.0
        if self.spk:
            vmax = max(p.spark_speed_min, p.spark_speed_max)
            lmax = max(p.spark_life_min, p.spark_life_max)
            m += vmax / p.spark_drag + 0.5 * abs(p.spark_gravity) * lmax * lmax + 3 * p.spark_width * 1.3
        return (float(pts[:, 0].min() - m), float(pts[:, 1].min() - m),
                float(pts[:, 0].max() + m), float(pts[:, 1].max() + m))

    def get_bounds(self):
        return self._bounds

    def _crescent(self, view, u_t, u_h, thick, amp, edge, body, peak=1.6, soft=0.6):
        if u_h - u_t < 1e-3 or amp <= 0:
            return
        p = self.p
        R, sq, a0, sweep = p.radius, p.squash, self.a0, self.sweep
        cx, cy = float(self.c[0]), float(self.c[1])
        cr, sr = math.cos(self.rot), math.sin(self.rot)
        win = min(1.0, (u_h - u_t) / 0.6) ** 0.7

        def f(X, Y):
            dx, dy = X - cx, Y - cy
            lx = dx * cr + dy * sr
            ly = (-dx * sr + dy * cr) / sq
            r = np.hypot(lx, ly)
            phi = np.arctan2(ly, lx)
            u = (((phi - a0) * np.sign(sweep)) % TAU) / abs(sweep)
            v = (u - u_t) / (u_h - u_t)
            vc = np.clip(v, 0, 1)
            th = thick * np.sin(math.pi * vc ** peak) ** 0.85 * win
            d = R - r
            inside_v = smoothstep(-0.02, 0.0, v) * smoothstep(1.02, 1.0, v)
            outer = smoothstep(-soft, soft, d)
            dn = np.clip(d / np.maximum(th, 1e-3), 0, 1.5)
            inner = smoothstep(1.0, 0.75, dn)
            along = 0.25 + 0.75 * vc ** 0.8
            ew = max(0.8, thick * 0.07)
            taper = np.clip(th / (ew * 4), 0, 1)
            return amp * inside_v * outer * along * (
                body * (1 - np.clip(dn, 0, 1)) ** 1.3 * inner + edge * np.exp(-np.maximum(d, 0) / ew) * taper)

        m = R + 4
        b = self._bounds
        box = (max(cx - m, b[0]), max(cy - m, b[1]), min(cx + m, b[2]), min(cy + m, b[3]))
        if box[0] < box[2] and box[1] < box[3]:
            view.field(f, box)

    def draw(self, ctx, t):
        p = self.p
        v = ctx.light
        tt = t / p.duration
        if 0 <= tt <= 1:
            u_h, u_t, amp = _sweep_curve(tt)
            self._crescent(v, u_t, u_h, p.thickness * (1 - 0.3 * tt), amp * 1.2 * p.brightness,
                           edge=2.8 * p.edge, body=0.9)
            if tt <= 0.75:
                k = (1 - tt / 0.75) * p.line_brightness * p.brightness
                for i in range(p.speed_lines):
                    if self.lu[i] > u_h:
                        continue
                    pt, tg = self.arc_point(self.lu[i], self.lr[i])
                    off = 0.1 * p.radius * tt
                    v.polyline([pt - tg * (self.ll[i] + off), pt + tg * off], 0.3, 0.0, r_end=0.7, a_end=k)
        drag, g = p.spark_drag, p.spark_gravity
        for t0, p0, v0, life, amp, wid in self.spk:
            q = t - t0
            if not 0 < q < life:
                continue
            fade = (1 - q / life) ** 1.3

            def pos(s):
                kk = (1 - math.exp(-drag * s)) / drag
                return p0 + v0 * kk + np.array([0.0, 0.5 * g * s * s])
            q0 = max(q - 0.04, 0.0)
            a = amp * fade
            v.polyline([pos(q0), pos((q0 + q) / 2), pos(q)], wid * 0.4, a * 0.1, r_end=wid, a_end=a)


@atom("cross_flash", tags=["glint", "light", "flash"])
class CrossFlash(Atom):
    """Four-point star glint that shoots its rays out fast and decays (blade-crossing flash,
    ping on a polished surface). Attach it to a slash's ``mid_arc``."""
    kind = "oneshot"
    outputs = {"light": Light(palette="steel")}

    class Params(Atom.Params):
        duration: float = P(0.25, gt=0, doc="seconds until gone")
        length: float = P(16.0, ge=0, doc="ray length at the start (px)")
        grow: float = P(140.0, ge=0, doc="ray length added per second (px/s)")
        brightness: float = P(2.2, ge=0, doc="peak intensity")
        decay: float = P(0.05, gt=0, doc="brightness time constant (s)")
        angle: float = P(34.0, doc="rotation of the cross (deg)")
        width: float = P(1.3, gt=0, doc="ray thickness (px)")

    def get_bounds(self):
        r = self.p.length + self.p.grow * self.p.duration + 3 * self.p.width * 1.6 + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        a = p.brightness * math.exp(-t / p.decay) * _end_fade(t, p.duration)
        if a < 0.02:
            return
        glint(ctx.light, 0.0, 0.0, p.length + p.grow * t, a, ang=math.radians(p.angle), r=p.width)


# =============================================================================
# hit impact
# =============================================================================
@atom("star_burst", tags=["impact", "light", "combat"])
class StarBurst(Atom):
    """Comic hit star: a spiky star polygon (soft outer star, hot inner star, white core)
    pops in then collapses."""
    kind = "oneshot"
    outputs = {"light": Light(palette="impact", bloom=[[1, 0.35], [4, 0.3], [12, 0.22]], exposure=1.15)}
    streams = ("shape",)

    class Params(Atom.Params):
        duration: float = P(0.333, gt=0, doc="seconds the hit lasts")
        points: int = P(9, ge=3, doc="points on the star")
        radius: float = P(62.0, gt=0, doc="star radius (px)")
        spikiness: float = P(1.0, ge=0, le=1.5, doc="notch depth (0 = disc, 1 = default, 1.5 = needles)")
        brightness: float = P(1.0, ge=0, doc="intensity multiplier")
        soft: float = P(0.7, gt=0, doc="edge softness (output px)")

    def setup(self):
        p = self.p
        rng = self.rng("shape")
        n = p.points
        self.a0 = rng.uniform(0, TAU)
        self.outer = rng.uniform(0.6, 1.0, n)
        self.outer[rng.integers(0, n)] = 1.0
        self.jit = rng.uniform(-0.25, 0.25, n)
        self.inner_k = max(0.08, 1 - 0.62 * p.spikiness)

    def get_bounds(self):
        r = self.p.radius + 4
        return (-r, -r, r, r)

    def _star(self, R, k_in):
        n = self.p.points
        pts = []
        for i in range(n):
            a = self.a0 + (i + self.jit[i] * 0.5) / n * TAU
            pts.append((math.cos(a) * R * self.outer[i], math.sin(a) * R * self.outer[i]))
            b = self.a0 + (i + 0.5) / n * TAU
            pts.append((math.cos(b) * R * k_in, math.sin(b) * R * k_in))
        return pts

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        pop = (0.7 + 0.3 * min(u / 0.12, 1)) * (1 - float(smoothstep(0.25, 0.7, u)) * 0.75)
        star_a = (1 - float(smoothstep(0.3, 0.72, u))) * p.brightness
        if star_a <= 0:
            return
        R0 = p.radius
        ctx.light.polygon(self._star(R0 * pop, self.inner_k), 0.5 * star_a, soft=p.soft)
        ctx.light.polygon(self._star(R0 * pop * 0.6, min(self.inner_k * 1.1, 0.9)), 1.6 * star_a, soft=p.soft)
        ctx.light.dot(0, 0, R0 * 0.18 * pop, 2.0 * star_a)


@atom("speed_lines", tags=["impact", "light", "lines"])
class SpeedLines(Atom):
    """Radial speed lines flying outward from a hit: tapered strokes that start on a ring
    and travel out while they fade."""
    kind = "oneshot"
    outputs = {"light": Light(palette="impact")}
    streams = ("lines",)

    class Params(Atom.Params):
        duration: float = P(0.333, gt=0, doc="seconds")
        count: int = P(16, ge=0, doc="number of lines")
        radius: float = P(62.0, gt=0, doc="reference radius (px); starts are a fraction of it")
        start_min: float = P(0.55, ge=0, doc="minimum start radius (fraction of radius)")
        start_max: float = P(0.95, ge=0, doc="maximum start radius (fraction of radius)")
        travel: float = P(0.9, ge=0, doc="outward travel over the duration (fraction of radius)")
        length_min: float = P(22.0, ge=0, doc="minimum line length (px)")
        length_max: float = P(44.0, ge=0, doc="maximum line length (px)")
        shrink: float = P(0.4, ge=0, le=1, doc="fraction of length lost over the duration")
        width_min: float = P(0.8, gt=0, doc="minimum line width at the inner end (px)")
        width_max: float = P(1.6, gt=0, doc="maximum line width at the inner end (px)")
        brightness: float = P(1.6, ge=0, doc="intensity")
        fade_start: float = P(0.15, ge=0, le=1, doc="fade starts (fraction of duration)")
        fade_end: float = P(0.6, ge=0, le=1, doc="fully faded (fraction of duration)")

    def setup(self):
        p = self.p
        rng = self.rng("lines")
        n = p.count
        self.ang = rng.uniform(0, TAU, n)
        self.r = rng.uniform(p.start_min, max(p.start_min, p.start_max), n)
        self.len = rng.uniform(p.length_min, max(p.length_min, p.length_max), n)
        self.w = rng.uniform(p.width_min, max(p.width_min, p.width_max), n)

    def get_bounds(self):
        p = self.p
        r = p.radius * (max(p.start_min, p.start_max) + p.travel) + max(p.length_min, p.length_max) \
            + 3 * max(p.width_min, p.width_max) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        lf = 1 - float(smoothstep(p.fade_start, max(p.fade_end, p.fade_start + 1e-3), u))
        if lf <= 0 or u >= 1:
            return
        for i in range(p.count):
            d = np.array([math.cos(self.ang[i]), math.sin(self.ang[i])])
            p0 = d * p.radius * (self.r[i] + p.travel * u)
            p1 = p0 + d * self.len[i] * (1 - p.shrink * u)
            ctx.light.polyline([p0, p1], self.w[i], p.brightness * lf, r_end=0.2, a_end=0.0)
