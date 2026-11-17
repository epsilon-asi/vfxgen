"""UI / game-feedback atoms: procedural stroke text, button shine sweep, spinning coin and
radial light rays.

Text and coins are opaque matter painted into a ``solid`` role (colour = channel palette
position); glows, shines and rays are light. Lengths are pixels at scale 1.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Solid, atom
from ..engine.canvas import capsule_sd
from ..engine.color import _hex, color_at, get_palette, ramp, register_palette
from ..engine.mathx import TAU, smoothstep
from ..errors import SpecError
from .combat import glint
from .glyphs import SUPPORTED, glyph, layout

register_palette("text_gold", "0:#7a3200,0.35:#e98a12,0.62:#ffc93c,0.85:#fff2a8,1:#ffffff")
register_palette("gold_coin", "0:#4a2400,0.25:#9a5a08,0.5:#e0a020,0.75:#ffd85a,1:#fff6c8")
register_palette("text_white", "0:#3c3c46,0.4:#a8acb8,0.7:#e6e8ee,0.88:#ffffff,1:#ffffff")
register_palette("text_crit", "0:#5a0000,0.35:#d01a0a,0.62:#ff6a1a,0.85:#ffd36a,1:#fffbe6")


def _ss(e0, e1, x) -> float:
    return float(smoothstep(e0, e1, x))


def spec_color(spec, k: float = 0.4) -> tuple[float, float, float]:
    """A single colour from a colour spec: a lone hex is used exactly, ramps are sampled at k."""
    s = str(spec).strip() if isinstance(spec, str) else None
    if s and s.startswith("#") and "," not in s:
        return _hex(s)
    return color_at(spec, k)


def paint_cov(view, ys, xs, cov, rgb, a: float = 1.0) -> None:
    """Paint coverage ``cov`` with per-pixel (or single) colour ``rgb`` (0..255) into a solid view.

    SolidView's public primitives take one colour per call; gradients need per-pixel colour, so
    this goes through the view's painter (honours gain, layer masks and painter's order)."""
    rgb = np.asarray(rgb, np.float32) / 255.0
    view._paint(ys, xs, cov.astype(np.float32), rgb, a)


def _pop_scale(q: float, overshoot: float) -> float:
    if q >= 1:
        return 1.0
    q = max(q, 0.0)
    return (1 - (1 - q) ** 3) * (1 + overshoot * math.sin(math.pi * q))


# =============================================================================
# text
# =============================================================================
@atom("text", tags=["ui", "text", "solid", "numbers"])
class Text(Atom):
    """Procedural stroke-font text (no system fonts): digits 0-9, A-Z, ``+ - ! % . :`` and ``x``
    (multiply sign). Thick round strokes with a vertical gradient fill (channel palette, top =
    ``fill_top``), a dark outline, a drop shadow and an optional glow (light role). Characters
    pop in one after another (``stagger``) with an overshoot, optionally bob, hold, then fade over
    the last ``fade`` seconds. Centred on the local origin (cap middle)."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="text_gold"),
               "light": Light(palette="holy", bloom=[[2, 0.4], [6, 0.35], [16, 0.25]], alpha_gamma=0.8)}
    streams = ("tilt",)

    class Params(Atom.Params):
        duration: float = P(1.2, gt=0, doc="seconds until fully faded")
        text: str = P("+1", doc=f"the string (supported: {SUPPORTED.strip()} and space; letters are upper-cased, "
                                "lower-case x = multiply sign)")
        size: float = P(32.0, gt=0, doc="cap height (px)")
        weight: float = P(0.2, gt=0, le=0.6, doc="stroke thickness (fraction of size)")
        spacing: float = P(0.04, ge=-0.3, doc="extra letter spacing (fraction of size)")
        outline: float = P(0.085, ge=0, doc="outline width (fraction of size); 0 = none")
        outline_palette: str = P("#2b1205", doc="outline / shadow colour (colour spec; a single hex is used as is)")
        shadow: float = P(0.07, ge=0, doc="drop shadow offset down (fraction of size); 0 = none")
        shadow_alpha: float = P(0.5, ge=0, le=1, doc="drop shadow opacity")
        fill_top: float = P(0.93, ge=0, le=1, doc="fill palette position at the top of the letters")
        gradient: float = P(0.4, ge=0, le=1, doc="fill palette drop from top to baseline")
        gloss: float = P(0.1, ge=0, le=0.5, doc="extra brightness of the upper half (two-tone gloss)")
        glow: float = P(0.0, ge=0, doc="glow brightness behind the letters (light role); 0 = none")
        glow_radius: float = P(0.22, gt=0, doc="glow radius (fraction of size)")
        delay: float = P(0.0, ge=0, doc="seconds before the first character pops in")
        stagger: float = P(0.04, ge=0, doc="seconds between characters popping in")
        pop: float = P(0.16, gt=0, doc="pop-in time per character (s)")
        overshoot: float = P(0.35, ge=0, le=1.5, doc="pop-in scale overshoot")
        rise: float = P(0.0, ge=0, doc="characters rise into place from this far below (fraction of size)")
        fade: float = P(0.3, gt=0, doc="fade-out time at the end (s)")
        fade_scale: float = P(1.0, ge=0, le=3, doc="scale reached at the end of the fade (1 = none)")
        bob: float = P(0.0, ge=0, doc="vertical bob amplitude (px)")
        bob_rate: float = P(2.0, ge=0, doc="bobs per second")
        bob_wave: float = P(0.7, doc="bob phase step between characters (radians): a travelling wave")
        tilt: float = P(0.0, ge=0, le=45, doc="random per-character tilt (deg)")

    def setup(self):
        p = self.p
        bad = sorted({c for c in p.text if c not in SUPPORTED and c.upper() not in SUPPORTED})
        if bad:
            raise SpecError(f"text: unsupported character(s) {''.join(bad)!r}; supported: {SUPPORTED!r}")
        self.r = p.weight * p.size / 2
        self.items, self.total = layout(p.text, p.size, self.r, p.spacing * p.size)
        rng = self.rng("tilt")
        self.tilts = np.radians(rng.uniform(-1, 1, max(len(self.items), 1)) * p.tilt)
        self.ow = p.outline * p.size
        self.sh = p.shadow * p.size
        self.outline_rgb = spec_color(p.outline_palette, 0.15)

    def get_bounds(self):
        p = self.p
        k = 1 + p.overshoot
        k = max(k, p.fade_scale)
        m = self.r + self.ow + 3 * p.glow_radius * p.size * (p.glow > 0) + 2
        hw = self.total / 2 * k + m + self.sh
        hh = p.size / 2 * k + m + p.rise * p.size + p.bob + self.sh
        return (-hw - 2, -hh - 2, hw + 2, hh + 2)

    # -- per-character state --------------------------------------------------------
    def _chars(self, t):
        p = self.p
        out = []
        fade = _ss(p.duration, p.duration - p.fade, t) if t < p.duration else 0.0
        if fade <= 0:
            return out
        fs = 1 + (p.fade_scale - 1) * (1 - fade)
        for i, (ch, pts, (cx, cy), adv) in enumerate(self.items):
            ti = t - p.delay - i * p.stagger
            if ti < 0:
                continue
            q = min(ti / p.pop, 1.0)
            s = _pop_scale(q, p.overshoot) * fs
            a = _ss(0, 0.3, q) * fade
            if s <= 1e-3 or a <= 1e-3:
                continue
            dy = p.rise * p.size * (1 - q) ** 3
            if p.bob > 0:
                dy += p.bob * math.sin(TAU * p.bob_rate * t - i * p.bob_wave)
            th = float(self.tilts[i]) * (1 - 0.5 * q)
            cs, sn = math.cos(th) * s, math.sin(th) * s
            M = np.array([[cs, sn], [-sn, cs]], np.float32)
            c = np.array([cx, cy], np.float32)
            tp = [(q_ - c) @ M + c + np.array([0, dy], np.float32) for q_ in pts]
            out.append((tp, s, a, cy + dy))
        return out

    def _sdf(self, view, tp, r, grow, dx=0.0, dy=0.0):
        allp = np.concatenate(tp)
        m = r + grow + 3.0 / max(view.px, 1e-6)
        w = view.window(float(allp[:, 0].min()) + dx - m, float(allp[:, 1].min()) + dy - m,
                        float(allp[:, 0].max()) + dx + m, float(allp[:, 1].max()) + dy + m)
        if w is None:
            return None
        ys, xs, X, Y = w
        X = X - dx
        Y = Y - dy
        d = np.full(X.shape, np.inf, np.float32)
        for s in tp:
            if len(s) == 1:
                s = np.vstack([s, s])
            for j in range(len(s) - 1):
                np.minimum(d, capsule_sd(X, Y, s[j], s[j + 1], r, r), out=d)
        return ys, xs, X, Y, d

    def draw(self, ctx, t):
        p = self.p
        chars = self._chars(t)
        if not chars:
            return
        sol = ctx.solid
        soft = 0.6 / max(sol.px, 1e-6)
        if ctx.has("light") and p.glow > 0:
            for tp, s, a, _ in chars:
                for st in tp:
                    if len(st) == 1:
                        ctx.light.dot(float(st[0, 0]), float(st[0, 1]), p.glow_radius * p.size * s, p.glow * a)
                    else:
                        ctx.light.polyline(st, p.glow_radius * p.size * s, p.glow * a)
        orgb = self.outline_rgb
        if self.sh > 0 and p.shadow_alpha > 0:
            for tp, s, a, _ in chars:
                res = self._sdf(sol, tp, self.r * s, self.ow * s, 0.0, self.sh * s)
                if res is not None:
                    ys, xs, X, Y, d = res
                    paint_cov(sol, ys, xs, smoothstep(soft, -soft, d - self.ow * s), orgb, a * p.shadow_alpha)
        if self.ow > 0:
            for tp, s, a, _ in chars:
                res = self._sdf(sol, tp, self.r * s, self.ow * s)
                if res is not None:
                    ys, xs, X, Y, d = res
                    paint_cov(sol, ys, xs, smoothstep(soft, -soft, d - self.ow * s), orgb, a)
        stops = sol.palette or get_palette("text_gold")
        for tp, s, a, cy in chars:
            res = self._sdf(sol, tp, self.r * s, 0.0)
            if res is None:
                continue
            ys, xs, X, Y, d = res
            h = p.size * s
            v = np.clip((Y - (cy - h / 2 - self.r * s)) / (h + 2 * self.r * s), 0, 1)
            k = p.fill_top - p.gradient * v + p.gloss * smoothstep(0.52, 0.46, v)
            paint_cov(sol, ys, xs, smoothstep(soft, -soft, d), ramp(stops, np.clip(k, 0, 1)), a)


# =============================================================================
# shine sweep
# =============================================================================
def _shape_sd(X, Y, shape, w, h, r):
    if shape == "ellipse":
        a, b = w / 2, h / 2
        return (np.sqrt((X / a) ** 2 + (Y / b) ** 2) - 1) * min(a, b)
    r = min(r, w / 2, h / 2)
    qx = np.abs(X) - (w / 2 - r)
    qy = np.abs(Y) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


@atom("shine_sweep", tags=["ui", "light", "loop", "button", "shine"])
class ShineSweep(Atom):
    """Diagonal highlight band (plus a thinner trailing band) sweeping across a rounded
    rectangle (button) or ellipse once per period, lighting the shape's rim as it passes and
    leaving a corner sparkle. Between sweeps only a faint ``idle`` rim stays lit. Centred on the
    origin. Overlay it additively on a button sprite."""
    kind = "loop"
    outputs = {"light": Light(palette="glint", bloom=[[1.5, 0.35], [5, 0.25]], alpha_gamma=0.9)}

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="interval between sweeps = loop length (s)")
        shape: str = P("rect", doc="area the shine is clipped to", choices=["rect", "ellipse"])
        width: float = P(180.0, gt=0, doc="shape width (px)")
        height: float = P(60.0, gt=0, doc="shape height (px)")
        corner_radius: float = P(18.0, ge=0, doc="rect corner radius (px)")
        angle: float = P(25.0, ge=-80, le=80, doc="band tilt from vertical (deg)")
        band_width: float = P(14.0, gt=0, doc="main band half-width (px, gaussian sigma)")
        band2: float = P(0.55, ge=0, doc="trailing thin band brightness (0 = none)")
        sweep_time: float = P(0.55, gt=0, doc="seconds the band takes to cross")
        delay: float = P(0.0, ge=0, doc="sweep start within the period (s)")
        brightness: float = P(0.9, ge=0, doc="band intensity")
        rim: float = P(0.9, ge=0, doc="rim highlight brightness near the band")
        idle: float = P(0.0, ge=0, doc="steady faint rim glow (also between sweeps)")
        edge_soft: float = P(1.0, gt=0, doc="edge softness of the shape (px)")
        sparkle: float = P(1.0, ge=0, doc="brightness of the glint left at the far corner (0 = none)")

    def get_bounds(self):
        p = self.p
        m = 8 + 3 * p.edge_soft
        sp = 30 if p.sparkle > 0 else 0
        hw = p.width / 2 + max(m, sp * 0.6)
        hh = p.height / 2 + max(m, sp * 0.6)
        return (-hw, -hh, hw, hh)

    def _phase(self, t):
        p = self.p
        return ((t - p.delay) % p.period) / max(p.sweep_time, 1e-6)

    def draw(self, ctx, t):
        p = self.p
        th = math.radians(p.angle)
        nx, ny = math.cos(th), math.sin(th)
        ext = abs(nx) * p.width / 2 + abs(ny) * p.height / 2
        u = self._phase(t)
        L = ctx.light
        b = self.get_bounds()
        span = ext + 3 * p.band_width
        if u < 1:
            e = u * u * (3 - 2 * u)
            c = -span + 2 * span * e
            bw = p.band_width

            def f(X, Y):
                sd = _shape_sd(X, Y, p.shape, p.width, p.height, p.corner_radius)
                inside = smoothstep(p.edge_soft, -p.edge_soft, sd)
                uu = X * nx + Y * ny
                band = np.exp(-((uu - c) / bw) ** 2 / 2)
                if p.band2 > 0:
                    band = band + p.band2 * np.exp(-((uu - c + 2.2 * bw) / (0.35 * bw)) ** 2 / 2)
                v = p.brightness * band * inside
                if p.rim > 0:
                    rimm = np.exp(-(sd / 1.3) ** 2 / 2)
                    v = v + p.rim * rimm * np.exp(-((uu - c) / (3 * bw)) ** 2 / 2)
                if p.idle > 0:
                    v = v + p.idle * np.exp(-(sd / 1.5) ** 2 / 2)
                return v
            L.field(f, b)
        elif p.idle > 0:
            L.field(lambda X, Y: p.idle * np.exp(-(_shape_sd(X, Y, p.shape, p.width, p.height,
                                                               p.corner_radius) / 1.5) ** 2 / 2), b)
        if p.sparkle > 0:
            # glint pops at the corner the band leaves from, as it exits
            q = (u - 0.75) / 0.7
            if 0 <= q < 1:
                a = p.sparkle * 2.2 * math.sin(math.pi * q) ** 1.5
                if p.shape == "rect":
                    rr = min(p.corner_radius, p.width / 2, p.height / 2)
                    gx = p.width / 2 - rr * (1 - 0.7071)
                    gy = -p.height / 2 + rr * (1 - 0.7071)
                else:
                    gx, gy = p.width / 2 * 0.7071, -p.height / 2 * 0.7071
                sgn = 1 if nx >= 0 else -1
                Lg = min(18.0, 0.5 * min(p.width, p.height) + 4) * (0.6 + 0.4 * math.sin(math.pi * q))
                glint(L, sgn * gx, gy, Lg, a, ang=0.3 * q, r=1.1)


# =============================================================================
# coin
# =============================================================================
def _ellipse(cx, cy, rx, ry, n=48, a0=0.0, a1=360.0):
    th = np.radians(np.linspace(a0, a1, n, endpoint=False))
    return np.column_stack([cx + rx * np.cos(th), cy + ry * np.sin(th)])


def _star(cx, cy, R, kx, n=5, inner=0.45, rot=-90.0):
    pts = []
    for i in range(2 * n):
        a = math.radians(rot + i * 180.0 / n)
        rr = R if i % 2 == 0 else R * inner
        pts.append((cx + kx * rr * math.cos(a), cy + rr * math.sin(a)))
    return np.asarray(pts, np.float32)


@atom("coin_spin", tags=["ui", "solid", "loop", "pickup", "coin"])
class CoinSpin(Atom):
    """A procedural coin spinning about its vertical axis (x-scale = cos of the spin angle):
    thick edge band, raised rim, inner face and an embossed star, all flat-shaded solid ellipses
    (channel palette = metal ramp, shadow -> lit) whose shade follows the spin, plus a specular
    glint (light role) when it faces the viewer. Centred on the origin."""
    kind = "loop"
    outputs = {"solid": Solid(palette="gold_coin"),
               "light": Light(palette="holy", bloom=[[1.2, 0.4], [4, 0.3]], alpha_gamma=0.85)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        spins: int = P(2, doc="full turns per period (whole number; negative = reverse)")
        phase: float = P(0.0, doc="spin angle offset (turns)")
        radius: float = P(18.0, gt=0, doc="coin radius (px)")
        thickness: float = P(0.24, ge=0, le=1, doc="edge thickness (fraction of radius)")
        emblem: str = P("star", doc="face emblem", choices=["star", "ring", "none"])
        shine: float = P(1.2, ge=0, doc="specular glint brightness when facing the viewer (0 = none)")

    def get_bounds(self):
        r = self.p.radius * (1 + self.p.thickness) + 2
        g = self.p.radius * 1.2 if self.p.shine > 0 else 0
        r = max(r, g + self.p.radius * 0.5) + 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        th = TAU * (p.spins * t / p.period + p.phase)
        c, s = math.cos(th), math.sin(th)
        R = p.radius
        rx = R * abs(c)
        off = 0.5 * p.thickness * R * abs(s)
        sgn = 1.0 if s * c >= 0 else -1.0          # side the face is shifted to
        sol = ctx.solid
        # edge band: hull of the two faces
        right = _ellipse(off, 0, rx, R, 24, -90, 90.0001)
        left = _ellipse(-off, 0, rx, R, 24, 90, 270.0001)
        hull = np.vstack([right, left])
        sol.polygon(hull, k=0.18, soft=0.6)
        if off > 0.6:      # reeded edge: a lit band down the middle of the edge
            ex = -sgn * off * 0.9
            band = np.vstack([_ellipse(ex - off * 0.35, 0, rx, R * 0.97, 16, 90, 270.0001),
                              _ellipse(ex + off * 0.35, 0, rx, R * 0.97, 16, -90, 90.0001)[::-1]])
            sol.polygon(np.vstack([band]), k=0.32, a=0.6, soft=0.6)
        if rx < 0.35:
            return
        fx = sgn * off
        lit = 0.08 * (-s if c >= 0 else s)        # face shading follows the spin (light from the left)
        sol.polygon(_ellipse(fx, 0, rx, R), k=0.6 + lit, soft=0.6)
        sol.polygon(_ellipse(fx, 0, rx * 0.8, R * 0.8), k=0.42 + lit, soft=0.6)
        sol.polygon(_ellipse(fx + 0.04 * rx, 0.04 * R, rx * 0.76, R * 0.76), k=0.52 + lit, soft=0.6)
        if p.emblem == "star":
            sol.polygon(_star(fx + 0.05 * rx, 0.06 * R, R * 0.5, abs(c)), k=0.3 + lit, soft=0.6)
            sol.polygon(_star(fx, 0, R * 0.5, abs(c)), k=0.86 + lit, soft=0.6)
        elif p.emblem == "ring":
            sol.stroke(_ellipse(fx, 0, rx * 0.45, R * 0.45, 32).tolist() + [_ellipse(fx, 0, rx * 0.45, R * 0.45, 32)[0]],
                       R * 0.08, k=0.86 + lit)
        # rim highlight on the upper left
        arc = _ellipse(fx, 0, rx * 0.9, R * 0.9, 10, 195, 255)
        sol.stroke(arc, R * 0.06, k=1.0, a=0.85, r_end=R * 0.03)
        if p.shine > 0 and ctx.has("light"):
            a = p.shine * _ss(0.8, 1.0, abs(c))
            if a > 1e-3:
                glint(ctx.light, fx - 0.35 * rx, -0.4 * R, R * 1.0 * (0.5 + 0.5 * a / p.shine), a * 1.6,
                      ang=0.35, r=max(0.7, R * 0.05))


# =============================================================================
# radial rays
# =============================================================================
@atom("ray_burst", tags=["ui", "light", "rays", "burst", "unlock"])
class RayBurst(Atom):
    """Radial light rays (sunburst / god rays): tapered wedges fanning out from the centre,
    growing in, slowly rotating, then fading. Alternate rays are shorter. Unlock, reward and
    level-up backdrops."""
    kind = "oneshot"
    outputs = {"light": Light(palette="holy", bloom=[[2, 0.4], [8, 0.35], [20, 0.25]], alpha_gamma=0.8)}
    streams = ("rays",)

    class Params(Atom.Params):
        duration: float = P(1.2, gt=0, doc="seconds until faded")
        count: int = P(12, ge=1, doc="number of rays")
        radius: float = P(110.0, gt=0, doc="length of the long rays (px)")
        short: float = P(0.65, ge=0, le=1, doc="length of every other ray (fraction)")
        width: float = P(10.0, gt=0, doc="ray angular width (deg)")
        inner: float = P(10.0, ge=0, doc="rays start this far from the centre (px)")
        grow: float = P(0.18, gt=0, doc="seconds for the rays to shoot out")
        spin: float = P(20.0, doc="rotation (deg per second)")
        brightness: float = P(0.9, ge=0, doc="ray intensity")
        jitter: float = P(0.25, ge=0, le=1, doc="random length variation")
        fade_start: float = P(0.45, ge=0, lt=1, doc="fade-out starts (fraction of duration)")

    def setup(self):
        rng = self.rng("rays")
        n = self.p.count
        self.len = 1 - self.p.jitter * rng.random(n)
        self.a0 = rng.uniform(0, TAU)

    def get_bounds(self):
        r = self.p.radius + 8
        return (-r, -r, r, r)

    def draw(self, ctx, t):
        p = self.p
        u = t / p.duration
        if u >= 1:
            return
        env = _ss(1.0, p.fade_start, u) * _ss(0, 0.05, t)
        g = 1 - math.exp(-t / (p.grow / 3))
        if env <= 1e-3:
            return
        rot = self.a0 + math.radians(p.spin * t)
        hw = math.radians(p.width) / 2
        n = p.count
        lens = p.radius * g * self.len * np.where(np.arange(n) % 2 == 0, 1.0, p.short)

        def f(X, Y):
            r = np.hypot(X, Y)
            ang = (np.arctan2(Y, X) - rot) % TAU
            k = ang / TAU * n
            idx = np.floor(k + 0.5).astype(int) % n
            dth = (k - np.round(k)) * TAU / n            # angle from the nearest ray axis
            Lr = lens[idx]
            # wedge narrows toward the tip
            along = np.clip((r - p.inner) / np.maximum(Lr - p.inner, 1e-3), 0, 1)
            wid = hw * (1 - 0.65 * along) + 1e-4
            v = np.exp(-(dth / wid) ** 2 * 1.5) * smoothstep(p.inner * 0.6, p.inner * 1.6 + 1, r)
            v = v * (1 - along) ** 1.2 * (r < Lr)
            return p.brightness * env * v
        ctx.light.field(f, self.get_bounds(), lowres="bilinear")
