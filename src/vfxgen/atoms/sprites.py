"""Sprite / texture atoms: effects applied to *your own* image assets.

* ``sprite`` (one-shot): paints an image with an effect: hit ``flash``, noise ``dissolve`` with a
  burning edge, ``teleport_in`` / ``teleport_out`` (slice wipe + stretch + beam), pulsing
  ``outline_glow``, creeping ``tint`` (freeze / poison).
* ``sprite_loop`` (loop): the image with steady effects (tint patches, outline glow, rim light,
  sheen sweep, flicker flashes, shiver).
* ``sprite_emitter`` (loop): particles, flames, arcs, frost crystals and mist emitted from the
  silhouette (interior / edge / top edge) of an image. Does not draw the image itself.
* ``texture_particles`` (loop) / ``texture_burst`` (one-shot): particles drawn as an image
  (coins, leaves, shards) with spin, flip, scale over life, gravity, drag, curl, floor bounce.

Convention for the sprite atoms: the local origin is the image's **bottom centre** (the
character's feet); the image covers ``x in [-w/2, w/2]``, ``y in [-h, 0]`` where ``h`` is the
``height`` param (image pixels when 0). Images are painted into a ``solid`` role, glows into
``light``. Colours of the image itself come from the image; ``*_palette`` params give flash /
tint / ice colours; light colours come from the light channel palette.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, Solid, atom
from ..engine.assets import bilinear, edge_mask, glow_mask, sample_points
from ..engine.color import color_at
from ..engine.mathx import TAU, blur, refine, smoothstep
from ..engine.noise import Noise
from ..engine.particles import Emitter, EmitterConfig, life_alpha

_PSTYLES = ["dot", "streak", "glint", "plus", "puff"]
_REGIONS = ["interior", "edge", "top_edge"]


def _ss(e0, e1, x) -> float:
    return float(smoothstep(e0, e1, x))


def _rgb(spec, k: float = 0.4) -> tuple[float, float, float]:
    """A colour from a colour spec (single hex colours sit at ramp position 0.4)."""
    return color_at(spec, k)


def _rank(field: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rank-normalise ``field`` over ``mask`` pixels to a uniform 0..1 distribution (so
    threshold sweeps remove area at a constant rate). Outside the mask: 1."""
    out = np.ones(field.shape, np.float32)
    idx = np.flatnonzero(mask.ravel())
    if idx.size:
        vals = field.ravel()[idx]
        order = np.argsort(vals, kind="stable")
        r = np.empty(idx.size, np.float32)
        r[order] = np.linspace(0, 1, idx.size, dtype=np.float32)
        out.ravel()[idx] = r
    return out


def _tinted(img: np.ndarray, rgb, amount: float) -> np.ndarray:
    """Premultiplied RGBA recoloured toward ``rgb`` (0..255), keeping the luminance detail."""
    a = img[..., 3:4]
    straight = np.where(a > 1e-6, img[..., :3] / np.maximum(a, 1e-6), 0)
    lum = straight @ np.array([0.299, 0.587, 0.114], np.float32)
    col = np.asarray(rgb, np.float32) / 255.0
    tint = np.clip(col[None, None, :] * (0.55 + 0.75 * lum[..., None]) + 0.3 * lum[..., None] ** 2, 0, 1)
    out = straight + (tint - straight) * float(amount)
    return np.dstack([out * a, a]).astype(np.float32)


class _SpriteImage:
    """Mixin: the image, its display size and silhouette maps (padded so halos fit)."""

    def _load_image(self, height: float, pad: float):
        img = self.asset(self.p.image)
        self.img = img
        self.ih, self.iw = img.shape[:2]
        self.sc = (height / self.ih) if height > 0 else 1.0     # local units per image pixel
        self.W, self.H = self.iw * self.sc, self.ih * self.sc
        self.pad = int(math.ceil(max(pad, 1.0) / self.sc)) + 2   # image px of padding
        P = self.pad
        a = np.zeros((self.ih + 2 * P, self.iw + 2 * P), np.float32)
        a[P:P + self.ih, P:P + self.iw] = img[..., 3]
        self.apad = a

    def _to_local(self, px, py, padded: bool = True):
        """Image pixel coords (padded or not) -> local coords."""
        P = self.pad if padded else 0
        x = (np.asarray(px, np.float32) - P - self.iw / 2) * self.sc
        y = (np.asarray(py, np.float32) - P - self.ih) * self.sc
        return x, y

    def _to_img(self, X, Y, padded: bool = True):
        P = self.pad if padded else 0
        return X / self.sc + self.iw / 2 + P, Y / self.sc + self.ih + P

    def _stamp(self, view, arr, a: float, x: float = 0.0, sx: float = 1.0, sy: float = 1.0) -> None:
        """Deposit a padded image-sized map, placed exactly like the sprite."""
        if a <= 1e-5:
            return
        ph, pw = arr.shape[:2]
        view.stamp(arr, x, 0.0, w=pw * self.sc * sx, h=ph * self.sc * sy, a=a,
                   anchor=(0.5, (self.pad + self.ih) / ph))

    def _paint(self, view, img, x: float = 0.0, sx: float = 1.0, sy: float = 1.0, **kw) -> None:
        view.image(img, x, 0.0, w=self.W * sx, h=self.H * sy, anchor=(0.5, 1.0), **kw)


def _sample_uv(fmap: np.ndarray, u, v) -> np.ndarray:
    h, w = fmap.shape
    return bilinear(fmap, u * w, v * h)


def _region_weight(apad: np.ndarray, region: str, width: float):
    """Weight map for emission + outward normals (nx, ny) of the blurred silhouette."""
    b = blur(apad, max(width, 0.75))
    gy, gx = np.gradient(b)
    n = np.hypot(gx, gy) + 1e-6
    nx, ny = -gx / n, -gy / n
    if region == "interior":
        w = apad.copy()
    else:
        w = edge_mask(apad, width)
        if region == "top_edge":
            w = w * np.clip(-ny * 1.4 - 0.15, 0, 1)
    return w.astype(np.float32), nx.astype(np.float32), ny.astype(np.float32)


def _sprite_anchors():
    return {"feet": (0.0, 0.0),
            "head": lambda self, t: (0.0, -self.H),
            "body": lambda self, t: (0.0, -self.H * 0.5)}


# =============================================================================
# sprite (one-shot)
# =============================================================================
@atom("sprite", tags=["sprite", "image", "solid", "light", "oneshot"])
class Sprite(_SpriteImage, Atom):
    """Your image with a one-shot effect (origin = image bottom centre, i.e. the feet).

    Modes:
    * ``flash``: hit flash to ``flash_palette`` (``pulses`` blinks), optional shake, halo glow.
    * ``dissolve``: noise-threshold dissolve (optionally sweeping ``direction``), charred band
      and a burning edge in light, embers (and smoke) flying off the edge.
    * ``teleport_out`` / ``teleport_in``: charge flash, horizontal squeeze + vertical stretch,
      the image splits into scanline slices that jitter and vanish bottom-up, then a vertical
      beam with rising sparks (``teleport_in`` plays it backwards: beam down, slices assemble).
    * ``outline_glow``: pulsing glow halo + inner rim light around the silhouette.
    * ``tint``: freeze / poison tint creeping in from the edges, holding, then thawing.

    Draws nothing at ``t >= duration`` (the game shows its own sprite again)."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="fire", bloom=[[2, 0.5], [6, 0.35]], alpha_gamma=0.8),
               "smoke": Smoke(palette="dark_smoke", density=1.0, shadow=0.4)}
    streams = ("noise", "embers", "slices")
    anchors = _sprite_anchors()

    class Params(Atom.Params):
        duration: float = P(0.5, gt=0, doc="seconds")
        image: str = P("hero.png", doc="image asset (PNG with alpha), relative to the scene file / VFXGEN_PATH / library assets")
        height: float = P(128.0, ge=0, doc="displayed image height (px); 0 = native image size")
        mode: str = P("flash", doc="effect", choices=["flash", "dissolve", "teleport_in", "teleport_out",
                                                       "outline_glow", "tint"])
        strength: float = P(1.0, ge=0, le=1, doc="flash / tint amount (0..1)")
        flash_palette: str = P("#ffffff", doc="flash colour (also the teleport charge colour)")
        tint_palette: str = P("frost", doc="tint colour (tint mode)")
        pulses: int = P(1, ge=1, le=12, doc="flash: number of blinks; outline_glow: glow pulses")
        rise: float = P(0.06, ge=0, le=1, doc="fraction of the duration to reach full flash / tint / glow")
        hold: float = P(0.15, ge=0, le=1, doc="flash: fraction of the duration held at full flash")
        release: float = P(0.4, ge=0, le=1, doc="tint / outline_glow: fraction of the duration fading out")
        glow: float = P(1.0, ge=0, doc="light brightness (halo, burning edge, beam)")
        glow_radius: float = P(6.0, gt=0, doc="glow halo radius (px)")
        shake: float = P(0.0, ge=0, doc="flash: sideways hit shake amplitude (px), decaying")
        edge_width: float = P(0.05, gt=0, le=0.5, doc="dissolve: burning edge width (fraction of the dissolve sweep)")
        char: float = P(0.85, ge=0, le=1, doc="dissolve: darkening of the band behind the burning edge")
        char_palette: str = P("#2a1206", doc="dissolve: colour of the charred band")
        noise_scale: float = P(9.0, gt=0, doc="dissolve / tint noise cell size (px)")
        direction: str = P("down", doc="dissolve: sweep direction (head->feet = down)", choices=["none", "up", "down", "left", "right"])
        sweep: float = P(0.45, ge=0, le=1, doc="dissolve / tint: how strongly the sweep (dissolve) or the edges (tint) lead vs noise")
        dissolve_end: float = P(0.8, gt=0.1, le=1, doc="dissolve: fraction of the duration when the image is gone")
        embers: int = P(50, ge=0, le=2000, doc="dissolve / teleport: number of embers or sparks")
        ember_size: float = P(1.2, gt=0, doc="ember / spark size (px)")
        smoke: float = P(0.0, ge=0, doc="dissolve: smoke wisps density from the burning edge (0 = off)")
        stretch: float = P(1.0, ge=0, le=4, doc="teleport: extra vertical stretch at the end (1 = twice as tall)")
        squeeze: float = P(0.9, ge=0, le=1, doc="teleport: horizontal squeeze at the end (fraction of the width lost)")
        lines: int = P(18, ge=1, le=128, doc="teleport: number of scanline slices")
        jitter: float = P(8.0, ge=0, doc="teleport: sideways slice jitter (px)")
        beam: float = P(1.0, ge=0, doc="teleport: beam brightness (0 = off)")

    def setup(self):
        p = self.p
        self._load_image(p.height, 3 * p.glow_radius * 2.5 + 4)
        self.flash_rgb = _rgb(p.flash_palette)
        a = self.img[..., 3]
        self.halo = glow_mask(self.apad, p.glow_radius / self.sc)
        self.halo_wide = glow_mask(self.apad, 2.5 * p.glow_radius / self.sc)
        self.rim = edge_mask(self.apad, 1.2 / self.sc) * self.apad
        rng_n = self.rng("noise")
        noise = Noise(rng_n, period=(32, 32, 8))
        yy, xx = np.mgrid[0:self.ih, 0:self.iw].astype(np.float32) + 0.5
        cell = max(p.noise_scale / self.sc, 1.0)
        n = noise.fbm(xx / cell, yy / cell, 0.37, octaves=3)
        inside = a > 0.02
        nrank = _rank(n, inside)
        # dissolve field: noise blended with a directional sweep
        sweep = {"none": None, "down": yy / self.ih, "up": 1 - yy / self.ih,
                 "right": xx / self.iw, "left": 1 - xx / self.iw}[p.direction]
        f = nrank if sweep is None else (1 - p.sweep) * nrank + p.sweep * sweep
        self.fmap = _rank(f, inside)
        # tint coverage: edges first (low depth), noise breaks it up
        depth = blur(a, 4.0 / self.sc + 1)
        depth = depth / max(float(depth.max()), 1e-6)
        self.cmap = _rank(p.sweep * depth + (1 - p.sweep) * nrank, inside)
        self.tinted = _tinted(self.img, _rgb(p.tint_palette), p.strength) if p.mode == "tint" else None
        self.char_rgb = _rgb(p.char_palette)
        # embers: points on the silhouette, released when the dissolve front passes them
        rng = self.rng("embers")
        m = int(p.embers)
        pts = sample_points(a, m, rng)
        self.e_x, self.e_y = self._to_local(pts[:, 0], pts[:, 1], padded=False)
        self.e_f = bilinear(self.fmap, pts[:, 0], pts[:, 1]) if m else np.zeros(0, np.float32)
        self.e_v = np.stack([rng.normal(0, 14, m), -rng.uniform(15, 55, m)], 1) if m else np.zeros((0, 2))
        self.e_life = rng.uniform(0.35, 0.8, m)
        self.e_r = rng.random((m, 3))
        # teleport slices
        rs = self.rng("slices")
        L = int(p.lines)
        self.sl_rows = np.round(np.linspace(0, self.ih, L + 1)).astype(int)
        order = 1 - (np.arange(L) + 0.5) / L                      # bottom slices vanish first
        self.sl_t = np.clip(0.2 + 0.7 * order + rs.uniform(-0.08, 0.08, L), 0.12, 0.98)
        self.sl_j = rs.normal(0, 1, L) * np.where(np.arange(L) % 2, 1, -1)
        self.sp = rs.random((max(m, 1), 4))

    def get_bounds(self):
        p = self.p
        m = 3 * p.glow_radius * 2.5 + 6
        top = self.H * (1 + (p.stretch if p.mode.startswith("teleport") else 0)) + m + 40
        side = self.W / 2 + m + (p.jitter * 3 + p.shake if p.mode != "dissolve" else 20)
        return (-side, -top - 50, side, m)

    # -- helpers ----------------------------------------------------------------------
    def _halo(self, view, amt, x=0.0, sx=1.0, sy=1.0):
        g = self.p.glow
        self._stamp(view, self.halo, 0.9 * g * amt, x, sx, sy)
        self._stamp(view, self.halo_wide, 0.25 * g * amt, x, sx, sy)

    def draw(self, ctx, t):
        p = self.p
        if t < 0 or t >= p.duration:
            return
        u = t / p.duration
        getattr(self, "_draw_" + p.mode.replace("teleport_in", "teleport").replace("teleport_out", "teleport"))(ctx, t, u)

    # -- flash ------------------------------------------------------------------------------
    def _draw_flash(self, ctx, t, u):
        p = self.p
        if p.pulses <= 1:
            amt = _ss(0, max(p.rise, 1e-3), u) * (1 - _ss(p.rise + p.hold, 1.0, u)) ** 1.6
        else:
            x = (u * p.pulses) % 1.0
            amt = _ss(0, 0.06, x) * _ss(0.6, 0.42, x) * (1 - u) ** 0.6
        amt *= p.strength
        dx = p.shake * (1 - u) ** 2 * math.sin(TAU * 5 * u) if p.shake else 0.0
        self._paint(ctx.solid, self.img, dx, tint=self.flash_rgb, tint_amount=amt)
        self._halo(ctx.light, amt, dx)

    # -- outline glow --------------------------------------------------------------------------
    def _draw_outline_glow(self, ctx, t, u):
        p = self.p
        env = _ss(0, max(p.rise, 1e-3), u) * _ss(1.0, 1.0 - max(p.release, 1e-3), u)
        mod = 0.72 + 0.28 * math.cos(TAU * p.pulses * u) if p.pulses > 1 else 1.0
        self._paint(ctx.solid, self.img)
        self._halo(ctx.light, env * mod)
        self._stamp(ctx.light, self.rim, 0.9 * p.glow * env * mod)

    # -- tint ---------------------------------------------------------------------------------
    def _draw_tint(self, ctx, t, u):
        p = self.p
        env = _ss(0, max(p.rise, 1e-3), u) * _ss(1.0, 1.0 - max(p.release, 1e-3), u)
        self._paint(ctx.solid, self.img)
        if env > 1e-3:
            lvl = env * 1.12
            cm = self.cmap
            self._paint(ctx.solid, self.tinted,
                        mask=lambda uu, vv: smoothstep(lvl, lvl - 0.1, _sample_uv(cm, uu, vv)))
            self._stamp(ctx.light, self.halo, 0.7 * p.glow * env)
            self._stamp(ctx.light, self.rim, 0.35 * p.glow * env)

    # -- dissolve -------------------------------------------------------------------------------
    def _threshold(self, t):
        p = self.p
        e = p.edge_width
        q = min(max(t / (p.dissolve_end * p.duration), 0.0), 1.0)
        return -e + (1 + 2 * e) * q ** 1.15

    def _draw_dissolve(self, ctx, t, u):
        p = self.p
        e = p.edge_width
        th = self._threshold(t)
        fm = self.fmap
        soft = 0.012
        self._paint(ctx.solid, self.img, mask=lambda uu, vv: smoothstep(th, th + soft, _sample_uv(fm, uu, vv)))
        if p.char > 0:
            cr = self.char_rgb
            self._paint(ctx.solid, self.img, tint=cr, tint_amount=1.0,
                        mask=lambda uu, vv: (lambda f: smoothstep(th, th + soft, f) * smoothstep(th + 2.6 * e, th + 0.6 * e, f)
                                             * p.char)(_sample_uv(fm, uu, vv)))
        # burning edge (light): a band just on the remaining side of the front
        a = self.img[..., 3]
        d = fm - th
        band = a * np.exp(-(np.maximum(d, 0) / (0.55 * e)) ** 2) * smoothstep(-0.25 * e, 0.0, d)
        band = band * (fm < 1.0)
        bpad = np.zeros_like(self.apad)
        P = self.pad
        bpad[P:P + self.ih, P:P + self.iw] = band
        end = _ss(p.duration, p.duration * 0.9, t)
        self._stamp(ctx.light, bpad, 3.0 * p.glow * end)
        # embers released as the front passes
        if len(self.e_f):
            tq = ((self.e_f + e) / (1 + 2 * e)) ** (1 / 1.15) * p.dissolve_end * p.duration
            age = t - tq
            live = (age >= 0) & (age < self.e_life)
            for i in np.flatnonzero(live):
                ag = float(age[i])
                q = ag / float(self.e_life[i])
                x = float(self.e_x[i] + self.e_v[i, 0] * ag + 6 * math.sin(TAU * (1.3 * ag + self.e_r[i, 0])))
                y = float(self.e_y[i] + self.e_v[i, 1] * ag - 30 * ag * ag)
                br = 1.8 * p.glow * (1 - q) ** 1.3 * _ss(0, 0.08, q) * (0.6 + 0.8 * self.e_r[i, 1]) * end
                ctx.light.dot(x, y, p.ember_size * (0.7 + 0.6 * self.e_r[i, 2]), br)
                if p.smoke > 0 and i % 3 == 0:
                    ctx.smoke.dot(x * 0.8 + 0.2 * float(self.e_x[i]), y - 10 * ag, 3 + 10 * q,
                                  0.12 * p.smoke * math.sin(math.pi * q) * end)

    # -- teleport ------------------------------------------------------------------------------
    def _draw_teleport(self, ctx, t, u):
        p = self.p
        if p.mode == "teleport_in":
            u = 1.0 - u
        A, B = 0.28, 0.7
        charge = _ss(0, A, u)
        k = min(max((u - A) / (B - A), 0.0), 1.0)
        k2 = k * k
        sx = 1 - p.squeeze * k2
        sy = 1 + p.stretch * k2
        fl = min(1.0, 0.85 * charge + k) * p.strength
        L = len(self.sl_t)
        g = p.glow
        visible_any = False
        vis_sum = 0.0
        for i in range(L):
            vis = 1 - _ss(self.sl_t[i] - 0.1, self.sl_t[i], k)
            vis_sum += vis / L
            if vis <= 1e-3:
                continue
            visible_any = True
            r0, r1 = int(self.sl_rows[i]), int(self.sl_rows[i + 1])
            if r1 <= r0:
                continue
            c0, c1 = max(r0 - 1, 0), min(r1 + 1, self.ih)
            crop = self.img[c0:c1]
            ch = (c1 - c0) / self.ih * self.H * sy
            y_bot = -(self.ih - c1) / self.ih * self.H * sy
            x = float(self.sl_j[i]) * p.jitter * math.sin(math.pi * min(1.0, k * 1.6)) ** 0.7
            n = c1 - c0
            lo, hi = (r0 - c0) / n, (r1 - c0) / n
            duty = 1 - 0.65 * k
            hi_k = lo + (hi - lo) * duty
            ctx.solid.image(crop, x, y_bot, w=self.W * sx, h=ch, anchor=(0.5, 1.0), a=vis,
                            tint=self.flash_rgb, tint_amount=fl,
                            mask=lambda uu, vv, lo=lo, hi=hi_k: smoothstep(lo - 0.002, lo + 0.002, vv)
                            * smoothstep(hi + 0.002, hi - 0.002, vv))
        if visible_any:
            vis_all = vis_sum ** 1.5
            self._halo(ctx.light, (0.5 * charge + 0.9 * k) * vis_all, 0.0, sx, sy)
            self._stamp(ctx.light, self.rim, 0.8 * g * charge * vis_all, 0.0, sx, sy)
            # bright scan line riding the wipe front (bottom-up)
            if 0 < k < 1:
                yf = -self.H * sy * min(1.0, max(0.0, (k - 0.15) / 0.8))
                wl = self.W * sx * 0.65 + 4
                ctx.light.polyline([(-wl, yf), (0, yf), (wl, yf)], 1.2, 0.0, a_end=0.0)
                ctx.light.polyline([(-wl, yf), (0, yf)], 1.1, 0.0, a_end=1.6 * g * math.sin(math.pi * k))
                ctx.light.polyline([(0, yf), (wl, yf)], 1.1, 1.6 * g * math.sin(math.pi * k), a_end=0.0)
        # beam + sparks
        if p.beam > 0:
            bc = _ss(B - 0.2, B, u) * (1 - _ss(B, 1.0, u)) ** 1.4
            if bc > 1e-3:
                top = -self.H * (1 + p.stretch) - 40
                bw = max(self.W * 0.16 * (1 - 0.6 * _ss(B, 1.0, u)), 1.5)
                a = p.beam * g * bc

                def beam(X, Y):
                    vert = smoothstep(top, top + 70, Y) * smoothstep(8, -4, Y)
                    return a * vert * (0.9 * np.exp(-X * X / (2 * bw * bw)) + 1.6 * np.exp(-X * X / (2 * (bw * 0.25) ** 2)))
                ctx.light.field(beam, (-4 * bw, top, 4 * bw, 12), lowres="bilinear")
                ctx.light.dot(0, 0, bw * 1.2, 0.8 * a)
        m = len(self.e_x)
        if m:
            sp = self.sp
            t0 = B - 0.12 + 0.2 * sp[:m, 0]
            life = 0.12 + 0.16 * sp[:m, 1]
            age = (u - t0) / life
            for i in np.flatnonzero((age >= 0) & (age < 1)):
                q = float(age[i])
                x0 = (sp[i, 2] - 0.5) * self.W * 0.5
                y0 = -sp[i, 3] * self.H
                y = y0 - (90 + 160 * sp[i, 1]) * q * p.duration
                ctx.light.dot(float(x0 * (1 - 0.6 * q)), float(y), p.ember_size, 1.6 * g * (1 - q) ** 1.2 * _ss(0, 0.1, q))


# =============================================================================
# sprite_loop
# =============================================================================
@atom("sprite_loop", tags=["sprite", "image", "solid", "light", "loop"])
class SpriteLoop(_SpriteImage, Atom):
    """Your image with steady, seamlessly looping effects (origin = image bottom centre).

    Combine freely: ``tint`` patches (``cover`` = fraction covered, edges first, noisy),
    pulsing ``outline`` halo, inner ``rim`` light, a ``sheen`` highlight sweeping across the
    tinted parts once per period, random ``flicker`` flashes to ``flash_palette`` (electrified)
    and ``shiver`` (sideways jitter). Burning / frozen / electrified / poisoned bodies,
    selection highlight, invulnerability blink."""
    kind = "loop"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="fire", bloom=[[2, 0.5], [6, 0.35]], alpha_gamma=0.8)}
    streams = ("noise", "flicker")
    anchors = _sprite_anchors()

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        image: str = P("hero.png", doc="image asset (PNG with alpha)")
        height: float = P(128.0, ge=0, doc="displayed image height (px); 0 = native image size")
        tint: float = P(0.0, ge=0, le=1, doc="tint amount (0 = off)")
        tint_palette: str = P("frost", doc="tint colour")
        cover: float = P(1.0, ge=0, le=1, doc="fraction of the image covered by the tint (edges first)")
        noise_scale: float = P(9.0, gt=0, doc="tint patch noise cell size (px)")
        sweep: float = P(0.5, ge=0, le=1, doc="how strongly the tint patches follow the edges vs noise")
        outline: float = P(0.0, ge=0, doc="glow halo brightness (0 = off)")
        rim: float = P(0.0, ge=0, doc="inner rim light brightness (0 = off)")
        glow_radius: float = P(6.0, gt=0, doc="halo radius (px)")
        sheen: float = P(0.0, ge=0, doc="highlight band sweeping over the tinted area once per period (0 = off)")
        pulse: float = P(0.25, ge=0, le=1, doc="pulse depth of the outline / rim")
        pulse_cycles: int = P(2, ge=0, doc="pulses per period (whole number)")
        flicker: float = P(0.0, ge=0, le=1, doc="chance per tick of a flash frame (0 = off)")
        flicker_rate: float = P(16.0, gt=0, doc="flicker ticks per second")
        flash: float = P(0.7, ge=0, le=1, doc="flash amount on flicker frames")
        flash_palette: str = P("#ffffff", doc="flash colour")
        shiver: float = P(0.0, ge=0, doc="random sideways jitter (px) per tick")

    def setup(self):
        p = self.p
        self._load_image(p.height, 3 * p.glow_radius * 2.5 + 4)
        a = self.img[..., 3]
        self.halo = glow_mask(self.apad, p.glow_radius / self.sc)
        self.halo_wide = glow_mask(self.apad, 2.5 * p.glow_radius / self.sc)
        self.rim = edge_mask(self.apad, 1.2 / self.sc) * self.apad
        noise = Noise(self.rng("noise"), period=(32, 32, 8))
        yy, xx = np.mgrid[0:self.ih, 0:self.iw].astype(np.float32) + 0.5
        cell = max(p.noise_scale / self.sc, 1.0)
        n = noise.fbm(xx / cell, yy / cell, 0.37, octaves=3)
        inside = a > 0.02
        depth = blur(a, 4.0 / self.sc + 1)
        depth = depth / max(float(depth.max()), 1e-6)
        self.cmap = _rank(p.sweep * depth + (1 - p.sweep) * _rank(n, inside), inside)
        self.tinted = _tinted(self.img, _rgb(p.tint_palette), p.tint) if p.tint > 0 else None
        self.flash_rgb = _rgb(p.flash_palette)
        cov = np.zeros_like(self.apad)
        P = self.pad
        if p.tint > 0:
            cov[P:P + self.ih, P:P + self.iw] = a * smoothstep(p.cover * 1.12, p.cover * 1.12 - 0.1, self.cmap)
        else:
            cov[P:P + self.ih, P:P + self.iw] = a
        self.cov = cov

    def get_bounds(self):
        m = 3 * self.p.glow_radius * 2.5 + 6 + self.p.shiver
        return (-self.W / 2 - m, -self.H - m, self.W / 2 + m, m)

    def draw(self, ctx, t):
        p = self.p
        ph = t / p.period
        k = 1 + p.pulse * math.sin(TAU * p.pulse_cycles * ph) if p.pulse_cycles else 1.0
        fl, dx = 0.0, 0.0
        if p.flicker > 0 or p.shiver > 0:
            r = self.tick_rng("flicker", t, p.flicker_rate).random(3)
            if p.flicker > 0 and r[0] < p.flicker:
                fl = p.flash * (0.6 + 0.4 * r[1])
            if p.shiver > 0:
                dx = p.shiver * (2 * r[2] - 1)
        self._paint(ctx.solid, self.img, dx, tint=self.flash_rgb, tint_amount=fl)
        if self.tinted is not None and p.cover > 0:
            lvl = p.cover * 1.12
            cm = self.cmap
            self._paint(ctx.solid, self.tinted, dx, tint=self.flash_rgb, tint_amount=fl,
                        mask=lambda uu, vv: smoothstep(lvl, lvl - 0.1, _sample_uv(cm, uu, vv)))
        kk = k * (1 + 1.5 * fl)
        if p.outline > 0:
            self._stamp(ctx.light, self.halo, 1.3 * p.outline * kk, dx)
            self._stamp(ctx.light, self.halo_wide, 0.45 * p.outline * kk, dx)
        if p.rim > 0:
            self._stamp(ctx.light, self.rim, p.rim * kk, dx)
        if p.sheen > 0:
            # diagonal band crossing the image once per period (plus a gap so it is not constant)
            span = self.W + self.H * 0.6
            pos = -span / 2 - 30 + (span + 60) * ph * 1.0
            bw = max(self.W * 0.1, 3)
            cov = self.cov

            def fn(X, Y):
                d = (X + (Y + self.H / 2) * 0.6) - pos
                px, py = self._to_img(X - dx, Y)
                return p.sheen * np.exp(-d * d / (2 * bw * bw)) * bilinear(cov, px, py)
            ctx.light.field(fn, (-self.W / 2 - 2 + dx, -self.H - 2, self.W / 2 + 2 + dx, 2), lowres="bilinear")


# =============================================================================
# sprite_emitter
# =============================================================================
def _draw_points(view, pos, alpha, size, style, prev=None, rand=None):
    for i in np.flatnonzero(alpha > 1e-3):
        x, y = float(pos[i, 0]), float(pos[i, 1])
        a, s = float(alpha[i]), float(size[i])
        if style in ("dot", "puff"):
            view.dot(x, y, s, a)
        elif style == "streak":
            tx, ty = float(prev[i, 0]), float(prev[i, 1])
            view.polyline([(tx, ty), ((x + tx) / 2, (y + ty) / 2), (x, y)], s * 0.45, a * 0.15, r_end=s, a_end=a)
        else:  # glint / plus
            L = s * (5 if style == "glint" else 3.5)
            w = s * (0.45 if style == "glint" else 0.6)
            rot = float(rand[i]) * math.pi if style == "glint" else 0.0
            for kk in range(4):
                ang = rot + kk * math.pi / 2
                view.polyline([(x, y), (x + L * math.cos(ang), y + L * math.sin(ang))], w, a, r_end=w * 0.3, a_end=0.0)
            view.dot(x, y, s * 0.9, a)


@atom("sprite_emitter", tags=["sprite", "particles", "fire", "electric", "ice", "loop"])
class SpriteEmitter(_SpriteImage, Atom):
    """Particles and status matter emitted from an image's silhouette (does NOT draw the image:
    stack it over a ``sprite`` / ``sprite_loop`` layer with the same image, height and position).

    ``region``: where things spawn: ``interior`` (anywhere on the silhouette), ``edge`` (outline
    band) or ``top_edge`` (upward-facing edges: head, shoulders, raised arms).

    ``style`` adds a status look on top of the generic particles (``count``, speeds, gravity,
    curl, ``particle_style``...):
    * ``burning``: flickering flame tongues licking up from the region, a heat glow on the
      edges, optional dark smoke (``smoke``); particles = rising embers.
    * ``electrified``: jagged arcs crawling between nearby edge points, re-jagged ``arc_rate``
      times per second, with sparks at their ends; particles = sparks.
    * ``frozen``: solid ice spikes growing out of the edges (``crystals``, ``ice_palette``),
      twinkling tips, cold mist sinking off the body (``mist``); particles = sparkles.
    * ``particles``: only the generic particles."""
    kind = "loop"
    outputs = {"light": Light(palette="fire", bloom=[[2, 0.45], [6, 0.3]], alpha_gamma=0.85),
               "smoke": Smoke(palette="dark_smoke", density=1.0, shadow=0.4),
               "solid": Solid(palette="ice")}
    streams = ("points", "particles", "flames", "arcs", "crystals", "mist")
    anchors = _sprite_anchors()

    class Params(Atom.Params):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        image: str = P("hero.png", doc="image asset whose silhouette emits (PNG with alpha)")
        height: float = P(128.0, ge=0, doc="displayed image height (px); 0 = native image size")
        region: str = P("top_edge", doc="where things spawn", choices=_REGIONS)
        edge_width: float = P(2.5, gt=0, doc="width of the silhouette edge band (px)")
        style: str = P("burning", doc="status look", choices=["burning", "electrified", "frozen", "particles"])
        # generic particles
        count: int = P(40, ge=0, le=5000, doc="number of particles (embers / sparks / sparkles)")
        direction: str = P("angle", doc="initial direction: fixed angle cone, or the outward edge normal", choices=["angle", "normal"])
        angle: float = P(-90.0, doc="emission angle (deg, clockwise from +x; -90 = up)")
        spread: float = P(40.0, ge=0, le=360, doc="emission cone width (deg)")
        speed_min: float = P(20.0, doc="minimum initial speed (px/s)")
        speed_max: float = P(60.0, doc="maximum initial speed (px/s)")
        gravity: float = P(-40.0, doc="gravity (px/s², +down; negative = rises)")
        drag: float = P(0.5, ge=0, doc="linear drag (1/s)")
        curl: float = P(25.0, ge=0, doc="swirling flow strength (px/s); 0 = straight")
        curl_scale: float = P(40.0, gt=0, doc="swirl eddy size (px)")
        laps_min: int = P(1, ge=1, doc="minimum particle lifetimes per period (whole number)")
        laps_max: int = P(3, ge=1, doc="maximum particle lifetimes per period (whole number)")
        particle_style: str = P("dot", doc="how particles are drawn", choices=_PSTYLES)
        size: float = P(1.1, gt=0, doc="particle size (px)")
        size_end: float = P(-1.0, doc="particle size at end of life (px); -1 = constant")
        brightness: float = P(1.6, ge=0, doc="particle intensity")
        fade_in: float = P(0.1, ge=0, le=1, doc="particle fade-in fraction of life")
        fade_out: float = P(1.3, ge=0, doc="particle fade-out power")
        twinkle: float = P(0.0, ge=0, le=1, doc="particle twinkle depth")
        halo: float = P(0.15, ge=0, doc="soft halo around each particle")
        output: str = P("light", doc="role the generic particles draw into", choices=["light", "smoke"])
        extent: float = P(60.0, ge=0, doc="how far beyond the image particles may travel (px; also the bounds)")
        # burning
        flames: int = P(14, ge=0, le=200, doc="burning: number of flame tongues")
        flame_height: float = P(20.0, gt=0, doc="burning: flame tongue height (px)")
        flame_width: float = P(3.5, gt=0, doc="burning: flame tongue base radius (px)")
        heat: float = P(0.5, ge=0, doc="burning / electrified: glow on the emitting edges")
        smoke: float = P(0.0, ge=0, doc="burning: smoke density (0 = off)")
        # electrified
        arcs: int = P(5, ge=0, le=60, doc="electrified: arcs alive at a time")
        arc_length: float = P(26.0, gt=0, doc="electrified: arc span (px)")
        arc_rate: float = P(14.0, gt=0, doc="electrified: re-jags per second")
        # frozen
        crystals: int = P(22, ge=0, le=400, doc="frozen: ice spikes on the edges")
        crystal_size: float = P(7.0, gt=0, doc="frozen: ice spike length (px)")
        ice_palette: str = P("ice", doc="frozen: ice spike colours (shadow -> lit)")
        mist: float = P(0.0, ge=0, doc="frozen: cold mist density sinking off the body (0 = off)")

    def setup(self):
        p = self.p
        self._load_image(p.height, p.edge_width * 3 + 4)
        w, self.nx, self.ny = _region_weight(self.apad, p.region, p.edge_width / self.sc)
        self.weight = w
        def pts(n, rng):
            q = sample_points(self.weight, n, rng)
            x, y = self._to_local(q[:, 0], q[:, 1])
            nx = bilinear(self.nx, q[:, 0], q[:, 1])
            ny = bilinear(self.ny, q[:, 0], q[:, 1])
            nn = np.hypot(nx, ny) + 1e-6
            return np.stack([x, y], 1), np.stack([nx / nn, ny / nn], 1)
        self._pts = pts
        # generic particles
        lm = max(p.laps_min, p.laps_max)
        cfg = EmitterConfig(count=int(p.count), shape="point", direction="angle", angle=p.angle,
                            spread=p.spread, speed=(p.speed_min, p.speed_max), life=(0.5, 1.0),
                            gravity=p.gravity, drag=p.drag, loop=True, period=p.period,
                            laps=(p.laps_min, lm), curl=p.curl, curl_scale=p.curl_scale)
        self.em = Emitter(self.rng("particles"), cfg)
        if p.count:
            pp, nn = pts(int(p.count), self.rng("points"))
            self.em.p0 = pp.astype(np.float64)
            if p.direction == "normal":
                spd = np.hypot(self.em.v0[:, 0], self.em.v0[:, 1])
                self.em.v0 = nn * spd[:, None]
        # burning flames
        rf = self.rng("flames")
        nf = int(p.flames) if p.style == "burning" else 0
        self.f_laps = rf.integers(1, 3, nf) if nf else np.zeros(0, int)
        self.f_r = rf.random((nf, 5))
        self.s_r = rf.random((16, 4))
        self.f_pos = [pts(2, rf)[0] for _ in range(nf)]      # a base point per lap parity
        self.s_pos = pts(16, rf)[0] if p.smoke > 0 and p.style == "burning" else None
        self.heat_map = (w / max(float(w.max()), 1e-6)).astype(np.float32)
        # electrified: candidate arc endpoints along the region
        self.arc_pts = pts(300, self.rng("arcs"))[0] if p.style == "electrified" else np.zeros((0, 2))
        # frozen crystals
        rc = self.rng("crystals")
        nc = int(p.crystals) if p.style == "frozen" else 0
        if nc:
            cp, cn = pts(nc, rc)
            ang = np.arctan2(cn[:, 1], cn[:, 0]) + np.radians(rc.normal(0, 18, nc))
            up = -math.pi / 2
            ang = ang + 0.25 * np.sin(up - ang)                 # lean a little upward
            self.cr = [(cp[i], float(ang[i]), p.crystal_size * (0.55 + 0.9 * rc.random()),
                        0.22 + 0.12 * rc.random(), rc.random()) for i in range(nc)]
            self.cr.sort(key=lambda c: c[2], reverse=True)       # big ones behind
        else:
            self.cr = []
        rm = self.rng("mist")
        self.m_pos = pts(18, rm)[0] if p.style == "frozen" and p.mist > 0 else None
        self.m_r = rm.random((18, 5))
        self.ice_dark = np.asarray(color_at(p.ice_palette, 0.0)) * 0.55
        self.ice_lit = color_at(p.ice_palette, 1.0)
        self.ice_mid = color_at(p.ice_palette, 0.55)

    def get_bounds(self):
        e = self.p.extent
        return (-self.W / 2 - e, -self.H - e, self.W / 2 + e, e * 0.6 + 4)

    def _box_fade(self, pos):
        x0, y0, x1, y1 = self.get_bounds()
        m = 0.18 * self.p.extent + 4
        d = np.minimum.reduce([pos[:, 0] - x0, x1 - pos[:, 0], pos[:, 1] - y0, y1 - pos[:, 1]])
        return smoothstep(0, m, d)

    def draw(self, ctx, t):
        p = self.p
        ph = t / p.period
        st = p.style
        if st == "frozen":
            self._draw_frozen(ctx, t, ph)
        elif st == "burning":
            self._draw_burning(ctx, t, ph)
        elif st == "electrified":
            self._draw_electric(ctx, t, ph)
        self._draw_particles(ctx, t)

    def _draw_particles(self, ctx, t):
        p = self.p
        em = self.em
        if em.n == 0:
            return
        pos, age, q, alive = em.state(t)
        a = life_alpha(q, p.fade_in, p.fade_out) * p.brightness * (0.6 + 0.8 * em.rand[:, 0])
        if p.twinkle > 0:
            a = a * (1 - p.twinkle + p.twinkle * (0.5 + 0.5 * np.sin(TAU * (4 * q + em.rand[:, 1]))) ** 2)
        a = a * self._box_fade(pos)
        sz = p.size * (0.7 + 0.6 * em.rand[:, 2])
        if p.size_end >= 0:
            sz = sz * (1 + (p.size_end / p.size - 1) * q)
        prev = None
        if p.particle_style == "streak":
            pp, ag2, _, _ = em.state(t - 0.04)
            prev = np.where((ag2 < age)[:, None], pp, pos)
        view = getattr(ctx, p.output)
        _draw_points(view, pos, a, sz, p.particle_style, prev, em.rand[:, 3])
        if p.halo > 0 and p.output == "light":
            for i in np.flatnonzero(a > 0.02):
                view.dot(float(pos[i, 0]), float(pos[i, 1]), float(sz[i]) * 4, float(a[i]) * p.halo)

    # -- burning -----------------------------------------------------------------------------
    def _draw_burning(self, ctx, t, ph):
        p = self.p
        if p.heat > 0:
            fl = 0.85 + 0.15 * math.sin(TAU * 3 * ph) * math.sin(TAU * 5 * ph + 1.3)
            self._stamp(ctx.light, self.heat_map, p.heat * fl)
        for i in range(len(self.f_laps)):
            laps = int(self.f_laps[i])
            r = self.f_r[i]
            x_ = ph * laps + r[0]
            q = x_ % 1.0
            lap = int(math.floor(x_)) % laps
            bx, by = (float(v) for v in self.f_pos[i][lap])
            env = math.sin(math.pi * q) ** 0.7
            Hh = p.flame_height * (0.55 + 0.9 * r[1]) * (0.35 + 0.65 * env)
            rad = p.flame_width * (0.6 + 0.6 * r[2]) * (0.5 + 0.5 * env)
            pts = []
            n = 7
            for j in range(n):
                s = j / (n - 1)
                sway = (3.5 + 3 * r[3]) * s ** 1.4 * math.sin(TAU * (2 * ph + r[4]) + 2.8 * s) \
                    + 1.5 * s * math.sin(TAU * (5 * ph + r[0]) + 4 * s)
                pts.append((bx + sway, by - Hh * s + rad * 0.6))
            a = 1.25 * env * (0.7 + 0.5 * r[3])
            ctx.light.polyline(pts, rad, a, r_end=rad * 0.15, a_end=a * 0.15)
            ctx.light.polyline(pts[:4], rad * 0.45, a * 1.1, r_end=rad * 0.15, a_end=a * 0.3)
        if self.s_pos is not None:
            for i, (bx, by) in enumerate(self.s_pos):
                r = self.s_r[i]
                q = (ph + r[0]) % 1.0
                x = bx + 10 * q * (r[1] - 0.5) + 4 * math.sin(TAU * (ph + r[2]))
                y = by - 70 * q - 6
                rr = 4 + 11 * q
                d = 0.16 * p.smoke * math.sin(math.pi * q) ** 1.2
                fade = float(self._box_fade(np.array([[x, y - rr]]))[0])
                ctx.smoke.dot(float(x), float(y), rr, d * fade)

    # -- electrified ---------------------------------------------------------------------------
    def _draw_electric(self, ctx, t, ph):
        p = self.p
        rng = self.tick_rng("arcs", t, p.arc_rate)
        if p.heat > 0:
            self._stamp(ctx.light, self.heat_map, p.heat * (0.6 + 0.6 * rng.random()))
        P_ = self.arc_pts
        if len(P_) < 2:
            return
        for _ in range(int(p.arcs)):
            i = int(rng.integers(len(P_)))
            a0 = P_[i]
            d = np.hypot(*(P_ - a0).T)
            ok = np.flatnonzero((d > p.arc_length * 0.35) & (d < p.arc_length * 1.2))
            if not len(ok):
                continue
            b0 = P_[int(ok[rng.integers(len(ok))])]
            mid = (a0 + b0) / 2 + rng.normal(0, p.arc_length * 0.15, 2)
            path = refine(rng, [a0, mid, b0], 0.32, 2.0)
            br = 0.7 + 0.6 * rng.random()
            ctx.light.polyline(path, 2.4, 0.35 * br)
            ctx.light.polyline(path, 0.6, 2.2 * br)
            if rng.random() < 0.5 and len(path) > 4:      # small fork
                j = int(rng.integers(1, len(path) - 2))
                tip = path[j] + rng.normal(0, p.arc_length * 0.3, 2)
                fork = refine(rng, [path[j], tip], 0.35, 2.0)
                ctx.light.polyline(fork, 0.45, 1.4 * br, a_end=0.2)
            for e in (a0, b0):
                ctx.light.dot(float(e[0]), float(e[1]), 1.6, 1.2 * br)

    # -- frozen ---------------------------------------------------------------------------------
    def _draw_frozen(self, ctx, t, ph):
        p = self.p
        if self.m_pos is not None:
            for i, (bx, by) in enumerate(self.m_pos):
                r = self.m_r[i]
                laps = 1 + int(r[4] * 2)
                q = (ph * laps + r[0]) % 1.0
                x = bx + (r[1] - 0.5) * 30 * q + 4 * math.sin(TAU * (ph + r[2]))
                y = by + 34 * q ** 1.3 - 4
                rr = 5 + 13 * q
                d = 0.1 * p.mist * math.sin(math.pi * q) ** 1.3 * (0.7 + 0.6 * r[3])
                fade = float(self._box_fade(np.array([[x, y + rr]]))[0])
                ctx.smoke.dot(float(x), float(y), rr, d * fade)
        cols = (tuple(float(v) for v in self.ice_dark), self.ice_mid, self.ice_lit)
        for (c, ang, L, wf, r) in self.cr:
            tw = max(0.0, math.sin(TAU * (ph * (1 + int(r * 3)) + r))) ** 6
            ice_spike(ctx, c, ang, L, wf, cols, tw)


def ice_spike(ctx, c, ang: float, L: float, wf: float, cols, twinkle: float = 0.0) -> None:
    """A two-facet solid ice spike from point ``c`` toward angle ``ang`` (radians), length
    ``L``, half-width ``wf * L``; ``cols`` = (outline, shadow, lit) RGB 0..255. ``twinkle``
    (0..1) adds a light glint at the tip."""
    dark, mid, lit = cols
    c = np.asarray(c, np.float64)
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    base = c - d * L * 0.3
    tip = c + d * L
    w = L * wf
    b1, b2 = base + n * w, base - n * w
    ctx.solid.polygon([b1 + (b1 - tip) * 0.12 + n * 0.6, tip + d * 1.1, b2 + (b2 - tip) * 0.12 - n * 0.6],
                      rgb=dark, soft=0.6)
    lit_left = (n[0] * -0.55 + n[1] * -0.85) > 0
    ctx.solid.polygon([b1, tip, base], rgb=lit if lit_left else mid, soft=0.5)
    ctx.solid.polygon([base, tip, b2], rgb=mid if lit_left else lit, soft=0.5)
    if twinkle > 0.02 and ctx.has("light"):
        tw = twinkle
        ctx.light.dot(float(tip[0]), float(tip[1]), 1.2, 1.6 * tw)
        ctx.light.polyline([(tip[0] - 5 * tw, tip[1]), (tip[0] + 5 * tw, tip[1])], 0.5, 1.2 * tw)
        ctx.light.polyline([(tip[0], tip[1] - 5 * tw), (tip[0], tip[1] + 5 * tw)], 0.5, 1.2 * tw)


# =============================================================================
# texture particles
# =============================================================================
class _TexParams(Atom.Params):
    image: str = P("coin.png", doc="image asset drawn for every particle (PNG with alpha)")
    size: float = P(16.0, gt=0, doc="particle height (px); width follows the image aspect")
    size_var: float = P(0.25, ge=0, le=1, doc="random size variation (fraction)")
    scale_start: float = P(1.0, ge=0, doc="size multiplier at birth")
    scale_end: float = P(1.0, ge=0, doc="size multiplier at end of life")
    count: int = P(24, ge=0, le=2000, doc="number of particles")
    shape: str = P("point", doc="spawn shape", choices=["point", "circle", "disk", "line", "box", "ellipse"])
    radius: float = P(0.0, ge=0, doc="spawn shape size (px)")
    radius_y: float = P(-1.0, doc="ellipse y radius / box half-height (px); -1 = same as radius")
    direction: str = P("angle", doc="initial direction mode", choices=["angle", "radial", "inward", "tangent"])
    angle: float = P(-90.0, doc="emission angle (deg, clockwise from +x; -90 = up)")
    spread: float = P(60.0, ge=0, le=360, doc="emission cone width (deg)")
    speed_min: float = P(120.0, doc="minimum initial speed (px/s)")
    speed_max: float = P(260.0, doc="maximum initial speed (px/s)")
    gravity: float = P(500.0, doc="gravity (px/s², +down)")
    drag: float = P(0.4, ge=0, doc="linear drag (1/s)")
    curl: float = P(0.0, ge=0, doc="swirling flow strength (px/s); 0 = straight (leaves: 40+)")
    curl_scale: float = P(60.0, gt=0, doc="swirl eddy size (px)")
    sway: float = P(0.0, ge=0, doc="sideways sway amplitude (px) (falling leaves)")
    spin: float = P(240.0, ge=0, doc="spin speed (deg/s, random direction)")
    spin_var: float = P(0.6, ge=0, le=1, doc="random spin speed variation")
    flip: float = P(0.0, ge=0, doc="3D flip: width oscillations per second (spinning coins); 0 = off")
    orient: str = P("spin", doc="spin = random rotation + spin; velocity = point along the flight (shards)", choices=["spin", "velocity"])
    fade_in: float = P(0.05, ge=0, le=1, doc="fade-in fraction of life")
    fade_out: float = P(0.15, ge=0, le=1, doc="fade-out fraction at the end of life")
    glint: float = P(0.0, ge=0, doc="twinkling light glint on each particle (0 = off)")
    extent: float = P(220.0, gt=0, doc="particles fade before |x| or |y| reaches this (px); also the bounds")


def _tex_setup(self, cfg):
    p = self.p
    self.img = self.asset(p.image)
    ih, iw = self.img.shape[:2]
    self.aspect = iw / ih
    self.em = Emitter(self.rng("particles"), cfg)
    r = self.rng("style").random((max(cfg.count, 1), 4))
    self.sz = p.size * (1 + p.size_var * (2 * r[:, 0] - 1))
    sign = np.where(r[:, 1] < 0.5, -1.0, 1.0)
    self.spin = sign * p.spin * (1 - p.spin_var * r[:, 2])
    self.rot0 = r[:, 3] * 360.0
    self.fl_ph = self.em.rand[:, 3]


def _tex_draw(self, ctx, pos, age, q, alive, vel=None):
    p = self.p
    em = self.em
    a = smoothstep(0, max(p.fade_in, 1e-4), q) * smoothstep(1, 1 - max(p.fade_out, 1e-4), q) if p.fade_in > 0 \
        else smoothstep(1, 1 - max(p.fade_out, 1e-4), q)
    d = np.maximum(np.abs(pos[:, 0]), np.abs(pos[:, 1]))
    a = a * smoothstep(p.extent, p.extent * 0.85, d) * alive
    sc = p.scale_start + (p.scale_end - p.scale_start) * np.clip(q, 0, 1)
    tvar = age if not em.cfg.loop else q * em.cfg.period / np.maximum(em.laps, 1)
    for i in np.flatnonzero(a > 2e-3):
        h = float(self.sz[i] * sc[i])
        if h <= 0.3:
            continue
        w = h * self.aspect
        if p.flip > 0:
            w *= abs(math.cos(math.pi * (p.flip * float(tvar[i]) + float(self.fl_ph[i]))))
            w = max(w, h * 0.06)
        if p.orient == "velocity" and vel is not None:
            rot = math.degrees(math.atan2(vel[i, 1], vel[i, 0])) + 90.0
        else:
            rot = float(self.rot0[i] + self.spin[i] * float(tvar[i]))
        x, y = float(pos[i, 0]), float(pos[i, 1])
        ctx.solid.image(self.img, x, y, w=w, h=h, rot=rot, a=float(a[i]))
        if p.glint > 0:
            tw = max(0.0, math.sin(TAU * (1.5 * float(tvar[i]) + float(em.rand[i, 1])))) ** 8
            if tw > 0.02:
                g = p.glint * tw * float(a[i])
                L = h * 0.45
                gx, gy = x - h * 0.15, y - h * 0.15
                for ang in (0, math.pi / 2, math.pi, 1.5 * math.pi):
                    ctx.light.polyline([(gx, gy), (gx + L * math.cos(ang), gy + L * math.sin(ang))], 0.6, g, r_end=0.2, a_end=0.0)
                ctx.light.dot(gx, gy, 1.2, g)


def _velocity(em, age, dt=1e-3):
    p1 = em.positions(age + dt, age)
    p0 = em.positions(np.maximum(age - dt, 0), age)
    return (p1 - p0) / (2 * dt)


@atom("texture_particles", tags=["particles", "texture", "image", "solid", "loop"])
class TextureParticles(Atom):
    """Seamlessly looping particles drawn as an image (falling leaves, drifting coins, petals,
    logos): spin, 3D flip, scale over life, fade, gravity, drag, curl, sway."""
    kind = "loop"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="holy", bloom=[[2, 0.4]], alpha_gamma=0.85)}
    streams = ("particles", "style")

    class Params(_TexParams):
        period: float = P(2.0, gt=0, doc="loop length (s)")
        laps_min: int = P(1, ge=1, doc="minimum lifetimes per period (whole number)")
        laps_max: int = P(1, ge=1, doc="maximum lifetimes per period (whole number)")

    def setup(self):
        p = self.p
        cfg = EmitterConfig(count=p.count, shape=p.shape, radius=p.radius,
                            radius_y=None if p.radius_y < 0 else p.radius_y, direction=p.direction,
                            angle=p.angle, spread=p.spread, speed=(p.speed_min, p.speed_max), gravity=p.gravity,
                            drag=p.drag, sway=p.sway, sway_freq=1.0, loop=True, period=p.period,
                            laps=(p.laps_min, max(p.laps_min, p.laps_max)), curl=p.curl, curl_scale=p.curl_scale)
        _tex_setup(self, cfg)

    def get_bounds(self):
        e = self.p.extent + self.p.size
        return (-e, -e, e, e)

    def draw(self, ctx, t):
        pos, age, q, alive = self.em.state(t)
        vel = _velocity(self.em, age) if self.p.orient == "velocity" else None
        _tex_draw(self, ctx, pos, age, q, alive, vel)


@atom("texture_burst", tags=["particles", "texture", "image", "solid", "oneshot", "coins"])
class TextureBurst(Atom):
    """One-shot burst of particles drawn as an image (coin fountain, shard burst, leaf puff):
    spin, 3D flip, scale over life, fade, gravity, drag, curl and an optional floor they bounce
    on (``floor`` px below the origin)."""
    kind = "oneshot"
    outputs = {"solid": Solid(palette="rock"),
               "light": Light(palette="holy", bloom=[[2, 0.4]], alpha_gamma=0.85)}
    streams = ("particles", "style")

    class Params(_TexParams):
        duration: float = P(1.6, gt=0, doc="total length (s); particles are cut at this time")
        life_min: float = P(0.9, gt=0, doc="minimum particle lifetime (s)")
        life_max: float = P(1.4, gt=0, doc="maximum particle lifetime (s)")
        stagger: float = P(0.15, ge=0, doc="spawn times spread over this many seconds")
        floor: float = P(-1.0, doc="floor height below the origin (px) particles bounce on; -1 = no floor")
        bounce: float = P(0.45, ge=0, le=1, doc="floor: restitution (fraction of speed kept per bounce)")
        friction: float = P(0.6, ge=0, le=1, doc="floor: fraction of sideways speed kept per bounce")

    def setup(self):
        p = self.p
        cfg = EmitterConfig(count=p.count, shape=p.shape, radius=p.radius,
                            radius_y=None if p.radius_y < 0 else p.radius_y, direction=p.direction,
                            angle=p.angle, spread=p.spread, speed=(p.speed_min, p.speed_max),
                            life=(p.life_min, max(p.life_min, p.life_max)), gravity=p.gravity, drag=p.drag,
                            sway=p.sway, sway_freq=1.0, stagger=p.stagger, curl=p.curl, curl_scale=p.curl_scale)
        _tex_setup(self, cfg)

    def get_bounds(self):
        e = self.p.extent + self.p.size
        return (-e, -e, e, e)

    def _bounce(self, age):
        """Closed-form floor bounces (no drag vertically): positions, velocities, rest mask."""
        p, em = self.p, self.em
        g = max(p.gravity, 1e-3)
        fl = p.floor
        x0, y0 = em.p0[:, 0].astype(np.float64), em.p0[:, 1].astype(np.float64)
        vx, vy = em.v0[:, 0].astype(np.float64), em.v0[:, 1].astype(np.float64)
        a = np.maximum(age, 0).astype(np.float64)
        n = len(a)
        tb = np.zeros(n)                         # start time of the current arc
        yb, vyb, vxb = y0.copy(), vy.copy(), vx.copy()
        xb = x0.copy()
        rest = np.zeros(n, bool)
        for _ in range(4):
            disc = vyb * vyb + 2 * g * np.maximum(fl - yb, 0)
            th = (-vyb + np.sqrt(disc)) / g          # time to reach the floor from this arc start
            hit = (tb + th <= a) & ~rest
            if not hit.any():
                break
            v_hit = vyb + g * th
            xb = np.where(hit, xb + vxb * th, xb)
            tb = np.where(hit, tb + th, tb)
            yb = np.where(hit, fl, yb)
            vyb = np.where(hit, -p.bounce * v_hit, vyb)
            vxb = np.where(hit, vxb * p.friction, vxb)
            rest |= hit & (np.abs(vyb) < 25)
        dt = a - tb
        y = np.where(rest, fl, yb + vyb * dt + 0.5 * g * dt * dt)
        y = np.minimum(y, fl)
        x = np.where(rest, xb, xb + vxb * dt)
        vel = np.stack([np.where(rest, 0, vxb), np.where(rest, 0, vyb + g * dt)], 1)
        return np.stack([x, y], 1), vel, rest

    def draw(self, ctx, t):
        p = self.p
        if t >= p.duration:
            return
        em = self.em
        age, q, alive = em.age(t)
        if p.floor >= 0:
            pos, vel, rest = self._bounce(age)
        else:
            pos, age, q, alive = em.state(t)
            vel = _velocity(em, age) if p.orient == "velocity" else None
        q = np.where(alive, q, 1.0)
        end = float(smoothstep(p.duration, p.duration * 0.88, t))
        _tex_draw(self, ctx, pos, age, q, alive.astype(np.float32) * end, vel)
