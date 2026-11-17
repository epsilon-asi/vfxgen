"""Physical / elemental / combat effects.

Physical matter (dust, water, rock, ice) is mostly NON-emissive here: it is
either lit density (`_shade_volume`, a normal-mapped variant of the core smoke
shading) or flat-shaded `Solid` shapes, with restrained light only for
highlights, flashes and sparks. Every effect exposes its knobs as keyword
arguments (see each docstring's Params block); every colour ramp other than
the main `palette` is its own `*_palette` argument (preset name or hex).
"""
import math

import numpy as np

from .core import (SS, TAU, Canvas, Noise, PALETTES, _edge_mask, blur, effect,  # noqa: F401
                   get_palette, grid, ramp, refine, shift, smoothstep)

_grid = grid

# Extra ramps used by this module (all overridable per effect).
PALETTES.setdefault("water", [(0.0, (14, 42, 78)), (0.45, (42, 100, 150)),
                              (0.8, (130, 186, 220)), (1.0, (222, 242, 252))])
PALETTES.setdefault("glint", [(0.0, (120, 190, 235)), (0.5, (205, 238, 255)), (1.0, (255, 255, 255))])
PALETTES.setdefault("steel", [(0.0, (70, 110, 210)), (0.45, (150, 195, 255)),
                              (0.8, (225, 240, 255)), (1.0, (255, 255, 255))])
PALETTES.setdefault("impact", [(0.0, (230, 120, 40)), (0.4, (255, 200, 110)),
                               (0.75, (255, 240, 200)), (1.0, (255, 255, 255))])
# solid matter: index = facet lighting (0 = shadow side, 1 = lit side)
PALETTES.setdefault("ice", [(0.0, (28, 92, 150)), (1.0, (205, 240, 255))])
# rock: 0 = crack interior, 0.15 = rock shadow, 0.6 = rock lit face, 1 = crack lip
PALETTES.setdefault("rock", [(0.0, (30, 22, 17)), (0.15, (40, 33, 28)),
                             (0.6, (112, 94, 76)), (1.0, (150, 126, 96))])


def _col(stops, k):
    """One RGB tuple (0..255) from a ramp at position k."""
    return tuple(float(v) for v in ramp(stops, np.array(float(np.clip(k, 0, 1)))))


# ===========================================================================
# Original light-only effects (now parameterised; defaults = original look)
# ===========================================================================
@effect("whirlwind", "wind", "major", "Tornado funnel of spiralling wind streaks with base dust", loop=True)
def fx_whirlwind(rng, palette, scale=1.0, frames=16, streaks=340, bands=50, dust=70,
                 spin=1, top_radius=1.0, sway=1.0, streak_width=1.0, glow=1.0, dust_ring=True):
    """Spiralling streaks rise and wrap around a swaying funnel; seamless loop.

    Params:
      scale: output size multiplier (base 256x384)
      frames: frames in the loop
      streaks: number of thin wind streaks
      bands: number of soft wide bands behind the streaks
      dust: number of dust specks circling the base
      spin: extra integer spin multiplier (keeps the loop seamless)
      top_radius: funnel width at the top (1 = default)
      sway: how far the funnel snakes sideways
      streak_width: width multiplier for streaks and bands
      glow: light exposure / brightness
      dust_ring: draw the dust ring at the base
    """
    w, h = int(256 * scale), int(384 * scale)
    s = w / 128
    yb, yt = h - 16 * s, 14 * s
    H = yb - yt
    rb, rt = 5 * s, min(46 * top_radius, 56) * s
    spin = max(1, int(spin))

    def radius(hh):
        return rb + (rt - rb) * hh ** 1.35

    def centre_x(hh, t):
        return w / 2 + 7 * sway * s * hh * np.sin(TAU * (0.9 * hh - t))

    N = int(streaks)
    P = dict(h0=rng.random(N), th0=rng.uniform(0, TAU, N), k=rng.choice([2, 3], N) * spin,
             rj=rng.uniform(0.75, 1.06, N), a=rng.uniform(0.5, 1.0, N),
             arc=rng.uniform(0.7, 1.6, N))
    NB = int(bands)
    B = dict(h0=rng.random(NB), th0=rng.uniform(0, TAU, NB), k=rng.choice([2, 3], NB) * spin,
             rj=rng.uniform(0.8, 1.05, NB), arc=rng.uniform(1.6, 2.6, NB))
    ND = int(dust) if dust_ring else 0
    D = dict(th0=rng.uniform(0, TAU, ND), k=rng.choice([1, 2], ND) * spin,
             rad=rng.uniform(8, 40, ND) * s, lift=rng.uniform(0, 10, ND) * s,
             a=rng.uniform(0.2, 0.55, ND))

    def streak(c, hh, th, k, rj, arc, width, a):
        steps = 9
        ths = th - np.sign(k) * np.linspace(arc, 0, steps)   # tail -> head
        rad = radius(hh) * rj
        xs = centre_x(hh, t_) + rad * np.cos(ths)
        ys = yb - hh * H + 0.2 * rad * np.sin(ths)
        depth = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(th))
        fade = math.sin(math.pi * hh) ** 1.3
        c.polyline(np.stack([xs, ys], 1), width * 0.6, a * depth * fade * 0.15,
                   r_end=width, a_end=a * depth * fade)

    out = []
    for f in range(frames):
        t_ = f / frames  # integer spin/rise counts => seamless loop
        c = Canvas(w, h)
        for i in range(NB):
            hh = (B["h0"][i] + t_) % 1.0
            th = B["th0"][i] + TAU * B["k"][i] * t_
            streak(c, hh, th, B["k"][i], B["rj"][i], B["arc"][i], 3.5 * s * streak_width, 0.25)
        for i in range(N):
            hh = (P["h0"][i] + t_) % 1.0
            th = P["th0"][i] + TAU * P["k"][i] * t_
            streak(c, hh, th, P["k"][i], P["rj"][i], P["arc"][i], 0.7 * s * streak_width,
                   0.9 * P["a"][i])
        for i in range(ND):
            th = D["th0"][i] + TAU * D["k"][i] * t_
            x = w / 2 + D["rad"][i] * math.cos(th)
            y = yb + 4 * s - D["lift"][i] + 0.22 * D["rad"][i] * math.sin(th)
            depth = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(th))
            c.dot(x, y, 0.7 * s, D["a"][i] * depth)
        out.append(c.render(palette, bloom=((1.2 * s, 0.35), (4 * s, 0.25)),
                            exposure=1.1 * glow, alpha_gamma=1.0))
    return out, 16


@effect("sparks", "fire", "minor", "Spark burst: flash, shockwave ring, ballistic sparks", loop=False)
def fx_sparks(rng, palette, scale=1.0, frames=14, count=90, speed=1.0, gravity=1.0, drag=3.0,
              life=1.0, spark_width=1.0, brightness=1.0, duration=0.55, flash=True, shockwave=True):
    """Radial burst of ballistic sparks with a flash and a quick shockwave ring.

    Params:
      scale: output size multiplier (base 256x256)
      frames: number of frames
      count: number of sparks
      speed: launch speed multiplier
      gravity: gravity multiplier (0 = no fall)
      drag: air drag (higher = sparks stop sooner)
      life: spark lifetime multiplier
      spark_width: spark thickness multiplier
      brightness: spark intensity multiplier
      duration: seconds covered by the animation
      flash: draw the central flash + afterglow
      shockwave: draw the expanding ring
    """
    size = int(256 * scale)
    s = size / 128
    cx = cy = size / 2
    T = duration
    grav = 70 * s * gravity
    N = int(count)
    ang = rng.uniform(0, TAU, N)
    spd = rng.uniform(70, 200, N) * s * speed
    lif = rng.uniform(0.25, 0.55, N) * life
    amp = rng.uniform(1.4, 3.2, N) * brightness
    wid = rng.uniform(0.55, 0.95, N) * s * spark_width

    def pos(i, t):
        k = (1 - math.exp(-drag * t)) / drag
        return (cx + math.cos(ang[i]) * spd[i] * k,
                cy + math.sin(ang[i]) * spd[i] * k + 0.5 * grav * t * t)

    out = []
    for f in range(frames):
        t = (f + 0.35) * T / frames
        c = Canvas(size, size)
        if flash:
            c.dot(cx, cy, (5 + 40 * t) * s, 3.2 * math.exp(-t / 0.05))     # flash
            c.dot(cx, cy, 16 * s, 0.6 * math.exp(-t / 0.12))               # afterglow
        R = (6 + 46 * (1 - math.exp(-t / 0.07))) * s                        # shockwave
        ring = 0.9 * math.exp(-t / 0.09)
        if shockwave and ring > 0.01:
            X, Y = _grid(size, size)
            rr = np.hypot(X - cx, Y - cy)
            c.buf += ring * np.exp(-((rr - R) / (2.2 * s)) ** 2)
        for i in range(N):
            if t >= lif[i]:
                continue
            fade = (1 - t / lif[i]) ** 1.4
            tail = max(t - 0.07, 0.0)
            pts = [pos(i, tail), pos(i, (tail + t) / 2), pos(i, t)]
            c.polyline(pts, wid[i] * 0.5, amp[i] * fade * 0.15,
                       r_end=wid[i], a_end=amp[i] * fade)
        out.append(c.render(palette, bloom=((1.5 * s, 0.6), (5 * s, 0.6), (14 * s, 0.4))))
    return out, 24


# ===========================================================================
# Shared helpers for the physical / combat effects below.
#
# Besides the core Canvas (light + smoke) these effects use a third kind of
# layer, `Solid`: opaque-ish painted matter (ice shards, rock chunks, crack
# decals) composited with plain "over" at SS resolution. Each effect renders
# one or more layers and stacks them with `_stack` (bottom first).
# ===========================================================================
def _bb(shape, xmin, xmax, ymin, ymax):
    x0 = max(int(xmin * SS), 0)
    x1 = min(int(xmax * SS) + 2, shape[1])
    y0 = max(int(ymin * SS), 0)
    y1 = min(int(ymax * SS) + 2, shape[0])
    return x0, x1, y0, y1


def _poly_sd(X, Y, pts):
    """Signed distance to a polygon (negative inside, even-odd)."""
    inside = np.zeros(X.shape, bool)
    dmin = np.full(X.shape, np.inf, np.float32)
    n = len(pts)
    for i in range(n):
        xa, ya = pts[i]
        xb, yb = pts[(i + 1) % n]
        cond = (ya > Y) != (yb > Y)
        xcross = xa + (Y - ya) * (xb - xa) / (yb - ya + 1e-12)
        inside ^= cond & (X < xcross)
        vx, vy = xb - xa, yb - ya
        t = np.clip(((X - xa) * vx + (Y - ya) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
        np.minimum(dmin, np.hypot(X - xa - t * vx, Y - ya - t * vy), out=dmin)
    return np.where(inside, -dmin, dmin)


def _capsule_sd(X, Y, p0, p1, r0, r1):
    """Signed distance to a tapered capsule (round ends)."""
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    t = np.clip(((X - p0[0]) * vx + (Y - p0[1]) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
    return np.hypot(X - p0[0] - t * vx, Y - p0[1] - t * vy) - (r0 + (r1 - r0) * t)


def capsule(c, p0, p1, r0, r1, a, soft=0.6, target=None):
    """Hard-edged (anti-aliased) tapered capsule into a Canvas buffer.
    Crisp counterpart of polyline: droplets, debris streaks, speed lines."""
    buf = c.dens if target == "dens" else c.buf
    m = max(r0, r1) + 2 * soft + 1
    x0, x1, y0, y1 = _bb(buf.shape, min(p0[0], p1[0]) - m, max(p0[0], p1[0]) + m,
                         min(p0[1], p1[1]) - m, max(p0[1], p1[1]) + m)
    if x0 >= x1 or y0 >= y1:
        return
    sd = _capsule_sd(c.X[y0:y1, x0:x1], c.Y[y0:y1, x0:x1], p0, p1, r0, r1)
    buf[y0:y1, x0:x1] += a * smoothstep(soft, -soft, sd)


class Solid:
    """Painted, non-emissive matter: straight 'over' compositing of flat-shaded
    shapes at SS resolution (ice facets, rock chunks, crack decals)."""

    def __init__(self, w, h):
        self.w, self.h = int(w), int(h)
        self.X, self.Y = grid(self.w, self.h)
        self.rgb = np.zeros(self.X.shape + (3,), np.float32)   # premultiplied
        self.a = np.zeros(self.X.shape, np.float32)

    def _paint(self, x0, x1, y0, y1, cov, rgb, a):
        k = np.clip(cov * a, 0, 1)
        sub = self.rgb[y0:y1, x0:x1]
        sub *= (1 - k)[..., None]
        sub += k[..., None] * (np.asarray(rgb, np.float32) / 255.0)
        sa = self.a[y0:y1, x0:x1]
        sa *= 1 - k
        sa += k

    def polygon(self, pts, rgb, a=1.0, soft=0.6):
        pts = np.asarray(pts, np.float32)
        m = 2 * soft + 1
        x0, x1, y0, y1 = _bb(self.a.shape, pts[:, 0].min() - m, pts[:, 0].max() + m,
                             pts[:, 1].min() - m, pts[:, 1].max() + m)
        if x0 >= x1 or y0 >= y1:
            return
        sd = _poly_sd(self.X[y0:y1, x0:x1], self.Y[y0:y1, x0:x1], pts)
        self._paint(x0, x1, y0, y1, smoothstep(soft, -soft, sd), rgb, a)

    def capsule(self, p0, p1, r0, r1, rgb, a=1.0, soft=0.6):
        m = max(r0, r1) + 2 * soft + 1
        x0, x1, y0, y1 = _bb(self.a.shape, min(p0[0], p1[0]) - m, max(p0[0], p1[0]) + m,
                             min(p0[1], p1[1]) - m, max(p0[1], p1[1]) + m)
        if x0 >= x1 or y0 >= y1:
            return
        sd = _capsule_sd(self.X[y0:y1, x0:x1], self.Y[y0:y1, x0:x1], p0, p1, r0, r1)
        self._paint(x0, x1, y0, y1, smoothstep(soft, -soft, sd), rgb, a)

    def stroke(self, pts, r, rgb, a=1.0, soft=0.6, r_end=None):
        """Open polyline of capsules (cracks)."""
        pts = np.asarray(pts, np.float32)
        rs = np.linspace(r, r if r_end is None else r_end, len(pts))
        m = rs.max() + 2 * soft + 1
        x0, x1, y0, y1 = _bb(self.a.shape, pts[:, 0].min() - m, pts[:, 0].max() + m,
                             pts[:, 1].min() - m, pts[:, 1].max() + m)
        if x0 >= x1 or y0 >= y1:
            return
        X, Y = self.X[y0:y1, x0:x1], self.Y[y0:y1, x0:x1]
        sd = np.full(X.shape, np.inf, np.float32)
        for i in range(len(pts) - 1):
            np.minimum(sd, _capsule_sd(X, Y, pts[i], pts[i + 1], rs[i], rs[i + 1]), out=sd)
        self._paint(x0, x1, y0, y1, smoothstep(soft, -soft, sd), rgb, a)

    def rgba(self):
        h, w = self.h, self.w
        a = self.a.reshape(h, SS, w, SS).mean(axis=(1, 3)) * _edge_mask(w, h)
        prem = self.rgb.reshape(h, SS, w, SS, 3).mean(axis=(1, 3))
        rgb = prem / np.maximum(self.a.reshape(h, SS, w, SS).mean(axis=(1, 3)), 1e-6)[..., None]
        return np.dstack([np.clip(rgb, 0, 1), a]).astype(np.float32)


def _stack(*layers):
    """Composite straight-alpha layers (bottom first) -> RGBA uint8.
    Accepts uint8 RGBA from Canvas.render or float RGBA from Solid.rgba."""
    out = None
    for L in layers:
        if L is None:
            continue
        L = L.astype(np.float32) / 255.0 if L.dtype == np.uint8 else L
        if out is None:
            out = L.copy()
            continue
        ta, ba = L[..., 3:4], out[..., 3:4]
        a = ta + ba * (1 - ta)
        prem = L[..., :3] * ta + out[..., :3] * ba * (1 - ta)
        out = np.dstack([prem / np.maximum(a, 1e-6), a])
    a = out[..., 3]
    out[..., :3][a < 1.5 / 255] = 0
    out[..., 3][a < 1.5 / 255] = 0
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def _noise_field(nz, w, h, x, y, z, octaves=4):
    """fbm evaluated at OUTPUT resolution, upsampled to the SS buffer.
    x, y: callables or arrays at output res (noise-cell units)."""
    v = nz.fbm(x, y, z, octaves=octaves)
    return np.repeat(np.repeat(v, SS, axis=0), SS, axis=1).astype(np.float32)


def _shade_volume(dens, w, h, stops, opacity=1.0, bump=6.0, soft=1.2, shadow=1.0,
                  wrap=0.35, spec=0.0, spec_pow=24.0, rim=0.0, light=(-0.5, -0.72, 0.48)):
    """Shade an SS density buffer as a bumpy volume -> float RGBA (straight).

    Unlike Canvas.render's flat self-shadow, this derives a surface normal from
    the (blurred, sqrt-compressed) density, so billows get lit tops and shaded
    undersides (cauliflower dust) and liquids get a rounded, glossy body.
    stops: colour ramp indexed by light (0 = shadow, 1 = fully lit).
    spec > 0 adds a white Blinn highlight, rim > 0 a fresnel edge (water)."""
    D = np.maximum(dens.reshape(h, SS, w, SS).mean(axis=(1, 3)), 0) * _edge_mask(w, h)
    Hf = blur(np.sqrt(D), soft)
    gy_, gx_ = np.gradient(Hf)
    nx, ny, nzz = -gx_ * bump, -gy_ * bump, np.ones_like(Hf)
    nl = np.sqrt(nx * nx + ny * ny + 1)
    L = np.array(light, np.float32)
    L = L / np.linalg.norm(L)
    ndl = (nx * L[0] + ny * L[1] + nzz * L[2]) / nl
    diff = np.clip((ndl + wrap) / (1 + wrap), 0, 1)
    size = max(w, h)
    off = size / 45
    Db = blur(D, max(1.5, size / 90))
    occl = shift(Db, L[0] * off, L[1] * off) + 0.5 * shift(Db, L[0] * off * 2.2, L[1] * off * 2.2)
    lit = diff * np.exp(-shadow * occl)
    rgb = ramp(stops, np.clip(lit, 0, 1)) / 255.0
    a = 1 - np.exp(-D * opacity)
    if rim > 0:
        fres = np.clip(1 - nzz / nl, 0, 1) ** 1.5 * rim * smoothstep(0.01, 0.15, a)
        top = np.array(get_palette(stops)[-1][1], np.float32) / 255
        rgb = rgb + (top - rgb) * fres[..., None] * 0.7
        a = a + (1 - a) * fres * 0.6
    if spec > 0:
        Hv = np.array([L[0], L[1], L[2] + 1.0])
        Hv = Hv / np.linalg.norm(Hv)
        sp = np.clip((nx * Hv[0] + ny * Hv[1] + nzz * Hv[2]) / nl, 0, 1) ** spec_pow * spec
        sp = sp * smoothstep(0.02, 0.25, a)
        rgb = rgb + (1 - rgb) * np.clip(sp, 0, 1)[..., None]
        a = a + (1 - a) * np.clip(sp, 0, 1) * 0.8
    return np.dstack([rgb, a]).astype(np.float32)


def _out_grid(w, h):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs + 0.5, ys + 0.5


LIGHT = np.array([-0.55, -0.85]) / np.hypot(-0.55, -0.85)   # toward the light


def _ice_shard(sol, c, p, ang, L, W, alpha, edge, skew=0.0, stops="ice"):
    """Kite-shaped ice shard: two flat-shaded facets in `sol` (colours from the
    `stops` ramp by facet lighting), bright outline + spine in light canvas `c`.
    p = centre, ang = axis direction."""
    d = np.array([math.cos(ang), math.sin(ang)])
    n = np.array([-d[1], d[0]])
    tip = p + d * L * 0.62
    back = p - d * L * 0.38
    mid = p + d * L * (0.02 + skew)
    sl = mid + n * W * 0.5
    sr = mid - n * W * 0.62
    for side, nn in ((sl, n), (sr, -n)):
        k = 0.5 + 0.5 * float(nn @ LIGHT)
        col = _col(stops, k ** 0.9)
        sol.polygon([tip, side, back], col, alpha * (0.72 + 0.2 * k), soft=0.55)
    if edge > 0 and c is not None:
        r = max(0.35, W * 0.035)
        lit = sl if float(n @ LIGHT) > 0 else sr
        c.polyline([back, lit, tip], r, edge * 0.5, r_end=r * 0.7, a_end=edge)
        c.polyline([back, tip], r * 0.8, edge * 0.25, a_end=edge * 0.7)
        c.dot(tip[0], tip[1], r * 2.2, edge * 0.6)


def _crescent(c, cx, cy, R, rot, squash, a0, sweep, u_t, u_h, thick, amp,
              edge=3.0, body=0.55, peak=1.6, soft=0.6):
    """Sword-slash crescent into c.buf. An (optionally squashed/rotated) arc of
    radius R from angle a0 sweeping `sweep` rad; only u in [u_t, u_h] (0..1
    along the sweep) is visible. Thickest towards the head, a sharp hot outer
    (cutting) edge, glow body fading inward and towards the tail."""
    if u_h - u_t < 1e-3 or amp <= 0:
        return
    m = R + 4
    x0, x1, y0, y1 = _bb(c.buf.shape, cx - m, cx + m, cy - m, cy + m)
    if x0 >= x1 or y0 >= y1:
        return
    dx = c.X[y0:y1, x0:x1] - cx
    dy = c.Y[y0:y1, x0:x1] - cy
    cr, sr = math.cos(rot), math.sin(rot)
    lx = dx * cr + dy * sr
    ly = (-dx * sr + dy * cr) / squash
    r = np.hypot(lx, ly)
    phi = np.arctan2(ly, lx)
    u = (((phi - a0) * np.sign(sweep)) % TAU) / abs(sweep)
    v = (u - u_t) / (u_h - u_t)
    vc = np.clip(v, 0, 1)
    # thickness profile within the visible window; thins as the window closes
    # so the dying tail never leaves a fat wedge
    th = thick * np.sin(math.pi * vc ** peak) ** 0.85 * min(1.0, (u_h - u_t) / 0.6) ** 0.7
    d = R - r                                   # inward distance from cutting edge
    inside_v = smoothstep(-0.02, 0.0, v) * smoothstep(1.02, 1.0, v)
    outer = smoothstep(-soft, soft, d)
    dn = np.clip(d / np.maximum(th, 1e-3), 0, 1.5)
    inner = smoothstep(1.0, 0.75, dn)
    along = 0.25 + 0.75 * vc ** 0.8
    ew = max(0.8, thick * 0.07)
    taper = np.clip(th / (ew * 4), 0, 1)
    I = amp * inside_v * outer * along * (
        body * (1 - np.clip(dn, 0, 1)) ** 1.3 * inner +
        edge * np.exp(-np.maximum(d, 0) / ew) * taper)
    c.buf[y0:y1, x0:x1] += I.astype(np.float32)


def _sweep_curve(tt, hit=0.3):
    """Head/tail positions (0..1 along the arc) and brightness for a slash at
    normalised time tt: the head whips in fast, the tail chases it out."""
    u_h = 0.35 + 0.65 * (1 - (1 - min(tt / hit, 1)) ** 2.2)
    u_t = 0.98 * smoothstep(0.08, 1.0, tt) ** 1.15
    amp = 1.0 if tt < hit else (1 - smoothstep(hit, 1.0, tt)) ** 0.8
    return u_h, min(u_t, u_h), amp


def _make_slash(rng, size, s, R, rot, sq=0.6, thick=46, lines=7, arc_deg=165.0, through=0.0):
    """Geometry for one crescent slash, positioned so the crescent's bounding
    box is centred in the frame; `through` (0..1) instead pulls the blade's
    midpoint toward the centre (so several slashes cross there)."""
    sweep = math.radians(min(max(arc_deg, 30.0), 300.0))
    sl = dict(R=R, rot=rot, sq=sq, thick=thick * s, cx=0.0, cy=0.0,
              a0=math.radians(272.5) - sweep / 2, sweep=sweep,
              lu=rng.uniform(0.25, 0.95, lines), lr=R + rng.uniform(-0.3, 0.12, lines) * R,
              ll=rng.uniform(0.16, 0.34, lines) * R)
    pts = np.array([_arc_point(sl, u, R - thick * s * 0.3)[0] for u in np.linspace(0, 1, 33)])
    ref = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2])
    ref = ref * (1 - through) + pts[16] * through
    sl["cx"], sl["cy"] = size / 2 - ref[0], size / 2 - ref[1]
    return sl


def _arc_point(sl, u, rad=None):
    """Screen point (and unit tangent) at fraction u along a slash arc."""
    ph = sl["a0"] + sl["sweep"] * u
    rad = sl["R"] if rad is None else rad
    lx, ly = rad * math.cos(ph), rad * math.sin(ph) * sl["sq"]
    tx, ty = -math.sin(ph), math.cos(ph) * sl["sq"]
    cr, sr = math.cos(sl["rot"]), math.sin(sl["rot"])
    tn = math.hypot(tx, ty)
    return (np.array([sl["cx"] + lx * cr - ly * sr, sl["cy"] + lx * sr + ly * cr]),
            np.array([tx * cr - ty * sr, tx * sr + ty * cr]) / tn)


def _draw_slash(c, sl, tt, s, gain=1.0, edge=1.0):
    """Crescent + speed lines for normalised slash time tt in [0, 1]."""
    if not 0 <= tt <= 1:
        return
    u_h, u_t, amp = _sweep_curve(tt)
    _crescent(c, sl["cx"], sl["cy"], sl["R"], sl["rot"], sl["sq"], sl["a0"], sl["sweep"],
              u_t, u_h, sl["thick"] * (1 - 0.3 * tt), amp * 1.2 * gain, edge=2.8 * edge, body=0.9)
    if tt > 0.75:
        return
    k = (1 - tt / 0.75) * 0.5 * gain
    for i in range(len(sl["lu"])):
        if sl["lu"][i] > u_h:
            continue
        p, tg = _arc_point(sl, sl["lu"][i], sl["lr"][i])
        off = 0.1 * sl["R"] * tt
        c.polyline([p - tg * (sl["ll"][i] + off), p + tg * off], 0.3 * s, 0.0,
                   r_end=0.7 * s, a_end=k)


def _glint(c, x, y, L, a, ang=0.0, r=0.5):
    """Four-point star sparkle: two crossed tapered strokes + a hot core."""
    for k in range(2):
        th = ang + k * math.pi / 2
        dx, dy = math.cos(th) * L, math.sin(th) * L
        c.polyline([(x - dx, y - dy), (x, y)], r * 0.3, 0.0, r_end=r, a_end=a)
        c.polyline([(x, y), (x + dx, y + dy)], r, a, r_end=r * 0.3, a_end=0.0)
    c.dot(x, y, r * 1.6, a * 0.8)


def _ballistic_streak(c, p, v, tt, drag, grav, wid, amp, tail=0.04):
    """Draw a spark streak at time tt for a particle launched at p with v."""
    def pos(q):
        kk = (1 - math.exp(-drag * q)) / drag
        return p + v * kk + np.array([0.0, 0.5 * grav * q * q])
    t0 = max(tt - tail, 0.0)
    c.polyline([pos(t0), pos((t0 + tt) / 2), pos(tt)], wid * 0.4, amp * 0.1, r_end=wid, a_end=amp)


# ===========================================================================
# Slashes
# ===========================================================================
@effect("slash", "steel", "minor", "Single crescent sword slash with hot cutting edge", loop=False)
def fx_slash(rng, palette, scale=1.0, frames=8, radius=1.0, thickness=1.0, arc_angle=165.0,
             squash=0.6, angle=0.0, angle_jitter=26.0, flip=-1, edge=1.0, glow=1.0,
             speed_lines=7, sparks=0):
    """One crescent: the arc whips in over ~2 frames, the tail chases the head out.

    Params:
      scale: output size multiplier (base 256x256)
      frames: number of frames (6-10 reads best)
      radius: arc radius multiplier
      thickness: crescent thickness multiplier
      arc_angle: degrees of arc the blade sweeps
      squash: vertical squash of the arc (1 = circle, lower = flatter, more perspective)
      angle: rotation of the slash in degrees
      angle_jitter: random extra rotation, +/- degrees
      flip: -1 random, 0 normal, 1 upside-down (mirror the swing)
      edge: brightness of the hot cutting edge
      glow: overall light exposure
      speed_lines: number of thin speed lines along the cut
      sparks: number of small sparks thrown off the edge
    """
    size = int(256 * scale)
    s = scale
    fl = int(rng.integers(0, 2)) if flip < 0 else int(flip)
    rot = math.radians(angle + rng.uniform(-1, 1) * angle_jitter) + math.pi * fl
    R = min(100 * radius, 112) * s
    sl = _make_slash(rng, size, s, R, rot, sq=squash, thick=46 * thickness, lines=int(speed_lines),
                     arc_deg=arc_angle)
    sp = []
    for _ in range(int(sparks)):
        u = rng.uniform(0.35, 0.95)
        p, tg = _arc_point(sl, u)
        a = math.atan2(tg[1], tg[0]) + rng.normal(0, 0.4)
        sp.append((u * 0.3, p, np.array([math.cos(a), math.sin(a)]) * rng.uniform(150, 320) * s,
                   rng.uniform(0.6, 1.0)))
    out = []
    for f in range(frames):
        tt = f / (frames - 1)
        c = Canvas(size, size)
        _draw_slash(c, sl, tt, s, edge=edge)
        for t0, p, v, life in sp:
            q = (tt - t0) * 0.33
            if 0 < q < life * 0.25:
                fade = (1 - q / (life * 0.25)) ** 1.2
                _ballistic_streak(c, p, v, q, 5.0, 300 * s, 0.7 * s, 1.8 * fade)
        out.append(c.render(palette, bloom=((1.2 * s, 0.5), (4 * s, 0.45), (12 * s, 0.3)),
                            exposure=1.25 * glow, alpha_gamma=0.9))
    return out, 24


@effect("slash_combo", "steel", "major",
        "Three crossing sword slashes in quick succession with sparks and speed lines", loop=False)
def fx_slash_combo(rng, palette, scale=1.0, frames=20, slash_count=3, interval=0.17,
                   slash_duration=0.36, radius=1.0, thickness=1.0, arc_angle=165.0, sparks=26,
                   speed_lines=7, edge=1.0, glow=1.0, cross_flash=True, angle_jitter=7.0):
    """Staggered `_crescent` slashes (an X-cross, then a wide finishing cut),
    each shedding ballistic sparks off its edge, with a glint flash as each
    blade passes the centre.

    Params:
      scale: output size multiplier (base 384x384)
      frames: number of frames
      slash_count: number of slashes (pattern: X-cross, finisher, then alternating)
      interval: seconds between slash starts
      slash_duration: seconds each slash lives
      radius: arc radius multiplier
      thickness: crescent thickness multiplier
      arc_angle: degrees of arc each blade sweeps
      sparks: sparks shed per slash
      speed_lines: speed lines per slash
      edge: brightness of the hot cutting edges
      glow: overall light exposure
      cross_flash: glint flash at the centre as each blade crosses it
      angle_jitter: random rotation per slash, +/- degrees
    """
    size = int(384 * scale)
    s = scale
    n = max(1, int(slash_count))
    base = [(0.75, 0.55, 50, 140), (-0.75 + math.pi, 0.55, 50, 140), (0.0, 0.42, 58, 155)]
    slashes, starts = [], []
    for k in range(n):
        if k < 3:
            r0, sq, th, R = base[k]
        else:
            r0, sq, th, R = base[k % 2]
            r0 += rng.uniform(-0.4, 0.4)
        if k == 2:
            r0 += math.pi * rng.integers(0, 2)
        rot = r0 + math.radians(rng.uniform(-1, 1) * angle_jitter)
        slashes.append(_make_slash(rng, size, s, min(R * radius, 168) * s, rot, sq=sq,
                                   thick=th * thickness, lines=int(speed_lines), arc_deg=arc_angle,
                                   through=0.65 if k < 2 else 0.4))
        starts.append(k * interval * (1.06 if k == 2 else 1.0))
    dur = slash_duration
    T = starts[-1] + dur + 0.12

    sp = []
    for k, sl in enumerate(slashes):
        for _ in range(int(sparks)):
            u = rng.uniform(0.3, 0.95)
            p, tg = _arc_point(sl, u)
            ang = math.atan2(tg[1], tg[0]) + rng.normal(0, 0.45)
            sp.append(dict(t0=starts[k] + dur * 0.18 * (u - 0.3) / 0.65, p=p,
                           v=np.array([math.cos(ang), math.sin(ang)]) * rng.uniform(140, 380) * s,
                           life=rng.uniform(0.18, 0.34), amp=rng.uniform(1.2, 2.4),
                           wid=rng.uniform(0.5, 0.9) * s))

    out = []
    for f in range(frames):
        t = (f + 0.6) / frames * T
        c = Canvas(size, size)
        for k, sl in enumerate(slashes):
            last = k == n - 1 and n > 1
            _draw_slash(c, sl, (t - starts[k]) / dur, s, gain=1.15 if last else 1.0, edge=edge)
            th = t - starts[k] - dur * 0.12
            if cross_flash and th > 0:
                a = (2.2 if k else 1.6) * math.exp(-th / 0.05)
                if a > 0.02:
                    gp, _ = _arc_point(sl, 0.5, sl["R"] - sl["thick"] * 0.2)
                    _glint(c, gp[0], gp[1], (16 + 140 * th) * s, a, ang=sl["rot"] + 0.6, r=1.3 * s)
        for q in sp:
            tt = t - q["t0"]
            if 0 < tt < q["life"]:
                fade = (1 - tt / q["life"]) ** 1.3
                _ballistic_streak(c, q["p"], q["v"], tt, 5.0, 380 * s, q["wid"], q["amp"] * fade)
        out.append(c.render(palette, bloom=((1.2 * s, 0.5), (4 * s, 0.45), (13 * s, 0.32)),
                            exposure=1.2 * glow, alpha_gamma=0.9))
    return out, 24


@effect("hit_impact", "impact", "minor",
        "Comic hit: white-hot star burst, radial speed lines, small ring, chunky sparks", loop=False)
def fx_hit_impact(rng, palette, scale=1.0, frames=8, star_points=9, star_size=1.0, spikiness=1.0,
                  speed_lines=16, sparks=10, spark_size=1.0, ring=True, glow=1.0, duration=0.33):
    """A spiky star polygon (hot core, softer outer star) pops then collapses;
    tapered radial lines and chunky ballistic sparks fly out; a thin ring.

    Params:
      scale: output size multiplier (base 256x256)
      frames: number of frames
      star_points: points on the star burst
      star_size: star radius multiplier
      spikiness: how deep the star's notches are (0 = disc, 1 = default, 1.5 = needle-sharp)
      speed_lines: number of radial speed lines
      sparks: number of chunky sparks
      spark_size: spark thickness/length multiplier
      ring: draw the expanding ring
      glow: overall light exposure
      duration: seconds the hit lasts (sets how far sparks fly)
    """
    size = int(256 * scale)
    s = scale
    cx = cy = size / 2
    n = max(3, int(star_points))
    R0 = min(62 * star_size, 92) * s
    a0 = rng.uniform(0, TAU)
    outer = rng.uniform(0.6, 1.0, n)
    outer[rng.integers(0, n)] = 1.0
    jit = rng.uniform(-0.25, 0.25, n)
    inner_k = max(0.08, 1 - 0.62 * spikiness)

    def star(R, k_in):
        pts = []
        for i in range(n):
            a = a0 + (i + jit[i] * 0.5) / n * TAU
            pts.append((cx + math.cos(a) * R * outer[i], cy + math.sin(a) * R * outer[i]))
            b = a0 + (i + 0.5) / n * TAU
            pts.append((cx + math.cos(b) * R * k_in, cy + math.sin(b) * R * k_in))
        return pts

    NL = int(speed_lines)
    l_a = rng.uniform(0, TAU, NL)
    l_r = rng.uniform(0.55, 0.95, NL)
    l_len = rng.uniform(22, 44, NL) * s
    l_w = rng.uniform(0.8, 1.6, NL) * s
    NS = int(sparks)
    s_a = rng.uniform(0, TAU, NS)
    s_v = rng.uniform(320, 540, NS) * s
    s_w = rng.uniform(1.4, 2.4, NS) * s * spark_size

    out = []
    for f in range(frames):
        u = f / (frames - 1)                      # 0..1 over the hit
        t = u * duration
        c = Canvas(size, size)
        pop = (0.7 + 0.3 * min(u / 0.12, 1)) * (1 - smoothstep(0.25, 0.7, u) * 0.75)
        star_a = 1 - smoothstep(0.3, 0.72, u)
        if star_a > 0:
            c.polygon(star(R0 * pop, inner_k), 0.5 * star_a, soft=0.7 * s)
            c.polygon(star(R0 * pop * 0.6, min(inner_k * 1.1, 0.9)), 1.6 * star_a, soft=0.7 * s)
            c.dot(cx, cy, R0 * 0.18 * pop, 2.0 * star_a)
        if ring:
            Rr = R0 * (0.5 + 0.9 * (1 - math.exp(-u / 0.25)))
            c.ring(cx, cy, Rr, (0.9 + 1.2 * u) * s, 0.7 * (1 - smoothstep(0.1, 0.6, u)))
        lf = 1 - smoothstep(0.15, 0.6, u)
        for i in range(NL):
            if lf <= 0:
                break
            r1 = R0 * (l_r[i] + 0.9 * u)
            d = np.array([math.cos(l_a[i]), math.sin(l_a[i])])
            p0 = np.array([cx, cy]) + d * r1
            p1 = p0 + d * l_len[i] * (1 - 0.4 * u)
            c.polyline([p0, p1], l_w[i], 1.6 * lf, r_end=0.2 * s, a_end=0.0)
        for i in range(NS):
            life = duration * 0.95
            if t >= life:
                continue
            fade = (1 - t / life) ** 1.1
            d = np.array([math.cos(s_a[i]), math.sin(s_a[i])])
            _ballistic_streak(c, np.array([cx, cy]) + d * R0 * 0.3, d * s_v[i], t, 4.5, 500 * s,
                              s_w[i], 2.4 * fade, tail=0.06 * spark_size)
        out.append(c.render(palette, bloom=((1.0 * s, 0.35), (4 * s, 0.3), (12 * s, 0.22)),
                            exposure=1.15 * glow, alpha_gamma=0.85))
    return out, 24


# ===========================================================================
# Ice
# ===========================================================================
@effect("frost_nova", "frost", "major",
        "Ice explosion: crystal shards burst outward, frost ring, lingering cold mist, sparkles", loop=False)
def fx_frost_nova(rng, palette, scale=1.0, frames=24, shard_count=14, small_shards=30,
                  shard_size=1.0, spread=1.0, ring=True, ring_radius=1.0, mist_amount=1.0,
                  sparkles=40, ice_dust=70, glow=1.0,
                  ice_palette="ice", mist_palette="steam"):
    """Shards are flat-shaded `Solid` kites (two facets, ice_palette) with
    bright light-canvas edges; the frost ring is a noise-broken light ring;
    mist is noise-eroded lit density.

    Params:
      scale: output size multiplier (base 384x384)
      frames: number of frames
      shard_count: big shards (spread evenly around the burst)
      small_shards: small shards (random directions)
      shard_size: shard length multiplier
      spread: how far shards travel
      ring: draw the frost ring and its inner haze
      ring_radius: frost ring final radius multiplier
      mist_amount: cold mist density (0 = off)
      sparkles: twinkling four-point sparkles
      ice_dust: tiny fast ice streaks
      glow: light exposure for edges, ring, sparkles
      ice_palette: shard facet colours (shadow -> lit)
      mist_palette: mist colours (shadow -> lit)
    """
    size = int(384 * scale)
    s = scale
    cx = cy = size / 2
    nz = Noise(rng, period=(16, 16, 8))
    X, Y = _out_grid(size, size)
    Xs, Ys = grid(size, size)
    rr = np.hypot(Xs - cx, Ys - cy)
    phi = np.arctan2(Ys - cy, Xs - cx)
    Rmax = min(140 * ring_radius, 160)

    def ring_R(tn):
        return (12 + Rmax * (1 - math.exp(-tn / 0.17))) * s

    NB, NS = int(shard_count), int(small_shards)
    N = NB + NS
    ang = np.concatenate([(np.arange(NB) + rng.uniform(-0.3, 0.3, NB)) / max(NB, 1) * TAU,
                          rng.uniform(0, TAU, NS)])
    big = np.arange(N) < NB
    L = np.where(big, rng.uniform(36, 56, N), rng.uniform(10, 20, N)) * s * shard_size
    W = L * rng.uniform(0.28, 0.4, N)
    dist = np.where(big, rng.uniform(70, 112, N), rng.uniform(60, 140, N)) * s * spread
    dist = np.minimum(dist, size / 2 - 18 * s - L * 0.62)        # keep tips inside the frame
    tau = rng.uniform(0.09, 0.16, N)
    spin = rng.uniform(-1.2, 1.2, N)
    tilt = rng.uniform(-0.25, 0.25, N)
    skew = rng.uniform(-0.12, 0.12, N)
    f0 = rng.uniform(0.45, 0.7, N)            # fade window (normalised time)
    f1 = np.minimum(f0 + rng.uniform(0.2, 0.3, N), 0.94)

    NM = 90 if mist_amount > 0 else 0
    m_ang = rng.uniform(0, TAU, NM)
    m_k = 0.92 - rng.uniform(0, 1, NM) ** 2 * 0.6
    m_r = rng.uniform(5, 11, NM) * s
    m_a = rng.uniform(0.2, 0.45, NM) * mist_amount

    NG = int(sparkles)
    g_ang = rng.uniform(0, TAU, NG)
    g_k = rng.uniform(0.2, 1.0, NG)
    g_t0 = rng.uniform(0.05, 0.6, NG)
    g_life = rng.uniform(0.18, 0.32, NG)
    g_L = rng.uniform(4, 9, NG) * s
    ND = int(ice_dust)
    d_ang = rng.uniform(0, TAU, ND)
    d_spd = rng.uniform(120, 330, ND) * s * spread
    d_life = rng.uniform(0.35, 0.8, ND)

    out = []
    for f in range(frames):
        tn = (f + 0.6) / frames            # normalised time 0..1
        R = ring_R(tn)
        light = Canvas(size, size)
        mist = Canvas(size, size)
        sol = Solid(size, size)

        light.dot(cx, cy, (10 + 70 * tn) * s, 3.0 * math.exp(-tn / 0.05))
        light.dot(cx, cy, 26 * s, 0.7 * math.exp(-tn / 0.15))

        fade_r = 1 - smoothstep(0.3, 0.75, tn)
        if ring and fade_r > 0:
            brk = nz.fbm(np.cos(phi) * 3 + 8, np.sin(phi) * 3 + 8, tn * 2, octaves=3)
            mod = smoothstep(-0.5, 0.4, brk)
            wid = (1.6 + 3 * tn) * s
            d = rr - R
            ringv = np.exp(-(d / wid) ** 2) * (0.15 + 1.1 * mod)
            haze = smoothstep(R, R * 0.5, rr) * np.exp(np.minimum(d, 0) / (R * 0.18))
            light.buf += (fade_r * (1.4 * ringv + 0.18 * haze * mod)).astype(np.float32)

        mf = smoothstep(0.0, 0.12, tn) * (1 - smoothstep(0.4, 0.98, tn))
        for i in range(NM if mf > 0 else 0):
            rad = R * m_k[i]
            mist.dot(cx + math.cos(m_ang[i]) * rad, cy + math.sin(m_ang[i]) * rad - 10 * s * tn,
                     m_r[i] * (0.8 + 1.1 * tn), m_a[i] * mf, target="dens")
        if mist.dens.any():
            E = 1 + 1.5 * tn
            n = _noise_field(nz, size, size, (X - cx) / (18 * s * E), (Y - cy) / (18 * s * E), 3 + tn * 1.5)
            mist.dens *= smoothstep(-0.3, 0.5, n) * 1.6

        for i in np.argsort(L):
            fade = 1 - smoothstep(f0[i], f1[i], tn)
            if fade <= 0:
                continue
            rad = 4 * s + dist[i] * (1 - math.exp(-tn / tau[i]))
            p = np.array([cx + math.cos(ang[i]) * rad, cy + math.sin(ang[i]) * rad])
            grow = min(1.0, 0.55 + tn / 0.08)
            shrink = 0.6 + 0.4 * fade
            _ice_shard(sol, light, p, ang[i] + tilt[i] + spin[i] * tn, L[i] * grow * shrink,
                       W[i] * grow * shrink, fade ** 0.7, (1.6 if big[i] else 1.1) * fade, skew[i],
                       stops=ice_palette)

        for i in range(ND):
            if tn > d_life[i]:
                continue
            fade = (1 - tn / d_life[i]) ** 1.3
            d = np.array([math.cos(d_ang[i]), math.sin(d_ang[i])])
            _ballistic_streak(light, np.array([cx, cy]), d * d_spd[i], tn, 5.0, 0.0, 0.6 * s,
                              1.4 * fade, tail=0.05)

        for i in range(NG):
            u = (tn - g_t0[i]) / g_life[i]
            if 0 < u < 1:
                rad = R * g_k[i]
                _glint(light, cx + math.cos(g_ang[i]) * rad, cy + math.sin(g_ang[i]) * rad,
                       g_L[i] * (0.6 + 0.4 * math.sin(math.pi * u)), math.sin(math.pi * u) ** 2 * 2.2,
                       ang=0.3, r=0.55 * s)

        lay_light = light.render(palette, bloom=((1.2 * s, 0.45), (4.5 * s, 0.4), (14 * s, 0.3)),
                                 exposure=1.1 * glow, alpha_gamma=0.9)
        lay_mist = mist.render(palette, smoke=mist_palette, smoke_opacity=0.7, shadow=0.35)
        out.append(_stack(lay_mist, sol.rgba(), lay_light))
    return out, 24


def _crystal_pieces(rng, cx, cy, w, h, k):
    """Split a hexagonal crystal (front view of a pointed hex prism) into
    fragments: k wedges around an off-centre core point, each cut once more
    across the middle -> inner + outer pieces. Returns list of (pts, face)."""
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
    cuts = np.sort(rng.choice(len(per), k, replace=False))
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
    for p in pieces:
        cxp = p[:, 0].mean()
        face = 1.0 if cxp < -w * 0.17 else (0.0 if cxp > w * 0.17 else 0.55)   # hex prism faces
        out.append((p + (cx, cy), face))
    return out


@effect("ice_shatter", "frost", "minor", "Ice crystal cracks, then breaks into tumbling shards and glitter",
        loop=False)
def fx_ice_shatter(rng, palette, scale=1.0, frames=16, pieces=9, crystal_size=1.0, burst=1.0,
                   gravity=1.0, glitter=36, crack_frames=2, mist_amount=1.0, glow=1.0,
                   ice_palette="ice", mist_palette="steam"):
    """A faceted hex crystal (flat-shaded `Solid` fragments in ice_palette)
    shows glowing cracks, then bursts: fragments fly ballistically with spin,
    glitter twinkles and falls, a puff of cold mist.

    Params:
      scale: output size multiplier (base 256x256)
      frames: number of frames
      pieces: wedges the crystal splits into (each also splits in two)
      crystal_size: crystal size multiplier
      burst: shard launch speed multiplier
      gravity: gravity multiplier
      glitter: number of glitter sparkles
      crack_frames: frames the intact crystal shows cracks before breaking
      mist_amount: cold mist puff density (0 = off)
      glow: light exposure for edges, cracks and glitter
      ice_palette: crystal facet colours (shadow -> lit)
      mist_palette: mist colours (shadow -> lit)
    """
    size = int(256 * scale)
    s = scale
    cx, cy = size / 2, size / 2 - 4 * s
    cw, ch = min(54 * crystal_size, 90) * s, min(108 * crystal_size, 180) * s
    frags = _crystal_pieces(rng, cx, cy, cw, ch, max(3, int(pieces)))
    G = 380 * s * gravity
    cf = max(0, int(crack_frames))
    T_after = 0.6                       # seconds of flight after the break
    nz = Noise(rng, period=(16, 16, 8))
    X, Y = _out_grid(size, size)
    core = np.array([cx, cy])
    mv = []
    for pts, face in frags:
        cen = pts.mean(axis=0)
        d = cen - core + rng.normal(0, 2, 2) * s
        d = d / (np.linalg.norm(d) + 1e-6)
        sp = rng.uniform(90, 200) * s * burst
        mv.append(dict(cen=cen, rel=pts - cen, face=face, v=d * sp + np.array([0, -rng.uniform(40, 120)]) * s * burst,
                       w=rng.uniform(-7, 7), f0=rng.uniform(0.45, 0.75)))
    NG = int(glitter)
    g_p = core + rng.normal(0, 1, (NG, 2)) * np.array([cw * 0.4, ch * 0.35])
    g_v = rng.normal(0, 1, (NG, 2)) * 70 * s * burst - np.array([0, 40 * s])
    g_t0 = rng.uniform(0.0, 0.35, NG)
    g_life = rng.uniform(0.15, 0.3, NG)
    g_L = rng.uniform(3, 6.5, NG) * s
    n_after = max(frames - cf, 1)

    out = []
    for f in range(frames):
        light = Canvas(size, size)
        mist = Canvas(size, size)
        sol = Solid(size, size)
        if f < cf:                                          # intact, cracking
            k = (f + 1) / (cf + 1)
            for m in mv:
                pts = m["cen"] + m["rel"]
                col = _col(ice_palette, 0.25 + 0.6 * m["face"])
                sol.polygon(pts, col, 0.92, soft=0.5)
            for m in mv:
                pts = m["cen"] + m["rel"]
                light.polyline(np.vstack([pts, pts[:1]]), 0.5 * s, 1.4 * k)
            light.dot(cx, cy, 20 * s, 0.5 * k)
        else:
            t = (f - cf + 0.5) / n_after * T_after
            tn = t / T_after
            light.dot(cx, cy, (12 + 80 * t) * s, 2.2 * math.exp(-t / 0.05))
            for m in mv:
                p = m["cen"] + m["v"] * t + np.array([0, 0.5 * G * t * t])
                # fade with time, and early if a shard heads for the frame edge
                edge_d = min(p[0], size - p[0], p[1], size - p[1])
                fade = (1 - smoothstep(m["f0"], 0.97, tn)) * smoothstep(26 * s, 56 * s, edge_d)
                if fade <= 0:
                    continue
                a = m["w"] * t
                cr, sr = math.cos(a), math.sin(a)
                rel = m["rel"] @ np.array([[cr, sr], [-sr, cr]]) * (0.75 + 0.25 * fade)
                pts = p + rel
                kf = np.clip(0.25 + 0.6 * m["face"] + 0.25 * math.sin(a + m["face"] * 3), 0, 1)
                sol.polygon(pts, _col(ice_palette, kf), 0.9 * fade ** 0.7, soft=0.5)
                light.polyline(np.vstack([pts, pts[:1]]), 0.45 * s, 0.9 * fade)
                light.dot(pts[0][0], pts[0][1], 1.2 * s, 0.8 * fade)
            for i in range(NG):
                u = (tn - g_t0[i]) / g_life[i] if g_life[i] > 0 else -1
                if 0 < u < 1:
                    gp = g_p[i] + g_v[i] * t + np.array([0, 0.25 * G * t * t])
                    _glint(light, gp[0], gp[1], g_L[i], math.sin(math.pi * u) ** 2 * 2.0, ang=0.3,
                           r=0.5 * s)
            mf = mist_amount * smoothstep(0, 0.1, tn) * (1 - smoothstep(0.35, 1.0, tn))
            if mf > 0:
                for j in range(10):
                    a = j / 10 * TAU
                    rr_ = (10 + 50 * tn) * s
                    mist.dot(cx + math.cos(a) * rr_, cy + math.sin(a) * rr_ * 0.6 - 10 * s * tn,
                             (10 + 18 * tn) * s, 0.12 * mf, target="dens")
                n = _noise_field(nz, size, size, (X - cx) / (20 * s), (Y - cy) / (20 * s), 1 + tn * 2)
                mist.dens *= 0.3 + 1.1 * smoothstep(-0.5, 0.5, n)
        lay_light = light.render(palette, bloom=((1.0 * s, 0.45), (4 * s, 0.4), (11 * s, 0.25)),
                                 exposure=1.1 * glow, alpha_gamma=0.9)
        lay_mist = mist.render(palette, smoke=mist_palette, smoke_opacity=0.8, shadow=0.2)
        out.append(_stack(lay_mist, sol.rgba(), lay_light))
    return out, 24


# ===========================================================================
# Earth
# ===========================================================================
def _rock_shape(rng, r, n=None):
    """Random convex-ish rock outline (offsets around 0, radius ~r)."""
    n = n or int(rng.integers(5, 8))
    th = np.linspace(0, TAU, n, endpoint=False) + rng.uniform(-0.25, 0.25, n)
    rad = r * rng.uniform(0.7, 1.0, n)
    return np.stack([np.cos(th) * rad, np.sin(th) * rad], 1)


def _draw_rock(sol, shape, x, y, rot, alpha, stops="rock"):
    """Two-tone rock: shadow-colour silhouette + lit facet nudged toward the light."""
    cr, sr = math.cos(rot), math.sin(rot)
    pts = shape @ np.array([[cr, sr], [-sr, cr]]) + (x, y)
    sol.polygon(pts, _col(stops, 0.15), alpha, soft=0.5)
    r = np.abs(shape).max()
    facet = (pts - (x, y)) * 0.55 + (x, y) + LIGHT * r * 0.36
    sol.polygon(facet, _col(stops, 0.6), alpha * 0.9, soft=0.6)


@effect("ground_slam", "dust", "major",
        "Side-view ground impact: cracks, squashed shockwave, billowing dust, arcing rock debris", loop=False)
def fx_ground_slam(rng, palette, scale=1.0, frames=28, crack_count=9, crack_length=1.0,
                   dust_amount=1.0, dust_spread=1.0, dust_rise=1.0, rocks=20, pebbles=40,
                   rock_size=1.0, launch=1.0, gravity=1.0, duration=1.25, shockwave=True,
                   flash=True, rock_palette="rock", flash_palette="impact"):
    """Cracks and rocks are `Solid` matter; the cloud is billowing density
    (puffs with sub-billows, eroded by expanding fbm) shaded by `_shade_volume`
    with the main palette as its lit ramp; light is only the brief impact flash.

    Params:
      scale: output size multiplier (base 384x256)
      frames: number of frames
      crack_count: radial ground cracks (some get a side branch)
      crack_length: crack length multiplier
      dust_amount: dust cloud density (0 = no cloud)
      dust_spread: how far the dust rolls out sideways
      dust_rise: how high the central dust column climbs
      rocks: rock chunks thrown up
      pebbles: small dirt specks thrown up
      rock_size: rock chunk size multiplier
      launch: debris launch speed multiplier
      gravity: gravity multiplier for debris
      duration: seconds covered by the animation
      shockwave: draw the squashed ground shockwave (light + dust ring)
      flash: draw the impact flash and hot crack glow
      rock_palette: rock/crack colours (0 crack, 0.15 rock shadow, 0.6 rock lit, 1 crack lip)
      flash_palette: colour ramp of the impact flash
    """
    w, h = int(384 * scale), int(256 * scale)
    s = scale
    cx, gy = w / 2, h - 52 * s
    SQ = 0.26                       # ground-plane squash
    T = duration
    nz = Noise(rng, period=(16, 16, 8))
    X, Y = _out_grid(w, h)
    Xs, Ys = grid(w, h)
    # dust hugs the ground (flat underside) and thins out before the frame edge
    keep = (smoothstep(gy + 34 * s, gy + 8 * s, Ys) *
            smoothstep(6 * s, 52 * s, np.minimum(Xs, w - Xs)) *
            smoothstep(6 * s, 40 * s, Ys)).astype(np.float32)

    def proj(p):
        p = np.asarray(p)
        return np.stack([cx + p[:, 0], gy + p[:, 1] * SQ], 1)

    # cracks: radial on the ground plane; jagged in screen space so cracks
    # running toward the camera don't curl up
    cracks = []
    NC = max(0, int(crack_count))
    for i in range(NC):
        a = (i + rng.uniform(-0.35, 0.35)) / NC * TAU
        Lc = min(rng.uniform(70, 150) * crack_length, 160) * s * (0.55 + 0.45 * abs(math.cos(a)))
        end = proj([(math.cos(a) * Lc, math.sin(a) * Lc)])[0]
        pts = refine(rng, [(cx, gy), end], 0.14, 8 * s)
        cracks.append((pts, rng.uniform(3.0, 4.4) * s, 0.0))
        for _ in range(int(rng.integers(0, 2))):
            j = int(rng.integers(len(pts) // 4, len(pts) * 3 // 4))
            b = a + rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
            Lb = Lc * rng.uniform(0.25, 0.45)
            q = pts[j] + (math.cos(b) * Lb, math.sin(b) * Lb * SQ)
            cracks.append((refine(rng, [pts[j], q], 0.18, 6 * s), 1.8 * s, j / len(pts)))
    crack_c, lip_c = _col(rock_palette, 0.0), _col(rock_palette, 1.0)

    # dust puffs: mostly sideways along the ground + a rising core
    ND = 64 if dust_amount > 0 else 0
    side = rng.choice([-1, 1], ND)
    core = rng.random(ND) < 0.14
    vx = np.where(core, rng.uniform(-60, 60, ND), side * rng.uniform(100, 360, ND) * dust_spread) * s
    vy = np.where(core, -rng.uniform(120, 240, ND) * dust_rise, -rng.uniform(5, 80, ND)) * s
    r0 = rng.uniform(7, 13, ND) * s
    da = rng.uniform(0.3, 0.55, ND) * dust_amount
    dz = rng.uniform(-14, 14, ND) * s
    drag = rng.uniform(3.5, 5.0, ND)
    kid_a = -math.pi / 2 + rng.uniform(-1.6, 1.6, (ND, 3))     # sub-billows on the upper rim
    kid_k = rng.uniform(0.45, 0.65, (ND, 3))

    # debris
    NR = int(rocks)
    rk = [_rock_shape(rng, rng.uniform(3.5, 9.0) * s * rock_size) for _ in range(NR)]
    r_vx = rng.uniform(-230, 230, NR) * s * launch
    r_vy = -rng.uniform(230, 430, NR) * s * launch
    r_z = rng.uniform(-18, 18, NR) * s
    r_w = rng.uniform(-14, 14, NR)
    NP = int(pebbles)
    p_vx = rng.uniform(-260, 260, NP) * s * launch
    p_vy = -rng.uniform(150, 380, NP) * s * launch
    p_z = rng.uniform(-18, 18, NP) * s
    G = 1100 * s * max(gravity, 0.05)
    xlim = w / 2 - 30 * s

    def ballistic(x0, y0, vx_, vy_, zl, t):
        """Projectile that lands on the ground plane (y = gy + z*SQ), one bounce."""
        yl = gy + zl * SQ
        tl = (-vy_ + math.sqrt(vy_ * vy_ + 2 * G * max(yl - y0, 0))) / G
        if t < tl:
            x, y = x0 + vx_ * t, y0 + vy_ * t + 0.5 * G * t * t
        else:
            xl = x0 + vx_ * tl
            bvy = -0.25 * (vy_ + G * tl)
            tb = t - tl
            y = yl + bvy * tb + 0.5 * G * tb * tb
            if y > yl:
                x, y = xl + 0.3 * vx_ * min(tb, -2 * bvy / G), yl
            else:
                x = xl + 0.3 * vx_ * tb
        return float(np.clip(x, cx - xlim, cx + xlim)), max(y, 14 * s)

    rock_c = _col(rock_palette, 0.35)
    out = []
    for f in range(frames):
        t = (f + 0.5) / frames * T
        light = Canvas(w, h)
        dust = Canvas(w, h)
        under = Solid(w, h)
        over = Solid(w, h)
        endfade = 1 - smoothstep(0.62, 0.97, t / T)

        if flash:
            light.dot(cx, gy - 4 * s, (8 + 60 * t) * s, 2.4 * math.exp(-t / 0.045))
        Rw = (10 + 165 * (1 - math.exp(-t / 0.13))) * s
        if shockwave:
            light.ring(cx, gy, Rw, (1.2 + 4 * t) * s, 0.25 * math.exp(-t / 0.1), squash=SQ)
            sw = math.exp(-t / 0.3) * smoothstep(0, 0.03, t) * min(dust_amount, 1.5)
            if sw > 0.02:
                dust.ring(cx, gy - 3 * s, Rw * 0.97, (3 + 14 * t) * s, 0.55 * sw, squash=SQ * 1.15,
                          target="dens")

        # crater + cracks (decal)
        grow = 1 - math.exp(-t / 0.035)
        under.polygon(proj([(math.cos(a) * 20 * s, math.sin(a) * 20 * s)
                            for a in np.linspace(0, TAU, 24, endpoint=False)]),
                      _col(rock_palette, 0.1), 0.75 * endfade, soft=3 * s)
        for pts, wd, start in cracks:
            k = (grow - start) / (1 - start)
            if k <= 0:
                continue
            P = pts[:max(2, int(len(pts) * k))]
            under.stroke(P + (0, 0.9 * s), wd * 0.55, lip_c, 0.35 * endfade, r_end=0.3 * s)
            under.stroke(P, wd * 0.55, crack_c, 0.95 * endfade, r_end=0.3 * s)
            if flash:
                light.polyline(P, wd * 0.25, 0.9 * math.exp(-t / 0.08), r_end=0.3 * s, a_end=0.0)

        # dust
        for i in range(ND):
            k = (1 - math.exp(-drag[i] * t)) / drag[i]
            x = cx + vx[i] * k
            y = gy + dz[i] * SQ + vy[i] * k - (0 if core[i] else 18 * s * t)
            rad = r0[i] * (1 + 3.0 * (1 - math.exp(-t / 0.35)))
            y = min(y, gy + dz[i] * SQ - rad * 0.9)
            a = da[i] * smoothstep(0, 0.04, t) * (1 - smoothstep(0.35, 1.0, t / T)) / (1 + 1.5 * t)
            dust.dot(x, y, rad, a, target="dens")
            for j in range(3):
                dust.dot(x + math.cos(kid_a[i, j]) * rad * 0.9, y + math.sin(kid_a[i, j]) * rad * 0.9,
                         rad * kid_k[i, j], a * 0.8, target="dens")
        if dust.dens.any():
            E = 1 + 2.0 * t
            n = _noise_field(nz, w, h, (X - cx) / (24 * s * E), (Y - gy) / (20 * s * E), 2 + 2 * t)
            dust.dens *= (0.2 + 1.1 * smoothstep(-0.55, 0.55, n)) * keep

        # debris
        for i in range(NR):
            x, y = ballistic(cx + r_vx[i] * 0.05, gy - 6 * s, r_vx[i], r_vy[i], r_z[i], t)
            _draw_rock(over, rk[i], x, y - np.abs(rk[i]).max() * 0.4, r_w[i] * min(t, 0.6), endfade,
                       stops=rock_palette)
        for i in range(NP):
            x, y = ballistic(cx, gy - 4 * s, p_vx[i], p_vy[i], p_z[i], t)
            x2, y2 = ballistic(cx, gy - 4 * s, p_vx[i], p_vy[i], p_z[i], max(t - 0.025, 0))
            over.capsule((x2, y2), (x, y), 0.7 * s, 1.1 * s, rock_c, endfade)

        lay_light = light.render(flash_palette, bloom=((1.5 * s, 0.5), (6 * s, 0.4)), exposure=1.0)
        lay_dust = _shade_volume(dust.dens, w, h, palette, opacity=1.1, bump=40 * s,
                                 soft=1.5 * s, shadow=0.3, wrap=0.6) if ND else None
        out.append(_stack(lay_dust, under.rgba(), lay_light, over.rgba()))
    return out, 24


# ===========================================================================
# Water
# ===========================================================================
@effect("water_splash", "water", "major",
        "Side-view crown splash: rising water sheet breaks into droplets, central jet, spray", loop=False)
def fx_water_splash(rng, palette, scale=1.0, frames=26, crown_fingers=40, crown_height=1.0,
                    crown_width=1.0, drop_amount=1.0, spray_drops=26, jet=True, jet_height=1.0,
                    ripples=True, mist_amount=1.0, gravity=1.0, gloss=1.0, opacity=1.0,
                    duration=1.1, glint_palette="glint", mist_palette="steam"):
    """Water is a density field (crown = continuous sheet of quads + tapered
    fingers with rim beads, drops = stretched capsules) shaded by
    `_shade_volume` as a glossy, see-through body with fresnel rim and Blinn
    highlights (palette = water body ramp); glints in light; faint mist.

    Params:
      scale: output size multiplier (base 320x320)
      frames: number of frames
      crown_fingers: fingers around the crown rim (each sheds 1-2 drops)
      crown_height: crown height multiplier
      crown_width: crown radius multiplier
      drop_amount: multiplier on drops shed per finger (0 = none)
      spray_drops: fast small droplets thrown at the impact
      jet: central Worthington jet with a bead launched off its top
      jet_height: jet height multiplier
      ripples: surface ripple rings
      mist_amount: spray mist density (0 = off)
      gravity: gravity multiplier for drops
      gloss: specular highlight strength
      opacity: water body opacity multiplier
      duration: seconds covered by the animation
      glint_palette: colour of the light glints on drops
      mist_palette: spray mist colours (shadow -> lit)
    """
    size = int(320 * scale)
    s = scale
    cx, ys = size / 2, size * 0.74
    SQ = 0.3
    T = duration
    G = 1100 * s * max(gravity, 0.05)
    nz = Noise(rng, period=(16, 16, 8))
    X, Y = _out_grid(size, size)
    Hmax = min(92 * crown_height, 150) * s
    cw = min(crown_width, 1.6)

    NF = max(6, int(crown_fingers))
    phi = (np.arange(NF) + rng.uniform(-0.3, 0.3, NF)) / NF * TAU
    fh = rng.uniform(0.75, 1.15, NF)              # finger height jitter
    fd = rng.uniform(0.14, 0.26, NF)              # drop pinch-off time
    fr = rng.uniform(2.0, 3.6, NF) * s            # rim bead radius

    def crown(t):
        Rb = (12 + 46 * cw * (1 - math.exp(-t / 0.12))) * s
        H = Hmax * (1 - math.exp(-t / 0.06)) * (1 - smoothstep(0.22, 0.62, t)) ** 1.2
        return Rb, Rb * (1.25 + 0.5 * min(t / 0.3, 1)), H

    def tip(j, t):
        Rb, Rr, H = crown(t)
        return (cx + Rr * math.cos(phi[j]), ys + Rr * math.sin(phi[j]) * SQ - H * fh[j])

    drops = []
    for j in range(NF):
        for k in range(int(round(rng.integers(1, 3) * drop_amount))):
            td = fd[j] + k * rng.uniform(0.03, 0.08)
            x0, y0 = tip(j, td)
            out_v = rng.uniform(25, 70) * s
            drops.append(dict(t0=td, x0=x0, y0=y0, vx=math.cos(phi[j]) * out_v + rng.normal(0, 10) * s,
                              vy=-rng.uniform(120, 300) * s, r=fr[j] * (0.9 if k == 0 else 0.6),
                              yl=ys + math.sin(phi[j]) * 50 * s * SQ, front=math.sin(phi[j]) > 0))
    for _ in range(int(spray_drops)):
        a = rng.uniform(-math.pi * 0.95, -math.pi * 0.05)
        v = rng.uniform(180, 380) * s
        drops.append(dict(t0=0.0, x0=cx, y0=ys - 4 * s, vx=math.cos(a) * v * 0.45, vy=math.sin(a) * v,
                          r=rng.uniform(1.0, 2.0) * s, yl=ys + rng.uniform(-6, 10) * s, front=True))

    jt0, jt1 = 0.3, 0.85
    jH = min(rng.uniform(95, 115) * jet_height * s, ys - 50 * s)

    out = []
    for f in range(frames):
        t = (f + 1.0) / frames * T
        water = Canvas(size, size)
        mist = Canvas(size, size)
        light = Canvas(size, size)
        Rb, Rr, H = crown(t)
        surf = 1 - smoothstep(0.55, 1.0, t / T)

        water.ring(cx, ys, Rb * 1.1, (4 + 4 * t) * s, 0.9 * surf, squash=SQ, target="dens")
        if ripples:
            for k in range(2):
                rt = t - k * 0.15
                if rt > 0:
                    Rr_ = (20 + 140 * (1 - math.exp(-rt / 0.35))) * s
                    water.ring(cx, ys, Rr_, (1.5 + 3 * rt) * s, 0.45 * math.exp(-rt / 0.45) * surf,
                               squash=SQ, target="dens")

        # crown: continuous sheet (quads between neighbouring fingers) up to ~70%
        # height, tapered fingers + rim beads above it; back half first
        if H > 1 * s:
            hk_ = smoothstep(0.1, 0.45, H / Hmax)          # sheet thins as it collapses
            order = np.argsort(np.sin(phi))
            for j in order:
                dens = 0.55 if math.sin(phi[j]) > 0 else 0.3
                j2 = (j + 1) % NF
                quad = []
                for jj, hk in ((j, 0.0), (j2, 0.0), (j2, 0.68), (j, 0.68)):
                    tx, ty = tip(jj, t)
                    bx = cx + Rb * math.cos(phi[jj])
                    by = ys + Rb * math.sin(phi[jj]) * SQ
                    quad.append((bx + (tx - bx) * hk, by + (ty - by) * hk))
                water.polygon(quad, dens * (0.4 + 0.6 * hk_), soft=1.0, target="dens")
            for j in order:
                front = math.sin(phi[j])
                bx = cx + Rb * math.cos(phi[j])
                by = ys + Rb * math.sin(phi[j]) * SQ
                tx, ty = tip(j, t)
                mx, my = bx + (tx - bx) * 0.45, by + (ty - by) * 0.45
                wt = max(fr[j] * (1 - smoothstep(fd[j] - 0.05, fd[j], t)) * 0.8, 0.8 * s)
                capsule(water, (mx, my), (tx, ty), TAU * Rb / NF * 0.45 + 1.0 * s, wt * 0.7,
                        (0.6 if front > 0 else 0.35) * hk_, soft=0.8, target="dens")
                if t < fd[j]:
                    water.dot(tx, ty, fr[j] * 0.6, 1.2, target="dens")

        if jet and jt0 < t < jt1:
            u = (t - jt0) / (jt1 - jt0)
            hj = jH * math.sin(math.pi * min(u * 1.3, 1)) ** 0.8
            wj = 6 * s * (1 - 0.4 * u)
            if hj > 2 * s:
                capsule(water, (cx, ys + 2 * s), (cx, ys - hj), wj, wj * 0.55, 0.9, soft=0.8,
                        target="dens")
            if u > 0.35:
                tb = (u - 0.35) * (jt1 - jt0)
                v0 = 230 * s * math.sqrt(max(jet_height, 0.1))
                yb = ys - jH * 0.9 - v0 * tb + 0.5 * G * tb * tb
                if yb < ys:
                    capsule(water, (cx, yb + 3 * s), (cx, yb), 4.5 * s, 4 * s, 1.4, target="dens")

        for d in drops:
            tt = t - d["t0"]
            if tt <= 0:
                continue
            x = d["x0"] + d["vx"] * tt
            y = d["y0"] + d["vy"] * tt + 0.5 * G * tt * tt
            vy = d["vy"] + G * tt
            if y > d["yl"] or y < 12 * s:
                continue
            sp = math.hypot(d["vx"], vy) + 1e-6
            st = min(sp * 0.018, 14 * s)
            px, py = x - d["vx"] / sp * st, y - vy / sp * st
            capsule(water, (px, py), (x, y), d["r"] * 0.45, d["r"], 1.3 if d["front"] else 0.8,
                    soft=0.6, target="dens")
            light.dot(x - d["r"] * 0.35, y - d["r"] * 0.4, d["r"] * 0.32, 1.3 * gloss)

        mf = mist_amount * smoothstep(0.02, 0.15, t) * (1 - smoothstep(0.35, 0.95, t / T))
        if mf > 0:
            for k in range(14):
                a = k / 14 * TAU
                mist.dot(cx + math.cos(a) * Rr * 1.1, ys + math.sin(a) * Rr * SQ - H * 0.8 - 12 * s * t,
                         (10 + 22 * t) * s, 0.08 * mf, target="dens")
            n = _noise_field(nz, size, size, (X - cx) / (22 * s), (Y - ys) / (22 * s) + t * 2, t * 2)
            mist.dens *= 0.3 + 1.0 * smoothstep(-0.5, 0.6, n)

        lay_water = _shade_volume(water.dens, size, size, palette, opacity=1.0 * opacity,
                                  bump=6 * s, soft=1.0 * s, shadow=0.15, wrap=0.8,
                                  spec=1.0 * gloss, spec_pow=30, rim=0.8)
        lay_mist = mist.render(glint_palette, smoke=mist_palette, smoke_opacity=1.0, shadow=0.0)
        lay_light = light.render(glint_palette, bloom=((1.0 * s, 0.4), (3 * s, 0.25)), exposure=1.2)
        out.append(_stack(lay_mist, lay_water, lay_light))
    return out, 24
