"""Fire / smoke effects."""
import math

import numpy as np

from .core import SS, TAU, Canvas, Noise, blur, effect, grid, ramp, refine, shift, smoothstep, PALETTES  # noqa: F401
from .core import _edge_mask

_grid = grid

# warm-tinted dark smoke for explosions (index = lighting)
PALETTES.setdefault("explosion_smoke", [(0.0, (30, 26, 26)), (0.5, (74, 67, 64)), (1.0, (150, 140, 130))])


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
def _coords(w, h):
    """Output-resolution pixel-centre coordinates (noise fields are evaluated
    here, then upsampled into the supersampled canvas)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs + 0.5, ys + 0.5


def _up(a):
    """Output-res field -> supersampled canvas resolution (exact under the
    canvas' box downsample)."""
    return np.repeat(np.repeat(a, SS, axis=0), SS, axis=1)


def _loop_t(f, frames):
    return f / frames


class _Tex:
    """Tileable 2D fbm texture with bilinear lookup. Cheap noise for many
    particles (smoke puffs): sample it in each puff's local frame so the
    detail travels and grows with the puff."""

    def __init__(self, rng, n=256, cells=8, octaves=5, gain=0.5):
        nz = Noise(rng, period=(cells, cells, 2))
        ys, xs = np.mgrid[0:n, 0:n].astype(np.float32) * (cells / n)
        t = nz.fbm(xs, ys, 0.0, octaves=octaves, gain=gain)
        t = (t - t.mean()) / (t.std() + 1e-6)
        self.t = t.astype(np.float32)
        self.n = n

    def __call__(self, u, v):
        n = self.n
        u = np.asarray(u, np.float32) * n
        v = np.asarray(v, np.float32) * n
        i0, j0 = np.floor(u), np.floor(v)
        fu, fv = u - i0, v - j0
        i0 = i0.astype(np.int64) % n
        j0 = j0.astype(np.int64) % n
        i1, j1 = (i0 + 1) % n, (j0 + 1) % n
        T = self.t
        a = T[j0, i0] * (1 - fu) + T[j0, i1] * fu
        b = T[j1, i0] * (1 - fu) + T[j1, i1] * fu
        return a * (1 - fv) + b * fv


def _puff_field(c, tex, x, y, r, ox=0.0, oy=0.0, rot=0.0, detail=0.55):
    """Soft ball + noise over its bounding box. Returns (slice, body, n):
    gaussian body and unit-variance texture `n` sampled in the puff's own
    rotated/scaled frame, so billow detail travels and grows with the puff.
    Combine as body * (1 + amp * n) - threshold. None if off-canvas."""
    R = 2.4 * r
    x0, x1, y0, y1 = c._bbox(x - R, x + R, y - R, y + R)
    if x0 >= x1 or y0 >= y1:
        return None
    dx = (c.X[y0:y1, x0:x1] - x) / r
    dy = (c.Y[y0:y1, x0:x1] - y) / r
    cr, sr = math.cos(rot), math.sin(rot)
    u = cr * dx - sr * dy
    v = sr * dx + cr * dy
    body = np.exp(-(dx * dx + dy * dy) * 0.75)
    n = tex(u * detail * 0.25 + ox, v * detail * 0.25 + oy)
    return (slice(y0, y1), slice(x0, x1)), body, n


def _puff(c, tex, x, y, r, a, ox=0.0, oy=0.0, rot=0.0, detail=0.55, amp=0.55,
          thr=0.18, target="dens", crisp=0.0):
    """Paint a noise-eroded puff (see _puff_field). `thr` = erosion threshold:
    raise it to break the puff into wisps as it dissipates. `crisp` > 0 gives
    a defined cauliflower edge (smoothstep width) instead of a linear ramp."""
    if a <= 0:
        return
    res = _puff_field(c, tex, x, y, r, ox, oy, rot, detail)
    if res is None:
        return
    sl, body, n = res
    buf = c.dens if target == "dens" else c.buf
    val = body * (1 + amp * n) - thr
    if crisp > 0:
        buf[sl] += a * (smoothstep(0, crisp, val) * 0.6 + np.clip(val, 0, None))
    else:
        buf[sl] += a * np.clip(val, 0, None)


def _edge_fade(x, y, w, h, s, margin=26):
    """1 well inside the frame, 0 within ~20px of the border (for particles)."""
    d = min(x, y, w - x, h - y)
    return float(smoothstep(20 * s, (20 + margin) * s, d))


def _render_shaded(c, palette, smoke_palette, opacity=1.0, light_dir=(-0.55, -0.85), relief=1.0,
                   shadow=1.0, ambient=0.18, bloom=(), exposure=1.0, alpha_gamma=0.85, height_gain=0.8):
    """Like Canvas.render, but smoke gets pseudo-normal lighting: a multi-
    scale blurred density acts as a height field, so every billow is lit on
    its light-facing side and shaded on the other (plus the core's soft
    self-shadow). Emitted light (`buf`) is composited OVER the smoke."""
    w, h = c.w, c.h
    size = max(w, h)
    D = np.maximum(c.dens.reshape(h, SS, w, SS).mean(axis=(1, 3)), 0) * _edge_mask(w, h)
    a = 1.0 - np.exp(-D * opacity)
    Hh = 0.0
    for sig, wt in ((size / 90, 0.2), (size / 45, 0.4), (size / 20, 0.4)):
        Hh = Hh + wt * (1 - np.exp(-blur(D, sig) * opacity * height_gain))
    gy, gx = np.gradient(Hh)
    k = relief * size / 7
    nx, ny, nz = -gx * k, -gy * k, np.ones_like(Hh)
    nl = np.sqrt(nx * nx + ny * ny + 1)
    lx, ly = light_dir
    lz = 0.75
    ll = math.sqrt(lx * lx + ly * ly + lz * lz)
    lam = np.clip((nx * lx + ny * ly + nz * lz) / (nl * ll), 0, 1)
    Db = blur(D, max(1.5, size / 90))
    off = size / 40
    occl = shift(Db, lx * off, ly * off) + 0.5 * shift(Db, lx * off * 2.2, ly * off * 2.2)
    sh = np.exp(-shadow * occl * opacity * 0.5)
    lit = np.clip(ambient + (1 - ambient) * lam ** 1.6 * (0.25 + 0.75 * sh), 0, 1)
    s_rgb = ramp(smoke_palette, lit)
    if c.buf.any():
        L = c.render(palette, bloom=bloom, exposure=exposure, alpha_gamma=alpha_gamma).astype(np.float32)
        e_a = L[..., 3] / 255
        e_rgb = L[..., :3]
    else:
        e_a = np.zeros_like(a)
        e_rgb = np.zeros_like(s_rgb)
    out_a = e_a + a * (1 - e_a)
    rgb = (e_rgb * e_a[..., None] + s_rgb * (a * (1 - e_a))[..., None]) / np.maximum(out_a, 1e-6)[..., None]
    out_a[out_a < 1.5 / 255] = 0
    rgb[out_a == 0] = 0
    return np.clip(np.dstack([rgb, out_a * 255]) + 0.5, 0, 255).astype(np.uint8)


def _flame_field(N, xs, ys, cx, yb, half_w, height, t, warp=None, scroll_cells=8, cell_y=48.0,
                 cell_x=34.0, zcells=2, top=0.9, noise_amp=(0.22, 0.75), octaves=5):
    """Scalar flame 'fuel' field (>0 inside flame). Upward-scrolling fbm,
    domain-warped sideways, added to a teardrop envelope that tapers with
    height so noise peaks break off into licking tongues and wisps.
    The scroll covers `scroll_cells` noise cells per loop (= y period)."""
    v = (yb - ys) / height                      # 0 at base, 1 at nominal tip
    sway = 0.0
    if warp is not None:
        sway = warp.fbm(xs / (cell_x * 3), ys / (cell_y * 3) + t * warp.py, t * warp.pz, octaves=2)
    u = (xs - cx) / half_w + sway * 0.55 * np.clip(v, 0, None) ** 1.2
    n = N.fbm(xs / cell_x, ys / cell_y + t * scroll_cells, t * zcells, octaves=octaves)
    vp = np.clip(v, 0, None)
    wv = np.clip(1.0 - 0.75 * vp, 0.08, None) ** 0.9 * (0.9 + 0.1 * smoothstep(-0.06, 0.16, v))
    shape = 1.0 - (np.abs(u) / wv) ** 1.8
    shape -= top * vp ** 1.1
    # rounded bottom (bowl-shaped, so the base corners lift instead of a flat cut)
    vb = v - 0.05 * u * u
    shape -= np.clip(-vb, 0, None) ** 2 * 50
    a0, a1 = noise_amp
    amp = a0 + (a1 - a0) * np.clip(v, 0, 1.2)
    return shape + amp * n, v


# ---------------------------------------------------------------------------
# fire (bonfire)
# ---------------------------------------------------------------------------
@effect("fire", "fire", "major", "Bonfire: rising noise flame with licking tongues, wisps and embers", loop=True)
def fx_fire(rng, palette, scale=1.0, frames=24, height=1.0, width=1.0, intensity=1.0, turbulence=1.0,
            speed=1, sway=1.0, core=1.0, embers=26, glow=1.0):
    """Bonfire flame. Domain-warped fbm scrolls upward through a teardrop
    envelope; noise peaks above the envelope tip detach into wisps. Hot
    white-yellow core where the field is deep, red transparent edges.
    Integer scroll / time cells per loop => seamless.

    Params:
      height: flame height multiplier (keep <= ~1.2 to stay in frame)
      width: flame base width multiplier
      intensity: overall brightness (higher = whiter / hotter)
      turbulence: noise strength: more tongues, wisps and break-up
      speed: rise speed in noise periods per loop (integer keeps the loop seamless)
      sway: sideways lick / domain-warp amount
      core: extra heat of the white-hot core near the base
      embers: number of rising embers (0 = off)
      glow: bloom / halo strength
    """
    w, h = int(256 * scale), int(384 * scale)
    s = w / 256
    cx, yb = w / 2, h - 64 * s
    speed = max(1, int(round(speed)))
    xs, ys = _coords(w, h)
    N = Noise(rng, period=(64, 8, 2))
    W = Noise(rng, period=(64, 4, 1))
    D = Noise(rng, period=(64, 16, 3))     # fast vertical streak detail

    NE = int(embers)
    E = dict(x0=rng.normal(0, 26, NE) * s * width, ph=rng.random(NE), k=rng.choice([1, 1, 2], NE) * speed,
             sw=rng.uniform(6, 20, NE) * s, sk=rng.choice([1, 2, 3], NE), sph=rng.uniform(0, TAU, NE),
             top=rng.uniform(0.55, 0.95, NE), a=rng.uniform(0.8, 1.8, NE),
             tw=rng.choice([2, 3, 4], NE), drift=rng.normal(0, 18, NE) * s)

    def ember(i, tt):
        hh = (E["ph"][i] + E["k"][i] * tt) % 1.0
        y = yb - 40 * s - hh * (yb - 40 * s - 34 * s) * E["top"][i]
        x = cx + E["x0"][i] + E["drift"][i] * hh + E["sw"][i] * hh * math.sin(TAU * E["sk"][i] * tt + E["sph"][i])
        return x, y, hh

    class _Warp:                           # scale the sway warp without touching the noise
        py, pz = W.py, W.pz

        @staticmethod
        def fbm(*a, **k):
            return W.fbm(*a, **k) * sway

    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(w, h)
        F, v = _flame_field(N, xs, ys, cx, yb, 68 * s * width, 240 * s * height, t, warp=_Warp,
                            cell_x=15 * s * width ** 0.5, cell_y=46 * s * height ** 0.5,
                            scroll_cells=8 * speed, zcells=2,
                            noise_amp=(0.5 * turbulence, 1.45 * turbulence), top=1.15)
        m = smoothstep(0.0, 0.1, F)
        d = D.fbm(xs / (10 * s), ys / (30 * s) + t * 16 * speed, t * 3, octaves=3)
        heat = 1.5 * np.exp(-1.6 * np.clip(v, 0, None))
        I = m * (0.3 + 1.2 * np.clip(F, 0, 1.6) ** 1.3) * heat * np.clip(0.75 + 0.7 * d, 0.15, None)
        I *= 0.55 + 0.45 * np.exp(-((xs - cx) / (46 * s * width)) ** 2)      # cooler flanks
        hot = np.exp(-((xs - cx) / (30 * s * width)) ** 2) * np.exp(-((ys - (yb - 30 * s)) / (40 * s * height)) ** 2)
        I *= (1 + 0.9 * core * hot) * smoothstep(14 * s, 70 * s, ys) * intensity
        c.buf += _up(I.astype(np.float32))
        c.dot(cx, yb - 16 * s, 40 * s * width, 0.08 * glow * intensity)      # warm base glow
        for i in range(NE):
            x, y, hh = ember(i, t)
            px, py, ph = ember(i, t - 0.012 * E["k"][i])
            fade = math.sin(math.pi * hh) ** 0.8 * _edge_fade(x, y, w, h, s, 20)
            tw = 0.6 + 0.4 * math.sin(TAU * E["tw"][i] * t + E["sph"][i] * 3)
            amp = E["a"][i] * fade * tw
            if ph < hh:   # skip the wrap-around frame's tail
                c.polyline([(px, py), (x, y)], 0.5 * s, amp * 0.3, r_end=0.9 * s, a_end=amp)
        out.append(c.render(palette, bloom=((1.5 * s, 0.2 * glow), (6 * s, 0.18 * glow), (20 * s, 0.14 * glow)),
                            exposure=0.9, alpha_gamma=0.65))
    return out, 24


# ---------------------------------------------------------------------------
# explosion
# ---------------------------------------------------------------------------
@effect("explosion", "fire", "major", "Fireball explosion: flash, shockwave, debris, cooling into billowing smoke",
        loop=False)
def fx_explosion(rng, palette, scale=1.0, frames=24, radius=1.0, intensity=1.0, burn_time=1.0,
                 smoke_amount=1.0, smoke_palette="explosion_smoke", puffs=22, turbulence=1.0, rise=1.0,
                 sparks=70, debris=8, shockwave=True, flash=True, fire_light=0.8):
    """Showpiece explosion. A cluster of noise-eroded puffs expands with a
    decelerating radius; each puff carries its own heat that decays (inner
    puffs stay hot longest), so the fireball cools from white to orange to
    red pockets inside dark smoke that keeps billowing up and erodes away.
    Flash + shockwave ring up front, ballistic sparks and smoke-trailing
    burning debris on top.

    Params:
      radius: fireball / smoke cloud size multiplier (keep <= ~1.15 to stay in frame)
      intensity: fire brightness multiplier
      burn_time: how long the fire keeps burning inside the smoke (multiplier)
      smoke_amount: smoke density multiplier (0 = pure fireball, no smoke)
      smoke_palette: colour ramp of the smoke (shadow -> lit), preset or hex list
      puffs: number of billow puffs forming the cloud
      turbulence: billow / break-up noise strength
      rise: how far the cloud drifts upward
      sparks: number of flying spark streaks (0 = off)
      debris: number of burning debris chunks with smoke trails (0 = off)
      shockwave: draw the expanding shockwave ring
      flash: draw the initial white flash
      fire_light: how much the fire tints the surrounding smoke
    """
    size = int(384 * scale)
    s = size / 384
    cx, cy0 = size / 2, size * 0.62
    T = 1.5                         # seconds covered by the animation
    RMAX = 84 * s * radius
    tex = _Tex(rng, n=256, cells=8, octaves=5)

    def R_of(t):                    # fireball radius: fast blast, slow creep
        return RMAX * (1 - math.exp(-t / 0.09)) + 30 * s * radius * t

    def cy_of(t):                   # buoyant rise of the whole cloud
        return cy0 - 36 * s * rise * t ** 1.4

    NP = max(1, int(puffs))
    P = dict(th=rng.uniform(0, TAU, NP), d=np.sqrt(rng.uniform(0.0, 1.0, NP)) * 0.72,
             r=rng.uniform(0.3, 0.58, NP), rise=rng.uniform(0, 36, NP) * s * rise,
             spin=rng.normal(0, 1.2, NP), rot=rng.uniform(0, TAU, NP),
             ox=rng.random(NP), oy=rng.random(NP))
    P["d"][0], P["r"][0] = 0.0, 0.72                  # core puff
    P["tau"] = (0.10 + 0.24 * (1 - P["d"] / 0.72) + rng.uniform(-0.03, 0.05, NP)) * burn_time

    NS = int(sparks)                # spark streaks
    S = dict(ang=rng.uniform(0, TAU, NS), spd=rng.uniform(200, 380, NS) * s,
             life=rng.uniform(0.2, 0.6, NS), a=rng.uniform(1.2, 2.6, NS),
             wid=rng.uniform(0.6, 1.1, NS) * s)
    ND = int(debris)                # burning debris with smoke trails
    Dd = dict(ang=-math.pi / 2 + rng.uniform(-1.4, 1.4, ND), spd=rng.uniform(260, 500, ND) * s,
              life=rng.uniform(0.55, 1.15, ND), t0=rng.uniform(0, 0.16, ND))
    drag, grav = 2.2, 260 * s

    def ballistic(x0, y0, ang, spd, t, drag=drag):
        k = (1 - math.exp(-drag * t)) / drag
        return x0 + math.cos(ang) * spd * k, y0 + math.sin(ang) * spd * k + 0.5 * grav * t * t

    X, Y = _grid(size, size)
    rr = np.hypot(X - cx, Y - cy0)
    ring_mod = 0.7 + 0.3 * np.clip(tex(np.arctan2(Y - cy0, X - cx) / TAU * 3, 0.37), -1, 1)

    out = []
    for f in range(frames):
        u = f / (frames - 1)
        t = T * (0.02 + 0.98 * u ** 1.35)          # dense early frames for the blast
        c = Canvas(size, size)
        R = R_of(t)
        cy = cy_of(t)
        fade = 1 - smoothstep(0.45, 1.0, u)        # global smoke dissipation
        erode = 0.18 + 0.75 * smoothstep(0.35, 1.0, u)

        if flash:
            c.dot(cx, cy0, 40 * s * radius, 4.0 * intensity * math.exp(-t / 0.035))
            c.dot(cx, cy0, 70 * s * radius, 0.5 * intensity * math.exp(-t / 0.05))

        if shockwave and t < 0.35:
            Rs = (24 + 140 * (1 - math.exp(-t / 0.1))) * s
            ra = 0.7 * math.exp(-t / 0.07)
            dr = (rr - Rs) / ((2.0 + 9 * t) * s)
            c.buf += ra * ring_mod * np.exp(-dr * dr)

        for i in range(NP):
            th = P["th"][i]
            dist = P["d"][i] * R
            x = cx + math.cos(th) * dist
            y = cy + math.sin(th) * dist * 0.85 - P["rise"][i] * t * (1 + 0.5 * math.sin(th) * -1)
            r = P["r"][i] * R * (1 + 0.25 * t)
            rot = P["rot"][i] + P["spin"][i] * t
            heat = math.exp(-t / P["tau"][i])
            res = _puff_field(c, tex, x, y, r, P["ox"][i], P["oy"][i], rot, detail=0.6)
            if res is None:
                continue
            sl, body, n = res
            fld = body * (1 + 0.8 * turbulence * n)
            # smoke body: always there (dark fringe early), eroded away at the end
            sa = (0.5 + 0.9 * smoothstep(0.1, 0.8, 1 - heat)) * fade * smoke_amount
            c.dens[sl] += sa * np.clip(fld - erode, 0, None) * 1.6
            # fire: same puff, eroded harder as it cools -> shrinks into hot pockets
            fa = 3.4 * heat ** 1.4 * intensity
            if fa > 0.06:
                hot = body * (1 + 1.1 * turbulence * n) - (0.2 + 0.95 * (1 - heat))
                c.buf[sl] += fa * np.clip(hot, 0, None) ** 0.9

        for i in range(NS):
            if t >= S["life"][i]:
                continue
            tail = max(t - 0.05, 0.0)
            pts = [ballistic(cx, cy0, S["ang"][i], S["spd"][i], tt) for tt in (tail, (tail + t) / 2, t)]
            fd = (1 - t / S["life"][i]) ** 1.3 * _edge_fade(*pts[-1], size, size, s)
            c.polyline(pts, S["wid"][i] * 0.4, S["a"][i] * fd * 0.1, r_end=S["wid"][i], a_end=S["a"][i] * fd)

        for i in range(ND):
            tt = min(t, Dd["life"][i]) - Dd["t0"][i]
            if tt <= 0:
                continue
            hist = np.linspace(max(0.0, tt - 0.6), tt, 12)
            pts = [ballistic(cx, cy0, Dd["ang"][i], Dd["spd"][i], q, drag=0.9) for q in hist]
            df = (1 - smoothstep(0.7, 1.0, (tt + Dd["t0"][i]) / Dd["life"][i])) * _edge_fade(*pts[-1], size, size, s, 40)
            c.polyline(pts, 6.0 * s, 0.35 * df * fade * smoke_amount, r_end=2.0 * s,
                       a_end=1.3 * df * fade * smoke_amount, target="dens")
            if t < Dd["life"][i]:
                hx, hy = pts[-1]
                c.dot(hx, hy, 2.6 * s, 2.4 * df)
                c.polyline(pts[-5:], 0.8 * s, 0.0, r_end=2.2 * s, a_end=1.2 * df)

        out.append(c.render(palette, bloom=((2 * s, 0.25), (8 * s, 0.2), (24 * s, 0.15)),
                            exposure=1.0, alpha_gamma=0.75, smoke=smoke_palette, smoke_opacity=1.2,
                            shadow=1.0, fire_light=fire_light))
    return out, 18


# ---------------------------------------------------------------------------
# fireball (projectile)
# ---------------------------------------------------------------------------
@effect("fireball", "fire", "major", "Flying fireball: turbulent hot head, flame trail breaking into wisps and embers",
        loop=True)
def fx_fireball(rng, palette, scale=1.0, frames=20, head_size=1.0, trail_length=1.0, trail_width=1.0,
                turbulence=1.0, speed=1, intensity=1.0, embers=40, glow=1.0):
    """Projectile flying right. One flame field in head-relative coords: a
    round hot head plus a widening tail, both modulated by fbm that scrolls
    left (integer noise cells per loop) so the trail streams backwards and
    tears into detached wisps. Embers shed from the head drift back.

    Params:
      head_size: radius multiplier of the hot head
      trail_length: flame trail length multiplier
      trail_width: flame trail thickness multiplier
      turbulence: noise strength: wobble, tongues and wisp break-up
      speed: trail streaming speed in noise periods per loop (integer keeps the loop seamless)
      intensity: brightness multiplier
      embers: number of embers shed from the head (0 = off)
      glow: bloom / halo strength
    """
    w, h = int(384 * scale), int(256 * scale)
    s = h / 256
    hx, hy = w * 0.70, h * 0.5
    RH = 26 * s * head_size         # head radius
    L = 210 * s * trail_length      # trail length scale
    speed = max(1, int(round(speed)))
    xs, ys = _coords(w, h)
    CELL = 28 * s
    PX = 12                         # noise cells scrolled per loop (x period)
    N = Noise(rng, period=(PX, 64, 3))
    Wp = Noise(rng, period=(PX // 2, 64, 2))
    dx = hx - xs                    # distance behind the head (>0 = trail side)
    dy = ys - hy

    NE = int(embers)
    E = dict(ph=rng.random(NE), k=rng.choice([1, 2], NE) * speed, dy=rng.normal(0, 0.45, NE),
             spread=rng.uniform(10, 40, NE) * s, len=rng.uniform(0.55, 1.0, NE),
             a=rng.uniform(0.8, 1.8, NE), wob=rng.uniform(2, 8, NE) * s, wk=rng.choice([1, 2, 3], NE),
             wph=rng.uniform(0, TAU, NE))

    def ember(i, tt):
        q = (E["ph"][i] + E["k"][i] * tt) % 1.0
        x = hx - 10 * s - q * E["len"][i] * (hx - 40 * s)
        y = hy + E["dy"][i] * RH + q * E["spread"][i] * np.sign(E["dy"][i]) \
            + E["wob"][i] * q * math.sin(TAU * E["wk"][i] * tt + E["wph"][i])
        return x, y, q

    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(w, h)
        warp = Wp.fbm(xs / (CELL * 2) + t * (PX // 2) * speed, ys / (CELL * 2), t * 2, octaves=2)
        dyw = dy + warp * 38 * s * turbulence * np.clip(dx / L, 0, 1.5) ** 1.2
        n = N.fbm(xs / (CELL * 1.5) + t * PX * speed, ys / (CELL * 0.55), t * 3, octaves=5)
        a = np.clip(dx, 0, None) / L                       # 0 at head, 1 at tail
        width = RH * trail_width * (1.0 + 0.9 * a) * np.clip(1 - 0.5 * a, 0.2, None)
        trail = 1.0 - np.abs(dyw / width) ** 1.7 - 0.75 * a ** 1.3
        # head: round front
        dfront = np.hypot(np.clip(-dx, 0, None), dy) / RH
        trail = np.where(dx < 0, 1.0 - dfront ** 2, trail)
        amp = (0.55 + 0.8 * np.clip(a, 0, 1.3)) * turbulence
        F = trail + amp * n
        m = smoothstep(0.0, 0.12, F)
        heat = 1.7 * np.exp(-3.0 * a) + 0.12
        I = m * (0.25 + 1.0 * np.clip(F, 0, 1.6) ** 1.6) * heat
        I *= smoothstep(14 * s, 60 * s, xs) * intensity      # never reach the left edge
        c.buf += _up(I.astype(np.float32))
        # white-hot core
        c.dot(hx + 2 * s, hy, RH * 0.45, 1.4 * intensity)
        c.dot(hx, hy, RH * 1.6, 0.25 * intensity * glow)
        for i in range(NE):
            x, y, q = ember(i, t)
            px, py, pq = ember(i, t - 0.015 * E["k"][i])
            if pq > q:
                continue
            fd = math.sin(math.pi * q) ** 0.7 * (1 - q) ** 0.5
            amp_e = E["a"][i] * fd * _edge_fade(x, y, w, h, s, 30)
            c.polyline([(px, py), (x, y)], 0.45 * s, amp_e * 0.25, r_end=0.9 * s, a_end=amp_e)
        out.append(c.render(palette, bloom=((1.5 * s, 0.25 * glow), (6 * s, 0.25 * glow), (18 * s, 0.2 * glow)),
                            exposure=0.95, alpha_gamma=0.6))
    return out, 20


# ---------------------------------------------------------------------------
# smoke column
# ---------------------------------------------------------------------------
def _rising_puffs(c, tex, P, t, i_range, path, a_scale, target="dens", crisp=0.0, amp=0.75):
    """Draw looping puffs. Each puff's age q = (phase + t) % 1 drives its
    position/size/opacity via `path(i, q, t) -> (x, y, r, a, thr)`. Opacity
    must be 0 at q = 0 and q = 1 so the wrap is invisible."""
    for i in i_range:
        q = (P["ph"][i] + t) % 1.0
        x, y, r, a, thr = path(i, q, t)
        if a <= 1e-3:
            continue
        _puff(c, tex, x, y, r, a * a_scale, P["ox"][i] + 0.25 * q, P["oy"][i] - 0.35 * q,
              P["rot"][i] + P["spin"][i] * q, detail=0.6, amp=amp, thr=thr, target=target, crisp=crisp)


@effect("smoke", "smoke", "major", "Rising column of billowing, self-shadowed grey smoke", loop=True)
def fx_smoke(rng, palette, scale=1.0, frames=32, puffs=34, height=1.0, width=1.0, density=1.0, speed=1,
             sway=1.0, wispiness=1.0, relief=1.0, light_angle=-122.0):
    """Billowing smoke column. Looping noise-eroded puffs are born small at
    the base, rise, swell, drift and erode into wisps; pseudo-normal
    shading plus self-shadow gives them volume. Every puff's age wraps once
    per loop and its opacity is zero at birth/death => seamless. `palette`
    is the smoke ramp (shadow -> lit).

    Params:
      puffs: number of puffs in the column
      height: column height multiplier (keep <= ~1.15 to stay in frame)
      width: column width / puff size multiplier
      density: smoke opacity multiplier
      speed: rise laps per loop (integer keeps the loop seamless)
      sway: sideways drift / sway amount
      wispiness: how strongly puffs erode into wisps as they rise
      relief: strength of the billow lighting (0 = flat)
      light_angle: direction toward the light in degrees (screen, -90 = straight up)
    """
    w, h = int(256 * scale), int(384 * scale)
    s = w / 256
    cx, yb = w / 2, h - 52 * s
    ytop = yb - (yb - 80 * s) * height
    speed = max(1, int(round(speed)))
    la = math.radians(light_angle)
    tex = _Tex(rng, n=256, cells=8, octaves=5)
    NP = max(1, int(puffs))
    P = dict(ph=(np.arange(NP) + rng.uniform(0, 0.8, NP)) / NP, ox=rng.random(NP), oy=rng.random(NP),
             rot=rng.uniform(0, TAU, NP), spin=rng.normal(0, 1.0, NP),
             side=rng.normal(0, 1, NP), r0=rng.uniform(0.8, 1.2, NP), sw=rng.uniform(0, TAU, NP))

    def path(i, q, t):
        y = yb - (yb - ytop) * q ** 0.9
        sw = 14 * s * sway * q * math.sin(TAU * t + P["sw"][i] + 2.5 * q)
        x = cx + P["side"][i] * (5 + 20 * q) * s * width + sw + 10 * s * sway * q * q
        r = (14 + 40 * q) * s * P["r0"][i] * width
        a = smoothstep(0.0, 0.12, q) * (1 - smoothstep(0.5, 1.0, q))
        thr = 0.15 + 0.45 * wispiness * q ** 1.5
        return x, y, r, a, thr

    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(w, h)
        _rising_puffs(c, tex, P, (t * speed) % 1.0, range(NP), path, 0.2 * 34 / NP, amp=0.85)
        out.append(_render_shaded(c, palette, palette, opacity=5.0 * density, shadow=1.0, relief=relief,
                                  light_dir=(math.cos(la), math.sin(la))))
    return out, 16


# ---------------------------------------------------------------------------
# poison cloud
# ---------------------------------------------------------------------------
@effect("poison_cloud", "toxic", "major", "Toxic green gas cloud churning in place, with glowing motes and bubbles",
        loop=True)
def fx_poison_cloud(rng, palette, scale=1.0, frames=32, gas_palette="poison_gas", puffs=30, spread=1.0,
                    density=1.0, glow=0.35, motes=26, bubbles=6, speed=1, churn=1.0):
    """Low, wide toxic cloud. Looping noise-eroded gas puffs well up from the
    middle, roll outward and dissolve (pseudo-normal shading keeps them
    volumetric); a faint inner glow, rising motes and popping bubbles use the
    emissive `palette`.

    Params:
      gas_palette: colour ramp of the gas (shadow -> lit), preset or hex list
      puffs: number of gas puffs alive at once
      spread: cloud width/height multiplier
      density: gas opacity multiplier
      glow: strength of the faint emissive glow inside the gas (0 = off)
      motes: number of rising glowing motes
      bubbles: number of popping bubbles
      speed: loop cycles per animation (integer keeps the loop seamless)
      churn: how much puffs spin and roll
    """
    w, h = int(320 * scale), int(256 * scale)
    s = h / 256
    cx, cy = w / 2, h * 0.58
    speed = max(1, int(round(speed)))
    tex = _Tex(rng, n=256, cells=8, octaves=5)
    NP = int(puffs)
    ang = rng.uniform(0, TAU, NP)
    P = dict(ph=(np.arange(NP) + rng.uniform(0, 0.9, NP)) / max(NP, 1), ox=rng.random(NP), oy=rng.random(NP),
             rot=rng.uniform(0, TAU, NP), spin=rng.normal(0, 1.4, NP) * churn,
             dx=np.cos(ang), dy=np.sin(ang), d0=rng.uniform(0, 30, NP) * s, r0=rng.uniform(0.8, 1.25, NP))

    def path(i, q, t):
        dist = P["d0"][i] + 46 * s * spread * q
        x = cx + P["dx"][i] * dist * 1.3
        y = cy + P["dy"][i] * dist * 0.55 - 18 * s * q
        r = (22 + 26 * q) * s * P["r0"][i] * spread ** 0.5
        a = smoothstep(0.0, 0.25, q) * (1 - smoothstep(0.55, 1.0, q))
        return x, y, r, a, 0.16 + 0.4 * q ** 1.5

    NM = int(motes)
    M = dict(ph=rng.random(NM), x0=rng.normal(0, 55, NM) * s * spread, y0=rng.normal(12, 12, NM) * s,
             rise=rng.uniform(50, 110, NM) * s, sw=rng.uniform(3, 9, NM) * s, swp=rng.uniform(0, TAU, NM),
             a=rng.uniform(0.5, 1.3, NM), r=rng.uniform(0.7, 1.4, NM) * s, k=rng.choice([1, 2], NM))
    NB = int(bubbles)
    B = dict(ph=rng.random(NB), x=rng.normal(0, 50, NB) * s * spread, y=rng.normal(10, 14, NB) * s,
             R=rng.uniform(4, 9, NB) * s)

    out = []
    for f in range(frames):
        t = f / frames
        tl = (t * speed) % 1.0
        c = Canvas(w, h)
        _rising_puffs(c, tex, P, tl, range(NP), path, 0.16 * density, amp=0.8)
        if glow > 0:
            c.buf += glow * 0.25 * blur(c.dens, 10 * s * SS)
        for i in range(NM):
            q = (M["ph"][i] + M["k"][i] * tl) % 1.0
            x = cx + M["x0"][i] + M["sw"][i] * math.sin(TAU * (M["k"][i] * tl) * 2 + M["swp"][i])
            y = cy + M["y0"][i] - M["rise"][i] * q
            fd = math.sin(math.pi * q) ** 1.5
            c.dot(x, y, M["r"][i], M["a"][i] * fd)
            c.dot(x, y, M["r"][i] * 3.5, 0.12 * M["a"][i] * fd)
        for i in range(NB):
            q = (B["ph"][i] + tl) % 1.0
            x, y = cx + B["x"][i], cy + B["y"][i] - 30 * s * q
            if q < 0.82:                       # bubble grows ...
                R = B["R"][i] * (0.3 + 0.7 * q / 0.82)
                c.ring(x, y, R, 0.5 * s, 0.32 * smoothstep(0, 0.15, q))
                c.dot(x - R * 0.35, y - R * 0.35, 0.8 * s, 0.35 * smoothstep(0, 0.15, q))
            else:                              # ... and pops into a fading flash ring
                k = (q - 0.82) / 0.18
                c.ring(x, y, B["R"][i] * (1 + 1.2 * k), 0.5 * s, 0.45 * (1 - k) ** 2)
        out.append(_render_shaded(c, palette, gas_palette, opacity=6.0, shadow=0.8, ambient=0.25,
                                  bloom=((1.5 * s, 0.4), (5 * s, 0.35)), alpha_gamma=0.9))
    return out, 16


# ---------------------------------------------------------------------------
# torch flame
# ---------------------------------------------------------------------------
@effect("torch_flame", "fire", "minor", "Small steady torch / candle flame with flicker", loop=True)
def fx_torch_flame(rng, palette, scale=1.0, frames=24, height=1.0, width=1.0, flicker=1.0, intensity=1.0,
                   speed=2, glow=1.0, sparks=6):
    """Compact torch flame: the bonfire flame field with a narrow teardrop
    envelope, fewer octaves and a gentle sway, a warm halo and the odd
    rising spark. Integer noise cells per loop => seamless.

    Params:
      height: flame height multiplier
      width: flame width multiplier
      flicker: strength of the noise flicker / tongue break-up
      intensity: brightness multiplier
      speed: rise cycles per loop (integer keeps the loop seamless)
      glow: strength of the surrounding warm halo (0 = off)
      sparks: number of small rising sparks
    """
    size = int(256 * scale)
    s = size / 256
    cx, yb = size / 2, size * 0.72
    speed = max(1, int(round(speed)))
    xs, ys = _coords(size, size)
    N = Noise(rng, period=(64, 6, 2))
    W = Noise(rng, period=(64, 3, 1))
    NS = int(sparks)
    S = dict(ph=rng.random(NS), x0=rng.normal(0, 5, NS) * s, sw=rng.uniform(4, 12, NS) * s,
             swp=rng.uniform(0, TAU, NS), a=rng.uniform(0.8, 1.5, NS))
    H = 120 * s * height
    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(size, size)
        F, v = _flame_field(N, xs, ys, cx, yb, 21 * s * width, H, t, warp=W,
                            cell_x=13 * s, cell_y=34 * s, scroll_cells=6 * speed, zcells=2, top=1.15,
                            noise_amp=(0.18 * flicker, 0.8 * flicker), octaves=4)
        m = smoothstep(0.0, 0.12, F)
        heat = 1.6 * np.exp(-1.5 * np.clip(v, 0, None))
        I = m * (0.3 + 1.3 * np.clip(F, 0, 1.4) ** 1.2) * heat * intensity
        # blue-ish dim root is not in the palette; keep the root slightly darker
        I *= 0.55 + 0.45 * smoothstep(-0.04, 0.12, v)
        c.buf += _up(I.astype(np.float32))
        fl = 1 + 0.06 * flicker * math.sin(TAU * 3 * t * speed) + 0.04 * flicker * math.sin(TAU * 5 * t * speed + 1)
        if glow > 0:
            c.dot(cx, yb - H * 0.3, 26 * s, 0.10 * glow * fl * intensity)
            c.dot(cx, yb - H * 0.3, 44 * s, 0.03 * glow * fl * intensity)
        for i in range(NS):
            q = (S["ph"][i] + t * speed) % 1.0
            x = cx + S["x0"][i] + S["sw"][i] * q * math.sin(TAU * t * speed + S["swp"][i])
            y = yb - H * 0.6 - q * (yb - H * 0.6 - 30 * s)
            fd = math.sin(math.pi * q) ** 1.2 * (1 - q)
            c.dot(x, y, 0.8 * s, S["a"][i] * fd * 1.6)
        out.append(c.render(palette, bloom=((1.5 * s, 0.3), (5 * s, 0.25), (14 * s, 0.2)),
                            exposure=0.95, alpha_gamma=0.65))
    return out, 24


# ---------------------------------------------------------------------------
# dust puff
# ---------------------------------------------------------------------------
@effect("dust_puff", "dust", "minor", "Landing / impact dust puff rolling out along the ground and settling",
        loop=False)
def fx_dust_puff(rng, palette, scale=1.0, frames=20, puffs=34, spread=1.0, height=1.0, density=1.0,
                 duration=0.9, pebbles=14, symmetric=True):
    """Ground impact: noise-eroded dust puffs shoot sideways from the impact
    point with drag, swell, lift a little and erode away; a few grains arc
    up and fall back. `palette` is the dust ramp (shadow -> lit).

    Params:
      puffs: number of dust puffs
      spread: how far the dust rolls sideways
      height: how high the dust billows
      density: dust opacity multiplier
      duration: seconds the effect lasts (with `frames`, sets playback speed)
      pebbles: number of small grains thrown up (0 = off)
      symmetric: True = spreads both ways; False = mostly to the right
    """
    size = int(256 * scale)
    s = size / 256
    cx, gy = size / 2, size * 0.7
    tex = _Tex(rng, n=256, cells=8, octaves=5)
    NP = int(puffs)
    sign = rng.choice([-1, 1], NP) if symmetric else np.where(rng.random(NP) < 0.85, 1, -1)
    lateral = rng.uniform(0.03, 1.0, NP) ** 0.8
    P = dict(vx=sign * lateral * 400 * s * spread, up=(1 - lateral) ** 1.5 * rng.uniform(30, 110, NP) * s * height,
             r0=rng.uniform(10, 16, NP) * s, grow=rng.uniform(12, 24, NP) * s, ox=rng.random(NP),
             oy=rng.random(NP), rot=rng.uniform(0, TAU, NP), spin=rng.normal(0, 1.5, NP),
             t0=rng.uniform(0, 0.04, NP), life=rng.uniform(0.75, 1.0, NP))
    NG = int(pebbles)
    G = dict(vx=rng.normal(0, 110, NG) * s * spread, vy=-rng.uniform(120, 420, NG) * s * height,
             r=rng.uniform(0.7, 1.5, NG) * s)
    drag = 4.5
    out = []
    for f in range(frames):
        u = f / (frames - 1)
        t = duration * (0.07 + 0.93 * u ** 1.25)
        c = Canvas(size, size)
        for i in range(NP):
            tt = t - P["t0"][i]
            if tt <= 0:
                continue
            q = min(tt / (duration * P["life"][i]), 1.0)
            k = (1 - math.exp(-drag * tt)) / drag
            x = cx + P["vx"][i] * k
            r = P["r0"][i] + P["grow"][i] * q ** 0.6 * (0.7 + 0.6 * abs(P["vx"][i]) / (400 * s * spread))
            y = gy - r * 0.55 - P["up"][i] * k * 2.2 - 10 * s * height * q
            a = smoothstep(0, 0.03, q) * (1 - smoothstep(0.35, 1.0, q)) * _edge_fade(x, y, size, size, s, 70)
            _puff(c, tex, x, y, r, 0.24 * density * a, P["ox"][i] + 0.2 * q, P["oy"][i] - 0.2 * q,
                  P["rot"][i] + P["spin"][i] * q, detail=0.6, amp=0.8, thr=0.15 + 0.5 * q ** 1.4)
        for i in range(NG):
            x = cx + G["vx"][i] * t
            y = gy - 4 * s + G["vy"][i] * t + 0.5 * 1500 * s * t * t
            if y > gy or t > duration * 0.7:
                continue
            c.dot(x, y, G["r"][i], 3.0 * density * (1 - t / (duration * 0.7)), target="dens")
        out.append(_render_shaded(c, palette, palette, opacity=6.0, shadow=0.9, ambient=0.25, relief=1.6,
                                  height_gain=0.3))
    return out, round(frames / duration)


# ---------------------------------------------------------------------------
# embers
# ---------------------------------------------------------------------------
@effect("embers", "fire", "minor", "Field of drifting, swaying, twinkling embers and sparks", loop=True)
def fx_embers(rng, palette, scale=1.0, frames=32, count=60, speed=1, sway=1.0, size_mult=1.0,
              brightness=1.0, trail=1.0, twinkle=1.0):
    """Ambient embers: each rises on its own looping track (integer laps per
    loop) with sinusoidal sway, a short motion-blur streak, a soft halo and
    a twinkle; hot ones read yellow-white, faint ones deep red.

    Params:
      count: number of embers
      speed: base rise laps per loop (integer keeps the loop seamless)
      sway: sideways sway amount
      size_mult: ember size multiplier
      brightness: brightness multiplier
      trail: motion streak length multiplier (0 = round dots)
      twinkle: twinkle strength (0 = steady)
    """
    size = int(256 * scale)
    s = size / 256
    speed = max(1, int(round(speed)))
    n = int(count)
    E = dict(ph=rng.random(n), x0=rng.uniform(28, 228, n) * s, k=rng.choice([1, 1, 2], n) * speed,
             sw=rng.uniform(5, 20, n) * s * sway, sk=rng.choice([1, 2], n), swp=rng.uniform(0, TAU, n),
             drift=rng.normal(0, 18, n) * s, a=rng.uniform(0.35, 1.0, n) ** 1.5 * 2.2,
             r=rng.uniform(0.6, 1.3, n) * s * size_mult, tw=rng.choice([2, 3, 4, 5], n),
             twp=rng.uniform(0, TAU, n))

    def pos(i, tt):
        q = (E["ph"][i] + E["k"][i] * tt) % 1.0
        y = size - 20 * s - q * (size - 40 * s)
        x = E["x0"][i] + E["drift"][i] * q + E["sw"][i] * math.sin(TAU * E["sk"][i] * tt + E["swp"][i] + 3 * q)
        return x, y, q

    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(size, size)
        for i in range(n):
            x, y, q = pos(i, t)
            fd = math.sin(math.pi * q) ** 0.9 * _edge_fade(x, y, size, size, s, 20)
            tw = 1 - twinkle * 0.45 * (0.5 + 0.5 * math.sin(TAU * E["tw"][i] * t + E["twp"][i]))
            a = E["a"][i] * fd * tw * brightness
            if a < 0.01:
                continue
            px, py, pq = pos(i, t - 0.02 * trail)
            if trail > 0 and pq < q:
                c.polyline([(px, py), (x, y)], E["r"][i] * 0.4, a * 0.15, r_end=E["r"][i], a_end=a)
            else:
                c.dot(x, y, E["r"][i], a)
            c.dot(x, y, E["r"][i] * 4, a * 0.06)
        out.append(c.render(palette, bloom=((1.5 * s, 0.5), (5 * s, 0.35)), exposure=1.0, alpha_gamma=0.8))
    return out, 24


# ---------------------------------------------------------------------------
# muzzle flash
# ---------------------------------------------------------------------------
@effect("muzzle_flash", "fire", "minor", "Gun muzzle flash: star burst and forward cone, quick smoke wisp",
        loop=False)
def fx_muzzle_flash(rng, palette, scale=1.0, frames=7, length=1.0, spikes=5, spread=1.0, intensity=1.0,
                    smoke=1.0, smoke_palette="smoke", sparks=8):
    """Muzzle flash pointing right: a jagged forward cone of tapered flame
    petals plus a short side star, white-hot at the muzzle; it collapses in
    2-3 frames while a small smoke wisp drifts forward and fades.

    Params:
      length: forward cone length multiplier
      spikes: number of side star spikes
      spread: half-angle multiplier of the flame cone
      intensity: flash brightness multiplier
      smoke: smoke wisp amount (0 = off)
      smoke_palette: colour ramp of the smoke wisp (shadow -> lit)
      sparks: number of tiny forward sparks
    """
    size = int(256 * scale)
    s = size / 256
    mx, my = size * 0.2, size * 0.5
    tex = _Tex(rng, n=128, cells=4, octaves=4)
    NPET = 9
    pet_ang = rng.normal(0, 0.16, NPET) * spread
    pet_ang[0] = 0.0
    pet_len = rng.uniform(0.45, 1.0, NPET)
    pet_len[0] = 1.0
    side = np.linspace(0, TAU, int(spikes), endpoint=False) + rng.uniform(0, TAU)
    side_len = rng.uniform(0.5, 1.0, int(spikes))
    NS = int(sparks)
    SP = dict(ang=rng.normal(0, 0.25, NS) * spread, spd=rng.uniform(500, 900, NS) * s, a=rng.uniform(0.8, 1.6, NS))
    NW = 7
    Wp = dict(off=rng.uniform(0, 1, NW), dy=rng.normal(0, 6, NW) * s, r=rng.uniform(7, 12, NW) * s,
              ox=rng.random(NW), oy=rng.random(NW), rot=rng.uniform(0, TAU, NW))
    out = []
    for f in range(frames):
        u = f / max(frames - 1, 1)
        c = Canvas(size, size)
        e = math.exp(-u / 0.2) * intensity              # flash envelope
        fr = np.random.default_rng(int(rng.integers(1 << 31)))
        if e > 0.01:
            L = 150 * s * length * (0.65 + 0.35 * e)
            for i in range(NPET):
                a = pet_ang[i] + fr.normal(0, 0.03)
                ln = L * pet_len[i] * fr.uniform(0.8, 1.1)
                wid = (9 if i == 0 else 6) * s * (0.6 + 0.4 * e)
                tip = (mx + math.cos(a) * ln, my + math.sin(a) * ln)
                mid = (mx + math.cos(a) * ln * 0.35, my + math.sin(a) * ln * 0.35)
                nx, ny = -math.sin(a), math.cos(a)
                poly = [(mx, my - wid * 0.4 * ny), (mid[0] + nx * wid, mid[1] + ny * wid), tip,
                        (mid[0] - nx * wid, mid[1] - ny * wid), (mx, my + wid * 0.4 * ny)]
                c.polygon(poly, 0.9 * e * (1.0 if i == 0 else 0.7), soft=2.5 * s)
            for j in range(len(side)):
                a = side[j]
                ln = 34 * s * side_len[j] * (0.5 + 0.5 * e) * length ** 0.5
                c.polyline([(mx, my), (mx + math.cos(a) * ln, my + math.sin(a) * ln)], 2.6 * s, 1.2 * e,
                           r_end=0.4 * s, a_end=0.1 * e)
            c.dot(mx + 6 * s, my, 9 * s, 3.0 * e)
            c.dot(mx + 30 * s, my, 30 * s, 0.35 * e)
        for i in range(NS):
            tt = u * 0.12
            d = SP["spd"][i] * tt + 10 * s
            x, y = mx + math.cos(SP["ang"][i]) * d, my + math.sin(SP["ang"][i]) * d
            fd = (1 - u) ** 2 * _edge_fade(x, y, size, size, s, 30)
            x0, y0 = mx + math.cos(SP["ang"][i]) * d * 0.8, my + math.sin(SP["ang"][i]) * d * 0.8
            c.polyline([(x0, y0), (x, y)], 0.3 * s, 0.2 * fd, r_end=0.8 * s, a_end=SP["a"][i] * fd)
        if smoke > 0:
            for i in range(NW):
                q = 0.15 + 0.85 * u
                x = mx + (20 + 90 * Wp["off"][i]) * s * (0.4 + 0.6 * q ** 0.5)
                y = my + Wp["dy"][i] - 12 * s * q * (0.5 + Wp["off"][i])
                r = Wp["r"][i] * (0.8 + 1.4 * q)
                a = smoothstep(0, 0.2, u) * (1 - smoothstep(0.45, 1.0, u))
                _puff(c, tex, x, y, r, 0.22 * smoke * a, Wp["ox"][i] + 0.2 * q, Wp["oy"][i],
                      Wp["rot"][i] + q, detail=0.6, amp=0.8, thr=0.2 + 0.4 * q)
        out.append(_render_shaded(c, palette, smoke_palette, opacity=5.0, shadow=0.8, ambient=0.3,
                                  bloom=((1.5 * s, 0.5), (5 * s, 0.4), (14 * s, 0.3)), alpha_gamma=0.7))
    return out, 24
