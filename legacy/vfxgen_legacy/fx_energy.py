"""Magic / energy effects.

Every effect exposes its knobs as keyword arguments (see `--params NAME`).
Loop-critical speeds are integers (cycles / symmetry steps per loop) so any
value keeps the loop seamless; 0 freezes that motion.
"""
import math

import numpy as np

from .core import SS, TAU, Canvas, Noise, PALETTES, blur, effect, grid, refine, smoothstep  # noqa: F401

_grid = grid

# Smoke ramp for the portal's see-through dark eye (deep violet, never grey).
PALETTES.setdefault("abyss", [(0.0, (8, 0, 18)), (0.5, (22, 6, 40)), (1.0, (48, 16, 78))])


# ===========================================================================
# Shared helpers
# ===========================================================================
def _bloom(glow, *passes):
    """((sigma, weight), ...) with every weight scaled by `glow`."""
    return tuple((s, w * glow) for s, w in passes)


def _lo_grid(w, h):
    """Output-resolution pixel-centre coords (for expensive noise fields)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return xs + 0.5, ys + 0.5


def _up(a):
    """Bilinear upsample an output-res field to the supersampled canvas grid."""
    def axis(a, n, ax):
        src = np.clip((np.arange(n * SS) + 0.5) / SS - 0.5, 0, n - 1)
        i0 = np.floor(src).astype(np.int64)
        i1 = np.minimum(i0 + 1, n - 1)
        f = (src - i0).astype(np.float32)
        if ax == 0:
            return a[i0] * (1 - f)[:, None] + a[i1] * f[:, None]
        return a[:, i0] * (1 - f) + a[:, i1] * f
    h, w = a.shape
    return axis(axis(a, h, 0), w, 1).astype(np.float32)


def _glint(c, x, y, L, a, rot=0.0, w=None, diag=0.4):
    """4-point star glint: tapered cross streaks, shorter diagonals, hot core."""
    w = w or max(L * 0.07, 0.45)
    for k in range(4):
        ang = rot + k * TAU / 4
        c.polyline([(x, y), (x + L * math.cos(ang), y + L * math.sin(ang))],
                   w, a, r_end=w * 0.3, a_end=0.0)
        if diag:
            ang += TAU / 8
            Ld = L * diag
            c.polyline([(x, y), (x + Ld * math.cos(ang), y + Ld * math.sin(ang))],
                       w * 0.8, a * 0.55, r_end=w * 0.3, a_end=0.0)
    c.dot(x, y, max(L * 0.09, 0.6), a * 1.3)
    c.dot(x, y, L * 0.3, a * 0.18)


def _plus(c, x, y, L, a, w):
    """Chunky plus-sign sparkle (heal)."""
    c.polyline([(x - L, y), (x + L, y)], w, a)
    c.polyline([(x, y - L), (x, y + L)], w, a)
    c.dot(x, y, L * 0.5, a * 0.35)


def _ease_out(x, p=3):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** p


# ===========================================================================
# Original effects (now parameterised; defaults = original look)
# ===========================================================================
@effect("lightning", "electric", "major", "Branching lightning strike: hit, flicker, re-strike, fade", loop=False)
def fx_lightning(rng, palette, scale=1.0, frames=10, fps=16, intensity=1.0, width=1.0,
                 roughness=0.24, jitter=0.21, branches=-1, sub_branches=True, sway=0.08,
                 impact=1.0, restrike=True, glow=1.0):
    """Midpoint-displaced main bolt + branches, re-jittered every frame.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: animation length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      width: stroke width multiplier
      roughness: jaggedness of the bolt's overall shape
      jitter: per-frame crackle of the path
      branches: number of side branches (-1 = random 4-6)
      sub_branches: allow branches to fork again
      sway: random horizontal offset of start/end (fraction of width)
      impact: brightness of the ground impact flare (0 = off)
      restrike: flicker + re-strike envelope (False = single smooth fade)
      glow: bloom strength
    """
    w, h = int(256 * scale), int(512 * scale)
    s = scale * 2  # detail sizes below were tuned at 128x256
    cx = w / 2
    start = np.array([cx + rng.uniform(-sway, sway) * w, 10 * s])
    end = np.array([cx + rng.uniform(-sway, sway) * w, h - 22 * s])

    # Coarse skeleton: kept across frames so the strike has a stable shape.
    main = refine(rng, [start, end], rough=roughness, detail=h / 28)
    span = np.abs(main[:, 0] - cx).max()
    limit = w / 2 - 34 * s
    if span > limit:
        main[:, 0] = cx + (main[:, 0] - cx) * (limit / span)

    # Branch skeletons: (coarse pts, parent index, width, intensity)
    branch_list = [(main, -1, 1.0 * s * width, 3.4)]
    b_rough = 0.26 * roughness / 0.24

    def grow(parent_idx, path, depth, count):
        n = len(path)
        for _ in range(count):
            i = int(rng.integers(max(1, int(n * 0.12)), max(2, int(n * 0.85))))
            p = path[i]
            d = path[min(i + 2, n - 1)] - path[max(i - 1, 0)]
            d = d / (np.linalg.norm(d) + 1e-9)
            ang = rng.choice([-1, 1]) * rng.uniform(0.4, 1.0)
            ca, sa = math.cos(ang), math.sin(ang)
            dirv = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
            remaining = 1 - i / n
            L = rng.uniform(0.18, 0.38) * h * (0.5 if depth else 1.0) * (0.4 + 0.6 * remaining)
            q = p + dirv * L
            q[0] = np.clip(q[0], 30 * s, w - 30 * s)
            q[1] = np.clip(q[1], 12 * s, h - 24 * s)
            bp = refine(rng, [p, q], rough=b_rough, detail=max(L / 6, 5 * s))
            branch_list.append((bp, parent_idx, (0.72 if depth == 0 else 0.55) * s * width,
                                1.7 if depth == 0 else 0.9))
            if depth == 0:
                k = int(rng.integers(0, 3))
                grow(len(branch_list) - 1, bp, 1, k if sub_branches else 0)

    n_br = int(rng.integers(4, 7))
    grow(0, main, 0, n_br if branches < 0 else int(branches))

    # Strike, flicker, re-strike, fade.
    if restrike:
        env = np.array([0.8, 1.0, 0.45, 0.95, 0.7, 0.5, 0.35, 0.22, 0.12, 0.05])
    else:
        env = np.array([1.0, 0.85, 0.7, 0.55, 0.42, 0.3, 0.2, 0.12, 0.06, 0.02])
    env = np.interp(np.linspace(0, 1, frames), np.linspace(0, 1, len(env)), env)

    out = []
    for f in range(frames):
        fr = np.random.default_rng(int(rng.integers(1 << 31)))
        c = Canvas(w, h)
        e = env[f]
        visible = []
        for bi, (pts, parent, r, a) in enumerate(branch_list):
            ok = parent == -1 or (visible[parent] and fr.random() > 0.2 + 0.3 * (1 - e))
            visible.append(ok)
            if not ok:
                continue
            detail = 1.6 * s if parent == -1 else 1.4 * s
            path = refine(fr, pts, rough=jitter, detail=detail)
            path[:, 0] = np.clip(path[:, 0], 20 * s, w - 20 * s)
            if parent == -1:
                c.polyline(path, r, a * e)
            else:
                c.polyline(path, r, a * e, r_end=r * 0.5, a_end=a * e * 0.12)
        if impact > 0:
            c.dot(*end, 7 * s, 1.1 * e * impact)      # impact flare
            c.dot(*end, 18 * s, 0.25 * e * impact)    # wide impact glow
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * s, 0.7), (5 * s, 0.9), (16 * s, 0.9)),
                            exposure=1.0 * intensity))
    return out, fps


@effect("orb", "arcane", "major", "Energy orb with wobbling shells and orbiting motes", loop=True)
def fx_orb(rng, palette, scale=1.0, frames=16, fps=16, intensity=1.0, core=1.0, shells=3,
           wobble=1.0, motes=14, orbit=1.0, pulse=0.07, spin=1, glow=1.0):
    """Hot core inside wobbling harmonic shells, with orbiting motes.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      core: brightness of the central core
      shells: number of wobbling energy shells
      wobble: shell shape distortion multiplier
      motes: number of orbiting motes (0 = none)
      orbit: mote orbit radius multiplier
      pulse: size pulse amplitude (0 = steady)
      spin: integer speed of shells and motes (cycles per loop; 0 = frozen)
      glow: bloom strength
    """
    size = int(256 * scale)
    s = size / 128
    cx = cy = size / 2

    def harmonics(n):
        return [(int(rng.integers(2, 8)), int(rng.choice([-2, -1, 1, 2])),
                 rng.uniform(0, TAU), rng.uniform(0.4, 1.0)) for _ in range(n)]

    # Wobbling energy shells: radius, shape harmonics, brightness harmonics, strength
    shell_list = [(26 * s, harmonics(4), harmonics(2), 1.0),
                  (34 * s, harmonics(4), harmonics(2), 0.65),
                  (19 * s, harmonics(3), harmonics(2), 0.5)]

    mote_list = []
    for _ in range(motes):
        mote_list.append(dict(rho=rng.uniform(40, 52) * s * orbit, phi=rng.uniform(0, TAU),
                              k=int(rng.choice([1, 1, -1, 2])), squash=rng.uniform(0.2, 0.9),
                              tilt=rng.uniform(0, math.pi), a=rng.uniform(0.8, 1.6)))
    for _ in range(max(0, shells - 3)):
        shell_list.append((rng.uniform(16, 38) * s, harmonics(4), harmonics(2), rng.uniform(0.4, 0.7)))
    shell_list = shell_list[:max(0, shells)]

    X, Y = _grid(size, size)
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)

    out = []
    for f in range(frames):
        t = f / frames  # integer frequencies below => seamless loop
        ts = t * spin
        c = Canvas(size, size)
        pl = 1 + pulse * math.sin(TAU * t)
        c.buf += 3.0 * core * np.exp(-(r / (9 * s * pl)) ** 2)
        c.buf += 0.55 * core * np.exp(-(r / (24 * s * pl)) ** 2)
        for R0, shape, bright, strength in shell_list:
            n = sum(a * np.cos(m * th + TAU * k * ts + ph) for m, k, ph, a in shape)
            n /= sum(a for *_, a in shape)
            b = sum(a * np.cos(m * th - TAU * k * ts + ph) for m, k, ph, a in bright)
            b = np.clip(0.55 + 0.6 * b / sum(a for *_, a in bright), 0, None)
            R = R0 * pl + 6 * s * wobble * n
            c.buf += strength * 1.6 * b * b * np.exp(-((r - R) / (1.1 * s)) ** 2)
        for m in mote_list:
            ct, st = math.cos(m["tilt"]), math.sin(m["tilt"])
            head = m["phi"] + TAU * m["k"] * ts
            phis = head - np.sign(m["k"]) * np.linspace(0.55, 0, 10)  # tail -> head
            ex, ey = m["rho"] * np.cos(phis), m["rho"] * m["squash"] * np.sin(phis)
            pts = np.stack([cx + ct * ex - st * ey, cy + st * ex + ct * ey], 1)
            depth = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(head))
            c.polyline(pts, 0.3 * s, 0.0, r_end=1.0 * s, a_end=m["a"] * depth * 1.4)
            c.dot(*pts[-1], 1.3 * s, m["a"] * depth)
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * s, 0.6), (4 * s, 0.6), (11 * s, 0.55)),
                            exposure=1.0 * intensity))
    return out, fps


# ===========================================================================
# Major effects
# ===========================================================================
def _make_glyph(rng):
    """A procedural rune: list of strokes in a unit cell (x in [-.3,.3], y in
    [-.5,.5]); strokes walk an anchor grid, plus optional stave / loop / dot."""
    cols = [-0.3, 0.0, 0.3]
    rows = [-0.5, -1 / 6, 1 / 6, 0.5]
    strokes = []
    if rng.random() < 0.7:
        ci = int(rng.integers(0, 3))
        r0, r1 = (0, 3) if rng.random() < 0.7 else (0, 2)
        strokes.append([(cols[ci], rows[r0]), (cols[ci], rows[r1])])
    for _ in range(int(rng.integers(2, 4))):
        ci, ri = int(rng.integers(0, 3)), int(rng.integers(0, 4))
        pts = [(cols[ci], rows[ri])]
        for _ in range(int(rng.integers(1, 3))):
            for _try in range(8):
                dc, dr = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
                if dr == 0 and dc == 0:
                    continue
                nc, nr = ci + dc * int(rng.integers(1, 3)), ri + dr
                if 0 <= nc < 3 and 0 <= nr < 4:
                    ci, ri = nc, nr
                    pts.append((cols[ci], rows[ri]))
                    break
        if len(pts) > 1:
            strokes.append(pts)
    extra = rng.random()
    if extra < 0.3:   # small loop
        ox, oy = cols[int(rng.integers(0, 3))], rows[int(rng.integers(1, 3))]
        a = np.linspace(0, TAU, 13)
        strokes.append(list(zip(ox + 0.13 * np.cos(a), oy + 0.13 * np.sin(a))))
    elif extra < 0.55:  # dot
        strokes.append([(cols[int(rng.integers(0, 3))], rows[int(rng.integers(0, 4))])])
    return strokes


def _glyph_ring(c, glyphs, cx, cy, R, H, rot, a, lw, sweep):
    """Draw glyphs (list of stroke lists) around a ring of radius R, glyph
    height H, upright = outward. sweep(angle) -> brightness multiplier."""
    n = len(glyphs)
    for i, strokes in enumerate(glyphs):
        ang0 = rot + i * TAU / n
        aa = a * sweep(ang0)
        for st in strokes:
            st = np.asarray(st, np.float64)
            if len(st) == 1:   # dot
                gx, gy = st[0]
                ang, rad = ang0 + gx * H / R, R - gy * H
                c.dot(cx + rad * math.cos(ang), cy + rad * math.sin(ang), lw * 1.5, aa * 0.9)
                continue
            dense = []
            for p, q in zip(st[:-1], st[1:]):
                k = max(2, int(np.linalg.norm(q - p) * H / 2.5))
                dense.append(p + (q - p) * np.linspace(0, 1, k, endpoint=False)[:, None])
            dense.append(st[-1:])
            st = np.vstack(dense)
            ang = ang0 + st[:, 0] * H / R
            rad = R - st[:, 1] * H
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1)
            c.polyline(pts, lw, aa)


@effect("magic_circle", "arcane", "major",
        "Summoning circle: counter-rotating rune bands, hexagram, ticks, orbiting sparks", loop=True)
def fx_magic_circle(rng, palette, scale=1.0, frames=32, fps=16, intensity=1.0, rotation_speed=1,
                    glyphs=9, glyph_repeat=4, inner_glyphs=True, hexagram=True, ticks=True,
                    star=True, sparks=9, inner_sparks=5, pulse=0.12, sweep=1.0, haze=1.0,
                    line_width=1.0, glow=1.0):
    """Front-on summoning circle. Every rotating band is P-fold symmetric and
    turns a whole number of symmetry steps per loop, so the loop is seamless.
    A soft highlight sweeps once around the circle; everything pulses gently.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      rotation_speed: integer band rotation (symmetry steps per loop; negative reverses, 0 = still)
      glyphs: number of unique runes in the outer band
      glyph_repeat: times the rune set repeats around the band (total = glyphs * glyph_repeat)
      inner_glyphs: draw the small inner rune ring
      hexagram: draw the hexagram, tip circles and inner hexagon
      ticks: draw the tick / dot scale band
      star: draw the central 8-point star
      sparks: sparks orbiting outside the circle
      inner_sparks: sparks orbiting inside the hexagram
      pulse: brightness pulse amplitude
      sweep: strength of the highlight sweeping around the circle
      haze: faint glow filling the disc
      line_width: stroke width multiplier
      glow: bloom strength
    """
    size = int(320 * scale)
    u = size / 320
    cx = cy = size / 2
    X, Y = grid(size, size)
    dx, dy = X - cx, Y - cy
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    lw = 0.62 * u * line_width

    def ringf(R, wid=lw):
        return np.exp(-0.5 * ((r - R) / wid) ** 2)

    def tickf(n, rot, r0, r1, wid=lw):
        f = (th - rot) * n / TAU
        f = f - np.round(f)
        d = f * TAU / n * r
        win = smoothstep(r0 - 0.8 * u, r0 + 0.4 * u, r) * smoothstep(r1 + 0.8 * u, r1 - 0.4 * u, r)
        return np.exp(-0.5 * (d / wid) ** 2) * win

    def dashes(n, rot, R, duty, wid=lw):
        f = ((th - rot) * n / TAU) % 1.0
        e = (1.2 * u) / (TAU * R / n)
        return ringf(R, wid) * smoothstep(0, e, f) * smoothstep(duty, duty - e, f)

    # layout (radii in px at 320)
    R_out, R_out2 = 138 * u, 133 * u
    R_tk0, R_tk1 = 119 * u, 129 * u
    R_gb_o, R_gb_i = 116 * u, 95 * u
    R_gly, H_gly = 105.5 * u, 11 * u
    R_hex = 92 * u
    R_in = R_hex * 0.5
    R_gly2, H_gly2 = 33 * u, 7.5 * u
    lwu = line_width * u

    # static rings
    static = (1.1 * ringf(R_out, 0.8 * lwu) + 0.7 * ringf(R_out2) + 0.8 * ringf(R_gb_o)
              + 0.9 * ringf(R_gb_i) + 0.45 * ringf(R_gb_i - 3 * u, 0.5 * lwu)
              + 0.85 * ringf(R_in) + 0.5 * ringf(14 * u))
    # faint magical haze filling the disc
    hz = haze * (0.045 * smoothstep(R_out + 2 * u, R_out - 30 * u, r) + 0.1 * np.exp(-(r / (40 * u)) ** 2))

    # glyph sets: U unique glyphs repeated P times -> rotate by TAU/P per loop
    P1 = max(1, int(glyph_repeat))
    g1 = [_make_glyph(rng) for _ in range(max(1, int(glyphs)))] * P1
    P2 = 4
    g2 = [_make_glyph(rng) for _ in range(3)] * P2

    spark_list = [dict(rho=rng.uniform(143, 148) * u, phi=rng.uniform(0, TAU),
                       k=int(rng.choice([1, 1, 2, -1])), a=rng.uniform(0.7, 1.3)) for _ in range(sparks)]
    inner_list = [dict(rho=rng.uniform(55, 80) * u, phi=rng.uniform(0, TAU),
                       k=int(rng.choice([-1, -2, 1])), a=rng.uniform(0.5, 0.9)) for _ in range(inner_sparks)]

    out = []
    rs = int(rotation_speed)
    for f in range(frames):
        t = f / frames
        c = Canvas(size, size)
        pl = 1 + pulse * math.sin(TAU * t)
        sw_ang = TAU * t

        def sweepf(a):
            return 0.72 + 0.45 * sweep * max(0.0, math.cos(a - sw_ang)) ** 2

        rot_tk = rs * TAU / 24 * t          # ticks: 24-fold majors
        rot_g1 = -rs * TAU / P1 * t         # outer runes, counter
        rot_hex = rs * TAU / 6 * t          # hexagram
        rot_ds = -rs * TAU / 12 * t         # inner dashes
        rot_g2 = rs * TAU / P2 * t          # inner runes
        rot_st = -rs * TAU / 8 * t          # central star

        layer = static.copy()
        if ticks:
            layer += 0.75 * tickf(96, rot_tk, R_tk0 + 3 * u, R_tk1 - 2 * u)
            layer += 1.0 * tickf(24, rot_tk, R_tk0 - 1 * u, R_tk1 + 1 * u, 0.75 * lwu)
            # dots between the major ticks
            layer += 0.9 * tickf(24, rot_tk + TAU / 48, R_tk1 - 0.5 * u, R_tk1 + 0.5 * u, 1.1 * lwu)
        layer += 0.8 * dashes(12, rot_ds, R_in - 5 * u, 0.55)
        layer += 0.55 * dashes(36, -rot_ds, R_out2 + 2.5 * u, 0.4, 0.5 * lwu)
        c.buf += layer + hz

        _glyph_ring(c, g1, cx, cy, R_gly, H_gly, rot_g1, 1.15, lw, sweepf)
        if inner_glyphs:
            _glyph_ring(c, g2, cx, cy, R_gly2, H_gly2, rot_g2, 1.0, lw * 0.85, sweepf)

        if hexagram:
            tips = [(cx + R_hex * math.cos(rot_hex + k * TAU / 6 - TAU / 4),
                     cy + R_hex * math.sin(rot_hex + k * TAU / 6 - TAU / 4)) for k in range(6)]
            for tri in (tips[0::2], tips[1::2]):
                for p, q in zip(tri, tri[1:] + tri[:1]):
                    c.polyline([p, q], lw * 1.05, 1.15)
            for k, (px, py) in enumerate(tips):
                c.ring(px, py, 6.5 * u, lw, 1.0 * sweepf(rot_hex + k * TAU / 6 - TAU / 4))
                c.dot(px, py, 1.6 * u, 1.0)
            # inner hexagon (rotates the other way)
            hx = [(cx + R_in * 0.82 * math.cos(-rot_hex + k * TAU / 6),
                   cy + R_in * 0.82 * math.sin(-rot_hex + k * TAU / 6)) for k in range(6)]
            for p, q in zip(hx, hx[1:] + hx[:1]):
                c.polyline([p, q], lw * 0.8, 0.45)

        if star:   # central 8-point star (two squares)
            for off in (0.0, TAU / 8):
                sq = [(cx + 21 * u * math.cos(rot_st + off + k * TAU / 4),
                       cy + 21 * u * math.sin(rot_st + off + k * TAU / 4)) for k in range(5)]
                c.polyline(sq, lw, 0.85)

        # rotating highlight sweep across the whole circle (stronger on outer parts)
        m = 0.8 + 0.4 * sweep * np.maximum(np.cos(th - sw_ang), 0) ** 2 * smoothstep(10 * u, 70 * u, r)
        c.buf *= m * pl

        # core glow
        c.dot(cx, cy, 5 * u, 1.4 * pl)
        c.dot(cx, cy, 16 * u, 0.35 * pl)

        # orbiting sparks with tapered trails
        for sp, n_tr in [(s_, 0.5) for s_ in spark_list] + [(s_, 0.8) for s_ in inner_list]:
            head = sp["phi"] + TAU * sp["k"] * t
            phis = head - np.sign(sp["k"]) * np.linspace(n_tr, 0, 12)
            pts = np.stack([cx + sp["rho"] * np.cos(phis), cy + sp["rho"] * np.sin(phis)], 1)
            c.polyline(pts, 0.25 * u, 0.0, r_end=0.9 * u, a_end=sp["a"] * 1.2)
            c.dot(*pts[-1], 1.3 * u, sp["a"] * 1.2)
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.5), (4 * u, 0.45), (12 * u, 0.4)),
                            exposure=1.0 * intensity, alpha_gamma=0.8))
    return out, fps


@effect("portal", "void", "major",
        "Swirling vortex portal: spiralling arms, bright rim, dark eye, particles sucked in", loop=True)
def fx_portal(rng, palette, scale=1.0, frames=32, fps=20, intensity=1.0, arms=3, twist=1.15,
              inflow_speed=1, rotation_speed=0, warp=1.0, particles=46, rim=1.0, radius=1.0,
              eye_darkness=0.55, eye_palette="abyss", glow=1.0):
    """Domain-warped fbm in log-polar coordinates. Noise x = angle (periodic
    around the ring) + twist * log(r); noise y = log(r) scrolled by whole noise
    periods per loop, so features spiral inward and the loop is exact.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      arms: number of spiral arms
      twist: how tightly the arms wind
      inflow_speed: integer inward flow (noise periods per loop; 0 = no inflow)
      rotation_speed: integer extra spin (full turns per loop; negative reverses)
      warp: domain-warp turbulence strength
      particles: number of particles being sucked in
      rim: brightness of the rim
      radius: portal size multiplier (<= 1.05)
      eye_darkness: opacity of the dark see-through eye (0 = fully transparent)
      eye_palette: colour ramp of the dark eye (preset or hex list)
      glow: bloom strength
    """
    w, h = int(320 * scale), int(384 * scale)
    u = w / 320
    cx, cy = w / 2, h / 2
    rad = min(radius, 1.05)
    rx, ry = 116 * u * rad, 148 * u * rad
    NX, NY, NZ = 12, 8, 4
    C, TW = 3.6, twist           # radial cells per log unit, spiral twist
    noise = Noise(rng, period=(NX, NY, NZ))
    wnoise = Noise(rng, period=(NX, NY, NZ))
    rimn = Noise(rng, period=(24, 4, NZ))

    xs, ys = _lo_grid(w, h)
    ex, ey = (xs - cx) / rx, (ys - cy) / ry
    rho = np.hypot(ex, ey)
    th = np.arctan2(ey, ex)
    L = np.log(rho + 0.02)
    bx0 = th / TAU * NX + TW * C * L
    by0 = C * L
    env = smoothstep(0.06, 0.6, rho) * smoothstep(1.13, 0.94, rho) * (0.25 + 0.75 * np.clip(rho, 0, 1) ** 1.6)

    X, Y = grid(w, h)
    EX, EY = (X - cx) / rx, (Y - cy) / ry
    RHO = np.hypot(EX, EY)
    TH = np.arctan2(EY, EX)
    ang1d = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)
    eye = 1.6 * smoothstep(0.85, 0.05, RHO) ** 1.5      # dark see-through eye (smoke)

    # infalling particles follow the same log-spiral as the noise arms
    dth_dL = -TAU * TW * C / NX
    parts = [dict(L0=math.log(rng.uniform(0.98, 1.16)), th0=rng.uniform(0, TAU),
                  ph=rng.uniform(0, 1), k=int(rng.choice([1, 1, 2])),
                  K=rng.uniform(2.0, 2.6), a=rng.uniform(0.6, 1.4)) for _ in range(particles)]

    out = []
    for f in range(frames):
        t = f / frames
        z = t * NZ
        bx = bx0 - NX * int(rotation_speed) * t
        by = by0 + NY * int(inflow_speed) * t
        wx = wnoise.fbm(bx, by, z, octaves=2)
        wy = wnoise.fbm(bx + 3.0, by + 5.0, z, octaves=2)
        n = noise.fbm(bx + 0.9 * warp * wx, by + 0.7 * warp * wy, z, octaves=4)
        ridge = (1 - np.abs(n)) ** 6
        body = smoothstep(0.0, 0.75, n)
        # spiral arms, scrolled with the flow
        arm = 0.5 + 0.5 * np.cos(TAU * (int(arms) * bx / NX + by / NY) + 2.2 * warp * wx)
        I = env * (0.3 + 0.7 * arm ** 1.5) * (1.7 * ridge + 1.1 * body * body + 0.12)
        # ragged energy licking just outside the rim
        I += rim * np.exp(-np.maximum(rho - 1, 0) / 0.035) \
            * smoothstep(0.93, 1.0, rho) * (0.15 + 0.9 * body) * (0.5 + 0.5 * arm)
        c = Canvas(w, h)
        c.buf += _up(I)
        # rim: thin bright noisy line + soft halo
        rn = np.interp(TH, ang1d, rimn.fbm(ang1d / TAU * 24, 0.0, z, octaves=3))
        rimf = (1.5 + 1.2 * rn) * np.exp(-0.5 * ((RHO - 1.0) / 0.011) ** 2) \
            + (0.45 + 0.35 * rn) * np.exp(-0.5 * ((RHO - 0.985) / 0.045) ** 2)
        c.buf += rim * np.clip(rimf, 0, None)
        c.dens += eye

        spin = TAU * int(rotation_speed) * t
        for p in parts:
            s = (p["k"] * t + p["ph"]) % 1.0
            ss = np.linspace(max(s - 0.07, 0), s, 9)
            Ls = p["L0"] - p["K"] * ss
            rr = np.exp(Ls)
            ths = p["th0"] + dth_dL * (Ls - p["L0"]) + spin
            pts = np.stack([cx + rr * rx * np.cos(ths), cy + rr * ry * np.sin(ths)], 1)
            fade = smoothstep(0.0, 0.12, s) * smoothstep(0.12, 0.4, rr[-1])
            if fade <= 0.01:
                continue
            c.polyline(pts, 0.25 * u, 0.0, r_end=0.95 * u, a_end=p["a"] * 1.3 * fade)
            c.dot(*pts[-1], 1.2 * u, p["a"] * fade)
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * u, 0.45), (5 * u, 0.4), (14 * u, 0.35)),
                            exposure=0.9 * intensity, alpha_gamma=0.85,
                            smoke=eye_palette if eye_darkness > 0 else None,
                            smoke_opacity=eye_darkness, shadow=0.3))
    return out, fps


@effect("beam", "electric", "major",
        "Horizontal energy laser: blinding core, turbulent scrolling sheath, emitter flare, impact splash", loop=True)
def fx_beam(rng, palette, scale=1.0, frames=24, fps=20, intensity=1.0, core_width=1.0,
            sheath_width=1.0, turbulence=1.0, filaments=True, scroll_speed=1, surge=0.22,
            flicker=1.0, emitter=True, impact=True, sparks=40, spark_spread=1.3, glow=1.0):
    """Core + anisotropic fbm sheath scrolling whole noise periods per loop;
    emitter flare with rotating 6-fold spikes; impact splash from angular
    noise, pulsing shock rings and looping spark particles.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      core_width: width of the blinding core
      sheath_width: width of the turbulent outer sheath
      turbulence: brightness of the sheath plasma
      filaments: draw the bright twisting filaments in the sheath
      scroll_speed: integer flow speed along the beam (noise periods per loop)
      surge: amplitude of energy pulses racing along the core
      flicker: brightness flicker amount
      emitter: draw the emitter flare (left)
      impact: draw the impact splash, shock rings and sparks (right)
      sparks: number of impact sparks
      spark_spread: spray half-angle of the sparks (radians)
      glow: bloom strength
    """
    w, h = int(512 * scale), int(192 * scale)
    u = h / 192
    cy = h / 2
    x0, x1 = 46 * u, w - 54 * u
    PX, PZ = 16, 4
    sp_k = int(scroll_speed)
    nA = Noise(rng, period=(PX, 16, PZ))
    nB = Noise(rng, period=(PX, 16, PZ))
    nW = Noise(rng, period=(PX, 4, PZ))
    nS = Noise(rng, period=(8, 4, PZ))

    band = int(min(cy, 58 * u))
    xs, ys = _lo_grid(w, h)
    ys_c = ys - cy
    rows = slice(int(cy - band), int(cy + band))
    xb, yb = xs[rows], ys_c[rows]
    along = smoothstep(x0 - 6 * u, x0 + 18 * u, xb) * smoothstep(x1 + 10 * u, x1 - 6 * u, xb)

    X, Y = grid(w, h)
    DY = Y - cy
    along_hi = smoothstep(x0 - 3 * u, x0 + 8 * u, X) * smoothstep(x1 + 6 * u, x1 - 2 * u, X)
    IX, IY = X - x1, Y - cy
    IR = np.hypot(IX, IY)
    ITH = np.arctan2(IY, IX)
    ang1d = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)

    spark_list = []
    for _ in range(sparks):
        ang = math.pi + rng.uniform(-spark_spread, spark_spread)
        spark_list.append(dict(ang=ang, D=rng.uniform(35, 80) * u, ph=rng.uniform(0, 1),
                               k=int(rng.choice([1, 2, 2, 3])), a=rng.uniform(0.8, 1.6),
                               g=rng.uniform(10, 26) * u))

    out = []
    for f in range(frames):
        t = f / frames
        z = t * PZ
        c = Canvas(w, h)
        flick = 1 + flicker * (0.06 * math.sin(TAU * 3 * t) + 0.04 * math.sin(TAU * 5 * t + 1.3))
        # sheath (lo-res, band rows only)
        sx = xb / (34 * u) - PX * sp_k * t
        wob = nW.fbm(xb / (68 * u) - PX * sp_k * t, 0.0, z, octaves=2)
        W = 17 * u * sheath_width * (1 + 0.35 * wob)
        a_ = nA.fbm(sx, yb / (11 * u * sheath_width), z, octaves=3)
        sheath = np.exp(-(yb / W) ** 2) * (0.08 + 0.42 * smoothstep(-0.1, 0.7, a_))
        outer = 0.07 * np.exp(-(yb / (W * 2.0)) ** 2) * (0.6 + 0.4 * a_)
        I = np.zeros_like(xs)
        Ib = sheath + outer
        if filaments:
            b_ = nB.fbm(xb / (64 * u) - 2 * PX * sp_k * t, yb / (9 * u * sheath_width), z, octaves=3)
            Ib = Ib + 0.75 * (1 - np.abs(b_)) ** 8 * np.exp(-(yb / (W * 1.3)) ** 2)
        I[rows] = Ib * turbulence * along * flick
        c.buf += _up(I)
        # core (hi-res)
        wob_hi = 1 + 0.12 * math.sin(TAU * 2 * t)
        sg = 1 + surge * np.sin(TAU * (X / (70 * u) - 4 * t))   # pulses racing along
        c.buf += along_hi * flick * (2.8 * np.exp(-(DY / (2.0 * u * core_width * wob_hi)) ** 2)
                                     + 0.6 * sg * np.exp(-(DY / (5.5 * u * core_width)) ** 2))

        if emitter:
            ep = 1 + 0.15 * math.sin(TAU * 2 * t)
            c.dot(x0, cy, 6 * u * ep, 2.4)
            c.dot(x0, cy, 16 * u * ep, 0.55)
            for sgn in (-1, 1):
                c.polyline([(x0, cy), (x0, cy + sgn * 34 * u * ep)], 1.4 * u, 1.6, r_end=0.3 * u, a_end=0)
            c.polyline([(x0, cy), (x0 - 30 * u * ep, cy)], 1.6 * u, 1.2, r_end=0.3 * u, a_end=0)
            for k in range(6):
                ang = TAU / 6 * t + k * TAU / 6 + 0.3
                L = (14 + 6 * (k % 2)) * u * ep
                c.polyline([(x0, cy), (x0 + L * math.cos(ang), cy + L * math.sin(ang))],
                           0.9 * u, 0.9, r_end=0.2 * u, a_end=0)

        if impact:
            # noisy splash fan pointing back toward the emitter
            sn = np.interp(ITH, ang1d, nS.fbm(ang1d / TAU * 16, 0.0, z * 2, octaves=3))
            back = 0.25 + 0.75 * np.maximum(-np.cos(ITH), 0)
            Lr = (6 + 30 * back * np.clip(0.45 + 0.8 * sn, 0.05, None)) * u
            c.buf += 0.9 * np.exp(-IR / np.maximum(Lr, 1e-3) * 2.5) * (IR < 70 * u)
            c.dot(x1, cy, 7 * u, 2.8 * flick)
            c.dot(x1, cy, 20 * u, 0.35)
            c.buf += 0.5 * np.exp(-(IX / (5 * u)) ** 2 - (IY / (34 * u)) ** 2)   # flare on the wall
            for k in range(2):
                ph = (2 * t + k * 0.5) % 1.0
                R = (4 + 26 * _ease_out(ph, 2)) * u
                a = 0.9 * (1 - ph) ** 2
                if a > 0.01:
                    d = np.hypot(IX / 0.45, IY) - R
                    c.buf += a * np.exp(-0.5 * (d / (1.1 * u)) ** 2)
            for sp in spark_list:
                s = (sp["k"] * t + sp["ph"]) % 1.0

                def pos(sv):
                    e = 1 - (1 - sv) ** 2
                    return (x1 + math.cos(sp["ang"]) * sp["D"] * e,
                            cy + math.sin(sp["ang"]) * sp["D"] * e + sp["g"] * sv * sv)
                p_tail, p_head = pos(max(s - 0.14, 0)), pos(s)
                a = sp["a"] * (1 - s) ** 1.2 * smoothstep(0, 0.04, s)
                c.polyline([p_tail, p_head], 0.25 * u, 0.0, r_end=0.7 * u, a_end=a * 2.2)
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * u, 0.5), (5 * u, 0.4), (15 * u, 0.35)),
                            exposure=1.0 * intensity, alpha_gamma=0.85))
    return out, fps


@effect("shield", "frost", "major",
        "Protective bubble: fresnel rim, hexagon grid on the sphere, shimmer wave sweeping across", loop=True)
def fx_shield(rng, palette, scale=1.0, frames=32, fps=16, intensity=1.0, radius=1.0,
              hex_density=7.0, grid_brightness=1.0, wave=1.0, waves=1, twinkle=0.3, rim=1.0,
              fill=1.0, highlight=True, glow=1.0):
    """Sphere normal from the disc; fresnel rim (1 - n.z)^3; hex grid in
    stereographic coords so cells foreshorten toward the rim; a shimmer band
    (dot(n, dir) sweeping across, off-sphere at both ends) lights the grid.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      radius: bubble size multiplier (<= 1.08)
      hex_density: hexagon cells across the sphere (higher = smaller cells)
      grid_brightness: resting brightness of the hex grid lines
      wave: strength of the shimmer wave
      waves: integer number of shimmer sweeps per loop (0 = none)
      twinkle: fraction of cells that randomly glint
      rim: brightness of the fresnel rim
      fill: brightness of the faint interior tint
      highlight: small specular highlight top-left
      glow: bloom strength
    """
    size = int(320 * scale)
    u = size / 320
    cx = cy = size / 2
    R = 128 * u * min(radius, 1.08)
    X, Y = grid(size, size)
    nx, ny = (X - cx) / R, (Y - cy) / R
    rr = np.hypot(nx, ny)
    inside = rr < 1
    nz = np.sqrt(np.clip(1 - rr * rr, 0, 1))
    th = np.arctan2(ny, nx)

    # hex grid (flat, cell pitch 1) on stereographic coords
    K = hex_density
    px, py = nx / (1 + nz) * K, ny / (1 + nz) * K
    sq3 = math.sqrt(3)
    ax, ay = np.mod(px, 1.0) - 0.5, np.mod(py, sq3) - sq3 / 2
    bx, by = np.mod(px - 0.5, 1.0) - 0.5, np.mod(py - sq3 / 2, sq3) - sq3 / 2
    use_a = ax * ax + ay * ay < bx * bx + by * by
    gx, gy = np.where(use_a, ax, bx), np.where(use_a, ay, by)
    hexd = np.maximum(np.abs(gx) * 0.5 + np.abs(gy) * sq3 / 2, np.abs(gx))
    edge = 0.5 - hexd                                  # grid units to the cell edge
    # grid units per output pixel -> constant-width anti-aliased lines
    dpx = np.hypot(*np.gradient(px)) * SS
    lines = np.exp(-0.5 * (edge / (0.7 * dpx + 1e-4)) ** 2) * smoothstep(0.08, 0.4, nz) * inside
    fillm = smoothstep(0.0, 0.12, edge) * smoothstep(0.05, 0.35, nz) * inside
    # stable per-cell hash
    idx = np.round((px - gx) * 2).astype(np.int64)
    idy = np.round((py - gy) / (sq3 / 2)).astype(np.int64)
    hsh = ((idx * 73856093) ^ (idy * 19349663)) % 1009 / 1009.0
    hk = 1 + (hsh * 7).astype(np.int64) % 2
    lucky = (((idx * 83492791) ^ (idy * 2654435761)) % 997 / 997.0) < twinkle

    fres = (1 - nz) ** 3 * inside
    rim_line = np.exp(-0.5 * ((rr - 1) * R / (1.3 * u)) ** 2)
    halo = np.exp(-np.maximum(rr - 1, 0) * R / (7 * u)) * (rr > 1) * 0.35
    d3 = np.array([0.62, -0.48, 0.62])
    d3 /= np.linalg.norm(d3)
    sdir = nx * d3[0] + ny * d3[1] + nz * d3[2]

    rimnoise = Noise(rng, period=(16, 4, 4))
    ang1d = np.linspace(-math.pi, math.pi, 2049).astype(np.float32)

    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(size, size)
        if waves > 0:
            ph = -1.25 + 2.65 * ((int(waves) * t) % 1.0)
            wv = wave * np.exp(-0.5 * ((sdir - ph) / 0.12) ** 2) * inside
            trail = wave * np.exp(-0.5 * ((sdir - ph + 0.25) / 0.25) ** 2) * inside
        else:
            wv = trail = 0.0
        tw = np.maximum(0, np.sin(TAU * (hk * t + hsh))) ** 10 * lucky
        rn = np.interp(th, ang1d, rimnoise.fbm(ang1d / TAU * 16, 0.0, t * 4, octaves=3))
        breathe = 1 + 0.06 * math.sin(TAU * t)
        I = (0.05 * fill * inside + rim * 0.75 * fres * (0.85 + 0.3 * rn)
             + rim * ((1.25 + 0.5 * rn) * rim_line + halo)
             + lines * (0.16 * grid_brightness + 1.5 * wv + 0.35 * trail + 0.5 * tw)
             + fillm * (0.12 * wv + 0.06 * trail + 0.09 * tw)
             + 0.6 * wv * fres)
        c.buf += I * breathe
        if highlight:
            c.dot(cx - 0.45 * R, cy - 0.5 * R, 9 * u, 0.12)
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.45), (4 * u, 0.4), (12 * u, 0.35)),
                            exposure=1.0 * intensity, alpha_gamma=0.9))
    return out, fps


@effect("heal_aura", "heal", "major",
        "Healing aura: glowing ground ring under a rising column of light, motes and plus sparkles", loop=True)
def fx_heal_aura(rng, palette, scale=1.0, frames=32, fps=16, intensity=1.0, ring_radius=1.0,
                 column=1.0, column_height=1.0, motes=34, pluses=9, rise_speed=1, rotation_speed=1,
                 pulse=0.1, inner_rings=True, glow=1.0):
    """Ground-plane rings in squashed polar coords (dashes rotate whole symmetry
    steps per loop); column = edge-brightened profile x vertical fade x upward
    scrolling streak noise; motes/pluses live on (k*t + phase) % 1 cycles.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      ring_radius: ground ring size multiplier (<= 1.15)
      column: brightness of the rising light column (0 = off)
      column_height: height of the column multiplier
      motes: number of rising motes
      pluses: number of plus-sign sparkles
      rise_speed: integer rise speed of motes and column streaks (cycles per loop)
      rotation_speed: integer rotation of the dashed ground rings (steps per loop)
      pulse: brightness pulse amplitude
      inner_rings: draw the dashed inner ground rings
      glow: bloom strength
    """
    w, h = int(256 * scale), int(384 * scale)
    u = w / 256
    cx, gy = w / 2, h - 66 * u
    R, sq = 98 * u * min(ring_radius, 1.15), 0.3
    top = max(20 * u, gy - (gy - 46 * u) * column_height)
    X, Y = grid(w, h)
    gx_, gy_ = X - cx, (Y - gy) / sq
    gr = np.hypot(gx_, gy_)
    gth = np.arctan2(gy_, gx_)

    def ellip(RR, wid):
        return np.exp(-0.5 * ((gr - RR) / wid) ** 2)
    disc = smoothstep(R, R * 0.2, gr)

    # column (lo-res noise)
    PY = 8
    nz = Noise(rng, period=(16, PY, 4))
    xs, ys = _lo_grid(w, h)
    dxl = np.abs(xs - cx) / (R * 0.86)
    prof = smoothstep(1.0, 0.72, dxl) * (0.35 + 0.65 * dxl ** 2)
    front = gy + sq * R * 0.86 * np.sqrt(np.clip(1 - dxl ** 2, 0, 1))
    vert = smoothstep(top, gy, ys) ** 1.6 * smoothstep(front + 6 * u, front - 4 * u, ys)
    hs = (gy - top) / (gy - 46 * u)    # mote rise heights follow the column

    mote_list = [dict(r=math.sqrt(rng.uniform(0.02, 1)) * R * 0.85, a=rng.uniform(0, TAU),
                      H=rng.uniform(150, 260) * u * hs, ph=rng.uniform(0, 1), k=int(rng.choice([1, 1, 2])),
                      sz=rng.uniform(0.9, 1.7) * u, sway=rng.uniform(2, 7) * u, br=rng.uniform(0.6, 1.3))
                 for _ in range(motes)]
    plus_list = [dict(r=math.sqrt(rng.uniform(0.05, 1)) * R * 0.9, a=rng.uniform(0, TAU),
                      H=rng.uniform(120, 220) * u * hs, ph=rng.uniform(0, 1), k=1,
                      L=rng.uniform(3.5, 6.5) * u, br=rng.uniform(0.8, 1.3)) for _ in range(pluses)]

    rise, rot = int(rise_speed), int(rotation_speed)
    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(w, h)
        pl = 1 + pulse * math.sin(TAU * t)
        if column > 0:
            n = nz.fbm(xs / (14 * u), ys / (60 * u) + PY * rise * t, t * 4, octaves=3)
            streak = 0.45 + 0.75 * smoothstep(-0.2, 0.7, n)
            c.buf += _up(prof * vert * streak * 0.65 * column * pl)
        # ground rings
        G = 1.15 * ellip(R, 2.0 * u) + 0.4 * ellip(R, 7 * u) + 0.18 * disc
        if inner_rings:
            dash = smoothstep(0.35, 0.45, np.abs(((gth + rot * TAU / 16 * t) * 16 / TAU) % 1.0 - 0.5) * 2)
            dash2 = smoothstep(0.55, 0.62, np.abs(((gth - rot * TAU / 24 * t) * 24 / TAU) % 1.0 - 0.5) * 2)
            G = G + 0.9 * ellip(R * 0.84, 1.8 * u) * dash + 0.6 * ellip(R * 0.66, 1.5 * u) * (1 - dash2)
        c.buf += G * pl
        for m in mote_list:
            s = (m["k"] * rise * t + m["ph"]) % 1.0
            x = cx + m["r"] * math.cos(m["a"]) + m["sway"] * math.sin(TAU * (s * 2) + m["a"])
            y = gy + m["r"] * math.sin(m["a"]) * sq - m["H"] * s
            a = m["br"] * smoothstep(0, 0.15, s) * (1 - s) ** 1.3
            c.dot(x, y, m["sz"], a * 1.4)
            c.polyline([(x, y + 6 * u), (x, y)], 0.2 * u, 0.0, r_end=m["sz"] * 0.7, a_end=a * 0.6)
        for m in plus_list:
            s = (m["k"] * rise * t + m["ph"]) % 1.0
            x = cx + m["r"] * math.cos(m["a"])
            y = gy + m["r"] * math.sin(m["a"]) * sq - m["H"] * s
            a = m["br"] * math.sin(math.pi * s) ** 1.5
            L = m["L"] * (0.6 + 0.4 * math.sin(math.pi * s))
            _plus(c, x, y, L, a * 1.2, 0.75 * u)
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * u, 0.5), (5 * u, 0.45), (14 * u, 0.4)),
                            exposure=1.0 * intensity, alpha_gamma=0.9))
    return out, fps


@effect("holy_pillar", "holy", "major",
        "Pillar of light slams down from the sky, flares, widens and fades, with dust motes and a ground ring",
        loop=False)
def fx_holy_pillar(rng, palette, scale=1.0, frames=24, fps=20, intensity=1.0, width=1.0,
                   beam=1.0, impact=1.0, ground_ring=True, motes=46, dust=26, hold=0.45, glow=1.0):
    """Beam head races down in two frames, impact flare + expanding ground
    ring, width eases out while downward-scrolling streak noise fills it;
    everything rides a global envelope that reaches 0 on the last frame.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: animation length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      width: final pillar width multiplier
      beam: brightness of the pillar itself
      impact: strength of the ground impact flare and streak
      ground_ring: draw the expanding shock ring and glowing ground rings
      motes: number of glittering motes rising in the light
      dust: number of dust sparks kicked out along the ground
      hold: fraction of the animation at full strength before fading
      glow: bloom strength
    """
    w, h = int(256 * scale), int(512 * scale)
    u = w / 256
    cx, gy = w / 2, h - 72 * u
    top = 6 * u
    nz = Noise(rng, period=(16, 16, 4))
    xs, ys = _lo_grid(w, h)
    X, Y = grid(w, h)
    DX = X - cx
    topfade_lo = smoothstep(top, top + 110 * u, ys)
    topfade = smoothstep(top, top + 110 * u, Y)

    mote_list = [dict(x=cx + rng.normal(0, 30) * u, y=gy - rng.uniform(0, 300) * u,
                      vx=rng.normal(0, 10) * u, vy=-rng.uniform(20, 70) * u, sz=rng.uniform(0.9, 1.8) * u,
                      br=rng.uniform(0.8, 1.6), t0=rng.uniform(0.0, 0.35)) for _ in range(motes)]
    kicks = [dict(ang=rng.uniform(0, TAU), v=rng.uniform(40, 95) * u, sz=rng.uniform(0.8, 1.4) * u,
                  br=rng.uniform(0.6, 1.2)) for _ in range(dust)]
    hold = float(np.clip(hold, 0.05, 0.9))
    fk = frames / 24     # timings below were tuned at 24 frames

    out = []
    for f in range(frames):
        t = f / (frames - 1)
        c = Canvas(w, h)
        env = 1.0 if t < hold else float(smoothstep(1.0, hold, t)) ** 1.3
        if f == frames - 1:
            env = 0.0
        if env <= 0:
            out.append(c.render(palette))
            continue
        head = gy if f >= 1 else top + (gy - top) * 0.55
        wd = (3 + 26 * width * float(_ease_out(t / 0.4, 3))) * u
        core_w = wd * 0.32 + 1.5 * u
        bottom = smoothstep(head + 4 * u, head - 6 * u, Y)
        # streaky body
        n = nz.fbm(xs / (8 * u), ys / (70 * u) - 16 * t * 0.6, t * 2, octaves=3)
        body = np.exp(-((xs - cx) / wd) ** 2) * (0.2 + 0.9 * smoothstep(-0.1, 0.6, n))
        bot_lo = smoothstep(head + 4 * u, head - 6 * u, ys)
        c.buf += _up(body * topfade_lo * bot_lo * 0.6 * env * beam)
        flash = math.exp(-f / (2.2 * fk))
        c.buf += (1.8 * np.exp(-(DX / core_w) ** 2) + 0.18 * np.exp(-(DX / (wd * 2.2)) ** 2)) \
            * topfade * bottom * env * beam * (1 + 0.6 * flash)
        if f >= 1:
            # impact flare + horizontal anamorphic streak
            fl = (flash * 2.0 + 0.5 * env) * impact
            c.dot(cx, gy, 10 * u, 1.6 * fl)
            c.dot(cx, gy, 22 * u, 0.4 * fl)
            c.polyline([(cx - 104 * u, gy), (cx, gy)], 0.4 * u, 0.0, r_end=2.2 * u, a_end=1.2 * fl)
            c.polyline([(cx, gy), (cx + 104 * u, gy)], 2.2 * u, 1.2 * fl, r_end=0.4 * u, a_end=0.0)
            if ground_ring:
                gw = wd * 2.2
                c.ring(cx, gy, gw * 0.5, gw * 0.35, 0.5 * env, squash=0.28)
                ring_t = (f - 1) / (frames * 0.6)
                if ring_t < 1:
                    Rr = (12 + 92 * float(_ease_out(ring_t, 3))) * u
                    c.ring(cx, gy, Rr, (1.2 + 2.5 * ring_t) * u, 1.6 * (1 - ring_t) ** 1.5, squash=0.26)
                Rs = min(wd * 1.5 + 8 * u, 100 * u)
                c.ring(cx, gy, Rs, 1.6 * u, 0.9 * env, squash=0.26)
            # dust kicked out along the ground
            for k in kicks:
                kt = (f - 1) / (12 * fk)
                d = k["v"] * float(_ease_out(kt, 2))
                kx, ky = cx + d * math.cos(k["ang"]), gy + d * math.sin(k["ang"]) * 0.26 - 8 * u * kt
                a = k["br"] * max(0.0, 1 - (f - 1) / (14 * fk)) * env
                if a > 0.01:
                    c.dot(kx, ky, k["sz"], a)
        # rising motes inside the light
        for m in mote_list:
            tt = t - m["t0"]
            if tt < 0:
                continue
            x = m["x"] + m["vx"] * tt
            y = m["y"] + m["vy"] * tt
            a = m["br"] * smoothstep(0, 0.08, tt) * env * float(smoothstep(wd * 1.6, wd * 0.5, abs(x - cx)))
            c.dot(x, y, m["sz"], a)
        out.append(c.render(palette, bloom=_bloom(glow, (1.5 * u, 0.5), (5 * u, 0.4), (14 * u, 0.25)),
                            exposure=1.0 * intensity, alpha_gamma=0.72))
    return out, fps


# ===========================================================================
# Minor effects
# ===========================================================================
@effect("sparkle", "frost", "minor",
        "Cluster of twinkling 4-point star glints appearing at staggered times", loop=True)
def fx_sparkle(rng, palette, scale=1.0, frames=32, fps=16, intensity=1.0, glints=14, big=4,
               spread=1.0, size=1.0, duration=1.0, spin=1.0, diagonals=True, dust=24, glow=1.0):
    """Each glint lives on its own window of the (k*t + phase) % 1 cycle, so
    the cluster twinkles out of sync but loops exactly.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      glints: total number of star glints
      big: how many of them are large hero glints
      spread: cluster radius multiplier
      size: glint size multiplier
      duration: how long each glint stays lit (fraction-of-loop multiplier)
      spin: how much each glint rotates while lit
      diagonals: add the short diagonal rays
      dust: number of tiny twinkling specks
      glow: bloom strength
    """
    S = int(256 * scale)
    u = S / 256
    cx = cy = S / 2
    gl = []
    for i in range(glints):
        rad = math.sqrt(rng.uniform(0, 1)) * min(82 * spread, 100) * u
        ang = rng.uniform(0, TAU)
        is_big = i < big
        gl.append(dict(x=cx + rad * math.cos(ang), y=cy + rad * math.sin(ang),
                       L=(rng.uniform(36, 50) if is_big else rng.uniform(12, 24)) * u * size,
                       ph=(i * 0.618) % 1.0, dur=min(rng.uniform(0.3, 0.5) * duration, 1.0),
                       k=1 if is_big else int(rng.choice([1, 2])),
                       rot=rng.uniform(-0.25, 0.25), spin=rng.uniform(-0.6, 0.6) * spin,
                       br=rng.uniform(0.9, 1.4)))
    specks = [dict(x=cx + rng.normal(0, 45 * spread) * u, y=cy + rng.normal(0, 45 * spread) * u,
                   ph=rng.uniform(0, 1), k=int(rng.choice([1, 2, 3]))) for _ in range(dust)]
    out = []
    for f in range(frames):
        t = f / frames
        c = Canvas(S, S)
        for g in gl:
            s = ((g["k"] * t + g["ph"]) % 1.0) / g["dur"]
            if s >= 1:
                continue
            e = math.sin(math.pi * s) ** 2
            _glint(c, g["x"], g["y"], g["L"] * (0.35 + 0.65 * e), g["br"] * e * 1.3,
                   rot=g["rot"] + g["spin"] * (s - 0.5), w=max(0.55 * u, g["L"] * 0.045),
                   diag=0.4 if diagonals else 0)
        for d in specks:
            e = max(0.0, math.sin(TAU * (d["k"] * t + d["ph"]))) ** 4
            c.dot(d["x"], d["y"], 0.9 * u, 0.7 * e)
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.5), (4 * u, 0.4), (10 * u, 0.3)),
                            exposure=1.0 * intensity, alpha_gamma=0.85))
    return out, fps


@effect("electric_crackle", "electric", "minor",
        "Small erratic electric arcs jumping around a point, flickering", loop=True)
def fx_electric_crackle(rng, palette, scale=1.0, frames=24, fps=16, intensity=1.0, arcs=40,
                        reach=1.0, roughness=0.22, hop_chance=0.4, branch_chance=0.7,
                        micro_arcs=3, width=1.0, core=1.0, flicker=1.0, glow=1.0):
    """Arc events are scheduled on a circular timeline (start frame mod N),
    each holding a coarse skeleton re-jittered with `refine` every frame.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: loop length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      arcs: number of arc events across the loop (density)
      reach: arc length / radius multiplier
      roughness: jaggedness of the arcs
      hop_chance: fraction of arcs that hop around the core instead of shooting out
      branch_chance: chance an arc grows a small fork
      micro_arcs: tiny arcs hugging the core every frame
      width: stroke width multiplier
      core: brightness of the central spark
      flicker: core flicker amount
      glow: bloom strength
    """
    S = int(256 * scale)
    u = S / 256
    cx = cy = S / 2
    rch = min(reach, 1.25)
    events = []
    for i in range(arcs):
        start = int(rng.integers(0, frames))
        dur = int(rng.choice([1, 2, 2, 3]))
        if rng.random() < 1 - hop_chance:   # radial arc from the core
            a = rng.uniform(0, TAU)
            L = rng.uniform(45, 92) * u * rch
            bend = rng.uniform(-0.5, 0.5)
            pts = [(cx + 5 * u * math.cos(a), cy + 5 * u * math.sin(a)),
                   (cx + L * 0.5 * math.cos(a + bend), cy + L * 0.5 * math.sin(a + bend)),
                   (cx + L * math.cos(a + bend * 0.4), cy + L * math.sin(a + bend * 0.4))]
        else:                    # arc hopping around the core
            a0 = rng.uniform(0, TAU)
            span = rng.uniform(0.7, 1.7) * rng.choice([-1, 1])
            r0, r1 = rng.uniform(25, 60) * u * rch, rng.uniform(25, 70) * u * rch
            aa = np.linspace(a0, a0 + span, 5)
            rr = np.linspace(r0, r1, 5) + np.sin(np.linspace(0, math.pi, 5)) * rng.uniform(5, 18) * u * rch
            pts = list(zip(cx + rr * np.cos(aa), cy + rr * np.sin(aa)))
        events.append(dict(start=start, dur=dur, pts=np.array(pts), seed=int(rng.integers(1 << 31)),
                           br=rng.uniform(0.8, 1.3)))
    core_fl = 1 + (rng.uniform(0.75, 1.25, frames) - 1) * flicker
    micro_seeds = rng.integers(0, 1 << 31, frames)

    out = []
    for f in range(frames):
        c = Canvas(S, S)
        c.dot(cx, cy, 4 * u, 2.2 * core * core_fl[f])
        c.dot(cx, cy, 14 * u, 0.4 * core * core_fl[f])
        # constant micro-crackle hugging the core (fresh every frame)
        fr = np.random.default_rng(micro_seeds[f])
        for _ in range(micro_arcs):
            a = fr.uniform(0, TAU)
            L = fr.uniform(14, 30) * u * rch
            b = a + fr.uniform(-0.9, 0.9)
            pts = [(cx + 4 * u * math.cos(a), cy + 4 * u * math.sin(a)),
                   (cx + L * math.cos(b), cy + L * math.sin(b))]
            c.polyline(refine(fr, pts, rough=roughness * 0.3 / 0.22, detail=1.8 * u), 0.6 * u * width, 1.6,
                       r_end=0.35 * u * width, a_end=0.4)
        for ev in events:
            age = (f - ev["start"]) % frames
            if age >= ev["dur"]:
                continue
            fr = np.random.default_rng(ev["seed"] + f)
            e = ev["br"] * (1.0 if age == 0 else 0.6)
            path = refine(fr, ev["pts"], rough=roughness, detail=2.2 * u)
            c.polyline(path, 0.9 * u * width, 2.6 * e, r_end=0.55 * u * width, a_end=1.6 * e)
            c.dot(*path[-1], 2.5 * u, 0.9 * e)
            # small branch
            if len(path) > 8 and fr.random() < branch_chance:
                i = int(fr.integers(len(path) // 4, 3 * len(path) // 4))
                d = path[min(i + 2, len(path) - 1)] - path[i - 1]
                d = d / (np.linalg.norm(d) + 1e-9)
                ang = fr.choice([-1, 1]) * fr.uniform(0.5, 1.0)
                ca, sa = math.cos(ang), math.sin(ang)
                q = path[i] + np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]]) * fr.uniform(12, 26) * u
                bp = refine(fr, [path[i], q], rough=roughness * 0.25 / 0.22, detail=2.0 * u)
                c.polyline(bp, 0.5 * u * width, 1.3 * e, r_end=0.3 * u * width, a_end=0.2 * e)
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.6), (4 * u, 0.7), (12 * u, 0.6)),
                            exposure=1.0 * intensity))
    return out, fps


@effect("pickup_burst", "holy", "minor",
        "Collect / level-up pop: flash, expanding ring, star glints flying out", loop=False)
def fx_pickup_burst(rng, palette, scale=1.0, frames=16, fps=20, intensity=1.0, stars=7, star_size=1.0,
                    bits=22, rings=2, ring_radius=1.0, flash=1.0, center_glint=True, glow=1.0):
    """Quick flash, eased rings, glints launched on ease-out radial paths
    while shrinking and spinning; global fade to zero on the last frame.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: animation length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      stars: number of star glints flying out
      star_size: size multiplier of the flying glints
      bits: number of tiny sparks flying out
      rings: number of expanding rings (0-3)
      ring_radius: how far the rings and glints travel (multiplier)
      flash: brightness of the initial flash
      center_glint: big cross glint at the centre
      glow: bloom strength
    """
    S = int(256 * scale)
    u = S / 256
    cx = cy = S / 2
    rr = min(ring_radius, 1.15)
    star_list = [dict(ang=k * TAU / max(stars, 1) + rng.uniform(-0.25, 0.25), D=rng.uniform(62, 90) * u * rr,
                      L=rng.uniform(15, 22) * u * star_size, spin=rng.uniform(-1.5, 1.5)) for k in range(stars)]
    bit_list = [dict(ang=rng.uniform(0, TAU), D=rng.uniform(40, 100) * u * rr, br=rng.uniform(0.5, 1.1))
                for _ in range(bits)]
    ring_specs = [(0.0, 0.6, 1.4, 1.9, 92), (0.1, 0.65, 0.9, 1.1, 70), (0.2, 0.6, 0.8, 0.8, 52)][:max(0, rings)]
    out = []
    for f in range(frames):
        t = f / (frames - 1)
        c = Canvas(S, S)
        if f < frames - 1:
            fade = float(smoothstep(1.0, 0.55, t))
            fl = math.exp(-t * 15 / 1.6) * flash
            c.dot(cx, cy, (10 + 14 * t) * u, 2.6 * fl)
            c.dot(cx, cy, 34 * u, 0.6 * fl)
            if center_glint:
                _glint(c, cx, cy, (46 - 30 * t) * u, 1.6 * fl + 0.25 * fade, rot=0.0, w=1.6 * u)
            for delay, span, wd, br, travel in ring_specs:
                rt = (t - delay) / span
                if 0 <= rt < 1:
                    R = (8 + travel * rr * float(_ease_out(rt, 3))) * u
                    c.ring(cx, cy, R, (wd + 1.2 * rt) * u, br * (1 - rt) ** 2)
            e = float(_ease_out(t / 0.8, 3))
            for s in star_list:
                d = s["D"] * e
                x, y = cx + d * math.cos(s["ang"]), cy + d * math.sin(s["ang"])
                _glint(c, x, y, s["L"] * (1 - 0.55 * t), 1.5 * fade, rot=s["spin"] * t, w=0.9 * u)
            for b in bit_list:
                d = b["D"] * float(_ease_out(t / 0.7, 2))
                x, y = cx + d * math.cos(b["ang"]), cy + d * math.sin(b["ang"])
                c.dot(x, y, 1.0 * u, b["br"] * fade * (1 - t))
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.5), (4 * u, 0.45), (12 * u, 0.3)),
                            exposure=1.0 * intensity, alpha_gamma=0.75))
    return out, fps


@effect("charge_up", "arcane", "minor",
        "Energy particles streak inward to a brightening point, ending in a small pop", loop=False)
def fx_charge_up(rng, palette, scale=1.0, frames=28, fps=20, intensity=1.0, particles=64, radius=1.0,
                 swirl=1.0, trail=0.45, core=1.0, converge_rings=3, pop_at=0.72, pop=True,
                 pop_sparks=14, glow=1.0):
    """Particles on accelerating (s^2) spiral paths toward the centre with
    tapered trails; converging rings; the core grows, then pops at `pop_at`.

    Params:
      scale: output size multiplier (1.0 = base size)
      frames: animation length in frames
      fps: playback rate
      intensity: overall brightness (exposure)
      particles: number of inward-streaking particles
      radius: how far out particles start (multiplier)
      swirl: how much the particles spiral on the way in
      trail: trail length of each particle (fraction of its path)
      core: brightness of the charging core
      converge_rings: number of faint rings contracting inward
      pop_at: fraction of the animation when it pops
      pop: end with a pop (False = the charge just fades out)
      pop_sparks: sparks thrown out by the pop
      glow: bloom strength
    """
    S = int(256 * scale)
    u = S / 256
    cx = cy = S / 2
    T_POP = float(np.clip(pop_at, 0.3, 0.92))
    rad = min(radius, 1.2)
    parts = [dict(t0=rng.uniform(-0.3, T_POP - 0.28), D=rng.uniform(0.22, 0.34),
                  r0=rng.uniform(70, 104) * u * rad, a0=rng.uniform(0, TAU), sw=rng.uniform(0.6, 1.4) * swirl,
                  br=rng.uniform(0.7, 1.3)) for _ in range(particles)]
    pops = [dict(ang=rng.uniform(0, TAU), D=rng.uniform(30, 70) * u) for _ in range(pop_sparks)]
    out = []
    for f in range(frames):
        t = f / (frames - 1)
        c = Canvas(S, S)
        if f == frames - 1:
            out.append(c.render(palette))
            continue
        charge = min(t / T_POP, 1.0)
        if t < T_POP:
            cc = 0.5 + 3.0 * charge ** 2
            c.dot(cx, cy, (2.5 + 4.5 * charge) * u, cc * core)
            c.dot(cx, cy, (10 + 14 * charge) * u, (0.12 + 0.45 * charge ** 2) * core)
            for p in parts:
                s = (t - p["t0"]) / p["D"]
                if not 0 <= s < 1:
                    continue
                ss = np.linspace(max(s - trail, 0), s, 12)
                rr = p["r0"] * (1 - ss ** 2)
                aa = p["a0"] + p["sw"] * ss ** 2
                pts = np.stack([cx + rr * np.cos(aa), cy + rr * np.sin(aa)], 1)
                a = p["br"] * smoothstep(0, 0.15, s) * (0.6 + 0.6 * s)
                c.polyline(pts, 0.25 * u, 0.0, r_end=0.9 * u, a_end=a * 1.5)
            for k in range(converge_rings):
                rt = (charge * 3 - k * 0.8)
                if 0 <= rt < 1:
                    c.ring(cx, cy, (84 * rad - 74 * rt) * u, 1.0 * u, 0.6 * math.sin(math.pi * rt) ** 2)
        else:
            pt = (t - T_POP) / (1 - T_POP)        # 0..1 after the charge
            fade = (1 - pt) ** 1.5
            if not pop:
                c.dot(cx, cy, 7 * u * (1 - 0.6 * pt), 3.5 * core * fade)
                c.dot(cx, cy, 24 * u, 0.57 * core * fade)
            else:
                c.dot(cx, cy, (6 + 10 * pt) * u, 3.0 * core * math.exp(-pt * 5))
                c.dot(cx, cy, 28 * u, 0.6 * core * math.exp(-pt * 4))
                R = (8 + 70 * float(_ease_out(pt, 3))) * u
                c.ring(cx, cy, R, (1.6 + 3 * pt) * u, 1.6 * fade)
                for p in pops:
                    d = p["D"] * float(_ease_out(pt, 2))
                    d0 = max(d - 14 * u * (1 - pt), 0)
                    ca, sa = math.cos(p["ang"]), math.sin(p["ang"])
                    c.polyline([(cx + d0 * ca, cy + d0 * sa), (cx + d * ca, cy + d * sa)],
                               0.25 * u, 0.0, r_end=0.8 * u, a_end=1.4 * fade)
        out.append(c.render(palette, bloom=_bloom(glow, (1.2 * u, 0.5), (4 * u, 0.5), (12 * u, 0.45)),
                            exposure=1.0 * intensity, alpha_gamma=0.85))
    return out, fps
