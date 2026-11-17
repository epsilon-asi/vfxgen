"""
Rendering core shared by every effect.

Model
-----
A Canvas has two float buffers at SS x resolution:

  buf   emitted LIGHT (fire, energy, sparks, glow). Additive: overlaps get hotter.
  dens  smoke DENSITY (smoke, dust, steam, poison). Non-emissive, lit from a light
        direction with a cheap self-shadow.

Canvas.render() blooms the light, downsamples both, tone-maps light through a
palette ramp (intensity -> colour AND alpha: hot = white/opaque, faint =
saturated/transparent), shades smoke through a second ramp, then composites
light OVER smoke. Result: straight-alpha RGBA uint8, transparent where empty.

All coordinates/sizes passed to drawing calls are in OUTPUT pixels.
"""
import inspect
import math

import numpy as np

SS = 2           # supersampling factor
TAU = 2 * math.pi

# ---------------------------------------------------------------------------
# Palettes: (position 0..1, (r, g, b)).
# Emissive palettes index by tone-mapped brightness; smoke palettes by lighting.
# Effect modules may add their own entries.
# ---------------------------------------------------------------------------
PALETTES = {
    # emissive
    "electric": [(0.0, (40, 70, 255)), (0.35, (80, 140, 255)), (0.7, (175, 215, 255)), (1.0, (255, 255, 255))],
    "fire":     [(0.0, (170, 25, 0)), (0.35, (245, 95, 10)), (0.7, (255, 195, 70)), (1.0, (255, 255, 235))],
    "arcane":   [(0.0, (100, 25, 210)), (0.4, (190, 70, 255)), (0.75, (255, 165, 255)), (1.0, (255, 255, 255))],
    "toxic":    [(0.0, (20, 130, 30)), (0.4, (90, 225, 50)), (0.75, (205, 255, 130)), (1.0, (255, 255, 235))],
    "frost":    [(0.0, (40, 150, 200)), (0.4, (110, 210, 240)), (0.75, (200, 245, 255)), (1.0, (255, 255, 255))],
    "holy":     [(0.0, (200, 140, 30)), (0.4, (255, 205, 80)), (0.75, (255, 240, 180)), (1.0, (255, 255, 255))],
    "heal":     [(0.0, (20, 170, 110)), (0.4, (70, 230, 150)), (0.75, (190, 255, 215)), (1.0, (255, 255, 255))],
    "blood":    [(0.0, (120, 0, 10)), (0.4, (210, 20, 30)), (0.75, (255, 110, 100)), (1.0, (255, 230, 220))],
    "void":     [(0.0, (60, 10, 120)), (0.4, (130, 30, 200)), (0.75, (230, 120, 255)), (1.0, (255, 240, 255))],
    "wind":     [(0.0, (125, 165, 185)), (0.5, (200, 228, 240)), (1.0, (255, 255, 255))],
    # smoke (index = lighting: 0 shadowed, 1 lit)
    "smoke":      [(0.0, (62, 62, 68)), (0.5, (128, 128, 134)), (1.0, (214, 214, 220))],
    "dark_smoke": [(0.0, (22, 21, 23)), (0.5, (52, 50, 52)), (1.0, (110, 106, 104))],
    "dust":       [(0.0, (92, 72, 52)), (0.5, (160, 132, 100)), (1.0, (222, 200, 165))],
    "steam":      [(0.0, (150, 160, 170)), (0.5, (205, 212, 220)), (1.0, (250, 252, 255))],
    "poison_gas": [(0.0, (30, 70, 20)), (0.5, (80, 150, 40)), (1.0, (170, 230, 110))],
}


def _hex(c):
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise ValueError(f"bad hex colour {c!r}")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def get_palette(spec):
    """Resolve a colour spec to ramp stops. Accepts:
      - a preset name:            "fire", "smoke", ...
      - one hex colour:           "#ff3300"  -> auto ramp: deep -> colour -> pale -> white
      - a list of hex colours:    "#300,#f30,#fc6,#fff"  (evenly spaced, dark/faint first)
      - explicit positions:       "0:#300,0.4:#f30,1:#fff"
      - a list of stops already:  [(0.0, (r, g, b)), ...]
    """
    if isinstance(spec, (list, tuple)):
        return list(spec)
    spec = str(spec).strip()
    if spec in PALETTES:
        return PALETTES[spec]
    parts = [p for p in spec.split(",") if p.strip()]
    if not parts or not all("#" in p or len(p.split(":")[-1].strip()) in (3, 6) for p in parts):
        raise ValueError(f"unknown palette {spec!r}; use a preset ({', '.join(sorted(PALETTES))}) "
                         "or hex colours like '#ff3300' or '#300,#f30,#fff'")
    if len(parts) == 1:
        c = np.array(_hex(parts[0]), np.float32)
        deep = c * 0.7
        pale = c + (255 - c) * 0.6
        return [(0.0, tuple(deep)), (0.4, tuple(c)), (0.75, tuple(pale)), (1.0, (255, 255, 255))]
    stops = []
    for i, p in enumerate(parts):
        if ":" in p:
            pos, col = p.split(":", 1)
            stops.append((float(pos), _hex(col)))
        else:
            stops.append((i / (len(parts) - 1), _hex(p)))
    return sorted(stops)


def ramp(stops, t):
    stops = get_palette(stops)
    pos = np.array([s[0] for s in stops])
    cols = np.array([s[1] for s in stops], np.float32)
    return np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------------
# Effect registry
# ---------------------------------------------------------------------------
EFFECTS = {}


def effect(name, palette, tier, desc, loop):
    """Register an effect.

    fn(rng, palette, scale=1.0, frames=N, **params) -> (frames, fps)

    Every keyword argument after `rng, palette` is a user-tweakable parameter
    (exposed as `--set name=value` and listed by `--params NAME`). Document
    each one in the function docstring under a "Params:" block, one per line:
        Params:
          count: number of sparks
          smoke_palette: colour ramp for the smoke (preset or hex list)
    Any parameter whose name ends in `palette` accepts preset names or hex
    colours (see get_palette).

    tier: "minor" (small, quick: hits, puffs, glints) or "major" (set pieces).
    loop: True if the animation loops seamlessly, False for one-shots."""
    def deco(fn):
        sig = inspect.signature(fn)
        params = {k: v.default for k, v in list(sig.parameters.items())[2:]
                  if v.default is not inspect.Parameter.empty}
        docs = {}
        doc = inspect.getdoc(fn) or ""
        if "Params:" in doc:
            for line in doc.split("Params:", 1)[1].splitlines():
                k, _, v = line.strip().partition(":")
                if k.strip() in params:
                    docs[k.strip()] = v.strip()
        EFFECTS[name] = dict(fn=fn, palette=palette, tier=tier, desc=desc, loop=loop,
                             params=params, param_docs=docs)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Image ops
# ---------------------------------------------------------------------------
_grid_cache = {}


def grid(w, h):
    """Pixel-centre coords of the supersampled buffer, in output pixels."""
    key = (w, h)
    if key not in _grid_cache:
        ys, xs = np.mgrid[0:h * SS, 0:w * SS].astype(np.float32)
        _grid_cache[key] = ((xs + 0.5) / SS, (ys + 0.5) / SS)
    return _grid_cache[key]


def _box1d(a, r, axis):
    k = 2 * r + 1
    pad = [(0, 0), (0, 0)]
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad), axis=axis, dtype=np.float64)
    if axis == 1:
        return c[:, k:] - c[:, :-k]
    return c[k:, :] - c[:-k, :]


def blur(img, sigma):
    """Gaussian blur (sigma in pixels of `img`) via three box passes."""
    r = max(1, int(round((math.sqrt(4 * sigma * sigma + 1) - 1) / 2)))
    k = 2 * r + 1
    out = img
    for _ in range(3):
        out = _box1d(_box1d(out, r, 1), r, 0) / (k * k)
    return out.astype(np.float32)


def shift(img, dx, dy):
    """Sample img at (x + dx, y + dy), integer pixels, zero outside."""
    dx, dy = int(round(dx)), int(round(dy))
    out = np.zeros_like(img)
    h, w = img.shape
    xs0, xs1 = max(dx, 0), min(w + dx, w)
    ys0, ys1 = max(dy, 0), min(h + dy, h)
    if xs0 < xs1 and ys0 < ys1:
        out[ys0 - dy:ys1 - dy, xs0 - dx:xs1 - dx] = img[ys0:ys1, xs0:xs1]
    return out


def _edge_mask(w, h, margin=10):
    def sm(v):
        v = np.clip(v, 0, 1)
        return v * v * (3 - 2 * v)
    x = sm(np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / margin)
    y = sm(np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / margin)
    return np.outer(y, x).astype(np.float32)


def refine(rng, pts, rough, detail):
    """Midpoint displacement: subdivide until segments <= detail px, offsetting
    each new midpoint sideways by ~rough * segment length. Existing points are
    kept, so a coarse path keeps its shape. Great for lightning / cracks."""
    pts = np.asarray(pts, np.float64)
    while True:
        d = pts[1:] - pts[:-1]
        seglen = np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        if seglen.max() <= detail:
            return pts
        nrm = np.hstack([-d[:, 1:2], d[:, 0:1]]) / seglen
        off = np.clip(rng.normal(0, rough, (len(d), 1)), -2, 2) * seglen
        mid = (pts[:-1] + pts[1:]) / 2 + nrm * off
        out = np.empty((len(pts) + len(mid), 2))
        out[0::2] = pts
        out[1::2] = mid
        pts = out


# ---------------------------------------------------------------------------
# Noise: periodic 3D gradient (Perlin) noise. Make loops seamless by sampling
# z = t * period_z with t in [0, 1): every octave is an integer multiple of the
# period, so frame N wraps to frame 0 exactly.
# ---------------------------------------------------------------------------
class Noise:
    def __init__(self, rng, period=(32, 32, 8)):
        self.px, self.py, self.pz = period
        g = rng.normal(size=(self.pz, self.py, self.px, 3)).astype(np.float32)
        self.g = g / np.linalg.norm(g, axis=-1, keepdims=True)

    def __call__(self, x, y, z=0.0):
        """Single octave, roughly in [-1, 1]. x, y, z broadcastable arrays."""
        x = np.asarray(x, np.float32)
        y = np.asarray(y, np.float32)
        z = np.broadcast_to(np.asarray(z, np.float32), np.broadcast(x, y).shape)
        x, y = np.broadcast_arrays(x, y)
        xi, yi, zi = np.floor(x), np.floor(y), np.floor(z)
        fx, fy, fz = x - xi, y - yi, z - zi
        xi = xi.astype(np.int64)
        yi = yi.astype(np.int64)
        zi = zi.astype(np.int64)

        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        u, v, w = fade(fx), fade(fy), fade(fz)
        res = 0.0
        for dz in (0, 1):
            wz = w if dz else 1 - w
            for dy in (0, 1):
                wy = v if dy else 1 - v
                for dx in (0, 1):
                    wx = u if dx else 1 - u
                    gi = self.g[(zi + dz) % self.pz, (yi + dy) % self.py, (xi + dx) % self.px]
                    dot = gi[..., 0] * (fx - dx) + gi[..., 1] * (fy - dy) + gi[..., 2] * (fz - dz)
                    res = res + wx * wy * wz * dot
        return res * 1.6

    def fbm(self, x, y, z=0.0, octaves=4, lacunarity=2.0, gain=0.5):
        """Fractal sum, normalised to roughly [-1, 1]. Keep lacunarity an
        integer if you need the time axis (z) to loop."""
        total, amp, norm, f = 0.0, 1.0, 0.0, 1.0
        for o in range(octaves):
            total = total + amp * self(x * f + o * 17.3, y * f + o * 9.1, np.asarray(z) * f)
            norm += amp
            amp *= gain
            f *= lacunarity
        return total / norm


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
class Canvas:
    def __init__(self, w, h):
        self.w, self.h = int(w), int(h)
        self.X, self.Y = grid(self.w, self.h)
        self.buf = np.zeros(self.X.shape, np.float32)   # light
        self.dens = np.zeros(self.X.shape, np.float32)  # smoke density

    # -- helpers ------------------------------------------------------------
    def _bbox(self, xmin, xmax, ymin, ymax):
        x0 = max(int(xmin * SS), 0)
        x1 = min(int(xmax * SS) + 2, self.buf.shape[1])
        y0 = max(int(ymin * SS), 0)
        y1 = min(int(ymax * SS) + 2, self.buf.shape[0])
        return x0, x1, y0, y1

    # -- light primitives ---------------------------------------------------
    def dot(self, x, y, r, a, target=None):
        """Soft gaussian blob, sigma r, peak a. target='dens' paints smoke."""
        buf = self.dens if target == "dens" else self.buf
        R = 3 * r
        x0, x1, y0, y1 = self._bbox(x - R, x + R, y - R, y + R)
        if x0 >= x1 or y0 >= y1:
            return
        dx = self.X[y0:y1, x0:x1] - x
        dy = self.Y[y0:y1, x0:x1] - y
        buf[y0:y1, x0:x1] += a * np.exp(-(dx * dx + dy * dy) / (2 * r * r))

    def ring(self, cx, cy, R, width, a, squash=1.0, target=None):
        """Gaussian-profile ring (ellipse if squash < 1 for ground-plane rings)."""
        buf = self.dens if target == "dens" else self.buf
        m = 3 * width
        x0, x1, y0, y1 = self._bbox(cx - R - m, cx + R + m, cy - R * squash - m, cy + R * squash + m)
        if x0 >= x1 or y0 >= y1:
            return
        dx = self.X[y0:y1, x0:x1] - cx
        dy = (self.Y[y0:y1, x0:x1] - cy) / max(squash, 1e-3)
        d = np.hypot(dx, dy) - R
        buf[y0:y1, x0:x1] += a * np.exp(-(d * d) / (2 * width * width))

    def polyline(self, pts, r, a, r_end=None, a_end=None, target=None):
        """Gaussian-profile stroke. Width/intensity taper linearly start->end.
        Segments combine with max() so joints don't bead."""
        buf = self.dens if target == "dens" else self.buf
        pts = np.asarray(pts, np.float32)
        n = len(pts)
        if n < 2:
            return
        rs = np.linspace(r, r if r_end is None else r_end, n)
        As = np.linspace(a, a if a_end is None else a_end, n)
        R = 3 * rs.max()
        x0, x1, y0, y1 = self._bbox(pts[:, 0].min() - R, pts[:, 0].max() + R,
                                    pts[:, 1].min() - R, pts[:, 1].max() + R)
        if x0 >= x1 or y0 >= y1:
            return
        acc = np.zeros((y1 - y0, x1 - x0), np.float32)
        for i in range(n - 1):
            xa, ya = pts[i]
            xb, yb = pts[i + 1]
            Ri = 3 * max(rs[i], rs[i + 1])
            sx0, sx1, sy0, sy1 = self._bbox(min(xa, xb) - Ri, max(xa, xb) + Ri,
                                            min(ya, yb) - Ri, max(ya, yb) + Ri)
            sx0, sx1 = max(sx0, x0), min(sx1, x1)
            sy0, sy1 = max(sy0, y0), min(sy1, y1)
            if sx0 >= sx1 or sy0 >= sy1:
                continue
            px = self.X[sy0:sy1, sx0:sx1] - xa
            py = self.Y[sy0:sy1, sx0:sx1] - ya
            vx, vy = xb - xa, yb - ya
            t = np.clip((px * vx + py * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
            dx, dy = px - t * vx, py - t * vy
            rr = rs[i] + (rs[i + 1] - rs[i]) * t
            amp = As[i] + (As[i + 1] - As[i]) * t
            val = amp * np.exp(-(dx * dx + dy * dy) / (2 * rr * rr))
            sub = acc[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
            np.maximum(sub, val, out=sub)
        buf[y0:y1, x0:x1] += acc

    def polygon(self, pts, a, soft=0.7, target=None):
        """Filled convex-or-not polygon (even-odd), edge softened by `soft` px.
        Good for shards, crystals, slash crescents, rune glyph fills."""
        buf = self.dens if target == "dens" else self.buf
        pts = np.asarray(pts, np.float32)
        m = 3 * soft + 1
        x0, x1, y0, y1 = self._bbox(pts[:, 0].min() - m, pts[:, 0].max() + m,
                                    pts[:, 1].min() - m, pts[:, 1].max() + m)
        if x0 >= x1 or y0 >= y1:
            return
        X = self.X[y0:y1, x0:x1]
        Y = self.Y[y0:y1, x0:x1]
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
            d = np.hypot(X - xa - t * vx, Y - ya - t * vy)
            np.minimum(dmin, d, out=dmin)
        sd = np.where(inside, -dmin, dmin)
        buf[y0:y1, x0:x1] += a * smoothstep(soft, -soft, sd)

    # -- output -------------------------------------------------------------
    def render(self, palette, bloom=((1.5, 0.6), (5, 0.5), (14, 0.35)),
               exposure=1.0, alpha_gamma=0.85,
               smoke=None, smoke_opacity=1.0, light_dir=(-0.55, -0.85),
               shadow=1.2, fire_light=0.0):
        """Light + smoke -> straight-alpha RGBA uint8.

        palette        emissive ramp for `buf`
        bloom          ((sigma_px, weight), ...) glow passes; () for none
        exposure       light gain before tone-map
        alpha_gamma    <1 = glow more opaque, >1 = more see-through
        smoke          smoke ramp name for `dens` (None = ignore dens)
        smoke_opacity  density -> alpha gain
        light_dir      unit vector pointing TOWARD the light (screen coords, y down)
        shadow         self-shadow strength for smoke
        fire_light     how much the fire's glow tints nearby smoke (0..~3)
        """
        w, h = self.w, self.h
        size = max(w, h)
        mask = _edge_mask(w, h)

        def down(a):
            return a.reshape(h, SS, w, SS).mean(axis=(1, 3))

        I = self.buf
        if bloom and I.any():
            I = I + sum(wt * blur(I, s * SS) for s, wt in bloom)
        I = down(I) * mask
        t = 1.0 - np.exp(-np.maximum(I, 0) * exposure)
        e_rgb = ramp(palette, t)
        e_a = t ** alpha_gamma

        if smoke and self.dens.any():
            D = np.maximum(down(self.dens), 0) * mask
            s_a = 1.0 - np.exp(-D * smoke_opacity)
            Db = blur(D, max(1.5, size / 90))
            off = size / 40
            occl = shift(Db, light_dir[0] * off, light_dir[1] * off) \
                + 0.5 * shift(Db, light_dir[0] * off * 2.2, light_dir[1] * off * 2.2)
            lit = np.exp(-shadow * occl)
            s_rgb = ramp(smoke, lit)
            if fire_light > 0 and I.any():
                gl = 1 - np.exp(-blur(I, size / 25) * fire_light)
                tint = np.array(ramp(palette, np.array(0.55)), np.float32)
                s_rgb = s_rgb * (1 - gl[..., None]) + (tint * 0.6 + s_rgb * 0.4) * gl[..., None] \
                    + tint * 0.25 * gl[..., None]
            # light OVER smoke (premultiplied)
            a = e_a + s_a * (1 - e_a)
            prem = e_rgb * e_a[..., None] + s_rgb * (s_a * (1 - e_a))[..., None]
            rgb = prem / np.maximum(a, 1e-6)[..., None]
        else:
            a, rgb = e_a, e_rgb

        a = a.copy()
        a[a < 1.5 / 255] = 0
        rgb[a == 0] = 0
        out = np.dstack([rgb, a * 255])
        return np.clip(out + 0.5, 0, 255).astype(np.uint8)
