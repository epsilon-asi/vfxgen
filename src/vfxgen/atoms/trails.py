"""Trail and link atoms: ``ribbon`` (a tapering strip that follows a moving layer or a path) and
``chain`` (lightning / tether / life-drain link between two targets).

Both atoms are loops (``period``) so they can sit in looping scenes; in one-shot scenes a
``follow`` ribbon ends by itself once its leader is gone, and a ``path`` ribbon in ``sweep`` mode
is empty at t = 0 and t = period (use ``time_mode: once`` on the layer).

Strips are drawn as the union of tapered capsules (minimum signed distance over segments), so
joints never bead or gap; colour / alpha along the strip come from the nearest segment.
"""
from __future__ import annotations

import math

import numpy as np

from ..atom import Atom, Light, P, Smoke, atom
from ..engine.mathx import TAU, refine, smoothstep
from ..engine.noise import Noise


# =============================================================================
# shared helpers
# =============================================================================
def _catmull(pts: np.ndarray, k: int) -> np.ndarray:
    """Centripetal-free (uniform) Catmull-Rom subdivision: ``k`` sub-steps per segment."""
    n = len(pts)
    if n < 3 or k <= 1:
        return pts
    P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    u = (np.arange(k) / k)[:, None]
    u2, u3 = u * u, u * u * u
    out = []
    for i in range(1, n):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    out.append(pts[-1:])
    return np.vstack(out)


def _resample(arr: np.ndarray, n_in: int, k: int) -> np.ndarray:
    """Linear resample of a per-vertex array to match :func:`_catmull` output length."""
    if n_in < 3 or k <= 1:
        return arr
    x = np.arange(n_in, dtype=np.float64)
    xi = np.concatenate([np.arange(i, i + 1, 1.0 / k) for i in range(n_in - 1)] + [[n_in - 1]])
    return np.interp(xi, x, arr)


def _normals(pts: np.ndarray) -> np.ndarray:
    d = np.gradient(pts, axis=0) if len(pts) > 1 else np.array([[1.0, 0.0]])
    ln = np.hypot(d[:, 0], d[:, 1])[:, None]
    d = d / np.maximum(ln, 1e-9)
    return np.stack([-d[:, 1], d[:, 0]], 1)


def draw_strip(view, pts, half_w, alpha, s=None, *, softness: float = 0.6, core: float = 0.0,
               core_width: float = 0.3, texture=None) -> None:
    """Deposit a smooth strip through ``pts`` (n, 2) with per-vertex half widths and alphas.

    The strip is the union of tapered capsules (min signed distance), so it has no beads at
    joints and no gaps. ``softness`` 0 = hard anti-aliased edge, 1 = soft falloff from the centre
    line. ``core`` adds a hot gaussian centre line (``core_width`` x half width).
    ``texture(S, A)`` -> multiplier, with S = position along the strip (``s``, default 0..1 from
    the first vertex) and A = signed distance across in half widths (-1..1)."""
    pts = np.asarray(pts, np.float64)
    n = len(pts)
    if n < 2:
        return
    half_w = np.maximum(np.asarray(half_w, np.float64), 1e-3)
    alpha = np.asarray(alpha, np.float64)
    if not np.any(alpha > 1e-4):
        return
    s = np.linspace(0, 1, n) if s is None else np.asarray(s, np.float64)
    aa = 0.8 / max(view.px, 1e-6)                                   # anti-aliasing (local units)
    R = float(half_w.max()) + 2 * aa
    win = view.window(float(pts[:, 0].min()) - R, float(pts[:, 1].min()) - R,
                      float(pts[:, 0].max()) + R, float(pts[:, 1].max()) + R)
    if win is None:
        return
    ys, xs, X, Y = win
    hh, ww = X.shape
    best = np.full(X.shape, np.inf, np.float32)      # signed distance to the strip edge
    bS = np.zeros(X.shape, np.float32)
    bA = np.zeros(X.shape, np.float32)
    bW = np.ones(X.shape, np.float32)
    bD = np.zeros(X.shape, np.float32)
    bX = np.zeros(X.shape, np.float32)
    ss = view.ss
    for i in range(n - 1):
        if alpha[i] <= 1e-4 and alpha[i + 1] <= 1e-4:
            continue
        xa, ya = pts[i]
        xb, yb = pts[i + 1]
        Ri = max(half_w[i], half_w[i + 1]) + 2 * aa
        bx0, by0, bx1, by1 = view.xf.bbox(min(xa, xb) - Ri, min(ya, yb) - Ri, max(xa, xb) + Ri, max(ya, yb) + Ri)
        c0 = max(int(np.floor(bx0 * ss)) - 1 - xs.start, 0)
        r0 = max(int(np.floor(by0 * ss)) - 1 - ys.start, 0)
        c1 = min(int(np.ceil(bx1 * ss)) + 2 - xs.start, ww)
        r1 = min(int(np.ceil(by1 * ss)) + 2 - ys.start, hh)
        if c0 >= c1 or r0 >= r1:
            continue
        px = X[r0:r1, c0:c1] - xa
        py = Y[r0:r1, c0:c1] - ya
        vx, vy = xb - xa, yb - ya
        L2 = vx * vx + vy * vy + 1e-9
        t = np.clip((px * vx + py * vy) / L2, 0, 1)
        dx, dy = px - t * vx, py - t * vy
        d = np.hypot(dx, dy)
        w = half_w[i] + (half_w[i + 1] - half_w[i]) * t
        sd = d - w
        sub = best[r0:r1, c0:c1]
        m = sd < sub
        if not m.any():
            continue
        sub[m] = sd[m]
        bS[r0:r1, c0:c1][m] = (s[i] + (s[i + 1] - s[i]) * t)[m]
        bA[r0:r1, c0:c1][m] = (alpha[i] + (alpha[i + 1] - alpha[i]) * t)[m]
        bW[r0:r1, c0:c1][m] = w[m]
        bD[r0:r1, c0:c1][m] = d[m]
        side = np.sign(vx * py - vy * px)
        bX[r0:r1, c0:c1][m] = (side * d / np.maximum(w, 1e-6))[m]
    inside = np.isfinite(best)
    if not inside.any():
        return
    best = np.where(inside, best, 1e9)
    soft_w = np.maximum(softness * bW, aa)
    val = smoothstep(aa, -soft_w, best)
    if core > 0:
        cw = np.maximum(core_width * bW, aa * 0.75)
        val = val + core * np.exp(-0.5 * (bD / cw) ** 2) * (best < 2 * aa)
    val = val * bA
    if texture is not None:
        val = val * np.clip(texture(bS, np.clip(bX, -1.5, 1.5)), 0, None)
    view._deposit(ys, xs, np.asarray(val, np.float32))


# =============================================================================
# ribbon
# =============================================================================
_RIB_OUT = ["light", "smoke"]


def _ribbon_head(self, t):
    if self.p.mode == "follow":
        return (0.0, 0.0)
    pts = self._path_points(t)[0]
    return (float(pts[0, 0]), float(pts[0, 1])) if len(pts) else tuple(self._path_at(np.array([1.0]))[0])


@atom("ribbon", tags=["trail", "ribbon", "light", "smoke", "loop"])
class Ribbon(Atom):
    """A strip trailing behind a moving head, its width and alpha tapering along its length.

    * ``mode: follow``: traces where the ``follow:`` layer's anchor has been over the last
      ``length_time`` seconds (``ctx.follow(dt)`` at ``samples`` steps, Catmull-Rom smoothed):
      sword trails, dash streaks, comet tails. Place it with ``anchor: {px: [0, 0]}``; the trail
      may go anywhere inside ``reach`` px of its origin. In LOOPING scenes set ``wrap`` to the
      scene duration so the trail continues across the loop seam.
    * ``mode: path`` (or follow without a ``follow:`` link): a parametric ``line`` / ``arc`` /
      ``sine`` path in local coordinates. ``head: sweep`` moves the head from the path start to
      past its end once per ``travel`` x period (empty at t = 0 and at the end: one-shot use with
      ``time_mode: once``); ``head: fixed`` keeps the head at the path END (the strip trails back
      along ``span`` of the path), for projectile tails in head-relative scenes.

    ``wobble`` bends the strip with flowing noise (more toward the tail); ``breakup`` tears it
    into streaks. ``output`` picks the light (glow) or smoke (dust / vapour) role.
    Anchor ``head`` = head position (path mode; origin in follow mode)."""
    kind = "loop"
    outputs = {"light": Light(palette="steel", bloom=[[1.5, 0.45], [5, 0.4], [14, 0.3]], exposure=1.1,
                              alpha_gamma=0.85),
               "smoke": Smoke(palette="dust", density=1.0, shadow=0.4)}
    streams = ("noise",)
    anchors = {"head": _ribbon_head}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s); path sweep / flow cycles are per period")
        mode: str = P("follow", doc="follow (trail of the `follow:` layer) or path (parametric path)",
                      choices=["follow", "path"])
        # follow
        length_time: float = P(0.15, gt=0, doc="follow: how many seconds of the leader's motion the trail covers")
        samples: int = P(24, ge=2, le=200, doc="points along the ribbon (follow: time samples)")
        smooth: int = P(3, ge=1, le=8, doc="Catmull-Rom sub-steps per sample (follow mode smoothing)")
        min_speed: float = P(0.0, ge=0, doc="follow: parts of the trail slower than this fade out (px/s; 0 = off)")
        reach: float = P(200.0, gt=0, doc="follow: half size of the box around the origin the trail may use (bounds)")
        wrap: float = P(0.0, ge=0, doc="follow: scene loop length (s) so the trail wraps across the loop seam (0 = off)")
        # path
        path: str = P("arc", doc="path mode: path shape", choices=["line", "arc", "sine"])
        head: str = P("sweep", doc="path mode: sweep (head travels the path) or fixed (head at the path end)",
                      choices=["sweep", "fixed"])
        x0: float = P(-100.0, doc="line / sine path start x (px)")
        y0: float = P(0.0, doc="line / sine path start y (px)")
        x1: float = P(100.0, doc="line / sine path end x (px)")
        y1: float = P(0.0, doc="line / sine path end y (px)")
        radius: float = P(90.0, gt=0, doc="arc path radius (px), centred on the origin")
        arc_from: float = P(200.0, doc="arc path start angle (deg, clockwise from +x)")
        arc_to: float = P(340.0, doc="arc path end angle (deg)")
        squash: float = P(1.0, gt=0, le=1, doc="arc path vertical squash")
        amplitude: float = P(20.0, doc="sine path amplitude (px)")
        waves: float = P(1.5, ge=0, doc="sine path: number of waves along the path")
        span: float = P(0.5, gt=0, le=1, doc="path mode: ribbon length as a fraction of the path")
        travel: float = P(0.8, gt=0, le=1, doc="sweep: fraction of the period the head needs to cross (and the tail to leave)")
        ease: float = P(1.0, gt=0, doc="sweep easing exponent of the head motion (<1 = fast start)")
        # look
        width_start: float = P(18.0, ge=0, doc="width at the head (px)")
        width_end: float = P(2.0, ge=0, doc="width at the tail (px)")
        taper: float = P(1.0, gt=0, doc="width taper exponent along the ribbon (>1 = stays wide longer)")
        brightness: float = P(1.0, ge=0, doc="intensity (light) or density (smoke) at the head")
        fade: float = P(1.5, ge=0, doc="alpha fade exponent toward the tail (0 = no fade)")
        head_fade: float = P(0.04, ge=0, le=0.5, doc="fraction of the length over which the head fades in (soft cap)")
        softness: float = P(0.7, ge=0, le=1, doc="edge softness (0 = crisp edge, 1 = soft from the centre)")
        core: float = P(0.8, ge=0, doc="hot centre line brightness (x brightness); 0 = none")
        core_width: float = P(0.25, gt=0, le=1, doc="hot centre line width (fraction of the half width)")
        wobble: float = P(0.0, ge=0, doc="noise wobble amplitude at the tail (px)")
        wobble_scale: float = P(2.0, gt=0, doc="wobble waves along the ribbon")
        breakup: float = P(0.0, ge=0, le=1, doc="noise streaks tearing the ribbon (0..1)")
        breakup_scale: float = P(4.0, gt=0, doc="breakup noise cells along the ribbon")
        flow: int = P(1, doc="noise periods the wobble / breakup flow toward the tail per period (whole number)")
        wave: float = P(0.0, doc="sine wave across the ribbon at the tail (px; 0 at the head): helix strands, wiggles")
        wave_count: float = P(2.0, ge=0, doc="sine waves along the ribbon")
        wave_speed: int = P(2, doc="wave cycles travelling toward the tail per period (whole number)")
        wave_phase: float = P(0.0, doc="wave phase offset (deg): 180 = mirrored strand")
        output: str = P("light", doc="which output role to draw into", choices=_RIB_OUT)

    def setup(self):
        p = self.p
        self.nx = max(1, int(round(max(p.wobble_scale, p.breakup_scale))) * 2)
        self.noise = Noise(self.rng("noise"), period=(self.nx, 8, 4))
        self._build_path()

    # -- path ------------------------------------------------------------------------
    def _raw_path(self, u):
        p = self.p
        if p.path == "arc":
            a = np.radians(p.arc_from + (p.arc_to - p.arc_from) * u)
            return np.stack([p.radius * np.cos(a), p.radius * np.sin(a) * p.squash], 1)
        a = np.array([p.x0, p.y0])
        b = np.array([p.x1, p.y1])
        pts = a[None] + (b - a)[None] * u[:, None]
        if p.path == "sine":
            d = b - a
            nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-9)
            pts = pts + nrm[None] * (p.amplitude * np.sin(TAU * p.waves * u))[:, None]
        return pts

    def _build_path(self):
        u = np.linspace(0, 1, 513)
        pts = self._raw_path(u)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        cum = np.concatenate([[0], np.cumsum(seg)])
        self._cum = cum / max(cum[-1], 1e-9)
        self._u = u

    def _path_at(self, q):
        """Points at arc-length fractions q (0..1)."""
        return self._raw_path(np.interp(q, self._cum, self._u))

    def _path_points(self, t):
        """(points head -> tail, s values) of the visible ribbon at time t (path mode)."""
        p = self.p
        n = p.samples * 2
        if p.head == "fixed":
            q = 1 - np.linspace(0, p.span, n)
            return self._path_at(q), np.linspace(0, 1, n)
        tn = (t % p.period) / p.period / p.travel
        if tn >= 1:
            return np.zeros((0, 2)), np.zeros(0)
        qh = tn ** p.ease * (1 + p.span)
        lo, hi = max(qh - p.span, 0.0), min(qh, 1.0)
        if hi - lo < 1e-4:
            return np.zeros((0, 2)), np.zeros(0)
        q = np.linspace(hi, lo, n)
        return self._path_at(q), (qh - q) / p.span

    def get_bounds(self):
        p = self.p
        if p.mode == "follow":
            r = p.reach
            return (-r, -r, r, r)
        pts = self._raw_path(np.linspace(0, 1, 129))
        m = max(p.width_start, p.width_end) / 2 + p.wobble + abs(p.wave) + 6
        return (float(pts[:, 0].min() - m), float(pts[:, 1].min() - m),
                float(pts[:, 0].max() + m), float(pts[:, 1].max() + m))

    # -- follow ------------------------------------------------------------------------
    def _follow_points(self, ctx):
        p = self.p
        n = p.samples
        dts = np.linspace(0, p.length_time, n)
        pts = []
        for dt in dts:
            q = ctx.follow(float(dt))
            if q is None and p.wrap > 0:
                q = ctx.follow(float(dt) - p.wrap * math.ceil((float(dt) - ctx.t_scene - 1e-9) / p.wrap))
            if q is None:
                break
            pts.append(q)
        if len(pts) < 2:
            return None
        pts = np.asarray(pts, np.float64)
        m = len(pts)
        s = dts[:m] / p.length_time
        a = np.ones(m)
        if p.min_speed > 0:
            step = p.length_time / (n - 1)
            v = np.hypot(*np.diff(pts, axis=0).T) / step
            v = np.concatenate([v, v[-1:]])
            a = smoothstep(0.0, p.min_speed, v).astype(np.float64)
        k = p.smooth
        pts2 = _catmull(pts, k)
        return pts2, _resample(s, m, k), _resample(a, m, k)

    # -- draw ------------------------------------------------------------------------------
    def draw(self, ctx, t):
        p = self.p
        view = getattr(ctx, p.output)
        follow = p.mode == "follow" and ctx.has_target("follow")
        if follow:
            r = self._follow_points(ctx)
            if r is None:
                return
            pts, s, extra = r
        else:
            pts, s = self._path_points(t)
            if len(pts) < 2:
                return
            extra = np.ones(len(pts))
        tn = (t % p.period) / p.period
        sc = np.clip(s, 0, 1)
        if p.wobble > 0:
            nrm = _normals(pts)
            # periodic: the wobble coordinate scrolls whole noise periods (nx) per loop
            wob = self.noise((sc * p.wobble_scale - p.flow * tn * self.nx) % self.nx, np.full_like(sc, 2.5),
                             tn * 4)
            pts = pts + nrm * (p.wobble * sc ** 0.8 * wob)[:, None]
        if p.wave:
            nrm = _normals(pts)
            off = p.wave * np.clip(sc / 0.35, 0, 1) ** 0.8 * np.sin(
                TAU * (p.wave_count * sc - p.wave_speed * tn) + math.radians(p.wave_phase))
            pts = pts + nrm * off[:, None]
        half = 0.5 * (p.width_start + (p.width_end - p.width_start) * sc ** p.taper)
        alpha = p.brightness * (1 - sc) ** p.fade * extra
        if p.head_fade > 0:
            alpha = alpha * (0.15 + 0.85 * smoothstep(0.0, p.head_fade, s))
        texture = None
        if p.breakup > 0:
            bk = p.breakup

            def texture(S, A):
                u = (S * p.breakup_scale - p.flow * tn * self.nx) % self.nx
                nv = self.noise.fbm(u, A * 1.3 + 4.0, tn * 4, octaves=2)
                return 1 - bk + bk * np.clip(0.55 + 1.6 * nv, 0, 1.6) * (1 - 0.35 * np.abs(A))
        draw_strip(view, pts, half, alpha, s, softness=p.softness, core=p.core, core_width=p.core_width,
                   texture=texture)


# =============================================================================
# chain
# =============================================================================
@atom("chain", tags=["chain", "lightning", "tether", "beam", "light", "loop"])
class Chain(Atom):
    """A link between two points: ``ctx.target("a")`` and ``ctx.target("b")`` (layer
    ``targets: {a: ..., b: ...}``), or the fallback points (ax, ay) / (bx, by) when there are no
    targets. Glowing flares sit on both ends.

    * ``lightning``: jagged bolt re-jagged ``tick_rate`` times per second, optional forks,
      flicker and a wide glow; several strands (``strands``) crackle at once.
    * ``tether``: smooth rope / energy line sagging by ``sag`` px, with a travelling wave and
      bright ``pulses`` running from a to b.
    * ``drain``: life-drain beam: intertwined wavy strands in a soft band, with motes flowing
      from a to b.

    Bounds: the box spanned by the fallback points plus ``margin``; with targets, set the
    fallback points so that box covers where the targets go. Anchors ``a``, ``b``, ``mid``
    (fallback points)."""
    kind = "loop"
    outputs = {"light": Light(palette="electric", bloom=[[2, 0.6], [7, 0.6], [22, 0.45]], exposure=1.0,
                              alpha_gamma=0.85)}
    streams = ("bolt", "motes", "noise")
    anchors = {"a": lambda self, t: (self.p.ax, self.p.ay), "b": lambda self, t: (self.p.bx, self.p.by),
               "mid": lambda self, t: ((self.p.ax + self.p.bx) / 2, (self.p.ay + self.p.by) / 2)}

    class Params(Atom.Params):
        period: float = P(1.0, gt=0, doc="loop length (s)")
        style: str = P("lightning", doc="lightning | tether | drain", choices=["lightning", "tether", "drain"])
        ax: float = P(-100.0, doc="fallback start x (px), used without targets; also sets the bounds")
        ay: float = P(0.0, doc="fallback start y (px)")
        bx: float = P(100.0, doc="fallback end x (px)")
        by: float = P(0.0, doc="fallback end y (px)")
        margin: float = P(60.0, ge=0, doc="bounds margin around the fallback points (px)")
        width: float = P(1.6, gt=0, doc="line width (px; lightning stroke sigma / tether and strand width)")
        brightness: float = P(1.0, ge=0, doc="line intensity")
        glow: float = P(0.35, ge=0, doc="wide soft glow along the link")
        glow_width: float = P(7.0, gt=0, doc="glow width (px)")
        flare: float = P(1.0, ge=0, doc="endpoint flare brightness (0 = none)")
        flare_radius: float = P(7.0, gt=0, doc="endpoint flare radius (px)")
        # lightning
        roughness: float = P(0.24, ge=0, doc="lightning: jaggedness")
        tick_rate: float = P(16.0, gt=0, doc="lightning: re-jag / flicker changes per second")
        strands: int = P(2, ge=1, le=6, doc="lightning: simultaneous strands; drain: intertwined strands")
        branches: int = P(3, ge=0, doc="lightning: forks per strand")
        branch_length: float = P(0.25, ge=0, le=1, doc="lightning: fork length (fraction of the link length)")
        flicker: float = P(0.4, ge=0, le=1, doc="lightning: per-tick brightness flicker")
        # tether / drain
        sag: float = P(24.0, doc="tether: droop at the middle (px, +down)")
        wave: float = P(6.0, ge=0, doc="tether / drain: wave amplitude (px)")
        waves: int = P(2, ge=0, doc="tether / drain: waves along the link")
        flow: int = P(2, ge=0, doc="tether / drain: waves and pulses travel a->b this many times per period")
        pulses: int = P(2, ge=0, doc="tether: bright pulses on the link at once")
        pulse_brightness: float = P(1.5, ge=0, doc="tether: pulse intensity")
        motes: int = P(28, ge=0, doc="drain: motes flowing from a to b")
        mote_size: float = P(1.8, gt=0, doc="drain: mote size (px)")
        band: float = P(10.0, ge=0, doc="drain: half width of the wavy band (px)")

    def setup(self):
        p = self.p
        rng = self.rng("motes")
        n = p.motes
        self.m_phase = rng.random(n)
        self.m_laps = rng.integers(1, 3, n) * max(p.flow, 1)
        self.m_off = rng.uniform(-1, 1, n)
        self.m_br = rng.uniform(0.6, 1.3, n)
        self.m_ph2 = rng.random(n)
        self.noise = Noise(self.rng("noise"), period=(8, 8, 4))

    def get_bounds(self):
        p = self.p
        m = p.margin + max(abs(p.sag), 0) + p.wave
        return (min(p.ax, p.bx) - m, min(p.ay, p.by) - m, max(p.ax, p.bx) + m, max(p.ay, p.by) + m)

    def _ends(self, ctx):
        p = self.p
        if ctx.has_target("a") and ctx.has_target("b"):
            a, b = ctx.target("a"), ctx.target("b")
            if a is None or b is None:
                return None
            return np.array(a, np.float64), np.array(b, np.float64)
        return np.array([p.ax, p.ay], np.float64), np.array([p.bx, p.by], np.float64)

    def draw(self, ctx, t):
        ends = self._ends(ctx)
        if ends is None:
            return
        a, b = ends
        p = self.p
        L = float(np.hypot(*(b - a)))
        tn = (t % p.period) / p.period
        v = ctx.light
        if L > 1e-3:
            getattr(self, "_" + p.style)(v, a, b, L, t, tn)
        if p.flare > 0:
            k = 1 + 0.25 * math.sin(TAU * 2 * tn)
            if p.style == "lightning":
                k *= 1 - 0.5 * p.flicker * float(self.tick_rng("bolt", t + 0.5 / p.tick_rate, p.tick_rate).random())
            for e, kk in ((a, 1.0), (b, 0.85 if p.style == "drain" else 1.0)):
                v.dot(float(e[0]), float(e[1]), p.flare_radius * 0.45, 1.6 * p.flare * k * kk)
                v.dot(float(e[0]), float(e[1]), p.flare_radius * 1.6, 0.35 * p.flare * k * kk)

    # -- styles ----------------------------------------------------------------------------
    def _lightning(self, v, a, b, L, t, tn):
        p = self.p
        rng = self.tick_rng("bolt", t, p.tick_rate)
        d = (b - a) / L
        e = 1 - p.flicker * float(rng.random())
        for k in range(p.strands):
            main = refine(rng, [a, b], rough=p.roughness, detail=max(L / 10, 6))
            off = (main - a[None]) @ np.array([-d[1], d[0]])
            lim = 0.16 * L
            if np.abs(off).max() > lim:           # keep the bolt near the a-b line (no big loops)
                main = main - np.outer(off * (1 - lim / np.abs(off).max()), [-d[1], d[0]])
            main = refine(rng, main, rough=p.roughness * 0.55, detail=max(L / 60, 2.5))
            amp = p.brightness * e * (1.0 if k == 0 else 0.55 * float(rng.uniform(0.4, 1.0)))
            if p.glow > 0 and k == 0:
                v.polyline(main, p.glow_width, p.glow * e)
            v.polyline(main, p.width * (1.0 if k == 0 else 0.7), amp)
            n = len(main)
            for _ in range(p.branches):
                if rng.random() < 0.35:
                    continue
                i = int(rng.integers(max(1, n // 8), max(2, n - n // 6)))
                ang = float(rng.choice([-1, 1]) * rng.uniform(0.35, 0.9))
                ca, sa = math.cos(ang), math.sin(ang)
                dirv = np.array([ca * d[0] - sa * d[1], sa * d[0] + ca * d[1]])
                bl = L * p.branch_length * float(rng.uniform(0.4, 1.0))
                q0 = main[i]
                bp = refine(rng, [q0, q0 + dirv * bl], rough=p.roughness * 1.1, detail=max(bl / 14, 2.5))
                v.polyline(bp, p.width * 0.7, amp * 0.7, r_end=p.width * 0.3, a_end=0.0)

    def _curve(self, a, b, L, tn, n, wave_amp, sag):
        p = self.p
        u = np.linspace(0, 1, n)
        d = (b - a) / L
        nrm = np.array([-d[1], d[0]])
        pts = a[None] + (b - a)[None] * u[:, None]
        pts[:, 1] += sag * 4 * u * (1 - u)
        env = np.sin(math.pi * u)
        if wave_amp:
            w = wave_amp * env * np.sin(TAU * (p.waves * u - p.flow * tn))
            pts = pts + nrm[None] * w[:, None]
        return u, pts, nrm, env

    def _tether(self, v, a, b, L, t, tn):
        p = self.p
        n = int(np.clip(L / 4, 24, 160))
        u, pts, nrm, env = self._curve(a, b, L, tn, n, p.wave, p.sag)
        pulse = np.zeros(n)
        if p.pulses > 0:
            for k in range(p.pulses):
                c = (k / p.pulses + p.flow * tn) % 1.0
                dd = np.minimum(np.abs(u - c), 1 - np.abs(u - c))
                pulse += np.exp(-0.5 * (dd / 0.045) ** 2)
        half = p.width * (1 + 0.8 * pulse)
        alpha = p.brightness * (0.75 + p.pulse_brightness * pulse)
        if p.glow > 0:
            draw_strip(v, pts, p.glow_width * (1 + 0.5 * pulse), p.glow * (0.6 + pulse), softness=1.0)
        draw_strip(v, pts, half, alpha, softness=0.5, core=0.8, core_width=0.35)

    def _drain(self, v, a, b, L, t, tn):
        p = self.p
        n = int(np.clip(L / 4, 24, 160))
        u, base, nrm, env = self._curve(a, b, L, tn, n, 0.0, 0.0)
        env = np.sin(math.pi * u) ** 0.6
        if p.glow > 0:
            texn = self.noise

            def tex(S, A):
                q = (S * 8 - p.flow * tn * 8) % 8
                return 0.65 + 0.7 * np.clip(0.5 + texn.fbm(q, A * 1.5 + 2, tn * 4, octaves=2), 0, 1)
            draw_strip(v, base, p.band * (0.5 + 0.5 * env) + p.glow_width * 0.3, p.glow * np.ones(n),
                       softness=1.0, texture=tex)
        for j in range(p.strands):
            ph = j / p.strands
            amp = p.band * (0.8 if j % 2 == 0 else 0.55)
            off = amp * env * np.sin(TAU * (p.waves * u - p.flow * tn + ph))
            pts = base + nrm[None] * off[:, None]
            al = p.brightness * (0.55 + 0.45 * np.sin(TAU * (3 * u - p.flow * tn + ph * 1.7)) ** 2) \
                * smoothstep(0, 0.08, u) * smoothstep(1, 0.92, u)
            draw_strip(v, pts, np.full(n, p.width * 0.8), al, softness=0.6, core=0.6, core_width=0.4)
        d = (b - a) / L
        for i in range(p.motes):
            q = (self.m_phase[i] + self.m_laps[i] * tn) % 1.0
            fade = float(smoothstep(0, 0.12, q) * smoothstep(1, 0.85, q))
            if fade <= 1e-3:
                continue
            sway = p.band * self.m_off[i] * math.sin(math.pi * q) ** 0.6 \
                * math.cos(TAU * (p.waves * q - p.flow * tn + self.m_ph2[i]))
            pos = a + (b - a) * q + nrm * sway
            tail = pos - d * min(L * 0.06, 14)
            al = fade * self.m_br[i] * p.brightness
            v.polyline([tail, pos], p.mote_size * 0.35, 0.0, r_end=p.mote_size * 0.6, a_end=al * 0.8)
            v.dot(float(pos[0]), float(pos[1]), p.mote_size, al)
