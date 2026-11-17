"""Channel rendering and compositing.

Each channel turns its scene-space buffer into premultiplied RGBA (float, 0..1):

* light   : (optional glow_from) -> bloom -> downsample -> tone-map through palette
* smoke   : flat self-shadowed smoke, or billow relief lighting (relief > 0)
* volume  : surface shading from the density's normals (dust, water)
* solid   : painted matter
* distort : not drawn; becomes the distortion map (engine heat-haze / refraction shaders)

Smoke, volume and solid channels can be lit by light channels (``lit_by``): ``tint`` colours
matter near light, ``surface`` lights it from the light's direction using surface normals.

Channels are stacked bottom -> top with their blend mode. Layers may draw into *depth slices*
(``name^N``) of a channel so e.g. smoke can pass in front of flames.

The shading maths are ported unchanged from the original effects so library scenes keep their look.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .canvas import Target
from .color import adjust_stops, get_palette, ramp
from .mathx import blur, gradient, shift, smoothstep

DEFAULT_PALETTE = {"light": "fire", "smoke": "smoke", "volume": "dust", "solid": "rock", "distort": "fire"}
KIND_RANK = {"volume": 0, "smoke": 1, "solid": 2, "light": 3, "distort": 4}
SCALAR_KINDS = {"light", "smoke", "volume", "distort"}
AUX_MAPS = ("normal", "emissive", "intensity", "distortion")


def base_name(name: str) -> str:
    """Channel name without its depth slice (^N) and palette sibling (@xxxxxx) suffixes."""
    return name.split("^")[0].split("@")[0]


def _resample(stops, n: int = 33) -> np.ndarray:
    pos = np.linspace(0, 1, n)
    return ramp(stops, pos)


@dataclass
class ChannelRuntime:
    """A channel as used during rendering."""
    name: str
    spec: Any
    length_scale: float = 1.0      # nested/draft scale applied to blur radii etc.
    declared: bool = True
    order: float = 0
    depth: int = 0
    stops: list = field(default_factory=list)
    _keys: list = field(default_factory=list)

    @property
    def kind(self) -> str:
        return self.spec.kind

    def finalize(self) -> None:
        pal = self.spec.palette if self.spec.palette is not None else DEFAULT_PALETTE[self.kind]
        stops = get_palette(pal)
        adj = self.spec.adjust
        if adj is not None:
            stops = adjust_stops(stops, adj.hue, adj.saturation, adj.brightness)
        self.stops = stops
        self._keys = []
        if getattr(self.spec, "palette_keys", None):
            for k in sorted(self.spec.palette_keys, key=lambda k: float(k[0])):
                st = get_palette(k[1])
                if adj is not None:
                    st = adjust_stops(st, adj.hue, adj.saturation, adj.brightness)
                self._keys.append((float(k[0]), _resample(st)))

    def stops_at(self, T: float):
        """Palette at scene time T (colour over time via palette_keys)."""
        if not self._keys:
            return self.stops
        keys = self._keys
        if T <= keys[0][0]:
            cols = keys[0][1]
        elif T >= keys[-1][0]:
            cols = keys[-1][1]
        else:
            for (t0, c0), (t1, c1) in zip(keys, keys[1:]):
                if t0 <= T <= t1:
                    k = (T - t0) / max(t1 - t0, 1e-9)
                    cols = c0 + (c1 - c0) * k
                    break
        pos = np.linspace(0, 1, len(cols))
        return [(float(p), tuple(float(v) for v in c)) for p, c in zip(pos, cols)]


def edge_mask(w: int, h: int, margin: float) -> np.ndarray:
    if margin <= 0:
        return np.ones((h, w), np.float32)

    def sm(v):
        v = np.clip(v, 0, 1)
        return v * v * (3 - 2 * v)
    x = sm(np.minimum(np.arange(w) + 0.5, w - np.arange(w) - 0.5) / margin)
    y = sm(np.minimum(np.arange(h) + 0.5, h - np.arange(h) - 0.5) / margin)
    return np.outer(y, x).astype(np.float32)


@dataclass
class Rendered:
    prem: np.ndarray       # (h, w, 3) premultiplied rgb 0..1
    a: np.ndarray          # (h, w) alpha 0..1
    intensity: np.ndarray | None = None   # light: bloomed intensity (for lit_by)
    tone: np.ndarray | None = None        # light: tone-mapped 0..1 (intensity map)
    normal: tuple | None = None           # (nx, ny, nl) where a surface normal exists
    density: np.ndarray | None = None


@dataclass
class Ctx:
    T: float = 0.0
    wrap: bool = False
    mask: Any = None


def render_light(ch: ChannelRuntime, tg: Target, targets: dict[str, Target], c: Ctx) -> Rendered:
    s = ch.spec
    ls, ss = ch.length_scale, tg.ss
    B = tg.data
    if s.glow_from is not None and s.glow_from.channel in targets:
        src = targets[s.glow_from.channel].data
        if src.any():
            B = B + s.glow_from.amount * blur(src, s.glow_from.blur * ls * ss, c.wrap)
    if s.bloom and B.any():
        B = B + sum(float(w) * blur(B, float(sig) * ls * ss, c.wrap) for sig, w in s.bloom)
    I = tg.downsample(B) * c.mask
    t = 1.0 - np.exp(-np.maximum(I, 0) * s.exposure)
    rgb = ramp(ch.stops_at(c.T), t) / 255.0
    a = (t ** s.alpha_gamma).astype(np.float32)
    return Rendered(rgb * a[..., None], a, I, t.astype(np.float32))


def _light_sum(lit_by, lights):
    srcs = [(c, r) for c, r in lights if base_name(c.name) in lit_by.channels and r.intensity is not None]
    if not srcs:
        return None, None
    I = sum(r.intensity for _, r in srcs)
    if not np.any(I):
        return None, None
    return I, srcs[0][0]


def _apply_lit_by(rgb, lit_by, lights, size, normal, c: Ctx, T):
    """rgb in 0..255. Returns lit rgb."""
    I, src = _light_sum(lit_by, lights)
    if I is None:
        return rgb
    radius = lit_by.radius or size / 25
    Ib = blur(I, radius, c.wrap)
    tint = ramp(src.stops_at(T), np.array(0.55))
    gl = (1 - np.exp(-Ib * lit_by.amount))
    if lit_by.mode == "surface" and normal is not None:
        nx, ny, nl = normal
        gy, gx = gradient(Ib, c.wrap)
        g = np.sqrt(gx * gx + gy * gy) + 1e-6
        lx, ly = gx / g * 0.8, gy / g * 0.8             # toward the light (brighter side)
        lz = 0.6
        ndl = np.clip((nx * lx + ny * ly + lz) / (nl * math.sqrt(0.8 ** 2 + lz ** 2)), 0, 1)
        k = (gl * (0.35 + 0.65 * ndl))[..., None]
        return np.clip(rgb * (1 - 0.5 * k) + tint * k * 0.9 + rgb * k * 0.4, 0, 255)
    k = gl[..., None]
    return rgb * (1 - k) + (tint * 0.6 + rgb * 0.4) * k + tint * 0.25 * k


def render_smoke(ch: ChannelRuntime, tg: Target, lights, c: Ctx) -> Rendered:
    s = ch.spec
    size = s.size_ref * ch.length_scale
    D = np.maximum(tg.downsample(), 0) * c.mask
    a = 1.0 - np.exp(-D * s.density)
    lx, ly = s.light_dir
    Db = blur(D, max(1.5, size / 90), c.wrap)
    off = size / 40
    occl = shift(Db, lx * off, ly * off, c.wrap) + 0.5 * shift(Db, lx * off * 2.2, ly * off * 2.2, c.wrap)
    normal = None
    if s.relief > 0:
        Hh = 0.0
        for sig, wt in ((size / 90, 0.2), (size / 45, 0.4), (size / 20, 0.4)):
            Hh = Hh + wt * (1 - np.exp(-blur(D, sig, c.wrap) * s.density * s.height_gain))
        gy, gx = gradient(Hh, c.wrap)
        k = s.relief * size / 7
        nx, ny = -gx * k, -gy * k
        nl = np.sqrt(nx * nx + ny * ny + 1)
        lz = 0.75
        ll = math.sqrt(lx * lx + ly * ly + lz * lz)
        lam = np.clip((nx * lx + ny * ly + lz) / (nl * ll), 0, 1)
        sh = np.exp(-s.shadow * occl * s.density * 0.5)
        lit = np.clip(s.ambient + (1 - s.ambient) * lam ** 1.6 * (0.25 + 0.75 * sh), 0, 1)
        normal = (nx, ny, nl)
    else:
        lit = np.exp(-s.shadow * occl)
    rgb = ramp(ch.stops_at(c.T), lit)
    if s.lit_by is not None and s.lit_by.amount > 0:
        rgb = _apply_lit_by(rgb, s.lit_by, lights, size, normal, c, c.T)
    rgb = np.clip(rgb / 255.0, 0, 1)
    a = a.astype(np.float32)
    return Rendered(rgb * a[..., None], a, normal=normal, density=D)


def render_volume(ch: ChannelRuntime, tg: Target, lights, c: Ctx) -> Rendered:
    s = ch.spec
    ls = ch.length_scale
    size = s.size_ref * ls
    D = np.maximum(tg.downsample(), 0) * c.mask
    Hf = blur(np.sqrt(D), s.soft * ls, c.wrap)
    gy_, gx_ = gradient(Hf, c.wrap)
    bump = s.bump * ls
    nx, ny = -gx_ * bump, -gy_ * bump
    nl = np.sqrt(nx * nx + ny * ny + 1)
    L = np.array(s.light, np.float32)
    L = L / np.linalg.norm(L)
    ndl = (nx * L[0] + ny * L[1] + L[2]) / nl
    diff = np.clip((ndl + s.wrap) / (1 + s.wrap), 0, 1)
    off = size / 45
    Db = blur(D, max(1.5, size / 90), c.wrap)
    occl = shift(Db, L[0] * off, L[1] * off, c.wrap) + 0.5 * shift(Db, L[0] * off * 2.2, L[1] * off * 2.2, c.wrap)
    lit = diff * np.exp(-s.shadow * occl)
    stops = ch.stops_at(c.T)
    rgb = ramp(stops, np.clip(lit, 0, 1))
    if s.lit_by is not None and s.lit_by.amount > 0:
        rgb = _apply_lit_by(rgb, s.lit_by, lights, size, (nx, ny, nl), c, c.T)
    rgb = rgb / 255.0
    a = 1 - np.exp(-D * s.density)
    if s.rim > 0:
        fres = np.clip(1 - 1 / nl, 0, 1) ** 1.5 * s.rim * smoothstep(0.01, 0.15, a)
        top = np.array(stops[-1][1], np.float32) / 255
        rgb = rgb + (top - rgb) * fres[..., None] * 0.7
        a = a + (1 - a) * fres * 0.6
    if s.spec > 0:
        Hv = np.array([L[0], L[1], L[2] + 1.0])
        Hv = Hv / np.linalg.norm(Hv)
        sp = np.clip((nx * Hv[0] + ny * Hv[1] + Hv[2]) / nl, 0, 1) ** s.spec_power * s.spec
        sp = np.clip(sp * smoothstep(0.02, 0.25, a), 0, 1)
        rgb = rgb + (1 - rgb) * sp[..., None]
        a = a + (1 - a) * sp * 0.8
    a = np.clip(a, 0, 1).astype(np.float32)
    return Rendered(np.clip(rgb, 0, 1) * a[..., None], a, normal=(nx, ny, nl), density=D)


def render_solid(ch: ChannelRuntime, tg: Target, lights, c: Ctx) -> Rendered:
    prem = tg.downsample(tg.rgb) * c.mask[..., None]
    a = tg.downsample(tg.a) * c.mask
    s = ch.spec
    if s.lit_by is not None and s.lit_by.amount > 0 and a.any():
        straight = np.where(a[..., None] > 1e-6, prem / np.maximum(a[..., None], 1e-6), 0) * 255
        lit = _apply_lit_by(straight, s.lit_by, lights, s.size_ref * ch.length_scale, None, c, c.T)
        prem = np.clip(lit / 255.0, 0, 1) * a[..., None]
    return Rendered(prem.astype(np.float32), a.astype(np.float32))


def render_distort(ch: ChannelRuntime, tg: Target, c: Ctx) -> np.ndarray:
    """Distortion map RGBA float: RG = displacement direction (0.5 = none), B = 0.5, A = strength."""
    s = ch.spec
    F = blur(tg.downsample() * c.mask, s.blur * ch.length_scale, c.wrap) * s.strength
    gy, gx = gradient(F, c.wrap)
    m = np.clip(F, 0, 1)
    out = np.zeros(F.shape + (4,), np.float32)
    out[..., 0] = np.clip(0.5 + gx * 4, 0, 1)
    out[..., 1] = np.clip(0.5 + gy * 4, 0, 1)
    out[..., 2] = 0.5
    out[..., 3] = m
    return out


def _to_u8(prem: np.ndarray, a: np.ndarray) -> np.ndarray:
    a = np.clip(a, 0, 1).astype(np.float32)
    rgb = np.clip(prem / np.maximum(a, 1e-6)[..., None], 0, 1)
    a = a.copy()
    a[a < 1.5 / 255] = 0
    rgb[a == 0] = 0
    return np.clip(np.dstack([rgb, a]) * 255 + 0.5, 0, 255).astype(np.uint8)


def _stack(items, w, h):
    out_p = np.zeros((h, w, 3), np.float32)
    out_a = np.zeros((h, w), np.float32)
    for ch, r in items:
        op = ch.spec.opacity
        cp, ca = (r.prem * op, r.a * op) if op != 1.0 else (r.prem, r.a)
        if ch.spec.blend == "over":
            out_p = cp + out_p * (1 - ca[..., None])
            out_a = ca + out_a * (1 - ca)
        elif ch.spec.blend == "add":
            out_p = out_p + cp
            out_a = ca + out_a - ca * out_a
        else:  # screen
            out_p = cp + out_p - cp * out_p
            out_a = ca + out_a - ca * out_a
    return out_p, np.clip(out_a, 0, 1)


def composite(channels: list[ChannelRuntime], targets: dict[str, Target], w: int, h: int,
              edge_fade: float, T: float = 0.0, wrap: bool = False, aux: tuple = (),
              as_float: bool = False):
    """Render every channel and stack them.

    Returns straight-alpha RGBA uint8, or ``(prem, a)`` floats with ``as_float``. With ``aux``
    (names from AUX_MAPS) returns ``(main, {name: RGBA uint8})``."""
    c = Ctx(T=T, wrap=wrap, mask=edge_mask(w, h, 0 if wrap else edge_fade))
    rendered: dict[str, Rendered] = {}
    lights: list = []
    for ch in channels:                       # lights first: other channels may be lit by them
        tg = targets.get(ch.name)
        if ch.kind == "light" and tg is not None and (tg.dirty or ch.spec.glow_from is not None):
            r = render_light(ch, tg, targets, c)
            rendered[ch.name] = r
            lights.append((ch, r))
    for ch in channels:
        tg = targets.get(ch.name)
        if tg is None or not tg.dirty or ch.name in rendered or ch.kind == "distort":
            continue
        fn = {"smoke": render_smoke, "volume": render_volume, "solid": render_solid}[ch.kind]
        rendered[ch.name] = fn(ch, tg, lights, c)
    items = [(ch, rendered[ch.name]) for ch in channels if ch.name in rendered]
    out_p, out_a = _stack(items, w, h)
    main = (out_p, out_a) if as_float else _to_u8(out_p, out_a)
    if not aux:
        return main
    maps: dict[str, np.ndarray] = {}
    if "emissive" in aux:
        ep, ea = _stack([(ch, r) for ch, r in items if ch.kind == "light"], w, h)
        maps["emissive"] = _to_u8(ep, ea)
    if "intensity" in aux:
        tone = np.zeros((h, w), np.float32)
        ia = np.zeros((h, w), np.float32)
        for ch, r in items:
            if r.tone is not None:
                tone = np.maximum(tone, r.tone)
                ia = ia + r.a * (1 - ia)
        g = np.clip(tone, 0, 1)
        maps["intensity"] = np.clip(np.dstack([g, g, g, ia]) * 255 + 0.5, 0, 255).astype(np.uint8)
    if "normal" in aux:
        nx = np.zeros((h, w), np.float32)
        ny = np.zeros((h, w), np.float32)
        wsum = np.zeros((h, w), np.float32)
        for ch, r in items:
            if r.normal is not None:
                n_x, n_y, n_l = r.normal
                nx += r.a * (n_x / n_l)
                ny += r.a * (n_y / n_l)
                wsum += r.a
        nz = np.sqrt(np.clip(1 - nx ** 2 - ny ** 2, 0, 1))
        flat = wsum < 1e-4
        nz[flat] = 1.0
        enc = np.dstack([nx * 0.5 + 0.5, -ny * 0.5 + 0.5, nz * 0.5 + 0.5, out_a])   # +y up (Unity / GL)
        maps["normal"] = np.clip(enc * 255 + 0.5, 0, 255).astype(np.uint8)
    if "distortion" in aux:
        acc = np.zeros((h, w, 4), np.float32)
        acc[..., :3] = 0.5
        for ch in channels:
            tg = targets.get(ch.name)
            if ch.kind == "distort" and tg is not None and tg.dirty:
                d = render_distort(ch, tg, c)
                acc[..., :2] += d[..., :2] - 0.5
                acc[..., 3] = np.maximum(acc[..., 3], d[..., 3])
        maps["distortion"] = np.clip(acc * 255 + 0.5, 0, 255).astype(np.uint8)
    return main, maps
