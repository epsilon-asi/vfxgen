"""Scene graph -> flat list of instances -> frames.

Building (:func:`build`):

* every layer is expanded into *copies* (``repeat``) and spawned children (``spawn``);
* atom layers become :class:`Instance` objects; scene layers recurse, adding a
  :class:`Level` (time map + transform + gain + mask) to the chain;
* channels are registered (declared ones, then auto-created ones from atom roles;
  nested scenes' channels are namespaced ``<layer-id>.<channel>``);
* an explicit, fully-specified copy of the scene is produced for the output YAML.

Rendering a frame at scene time ``T``: each instance walks its level chain to get its
local time, transform, gain and masks, then draws into views on the channel targets.
Channels are rendered and stacked (:mod:`vfxgen.engine.channels`).
"""
from __future__ import annotations

import copy as _copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..atom import Atom, DrawContext, resolve_anchor
from ..errors import LimitError, RenderError, SpecError
from ..spec.expr import evaluate, full_expr, is_expr
from ..spec.loader import EffectRef, LoadedScene, resolve_effect
from ..spec.models import Layer, Mask, Scene
from .anim import compile_value
from .canvas import SolidView, Target, View
from .channels import DEFAULT_PALETTE, KIND_RANK, SCALAR_KINDS, ChannelRuntime, composite
from .color import adjust_stops, get_palette, palette_key
from .mathx import TAU, derive_seed, ease, smoothstep
from .noise import Noise
from .paths import Path
from .transform import Affine

MAX_INSTANCES = 20000
MAX_SIZE = 4096
MAX_FRAMES = 1000
SEED_SCHEME = 1


# =============================================================================
# time / transform levels
# =============================================================================
@dataclass
class TimeMap:
    start: float = 0.0
    speed: float = 1.0
    length: Optional[float] = None      # clip in parent seconds
    mode: str = "once"
    natural: float = 1.0                # natural length in local seconds
    warp: Optional[tuple[str, float, float, Optional[float]]] = None   # curve, exp, offset, span
    fade_in: float = 0.0
    fade_out: float = 0.0
    holds: tuple = ()                   # hitstops: ((at_local_s, hold_s), ...)

    def local(self, Tp: float) -> Optional[float]:
        x = Tp - self.start
        if x < -1e-9:
            return None
        if self.length is not None and x > self.length + 1e-9:
            return None
        tl = max(x, 0.0) * self.speed
        for at, hold in self.holds:
            if tl > at:
                tl = max(at, tl - hold)
        L = max(self.natural, 1e-9)
        if self.mode == "once":
            if tl > L + 1e-9:
                return None
        elif self.mode == "loop":
            tl = tl % L
        elif self.mode == "hold":
            tl = min(tl, L)
        elif self.mode == "pingpong":
            m = tl % (2 * L)
            tl = m if m <= L else 2 * L - m
        if self.warp is not None:
            curve, ex, off, span = self.warp
            S = span or L
            if tl <= S:
                q = min(max(tl / S, 0.0), 1.0)
                f = q ** ex if curve == "power" else ease(curve, q)
                tl = S * (off + (1 - off) * f)
        return tl

    def span(self) -> Optional[float]:
        """Active duration in parent seconds (None = unbounded)."""
        if self.length is not None:
            return self.length
        if self.mode == "once":
            return (self.natural + sum(h for _, h in self.holds)) / self.speed
        return None

    def gain(self, Tp: float) -> float:
        g = 1.0
        x = Tp - self.start
        if self.fade_in > 0:
            g *= min(max(x / self.fade_in, 0.0), 1.0)
        sp = self.span()
        if self.fade_out > 0 and sp is not None:
            g *= min(max((sp - x) / self.fade_out, 0.0), 1.0)
        return g


@dataclass
class Level:
    time: TimeMap
    xf: Callable[[float, float], Affine]          # (t_local, T_parent) -> child->parent affine
    gain: Callable[[float, float], float]
    mask: Optional[Mask] = None
    label: str = ""


def _mask_fn(m: Mask) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    cx, cy = m.center
    if m.type == "sweep":
        def f(X, Y, t):
            ang = np.degrees(np.arctan2(Y - cy, X - cx))
            head = 360.0 * m.turns * t
            if m.profile == "lobe":
                d = np.mod(ang - head + 180.0, 360.0) - 180.0
                v = np.exp(-(d / max(m.width, 1e-6)) ** 2)
            else:
                d = np.mod(head - ang, 360.0)
                v = np.clip(1 - d / max(m.width, 1e-6), 0, 1) ** 2
            if m.inner > 0:
                v = v * smoothstep(m.inner * 0.6, m.inner, np.hypot(X - cx, Y - cy))
            return m.floor + (1 - m.floor) * v
    elif m.type == "radial":
        def f(X, Y, t):
            v = smoothstep(m.inner, m.outer, np.hypot(X - cx, Y - cy))
            return 1 - v if not m.invert else v
    elif m.type == "linear":
        (sx, sy), (ex, ey) = (m.start or [0, 0]), (m.end or [100, 0])
        L2 = (ex - sx) ** 2 + (ey - sy) ** 2 + 1e-9

        def f(X, Y, t):
            q = ((X - sx) * (ex - sx) + (Y - sy) * (ey - sy)) / L2
            v = smoothstep(0, 1, q)
            return 1 - v if m.invert else v
    else:  # noise
        nz = Noise(np.random.default_rng(m.seed))

        def f(X, Y, t):
            n = nz.fbm(X / m.scale, Y / m.scale + m.speed * t, 0.0, octaves=3)
            v = smoothstep(m.threshold - m.softness, m.threshold + m.softness, n)
            return 1 - v if m.invert else v
    return f


# =============================================================================
# instances & composition
# =============================================================================
@dataclass
class Instance:
    atom: Atom
    levels: list[Level]
    roles: dict[str, str]              # role -> channel name
    z: int
    order: int
    path: str
    links: dict = field(default_factory=dict)     # name -> (sibling level, ref, atom, anchor)

    def parent_state(self, T: float, root: Affine):
        """(A, local time) at the parent of this instance's own layer, or None if inactive."""
        A = root
        t = T
        for lv in self.levels[:-1]:
            tl = lv.time.local(t)
            if tl is None:
                return None
            A = A @ lv.xf(tl, t)
            t = tl
        return A, t

    def state(self, T: float, root: Affine):
        A = root
        g = 1.0
        masks = []
        t = T
        for lv in self.levels:
            tl = lv.time.local(t)
            if tl is None:
                return None
            A = A @ lv.xf(tl, t)
            g *= lv.gain(tl, t) * lv.time.gain(t)
            if g <= 1e-6:
                return None
            if lv.mask is not None:
                masks.append((A, _mask_fn(lv.mask), tl))
            t = tl
        return t, A, g, masks


@dataclass
class Composition:
    width: int
    height: int
    fps: float
    duration: float
    loop: bool
    edge_fade: float
    channels: list[ChannelRuntime]
    instances: list[Instance]
    explicit: dict
    root: Affine = field(default_factory=Affine.identity)
    name: str = "scene"
    blend: str = "alpha"
    pivot_local: tuple[float, float] = (0.0, 0.0)     # pivot in scene canvas coords (before root)
    tile: bool = False
    motion_blur: Optional[tuple[int, float]] = None   # (samples, shutter)
    events: list = field(default_factory=list)        # [{name, t, layer, data}] scene seconds
    tracks: list = field(default_factory=list)        # [(name, level, ref, atom, anchor)]
    assets_used: dict = field(default_factory=dict)   # asset path -> content hash

    def track_positions(self, T: float) -> dict:
        out = {}
        for name, lvl, ref, atom_obj, anchor in self.tracks:
            st = lvl.time.local(T)
            if st is None:
                out[name] = None
                continue
            ax, ay = _anchor_local(ref, atom_obj, anchor, st, sib_child_size(ref))
            x, y = (self.root @ lvl.xf(st, T)).apply(ax, ay)
            out[name] = [round(float(x), 3), round(float(y), 3)]
        return out

    @property
    def pivot(self) -> tuple[float, float]:
        """Normalised pivot (0..1) in the output frame."""
        x, y = self.root.apply(*self.pivot_local)
        return (round(float(x) / self.width, 6), round(float(y) / self.height, 6))

    @property
    def frame_count(self) -> int:
        if self.loop:
            return max(1, int(round(self.duration * self.fps)))
        n = int(math.floor(self.duration * self.fps + 1e-6)) + 1
        if (n - 1) / self.fps < self.duration - 1e-6:
            n += 1          # one-shots always get a frame exactly at `duration`
        return n

    def frame_time(self, i: int) -> float:
        return i / self.fps if self.loop else min(i / self.fps, self.duration)

    def render_frame(self, i: int, ss: int = 2, length_scale: float = 1.0) -> np.ndarray:
        return render_at(self, self.frame_time(i), ss=ss, length_scale=length_scale)


def _draw_instances(comp: Composition, T: float, ss: int) -> dict[str, Target]:
    W, H = comp.width, comp.height
    targets: dict[str, Target] = {}
    chans = {c.name: c for c in comp.channels}
    for inst in comp.instances:
        st = inst.state(T, comp.root)
        if st is None:
            continue
        t, A, g, masks = st
        b = inst.atom.get_bounds()
        x0, y0, x1, y1 = A.bbox(*b)
        if not comp.tile and (x1 < 0 or y1 < 0 or x0 > W or y0 > H):
            continue
        mask_fn = None
        if masks:
            invs = [(Am.inverse(), f, tl) for Am, f, tl in masks]

            def mask_fn(SX, SY, invs=invs):
                out = 1.0
                for inv, f, tl in invs:
                    lx, ly = inv.apply(SX, SY)
                    out = out * f(lx, ly, tl)
                return out
        views = {}
        groups = []                      # (private solid view, real solid view) to composite as a group
        for role, cname in inst.roles.items():
            ch = chans[cname]
            tg = targets.get(cname)
            if tg is None:
                tg = targets[cname] = Target(ch.kind if ch.kind != "distort" else "light", W, H, ss, wrap=comp.tile)
            if ch.kind == "solid" and (g < 1.0 - 1e-6 or mask_fn is not None):
                real = SolidView(tg, A, gain=g, mask=mask_fn, palette=ch.stops_at(T), clip=b)
                priv = SolidView(Target("solid", W, H, ss, wrap=comp.tile), A, palette=ch.stops_at(T), clip=b)
                views[role] = priv
                groups.append((priv, real))
                continue
            cls = SolidView if ch.kind == "solid" else View
            views[role] = cls(tg, A, gain=g, mask=mask_fn, palette=ch.stops_at(T), clip=b)
        target_fn = None
        if inst.links:
            Ainv = A.inverse()

            def target_fn(name, dt=0.0, inst=inst, Ainv=Ainv):
                if name not in inst.links:
                    raise KeyError(f"no target {name!r} on this layer (have: {', '.join(inst.links)})")
                T2 = T - dt
                if comp.loop and comp.duration > 0:
                    T2 %= comp.duration            # look back across the loop seam
                ps = inst.parent_state(T2, comp.root)
                if ps is None:
                    return None
                Ap, tp = ps
                slv, sref, satom, anc = inst.links[name]
                stl = slv.time.local(tp)
                if stl is None:
                    return None
                ax, ay = _anchor_local(sref, satom, anc, stl, sib_child_size(sref))
                px, py = (Ap @ slv.xf(stl, tp)).apply(ax, ay)
                lx, ly = Ainv.apply(px, py)
                return float(lx), float(ly)
        try:
            inst.atom.draw(DrawContext(views, A.linear_scale, target=target_fn, t_scene=T), t)
        except (SpecError, RenderError):
            raise
        except Exception as e:  # pragma: no cover - surfaced with location
            raise RenderError(f"{inst.path}: {type(e).__name__}: {e}") from e
        for priv, real in groups:
            real.add(priv)
    return targets


def _channels_scaled(comp: Composition, length_scale: float):
    if length_scale == 1.0:
        return comp.channels
    chs = [_copy.copy(c) for c in comp.channels]
    for c in chs:
        c.length_scale *= length_scale
    return chs


def render_at(comp: Composition, T: float, ss: int = 2, length_scale: float = 1.0, aux: tuple = (),
              motion_blur: Optional[tuple[int, float]] = None):
    """Render scene time T -> RGBA uint8 (and ``{map: RGBA}`` when ``aux`` is given).

    Motion blur averages ``samples`` renders spread over ``shutter`` x the frame interval ending at T."""
    W, H = comp.width, comp.height
    chs = _channels_scaled(comp, length_scale)
    edge = comp.edge_fade * (length_scale if length_scale < 1 else 1)
    mb = motion_blur if motion_blur is not None else comp.motion_blur
    if not mb or mb[0] <= 1:
        targets = _draw_instances(comp, T, ss)
        return composite(chs, targets, W, H, edge, T=T, wrap=comp.tile, aux=tuple(aux))
    n, shutter = mb
    acc_p = np.zeros((H, W, 3), np.float32)
    acc_a = np.zeros((H, W), np.float32)
    maps = None
    for k in range(n):
        Tk = T - shutter / comp.fps * (k / (n - 1))
        if comp.loop:
            Tk %= comp.duration
        targets = _draw_instances(comp, Tk, ss)
        if k == 0 and aux:
            (p, a), maps = composite(chs, targets, W, H, edge, T=Tk, wrap=comp.tile, aux=tuple(aux), as_float=True)
        else:
            p, a = composite(chs, targets, W, H, edge, T=Tk, wrap=comp.tile, as_float=True)
        acc_p += p
        acc_a += a
    from .channels import _to_u8
    main = _to_u8(acc_p / n, acc_a / n)
    return (main, maps) if aux else main


# =============================================================================
# building
# =============================================================================
@dataclass
class Copy:
    layer: Layer
    lid: str
    index: int                 # layer index in its scene
    seed: int
    dt: float = 0.0
    pos: Optional[tuple[float, float]] = None       # absolute position override (repeat.along / spawn)
    dp: tuple[float, float] = (0.0, 0.0)
    rot_add: float = 0.0
    scale_mul: float = 1.0
    opacity_mul: float = 1.0
    speed_mul: float = 1.0
    hue: float = 0.0
    params: Optional[dict] = None
    effect: Any = None
    gen: int = 0
    label: str = ""


def _ns_spec(spec, ns: str):
    """Namespace channel references inside a channel spec (lit_by / glow_from) for nesting."""
    if not ns:
        return spec
    upd = {}
    lb = getattr(spec, "lit_by", None)
    if lb is not None:
        upd["lit_by"] = lb.model_copy(update={"channels": [ns + c for c in lb.channels]})
    gf = getattr(spec, "glow_from", None)
    if gf is not None:
        upd["glow_from"] = gf.model_copy(update={"channel": ns + gf.channel})
    return spec.model_copy(update=upd) if upd else spec


class _Builder:
    def __init__(self):
        self.channels: dict[str, ChannelRuntime] = {}
        self.instances: list[Instance] = []
        self.order = 0

    # -- channels ----------------------------------------------------------------
    def add_declared(self, name: str, spec, length_scale: float) -> None:
        if name in self.channels:
            return
        ns = name[: len(name) - len(name.split(".")[-1])]
        ch = ChannelRuntime(name=name, spec=_ns_spec(spec, ns), length_scale=length_scale, declared=True,
                            order=len(self.channels))
        ch.finalize()
        self.channels[name] = ch

    def channel_for(self, ns: str, role: str, role_def, layer: Layer, copy: Copy, length_scale: float,
                    main: bool) -> str:
        name = self._resolve(ns, role, role_def, layer, copy, length_scale, main)
        if layer.depth:
            return self._depth_slice(name, int(layer.depth))
        return name

    def _depth_slice(self, name: str, depth: int) -> str:
        key = f"{name}^{depth}"
        if key not in self.channels:
            base = self.channels[name]
            ch = ChannelRuntime(name=key, spec=base.spec, length_scale=base.length_scale,
                                declared=base.declared, order=base.order + 0.5, depth=depth)
            ch.finalize()
            self.channels[key] = ch
        return key

    def _resolve(self, ns, role, role_def, layer, copy, length_scale, main) -> str:
        from ..spec.models import DistortChannel, LightChannel, SmokeChannel, SolidChannel, VolumeChannel
        target = ns + layer.channels.get(role, role)
        pal = layer.palettes.get(role) or (layer.colors if main else None)
        adj = layer.adjust
        hue = copy.hue
        existing = self.channels.get(target)
        want_stops = None
        if pal is not None or adj is not None or hue:
            base = pal if pal is not None else (existing.spec.palette if existing and existing.spec.palette
                                                is not None else role_def.settings.get("palette"))
            stops = get_palette(base) if base is not None else get_palette(DEFAULT_PALETTE[role_def.kind])
            if adj is not None or hue:
                stops = adjust_stops(stops, (adj.hue if adj else 0) + hue,
                                     adj.saturation if adj else 1, adj.brightness if adj else 1)
            want_stops = [list(map(float, [p])) + [list(map(float, c))] for p, c in stops]
        if existing is not None:
            ok = existing.kind == role_def.kind or (existing.kind in SCALAR_KINDS and role_def.kind in SCALAR_KINDS)
            if not ok:
                raise SpecError(f"channel `{target}` is a {existing.kind} channel but this atom's "
                                f"`{role}` output draws {role_def.kind} (solid outputs need solid channels)",
                                path=copy.label)
            if want_stops is None or palette_key(want_stops) == palette_key(existing.stops):
                return target
            sib = f"{target}@{derive_seed(palette_key(want_stops)) % 10**6:06d}"
            if sib not in self.channels:
                spec = existing.spec.model_copy(update={"palette": want_stops, "palette_keys": None})
                ch = ChannelRuntime(name=sib, spec=spec, length_scale=existing.length_scale,
                                    declared=False, order=len(self.channels))
                ch.finalize()
                self.channels[sib] = ch
            return sib
        settings = dict(role_def.settings)
        if want_stops is not None:
            settings["palette"] = want_stops
        cls = {"light": LightChannel, "smoke": SmokeChannel, "volume": VolumeChannel,
               "solid": SolidChannel, "distort": DistortChannel}[role_def.kind]
        spec = _ns_spec(cls(**settings), ns)
        ch = ChannelRuntime(name=target, spec=spec, length_scale=length_scale, declared=False,
                            order=len(self.channels))
        ch.finalize()
        self.channels[target] = ch
        return target

    def ordered_channels(self) -> list[ChannelRuntime]:
        decl = [c for c in self.channels.values() if c.declared and c.depth == 0]
        auto = sorted((c for c in self.channels.values() if not c.declared and c.depth == 0),
                      key=lambda c: (KIND_RANK[c.kind], c.order))
        base = decl + auto
        rank = {c.name: i for i, c in enumerate(base)}
        slices = sorted((c for c in self.channels.values() if c.depth != 0),
                        key=lambda c: (c.depth, rank.get(c.name.split("^")[0], 0)))
        return [c for c in slices if c.depth < 0] + base + [c for c in slices if c.depth > 0]


def _expand(layer: Layer, index: int, scene_seed: int) -> list[Copy]:
    lid = layer.id or f"layer{index}"
    base_seed = int(layer.seed) if layer.seed is not None else derive_seed(scene_seed, "layer", lid)
    label = f"layers[{index}]({layer.effect if isinstance(layer.effect, str) else 'inline'})"
    if layer.repeat is None:
        return [Copy(layer, lid, index, base_seed, label=label)]
    r = layer.repeat
    n = r.count
    rng = np.random.default_rng(derive_seed(base_seed, "repeat"))
    if r.order == "reverse":
        rank = list(range(n - 1, -1, -1))
    elif r.order == "random":
        rank = list(rng.permutation(n))
    elif r.order == "pingpong":
        half = list(range(0, n, 2)) + list(range(n - 1 - (n % 2 == 1), 0, -2))
        rank = [half.index(i) if i in half else i for i in range(n)]
    else:
        rank = list(range(n))
    pts = Path(r.along).sample(n, r.along.spacing, rng) if r.along is not None else None
    if r.seeds is not None and len(r.seeds) != n:
        raise SpecError(f"repeat.seeds has {len(r.seeds)} entries but count is {n}", path=label)
    out = []
    j = r.jitter
    for i in range(n):
        k = 0.0 if n == 1 else i / (n - 1)
        u = lambda a: float(rng.uniform(-a, a)) if a > 0 else 0.0  # noqa: E731
        c = Copy(layer, lid if n == 1 else f"{lid}#{i}", index,
                 seed=(r.seeds[i] if r.seeds is not None else
                       derive_seed(base_seed, "copy", i) if r.new_seed else base_seed),
                 label=f"{label}#copy{i}")
        c.dt = rank[i] * r.every + u(j.time)
        if pts is not None:
            (px, py), tang = pts[i]
            c.pos = (px, py)
            if r.align:
                c.rot_add += tang
        c.dp = (u(j.position), u(j.position))
        c.rot_add += u(j.rotate) + (r.ramp.rotate[0] + (r.ramp.rotate[1] - r.ramp.rotate[0]) * k
                                    if r.ramp.rotate else 0.0)
        c.scale_mul = (1 + u(j.scale)) * (r.ramp.scale[0] + (r.ramp.scale[1] - r.ramp.scale[0]) * k
                                          if r.ramp.scale else 1.0)
        c.opacity_mul = (r.ramp.opacity[0] + (r.ramp.opacity[1] - r.ramp.opacity[0]) * k
                         if r.ramp.opacity else 1.0)
        c.speed_mul = 1 + u(j.speed)
        c.hue = u(j.hue)
        out.append(c)
    return out


def _spawn(c: Copy, static_pos: tuple[float, float]) -> list[Copy]:
    s = c.layer.spawn
    if s is None:
        return []
    out: list[Copy] = []
    parents = [(c, static_pos)]
    rng = np.random.default_rng(derive_seed(c.seed, "spawn"))
    k = 0
    for g in range(1, s.depth + 1):
        nxt = []
        for par, ppos in parents:
            offs = (Path(s.at).sample(s.count, s.at.spacing, rng) if s.at is not None else
                    [((s.radius * math.cos(TAU * j / s.count), s.radius * math.sin(TAU * j / s.count)), 0.0)
                     for j in range(s.count)])
            for j in range(s.count):
                (ox, oy), _ = offs[j]
                sc = par.scale_mul
                when = s.when if isinstance(s.when, (int, float)) else float(rng.uniform(*s.when))
                jt = s.jitter
                u = lambda a: float(rng.uniform(-a, a)) if a > 0 else 0.0  # noqa: E731
                pos = (ppos[0] + ox * sc + u(jt.position), ppos[1] + oy * sc + u(jt.position))
                params = dict(c.layer.params) if s.effect == "self" else {}
                for key, val in s.params.items():
                    params[key] = (evaluate(full_expr(val), {"gen": g}) if isinstance(val, str)
                                   and full_expr(val) else val)
                lay = c.layer.model_copy(update={
                    "spawn": None, "repeat": None, "params": params,
                    "effect": c.layer.effect if s.effect == "self" else s.effect,
                    "move": None, "attach": None})
                seed = (s.seeds[k] if s.seeds is not None and k < len(s.seeds)
                        else derive_seed(c.seed, "spawn", g, j, k))
                ch = Copy(lay, f"{c.lid}~{g}.{k}", c.index, seed,
                          dt=par.dt + when + u(jt.time), pos=pos, rot_add=par.rot_add + u(jt.rotate),
                          scale_mul=sc * s.scale * (1 + u(jt.scale)),
                          opacity_mul=par.opacity_mul * s.opacity, speed_mul=par.speed_mul,
                          hue=par.hue + u(jt.hue), gen=g, label=f"{c.label}~gen{g}#{j}")
                out.append(ch)
                nxt.append((ch, pos))
                k += 1
        parents = nxt
    return out


def _position(at, size, path, variables=None):
    """Compile ``at`` -> f(t, T, p) -> (x, y) in parent coordinates."""
    w, h = size
    if at is None:
        return lambda t, T, p: (w / 2, h / 2)
    if hasattr(at, "uv") and not hasattr(at, "keys"):
        x, y = w * at.uv[0], h * at.uv[1]
        return lambda t, T, p: (x, y)
    f = compile_value(at, variables, path=path)
    return f


class _SceneCtx:
    """Per-scene build state (siblings for attach, size)."""

    def __init__(self, loaded: LoadedScene, size: tuple[float, float]):
        self.loaded = loaded
        self.size = size
        self.siblings: dict[str, tuple[Level, Any, Any]] = {}   # id -> (level, ref, copy)


def _scene_anchor_spec(v):
    """In a scene's ``anchors:`` block a bare [x, y] means PIXELS of that scene's canvas."""
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return {"px": [float(v[0]), float(v[1])]}
    return v


def _anchor_local(ref: EffectRef, atom_obj: Optional[Atom], spec, t: float, child_size) -> tuple[float, float]:
    if atom_obj is not None:
        return atom_obj.anchor_point(spec, t)
    sc = ref.scene.scene
    box = (0.0, 0.0, float(child_size[0]), float(child_size[1]))

    def named(n, tt):
        if n in sc.anchors:
            return resolve_anchor(_scene_anchor_spec(sc.anchors[n]), box, None, tt, owner=sc.name or "scene")
        return None
    return resolve_anchor(spec, box, named, t, owner=sc.name or "scene")


def _feel(feel, t: float, seed: int) -> tuple[float, float, float, float]:
    """Game-feel modifiers at local time t -> (scale_x, scale_y, offset_x, offset_y)."""
    sx = sy = 1.0
    dx = dy = 0.0
    if feel.anticipation is not None:
        an = feel.anticipation
        if 0 <= t <= an.duration:
            k = math.sin(math.pi * t / an.duration)
            sx *= 1 - an.amount * k
            sy *= 1 - an.amount * k
            dx += an.offset[0] * k
            dy += an.offset[1] * k
    for pu in feel.squash:
        q = (t - pu.at) / pu.duration
        if 0 <= q <= 1:
            env = math.sin(2 * math.pi * q) * (1 - q)
            m = max(1 - pu.amount * env, 0.05)
            if feel.squash_axis == "y":
                sy, sx = sy * m, sx / m
            else:
                sx, sy = sx * m, sy / m
    for pu in feel.punch:
        q = (t - pu.at) / pu.duration
        if 0 <= q <= 1:
            v = 1 + pu.amount * math.exp(-4 * q) * math.cos(3 * math.pi * q) * (1 - q)
            sx, sy = sx * v, sy * v
    for i, sh in enumerate(feel.shake):
        q = (t - sh.at) / sh.duration
        if 0 <= q <= 1:
            ph = (derive_seed(seed, "shake", i) % 10000) / 10000 * TAU
            a = sh.amplitude * (1 - q)
            dx += a * math.sin(TAU * sh.frequency * t + ph)
            dy += a * math.sin(TAU * sh.frequency * 1.37 * t + ph * 1.7)
    return sx, sy, dx, dy


def _make_level(c: Copy, ref: EffectRef, atom_obj: Optional[Atom], natural: float, child_size,
                sctx: _SceneCtx, default_mode: str) -> Level:
    L = c.layer
    path = c.label
    V = sctx.loaded.vars            # scene vars are visible to time expressions
    at_f = _position(L.at, sctx.size, path, V)
    off_f = compile_value(L.offset, V, path=path + ".offset") if L.offset is not None else None
    rot_f = compile_value(L.rotate, V, path=path + ".rotate")
    if isinstance(L.scale, list):
        sx_, sy_ = float(L.scale[0]), float(L.scale[1])
        scale_f = lambda t, T, p: (sx_, sy_)  # noqa: E731
    else:
        scale_f = compile_value(L.scale, V, path=path + ".scale")
    op_f = compile_value(L.opacity, V, path=path + ".opacity")
    move = Path(L.move.path) if L.move is not None else None
    fx = -1.0 if L.flip in ("x", "xy") else 1.0
    fy = -1.0 if L.flip in ("y", "xy") else 1.0
    nat = max(natural, 1e-9)
    feel = L.feel
    tm = TimeMap(start=L.start + c.dt, speed=L.speed * c.speed_mul, length=L.length,
                 mode=L.time_mode or default_mode, natural=nat,
                 holds=tuple(sorted((h.at, h.hold) for h in feel.hitstop)) if feel else (),
                 warp=((L.time_warp.curve, L.time_warp.exp, L.time_warp.offset, L.time_warp.span)
                       if L.time_warp else None),
                 fade_in=L.fade.in_ if L.fade else 0.0, fade_out=L.fade.out if L.fade else 0.0)
    attach = L.attach
    const_anchor = None
    if atom_obj is None or not (isinstance(L.anchor, str) and callable(atom_obj.anchors.get(L.anchor))):
        const_anchor = _anchor_local(ref, atom_obj, L.anchor, 0.0, child_size)

    extra_rot = [0.0]

    def pos_at(t, Tp):
        p = t / nat
        extra_rot[0] = 0.0
        if c.pos is not None:
            x, y = c.pos
        elif move is not None:
            d = L.move.duration or (tm.span() or nat / tm.speed)
            q = (t / tm.speed - L.move.delay) / d if d > 0 else 1.0
            q = q % 1.0 if L.move.loop else min(max(q, 0.0), 1.0)
            x, y = move.point(ease(L.move.ease, q) if not L.move.loop else q)
        elif attach is not None:
            sib = sctx.siblings.get(attach.layer)
            if sib is None:
                raise SpecError(f"attach: no layer with id `{attach.layer}` before this one", path=path)
            slv, sref, satom = sib
            st = slv.time.local(Tp)
            if st is None:
                st = 0.0
            ap = _anchor_local(sref, satom, attach.anchor, st, sib_child_size(sref))
            sA = slv.xf(st, Tp)
            x, y = sA.apply(*ap)
            if attach.rotate:
                extra_rot[0] = sA.angle
        else:
            x, y = at_f(t, Tp, p)
        if off_f is not None:
            ox, oy = off_f(t, Tp, p)
            x, y = x + ox, y + oy
        return x + c.dp[0], y + c.dp[1]

    def xf(t, Tp):
        p = t / nat
        x, y = pos_at(t, Tp)
        rot = float(rot_f(t, Tp, p)) + c.rot_add + extra_rot[0]
        if L.face_motion and move is not None:
            d = L.move.duration or (tm.span() or nat / tm.speed)
            q = (t / tm.speed - L.move.delay) / d if d > 0 else 1.0
            q = q % 1.0 if L.move.loop else min(max(q, 0.0), 1.0)
            rot += move.tangent(ease(L.move.ease, q) if not L.move.loop else q)
        s = scale_f(t, Tp, p)
        sx, sy = (s, s) if not isinstance(s, tuple) else s
        sx, sy = sx * c.scale_mul * fx, sy * c.scale_mul * fy
        if feel is not None:
            fsx, fsy, fdx, fdy = _feel(feel, t, c.seed)
            sx, sy = sx * fsx, sy * fsy
            x, y = x + fdx, y + fdy
        ax, ay = const_anchor if const_anchor is not None else _anchor_local(ref, atom_obj, L.anchor, t, child_size)
        return (Affine.translate(x, y) @ Affine.rotate(rot) @ Affine.scale(sx, sy)
                @ Affine.translate(-ax, -ay))

    def gain(t, Tp):
        return float(op_f(t, Tp, t / nat)) * c.opacity_mul

    return Level(time=tm, xf=xf, gain=gain, mask=L.mask, label=path)


def sib_child_size(ref: EffectRef):
    if ref.kind == "scene":
        s = ref.scene.scene.size
        return (256, 256) if s == "auto" else tuple(s)
    return None


def _static_scale(L: Layer, c: Copy) -> float:
    try:
        if isinstance(L.scale, list):
            v = math.sqrt(abs(L.scale[0] * L.scale[1]))
        else:
            v = compile_value(L.scale)(0.0, 0.0, 0.0)
            v = math.sqrt(abs(v[0] * v[1])) if isinstance(v, tuple) else abs(v)
    except SpecError:
        v = 1.0
    return v * c.scale_mul


def _build_scene(b: _Builder, loaded: LoadedScene, levels: list[Level], ns: str, length_scale: float,
                 path_prefix: str, size: tuple[float, float], record_seeds: bool = True) -> tuple[dict, float, list]:
    """Build one scene's layers. Returns (explicit scene dict, natural duration, events in this
    scene's time).

    ``record_seeds``: write derived seeds into the explicit dict. Off inside scenes that are
    instantiated several times (repeat / spawn), where seeds must keep deriving per copy."""
    scene = loaded.scene
    for name, spec in scene.channels.items():
        b.add_declared(ns + name, spec, length_scale)
    sctx = _SceneCtx(loaded, size)
    if not levels:
        b.root_siblings = sctx.siblings
    evs: list = [{"name": e.name, "t": e.t, "layer": path_prefix.rstrip(" >") or "scene", "data": dict(e.data)}
                 for e in scene.events]
    explicit = _copy.deepcopy(loaded.raw)
    explicit_layers = []
    ends: list[float] = []
    loops: list[float] = []
    for idx, layer in enumerate(scene.layers):
        lay_explicit = _copy.deepcopy((loaded.raw.get("layers") or [])[idx]) if idx < len(
            loaded.raw.get("layers") or []) else {}
        if not layer.enabled:
            explicit_layers.append(lay_explicit)
            continue
        copies = _expand(layer, idx, scene.seed)
        all_copies: list[Copy] = []
        for c in copies:
            all_copies.append(c)
            if layer.spawn is not None:
                # static position of the parent for spawn placement
                pos = c.pos or (size[0] / 2, size[1] / 2)
                if c.pos is None and layer.at is not None:
                    try:
                        v = _position(layer.at, size, c.label)(0.0, 0.0, 0.0)
                        pos = (float(v[0]), float(v[1]))
                    except (SpecError, TypeError):
                        pass
                all_copies.extend(_spawn(c, pos))
        if len(b.instances) + len(all_copies) > MAX_INSTANCES:
            raise LimitError(f"more than {MAX_INSTANCES} instances", path=path_prefix + copies[0].label)
        first_atom_seeds = None
        child_explicit = None
        single = layer.repeat is None and layer.spawn is None
        for c in all_copies:
            L = c.layer
            pth = path_prefix + c.label
            ref = resolve_effect(L.effect, loaded, L.params, path=pth)
            if ref.kind == "atom":
                params = ref.atom_cls.validate_params(L.params, path=pth + ".params")
                try:
                    atom_obj = ref.atom_cls(params, seed=c.seed, seeds=L.seeds, base_dir=sctx.loaded.base_dir)
                except SpecError as e:
                    raise SpecError(e.message, path=pth) from None
                natural = atom_obj.length()
                default_mode = "loop" if ref.atom_cls.kind == "loop" else "once"
                lvl = _make_level(c, ref, atom_obj, natural, None, sctx, default_mode)
                roles = {}
                main = ref.atom_cls.main_role()
                for role, rdef in ref.atom_cls.outputs.items():
                    roles[role] = b.channel_for(ns, role, rdef, L, c, length_scale, role == main)
                links = {}
                for lname, tref in list(L.targets.items()) + ([("follow", L.follow)] if L.follow else []):
                    sib = sctx.siblings.get(tref.layer)
                    if sib is None:
                        raise SpecError(f"`{lname}` refers to layer `{tref.layer}`, which must be an earlier "
                                        f"layer in the same scene", path=pth)
                    links[lname] = (sib[0], sib[1], sib[2], tref.anchor)
                b.instances.append(Instance(atom_obj, levels + [lvl], roles, L.z, b.order, pth, links))
                b.order += 1
                if c is all_copies[0]:
                    first_atom_seeds = atom_obj.stream_seeds()
                    lay_explicit["params"] = params.model_dump()
                    sctx.siblings[layer.id or f"layer{idx}"] = (lvl, ref, atom_obj)
            else:
                child = ref.scene
                child.scene = child.scene.model_copy(update={"seed": c.seed})
                csize = child.scene.size
                csize = (256, 256) if csize == "auto" else tuple(csize)
                default_mode = "loop" if child.scene.loop else "once"
                lvl = _make_level(c, ref, None, 1.0, csize, sctx, default_mode)
                cns = ns + (c.lid if layer.isolate else layer.id or f"layer{idx}") + "."
                cexp, cdur, cevs = _build_scene(b, child, levels + [lvl], cns,
                                                length_scale * _static_scale(L, c), pth + " > ", csize,
                                                record_seeds=record_seeds and single)
                lvl.time.natural = max(cdur, 1e-6)
                for ev in cevs:
                    evs.append({**ev, "t": lvl.time.start + ev["t"] / lvl.time.speed})
                if c is all_copies[0]:
                    child_explicit = cexp
                    sctx.siblings[layer.id or f"layer{idx}"] = (lvl, ref, None)
            for ev in L.events:
                tl = ev.t + sum(h for at, h in lvl.time.holds if at < ev.t)
                evs.append({"name": ev.name, "t": lvl.time.start + tl / lvl.time.speed,
                            "layer": (path_prefix + c.label), "data": dict(ev.data)})
            sp = lvl.time.span()
            if lvl.time.mode == "loop" and lvl.time.length is None:
                loops.append(lvl.time.start + lvl.time.natural / lvl.time.speed)
            elif sp is not None:
                ends.append(lvl.time.start + sp)
        # explicit layer record
        lay_explicit["id"] = layer.id or f"layer{idx}"
        if record_seeds:
            lay_explicit["seed"] = copies[0].seed if layer.repeat is None else (
                int(layer.seed) if layer.seed is not None else derive_seed(scene.seed, "layer", lay_explicit["id"]))
            if layer.repeat is not None:
                lay_explicit.setdefault("repeat", {})["seeds"] = [c.seed for c in copies]
            if first_atom_seeds is not None and single:
                lay_explicit["seeds"] = first_atom_seeds
        if child_explicit is not None:
            lay_explicit["effect"] = {"scene": child_explicit}
            lay_explicit.pop("params", None)
        explicit_layers.append(lay_explicit)
    explicit["layers"] = explicit_layers
    if record_seeds:
        explicit["seed"] = scene.seed
    else:
        explicit.pop("seed", None)
    if scene.duration == "auto":
        dur = max(ends + loops) if (ends or loops) else 1.0
    else:
        dur = float(scene.duration)
    explicit["duration"] = float(dur)
    return explicit, dur, evs


def apply_theme(channels: list[ChannelRuntime], theme: dict) -> None:
    """Recolour channels: ``theme`` maps a channel name (its last segment, e.g. ``smoke``) or a
    channel kind (``light`` / ``smoke`` / ``volume`` / ``solid``) to an explicit colour ramp.
    Names win over kinds. Colour over time (palette_keys) is replaced too."""
    from .color import get_palette
    for spec in theme.values():
        get_palette(spec)                     # validate early
    for ch in channels:
        leaf = ch.name.split("^")[0].split("@")[0].split(".")[-1]
        spec = theme.get(leaf, theme.get(ch.kind))
        if spec is None:
            continue
        ch.spec = ch.spec.model_copy(update={"palette": spec, "palette_keys": None})
        ch.finalize()


def build(loaded: LoadedScene, theme: dict | None = None) -> Composition:
    """Build a renderable :class:`Composition` from a loaded scene."""
    scene: Scene = loaded.scene
    b = _Builder()
    size = (512, 512) if scene.size == "auto" else tuple(scene.size)
    if max(size) > MAX_SIZE:
        raise LimitError(f"size {size} exceeds {MAX_SIZE}px")
    explicit, dur, evs = _build_scene(b, loaded, [], "", 1.0, "", size)
    root = Affine.identity()
    if scene.size == "auto":
        x0 = y0 = math.inf
        x1 = y1 = -math.inf
        for k in range(9):
            T = dur * k / 8
            for inst in b.instances:
                st = inst.state(T, root)
                if st is None:
                    continue
                bx0, by0, bx1, by1 = st[1].bbox(*inst.atom.get_bounds())
                x0, y0, x1, y1 = min(x0, bx0), min(y0, by0), max(x1, bx1), max(y1, by1)
        if not math.isfinite(x0):
            x0, y0, x1, y1 = 0, 0, 256, 256
        m = 8
        size = (int(math.ceil(x1 - x0)) + 2 * m, int(math.ceil(y1 - y0)) + 2 * m)
        root = Affine.translate(m - x0, m - y0)
        explicit["size"] = list(size)
    comp = Composition(width=int(size[0]), height=int(size[1]), fps=float(scene.fps), duration=float(dur),
                       loop=bool(scene.loop), edge_fade=float(scene.edge_fade),
                       channels=b.ordered_channels(), instances=sorted(b.instances, key=lambda i: (i.z, i.order)),
                       explicit=explicit, root=root, name=scene.name or "scene", blend=scene.blend,
                       tile=bool(scene.tile),
                       motion_blur=(scene.motion_blur.samples, scene.motion_blur.shutter) if scene.motion_blur else None,
                       events=sorted((e for e in evs if -1e-9 <= e["t"] <= dur + 1e-9), key=lambda e: e["t"]))
    for tr in scene.track:
        sib = getattr(b, "root_siblings", {}).get(tr.layer)
        if sib is None:
            raise SpecError(f"track `{tr.name}`: no layer with id `{tr.layer}` in the scene")
        comp.tracks.append((tr.name, sib[0], sib[1], sib[2], tr.anchor))
    if comp.tile:
        comp.edge_fade = 0.0
    theme = theme or scene.theme
    if theme:
        apply_theme(comp.channels, theme)
    if scene.rotate:
        cx, cy = comp.width / 2, comp.height / 2
        comp.root = Affine.translate(cx, cy) @ Affine.rotate(scene.rotate) @ Affine.translate(-cx, -cy) @ comp.root
    box = (0.0, 0.0, float(size[0] if scene.size != "auto" else 512), float(size[1] if scene.size != "auto" else 512))

    def _named(n, t):
        if n in scene.anchors:
            return resolve_anchor(_scene_anchor_spec(scene.anchors[n]), box, None, t, owner=scene.name or "scene")
        return None
    comp.pivot_local = resolve_anchor(scene.pivot, box, _named, 0.0, owner=scene.name or "scene")
    if comp.loop:
        n = comp.duration * comp.fps
        if abs(n - round(n)) > 1e-4:
            raise SpecError(f"loop scene: duration × fps must be a whole number of frames "
                            f"({comp.duration} × {comp.fps} = {n:g})")
    if comp.frame_count > MAX_FRAMES:
        raise LimitError(f"{comp.frame_count} frames exceeds the limit of {MAX_FRAMES}")
    explicit["meta"] = {**(explicit.get("meta") or {}), "seed_scheme": SEED_SCHEME}
    return comp
