"""Public rendering entry point: scene source -> :class:`RenderResult`.

Rendering is a pure function of (scene, vars, overrides, seed, draft). Frames are
independent, so they are rendered in a process pool when worthwhile; results are
byte-identical to serial rendering.
"""
from __future__ import annotations

import copy as _copy
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .engine.compositor import Composition, build, render_at
from .engine.transform import Affine
from .errors import RenderCancelled, SpecError
from .spec.loader import LoadedScene, Source, _raw_from, load, resolve_vars  # noqa: F401
from .spec.models import Scene

__all__ = ["render", "render_frame", "render_themes", "render_directions", "validate", "RenderResult",
           "SceneInfo", "prepare", "apply_overrides"]


@dataclass
class RenderResult:
    frames: list[np.ndarray]
    fps: float
    width: int
    height: int
    loop: bool
    duration: float
    name: str
    explicit: dict
    times: list[float] = field(default_factory=list)
    elapsed: float = 0.0
    draft: bool = False
    source: Optional[str] = None
    blend: str = "alpha"                       # authored blend: "alpha" (normal) or "additive"
    pivot: tuple[float, float] = (0.5, 0.5)    # normalised pivot in the frame (0..1, y down)
    frame_indices: list[int] = field(default_factory=list)   # which frames of the full sequence
    frame_count_total: int = 0                 # frames in the full sequence
    events: list[dict] = field(default_factory=list)          # [{name, t, frame, layer, data}]
    tracks: dict[str, list] = field(default_factory=dict)     # name -> [[x, y] | None per frame]
    aux: dict[str, list] = field(default_factory=dict)        # extra maps -> frames

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    def sheet(self, columns: int | None = None, trim: bool = False):
        from .output.atlas import build_sheet
        return build_sheet(self.frames, self.fps, self.loop, columns=columns, trim=trim, times=self.times,
                           events=self.events, tracks=self.tracks, pivot=self.pivot, blend=self.blend)

    def save(self, out_dir: str | os.PathLike = "gallery", **kw):
        from .output.writer import save
        return save(self, out_dir, **kw)


# --- overrides -------------------------------------------------------------------
def _parse_scalar(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    low = s.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith(("[", "{")):
        from .spec.loader import read_yaml
        return read_yaml(s)
    return s


def apply_overrides(raw: dict, overrides: Mapping[str, Any]) -> dict:
    """Apply dotted-path overrides to a raw scene dict, e.g.
    ``layers.bolt.params.branches=8`` (layers addressed by id or index)."""
    raw = _copy.deepcopy(raw)
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        node: Any = raw
        for i, part in enumerate(parts):
            last = i == len(parts) - 1
            if isinstance(node, list):
                idx = None
                if part.isdigit():
                    idx = int(part)
                else:
                    for j, item in enumerate(node):
                        if isinstance(item, dict) and item.get("id") == part:
                            idx = j
                            break
                if idx is None or idx >= len(node):
                    raise SpecError(f"--set {dotted}: no list item {part!r}")
                if last:
                    node[idx] = _parse_scalar(value)
                else:
                    node = node[idx]
                continue
            if not isinstance(node, dict):
                raise SpecError(f"--set {dotted}: {'.'.join(parts[:i])} is not a mapping")
            if last:
                node[part] = _parse_scalar(value)
            else:
                if part not in node or node[part] is None:
                    node[part] = {}
                node = node[part]
    return raw


def _atom_scene(name: str, params: Mapping[str, Any], colors: Any = None) -> dict:
    """A one-layer scene wrapping an atom, sized to the atom's bounds."""
    from .atom import get_atom
    cls = get_atom(name)
    inst = cls(cls.validate_params(dict(params)), seed=0)
    x0, y0, x1, y1 = inst.get_bounds()
    w, h = int(math.ceil(x1 - x0)) + 16, int(math.ceil(y1 - y0)) + 16
    w, h = w + w % 2, h + h % 2
    fps = 24
    L = inst.length()
    dur = L if cls.kind == "oneshot" else max(round(L * fps) / fps, 1 / fps)
    layer: dict[str, Any] = {"id": name, "effect": name, "params": dict(params)}
    if colors is not None:
        layer["colors"] = colors
    return {"vfxgen": 1, "name": name, "size": [w, h], "fps": fps, "duration": dur,
            "loop": cls.kind == "loop", "layers": layer and [layer]}


def prepare(source: Source, vars: Mapping[str, Any] | None = None, *,
            overrides: Mapping[str, Any] | None = None, seed: int | None = None,
            colors: Any = None) -> LoadedScene:
    """Load a scene (path / name / dict / YAML text / Scene) with vars, overrides, seed and
    colours applied. A bare name resolves to a library effect or an atom."""
    from .atom import ATOMS
    from .atom import ensure_atoms
    ensure_atoms()
    from .spec.loader import LIBRARY_DIR
    vars = dict(vars or {})
    base_dir = None
    if isinstance(source, str) and "\n" not in source and "/" not in source \
            and not source.endswith((".yaml", ".yml", ".json")) and not Path(source).is_file():
        lib = LIBRARY_DIR / f"{source}.yaml"
        if lib.is_file():
            source = lib
        elif source in ATOMS:
            atom_params = {k: v for k, v in vars.items()}
            vars = {}
            source = _atom_scene(source, atom_params, colors)
            colors = None
        else:
            raise SpecError(f"{source!r} is not a file, library effect or atom")
    raw, bdir, file = _raw_from(source)
    base_dir = bdir
    if colors is not None:
        decl = raw.get("vars") or {}
        if "colors" in decl:
            vars.setdefault("colors", colors)
        else:
            for lay in raw.get("layers") or []:
                lay.setdefault("colors", colors)
    if seed is not None:
        raw = dict(raw)
        raw["seed"] = int(seed)
    if overrides:
        raw = apply_overrides(raw, overrides)
    loaded = load(raw, vars, base_dir=base_dir, name_hint=(file.stem if file else raw.get("name")))
    loaded.file = file
    if file is not None and not loaded.scene.name:
        loaded.scene.name = file.stem
    return loaded


# --- worker side --------------------------------------------------------------------
_W: dict[str, Any] = {}


def _worker_init(payload: dict) -> None:
    from .atom import ensure_atoms
    ensure_atoms()
    loaded = load(payload["raw"], payload["vars"], base_dir=Path(payload["base_dir"]),
                  name_hint=payload.get("name"))
    comp = build(loaded)
    _apply_draft(comp, payload["draft"])
    if payload.get("pixel"):
        _apply_pixel(comp, *payload["pixel"])
    _W["comp"] = comp
    _W["ss"] = payload["ss"]
    _W["ls"] = payload["ls"]
    _W["aux"] = tuple(payload.get("aux") or ())
    _W["mb"] = payload.get("mb")


def _worker_frames(times: list[float]) -> list[np.ndarray]:
    comp = _W["comp"]
    return [render_at(comp, T, ss=_W["ss"], length_scale=_W["ls"], aux=_W["aux"], motion_blur=_W["mb"])
            for T in times]


def _is_cancelled(cancel) -> bool:
    if cancel is None:
        return False
    if hasattr(cancel, "is_set"):
        return bool(cancel.is_set())
    return bool(cancel())


def _run_workers(payload: dict, chunks: list[list[float]], on_progress=None, cancel=None,
                 total: int = 0) -> list[list[np.ndarray]]:
    """Render frame chunks in parallel worker subprocesses (``python -m vfxgen._worker``).
    Workers report one progress tick per frame on stderr; cancelling kills them all."""
    import pickle
    import subprocess
    import sys
    import threading
    from .errors import RenderError
    procs, outs, errs = [], [], []
    done = [0]
    lock = threading.Lock()

    def read_out(pr, buf):
        buf.append(pr.stdout.read())

    def read_err(pr, buf):
        for line in pr.stderr:
            if line == b"P\n":
                with lock:
                    done[0] += 1
            else:
                buf.append(line)

    threads = []
    for ch in chunks:
        pr = subprocess.Popen([sys.executable, "-m", "vfxgen._worker"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pr.stdin.write(pickle.dumps((payload, ch), protocol=pickle.HIGHEST_PROTOCOL))
        pr.stdin.close()
        o, e = [], []
        for fn, buf in ((read_out, o), (read_err, e)):
            th = threading.Thread(target=fn, args=(pr, buf), daemon=True)
            th.start()
            threads.append(th)
        procs.append(pr)
        outs.append(o)
        errs.append(e)
    reported = -1
    try:
        while any(pr.poll() is None for pr in procs):
            if _is_cancelled(cancel):
                raise RenderCancelled("render cancelled")
            with lock:
                d = done[0]
            if on_progress is not None and d != reported:
                on_progress(d, total)
                reported = d
            time.sleep(0.02)
    except BaseException:
        for pr in procs:
            if pr.poll() is None:
                pr.kill()
        raise
    for th in threads:
        th.join()
    parts = []
    for pr, o, e in zip(procs, outs, errs):
        if pr.returncode != 0:
            msg = b"".join(e).decode(errors="replace").strip().splitlines()
            raise RenderError("render worker failed: " + (msg[-1] if msg else "unknown error"))
        parts.append(pickle.loads(o[0]))
    if on_progress is not None and reported != total:
        on_progress(total, total)
    return parts


def _apply_draft(comp: Composition, draft: bool) -> None:
    if not draft:
        return
    comp.width = max(2, int(math.ceil(comp.width / 2)))
    comp.height = max(2, int(math.ceil(comp.height / 2)))
    comp.root = Affine.scale(0.5) @ comp.root
    half = comp.fps / 2
    if not comp.loop or abs(comp.duration * half - round(comp.duration * half)) < 1e-6:
        comp.fps = half


@dataclass
class SceneInfo:
    """What a scene will produce, without rendering it (see :func:`validate`)."""
    name: str
    width: int
    height: int
    fps: float
    duration: float
    loop: bool
    frame_count: int
    instances: int
    channels: list[str]
    blend: str
    pivot: tuple[float, float]
    explicit: dict


def _apply_pixel(comp: Composition, size: int, fps: float | None) -> None:
    """Pixel-art geometry: native resolution 1/size, optional own frame rate."""
    k = 1.0 / size
    comp.width = max(2, int(round(comp.width * k)))
    comp.height = max(2, int(round(comp.height * k)))
    comp.root = Affine.scale(k) @ comp.root
    comp.edge_fade = comp.edge_fade * k
    if fps:
        comp.fps = float(fps)


def _bake_options(loaded: LoadedScene, opts: dict) -> LoadedScene:
    """Write render options into the scene (so the explicit spec reproduces them)."""
    opts = {k: v for k, v in opts.items() if v is not None}
    if not opts:
        return loaded
    from .spec.models import Scene
    raw = _copy.deepcopy(loaded.raw)
    for k, v in opts.items():
        if v is False:                 # explicit "off": remove the scene's own setting
            raw.pop(k, None)
        else:
            raw[k] = v
    try:
        scene = Scene.model_validate(raw)
    except Exception as e:  # pydantic ValidationError
        from .spec.loader import _format_validation
        raise _format_validation(e) if hasattr(e, "errors") else e
    return LoadedScene(scene=scene, raw=raw, base_dir=loaded.base_dir, file=loaded.file,
                       vars=loaded.vars, chain=loaded.chain)


def _load_and_build(source, vars, overrides, seed, colors, draft, opts=None):
    from .engine import assets as _assets
    loaded = source if isinstance(source, LoadedScene) else prepare(
        source, vars, overrides=overrides, seed=seed, colors=colors)
    loaded = _bake_options(loaded, opts or {})
    _assets.begin_tracking()
    try:
        comp = build(loaded)
    finally:
        used = _assets.end_tracking()
    comp.assets_used = used
    _apply_draft(comp, draft)
    return loaded, comp


def validate(source: Source | LoadedScene, vars: Mapping[str, Any] | None = None, *,
             overrides: Mapping[str, Any] | None = None, seed: int | None = None,
             colors: Any = None, draft: bool = False) -> SceneInfo:
    """Validate a scene completely (spec, references, atom params, limits) without rendering.

    Raises SpecError / ResolveError / LimitError exactly as :func:`render` would, before any
    pixel work. Typically a few milliseconds."""
    loaded, comp = _load_and_build(source, vars, overrides, seed, colors, draft)
    return SceneInfo(name=comp.name, width=comp.width, height=comp.height, fps=comp.fps,
                     duration=comp.duration, loop=comp.loop, frame_count=comp.frame_count,
                     instances=len(comp.instances), channels=[c.name for c in comp.channels],
                     blend=comp.blend, pivot=comp.pivot, explicit=comp.explicit)


def render(source: Source | LoadedScene, vars: Mapping[str, Any] | None = None, *,
           overrides: Mapping[str, Any] | None = None, seed: int | None = None,
           colors: Any = None, draft: bool = False, workers: int | None = None,
           cache: bool = True, frames: Sequence[int] | None = None,
           times: Sequence[float] | None = None, on_progress: Callable[[int, int], None] | None = None,
           cancel: Any = None, maps: Sequence[str] = (), motion_blur: int | None = None,
           rotate: float | None = None, pixel: Any = None, theme: Mapping[str, Any] | None = None) -> RenderResult:
    """Render a scene to frames.

    Args:
        source: scene path, library/atom name, YAML text, dict, :class:`Scene`, or a
            :class:`LoadedScene` from :func:`prepare`.
        vars: values for the scene's ``vars``. For a bare atom name: the atom's params.
        overrides: dotted-path overrides (``{"layers.bolt.params.branches": 8}``).
        seed: replaces the scene seed.
        colors: main colour spec (sets a ``colors`` var if declared, else every layer's colours).
        draft: half resolution, half fps, no supersampling (~8x faster).
        workers: process count (default: cores - 1). ``workers=1`` renders in this process and
            never starts a subprocess.
        cache: reuse an identical previous full render from the on-disk cache.
        frames: render only these frame indices of the full sequence (e.g. ``[0, 12]``).
        times: render only these scene times in seconds (overrides ``frames``).
        on_progress: ``fn(done, total)`` called as frames finish.
        cancel: a ``threading.Event`` (or zero-arg callable returning bool); when set, the render
            stops, workers are killed and :class:`~vfxgen.RenderCancelled` is raised.
        maps: extra maps to return in ``result.aux``: "normal", "emissive", "intensity", "distortion".
        motion_blur: samples per frame (overrides the scene's ``motion_blur``; 0/1 = off).
        rotate: rotate the whole effect (degrees, clockwise) about the frame centre.
        pixel: pixel-art output (dict / :class:`~vfxgen.spec.models.PixelArt`; overrides the
            scene's ``pixel``; ``False`` disables it).
        theme: recolour every channel: ``{channel name or kind: colour ramp}``, e.g.
            ``{"light": "#103,#46f,#bdf,#fff", "smoke": "#223,#556,#aab"}``.
    """
    from .engine.channels import AUX_MAPS
    bad = [m for m in maps if m not in AUX_MAPS]
    if bad:
        raise SpecError(f"unknown map(s) {bad}; available: {', '.join(AUX_MAPS)}")
    t0 = time.perf_counter()
    opts = {"theme": dict(theme) if theme else None, "rotate": rotate,
            "motion_blur": ({"samples": int(motion_blur), "shutter": 0.5} if motion_blur and motion_blur > 1
                            else (False if motion_blur is not None else None)),
            "pixel": (False if pixel is False else (pixel.model_dump() if hasattr(pixel, "model_dump") else pixel))}
    loaded, comp = _load_and_build(source, vars, overrides, seed, colors, draft, opts)
    ss, ls = (1, 0.5) if draft else (2, 1.0)
    px = loaded.scene.pixel
    if px is not None:
        _apply_pixel(comp, px.size, px.fps)
        ss, ls = 1, ls / px.size
    mb = comp.motion_blur
    aux = tuple(maps)
    n_total = comp.frame_count
    if times is not None:
        sel_times = [float(t) for t in times]
        indices = [int(round(t * comp.fps)) for t in sel_times]
    elif frames is not None:
        indices = [int(i) for i in frames]
        bad = [i for i in indices if not 0 <= i < n_total]
        if bad:
            raise SpecError(f"frame index {bad[0]} out of range 0..{n_total - 1}")
        sel_times = [comp.frame_time(i) for i in indices]
    else:
        indices = list(range(n_total))
        sel_times = [comp.frame_time(i) for i in indices]
    partial = frames is not None or times is not None
    n = len(sel_times)

    def result(frs, aux_frames=None):
        explicit = comp.explicit
        explicit["fps"] = float(comp.fps) if not draft else float(loaded.scene.fps)
        evs = [{**e, "t": round(e["t"], 6), "frame": min(n_total - 1, int(round(e["t"] * comp.fps)))}
               for e in comp.events]
        tracks: dict = {}
        if comp.tracks:
            per = [comp.track_positions(T) for T in sel_times]
            tracks = {name: [p[name] for p in per] for name, *_ in comp.tracks}
        return RenderResult(frames=frs, fps=comp.fps, width=comp.width, height=comp.height,
                            loop=comp.loop, duration=comp.duration, name=comp.name, explicit=explicit,
                            times=sel_times, elapsed=time.perf_counter() - t0, draft=draft,
                            source=str(loaded.file) if loaded.file else None, blend=comp.blend,
                            pivot=comp.pivot, frame_indices=indices, frame_count_total=n_total,
                            events=evs, tracks=tracks, aux=aux_frames or {})

    from .engine import cache as _cache
    use_cache = cache and not partial and os.environ.get("VFXGEN_NO_CACHE") != "1"
    use_cache = use_cache and not aux
    extra = {"mb": mb, "assets": comp.assets_used}
    ckey = _cache.key(comp.explicit, draft, extra) if use_cache else None
    cached = _cache.get(ckey) if ckey else None
    if cached is not None and len(cached) == n:
        if on_progress is not None:
            on_progress(n, n)
        return result(cached)
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, n // 2 if n >= 8 else 1))
    if workers == 1:
        out = []
        for k, T in enumerate(sel_times):
            if _is_cancelled(cancel):
                raise RenderCancelled("render cancelled")
            out.append(render_at(comp, T, ss=ss, length_scale=ls, aux=aux, motion_blur=mb))
            if on_progress is not None:
                on_progress(k + 1, n)
    else:
        payload = {"raw": loaded.raw, "vars": {}, "base_dir": str(loaded.base_dir),
                   "name": loaded.scene.name, "draft": draft, "ss": ss, "ls": ls,
                   "aux": aux, "mb": mb, "pixel": (px.size, px.fps) if px is not None else None}
        chunks = [sel_times[i::workers] for i in range(workers)]
        parts = _run_workers(payload, chunks, on_progress=on_progress, cancel=cancel, total=n)
        out = [None] * n  # type: ignore[list-item]
        for w, part in enumerate(parts):
            for j, fr in enumerate(part):
                out[w + j * workers] = fr
    aux_frames = None
    if aux:
        aux_frames = {m: [o[1][m] for o in out] for m in aux}
        out = [o[0] for o in out]
    if px is not None:
        from .engine.pixel import auto_palette, parse_palette, quantize_frame
        pal = parse_palette(px.palette) if px.palette else auto_palette(out, px.colors)
        out = [quantize_frame(f, pal, px.dither, px.dither_strength, px.alpha_threshold, px.outline) for f in out]
    if ckey:
        try:
            _cache.put(ckey, out)
        except OSError:  # pragma: no cover - cache is best effort
            pass
    return result(out, aux_frames)


def render_frame(source: Source | LoadedScene, t: float, vars: Mapping[str, Any] | None = None, *,
                 overrides: Mapping[str, Any] | None = None, seed: int | None = None,
                 colors: Any = None, draft: bool = False) -> np.ndarray:
    """One frame at scene time ``t`` (seconds), rendered in this process (no subprocess, no
    cache). For live previews: call on every slider move."""
    return render(source, vars, overrides=overrides, seed=seed, colors=colors, draft=draft,
                  workers=1, cache=False, times=[t]).frames[0]


def render_themes(source: Source | LoadedScene, themes: Mapping[str, Mapping[str, Any]],
                  vars: Mapping[str, Any] | None = None, **kw) -> dict[str, RenderResult]:
    """Render one effect in several colour themes in one pass: ``{theme name: RenderResult}``."""
    return {name: render(source, vars, theme=theme, **kw) for name, theme in themes.items()}


def render_directions(source: Source | LoadedScene, count: int = 8, vars: Mapping[str, Any] | None = None,
                      start: float = 0.0, **kw) -> dict[int, RenderResult]:
    """Render ``count`` evenly spaced rotations (8- or 16-way sprites): ``{degrees: RenderResult}``.
    0 degrees = as authored; angles increase clockwise."""
    out = {}
    for i in range(count):
        deg = round((start + 360.0 * i / count) % 360, 6)
        out[deg] = render(source, vars, rotate=deg or None, **kw)
    return out
