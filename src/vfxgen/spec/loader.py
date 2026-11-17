"""Loading scenes: YAML/JSON/dict -> variables substituted -> validated :class:`Scene`,
and resolving ``effect:`` references.

Reference resolution order for a layer's ``effect`` string:

1. a path (contains ``/`` or ends in ``.yaml``/``.yml``): relative to the referencing
   file's folder, then each ``VFXGEN_PATH`` folder, then the built-in library folder;
2. a library effect name (``explosion``);
3. an atom name (``ring``).

A name that is both a library effect and an atom is an error (ambiguous).
Nesting depth of scene references is limited (default 3); cycles are reported with the
full chain.
"""
from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from pydantic import ValidationError
from ruamel.yaml import YAML

from ..errors import ResolveError, SpecError
from .expr import evaluate, full_expr, is_expr, substitute
from .models import InlineScene, Scene

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "library"
MAX_DEPTH = 3

Source = Union[str, os.PathLike, Mapping[str, Any], Scene]


@dataclass
class LoadedScene:
    """A validated scene plus where it came from and the variable values it was loaded with."""
    scene: Scene
    raw: dict                      # substituted raw dict (before pydantic defaults)
    base_dir: Path
    file: Optional[Path] = None
    vars: dict[str, Any] = field(default_factory=dict)
    chain: tuple[str, ...] = ()    # reference chain (for cycle / depth reporting)

    @property
    def depth(self) -> int:
        return max(len(self.chain) - 1, 0)


def _yaml() -> YAML:
    y = YAML(typ="safe", pure=True)
    return y


def read_yaml(text: str) -> Any:
    try:
        return _yaml().load(io.StringIO(text))
    except Exception as e:  # ruamel raises many error types
        raise SpecError(f"YAML parse error: {e}") from None


def library_names() -> list[str]:
    return sorted(p.stem for p in LIBRARY_DIR.glob("*.yaml"))


def search_dirs(base_dir: Path | None) -> list[Path]:
    dirs = [base_dir] if base_dir else []
    for d in os.environ.get("VFXGEN_PATH", "").split(os.pathsep):
        if d.strip():
            dirs.append(Path(d).expanduser())
    dirs.append(LIBRARY_DIR)
    return [d for d in dirs if d is not None]


def find_scene_file(ref: str, base_dir: Path | None) -> Path:
    p = Path(ref).expanduser()
    if p.is_absolute():
        if p.is_file():
            return p.resolve()
        raise ResolveError(f"scene file not found: {ref}")
    for d in search_dirs(base_dir):
        cand = (d / p)
        if cand.is_file():
            return cand.resolve()
    raise ResolveError(f"scene file {ref!r} not found (looked in: "
                       f"{', '.join(str(d) for d in search_dirs(base_dir))})")


def _format_validation(e: ValidationError, prefix: str = "") -> SpecError:
    parts = []
    for err in e.errors():
        loc = []
        for x in err["loc"]:
            if isinstance(x, int):
                loc[-1:] = [f"{loc[-1]}[{x}]"] if loc else [f"[{x}]"]
            elif x in ("LightChannel", "SmokeChannel", "VolumeChannel", "SolidChannel",
                       "str", "InlineScene", "float", "Keys", "list[float]"):
                continue
            else:
                loc.append(str(x))
        path = ".".join(loc)
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown field"
        parts.append(f"{prefix + '.' if prefix and path else prefix}{path}: {msg}")
    # de-duplicate union noise
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return SpecError("invalid scene:\n  " + "\n  ".join(out[:12]))


def _raw_from(source: Source) -> tuple[dict, Path, Optional[Path]]:
    if isinstance(source, Scene):
        return source.model_dump(by_alias=True, exclude_none=True), Path.cwd(), None
    if isinstance(source, Mapping):
        return dict(source), Path.cwd(), None
    s = os.fspath(source)
    if "\n" not in s and (s.endswith((".yaml", ".yml", ".json")) or Path(s).is_file()):
        path = Path(s).expanduser().resolve()
        if not path.is_file():
            raise ResolveError(f"scene file not found: {s}")
        text = path.read_text()
        data = json.loads(text) if path.suffix == ".json" else read_yaml(text)
        if not isinstance(data, dict):
            raise SpecError(f"{path.name}: a scene file must be a mapping at the top level")
        return data, path.parent, path
    data = read_yaml(s)
    if not isinstance(data, dict):
        raise SpecError("scene text must be a YAML mapping")
    return data, Path.cwd(), None


def resolve_vars(raw: dict, overrides: Mapping[str, Any] | None, path: str = "") -> dict[str, Any]:
    """Var values: scene defaults (may reference built-ins and earlier vars), then overrides."""
    overrides = dict(overrides or {})
    declared = raw.get("vars") or {}
    if not isinstance(declared, dict):
        raise SpecError("`vars` must be a mapping", path=path)
    unknown = set(overrides) - set(declared)
    if unknown:
        raise SpecError(f"unknown variable(s) {sorted(unknown)}; this scene's vars: "
                        f"{sorted(declared) or 'none'}", path=path)
    builtins: dict[str, Any] = {}
    for k in ("seed", "fps"):
        v = raw.get(k)
        if v is not None and not is_expr(v):
            builtins[k] = v
    size = raw.get("size")
    if isinstance(size, list) and len(size) == 2 and not any(is_expr(x) for x in size):
        builtins["width"], builtins["height"] = size
    values: dict[str, Any] = dict(builtins)
    for k, spec in declared.items():
        default = spec["default"] if isinstance(spec, dict) and "default" in spec else spec
        if k in overrides:
            val = overrides[k]
        elif is_expr(default) and full_expr(default) is not None:
            val = evaluate(full_expr(default), values)
        else:
            val = default
        if isinstance(spec, dict):
            lo, hi, ch = spec.get("min"), spec.get("max"), spec.get("choices")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                if lo is not None and val < lo or hi is not None and val > hi:
                    raise SpecError(f"var `{k}` = {val} outside [{lo}, {hi}]", path=path)
            if ch is not None and val not in ch:
                raise SpecError(f"var `{k}` = {val!r} not one of {ch}", path=path)
        values[k] = val
    return values


def load(source: Source, vars: Mapping[str, Any] | None = None, *, chain: tuple[str, ...] = (),
         base_dir: Path | None = None, name_hint: str | None = None) -> LoadedScene:
    """Load and validate a scene from a path, YAML text, dict or Scene."""
    raw, bdir, file = _raw_from(source)
    if base_dir is not None and file is None:
        bdir = base_dir
    key = str(file) if file else (name_hint or "<inline>")
    if file is not None and str(file) in chain:
        raise ResolveError("reference cycle: " + " -> ".join([*chain, str(file)]))
    chain = (*chain, key)
    if len(chain) - 1 > MAX_DEPTH:
        raise ResolveError(f"scene nesting deeper than {MAX_DEPTH}: " + " -> ".join(chain))
    where = file.name if file else (name_hint or "")
    values = resolve_vars(raw, vars, path=where)
    body = {k: v for k, v in raw.items() if k != "vars"}
    try:
        body = substitute(body, values)
    except SpecError as e:
        raise SpecError(e.message, path=f"{where}: {e.path}" if where else e.path) from None
    # keep vars, recording the values actually used
    if raw.get("vars"):
        vv = {}
        for k, spec in raw["vars"].items():
            entry = dict(spec) if isinstance(spec, dict) and "default" in spec else {"default": spec}
            entry["default"] = values[k]
            vv[k] = entry
        body["vars"] = vv
    if file is not None and not body.get("name"):
        body["name"] = file.stem
    try:
        scene = Scene.model_validate(body)
    except ValidationError as e:
        err = _format_validation(e)
        raise SpecError(err.message, path=where) from None
    return LoadedScene(scene=scene, raw=body, base_dir=bdir, file=file, vars=values, chain=chain)


@dataclass
class EffectRef:
    kind: str                      # "atom" | "scene"
    atom_cls: Any = None
    scene: Optional[LoadedScene] = None
    label: str = ""


def resolve_effect(effect: Union[str, InlineScene], parent: LoadedScene, params: Mapping[str, Any],
                   path: str = "") -> EffectRef:
    """Resolve a layer's ``effect`` to an atom class or a loaded child scene."""
    from ..atom import ATOMS
    from ..atom import ensure_atoms
    ensure_atoms()

    if isinstance(effect, InlineScene):
        child = load(effect.scene.model_dump(by_alias=True, exclude_none=True), params,
                     chain=parent.chain, base_dir=parent.base_dir, name_hint=effect.scene.name or "inline")
        return EffectRef("scene", scene=child, label=effect.scene.name or "inline")
    ref = str(effect)
    is_path = "/" in ref or ref.endswith((".yaml", ".yml", ".json"))
    try:
        if is_path:
            f = find_scene_file(ref, parent.base_dir)
            return EffectRef("scene", scene=load(f, params, chain=parent.chain), label=ref)
        lib = LIBRARY_DIR / f"{ref}.yaml"
        if lib.is_file() and ref in ATOMS:
            raise ResolveError(f"`{ref}` is both a library effect and an atom")
        if lib.is_file():
            return EffectRef("scene", scene=load(lib, params, chain=parent.chain), label=ref)
        if ref in ATOMS:
            return EffectRef("atom", atom_cls=ATOMS[ref], label=ref)
        for d in search_dirs(parent.base_dir)[:-1]:
            cand = d / f"{ref}.yaml"
            if cand.is_file():
                return EffectRef("scene", scene=load(cand, params, chain=parent.chain), label=ref)
    except SpecError as e:
        if path and not e.path.startswith(path):
            raise type(e)(e.message, path=f"{path} ({ref}) > {e.path}" if e.path else f"{path} ({ref})") from None
        raise
    raise ResolveError(f"unknown effect {ref!r}: not a file, library effect "
                       f"({', '.join(library_names()) or 'none yet'}) or atom "
                       f"({', '.join(sorted(ATOMS))})", path=path)
