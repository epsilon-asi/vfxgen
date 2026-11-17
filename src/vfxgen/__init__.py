"""vfxgen: procedural VFX sprite generator.

Quick start::

    import vfxgen as vfx

    result = vfx.render("explosion", {"radius": 1.2}, colors="#4af")
    gen = result.save("gallery")            # gallery/explosion/v001/...

    scene = vfx.load("my_scene.yaml")       # validated Scene model
    vfx.render("my_scene.yaml", draft=True).save()

    vfx.list_atoms()     # building blocks and their parameters
    vfx.list_effects()   # library effects and their vars
"""
from __future__ import annotations

__version__ = "1.1.0"

from typing import Any, Mapping

from .errors import LimitError, RenderCancelled, RenderError, ResolveError, SpecError, VfxError
from .render import (RenderResult, SceneInfo, prepare, render, render_directions, render_frame,
                     render_themes, validate)
from .spec.models import Layer, Scene

__all__ = [
    "__version__", "render", "render_frame", "render_themes", "render_directions", "validate", "prepare", "load", "RenderResult", "SceneInfo",
    "Scene", "Layer", "RenderCancelled",
    "list_atoms", "list_effects", "describe", "json_schema", "effect", "SpecError", "ResolveError",
    "LimitError", "RenderError", "VfxError",
]


def load(source, vars: Mapping[str, Any] | None = None, **kw):
    """Load and validate a scene; returns a :class:`~vfxgen.spec.loader.LoadedScene`
    (``.scene`` is the typed :class:`Scene`)."""
    return prepare(source, vars, **kw)


def effect(name: str, *, colors: Any = None, seed: int | None = None, **vars: Any):
    """Shorthand: a library effect or atom with vars/params, ready for :func:`render`."""
    return prepare(name, vars, colors=colors, seed=seed)


def list_atoms() -> list[dict]:
    """Every registered atom: name, doc, kind, params (default + doc), roles, anchors, streams."""
    from .atom import ATOMS
    from .atom import ensure_atoms
    ensure_atoms()
    return [ATOMS[n].info() for n in sorted(ATOMS)]


def list_effects() -> list[dict]:
    """Library effects (scene YAMLs shipped with the package) and their vars."""
    from .spec.loader import LIBRARY_DIR, read_yaml
    out = []
    for p in sorted(LIBRARY_DIR.glob("*.yaml")):
        raw = read_yaml(p.read_text()) or {}
        vars_ = {k: (v if isinstance(v, dict) else {"default": v}) for k, v in (raw.get("vars") or {}).items()}
        out.append({"name": p.stem, "description": raw.get("description", ""), "vars": vars_,
                    "loop": bool(raw.get("loop", False)), "file": str(p)})
    return out


def describe(name: str) -> dict:
    """Parameters / vars, docs and defaults of an atom or library effect."""
    from .atom import ATOMS
    from .atom import ensure_atoms
    ensure_atoms()
    for e in list_effects():
        if e["name"] == name:
            return {"type": "effect", **e}
    if name in ATOMS:
        return {"type": "atom", **ATOMS[name].info()}
    raise SpecError(f"unknown effect or atom {name!r}")


def json_schema() -> dict:
    """JSON Schema of the scene file format (top level inlined; nested scenes reference it)."""
    s = Scene.model_json_schema()
    ref = s.get("$ref", "")
    if ref.startswith("#/$defs/"):
        top = dict(s["$defs"][ref.split("/")[-1]])
        top["$defs"] = s["$defs"]
        top["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        top["title"] = "vfxgen scene"
        return top
    return s
