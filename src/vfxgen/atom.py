"""Atom authoring API.

An *atom* is a building block implemented in Python. It draws itself at any
local time ``t`` (seconds) into one or more output *roles* (light, smoke,
volume, solid), in its own local coordinates.

Example::

    from vfxgen.atom import Atom, atom, P, Light

    @atom("shockwave", tags=["ring"])
    class Shockwave(Atom):
        '''Expanding ring of light that fades as it grows.'''
        kind = "oneshot"
        bounds = (-140, -140, 140, 140)
        outputs = {"light": Light(palette="fire")}
        streams = ()

        class Params(Atom.Params):
            duration: float = P(0.35, gt=0, doc="seconds until faded")
            radius: float = P(120, ge=0, doc="final radius (px)")

        def draw(self, ctx, t):
            p = t / self.p.duration
            ctx.light.ring(0, 0, self.p.radius * p, 3, 0.9 * (1 - p) ** 2)

Contract (enforced by the shared atom tests):

* All randomness comes from ``self.rng(stream)`` (in ``setup``) or
  ``self.tick_rng(stream, t, rate)`` (in ``draw``). Declare stream names in
  ``streams``.
* ``draw`` depends only on ``t``: no frame index, no accumulated state.
* one-shots draw nothing for ``t >= duration``; loops satisfy
  ``draw(t) == draw(t + period)``.
* Drawing stays inside ``bounds`` (local units).
"""
from __future__ import annotations

import math
from typing import Any, Callable, ClassVar, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .engine.mathx import derive_seed
from .errors import SpecError

__all__ = ["Atom", "atom", "P", "Light", "Smoke", "Volume", "Solid", "ATOMS", "get_atom", "Role"]

ATOMS: dict[str, type["Atom"]] = {}


def P(default: Any, *, doc: str, ge: float | None = None, le: float | None = None,
      gt: float | None = None, lt: float | None = None, choices: list | None = None) -> Any:
    """Declare an atom parameter: default, documentation and bounds."""
    extra = {"choices": choices} if choices else None
    return Field(default, description=doc, ge=ge, le=le, gt=gt, lt=lt, json_schema_extra=extra)


class Role:
    """An output of an atom: the channel kind it deposits into and the default channel
    settings used when the scene does not declare that channel."""

    def __init__(self, kind: str, **settings: Any):
        self.kind = kind
        self.settings = settings

    def channel_spec(self) -> dict:
        return {"kind": self.kind, **self.settings}

    def __repr__(self) -> str:
        return f"{self.kind.title()}({', '.join(f'{k}={v!r}' for k, v in self.settings.items())})"


def Light(**settings: Any) -> Role:
    return Role("light", **settings)


def Smoke(**settings: Any) -> Role:
    return Role("smoke", **settings)


def Volume(**settings: Any) -> Role:
    return Role("volume", **settings)


def Solid(**settings: Any) -> Role:
    return Role("solid", **settings)


STANDARD_ANCHORS = {
    "center": (0.5, 0.5), "top": (0.5, 0.0), "bottom": (0.5, 1.0), "left": (0.0, 0.5),
    "right": (1.0, 0.5), "top_left": (0.0, 0.0), "top_right": (1.0, 0.0),
    "bottom_left": (0.0, 1.0), "bottom_right": (1.0, 1.0),
}


class AtomParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Atom:
    """Base class for building blocks. Subclass, set the class attributes, and decorate with
    :func:`atom`."""

    name: ClassVar[str] = ""
    tags: ClassVar[tuple[str, ...]] = ()
    kind: ClassVar[str] = "oneshot"                     # "oneshot" | "loop"
    bounds: ClassVar[tuple[float, float, float, float]] = (-128, -128, 128, 128)
    anchors: ClassVar[dict[str, Any]] = {}             # name -> (x, y) or callable(self, t)
    outputs: ClassVar[dict[str, Role]] = {"light": Role("light")}
    streams: ClassVar[tuple[str, ...]] = ()

    Params: ClassVar[type[AtomParams]] = AtomParams

    def __init__(self, params: AtomParams | dict | None = None, seed: int = 0,
                 seeds: Optional[dict[str, int]] = None, base_dir=None):
        from pathlib import Path as _P
        self.base_dir = _P(base_dir) if base_dir is not None else _P.cwd()
        self.p = params if isinstance(params, AtomParams) else self.validate_params(params or {})
        self.seed = int(seed)
        self._seed_overrides = dict(seeds or {})
        unknown = set(self._seed_overrides) - set(self.streams)
        if unknown:
            raise SpecError(f"atom `{self.name}` has no random stream(s) {sorted(unknown)}; "
                            f"streams: {list(self.streams) or 'none'}")
        self.setup()

    # -- class helpers -----------------------------------------------------------
    @classmethod
    def validate_params(cls, params: dict, path: str = "params") -> AtomParams:
        try:
            model = cls.Params(**params)
        except ValidationError as e:
            msgs = []
            for err in e.errors():
                loc = ".".join(str(x) for x in err["loc"])
                if err["type"] == "extra_forbidden":
                    msgs.append(f"unknown parameter `{loc}` for `{cls.name}`; valid: "
                                f"{', '.join(cls.Params.model_fields)}")
                else:
                    msgs.append(f"{loc}: {err['msg']}")
            raise SpecError("; ".join(msgs), path=path) from None
        for k, f in cls.Params.model_fields.items():
            extra = f.json_schema_extra if isinstance(f.json_schema_extra, dict) else {}
            ch = extra.get("choices")
            if ch and getattr(model, k) not in ch:
                raise SpecError(f"{k}: {getattr(model, k)!r} is not one of {ch}", path=path)
        return model

    @classmethod
    def main_role(cls) -> str:
        return next(iter(cls.outputs))

    @classmethod
    def info(cls) -> dict:
        fields = {}
        for k, f in cls.Params.model_fields.items():
            entry = {"default": f.default, "doc": f.description or "",
                     "type": getattr(f.annotation, "__name__", str(f.annotation))}
            for m in f.metadata:
                for attr, key in (("ge", "min"), ("gt", "min_exclusive"), ("le", "max"), ("lt", "max_exclusive")):
                    if getattr(m, attr, None) is not None:
                        entry[key] = getattr(m, attr)
            extra = f.json_schema_extra if isinstance(f.json_schema_extra, dict) else {}
            if extra.get("choices"):
                entry["choices"] = list(extra["choices"])
            fields[k] = entry
        try:
            bounds = list(cls(None, seed=0).get_bounds())
        except Exception:  # pragma: no cover - info must never fail
            bounds = list(cls.bounds)
        return {"name": cls.name, "doc": (cls.__doc__ or "").strip(), "kind": cls.kind,
                "tags": list(cls.tags), "bounds": [round(b, 2) for b in bounds],
                "anchors": sorted(set(STANDARD_ANCHORS) | set(cls.anchors)),
                "outputs": {r: role.channel_spec() for r, role in cls.outputs.items()},
                "streams": list(cls.streams), "params": fields}

    # -- assets ------------------------------------------------------------------------
    def asset(self, path: str):
        """Load an image param (your sprite / texture) relative to the scene file: premultiplied
        float RGBA (h, w, 4)."""
        from .engine.assets import load_image
        return load_image(path, self.base_dir)

    # -- randomness ------------------------------------------------------------------
    def stream_seed(self, name: str) -> int:
        if name in self._seed_overrides:
            return int(self._seed_overrides[name])
        return derive_seed(self.seed, "stream", name)

    def stream_seeds(self) -> dict[str, int]:
        """Seeds actually used for every declared stream (written to explicit YAML)."""
        return {s: self.stream_seed(s) for s in self.streams}

    def rng(self, name: str) -> np.random.Generator:
        """A fresh generator for stream ``name``. Fetch each stream once in ``setup``."""
        if name not in self.streams:
            raise RuntimeError(f"atom `{self.name}` uses undeclared random stream {name!r}; "
                               f"add it to `streams`")
        return np.random.default_rng(self.stream_seed(name))

    def tick_rng(self, name: str, t: float, rate: float) -> np.random.Generator:
        """Randomness that changes ``rate`` times per second (flicker, re-jag). Loops wrap the
        tick index by the period so the loop stays seamless."""
        tick = int(math.floor(t * rate + 1e-9))
        if self.kind == "loop":
            n = max(1, int(round(self.length() * rate)))
            tick %= n
        return np.random.default_rng(derive_seed(self.stream_seed(name), "tick", tick))

    # -- geometry / time ----------------------------------------------------------------
    def length(self) -> float:
        """Natural length in seconds: ``duration`` (one-shot) or ``period`` (loop)."""
        return float(getattr(self.p, "period" if self.kind == "loop" else "duration"))

    def get_bounds(self) -> tuple[float, float, float, float]:
        """Local bounding box; override when it depends on params."""
        return tuple(float(v) for v in self.bounds)  # type: ignore[return-value]

    def anchor_point(self, spec: Any, t: float = 0.0) -> tuple[float, float]:
        """Resolve an anchor spec (name, [u, v], {uv}, {px}) to local coordinates."""
        x0, y0, x1, y1 = self.get_bounds()
        return resolve_anchor(spec, (x0, y0, x1, y1), self._named_anchor, t, owner=self.name)

    def _named_anchor(self, name: str, t: float):
        if name in self.anchors:
            a = self.anchors[name]
            return a(self, t) if callable(a) else (float(a[0]), float(a[1]))
        return None

    # -- to implement -------------------------------------------------------------------
    def setup(self) -> None:
        """Precompute everything (and draw all non-tick randomness) here."""

    def draw(self, ctx: "DrawContext", t: float) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


def resolve_anchor(spec: Any, box: tuple[float, float, float, float],
                   named: Callable[[str, float], Optional[tuple[float, float]]] | None,
                   t: float = 0.0, owner: str = "") -> tuple[float, float]:
    x0, y0, x1, y1 = box
    if hasattr(spec, "model_dump"):
        spec = spec.model_dump()
    if isinstance(spec, str):
        if named is not None:
            p = named(spec, t)
            if p is not None:
                return p
        if spec in STANDARD_ANCHORS:
            u, v = STANDARD_ANCHORS[spec]
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * v
        raise SpecError(f"unknown anchor {spec!r} for `{owner}`")
    if isinstance(spec, dict) and "px" in spec:
        return float(spec["px"][0]), float(spec["px"][1])
    if isinstance(spec, dict) and "uv" in spec:
        spec = spec["uv"]
    if isinstance(spec, (list, tuple)) and len(spec) == 2:
        u, v = float(spec[0]), float(spec[1])
        return x0 + (x1 - x0) * u, y0 + (y1 - y0) * v
    raise SpecError(f"bad anchor {spec!r}")


class DrawContext:
    """What ``Atom.draw`` receives: one view per output role (``ctx.light``, ``ctx.smoke``,
    ``ctx.volume``, ``ctx.solid`` or custom role names), plus ``ctx.px`` (output pixels per
    local unit, useful for level-of-detail)."""

    def __init__(self, views: dict[str, Any], px: float, target=None, t_scene: float = 0.0):
        self._views = views
        self.px = px
        self.t_scene = t_scene            # scene time of this frame (seconds)
        self._target = target

    def target(self, name: str, dt: float = 0.0):
        """Local position (x, y) of a linked layer's anchor ``dt`` seconds ago (layer ``targets:``),
        or None when that layer is inactive then. Use for chains, tethers and beams."""
        if self._target is None:
            raise KeyError(f"this layer has no targets (add `targets: {{{name}: {{layer: ...}}}}`)")
        return self._target(name, dt)

    def follow(self, dt: float = 0.0):
        """Local position of the followed layer's anchor ``dt`` seconds ago (layer ``follow:``),
        for trails and ribbons. None when it was inactive."""
        return self.target("follow", dt)

    def has_target(self, name: str) -> bool:
        return self._target is not None and self._safe_has(name)

    def _safe_has(self, name):
        try:
            self._target(name, 0.0)
            return True
        except KeyError:
            return False

    def __getattr__(self, name: str):
        try:
            return self._views[name]
        except KeyError:
            raise AttributeError(f"no output role {name!r} on this atom") from None

    def has(self, role: str) -> bool:
        return role in self._views


def atom(name: str, tags: list[str] | tuple[str, ...] = ()) -> Callable[[type[Atom]], type[Atom]]:
    """Class decorator registering an atom under ``name``."""

    def deco(cls: type[Atom]) -> type[Atom]:
        if name in ATOMS and ATOMS[name] is not cls and ATOMS[name].__qualname__ != cls.__qualname__:
            raise ValueError(f"atom {name!r} is already registered by {ATOMS[name].__module__}")
        cls.name = name
        cls.tags = tuple(tags)
        need = "period" if cls.kind == "loop" else "duration"
        if cls.kind not in ("oneshot", "loop"):
            raise TypeError(f"atom {name}: kind must be 'oneshot' or 'loop'")
        if need not in cls.Params.model_fields:
            raise TypeError(f"atom {name}: {cls.kind} atoms need a `{need}` parameter (seconds)")
        for f, info in cls.Params.model_fields.items():
            if not info.description:
                raise TypeError(f"atom {name}: parameter `{f}` needs a doc (use P(..., doc=...))")
        ATOMS[name] = cls
        return cls

    return deco


def ensure_atoms() -> None:
    """Import the built-in atom modules (and user atoms on VFXGEN_PATH)."""
    import importlib
    mod = importlib.import_module("vfxgen.atoms")
    mod.load_user_atoms()


def get_atom(name: str) -> type[Atom]:
    ensure_atoms()
    try:
        return ATOMS[name]
    except KeyError:
        raise SpecError(f"unknown atom {name!r}") from None
