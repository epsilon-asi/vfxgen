"""Typed scene model (pydantic). YAML and JSON both load into these classes; the
JSON Schema of :class:`Scene` documents the whole file format.

Static ``${...}`` expressions are substituted before validation (see
:mod:`vfxgen.spec.expr`); time-dependent expressions stay strings and are
evaluated per frame.
"""
from __future__ import annotations

from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --- small value types ---------------------------------------------------------
class Keys(_M):
    """Keyframes: ``[[t, value], [t, value, ease], ...]`` with t in layer-local seconds.
    ``ease`` (applied from that key to the next): linear, in, out, in_out, step, sine, back,
    bounce, power:N."""
    keys: list[list[Any]] = Field(min_length=1)

    @field_validator("keys")
    @classmethod
    def _check(cls, v):
        for k in v:
            if len(k) not in (2, 3):
                raise ValueError("each key is [t, value] or [t, value, ease]")
        return sorted(v, key=lambda k: float(k[0]))


Expr = Annotated[str, Field(pattern=r"^\s*\$\{.*\}\s*$", description="time expression, e.g. '${360*t}'")]
Anim = Union[float, Keys, Expr]                              # animatable scalar
Vec2 = Annotated[list[float], Field(min_length=2, max_length=2)]
ExprVec = Annotated[list[Union[float, Expr]], Field(min_length=2, max_length=2)]
AnimVec = Union[Vec2, ExprVec, Keys, Expr]                     # animatable 2-vector


class UV(_M):
    """Fraction of a box: [0, 0] = top-left, [1, 1] = bottom-right."""
    uv: Vec2


class PX(_M):
    """Exact pixel in the effect's own coordinate space."""
    px: Vec2


Position = Union[Vec2, ExprVec, UV, Keys, Expr]
AnchorSpec = Union[str, Vec2, UV, PX]
ColorSpec = Union[str, list[Any]]


# --- paths ---------------------------------------------------------------------
class PathSpec(_M):
    """A path. Exactly one shape key must be set."""
    line: Optional[list[Vec2]] = None
    polyline: Optional[list[Vec2]] = None
    bezier: Optional[list[Vec2]] = None
    arc: Optional[dict[str, Any]] = None        # center, radius|[rx,ry], from, to (deg)
    circle: Optional[dict[str, Any]] = None     # center, radius
    ellipse: Optional[dict[str, Any]] = None    # center, radius [rx, ry], rotate
    spiral: Optional[dict[str, Any]] = None     # center, r0, r1, turns, from
    points: Optional[list[Vec2]] = None
    closed: bool = False
    spacing: Literal["even", "random"] = "even"

    @model_validator(mode="after")
    def _one(self):
        keys = [k for k in ("line", "polyline", "bezier", "arc", "circle", "ellipse", "spiral", "points")
                if getattr(self, k) is not None]
        if len(keys) != 1:
            raise ValueError("a path needs exactly one of: line, polyline, bezier, arc, circle, "
                             f"ellipse, spiral, points (got {keys or 'none'})")
        return self

    @property
    def shape(self) -> str:
        return next(k for k in ("line", "polyline", "bezier", "arc", "circle", "ellipse", "spiral", "points")
                    if getattr(self, k) is not None)


class Move(_M):
    """Animate ``at`` along a path."""
    path: PathSpec
    duration: Optional[float] = Field(None, gt=0, description="seconds to traverse (default: layer length)")
    delay: float = 0.0
    ease: str = "linear"
    loop: bool = False


class Attach(_M):
    """Follow another layer's anchor (in this scene). ``rotate: true`` also inherits that layer's
    rotation (added to this layer's own ``rotate``)."""
    layer: str
    anchor: AnchorSpec = "center"
    rotate: bool = False


class Jitter(_M):
    position: float = Field(0, ge=0, description="± px")
    rotate: float = Field(0, ge=0, description="± degrees")
    scale: float = Field(0, ge=0, description="± fraction (0.2 = 0.8..1.2)")
    time: float = Field(0, ge=0, description="± seconds")
    speed: float = Field(0, ge=0, description="± fraction")
    hue: float = Field(0, ge=0, description="± degrees of hue shift")


class Ramp(_M):
    """Linear interpolation from the first copy to the last."""
    scale: Optional[Vec2] = None
    opacity: Optional[Vec2] = None
    rotate: Optional[Vec2] = None


class Repeat(_M):
    count: int = Field(ge=1)
    every: float = Field(0.0, ge=0, description="seconds between copies")
    along: Optional[PathSpec] = None
    align: bool = False
    order: Literal["forward", "reverse", "random", "pingpong"] = "forward"
    jitter: Jitter = Field(default_factory=Jitter)
    ramp: Ramp = Field(default_factory=Ramp)
    new_seed: bool = True
    seeds: Optional[list[int]] = Field(None, description="explicit per-copy seeds")


class Spawn(_M):
    effect: str = "self"
    depth: int = Field(1, ge=1, le=3)
    count: int = Field(3, ge=1)
    at: Optional[PathSpec] = Field(None, description="placement relative to the parent anchor")
    radius: float = Field(60.0, ge=0, description="placement radius when `at` is omitted")
    when: Union[float, Vec2] = 0.15
    scale: float = Field(0.5, gt=0)
    opacity: float = Field(1.0, ge=0)
    jitter: Jitter = Field(default_factory=Jitter)
    params: dict[str, Any] = Field(default_factory=dict)
    seeds: Optional[list[int]] = None


class Fade(_M):
    in_: float = Field(0.0, ge=0, alias="in")
    out: float = Field(0.0, ge=0)


class TimeWarp(_M):
    """Remap local time over ``span`` seconds (default: the effect's natural length):
    ``t' = span * (offset + (1 - offset) * curve(t / span))``. ``curve: power`` uses ``exp``."""
    curve: str = "linear"
    exp: float = Field(1.0, gt=0)
    offset: float = Field(0.0, ge=0, lt=1)
    span: Optional[float] = Field(None, gt=0)


class Adjust(_M):
    hue: float = 0.0
    saturation: float = Field(1.0, ge=0)
    brightness: float = Field(1.0, ge=0)


class Mask(_M):
    """Scalar field multiplied into a layer's deposits (layer-local coordinates).

    * sweep: bright wedge rotating around ``center`` (``turns`` per second, ``width`` deg,
      ``floor`` = minimum).
    * radial: ``inner``..``outer`` radius ramp around ``center`` (``invert``).
    * linear: ramp from ``start`` to ``end`` point.
    * noise: fbm threshold (``scale`` px, ``threshold``, ``softness``, ``speed``).
    """
    type: Literal["sweep", "radial", "linear", "noise"]
    profile: Literal["trail", "lobe"] = Field("trail", description="sweep: trailing wedge or symmetric lobe")
    center: Vec2 = Field(default_factory=lambda: [0.0, 0.0])
    start: Optional[Vec2] = None
    end: Optional[Vec2] = None
    inner: float = 0.0
    outer: float = 100.0
    turns: float = 1.0
    width: float = 90.0
    floor: float = 0.0
    scale: float = 64.0
    threshold: float = 0.0
    softness: float = 0.3
    speed: float = 0.0
    invert: bool = False
    seed: int = 0


# --- channels ------------------------------------------------------------------
class _Channel(_M):
    palette: Optional[ColorSpec] = None
    palette_keys: Optional[list[list[Any]]] = Field(
        None, description="colour over time: [[scene_seconds, colour spec], ...] interpolated (overrides palette)")
    blend: Literal["over", "add", "screen"] = "over"
    opacity: float = Field(1.0, ge=0)
    adjust: Optional[Adjust] = None


class GlowFrom(_M):
    channel: str
    amount: float = 0.25
    blur: float = 5.0


class LitBy(_M):
    """Light from light channels falling on this channel.

    * ``tint``: colour the matter near light (legacy ``fire_light``).
    * ``surface``: light the matter's surface from the direction of the light, using its normals
      (volume, and smoke with ``relief`` > 0); falls back to tint elsewhere."""
    channels: list[str]
    amount: float = 0.8
    mode: Literal["tint", "surface"] = "tint"
    radius: float = Field(0.0, ge=0, description="light spread (px); 0 = automatic from size_ref")


class LightChannel(_Channel):
    kind: Literal["light"] = "light"
    bloom: list[Vec2] = Field(default_factory=lambda: [[1.5, 0.6], [5, 0.5], [14, 0.35]])
    exposure: float = Field(1.0, ge=0)
    alpha_gamma: float = Field(0.85, gt=0)
    glow_from: Optional[GlowFrom] = None


class SmokeChannel(_Channel):
    kind: Literal["smoke"] = "smoke"
    density: float = Field(1.0, ge=0, description="density -> alpha gain")
    shadow: float = 1.2
    light_dir: Vec2 = Field(default_factory=lambda: [-0.55, -0.85])
    relief: float = Field(0.0, ge=0, description="0 = flat smoke; >0 = billow relief lighting")
    ambient: float = 0.18
    height_gain: float = 0.8
    lit_by: Optional[LitBy] = None
    size_ref: float = Field(256, gt=0, description="reference size for shading blur scales (px)")


class VolumeChannel(_Channel):
    kind: Literal["volume"] = "volume"
    density: float = Field(1.0, ge=0)
    bump: float = 6.0
    soft: float = 1.2
    shadow: float = 1.0
    wrap: float = 0.35
    spec: float = 0.0
    spec_power: float = 24.0
    rim: float = 0.0
    light: list[float] = Field(default_factory=lambda: [-0.5, -0.72, 0.48], min_length=3, max_length=3)
    size_ref: float = Field(256, gt=0)
    lit_by: Optional[LitBy] = None


class SolidChannel(_Channel):
    kind: Literal["solid"] = "solid"
    lit_by: Optional[LitBy] = None
    size_ref: float = Field(256, gt=0)


class DistortChannel(_Channel):
    """Not drawn in colour: holds a scalar 'refraction strength' field. Exported as a distortion
    map (RG = displacement direction, A = strength) for engine heat-haze / shockwave shaders."""
    kind: Literal["distort"] = "distort"
    blur: float = Field(2.0, ge=0, description="smoothing before taking the gradient (px)")
    strength: float = Field(1.0, ge=0)


Channel = Annotated[Union[LightChannel, SmokeChannel, VolumeChannel, SolidChannel, DistortChannel],
                    Field(discriminator="kind")]


# --- layers & scenes -----------------------------------------------------------
class InlineScene(_M):
    scene: "Scene"


class Event(_M):
    """A gameplay event: ``t`` seconds (scene time for scene events, layer-local for layer
    events). Exported to the atlas JSON and as Unity AnimationEvents."""
    name: str
    t: float = Field(0.0, ge=0)
    data: dict[str, Any] = Field(default_factory=dict)


class Track(_M):
    """Record a layer anchor's scene position every frame (attach points, projectile head...)."""
    name: str
    layer: str
    anchor: AnchorSpec = "center"


class TargetRef(_M):
    layer: str
    anchor: AnchorSpec = "center"


class Hitstop(_M):
    at: float = Field(ge=0, description="local seconds")
    hold: float = Field(0.06, gt=0, description="seconds frozen")


class Pulse(_M):
    at: float = Field(0.0, ge=0)
    duration: float = Field(0.25, gt=0)
    amount: float = 0.25


class Anticipation(_M):
    duration: float = Field(0.12, gt=0, description="wind-up length at the start of the layer (s)")
    amount: float = Field(0.15, description="scale pull-back")
    offset: Vec2 = Field(default_factory=lambda: [0.0, 0.0], description="pull-back offset (px)")


class Shake(_M):
    at: float = Field(0.0, ge=0)
    duration: float = Field(0.3, gt=0)
    amplitude: float = Field(4.0, ge=0)
    frequency: float = Field(30.0, gt=0)


class Feel(_M):
    """Game-feel timing presets applied on top of the layer's own animation."""
    hitstop: list[Hitstop] = Field(default_factory=list)
    anticipation: Optional[Anticipation] = None
    squash: list[Pulse] = Field(default_factory=list, description="volume-preserving squash & stretch")
    squash_axis: Literal["x", "y"] = "y"
    punch: list[Pulse] = Field(default_factory=list, description="scale pop with overshoot")
    shake: list[Shake] = Field(default_factory=list)


class PixelArt(_M):
    """Pixel-art output: render natively at 1/size resolution, snap to a palette with ordered
    dithering, hard alpha, optional 1 px outline."""
    size: int = Field(4, ge=1, le=32, description="scene pixels per art pixel")
    palette: Optional[Union[str, list[str]]] = Field(None, description="hex colours; default: automatic")
    colors: int = Field(16, ge=2, le=256, description="automatic palette size")
    dither: Literal["none", "bayer2", "bayer4", "bayer8"] = "bayer4"
    dither_strength: float = Field(0.5, ge=0, le=2)
    alpha_threshold: float = Field(0.5, gt=0, lt=1)
    outline: Optional[str] = Field(None, description="hex colour of a 1 px outline")
    fps: Optional[float] = Field(None, gt=0, description="frame rate for the pixel version (e.g. 12)")


class MotionBlur(_M):
    samples: int = Field(5, ge=1, le=32)
    shutter: float = Field(0.5, gt=0, le=1, description="fraction of the frame interval")


class Layer(_M):
    id: Optional[str] = Field(None, pattern=r"^[A-Za-z_][A-Za-z0-9_\-]*$")
    effect: Union[str, InlineScene]
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    colors: Optional[ColorSpec] = None
    palettes: dict[str, ColorSpec] = Field(default_factory=dict, description="role -> colour spec")
    adjust: Optional[Adjust] = None
    # placement
    at: Optional[Position] = None
    anchor: AnchorSpec = "center"
    offset: Optional[AnimVec] = None
    rotate: Anim = 0.0
    scale: Union[Anim, Vec2] = 1.0
    flip: Optional[Literal["x", "y", "xy"]] = None
    opacity: Anim = 1.0
    move: Optional[Move] = None
    face_motion: bool = False
    attach: Optional[Attach] = None
    mask: Optional[Mask] = None
    # time
    start: float = 0.0
    length: Optional[float] = Field(None, gt=0)
    speed: float = Field(1.0, gt=0)
    time_mode: Optional[Literal["once", "loop", "hold", "pingpong"]] = None
    time_warp: Optional[TimeWarp] = None
    fade: Optional[Fade] = None
    # routing / order / randomness
    channels: dict[str, str] = Field(default_factory=dict, description="atom role -> scene channel")
    depth: int = Field(0, description="draw-order slice across channels: higher draws in front "
                                      "(e.g. smoke depth 1 in front of flames at depth 0)")
    z: int = 0
    isolate: bool = False
    seed: Optional[int] = None
    seeds: dict[str, int] = Field(default_factory=dict)
    repeat: Optional[Repeat] = None
    spawn: Optional[Spawn] = None
    # game feel, events, links to other layers
    feel: Optional[Feel] = None
    events: list[Event] = Field(default_factory=list)
    targets: dict[str, TargetRef] = Field(default_factory=dict,
                                          description="named links to other layers' anchors (chains, beams)")
    follow: Optional[TargetRef] = Field(None, description="the layer this one trails behind (ribbons)")

    @model_validator(mode="after")
    def _exclusive(self):
        if self.move is not None and self.attach is not None:
            raise ValueError("use either `move` or `attach`, not both")
        if self.repeat is not None and self.repeat.along is not None and self.move is not None:
            raise ValueError("`repeat.along` places copies on a path; it cannot be combined with `move`")
        return self


class VarSpec(_M):
    default: Any
    doc: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    choices: Optional[list[Any]] = None


class Scene(_M):
    vfxgen: int = 1
    name: Optional[str] = None
    description: str = ""
    size: Union[Annotated[list[int], Field(min_length=2, max_length=2)], Literal["auto"]] = Field(
        default_factory=lambda: [256, 256])
    fps: float = Field(24.0, gt=0)
    duration: Union[float, Literal["auto"]] = "auto"
    loop: bool = False
    seed: int = 1
    edge_fade: float = Field(10.0, ge=0, description="px of soft fade at the frame border")
    vars: dict[str, VarSpec] = Field(default_factory=dict)
    palettes: dict[str, ColorSpec] = Field(default_factory=dict)
    anchors: dict[str, AnchorSpec] = Field(default_factory=dict)
    channels: dict[str, Channel] = Field(default_factory=dict)
    layers: list[Layer] = Field(default_factory=list)
    meta: Optional[dict[str, Any]] = None
    defs: Optional[dict[str, Any]] = Field(None, description="free area for YAML anchors (&name) reused by layers")
    blend: Literal["alpha", "additive"] = Field(
        "alpha", description="how the effect is meant to be blended in-game (frames are always straight alpha)")
    pivot: AnchorSpec = Field("center", description="the effect's pivot in its frame (name, [u, v], {uv}, {px})")
    motion_blur: Optional[MotionBlur] = None
    pixel: Optional[PixelArt] = Field(None, description="render as pixel art")
    theme: Optional[dict[str, ColorSpec]] = Field(
        None, description="recolour channels: {channel name or kind: colour ramp}")
    rotate: float = Field(0.0, description="rotate the whole effect about the frame centre (degrees)")
    ends_visible: bool = Field(False, description="one-shot that intentionally ends on a visible frame "
                                                  "(e.g. a sprite effect that leaves the character)")
    tile: bool = Field(False, description="seamlessly tileable output: drawing wraps around the edges")
    events: list[Event] = Field(default_factory=list)
    track: list[Track] = Field(default_factory=list)

    @field_validator("vars", mode="before")
    @classmethod
    def _vars(cls, v):
        out = {}
        for k, spec in (v or {}).items():
            out[k] = spec if isinstance(spec, dict) and "default" in spec else {"default": spec}
        return out

    @field_validator("duration")
    @classmethod
    def _dur(cls, v):
        if v != "auto" and v <= 0:
            raise ValueError("duration must be > 0 or 'auto'")
        return v


InlineScene.model_rebuild()
Layer.model_rebuild()
