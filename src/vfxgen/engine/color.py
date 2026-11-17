"""Colour ramps ("palettes") and colour-spec parsing.

A palette is a list of stops ``[(position 0..1, (r, g, b)), ...]``.

* Light channels index the ramp by tone-mapped brightness (0 = faint edge,
  1 = white-hot core).
* Smoke / volume channels index it by lighting (0 = shadowed, 1 = lit).
* Solid channels index it by facet lighting / material position.

Anywhere a colour is accepted you may pass a *colour spec* (see
:func:`get_palette`).
"""
from __future__ import annotations

import colorsys
from typing import Sequence, Union

import numpy as np

Stops = list[tuple[float, tuple[float, float, float]]]
ColorSpec = Union[str, Sequence]

PALETTES: dict[str, Stops] = {
    # emissive
    "electric": [(0, (40, 70, 255)), (0.35, (80, 140, 255)), (0.7, (175, 215, 255)), (1, (255, 255, 255))],
    "fire": [(0, (170, 25, 0)), (0.35, (245, 95, 10)), (0.7, (255, 195, 70)), (1, (255, 255, 235))],
    "arcane": [(0, (100, 25, 210)), (0.4, (190, 70, 255)), (0.75, (255, 165, 255)), (1, (255, 255, 255))],
    "toxic": [(0, (20, 130, 30)), (0.4, (90, 225, 50)), (0.75, (205, 255, 130)), (1, (255, 255, 235))],
    "frost": [(0, (40, 150, 200)), (0.4, (110, 210, 240)), (0.75, (200, 245, 255)), (1, (255, 255, 255))],
    "holy": [(0, (200, 140, 30)), (0.4, (255, 205, 80)), (0.75, (255, 240, 180)), (1, (255, 255, 255))],
    "heal": [(0, (20, 170, 110)), (0.4, (70, 230, 150)), (0.75, (190, 255, 215)), (1, (255, 255, 255))],
    "blood": [(0, (120, 0, 10)), (0.4, (210, 20, 30)), (0.75, (255, 110, 100)), (1, (255, 230, 220))],
    "void": [(0, (60, 10, 120)), (0.4, (130, 30, 200)), (0.75, (230, 120, 255)), (1, (255, 240, 255))],
    "wind": [(0, (125, 165, 185)), (0.5, (200, 228, 240)), (1, (255, 255, 255))],
    "glint": [(0, (120, 190, 235)), (0.5, (205, 238, 255)), (1, (255, 255, 255))],
    "steel": [(0, (70, 110, 210)), (0.45, (150, 195, 255)), (0.8, (225, 240, 255)), (1, (255, 255, 255))],
    "impact": [(0, (230, 120, 40)), (0.4, (255, 200, 110)), (0.75, (255, 240, 200)), (1, (255, 255, 255))],
    # smoke / volume (index = lighting)
    "smoke": [(0, (62, 62, 68)), (0.5, (128, 128, 134)), (1, (214, 214, 220))],
    "dark_smoke": [(0, (22, 21, 23)), (0.5, (52, 50, 52)), (1, (110, 106, 104))],
    "explosion_smoke": [(0, (30, 26, 26)), (0.5, (74, 67, 64)), (1, (150, 140, 130))],
    "dust": [(0, (92, 72, 52)), (0.5, (160, 132, 100)), (1, (222, 200, 165))],
    "steam": [(0, (150, 160, 170)), (0.5, (205, 212, 220)), (1, (250, 252, 255))],
    "poison_gas": [(0, (30, 70, 20)), (0.5, (80, 150, 40)), (1, (170, 230, 110))],
    "abyss": [(0, (8, 0, 18)), (0.5, (22, 6, 40)), (1, (48, 16, 78))],
    "water": [(0, (14, 42, 78)), (0.45, (42, 100, 150)), (0.8, (130, 186, 220)), (1, (222, 242, 252))],
    # solid matter (index = facet lighting / material position)
    "ice": [(0, (28, 92, 150)), (1, (205, 240, 255))],
    "rock": [(0, (30, 22, 17)), (0.15, (40, 33, 28)), (0.6, (112, 94, 76)), (1, (150, 126, 96))],
}


class ColorError(ValueError):
    """An invalid colour spec."""


def register_palette(name: str, spec: ColorSpec, *, overwrite: bool = False) -> None:
    """Add a named palette (atom modules, scene ``palettes:`` blocks)."""
    if name in PALETTES and not overwrite:
        return
    PALETTES[name] = get_palette(spec)


def _hex(c: str) -> tuple[float, float, float]:
    c = c.strip().lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in c):
        raise ColorError(f"bad hex colour {c!r} (use #rgb or #rrggbb)")
    return tuple(float(int(c[i:i + 2], 16)) for i in (0, 2, 4))  # type: ignore[return-value]


def get_palette(spec: ColorSpec) -> Stops:
    """Resolve a colour spec to ramp stops.

    Accepted forms:

    * preset name: ``"fire"``
    * one hex colour: ``"#ff3300"`` (auto ramp: deep -> colour -> pale -> white)
    * hex list: ``"#300,#f30,#fc6,#fff"`` (evenly spaced, faint/dark first)
    * explicit positions: ``"0:#300,0.4:#f30,1:#fff"``
    * stops: ``[[0, [r, g, b]], ...]``
    """
    if isinstance(spec, (list, tuple)):
        try:
            stops = [(float(p), tuple(float(v) for v in c)) for p, c in spec]
        except (TypeError, ValueError) as e:
            raise ColorError(f"bad palette stops {spec!r}") from e
        if len(stops) < 2 or any(len(c) != 3 for _, c in stops):
            raise ColorError("a palette needs at least 2 stops of (position, (r, g, b))")
        return sorted(stops)  # type: ignore[return-value]
    s = str(spec).strip()
    if s in PALETTES:
        return PALETTES[s]
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts or not all(p.split(":")[-1].strip().startswith("#") for p in parts):
        raise ColorError(f"unknown palette {spec!r}; use a preset ({', '.join(sorted(PALETTES))}) "
                         "or hex colours like '#ff3300' or '#300,#f30,#fff'")
    if len(parts) == 1:
        c = np.array(_hex(parts[0]), np.float32)
        deep, pale = c * 0.7, c + (255 - c) * 0.6
        return [(0.0, tuple(map(float, deep))), (0.4, tuple(map(float, c))),
                (0.75, tuple(map(float, pale))), (1.0, (255.0, 255.0, 255.0))]
    stops: Stops = []
    for i, p in enumerate(parts):
        if ":" in p:
            pos, col = p.split(":", 1)
            stops.append((float(pos), _hex(col)))
        else:
            stops.append((i / (len(parts) - 1), _hex(p)))
    return sorted(stops)


def ramp(spec: ColorSpec, t) -> np.ndarray:
    """Sample a palette at positions ``t`` (array, 0..1) -> float32 RGB 0..255, shape (..., 3)."""
    stops = get_palette(spec)
    pos = np.array([s[0] for s in stops], np.float64)
    cols = np.array([s[1] for s in stops], np.float64)
    t = np.asarray(t, np.float64)
    return np.stack([np.interp(t, pos, cols[:, ch]) for ch in range(3)], axis=-1).astype(np.float32)


def color_at(spec: ColorSpec, k: float) -> tuple[float, float, float]:
    """One RGB colour (0..255) at ramp position k."""
    v = ramp(spec, np.array(float(np.clip(k, 0, 1))))
    return float(v[0]), float(v[1]), float(v[2])


def adjust_stops(spec: ColorSpec, hue: float = 0.0, saturation: float = 1.0,
                 brightness: float = 1.0) -> Stops:
    """Hue shift (degrees) and saturation / brightness multipliers applied to every stop."""
    out: Stops = []
    for p, (r, g, b) in get_palette(spec):
        h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
        h = (h + hue / 360.0) % 1.0
        s = min(max(s * saturation, 0.0), 1.0)
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        out.append((p, tuple(min(max(v * 255 * brightness, 0.0), 255.0) for v in (r2, g2, b2))))
    return out


def palette_key(spec: ColorSpec) -> str:
    """Stable identity of a colour spec (channel grouping, hashing)."""
    if isinstance(spec, str):
        return spec.strip()
    return repr([(round(p, 6), tuple(round(v, 3) for v in c)) for p, c in get_palette(spec)])
