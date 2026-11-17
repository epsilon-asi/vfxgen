"""Procedural stroke font (helper module: registers no atoms, no side effects).

Each glyph is a list of open polylines in *em units*: cap height 1 (y down, 0 = top of the
capitals, 1 = baseline) and its own advance width. Strokes are drawn as thick round-capped
capsules by the ``text`` atom, so the font needs no system fonts and stays crisp at any scale.

Supported: digits 0-9, capitals A-Z (lower-case letters are upper-cased), ``+ - ! % . :`` and
``x`` (lower-case x = multiplication sign, e.g. "x2"), space.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["GLYPHS", "SUPPORTED", "glyph", "layout"]


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int | None = None):
    """Points on an elliptical arc from angle a0 to a1 (degrees; 0 = right, 90 = down)."""
    if n is None:
        n = max(4, int(abs(a1 - a0) / 12) + 1)
    th = np.radians(np.linspace(a0, a1, n))
    return [(cx + rx * math.cos(t), cy + ry * math.sin(t)) for t in th]


def _P(*pts):
    return [tuple(map(float, p)) for p in pts]


def _build() -> dict[str, tuple[float, list[np.ndarray]]]:
    g: dict[str, tuple[float, list]] = {}
    # ---- digits (width 0.6) ------------------------------------------------------
    g["0"] = (0.62, [_arc(0.31, 0.5, 0.31, 0.5, -90, 270, 40)])
    g["1"] = (0.42, [_P((0.04, 0.2), (0.28, 0.0), (0.28, 1.0))])
    g["2"] = (0.6, [_arc(0.3, 0.29, 0.29, 0.29, 195, 355, 14) + _P((0.0, 1.0), (0.6, 1.0))])
    g["3"] = (0.6, [_arc(0.29, 0.25, 0.26, 0.25, 200, 450, 18),
                    _arc(0.29, 0.75, 0.31, 0.25, 270, 520, 18)])
    g["4"] = (0.64, [_P((0.48, 1.0), (0.48, 0.0), (0.0, 0.7), (0.64, 0.7))])
    g["5"] = (0.6, [_P((0.56, 0.0), (0.08, 0.0), (0.04, 0.47)) + _arc(0.3, 0.68, 0.3, 0.32, 218, 515, 22)])
    six = [_arc(0.3, 0.68, 0.3, 0.32, 0, 360, 30),
           _arc(0.62, 0.62, 0.62, 0.62, 180, 253, 8)]
    g["6"] = (0.6, six)
    g["9"] = (0.6, [[(0.6 - x, 1.0 - y) for x, y in s] for s in six])
    g["7"] = (0.6, [_P((0.0, 0.0), (0.6, 0.0), (0.2, 1.0))])
    g["8"] = (0.6, [_arc(0.3, 0.25, 0.25, 0.25, -90, 270, 26), _arc(0.3, 0.73, 0.3, 0.27, -90, 270, 30)])
    # ---- capitals ----------------------------------------------------------------
    g["A"] = (0.66, [_P((0.0, 1.0), (0.33, 0.0), (0.66, 1.0)), _P((0.13, 0.64), (0.53, 0.64))])
    g["B"] = (0.6, [_P((0.0, 0.5), (0.0, 0.0), (0.3, 0.0)) + _arc(0.3, 0.25, 0.25, 0.25, -90, 90, 12)
                    + _P((0.0, 0.5)),
                    _P((0.0, 0.5), (0.0, 1.0), (0.31, 1.0)) + _arc(0.31, 0.75, 0.29, 0.25, 90, -90, 12)
                    + _P((0.0, 0.5))])
    g["C"] = (0.62, [_arc(0.34, 0.5, 0.34, 0.5, -42, -318, 30)])
    g["D"] = (0.62, [_P((0.0, 0.0), (0.0, 1.0), (0.22, 1.0)) + _arc(0.22, 0.5, 0.4, 0.5, 90, -90, 18)
                     + _P((0.0, 0.0))])
    g["E"] = (0.54, [_P((0.54, 0.0), (0.0, 0.0), (0.0, 1.0), (0.54, 1.0)), _P((0.0, 0.5), (0.42, 0.5))])
    g["F"] = (0.52, [_P((0.52, 0.0), (0.0, 0.0), (0.0, 1.0)), _P((0.0, 0.5), (0.42, 0.5))])
    g["G"] = (0.64, [_arc(0.34, 0.5, 0.34, 0.5, -42, -360, 32) + _P((0.68, 0.56), (0.38, 0.56))])
    g["H"] = (0.6, [_P((0.0, 0.0), (0.0, 1.0)), _P((0.6, 0.0), (0.6, 1.0)), _P((0.0, 0.5), (0.6, 0.5))])
    g["I"] = (0.34, [_P((0.0, 0.0), (0.34, 0.0)), _P((0.17, 0.0), (0.17, 1.0)), _P((0.0, 1.0), (0.34, 1.0))])
    g["J"] = (0.56, [_P((0.56, 0.0), (0.56, 0.72)) + _arc(0.28, 0.72, 0.28, 0.28, 0, 180, 14)])
    g["K"] = (0.6, [_P((0.0, 0.0), (0.0, 1.0)), _P((0.58, 0.0), (0.0, 0.62)), _P((0.2, 0.45), (0.6, 1.0))])
    g["L"] = (0.52, [_P((0.0, 0.0), (0.0, 1.0), (0.52, 1.0))])
    g["M"] = (0.74, [_P((0.0, 1.0), (0.0, 0.0), (0.37, 0.62), (0.74, 0.0), (0.74, 1.0))])
    g["N"] = (0.62, [_P((0.0, 1.0), (0.0, 0.0), (0.62, 1.0), (0.62, 0.0))])
    g["O"] = (0.7, [_arc(0.35, 0.5, 0.35, 0.5, -90, 270, 40)])
    g["P"] = (0.58, [_P((0.0, 1.0), (0.0, 0.0), (0.3, 0.0)) + _arc(0.3, 0.27, 0.28, 0.27, -90, 90, 12)
                     + _P((0.0, 0.54))])
    g["Q"] = (0.7, [_arc(0.35, 0.5, 0.35, 0.5, -90, 270, 40), _P((0.42, 0.7), (0.72, 1.04))])
    g["R"] = (0.6, [_P((0.0, 1.0), (0.0, 0.0), (0.3, 0.0)) + _arc(0.3, 0.27, 0.28, 0.27, -90, 90, 12)
                    + _P((0.0, 0.54)), _P((0.28, 0.54), (0.6, 1.0))])
    g["S"] = (0.6, [_arc(0.3, 0.26, 0.28, 0.26, -25, -270, 18) + _arc(0.3, 0.75, 0.3, 0.25, -90, 155, 18)])
    g["T"] = (0.6, [_P((0.0, 0.0), (0.6, 0.0)), _P((0.3, 0.0), (0.3, 1.0))])
    g["U"] = (0.6, [_P((0.0, 0.0)) + _arc(0.3, 0.68, 0.3, 0.32, 180, 0, 16) + _P((0.6, 0.0))])
    g["V"] = (0.64, [_P((0.0, 0.0), (0.32, 1.0), (0.64, 0.0))])
    g["W"] = (0.84, [_P((0.0, 0.0), (0.2, 1.0), (0.42, 0.34), (0.64, 1.0), (0.84, 0.0))])
    g["X"] = (0.62, [_P((0.0, 0.0), (0.62, 1.0)), _P((0.62, 0.0), (0.0, 1.0))])
    g["Y"] = (0.62, [_P((0.0, 0.0), (0.31, 0.52), (0.62, 0.0)), _P((0.31, 0.52), (0.31, 1.0))])
    g["Z"] = (0.6, [_P((0.0, 0.0), (0.6, 0.0), (0.0, 1.0), (0.6, 1.0))])
    # ---- symbols -----------------------------------------------------------------
    g["+"] = (0.6, [_P((0.03, 0.55), (0.57, 0.55)), _P((0.3, 0.28), (0.3, 0.82))])
    g["-"] = (0.5, [_P((0.04, 0.55), (0.46, 0.55))])
    g["x"] = (0.54, [_P((0.06, 0.36), (0.48, 0.88)), _P((0.48, 0.36), (0.06, 0.88))])
    g["!"] = (0.16, [_P((0.08, 0.0), (0.08, 0.64)), _P((0.08, 0.98), (0.08, 0.98))])
    g["%"] = (0.74, [_arc(0.15, 0.2, 0.14, 0.17, -90, 270, 16), _P((0.66, 0.0), (0.08, 1.0)),
                     _arc(0.59, 0.8, 0.14, 0.17, -90, 270, 16)])
    g["."] = (0.16, [_P((0.08, 0.98), (0.08, 0.98))])
    g[":"] = (0.16, [_P((0.08, 0.4), (0.08, 0.4)), _P((0.08, 0.98), (0.08, 0.98))])
    g[" "] = (0.36, [])
    return {k: (w, [np.asarray(s, np.float32) for s in strokes]) for k, (w, strokes) in g.items()}


GLYPHS = _build()
SUPPORTED = "".join(sorted(GLYPHS))


def glyph(ch: str):
    """(advance width, strokes) for one character, upper-casing letters except ``x``."""
    if ch not in GLYPHS and ch.upper() in GLYPHS:
        ch = ch.upper()
    if ch not in GLYPHS:
        raise KeyError(ch)
    return GLYPHS[ch]


def layout(text: str, size: float, weight: float, spacing: float):
    """Lay a string out on one line, centred on (0, 0) (cap middle).

    Returns a list of (char, strokes in local px, centre (x, y), advance px) for every
    non-space character, plus the total width. ``weight`` is the stroke radius in px,
    ``spacing`` extra px between characters."""
    items = []
    x = 0.0
    for ch in text:
        w, strokes = glyph(ch)
        gw = w * size
        if strokes:
            pts = [np.column_stack([s[:, 0] * size + x + weight, s[:, 1] * size]) for s in strokes]
            items.append([ch, pts, (x + weight + gw / 2, size / 2), gw + 2 * weight])
        x += gw + 2 * weight + spacing
    total = max(x - spacing, 0.0)
    out = []
    for ch, pts, (cx, cy), adv in items:
        dx, dy = -total / 2, -size / 2
        out.append((ch, [p + np.array([dx, dy], np.float32) for p in pts], (cx + dx, cy + dy), adv))
    return out, total
