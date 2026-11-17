"""Shared test helpers."""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("VFXGEN_NO_CACHE", "1")     # tests exercise real renders

from vfxgen.atom import ATOMS, DrawContext, ensure_atoms
from vfxgen.engine.canvas import SolidView, Target, View
from vfxgen.engine.color import get_palette
from vfxgen.engine.transform import Affine

ensure_atoms()


def draw_atom(atom, t: float, ss: int = 1, scale: float = 0.5) -> dict[str, np.ndarray]:
    """Draw an atom instance directly (no compositor) centred on a canvas fitting its bounds.
    Returns role -> raw buffer (float32)."""
    x0, y0, x1, y1 = atom.get_bounds()
    w = int(np.ceil((x1 - x0) * scale)) + 8
    h = int(np.ceil((y1 - y0) * scale)) + 8
    A = Affine.translate(4 - x0 * scale, 4 - y0 * scale) @ Affine.scale(scale)
    views, tgs = {}, {}
    for role, rdef in atom.outputs.items():
        tg = Target(rdef.kind, w, h, ss)
        pal = get_palette(rdef.settings.get("palette", "fire"))
        cls = SolidView if rdef.kind == "solid" else View
        views[role] = cls(tg, A, palette=pal, clip=atom.get_bounds())
        tgs[role] = tg
    atom.draw(DrawContext(views, A.linear_scale), t)
    return {r: (tg.rgb if tg.kind == "solid" else tg.data).copy() for r, tg in tgs.items()}


@pytest.fixture
def tmp_gallery(tmp_path):
    return tmp_path / "gallery"


ALL_ATOMS = sorted(ATOMS)
