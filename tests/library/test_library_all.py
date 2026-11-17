"""Cross-cutting checks over every library scene (the per-group files test rendering in depth)."""
from __future__ import annotations

import pytest

import vfxgen as vfx
from vfxgen.atom import ATOMS
from vfxgen.spec.loader import LIBRARY_DIR, load

LIB = sorted(p.stem for p in LIBRARY_DIR.glob("*.yaml"))


def test_no_name_collisions_with_atoms():
    assert not set(LIB) & set(ATOMS), "library effect names must not equal atom names"


@pytest.mark.parametrize("name", LIB)
def test_library_scene_valid_and_documented(name):
    ls = load(LIBRARY_DIR / f"{name}.yaml")
    sc = ls.scene
    assert sc.description, "library scenes need a description"
    assert "colors" in sc.vars, "library scenes expose a `colors` var"
    for k, v in sc.vars.items():
        assert v.doc, f"var {k} has no doc"
    if sc.loop:
        n = float(sc.duration) * sc.fps
        assert abs(n - round(n)) < 1e-6


@pytest.mark.parametrize("name", LIB)
def test_library_scene_describes(name):
    info = vfx.describe(name)
    assert info["type"] == "effect" and info["vars"]
