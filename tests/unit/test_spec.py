"""Expressions, colours, models, loading and reference resolution."""
from __future__ import annotations

import numpy as np
import pytest

from vfxgen.engine.color import adjust_stops, get_palette, ramp
from vfxgen.errors import ResolveError, SpecError
from vfxgen.spec.expr import evaluate, substitute, uses_time
from vfxgen.spec.loader import load, resolve_effect

from . import probes  # noqa: F401


# --- expressions ---------------------------------------------------------------------
@pytest.mark.parametrize("src,expected", [
    ("1 + 2 * 3", 7), ("radius * 2", 2.4), ("max(1, radius)", 1.2), ("clamp(5, 0, 2)", 2),
    ("lerp(0, 10, 0.25)", 2.5), ("2 ** 3", 8), ("1 if radius > 1 else 0", 1), ("[1, radius]", [1, 1.2]),
    ("round(pi, 2)", 3.14),
])
def test_expr_eval(src, expected):
    assert evaluate(src, {"radius": 1.2}) == pytest.approx(expected) if not isinstance(expected, list) \
        else evaluate(src, {"radius": 1.2}) == expected


@pytest.mark.parametrize("src", [
    "radius.__class__", "__import__('os')", "(lambda: 1)()", "[x for x in [1]]", "open('x')",
    "radius[0]", "getattr(radius, 'x')",
])
def test_expr_rejects_unsafe(src):
    with pytest.raises(SpecError):
        evaluate(src, {"radius": 1.2})


def test_expr_unknown_var():
    with pytest.raises(SpecError, match="unknown variable"):
        evaluate("nope + 1", {})


def test_substitute_types_and_deferral():
    out = substitute({"a": "${x * 2}", "b": "size ${x}px", "c": "${360 * t}", "d": ["${x}", 3]}, {"x": 4})
    assert out["a"] == 8 and out["b"] == "size 4px" and out["c"] == "${360 * t}" and out["d"] == [4, 3]
    assert uses_time("360*t") and not uses_time("x+1")
    with pytest.raises(SpecError, match="whole value"):
        substitute({"a": "rot ${t}"}, {})


# --- colours -----------------------------------------------------------------------------
def test_palettes():
    assert get_palette("fire")[0][0] == 0
    one = get_palette("#2a7fff")
    assert len(one) == 4 and one[-1][1] == (255.0, 255.0, 255.0)
    lst = get_palette("#300,#f30,#fff")
    assert [p for p, _ in lst] == [0.0, 0.5, 1.0] and lst[1][1] == (255.0, 51.0, 0.0)
    pos = get_palette("0:#000,0.2:#f00,1:#fff")
    assert pos[1][0] == 0.2
    assert get_palette([[0, [0, 0, 0]], [1, [255, 0, 0]]])[1][1] == (255.0, 0.0, 0.0)
    assert np.allclose(ramp("#000,#fff", np.array([0.5])), [[127.5, 127.5, 127.5]])
    with pytest.raises(ValueError):
        get_palette("not_a_palette")
    with pytest.raises(ValueError):
        get_palette("#12")
    shifted = adjust_stops("#ff0000,#ff0000", hue=120)
    assert shifted[0][1][1] > 200 and shifted[0][1][0] < 10   # red -> green


# --- models & loading ----------------------------------------------------------------------
def test_validation_error_has_path():
    with pytest.raises(SpecError, match=r"layers\[0\]\.repeat\.count"):
        load({"vfxgen": 1, "layers": [{"effect": "glow", "repeat": {"count": 0}}]})
    with pytest.raises(SpecError, match="unknown field"):
        load({"vfxgen": 1, "layrs": []})


def test_vars_override_and_bounds():
    raw = {"vfxgen": 1, "vars": {"r": {"default": 2, "min": 0, "max": 5}, "d": "${r * 3}"},
           "layers": [{"effect": "glow", "params": {"radius": "${d}"}}]}
    ls = load(raw)
    assert ls.scene.layers[0].params["radius"] == 6
    assert load(raw, {"r": 1}).scene.layers[0].params["radius"] == 3
    with pytest.raises(SpecError, match="outside"):
        load(raw, {"r": 9})
    with pytest.raises(SpecError, match="unknown variable"):
        load(raw, {"zzz": 1})


def test_atom_param_validation():
    from vfxgen.render import render
    with pytest.raises(SpecError, match="unknown parameter `radiuz`"):
        render({"vfxgen": 1, "layers": [{"effect": "glow", "params": {"radiuz": 3}}]}, workers=1)
    with pytest.raises(SpecError, match="not one of"):
        render({"vfxgen": 1, "layers": [{"effect": "burst", "params": {"style": "blob"}}]}, workers=1)


def _write(p, text):
    p.write_text(text)
    return p


def test_resolution_and_relative_paths(tmp_path):
    (tmp_path / "sub").mkdir()
    _write(tmp_path / "sub" / "child.yaml", "vfxgen: 1\nsize: [64, 64]\nlayers:\n  - effect: glow\n")
    parent = load(_write(tmp_path / "p.yaml", "vfxgen: 1\nlayers:\n  - effect: sub/child.yaml\n"))
    ref = resolve_effect("sub/child.yaml", parent, {})
    assert ref.kind == "scene" and ref.scene.scene.size == [64, 64]
    assert resolve_effect("glow", parent, {}).kind == "atom"
    with pytest.raises(ResolveError, match="unknown effect"):
        resolve_effect("nonexistent_thing", parent, {})
    with pytest.raises(ResolveError, match="not found"):
        resolve_effect("missing/file.yaml", parent, {})


def test_cycle_detection(tmp_path):
    from vfxgen.render import render
    _write(tmp_path / "a.yaml", "vfxgen: 1\nlayers:\n  - effect: b.yaml\n")
    _write(tmp_path / "b.yaml", "vfxgen: 1\nlayers:\n  - effect: a.yaml\n")
    with pytest.raises(ResolveError, match="cycle"):
        render(str(tmp_path / "a.yaml"), workers=1)


def test_depth_limit(tmp_path):
    from vfxgen.render import render
    _write(tmp_path / "d3.yaml", "vfxgen: 1\nsize: [64, 64]\nduration: 0.2\nlayers:\n  - effect: _probe_marker\n")
    for i in (2, 1, 0):
        _write(tmp_path / f"d{i}.yaml", f"vfxgen: 1\nsize: [64, 64]\nlayers:\n  - effect: d{i + 1}.yaml\n")
    r = render(str(tmp_path / "d0.yaml"), workers=1, draft=True)       # 3 levels of nesting: allowed
    assert r.frames
    _write(tmp_path / "d4.yaml", "vfxgen: 1\nsize: [64, 64]\nlayers:\n  - effect: _probe_marker\n")
    _write(tmp_path / "d3.yaml", "vfxgen: 1\nsize: [64, 64]\nlayers:\n  - effect: d4.yaml\n")
    with pytest.raises(ResolveError, match="deeper than 3"):
        render(str(tmp_path / "d0.yaml"), workers=1)


def test_schema_exports():
    import vfxgen
    s = vfxgen.json_schema()
    assert "layers" in s["properties"] and "Layer" in s["$defs"]
