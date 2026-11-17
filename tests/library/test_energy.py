"""Energy / arcane library effects: render in draft, pass the quality checks, document every
var, recolour, and accept the legacy knobs."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen
from vfxgen.check import check_result

EFFECTS = ["lightning", "orb", "magic_circle", "portal", "beam", "shield", "heal_aura", "holy_pillar",
           "sparkle", "electric_crackle", "pickup_burst", "charge_up"]

# one legacy knob per effect that switches a part off / changes the structure
VARIANTS = {
    "lightning": {"restrike": False, "branches": 0, "impact": 0},
    "orb": {"motes": 0, "shells": 5, "spin": -1},
    "magic_circle": {"hexagram": False, "ticks": False, "rotation_speed": 2},
    "portal": {"eye_darkness": 0, "rotation_speed": 1, "eye_palette": "#102030"},
    "beam": {"emitter": False, "impact": False, "filaments": False},
    "shield": {"waves": 0, "highlight": False},
    "heal_aura": {"inner_rings": False, "column": 0},
    "holy_pillar": {"ground_ring": False, "motes": 0},
    "sparkle": {"diagonals": False, "dust": 0},
    "electric_crackle": {"micro_arcs": 0, "hop_chance": 1.0},
    "pickup_burst": {"rings": 3, "center_glint": False},
    "charge_up": {"pop": False},
}


def _render(name, vars=None, colors=None):
    return vfxgen.render(name, vars or {}, colors=colors, draft=True, workers=1, cache=False)


@pytest.fixture(scope="module")
def drafts():
    return {}


def _draft(drafts, name):
    if name not in drafts:
        drafts[name] = _render(name)
    return drafts[name]


@pytest.mark.parametrize("name", EFFECTS)
def test_renders_and_checks(name, drafts):
    r = _draft(drafts, name)
    assert r.frames
    for f in r.frames:
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 4
    rep = check_result(r)
    assert rep.ok, rep.summary()
    assert any(f[..., 3].any() for f in r.frames)


@pytest.mark.parametrize("name", EFFECTS)
def test_vars_documented(name):
    info = vfxgen.describe(name)
    assert info["type"] == "effect"
    assert "colors" in info["vars"]
    for k, v in info["vars"].items():
        assert v.get("doc"), f"{name}: var {k} has no doc"


@pytest.mark.parametrize("name", EFFECTS)
def test_recolour(name, drafts):
    base = _draft(drafts, name)
    target = np.array([255.0, 48.0, 80.0])
    red = _render(name, colors="#ff3050")
    i = int(np.argmax([f[..., 3].sum() for f in base.frames]))
    a, b = base.frames[i].astype(np.float64), red.frames[i].astype(np.float64)
    m = (a[..., 3] > 64) & (b[..., 3] > 64)
    assert m.any()

    def sim(f):   # colour direction of the lit pixels vs the requested colour (minus the white core)
        c = f[..., :3][m] - f[..., :3][m].min(axis=1, keepdims=True)
        v = c.mean(axis=0)
        t = target - target.min()
        return float(v @ t / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-9))
    assert sim(b) > sim(a) and sim(b) > 0.9


@pytest.mark.parametrize("name", EFFECTS)
def test_legacy_knobs(name, drafts):
    r = _render(name, VARIANTS[name])
    assert check_result(r).ok
    base = _draft(drafts, name)
    assert any(not np.array_equal(x, y) for x, y in zip(r.frames, base.frames)), "knobs had no effect"
