"""Environment / weather library effects and seamless tiles: render in draft, pass the quality
checks, document every var (with min / max for numbers), recolour, and tile seamlessly."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen
from vfxgen.check import check_result

TILES = ["fog_tile", "water_tile", "energy_tile", "rain_tile"]
EFFECTS = ["rain", "snow", "fireflies", "falling_leaves", "light_shafts", "dust_motes", "waterfall",
           "swirling_smoke"] + TILES


def _render(name, vars=None, colors=None, **kw):
    return vfxgen.render(name, vars or {}, colors=colors, draft=True, workers=1, cache=False, **kw)


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
    assert r.frames and r.loop
    for f in r.frames:
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 4
    rep = check_result(r)
    assert rep.ok, rep.summary()
    # no warnings either (a tile fills the frame by design: reaching the border is expected)
    allowed = {"edge_clip"} if name in TILES else set()
    assert not [i for i in rep.issues if i.code not in allowed], rep.summary()
    assert all(f[..., 3].any() for f in r.frames)


@pytest.mark.parametrize("name", EFFECTS)
def test_vars_documented(name):
    info = vfxgen.describe(name)
    assert info["type"] == "effect"
    assert info.get("description")
    assert "colors" in info["vars"]
    for k, v in info["vars"].items():
        assert v.get("doc"), f"{name}: var {k} has no doc"
        d = v.get("default")
        if isinstance(d, (int, float)) and not isinstance(d, bool):
            assert v.get("min") is not None and v.get("max") is not None, f"{name}: var {k} needs min / max"


def _sim(f, m, target):
    c = f[..., :3][m] - f[..., :3][m].min(axis=1, keepdims=True)
    v = c.mean(axis=0)
    t = target - target.min()
    return float(v @ t / (np.linalg.norm(v) * np.linalg.norm(t) + 1e-9))


@pytest.mark.parametrize("name", EFFECTS)
def test_recolour(name, drafts):
    base = _draft(drafts, name)
    target = np.array([255.0, 48.0, 80.0])
    i = int(np.argmax([f[..., 3].sum() for f in base.frames]))
    red = _render(name, colors="#ff3050", frames=[i]).frames[0]
    a, b = base.frames[i].astype(np.float64), red.astype(np.float64)
    m = (a[..., 3] > 64) & (b[..., 3] > 64)
    assert m.any()
    assert _sim(b, m, target) > _sim(a, m, target) and _sim(b, m, target) > 0.85


def _seam_ok(f):
    """Edge columns / rows continue across the wrap: the jump between the last and first
    column (row) is no bigger than typical neighbour steps inside the tile."""
    x = f.astype(np.float64)
    x = np.concatenate([x[..., :3] * x[..., 3:] / 255.0, x[..., 3:]], axis=-1)
    for axis in (0, 1):
        first = np.take(x, 0, axis=axis)
        last = np.take(x, -1, axis=axis)
        seam = np.abs(last - first).mean()
        steps = np.abs(np.diff(x, axis=axis)).mean(axis=tuple(k for k in range(3) if k != axis))
        inner = float(steps.mean())
        assert seam <= 2.0 * inner + 1.0, f"seam {seam:.2f} vs inner step {inner:.2f} (axis {axis})"
        assert seam <= np.percentile(steps, 99) + 1.0


@pytest.mark.parametrize("name", TILES)
def test_tiles_seamless(name, drafts):
    r = _draft(drafts, name)
    for f in r.frames[:: max(1, len(r.frames) // 3)]:
        _seam_ok(f)
    # full quality (supersampled) frame too: the wrap must not double-deposit the edge pixels
    full = vfxgen.render(name, frames=[3], workers=1, cache=False).frames[0]
    _seam_ok(full)
    big = np.concatenate([np.concatenate([full, full], axis=1)] * 2, axis=0)
    _seam_ok(big)                                              # 2x2 tiled still wraps cleanly


@pytest.mark.parametrize("name,vars", [("rain", {"splashes": 0, "wind": -0.4}), ("snow", {"swirl": 0, "wind": -30}),
                                       ("falling_leaves", {"shape": "maple", "flip": 0}),
                                       ("swirling_smoke", {"swirl": 0, "wisps": 0}),
                                       ("fog_tile", {"drift": -2, "banks": 4}), ("water_tile", {"caustics": 0})])
def test_knobs(name, vars, drafts):
    r = _render(name, vars, frames=[0, 2, 4])
    base = _draft(drafts, name)
    assert any(not np.array_equal(x, base.frames[k]) for x, k in zip(r.frames, (0, 2, 4))), "knobs had no effect"
