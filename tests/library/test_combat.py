"""Library effects ported from legacy fx_physical: combat, wind, ice, earth, water."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen as vfx
from vfxgen.check import check_result

EFFECTS = ["whirlwind", "sparks", "slash", "slash_combo", "hit_impact", "frost_nova", "ice_shatter",
           "ground_slam", "water_splash"]
LOOPS = {"whirlwind"}


def _info(name):
    return next(e for e in vfx.list_effects() if e["name"] == name)


@pytest.fixture(scope="module")
def renders():
    return {n: vfx.render(n, draft=True, workers=1, cache=False) for n in EFFECTS}


@pytest.mark.parametrize("name", EFFECTS)
def test_renders_rgba_and_checks_clean(renders, name):
    r = renders[name]
    assert r.frames, "no frames"
    for f in r.frames:
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 4
    assert r.loop == (name in LOOPS)
    assert any(f[..., 3].any() for f in r.frames), "every frame is empty"
    rep = check_result(r)
    assert rep.ok, rep.summary()
    if name not in LOOPS:
        assert int(r.frames[-1][..., 3].max()) <= 16, "one-shot does not end empty"


@pytest.mark.parametrize("name", EFFECTS)
def test_vars_documented_and_colors_exposed(name):
    vars_ = _info(name)["vars"]
    assert "colors" in vars_
    for k, v in vars_.items():
        assert v.get("doc"), f"var {k} has no doc"


@pytest.mark.parametrize("name", ["sparks", "frost_nova", "water_splash"])
def test_recolour(renders, name):
    base = renders[name]
    r = vfx.render(name, draft=True, workers=1, colors="#ff3050", cache=False)
    i = int(np.argmax([f[..., 3].sum() for f in base.frames]))
    assert not np.array_equal(base.frames[i], r.frames[i]), "colors had no effect"
    assert check_result(r).ok


def test_extra_palette_vars_recolour_matter():
    a = vfx.render("ground_slam", draft=True, workers=1, cache=False)
    b = vfx.render("ground_slam", {"rock_palette": "#3050ff"}, draft=True, workers=1, cache=False)
    i = len(a.frames) // 3
    assert not np.array_equal(a.frames[i], b.frames[i])


def test_toggles_and_knobs():
    full = vfx.render("hit_impact", draft=True, workers=1, cache=False)
    bare = vfx.render("hit_impact", {"ring": False, "sparks": 0, "speed_lines": 0}, draft=True, workers=1,
                      cache=False)
    assert sum(int(f[..., 3].sum()) for f in bare.frames) < sum(int(f[..., 3].sum()) for f in full.frames)
    one = vfx.render("slash_combo", {"slash_count": 1}, draft=True, workers=1, cache=False)
    assert check_result(one).ok
