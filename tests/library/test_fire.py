"""Fire / smoke library effects: render (draft), quality checks, documented vars, recolour."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen
from vfxgen.check import check_result

FIRE_EFFECTS = ["fire", "explosion", "fireball", "smoke", "poison_cloud", "torch_flame", "dust_puff",
                "embers", "muzzle_flash"]

# extra channel palettes each effect must expose as `<role>_palette` vars
EXTRA_PALETTES = {"explosion": ["smoke_palette"], "poison_cloud": ["gas_palette"],
                  "muzzle_flash": ["smoke_palette"]}


def _render(name, **kw):
    return vfxgen.render(name, draft=True, workers=1, cache=False, **kw)


@pytest.fixture(scope="module")
def renders():
    return {name: _render(name) for name in FIRE_EFFECTS}


@pytest.mark.parametrize("name", FIRE_EFFECTS)
def test_renders_rgba_and_checks_clean(renders, name):
    r = renders[name]
    assert r.frames, "no frames"
    for f in r.frames:
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 4
    assert any(f[..., 3].any() for f in r.frames), "everything transparent"
    rep = check_result(r)
    assert rep.ok, rep.summary()
    assert not rep.issues, rep.summary()          # no seams, clipping or leftovers either


@pytest.mark.parametrize("name", FIRE_EFFECTS)
def test_vars_documented(name):
    info = vfxgen.describe(name)
    assert info["type"] == "effect"
    vars_ = info["vars"]
    assert "colors" in vars_
    for extra in EXTRA_PALETTES.get(name, []):
        assert extra in vars_
    for k, v in vars_.items():
        assert v.get("doc"), f"var {k!r} of {name} has no doc"


def _mean_rgb(frames):
    acc, w = np.zeros(3), 0.0
    for f in frames:
        a = f[..., 3:4].astype(np.float64) / 255
        acc += (f[..., :3] * a).reshape(-1, 3).sum(0)
        w += float(a.sum())
    return acc / max(w, 1e-9)


@pytest.mark.parametrize("name,colors", [("fire", "#3080ff"), ("smoke", "#40c060")])
def test_recolour(renders, name, colors):
    base = _mean_rgb(renders[name].frames)
    r = _render(name, colors=colors)
    new = _mean_rgb(r.frames)
    assert check_result(r).ok
    if name == "fire":
        assert new[2] > new[0] and base[0] > base[2]       # orange -> blue
    else:
        assert new[1] > new[0] + 10 and new[1] > new[2] + 10   # grey -> green


def test_extra_palette_var_recolours_smoke():
    r = _render("explosion", vars={"smoke_palette": "#103080,#2050c0,#80a0ff", "sparks": 0, "debris": 0})
    late = r.frames[int(len(r.frames) * 0.8)]
    rgb = _mean_rgb([late])
    assert rgb[2] > rgb[0] + 20, rgb
