"""UI feedback effects (text, coins, shines, bursts) and per-material impact effects."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen as vfx
from vfxgen.check import check_result
from vfxgen.errors import SpecError

UI = ["coin_pickup", "button_shine", "level_up", "unlock_burst", "damage_pop"]
IMPACTS = ["impact_metal", "impact_wood", "impact_stone", "impact_blood", "impact_energy"]
EFFECTS = UI + IMPACTS
LOOPS = {"button_shine"}
EVENTS = {"coin_pickup": "collect", "button_shine": "shine", "level_up": "level_up", "unlock_burst": "unlock",
          "damage_pop": "show", **{n: "impact" for n in IMPACTS}}
ADDITIVE = {"button_shine", "unlock_burst", "impact_metal", "impact_energy"}


def _info(name):
    return next(e for e in vfx.list_effects() if e["name"] == name)


def _scene(name):
    from pathlib import Path

    from vfxgen.spec.loader import load
    return load(Path(vfx.__file__).parent / "library" / f"{name}.yaml").scene


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
def test_vars_documented_and_bounded(name):
    vars_ = _info(name)["vars"]
    assert "colors" in vars_
    for k, v in vars_.items():
        assert v.get("doc"), f"var {k} has no doc"
        if isinstance(v.get("default"), (int, float)) and not isinstance(v.get("default"), bool):
            assert v.get("min") is not None and v.get("max") is not None, f"numeric var {k} has no min/max"


@pytest.mark.parametrize("name", EFFECTS)
def test_events_pivot_blend(name):
    sc = _scene(name)
    assert sc.description
    assert EVENTS[name] in {e.name for e in sc.events}
    assert sc.blend == ("additive" if name in ADDITIVE else "alpha")


@pytest.mark.parametrize("name", ["damage_pop", "impact_metal", "impact_blood"])
def test_recolour(renders, name):
    base = renders[name]
    r = vfx.render(name, draft=True, workers=1, colors="#3080ff", cache=False)
    i = int(np.argmax([f[..., 3].sum() for f in base.frames]))
    assert not np.array_equal(base.frames[i], r.frames[i]), "colors had no effect"
    assert check_result(r).ok


def test_damage_pop_crit_is_bigger(renders):
    crit = vfx.render("damage_pop", {"crit": True, "text": "77"}, draft=True, workers=1, cache=False)
    assert check_result(crit).ok
    i = len(crit.frames) // 3
    assert int(crit.frames[i][..., 3].astype(bool).sum()) > int(renders["damage_pop"].frames[i][..., 3].astype(bool).sum())


def test_text_atom_rejects_unknown_characters():
    with pytest.raises(SpecError):
        vfx.render("damage_pop", {"text": "12#"}, draft=True, workers=1, cache=False)
