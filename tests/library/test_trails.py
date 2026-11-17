"""Library effects built on the trail / link / projectile atoms (ribbon, chain, projectile_core,
muzzle_ring): projectile families, movement effects and showcases."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vfxgen as vfx
from vfxgen.check import check_result
from vfxgen.spec.loader import read_yaml

FAMILIES = ["fire_bolt", "frost_shard", "arcane_missile", "lightning_orb"]
PROJECTILES = [f"{f}_{part}" for f in FAMILIES for part in ("launch", "travel", "impact")]
MOVEMENT = ["dash", "double_jump", "landing_dust", "footstep_dust", "blink"]
SHOWCASES = ["sword_trail", "chain_lightning", "tether", "life_drain"]
EFFECTS = PROJECTILES + MOVEMENT + SHOWCASES
LIB = Path(vfx.__file__).parent / "library"
LOOPS = {f"{f}_travel" for f in FAMILIES} | {"chain_lightning", "tether", "life_drain"}

EVENTS = {**{f"{f}_launch": "spawn_projectile" for f in FAMILIES},
          **{f"{f}_impact": "impact" for f in FAMILIES},
          "dash": "dash_start", "double_jump": "jump", "landing_dust": "land", "footstep_dust": "footstep",
          "blink": "appear", "sword_trail": "hit"}
TRACKS = {f"{f}_travel": "head" for f in FAMILIES}
ADDITIVE = {"fire_bolt_travel", "arcane_missile_travel", "arcane_missile_launch", "arcane_missile_impact",
            "lightning_orb_travel", "lightning_orb_launch", "lightning_orb_impact", "blink", "sword_trail",
            "chain_lightning", "tether", "life_drain"}


def _info(name):
    return next(e for e in vfx.list_effects() if e["name"] == name)


def _yaml(name):
    return read_yaml((LIB / f"{name}.yaml").read_text())


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
def test_vars_documented_with_ranges(name):
    vars_ = _info(name)["vars"]
    assert "colors" in vars_
    for k, v in vars_.items():
        assert v.get("doc"), f"var {k} has no doc"
        if isinstance(v.get("default"), (int, float)) and not isinstance(v.get("default"), bool):
            assert v.get("min") is not None and v.get("max") is not None, f"var {k} has no min/max"


@pytest.mark.parametrize("name", sorted(EVENTS))
def test_events(renders, name):
    names = {e["name"] for e in renders[name].events}
    assert EVENTS[name] in names, names


@pytest.mark.parametrize("name", sorted(TRACKS))
def test_travel_tracks_head_at_pivot(renders, name):
    r = renders[name]
    pts = r.tracks[TRACKS[name]]
    assert len(pts) == len(r.frames)
    # the head stays put (the engine moves the sprite) and sits on the pivot
    sc = _yaml(name)
    k = r.width / sc["size"][0]
    px, py = sc["pivot"]["px"]
    for p in pts:
        assert abs(p[0] - px * k) < 0.5 and abs(p[1] - py * k) < 0.5, (p, px * k, py * k)


@pytest.mark.parametrize("name", EFFECTS)
def test_blend_and_pivot_declared(name):
    sc = _yaml(name)
    assert sc["blend"] == ("additive" if name in ADDITIVE else "alpha")
    assert "pivot" in sc and sc.get("description")


@pytest.mark.parametrize("name", ["fire_bolt_travel", "frost_shard_impact", "chain_lightning"])
def test_recolour(renders, name):
    base = renders[name]
    r = vfx.render(name, draft=True, workers=1, colors="#ff3050", cache=False)
    i = int(np.argmax([f[..., 3].sum() for f in base.frames]))
    assert not np.array_equal(base.frames[i], r.frames[i]), "colors had no effect"
    assert check_result(r).ok
