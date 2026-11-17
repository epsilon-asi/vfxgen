"""Sprite, texture-particle and status library effects: render (draft), quality checks,
documented vars, recolour."""
from __future__ import annotations

import numpy as np
import pytest

import vfxgen
from vfxgen.check import check_result

SPRITE_EFFECTS = ["hit_flash", "dissolve_out", "teleport_in", "teleport_out", "outline_glow", "burning_sprite",
                  "electrified_sprite", "frozen_sprite"]
TEXTURE_EFFECTS = ["coin_fountain", "shard_burst"]
STATUS_EFFECTS = ["poison_status", "burn_status", "freeze_status", "stun_stars", "sleep_zs", "bleed_status",
                  "buff_aura", "debuff_aura"]
ALL = SPRITE_EFFECTS + TEXTURE_EFFECTS + STATUS_EFFECTS


def _render(name, **kw):
    return vfxgen.render(name, draft=True, workers=1, cache=False, **kw)


@pytest.fixture(scope="module")
def renders():
    return {}


def _get(renders, name):
    if name not in renders:
        renders[name] = _render(name)
    return renders[name]


@pytest.mark.parametrize("name", ALL)
def test_renders_rgba_and_checks_clean(renders, name):
    r = _get(renders, name)
    assert r.frames, "no frames"
    for f in r.frames:
        assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 4
    assert any(f[..., 3].any() for f in r.frames), "everything transparent"
    rep = check_result(r)
    assert rep.ok, rep.summary()
    assert not rep.issues, rep.summary()


@pytest.mark.parametrize("name", ALL)
def test_vars_documented(name):
    info = vfxgen.describe(name)
    assert info["type"] == "effect"
    vars_ = info["vars"]
    assert "colors" in vars_
    if name in SPRITE_EFFECTS:
        assert "image" in vars_, "sprite effects expose their image"
    for k, v in vars_.items():
        assert v.get("doc"), f"var {k!r} of {name} has no doc"


def _mean_rgb(frames):
    acc, w = np.zeros(3), 0.0
    for f in frames:
        a = f[..., 3:4].astype(np.float64) / 255
        acc += (f[..., :3] * a).reshape(-1, 3).sum(0)
        w += float(a.sum())
    return acc / max(w, 1e-9)


def test_recolour_outline_glow(renders):
    base = _mean_rgb(_get(renders, "outline_glow").frames)
    r = _render("outline_glow", colors="#30ff60")
    new = _mean_rgb(r.frames)
    assert check_result(r).ok
    assert new[1] - new[0] > base[1] - base[0] + 10, (base, new)      # gold glow -> green glow


def test_hit_flash_peaks_to_flash_colour():
    r = _render("hit_flash", colors="#ff2040")
    first, peak = r.frames[0], r.frames[1]
    sprite = first[..., 3] > 200
    c = peak[sprite][:, :3].astype(float).mean(0)
    assert c[0] > c[2] + 60 and c[0] > c[1] + 60, c                 # sprite is flashed red
    assert {e["name"] for e in r.events} >= {"hit", "peak", "recover"}
