"""The contract every atom must satisfy (runs over the whole registry).

Run for one atom while developing:  pytest tests/atoms -k <name>
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import ALL_ATOMS, draw_atom
from vfxgen.atom import ATOMS
from vfxgen.render import render


def _total(bufs):
    return sum(float(np.abs(b).sum()) for b in bufs.values())


@pytest.fixture(params=ALL_ATOMS)
def cls(request):
    return ATOMS[request.param]


def test_params_documented_and_bounded(cls):
    info = cls.info()
    assert info["params"], "atom has no params"
    for k, f in info["params"].items():
        assert f["doc"], f"{k} has no doc"
    assert cls.kind in ("oneshot", "loop")
    assert len(cls.outputs) >= 1


def test_deterministic(cls):
    a, b = cls({}, seed=11), cls({}, seed=11)
    L = a.length()
    for t in (0.0, 0.37 * L, 0.71 * L):
        ba, bb = draw_atom(a, t), draw_atom(b, t)
        for r in ba:
            assert np.array_equal(ba[r], bb[r]), f"role {r} differs at t={t}"


def test_draws_something(cls):
    a = cls({}, seed=3)
    L = a.length()
    assert any(_total(draw_atom(a, L * k / 8)) > 0 for k in range(8)), "atom never draws anything"


def test_time_contract(cls):
    a = cls({}, seed=5)
    L = a.length()
    if cls.kind == "loop":
        b0, b1 = draw_atom(a, 0.0), draw_atom(a, L)
        for r in b0:
            assert np.allclose(b0[r], b1[r], atol=1e-3 * max(1.0, float(np.abs(b0[r]).max()))), \
                f"loop not seamless in role {r}: draw(0) != draw(period)"
    else:
        peak = max(_total(draw_atom(a, L * k / 10)) for k in range(10))
        end = _total(draw_atom(a, L))
        assert end <= 0.02 * peak + 1e-6, f"one-shot not (nearly) empty at t=duration ({end:.3g} vs peak {peak:.3g})"


def test_stream_seeds(cls):
    a = cls({}, seed=1)
    seeds = a.stream_seeds()
    assert set(seeds) == set(cls.streams)
    if cls.streams:
        s = cls.streams[0]
        b = cls({}, seed=1, seeds={s: seeds[s] + 1})
        assert b.stream_seed(s) == seeds[s] + 1
        for other in cls.streams[1:]:
            assert b.stream_seed(other) == seeds[other]


def test_palette_params_accept_hex(cls):
    pal = [k for k in cls.Params.model_fields if k.endswith("palette")]
    if not pal:
        pytest.skip("no palette params")
    a = cls({k: "#ff8800" for k in pal}, seed=2)
    draw_atom(a, a.length() * 0.3)


def test_renders_via_cli_path(cls):
    """Rendered alone (draft): transparent RGBA, nothing hard-clipped at the border."""
    r = render(cls.name, draft=True, workers=1)
    assert r.frames and r.frames[0].dtype == np.uint8 and r.frames[0].shape[2] == 4
    border = max(int(max(f[0].max(), f[-1].max(), f[:, 0].max(), f[:, -1].max())) for f in r.frames)
    assert border <= 40, f"content touches the frame border (alpha {border})"


def test_all_atom_modules_import():
    from vfxgen.atoms import IMPORT_ERRORS
    assert not IMPORT_ERRORS, "\n".join(f"{k}:\n{v}" for k, v in IMPORT_ERRORS.items())
