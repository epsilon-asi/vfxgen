"""Compositor: time, placement, repeat/spawn, channels, determinism, reproducibility."""
from __future__ import annotations

import math

import numpy as np
import pytest

from vfxgen.engine.compositor import TimeMap, build
from vfxgen.errors import LimitError, SpecError
from vfxgen.render import prepare, render

from . import probes  # noqa: F401


def scene(layers, **kw):
    d = {"vfxgen": 1, "size": [200, 200], "fps": 10, "duration": 1.0, "edge_fade": 0, "seed": 1,
         "layers": layers}
    d.update(kw)
    return d


def centroid(frame):
    a = frame[..., 3].astype(np.float64)
    if a.sum() == 0:
        return None
    ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
    return float((xs * a).sum() / a.sum()) + 0.5, float((ys * a).sum() / a.sum()) + 0.5


def frame_at(spec, i=2, **kw):
    return render(spec, workers=1, **kw).frames[i]


# --- time -------------------------------------------------------------------------------
def test_timemap_modes():
    tm = TimeMap(start=0.5, speed=2.0, natural=1.0, mode="once")
    assert tm.local(0.4) is None and tm.local(0.5) == 0 and tm.local(0.75) == pytest.approx(0.5)
    assert tm.local(1.2) is None
    tm.mode = "loop"
    assert tm.local(1.25) == pytest.approx(0.5)
    tm.mode = "hold"
    assert tm.local(5) == 1.0
    tm.mode = "pingpong"
    assert tm.local(1.25) == pytest.approx(0.5) and tm.local(1.0) == pytest.approx(1.0)
    tm2 = TimeMap(natural=1.0, warp=("power", 2.0, 0.0, None))
    assert tm2.local(0.5) == pytest.approx(0.25)
    tm3 = TimeMap(natural=1.0, length=0.3, fade_in=0.1)
    assert tm3.local(0.4) is None and tm3.gain(0.05) == pytest.approx(0.5)


def test_frame_counts_and_times():
    one = build(prepare(scene([{"effect": "_probe_marker"}], fps=10, duration=1.0)))
    assert one.frame_count == 11 and one.frame_time(10) == pytest.approx(1.0)
    lp = build(prepare(scene([{"effect": "_probe_loop"}], fps=10, duration=1.0, loop=True)))
    assert lp.frame_count == 10
    with pytest.raises(SpecError, match="whole number"):
        build(prepare(scene([{"effect": "_probe_loop"}], fps=10, duration=1.05, loop=True)))


def test_fps_changes_sampling_not_motion():
    a = render(scene([{"effect": "_probe_loop"}], fps=10, loop=True), workers=1).frames[5]   # T=0.5
    b = render(scene([{"effect": "_probe_loop"}], fps=20, loop=True), workers=1).frames[10]  # T=0.5
    assert np.array_equal(a, b)


def test_tick_rate_independent_of_fps():
    lay = [{"effect": "_probe_loop", "params": {"flicker_rate": 5}}]
    a = render(scene(lay, fps=10, loop=True), workers=1).frames[3]
    b = render(scene(lay, fps=20, loop=True), workers=1).frames[6]
    assert np.array_equal(a, b)


def test_start_speed_and_loop_wrap():
    f = render(scene([{"effect": "_probe_marker", "start": 0.5}]), workers=1).frames
    assert f[4][..., 3].max() == 0 and f[6][..., 3].max() > 0
    f = render(scene([{"effect": "_probe_marker", "speed": 2}]), workers=1).frames
    assert f[3][..., 3].max() > 0 and f[6][..., 3].max() == 0     # finished at T=0.5


# --- placement ---------------------------------------------------------------------------
def test_at_and_default_anchor():
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100]}])))
    assert c == pytest.approx((120, 100), abs=0.6)                # dot at local (20,0), anchor = centre (0,0)


def test_rotation_is_clockwise():
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "rotate": 90}])))
    assert c == pytest.approx((100, 120), abs=0.6)


def test_scale_and_flip_and_offset():
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "scale": 2}])))
    assert c == pytest.approx((140, 100), abs=0.8)
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "flip": "x"}])))
    assert c == pytest.approx((80, 100), abs=0.6)
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "offset": [5, -7]}])))
    assert c == pytest.approx((125, 93), abs=0.6)


@pytest.mark.parametrize("anchor,expected", [
    ({"px": [20, 0]}, (100, 100)),        # pin the dot itself at `at`
    ("tip", (90, 100)),                   # named anchor (30, 0)
    ([1.0, 0.5], (80, 100)),              # right-centre of bounds = (40, 0)
    ({"uv": [0.0, 0.5]}, (160, 100)),     # left-centre (-40, 0)
])
def test_anchor_forms(anchor, expected):
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "anchor": anchor}])))
    assert c == pytest.approx(expected, abs=0.6)


def test_at_uv_and_keyframes_and_expressions():
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": {"uv": [0.25, 0.5]}}])))
    assert c == pytest.approx((70, 100), abs=0.6)
    kf = {"keys": [[0, [50, 100]], [1, [150, 100]]]}
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": kf}]), i=5))
    assert c == pytest.approx((120, 100), abs=0.6)
    c = centroid(frame_at(scene([{"effect": "_probe_marker", "at": [100, 100], "rotate": "${180 * t}"}]), i=5))
    assert c == pytest.approx((100, 120), abs=0.6)


def test_move_and_face_motion():
    lay = {"effect": "_probe_marker", "params": {"dx": 0},
           "move": {"path": {"line": [[50, 50], [150, 150]]}}}
    c = centroid(frame_at(scene([lay]), i=5))
    assert c == pytest.approx((100, 100), abs=1.0)
    lay2 = {"effect": "_probe_marker", "move": {"path": {"line": [[50, 100], [150, 100]]}}, "face_motion": True,
            "rotate": 90}
    c = centroid(frame_at(scene([lay2]), i=5))
    assert c == pytest.approx((100, 120), abs=1.0)


def test_attach_follows_moving_anchor():
    lays = [{"id": "a", "effect": "_probe_marker", "at": [60, 100], "params": {"a": 0.0001}},
            {"effect": "_probe_marker", "attach": {"layer": "a", "anchor": "moving"}, "params": {"dx": 0}}]
    c = centroid(frame_at(scene(lays), i=5))        # anchor 'moving' = (10t, 0) -> (65, 100) at t=0.5
    assert c == pytest.approx((65, 100), abs=0.8)


def test_mask_radial():
    base = frame_at(scene([{"effect": "_probe_marker", "at": [100, 100]}]))
    masked = frame_at(scene([{"effect": "_probe_marker", "at": [100, 100],
                              "mask": {"type": "radial", "inner": 5, "outer": 15}}]))
    assert masked[..., 3].sum() < 0.2 * base[..., 3].sum()


# --- repeat / spawn ----------------------------------------------------------------------------
def test_repeat_along_path_and_every():
    lay = {"effect": "_probe_marker", "params": {"dx": 0},
           "repeat": {"count": 3, "along": {"line": [[40, 100], [160, 100]]}, "every": 0.2}}
    comp = build(prepare(scene([lay])))
    assert len(comp.instances) == 3
    f = render(scene([lay]), workers=1).frames
    xs = [c[0] for c in [centroid(f[i]) for i in (1,)] if c]
    assert xs and xs[0] == pytest.approx(40, abs=1)            # only the first copy has started at T=0.1
    a = f[5][..., 3]
    cols = np.flatnonzero(a.max(axis=0) > 10)
    assert cols.min() < 50 and cols.max() > 150                 # all three visible later


def test_repeat_ramp_and_seeds_explicit():
    lay = {"id": "r", "effect": "_probe_loop", "repeat": {"count": 4, "ramp": {"opacity": [1, 0.25]}}}
    res = render(scene([lay], loop=True), workers=1)
    rep = res.explicit["layers"][0]["repeat"]
    assert len(rep["seeds"]) == 4


def test_spawn_generations():
    lay = {"effect": "_probe_marker", "at": [100, 100], "params": {"dx": 0},
           "spawn": {"depth": 2, "count": 3, "radius": 30, "when": 0.1, "scale": 0.5}}
    comp = build(prepare(scene([lay])))
    assert len(comp.instances) == 1 + 3 + 9
    with pytest.raises(SpecError):
        prepare(scene([{"effect": "_probe_marker", "spawn": {"depth": 4}}]))


def test_instance_limit():
    lay = {"effect": "_probe_marker", "repeat": {"count": 25000}}
    with pytest.raises(LimitError):
        build(prepare(scene([lay])))


# --- channels ----------------------------------------------------------------------------------
def test_same_palette_shares_channel_and_colours_split():
    comp = build(prepare(scene([{"effect": "glow"}, {"effect": "glow"}])))
    assert [c.name for c in comp.channels] == ["light"]
    comp = build(prepare(scene([{"effect": "glow"}, {"effect": "glow", "colors": "#00ff00"}])))
    assert len(comp.channels) == 2 and comp.channels[1].name.startswith("light@")


def test_overlap_sums_before_tonemap():
    one = frame_at(scene([{"effect": "_probe_marker", "params": {"a": 0.4}}]))
    two = frame_at(scene([{"effect": "_probe_marker", "params": {"a": 0.4}},
                          {"effect": "_probe_marker", "params": {"a": 0.4}}]))
    one_strong = frame_at(scene([{"effect": "_probe_marker", "params": {"a": 0.8}}]))
    assert np.array_equal(two, one_strong) and not np.array_equal(one, two)


def test_nested_scene_channels_namespaced(tmp_path):
    (tmp_path / "child.yaml").write_text(
        "vfxgen: 1\nsize: [100, 100]\nduration: 1\nchannels:\n  light: {kind: light, palette: fire}\n"
        "  smoke: {kind: smoke, lit_by: {channels: [light], amount: 1}}\n"
        "layers:\n  - effect: _probe_smoke\n  - effect: _probe_marker\n")
    p = tmp_path / "parent.yaml"
    p.write_text("vfxgen: 1\nsize: [200, 200]\nduration: 1\nlayers:\n  - {id: kid, effect: child.yaml, at: [100, 100]}\n")
    comp = build(prepare(str(p)))
    names = {c.name for c in comp.channels}
    assert {"kid.light", "kid.smoke"} <= names
    sm = next(c for c in comp.channels if c.name == "kid.smoke")
    assert sm.spec.lit_by.channels == ["kid.light"]
    assert render(str(p), workers=1).frames[3][..., 3].max() > 0


def test_smoke_lit_by_light_tints():
    lays = [{"effect": "_probe_smoke"}]
    plain = frame_at(scene(lays))
    lit = frame_at(scene(lays + [{"effect": "glow", "params": {"radius": 30, "brightness": 4}}]))
    assert not np.array_equal(plain, lit)


# --- determinism / reproducibility ------------------------------------------------------------------
COMPLEX = scene([
    {"id": "g", "effect": "glow", "params": {"flicker": 0.4}},
    {"id": "b", "effect": "burst", "at": [80, 120], "params": {"count": 20, "style": "streak"}},
    {"id": "s", "effect": "shockwave", "params": {"breakup": 0.5},
     "repeat": {"count": 2, "every": 0.2, "jitter": {"position": 10}}},
    {"id": "sp", "effect": "flash", "at": [150, 60], "params": {"radius": 3, "grow": 10},
     "spawn": {"count": 2, "depth": 1, "radius": 20}},
], fps=8)


def test_deterministic_and_parallel_identical():
    a = render(COMPLEX, workers=1)
    b = render(COMPLEX, workers=1)
    c = render(COMPLEX, workers=3)
    for x, y, z in zip(a.frames, b.frames, c.frames):
        assert np.array_equal(x, y) and np.array_equal(x, z)


def test_explicit_yaml_rerender_is_byte_identical(tmp_path):
    r = render(COMPLEX, workers=1)
    gen = r.save(tmp_path / "gallery")
    r2 = render(str(gen.yaml), workers=1)
    assert all(np.array_equal(x, y) for x, y in zip(r.frames, r2.frames))


def test_editing_one_layer_keeps_others_seeds():
    a = render(COMPLEX, workers=1).explicit
    edited = dict(COMPLEX)
    edited["layers"] = [dict(l) for l in COMPLEX["layers"]]
    edited["layers"][1] = {**edited["layers"][1], "params": {"count": 35, "style": "dot"}}
    b = render(edited, workers=1).explicit
    assert a["layers"][0]["seeds"] == b["layers"][0]["seeds"]
    assert a["layers"][2]["repeat"]["seeds"] == b["layers"][2]["repeat"]["seeds"]


def test_stream_seed_override_only_changes_that_part():
    base = render(scene([{"id": "b", "effect": "burst", "params": {"count": 10}}]), workers=1)
    s = base.explicit["layers"][0]["seeds"]["particles"]
    same = render(scene([{"id": "b", "effect": "burst", "params": {"count": 10}, "seeds": {"particles": s}}]),
                  workers=1)
    other = render(scene([{"id": "b", "effect": "burst", "params": {"count": 10},
                           "seeds": {"particles": s + 1}}]), workers=1)
    assert all(np.array_equal(x, y) for x, y in zip(base.frames, same.frames))
    assert any(not np.array_equal(x, y) for x, y in zip(base.frames, other.frames))


def test_scene_seed_changes_output():
    a = render(COMPLEX, workers=1, seed=1).frames
    b = render(COMPLEX, workers=1, seed=2).frames
    assert any(not np.array_equal(x, y) for x, y in zip(a, b))


def test_draft_mode():
    r = render(COMPLEX, workers=1, draft=True)
    assert r.size == (100, 100) and r.fps == 4


def test_size_auto():
    r = render(scene([{"effect": "_probe_marker", "at": [0, 0]}], size="auto"), workers=1)
    assert r.size == (96, 96)


# --- fixes from the porting round ------------------------------------------------------------
def test_time_expression_sees_scene_vars():
    sc = scene([{"effect": "_probe_marker", "at": [100, 100], "rotate": "${spin * t}"}],
               vars={"spin": {"default": 180, "doc": "deg/s"}})
    c = centroid(frame_at(sc, i=5))
    assert c == pytest.approx((100, 120), abs=0.6)


def test_defs_block_for_yaml_anchors():
    import io
    from ruamel.yaml import YAML
    text = """vfxgen: 1
size: [200, 200]
fps: 10
duration: 1
defs:
  common: &common {at: [100, 100], params: {dx: 0}}
layers:
  - {effect: _probe_marker, <<: *common}
"""
    raw = YAML(typ="safe", pure=True).load(io.StringIO(text))
    c = centroid(render(raw, workers=1).frames[2])
    assert c == pytest.approx((100, 100), abs=0.6)


def test_one_shot_last_frame_lands_on_duration():
    comp = build(prepare(scene([{"effect": "_probe_marker"}], fps=10, duration=0.95)))
    assert comp.frame_count == 11 and comp.frame_time(10) == pytest.approx(0.95)


def test_attach_rotate_inherits_rotation():
    lays = [{"id": "a", "effect": "_probe_marker", "at": [100, 100], "rotate": 90, "params": {"a": 0.0001}},
            {"effect": "_probe_marker", "attach": {"layer": "a", "anchor": "center", "rotate": True}}]
    c = centroid(frame_at(scene(lays)))
    assert c == pytest.approx((100, 120), abs=0.6)


def test_time_warp_offset_and_span():
    tm = TimeMap(natural=2.0, warp=("linear", 1.0, 0.5, 1.0))
    assert tm.local(0.0) == pytest.approx(0.5) and tm.local(1.0) == pytest.approx(1.0)
    assert tm.local(1.5) == pytest.approx(1.5)


def test_sweep_lobe_profile_is_symmetric():
    from vfxgen.engine.compositor import _mask_fn
    from vfxgen.spec.models import Mask
    f = _mask_fn(Mask(type="sweep", profile="lobe", width=30, turns=0.0))
    X = np.array([np.cos(np.radians(20)), np.cos(np.radians(-20))])
    Y = np.array([np.sin(np.radians(20)), np.sin(np.radians(-20))])
    v = f(X * 50, Y * 50, 0.0)
    assert v[0] == pytest.approx(v[1]) and 0 < v[0] < 1


def test_animated_nonuniform_scale():
    lay = {"effect": "_probe_marker", "at": [100, 100], "scale": {"keys": [[0, [1, 1]], [1, [3, 1]]]}}
    c = centroid(frame_at(scene([lay]), i=5))
    assert c == pytest.approx((140, 100), abs=1.0)


def test_view_multiply_lowres():
    from vfxgen.engine.canvas import Target, View
    from vfxgen.engine.transform import Affine
    v = View(Target("smoke", 40, 40, 2), Affine.translate(20, 20), clip=(-20, -20, 20, 20))
    v.dot(0, 0, 6, 1.0)
    before = v.target.data.sum()
    v.multiply(lambda X, Y: np.where(X > 0, 0.0, 1.0), lowres=True)
    assert 0.4 * before < v.target.data.sum() < 0.6 * before


def test_scene_named_anchor_list_is_pixels(tmp_path):
    (tmp_path / "child.yaml").write_text(
        "vfxgen: 1\nsize: [100, 100]\nduration: 1\nanchors: {spot: [20, 0]}\n"
        "layers:\n  - {effect: _probe_marker, at: [0, 0], params: {dx: 20}}\n")
    p = tmp_path / "parent.yaml"
    p.write_text("vfxgen: 1\nsize: [200, 200]\nfps: 10\nduration: 1\nedge_fade: 0\n"
                 "layers:\n  - {effect: child.yaml, at: [100, 100], anchor: spot}\n")
    c = centroid(render(str(p), workers=1).frames[2])
    assert c == pytest.approx((100, 100), abs=0.6)
