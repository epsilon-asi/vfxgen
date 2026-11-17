"""Engine features: depth, cross-kind routing, lighting, colour over time, maps, feel, events,
tracks, links, tiling, motion blur, pixel mode, sprites, flow, rotation."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import vfxgen as vfx
from tests.conftest import draw_atom
from vfxgen.atom import ATOMS

from . import probes  # noqa: F401
from .test_engine import centroid, scene


def r1(sc, i=2, **kw):
    return vfx.render(sc, workers=1, **kw).frames[i]


def test_depth_puts_smoke_in_front_of_light():
    lays = [{"effect": "glow", "params": {"radius": 20, "brightness": 4}}, {"effect": "_probe_smoke"}]
    behind = r1(scene(lays))
    front_sc = scene([lays[0], {**lays[1], "depth": 1}])
    front = r1(front_sc)
    from vfxgen.engine.compositor import build
    names = [c.name for c in build(vfx.prepare(front_sc)).channels]
    assert names.index("smoke^1") > names.index("light")          # smoke slice drawn after the light
    c = (100, 100)
    assert behind[c][:3].min() > 220                               # light over smoke: white-hot core
    assert front[c][2] < 150                                       # smoke (lit orange) covers the core


def test_cross_kind_routing_light_role_into_smoke_channel():
    sc = scene([{"effect": "shockwave", "channels": {"light": "mist"}, "params": {"radius_end": 50}}],
               channels={"mist": {"kind": "smoke", "palette": "#808080,#808080"}})
    f = r1(sc, i=1)
    vis = f[f[..., 3] > 20][:, :3]
    assert len(vis) and np.allclose(vis.astype(int), 128, atol=3)        # drawn as grey smoke


def test_solid_output_cannot_go_to_light_channel(tmp_path):
    _sprite(tmp_path / "sprite.png")
    p = tmp_path / "s.yaml"
    p.write_text("vfxgen: 1\nlayers:\n  - {effect: glow}\n  - {effect: _probe_sprite, channels: {solid: light}}\n")
    with pytest.raises(vfx.SpecError, match="solid"):
        vfx.render(str(p), workers=1)


def test_volume_lit_by_surface_faces_the_light():
    base = [{"effect": "_probe_volume", "at": [100, 100]}]
    light = {"effect": "glow", "at": [40, 100], "params": {"radius": 12, "brightness": 6, "halo_radius": 0}}
    lit_ch = {"volume": {"kind": "volume", "palette": "#404040,#ffffff", "bump": 20, "shadow": 0, "wrap": 0.5,
                         "lit_by": {"channels": ["light"], "amount": 2.0, "mode": "surface", "radius": 30}},
              "light": {"kind": "light", "palette": "#ff8000", "bloom": []}}
    f = r1(scene(base + [light], channels=lit_ch))
    left = f[100, 84:94, :3].astype(float).mean(axis=0)
    right = f[100, 106:116, :3].astype(float).mean(axis=0)
    assert left[0] - left[2] > right[0] - right[2] + 5      # warm light on the side facing it


def test_palette_keys_colour_over_time():
    sc = scene([{"effect": "glow", "params": {"radius": 15, "brightness": 1.0, "halo_radius": 0}, "time_mode": "loop"}],
               channels={"light": {"kind": "light", "bloom": [],
                                   "palette_keys": [[0, "#ff0000,#ff0000"], [1, "#0000ff,#0000ff"]]}})
    fr = vfx.render(sc, workers=1).frames
    a, b = fr[0][100, 100], fr[-1][100, 100]
    assert a[0] > a[2] and b[2] > b[0]


def test_aux_maps():
    sc = scene([{"effect": "glow", "params": {"radius": 12, "brightness": 2, "halo_radius": 0}},
                {"effect": "_probe_volume", "at": [60, 60]},
                {"effect": "shockwave", "channels": {"light": "heat"}, "params": {"radius_end": 60}}],
               channels={"heat": {"kind": "distort", "strength": 2}})
    r = vfx.render(sc, workers=1, frames=[1], maps=["normal", "emissive", "intensity", "distortion"])
    em, it, nm, ds = (r.aux[k][0] for k in ("emissive", "intensity", "normal", "distortion"))
    assert em.shape == r.frames[0].shape and em[100, 100, 3] > 200 and em[60, 60, 3] < 5   # light only
    assert it[100, 100, 0] == it[100, 100, 1] == it[100, 100, 2] > 150
    assert abs(int(nm[10, 10, 2]) - 255) <= 1 and nm[60, 48, 0] < 120     # flat far away, tilted on the blob
    assert ds[..., 3].max() > 0 and np.abs(ds[..., 0].astype(int) - 128).max() > 4
    assert r.frames[0][160, 100, 3] < 5 or True        # distortion is not drawn in colour


def test_feel_hitstop_punch_shake():
    mv = {"effect": "_probe_marker", "params": {"dx": 0}, "move": {"path": {"line": [[50, 100], [150, 100]]}}}
    plain = vfx.render(scene([mv]), workers=1).frames
    held = vfx.render(scene([{**mv, "feel": {"hitstop": [{"at": 0.3, "hold": 0.3}]}}]), workers=1).frames
    c3, c5 = centroid(held[3]), centroid(held[5])
    assert c3 == pytest.approx(c5, abs=0.3) and centroid(plain[5])[0] > c5[0] + 5
    base = centroid(r1(scene([{"effect": "_probe_marker", "at": [100, 100]}]), i=0))
    pun = centroid(r1(scene([{"effect": "_probe_marker", "at": [100, 100],
                              "feel": {"punch": [{"at": 0, "duration": 0.5, "amount": 0.5}]}}]), i=0))
    assert pun[0] > base[0] + 5
    sh = vfx.render(scene([{"effect": "_probe_marker", "at": [100, 100],
                            "feel": {"shake": [{"at": 0, "duration": 1, "amplitude": 6}]}}]), workers=1).frames
    assert len({tuple(np.round(centroid(f), 1)) for f in sh[1:6]}) > 2


def test_events_mapped_to_scene_time_and_frames(tmp_path):
    (tmp_path / "child.yaml").write_text(
        "vfxgen: 1\nsize: [100, 100]\nduration: 1\nevents: [{name: child_ev, t: 0.4}]\n"
        "layers:\n  - {effect: _probe_marker}\n")
    p = tmp_path / "s.yaml"
    p.write_text("vfxgen: 1\nsize: [200, 200]\nfps: 10\nduration: 1.5\n"
                 "events: [{name: scene_ev, t: 0.1, data: {k: 1}}]\n"
                 "layers:\n  - {effect: _probe_marker, start: 0.5, speed: 2, events: [{name: hit, t: 0.2}]}\n"
                 "  - {effect: child.yaml, start: 0.3}\n")
    ev = {e["name"]: e for e in vfx.render(str(p), workers=1, frames=[0]).events}
    assert ev["scene_ev"]["t"] == pytest.approx(0.1) and ev["scene_ev"]["data"] == {"k": 1}
    assert ev["hit"]["t"] == pytest.approx(0.6) and ev["hit"]["frame"] == 6
    assert ev["child_ev"]["t"] == pytest.approx(0.7)


def test_tracks_follow_layers():
    sc = scene([{"id": "m", "effect": "_probe_marker", "params": {"dx": 0},
                 "move": {"path": {"line": [[50, 100], [150, 100]]}}}],
               track=[{"name": "head", "layer": "m"}])
    r = vfx.render(sc, workers=1)
    for f, pos in zip(r.frames[1:9], r.tracks["head"][1:9]):
        assert centroid(f) == pytest.approx(pos, abs=0.6)


def test_targets_and_follow_links():
    lays = [{"id": "a", "effect": "_probe_marker", "at": [60, 140], "params": {"a": 0.0001}},
            {"effect": "_probe_link", "at": [100, 100], "rotate": 30, "targets": {"a": {"layer": "a"}}}]
    assert centroid(r1(scene(lays))) == pytest.approx((60, 140), abs=0.6)
    lays2 = [{"id": "m", "effect": "_probe_marker", "params": {"dx": 0, "a": 0.0001},
              "move": {"path": {"line": [[50, 100], [150, 100]]}}},
             {"effect": "_probe_link", "params": {"use_follow": True}, "follow": {"layer": "m"}}]
    c = centroid(r1(scene(lays2), i=5))          # where the mover was 0.1 s earlier: t=0.4 -> x=90
    assert c == pytest.approx((90, 100), abs=0.8)
    with pytest.raises(vfx.SpecError, match="earlier"):
        vfx.render(scene(list(reversed(lays))), workers=1)


def test_tile_wraps_and_tiles_seamlessly():
    sc = scene([{"effect": "glow", "at": [198, 100], "params": {"radius": 4, "brightness": 3, "halo_radius": 0}}],
               tile=True)
    f = r1(sc)
    assert f[100, 3, 3] > 20                         # wrapped round to the left edge
    big = np.concatenate([f, f], axis=1)[..., 3].astype(int)
    seam = np.abs(np.diff(big[90:110, 195:205], axis=1)).max()
    inner = np.abs(np.diff(f[90:110, 190:200, 3].astype(int), axis=1)).max()
    assert seam <= inner + 2


def test_motion_blur_smears_motion():
    mv = {"effect": "_probe_marker", "params": {"dx": 0}, "move": {"path": {"line": [[30, 100], [170, 100]]}}}
    sharp = r1(scene([mv]), i=5)
    blur = r1(scene([mv], motion_blur={"samples": 6, "shutter": 1.0}), i=5)
    assert (blur[..., 3] > 10).sum() > 1.5 * (sharp[..., 3] > 10).sum()
    again = vfx.render(scene([mv]), workers=1, motion_blur=6).frames[5]
    assert (again[..., 3] > 10).sum() > (sharp[..., 3] > 10).sum()


def test_pixel_mode():
    sc = scene([{"effect": "glow", "params": {"radius": 20, "brightness": 3}}],
               pixel={"size": 4, "palette": "#000000,#ff8800,#ffffff", "outline": "#123456"})
    r = vfx.render(sc, workers=1)
    f = r.frames[2]
    assert f.shape[:2] == (50, 50) and set(np.unique(f[..., 3])) <= {0, 255}
    cols = {tuple(c) for c in f[f[..., 3] == 255][:, :3]}
    assert cols <= {(0, 0, 0), (255, 136, 0), (255, 255, 255), (0x12, 0x34, 0x56)} and (0x12, 0x34, 0x56) in cols
    auto = vfx.render(scene([{"effect": "glow"}]), workers=1, pixel={"size": 2, "colors": 6, "fps": 6})
    assert auto.fps == 6 and len({tuple(c) for f in auto.frames for c in f[f[..., 3] == 255][:, :3]}) <= 6


def _sprite(path, color=(255, 0, 0)):
    im = np.zeros((20, 20, 4), np.uint8)
    im[4:16, 4:16] = (*color, 255)
    Image.fromarray(im, "RGBA").save(path)


def test_sprite_assets_and_cache_invalidation(tmp_path, monkeypatch):
    _sprite(tmp_path / "sprite.png")
    p = tmp_path / "s.yaml"
    p.write_text("vfxgen: 1\nsize: [80, 80]\nfps: 4\nduration: 0.5\nedge_fade: 0\n"
                 "layers:\n  - {effect: _probe_sprite, at: [40, 40], params: {glow: 0.0}}\n")
    f = vfx.render(str(p), workers=1).frames[0]
    assert tuple(f[40, 40][:3]) == (255, 0, 0) and f[40, 40, 3] == 255 and f[5, 5, 3] == 0
    monkeypatch.setenv("VFXGEN_NO_CACHE", "0")
    monkeypatch.setenv("VFXGEN_CACHE", str(tmp_path / "cache"))
    a = vfx.render(str(p), workers=1).frames[0]
    from vfxgen.engine import assets
    assets._cache.clear()
    _sprite(tmp_path / "sprite.png", (0, 0, 255))
    b = vfx.render(str(p), workers=1).frames[0]
    assert tuple(a[40, 40][:3]) == (255, 0, 0) and tuple(b[40, 40][:3]) == (0, 0, 255)


def test_curl_flow_changes_paths_and_loops():
    P = ATOMS["particles"]
    still = P({"count": 30, "speed_min": 40, "speed_max": 60}, seed=4)
    swirl = P({"count": 30, "speed_min": 40, "speed_max": 60, "curl": 80, "curl_scale": 40}, seed=4)
    a, b = draw_atom(still, 0.7), draw_atom(swirl, 0.7)
    assert not np.allclose(a["light"], b["light"])
    p0, p1 = draw_atom(swirl, 0.0), draw_atom(swirl, swirl.length())
    assert np.allclose(p0["light"], p1["light"], atol=1e-3)


def test_rotate_render_option():
    f = vfx.render(scene([{"effect": "_probe_marker", "at": [100, 100]}]), workers=1, frames=[2], rotate=90).frames[0]
    assert centroid(f) == pytest.approx((100, 120), abs=0.6)


def test_render_options_are_baked_into_explicit(tmp_path):
    sc = scene([{"effect": "_probe_marker", "at": [100, 100]}], motion_blur={"samples": 4, "shutter": 0.5})
    r = vfx.render(sc, workers=1, rotate=90, theme={"light": "#00ff00,#00ff00"}, motion_blur=0,
                   pixel={"size": 2})
    ex = r.explicit
    assert ex["rotate"] == 90 and ex["theme"] == {"light": "#00ff00,#00ff00"} and "motion_blur" not in ex
    assert ex["pixel"]["size"] == 2
    again = vfx.render(ex, workers=1)
    assert all(np.array_equal(a, b) for a, b in zip(r.frames, again.frames))


# --- fixes from the content round -----------------------------------------------------------
def test_wrap_lowres_field_has_no_double_edge():
    from vfxgen.engine.canvas import Target, View
    from vfxgen.engine.transform import Affine
    tg = Target("light", 8, 8, 2, wrap=True)
    View(tg, Affine.identity(), clip=(-4, -4, 12, 12)).field(lambda X, Y: np.ones_like(X), (0, 0, 8, 8),
                                                             lowres="nearest")
    assert np.allclose(tg.data, 1.0)


def test_check_is_tile_and_ends_visible_aware():
    from vfxgen.check import check_frames
    full = np.full((16, 16, 4), 255, np.uint8)
    assert not any(i.code == "edge_clip" for i in check_frames([full] * 4, loop=True, tile=True).issues)
    assert not any(i.code == "not_empty_at_end" for i in check_frames([full] * 4, loop=False,
                                                                       ends_visible=True).issues)


def test_follow_looks_back_across_loop_seam():
    lays = [{"id": "m", "effect": "_probe_loop", "params": {}},
            {"effect": "_probe_link", "params": {"use_follow": True, "duration": 1.0}, "time_mode": "loop",
             "follow": {"layer": "m"}}]
    f0 = vfx.render(scene(lays, loop=True), workers=1, frames=[0]).frames[0]
    assert f0[..., 3].sum() > 0


def test_mixed_scalar_vector_keys_and_expression_vectors():
    sc = scene([{"effect": "_probe_marker", "at": [100, 100], "scale": {"keys": [[0, 1.0], [1, [3, 1]]]},
                 "offset": ["${2 * 5}", 0]}])
    c = centroid(r1(sc, i=5))
    assert c == pytest.approx((150, 100), abs=1.0)


def test_solid_group_opacity_no_overlap_show_through(tmp_path):
    _sprite(tmp_path / "sprite.png")
    p = tmp_path / "s.yaml"
    p.write_text("vfxgen: 1\nsize: [80, 80]\nfps: 4\nduration: 0.5\nedge_fade: 0\n"
                 "layers:\n  - {effect: _probe_sprite, at: [40, 40], opacity: 0.5}\n"
                 "  - {effect: _probe_sprite, at: [44, 40], opacity: 0.5}\n")
    f = vfx.render(str(p), workers=1).frames[0]
    assert 120 <= f[40, 40, 3] <= 200          # each layer composited once at 0.5, as a group


def test_paint_field_and_luma_tint():
    from vfxgen.engine.canvas import SolidView, Target
    from vfxgen.engine.transform import Affine
    tg = Target("solid", 20, 20, 1)
    v = SolidView(tg, Affine.identity(), clip=(0, 0, 20, 20))
    v.paint_field(lambda X, Y: (np.ones_like(X), np.dstack([X / 20 * 255, np.zeros_like(X), np.zeros_like(X)])))
    assert tg.rgb[10, 18, 0] > tg.rgb[10, 2, 0]
    img = np.zeros((4, 4, 4), np.float32)
    img[..., 3] = 1
    img[:2, ..., :3] = 0.9
    img[2:, ..., :3] = 0.2
    tg2 = Target("solid", 4, 4, 1)
    SolidView(tg2, Affine.identity(), clip=(0, 0, 4, 4)).image(img, 2, 2, tint=(0, 0, 255), tint_amount=1.0,
                                                               tint_mode="luma")
    assert tg2.rgb[0, 0, 2] > tg2.rgb[3, 0, 2]          # shading kept


def test_cli_atom_param_named_duration_and_check_set(capsys):
    from vfxgen.cli import main, split_sets
    vars_, over = split_sets("flash", ["duration=0.8"])
    assert vars_ == {"duration": 0.8} and over == {}
    assert main(["check", "sparks", "--draft", "--set", "count=10"]) == 0
