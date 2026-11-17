"""CLI commands, the gallery writer, export, check, cache and watch helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.unit import probes  # noqa: F401
from vfxgen.check import check_frames
from vfxgen.cli import main
from vfxgen.render import render

SCENE = """vfxgen: 1
name: demo
size: [160, 160]
fps: 8
duration: 0.5
vars:
  colors: {default: fire, doc: main colours}
  power: {default: 1.0, min: 0, max: 4, doc: brightness}
layers:
  - id: f
    effect: flash
    colors: "${colors}"
    params: {brightness: "${3 * power}"}
  - id: b
    effect: burst
    params: {count: 12, style: streak}
"""


@pytest.fixture
def scene_file(tmp_path):
    p = tmp_path / "demo.yaml"
    p.write_text(SCENE)
    return p


def run(*args):
    return main([str(a) for a in args])


def test_render_versions_latest_and_overwrite(scene_file, tmp_path, capsys):
    out = tmp_path / "g"
    assert run("render", scene_file, "--out", out, "--workers", 1) == 0
    assert run("render", scene_file, "--out", out, "--workers", 1, "--set", "power=2") == 0
    d = out / "demo"
    assert (d / "v001").is_dir() and (d / "v002").is_dir() and (d / "latest").resolve().name == "v002"
    files = {p.name for p in (d / "v002").iterdir()}
    assert {"demo.yaml", "demo_sheet.png", "demo_sheet.json", "demo.webp", "frames"} <= files
    y = (d / "v002" / "demo.yaml").read_text()
    assert "brightness: 6.0" in y and "# peak intensity" in y and "vfxgen render" in y
    assert run("render", scene_file, "--out", out, "--workers", 1, "--overwrite") == 0
    assert not (d / "v003").exists()
    assert "brightness: 3.0" in (d / "v002" / "demo.yaml").read_text()


def test_render_atom_with_params_and_colors(tmp_path):
    out = tmp_path / "g"
    assert run("render", "shockwave", "--set", "breakup=0.5", "--colors", "#33ccff", "--out", out,
               "--draft", "--workers", 1) == 0
    assert (out / "shockwave_custom" / "v001" / "shockwave_custom_sheet.png").exists() or \
        any(out.glob("shockwave*/v001"))


def test_dotted_override_and_seed_stream(scene_file, tmp_path):
    out = tmp_path / "g"
    assert run("render", scene_file, "--out", out, "--workers", 1, "--set", "layers.b.params.count=3",
               "--set", "layers.b.seeds.particles=42", "--name", "dotted") == 0
    y = (out / "dotted" / "v001" / "dotted.yaml").read_text()
    assert "count: 3" in y and "particles: 42" in y


def test_unknown_var_is_a_clear_error(scene_file, tmp_path, capsys):
    assert run("render", scene_file, "--set", "nope=1", "--out", tmp_path) == 2
    assert "no var named 'nope'" in capsys.readouterr().err


def test_sweep_and_variants(scene_file, tmp_path):
    out = tmp_path / "g"
    assert run("render", scene_file, "--out", out, "--workers", 1, "--draft", "--sweep", "power=0.5,2") == 0
    assert (out / "demo__power-0.5").is_dir() and (out / "demo__power-2").is_dir()
    assert run("render", scene_file, "--out", out, "--workers", 1, "--draft", "--variants", "2", "--seed", "5") == 0
    assert (out / "demo_s5").is_dir() and (out / "demo_s6").is_dir()


def test_trim_and_alpha_modes(scene_file, tmp_path):
    r = render(str(scene_file), workers=1)
    g = r.save(tmp_path / "g", trim=True, alpha="premultiplied", formats=["sheet", "atlas", "frames", "gif"])
    atlas = json.loads(g.files["atlas"].read_text())
    assert atlas["frame_size"][0] <= 160 and atlas["alpha"] == "premultiplied" and "gif" in g.files
    f0 = np.asarray(Image.open(sorted(g.files["frames"].glob("*.png"))[0]))
    assert f0.shape[0] == atlas["frame_size"][1]


def test_list_describe_schema(capsys, tmp_path):
    assert run("list") == 0
    out = capsys.readouterr().out
    assert "shockwave" in out and "PALETTES" in out
    assert run("describe", "burst") == 0
    assert "random streams (seedable): particles" in capsys.readouterr().out
    assert run("schema", "-o", tmp_path / "s.json") == 0
    assert "layers" in json.loads((tmp_path / "s.json").read_text())["properties"]


def test_new_scene_and_atom(tmp_path, monkeypatch, capsys):
    assert run("new", "scene", "mine", "--dir", tmp_path) == 0
    p = tmp_path / "mine.yaml"
    assert p.exists() and (tmp_path / "vfxgen.schema.json").exists()
    assert render(str(p), workers=1, draft=True).frames
    assert run("new", "atom", "my_pop", "--dir", tmp_path) == 0
    assert (tmp_path / "atoms" / "my_pop.py").exists()
    monkeypatch.setenv("VFXGEN_PATH", str(tmp_path))
    from vfxgen.atom import ATOMS, ensure_atoms
    ensure_atoms()
    assert "my_pop" in ATOMS
    assert render("my_pop", workers=1, draft=True).frames


def test_check_and_export(scene_file, tmp_path):
    r = render(str(scene_file), workers=1)
    g = r.save(tmp_path / "g")
    assert run("check", g.dir) == 0
    assert run("export", g.dir, "--trim", "--alpha", "additive", "--fps", "4", "--size", "80x80") == 0
    exp = g.dir / "export"
    atlas = json.loads(next(exp.glob("*_sheet.json")).read_text())
    assert atlas["fps"] == 4 and atlas["alpha"] == "additive"


def test_check_detects_problems():
    a = np.zeros((32, 32, 4), np.uint8)
    b = a.copy()
    b[:, :, 3] = 255
    rep = check_frames([b, a, a, b * 0 + 5], loop=False)
    assert any(i.code == "edge_clip" for i in rep.issues)
    seam = [a.copy() for _ in range(4)]
    for k, f in enumerate(seam):          # dot moves 1px per frame, then jumps back 3px at the seam
        f[10:12, 10 + k:12 + k, 3] = 200
    assert any(i.code == "loop_seam" for i in check_frames(seam, loop=True).issues)
    assert not check_frames([a, a], loop=True).ok


def test_render_cache(scene_file, tmp_path, monkeypatch):
    monkeypatch.setenv("VFXGEN_NO_CACHE", "0")
    monkeypatch.setenv("VFXGEN_CACHE", str(tmp_path / "cache"))
    from vfxgen.engine import cache
    a = render(str(scene_file), workers=1)
    assert cache.info()["entries"] == 1
    b = render(str(scene_file), workers=1)
    assert all(np.array_equal(x, y) for x, y in zip(a.frames, b.frames))
    render(str(scene_file), {"power": 2.0}, workers=1)
    assert cache.info()["entries"] == 2
    assert cache.clear() == 2


def test_watch_helpers(tmp_path):
    from vfxgen.watch import referenced_files, relevant
    (tmp_path / "child.yaml").write_text("vfxgen: 1\nlayers:\n  - effect: glow\n")
    p = tmp_path / "top.yaml"
    p.write_text("vfxgen: 1\nlayers:\n  - effect: child.yaml\n")
    files = referenced_files(str(p))
    assert {f.name for f in files} == {"top.yaml", "child.yaml"}
    assert relevant([(1, str(tmp_path / "child.yaml"))], files)
    assert not relevant([(1, str(tmp_path / "notes.txt"))], files)
