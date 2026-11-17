"""Integration API used by the Workbench: validate, partial renders, render_frame, progress,
cancel, blend and pivot."""
from __future__ import annotations

import subprocess
import threading

import numpy as np
import pytest

import vfxgen as vfx
from vfxgen.errors import LimitError, RenderCancelled, SpecError

from . import probes  # noqa: F401

SCENE = {"vfxgen": 1, "size": [160, 120], "fps": 12, "duration": 1.0, "edge_fade": 0, "seed": 3,
         "layers": [{"effect": "burst", "params": {"count": 20, "style": "streak"}},
                    {"effect": "flash", "at": [60, 60]}]}


def test_validate_reports_without_rendering():
    info = vfx.validate(SCENE)
    assert (info.width, info.height, info.frame_count, info.instances) == (160, 120, 13, 2)
    assert info.blend == "alpha" and info.pivot == (0.5, 0.5)
    with pytest.raises(LimitError):
        vfx.validate({**SCENE, "size": [5000, 100]})
    with pytest.raises(LimitError):
        vfx.validate({**SCENE, "fps": 2000})
    with pytest.raises(SpecError):
        vfx.validate({**SCENE, "layers": [{"effect": "burst", "params": {"nope": 1}}]})


def test_partial_renders_match_full():
    full = vfx.render(SCENE, workers=1, cache=False)
    part = vfx.render(SCENE, workers=1, frames=[0, 6, 12])
    assert part.frame_indices == [0, 6, 12] and part.frame_count_total == 13
    for i, f in zip([0, 6, 12], part.frames):
        assert np.array_equal(f, full.frames[i])
    at = vfx.render(SCENE, workers=1, times=[0.5])
    assert np.array_equal(at.frames[0], full.frames[6])
    with pytest.raises(SpecError, match="out of range"):
        vfx.render(SCENE, workers=1, frames=[99])


def test_render_frame_starts_no_subprocess(monkeypatch):
    full = vfx.render(SCENE, workers=1, cache=False)

    def boom(*a, **k):
        raise AssertionError("subprocess started")
    monkeypatch.setattr(subprocess, "Popen", boom)
    f = vfx.render_frame(SCENE, 0.25)
    assert np.array_equal(f, full.frames[3])
    vfx.render(SCENE, workers=1, cache=False)          # full render with workers=1: no subprocess either


@pytest.mark.parametrize("workers", [1, 3])
def test_progress_reports_to_total(workers):
    seen = []
    big = {**SCENE, "duration": 2.0}
    r = vfx.render(big, workers=workers, cache=False, on_progress=lambda d, t: seen.append((d, t)))
    assert seen and seen[-1] == (len(r.frames), len(r.frames))
    assert all(a <= b for (a, _), (b, _) in zip(seen, seen[1:]))


def test_cancel_serial_and_parallel():
    ev = threading.Event()
    ev.set()
    with pytest.raises(RenderCancelled):
        vfx.render(SCENE, workers=1, cache=False, cancel=ev)
    ev2 = threading.Event()
    with pytest.raises(RenderCancelled):
        vfx.render({**SCENE, "duration": 3.0}, workers=3, cache=False, cancel=ev2,
                   on_progress=lambda d, t: ev2.set() if d >= 1 else None)
    with pytest.raises(RenderCancelled):
        vfx.render(SCENE, workers=1, cache=False, cancel=lambda: True)


def test_blend_and_pivot():
    r = vfx.render({**SCENE, "blend": "additive", "pivot": [0.5, 1.0]}, workers=1, frames=[0])
    assert r.blend == "additive" and r.pivot == (0.5, 1.0)
    r = vfx.render({**SCENE, "anchors": {"muzzle": [40, 30]}, "pivot": "muzzle"}, workers=1, frames=[0],
                   draft=True)
    assert r.pivot == (0.25, 0.25)
    r = vfx.render({**SCENE, "pivot": {"px": [80, 90]}}, workers=1, frames=[0])
    assert r.pivot == (0.5, 0.75)
