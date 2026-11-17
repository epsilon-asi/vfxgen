"""Fails when an engine or atom change alters how a library effect looks.

After an intended look change, refresh the references:  uv run python scripts/update_snapshots.py [names]
"""
from __future__ import annotations

import os

import pytest

import vfxgen as vfx

from . import snapshot as snap

os.environ.setdefault("VFXGEN_NO_CACHE", "1")
NAMES = [e["name"] for e in vfx.list_effects()]
MEAN_TOL = 0.5      # mean abs difference allowed (0..255)
MAX_TOL = 48        # largest single-pixel difference allowed


@pytest.mark.parametrize("name", NAMES)
def test_look_unchanged(name):
    ref = snap.load(name)
    if ref is None:
        pytest.skip(f"no reference snapshot for {name} (run scripts/update_snapshots.py {name})")
    mean, mx = snap.compare(snap.snapshot(name), ref)
    assert mean <= MEAN_TOL and mx <= MAX_TOL, (
        f"{name} looks different from its reference (mean diff {mean:.3f}, max {mx}). If intended, run "
        f"`uv run python scripts/update_snapshots.py {name}`.")
