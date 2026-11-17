#!/usr/bin/env python3
"""Re-create look-regression reference snapshots for library effects (all, or the names given)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["VFXGEN_NO_CACHE"] = "1"

import vfxgen as vfx  # noqa: E402
from tests.regression import snapshot as snap  # noqa: E402

names = sys.argv[1:] or [e["name"] for e in vfx.list_effects()]
for n in names:
    print(snap.save(n, snap.snapshot(n)))
