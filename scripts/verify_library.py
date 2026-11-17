#!/usr/bin/env python3
"""Render every library effect at full quality into a gallery, re-render each from its explicit
YAML and require byte-identical frames; also run quality checks. Prints one line per effect."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import vfxgen as vfx
from vfxgen.check import check_result

os.environ["VFXGEN_NO_CACHE"] = "1"
out = sys.argv[1] if len(sys.argv) > 1 else "gallery"
ok = True
t_all = time.time()
for e in vfx.list_effects():
    n = e["name"]
    r = vfx.render(n)
    g = r.save(out)
    r2 = vfx.render(str(g.yaml), workers=1)
    same = len(r.frames) == len(r2.frames) and all(np.array_equal(a, b) for a, b in zip(r.frames, r2.frames))
    rep = check_result(r)
    ok &= same and rep.ok
    issues = "; ".join(f"{i.code}" for i in rep.issues) or "-"
    print(f"{n:18s} {len(r.frames):3d}f {r.width}x{r.height} {r.elapsed:5.1f}s  reproducible={same}  check={'OK' if rep.ok else 'FAIL'} ({issues})", flush=True)
print(f"total {time.time() - t_all:.0f}s  {'ALL OK' if ok else 'PROBLEMS'}")
sys.exit(0 if ok else 1)
