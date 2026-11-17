"""``vfxgen watch``: re-render a scene whenever it, any scene it references, or any atom
source file changes. Each render runs in a fresh subprocess so edits to Python atom code
take effect without restarting the watcher. Renders overwrite the latest generation in place.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .errors import VfxError


def referenced_files(target: str) -> set[Path]:
    """The scene file and every scene file it references (recursively), best effort."""
    from .spec.loader import find_scene_file, read_yaml
    out: set[Path] = set()

    def walk(p: Path, depth: int) -> None:
        if p in out or depth > 4:
            return
        out.add(p)
        try:
            raw = read_yaml(p.read_text()) or {}
        except Exception:
            return
        for lay in raw.get("layers") or []:
            eff = lay.get("effect") if isinstance(lay, dict) else None
            spawn = (lay.get("spawn") or {}).get("effect") if isinstance(lay, dict) else None
            for e in (eff, spawn):
                if isinstance(e, str) and ("/" in e or e.endswith((".yaml", ".yml"))):
                    try:
                        walk(find_scene_file(e, p.parent), depth + 1)
                    except Exception:
                        pass
                elif isinstance(e, str):
                    from .spec.loader import LIBRARY_DIR
                    lib = LIBRARY_DIR / f"{e}.yaml"
                    if lib.is_file():
                        walk(lib, depth + 1)
    p = Path(target)
    if p.is_file():
        walk(p.resolve(), 0)
    return out


def watch_dirs(target: str) -> list[Path]:
    import vfxgen
    pkg = Path(vfxgen.__file__).resolve().parent
    dirs = {f.parent for f in referenced_files(target)} | {pkg / "atoms", pkg / "library"}
    for d in os.environ.get("VFXGEN_PATH", "").split(os.pathsep):
        if d.strip():
            dirs.add(Path(d).expanduser().resolve())
    return sorted(d for d in dirs if d.is_dir())


def relevant(changes, files: set[Path]) -> bool:
    for _, path in changes:
        p = Path(path)
        if p.suffix == ".py" or p.resolve() in files or p.suffix in (".yaml", ".yml"):
            return True
    return False


def _cmd(a, target: str) -> list[str]:
    cmd = [sys.executable, "-m", "vfxgen.cli", "render", target, "--overwrite", "--check"]
    if not getattr(a, "final", False):
        cmd.append("--draft")
    for opt in ("colors", "seed", "fps", "duration", "size", "out", "name", "formats", "alpha", "workers"):
        v = getattr(a, opt, None)
        if v is not None:
            cmd += [f"--{opt}", str(v)]
    for s in a.set or []:
        cmd += ["--set", s]
    if getattr(a, "trim", False):
        cmd.append("--trim")
    return cmd


def watch(a) -> int:
    from watchfiles import watch as _watch
    target = a.targets[0]
    if not Path(target).is_file():
        raise VfxError("watch needs a scene file (copy a library effect with `vfxgen new scene NAME --from EFFECT`)")
    files = referenced_files(target)
    dirs = watch_dirs(target)
    print(f"watching {target} ({len(files)} scene file(s), {len(dirs)} folder(s)); Ctrl-C to stop")
    run = lambda: subprocess.run(_cmd(a, target))  # noqa: E731
    t0 = time.time()
    run()
    print(f"  ({time.time() - t0:.1f}s)")
    try:
        for changes in _watch(*dirs, debounce=300, step=100):
            files = referenced_files(target)
            if not relevant(changes, files):
                continue
            names = ", ".join(sorted({Path(p).name for _, p in changes}))
            print(f"\nchanged: {names}")
            t0 = time.time()
            run()
            print(f"  ({time.time() - t0:.1f}s)")
    except KeyboardInterrupt:
        pass
    return 0
