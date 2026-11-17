"""Built-in atoms. Importing this package registers every atom.

User atoms: any ``*.py`` file inside an ``atoms/`` folder of a ``VFXGEN_PATH``
directory is imported too (create one with ``vfxgen new atom <name>``).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import pkgutil
import sys
from pathlib import Path

import traceback
import warnings

_loaded_user: set[str] = set()
IMPORT_ERRORS: dict[str, str] = {}   # module -> traceback, for modules that failed to import


def _import_builtin() -> None:
    """Import every built-in atom module. A broken module is reported (warning +
    IMPORT_ERRORS) instead of taking the whole registry down."""
    for m in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        if m.name.startswith("_"):
            continue
        try:
            importlib.import_module(f"{__name__}.{m.name}")
        except Exception:
            IMPORT_ERRORS[m.name] = traceback.format_exc()
            warnings.warn(f"vfxgen: atom module {m.name!r} failed to import:\n{IMPORT_ERRORS[m.name]}")


def load_user_atoms() -> list[str]:
    """Import atom modules from ``<dir>/atoms/*.py`` for each VFXGEN_PATH dir."""
    found = []
    for d in os.environ.get("VFXGEN_PATH", "").split(os.pathsep):
        if not d.strip():
            continue
        folder = Path(d).expanduser() / "atoms"
        for f in sorted(folder.glob("*.py")) if folder.is_dir() else []:
            key = str(f.resolve())
            if key in _loaded_user:
                continue
            name = f"vfxgen_user_atoms.{f.stem}"
            spec = importlib.util.spec_from_file_location(name, f)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            _loaded_user.add(key)
            found.append(key)
    return found


_import_builtin()
load_user_atoms()
