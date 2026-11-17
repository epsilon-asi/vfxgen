"""``vfxgen new``: starter files for atoms and scenes."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .errors import VfxError

ATOM_TEMPLATE = '''"""{title} - a user atom. Loaded automatically when this folder's parent is on VFXGEN_PATH:

    export VFXGEN_PATH={root}
    vfxgen describe {name}
    vfxgen render {name} --draft
"""
import math

from vfxgen.atom import Atom, Light, P, atom
from vfxgen.engine.mathx import TAU, smoothstep
from vfxgen.engine.noise import Noise


@atom("{name}", tags=["user"])
class {cls}(Atom):
    """{title}: describe what it looks like here."""
    kind = "oneshot"                       # or "loop" (then use a `period` param instead of `duration`)
    bounds = (-100, -100, 100, 100)        # local box containing everything ever drawn
    outputs = {{"light": Light(palette="arcane")}}
    streams = ("sparks",)                  # one named random stream per random part

    class Params(Atom.Params):
        duration: float = P(0.8, gt=0, doc="seconds until fully faded")
        radius: float = P(60.0, ge=0, doc="size of the effect (px)")
        count: int = P(12, ge=0, doc="number of sparks")
        brightness: float = P(1.5, ge=0, doc="peak intensity")

    def setup(self):
        rng = self.rng("sparks")
        self.angles = rng.uniform(0, TAU, self.p.count)

    def draw(self, ctx, t):
        p = self.p
        q = t / p.duration                 # 0 -> 1 over the effect
        fade = float(smoothstep(1.0, 0.6, q))
        ctx.light.dot(0, 0, 8 + 20 * q, p.brightness * fade)
        for a in self.angles:
            r = p.radius * q
            ctx.light.dot(r * math.cos(a), r * math.sin(a), 2.0, p.brightness * fade)
'''

SCENE_TEMPLATE = '''# yaml-language-server: $schema={schema}
vfxgen: 1
name: {name}
description: My effect.
size: [384, 384]
fps: 24
duration: 1.2
loop: false
seed: 1
vars:
  colors: {{default: arcane, doc: main colour ramp}}
  power: {{default: 1.0, min: 0, max: 3, doc: overall brightness}}
channels:
  light: {{kind: light, palette: "${{colors}}", exposure: "${{power}}"}}
layers:
  - id: flash
    effect: flash
    at: [192, 192]
  - id: wave
    effect: shockwave
    at: [192, 192]
    start: 0.05
    params: {{breakup: 0.5}}
  - id: sparks
    effect: burst
    at: [192, 192]
    params: {{count: 50, style: streak, speed_min: 120, speed_max: 280, drag: 2.5, gravity: 180}}
'''


def _ident(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise VfxError(f"{name!r} is not a valid name (letters, digits, underscores)")
    return name


def new_atom(name: str, directory: str = ".") -> Path:
    _ident(name)
    root = Path(directory).resolve()
    folder = root / "atoms"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.py"
    if path.exists():
        raise VfxError(f"{path} already exists")
    cls = "".join(w.capitalize() for w in name.split("_"))
    path.write_text(ATOM_TEMPLATE.format(name=name, cls=cls, title=name.replace("_", " ").title(), root=root))
    return path


def new_scene(name: str, directory: str = ".", from_: str | None = None) -> Path:
    _ident(name)
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.yaml"
    if path.exists():
        raise VfxError(f"{path} already exists")
    schema = root / "vfxgen.schema.json"
    if not schema.exists():
        import json
        import vfxgen
        schema.write_text(json.dumps(vfxgen.json_schema(), indent=1))
    if from_:
        from .spec.loader import LIBRARY_DIR, find_scene_file
        src = LIBRARY_DIR / f"{from_}.yaml"
        if not src.is_file():
            src = find_scene_file(from_, Path.cwd())
        text = src.read_text()
        text = re.sub(r"^name:.*$", f"name: {name}", text, count=1, flags=re.M)
        if not text.startswith("# yaml-language-server"):
            text = f"# yaml-language-server: $schema={schema.name}\n# copied from {src.name}\n" + text
        path.write_text(text)
    else:
        path.write_text(SCENE_TEMPLATE.format(name=name, schema=schema.name))
    return path
