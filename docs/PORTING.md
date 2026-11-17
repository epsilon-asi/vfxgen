# Porting legacy effects to atoms + library scenes

Project root: the folder containing `pyproject.toml`. Python: `.venv/bin/python` (package installed
editable; `vfxgen` CLI at `uv run vfxgen`). Read `SPEC.md` (§1–5) first.

## What you are doing

The legacy effects (`legacy/vfxgen_legacy/fx_*.py`; their renders are in `out/<name>/` and
`out/preview/<name>_sheet_on_checker.png`) are monolithic functions. Re-express each as:

1. **Atoms** (Python building blocks, `src/vfxgen/atoms/<your modules>.py`), and
2. a **library scene** `src/vfxgen/library/<effect>.yaml` composing those atoms (plus the shared
   atoms) so it looks like the legacy effect, with the same tweakable knobs as `vars`.

Keep strongly coupled parts inside one atom (shared per-particle state, geometry derived from
another component). Split everything else into reusable atoms. Prefer the shared atoms where
they fit: `glow`, `flash`, `shockwave`, `ring` (`atoms/basic.py`) and `particles` (loop) /
`burst` (one-shot) (`atoms/particles.py`). `vfxgen describe <name>` shows their params.

## Files you may edit

Only the atom modules and library YAMLs assigned to you, plus your test file
`tests/library/test_<group>.py`. Do **not** edit `src/vfxgen/engine/*`, `src/vfxgen/spec/*`,
`atom.py`, `render.py`, `cli.py`, `check.py`, `output/*`, `atoms/basic.py`,
`atoms/particles.py`, `tests/conftest.py`, `tests/atoms/*`, or another builder's files. If you hit an
engine bug or missing feature, work around it inside your module and report it (exact repro) in
your final message.

## Atom API cheat sheet (`src/vfxgen/atom.py`, `engine/canvas.py`)

```python
from ..atom import Atom, atom, P, Light, Smoke, Volume, Solid
from ..engine.mathx import TAU, smoothstep, refine, ballistic, ease_out, blur
from ..engine.noise import Noise
from ..engine.particles import Emitter, EmitterConfig, draw_particles
from ..engine.color import register_palette, color_at

register_palette("explosion_smoke", "#1e1a1a,#4a4340,#968c82")   # module-level, if you need new presets

@atom("fire_cloud", tags=["fire", "smoke"])
class FireCloud(Atom):
    """One-line summary.

    More detail."""
    kind = "oneshot"                      # or "loop"
    outputs = {"light": Light(palette="fire", bloom=[[2, .25], [8, .2], [24, .15]], alpha_gamma=.75),
               "smoke": Smoke(palette="explosion_smoke", density=1.2, shadow=1.0, size_ref=384,
                              lit_by={"channels": ["light"], "amount": 0.8})}
    streams = ("puffs", "noise")          # one named random stream per random part
    anchors = {"base": (0, 60)}           # optional named points (local coords); may be callables (self, t)

    class Params(Atom.Params):
        duration: float = P(1.5, gt=0, doc="seconds")        # one-shots: duration; loops: period
        radius: float = P(80, ge=0, doc="cloud radius (px)")

    def setup(self):                      # all randomness here, via named streams
        rng = self.rng("puffs")
        self.noise = Noise(self.rng("noise"))

    def get_bounds(self):                 # local box that contains everything ever drawn
        r = self.p.radius * 2
        return (-r, -r, r, r)

    def draw(self, ctx, t):               # t = local seconds; depends on t only
        ctx.light.dot(0, 0, 12, 2.0)
        ctx.smoke.dot(0, 0, 40, 0.5)
        ctx.light.field(lambda X, Y: ..., self.get_bounds(), lowres="bilinear")
```

* **Local coordinates**: the atom's origin is wherever you like (usually the effect's natural
  pivot: explosion centre, flame base, bolt top). Work in pixels at scale 1 (legacy scale=1 px).
  Transforms (position, rotation, scale) are applied by the engine; never use canvas size.
* **Views** (`ctx.<role>`): `dot(x,y,r,a)`, `ring(cx,cy,R,width,a,squash,dashes,duty,phase)`,
  `polyline(pts,r,a,r_end,a_end)`, `polygon(pts,a,soft)`, `capsule(p0,p1,r0,r1,a,soft)`,
  `field(fn(X,Y), bounds, lowres=False|"nearest"|"bilinear")` (X, Y are LOCAL coordinate arrays),
  `private()` + `map(fn(X,Y,v))` + `add(view)` to post-process only your own deposits (e.g. erode
  your density with noise). Solid views: `polygon(pts, k=ramp_pos, a, soft)`, `capsule(...)`,
  `stroke(pts, r, k=...)` painted in call order (painter's order); colour = channel palette at `k`
  (or `rgb=(r,g,b)`).
* **Roles → channels**: each role deposits into a scene channel (default: channel named like the
  role). Channel settings (palette, bloom, exposure, alpha_gamma, smoke/volume shading) belong
  to the **channel**, not the atom; your role defaults are used when the scene does not declare
  the channel. Library scenes should declare their channels explicitly with the legacy render
  settings. Bloom sigmas are output px; smoke/volume blur scales derive from `size_ref` (set it to
  the legacy canvas size, `max(w, h)`, to reproduce legacy shading exactly).
  Legacy `fire_light` → smoke channel `lit_by: {channels: [light], amount: X}`.
  Legacy `_render_shaded` → smoke channel with `relief > 0` (+ `ambient`, `height_gain`).
  Legacy `_shade_volume` → `volume` channel. Legacy `Solid` → `solid` channel.
  Legacy `_stack(a, b, c)` order → channel declaration order (bottom first).
  Legacy single tone-map over several components → put those atoms in the SAME light channel.
  Legacy edge mask (10 px) is applied by the scene (`edge_fade: 10`).
* **Time**: `t` is seconds. One-shots must draw nothing (or ~nothing) at `t >= duration`; loops must
  satisfy draw(t) == draw(t + period): speeds are whole numbers of cycles per period, noise time axis
  `z = (t/period) * pz` with integer lacunarity. **Never use frame index / frame count**. Legacy
  frame-indexed logic → seconds. Legacy per-frame RNG → `self.tick_rng("stream", t, rate_hz)`.
  Legacy normalised time `u = f/(frames-1)` → `u = t / duration`. Legacy eased sampling
  (`t = T*u**1.35`) → leave the atom linear in seconds and set `time_warp: {curve: power, exp: 1.35}`
  on the layer in the library scene.
* **Randomness**: only `self.rng(stream)` / `self.tick_rng(stream, t, rate)`. Exact legacy random
  layouts are not required (seeds changed); the look must match.
* **Params**: every knob a `P(default, doc=..., bounds)`; ints/floats/bools/strings; string choices
  via `P(..., choices=[...])`. Any param naming a colour ramp ends in `palette` (rare: channel
  palettes are set on channels; only use for atom-internal colour like solid facets).

## Library scene conventions (`src/vfxgen/library/<name>.yaml`)

```yaml
vfxgen: 1
name: explosion
description: Fireball explosion: flash, shockwave, debris, cooling into billowing smoke.
size: [384, 384]          # legacy canvas size
fps: 18                   # legacy fps
duration: 1.5             # seconds (one-shot: legacy simulated length; loop: frames/fps)
loop: false
vars:                     # same knob names as the legacy effect params where meaningful
  colors: {default: fire, doc: main colour ramp}
  smoke_palette: {default: explosion_smoke, doc: smoke colour ramp}
  radius: {default: 1.0, min: 0.1, max: 4, doc: blast size multiplier}
channels:                 # bottom -> top, legacy render settings
  smoke: {kind: smoke, palette: "${smoke_palette}", density: 1.2, shadow: 1.0, size_ref: 384,
          lit_by: {channels: [light], amount: 0.8}}
  light: {kind: light, palette: "${colors}", bloom: [[2, .25], [8, .2], [24, .15]], alpha_gamma: 0.75}
layers:
  - {id: flash, effect: flash, at: [192, 238], params: {brightness: "${3.2 * radius}"}}
  - id: cloud
    effect: fire_cloud
    at: [192, 238]
    time_warp: {curve: power, exp: 1.35}
    params: {radius: "${80 * radius}"}
```

* `at` is the anchor position in scene px (default: scene centre); `anchor` defaults to the atom's
  `center` (bounds centre) - use `anchor: {px: [0, 0]}` to pin the atom's local origin.
* Vars are substituted everywhere with `${...}` (safe arithmetic + min/max/clamp/lerp/sin/...).
* Expose `colors` (main light palette) and one `<role>_palette` var per extra channel palette.
* Loops: `loop: true` and `duration × fps` must be whole; every looping layer's period must divide
  the duration.

## Workflow

```bash
uv run python scripts/preview.py <effect-or-atom> --draft -o /tmp/<you>      # fast look
uv run python scripts/preview.py <effect> -o /tmp/<you> --every 2            # full quality
uv run pytest tests/atoms -k <atom> -q                              # atom contract
uv run vfxgen check <effect>                                                  # loop seam / edges / empty end
```

Compare against the legacy look: `out/preview/<effect>_sheet_on_checker.png` (and `out/<effect>/`
frames). Read your preview PNGs with the Read tool and iterate until the new effect matches the
legacy one in shape, motion, colour and timing. Use `--colors "#ff3050"` and a few `--set` tweaks
to confirm vars work.

Write `tests/library/test_<group>.py`: for each of your library effects, render in draft
(`vfxgen.render(name, draft=True, workers=1)`), assert frames are RGBA, `vfxgen.check` reports no
errors, every var has a doc, and one recolour works. Keep the file under ~60 s.
