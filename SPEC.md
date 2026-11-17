# vfxgen 1.0 — Specification

Status: **implemented (v1.1.0)**. Section 14 lists where the build differs from the original draft;
section 15 covers the v1.1 additions.

vfxgen is a Python package that generates transparent VFX sprites (frames, sprite
sheets, animated previews) from **YAML scene files**. A scene is a stack of
**layers**; each layer is a building block (**atom**) or another scene, placed
anywhere, at any time, in any direction, repeated along paths or spawning copies
of itself. The 30 existing effects ship as editable library scenes built from
atoms.

It is built as a publishable package with a clean SDK, CLI and test suite, so a
FastAPI service can later wrap it with a thin layer. Single user, single machine
(macOS, Python 3.15, 10 cores); no cross-platform or multi-tenant concerns.

---

## 1. Concepts

| Term | Meaning |
|---|---|
| **Atom** | A building block implemented in Python: `flame`, `ring`, `bolt`, `particles`, … Typed, documented parameters. Draws into one or more channels at any time *t* (seconds). |
| **Scene** | A YAML document: canvas size, fps, duration, channels, variables and a list of layers. A scene can be used as a layer inside another scene. |
| **Library effect** | A scene shipped with the package (`explosion`, `fire`, …). Referenced by name. |
| **Layer** | One entry in a scene's `layers:` list. References an atom or a scene, plus placement, timing, colours, overrides, repeat/spawn rules. |
| **Channel** | A shared buffer that layers draw into, with its own render settings. Everything in one channel is summed *before* it is converted to colour, so overlaps brighten correctly (fire + embers, crossing slashes). |
| **Instance** | One concrete drawing of an atom after repeats, spawns and nesting are expanded. The renderer works on a flat list of instances. |
| **Generation** | One render written to the gallery folder: frames, sheet, atlas, preview and a fully explicit YAML. |

### 1.1 Why channels (from the effect inventory)

The effects get their look from adding light up *before* the tone-map, the
non-linear step that turns brightness into colour and alpha. Compositing finished
per-layer images would visibly change every overlap. Smoke, dust and water are
shaded from the *summed* density field. So layers don't each produce an image.
They deposit into shared channels, and each channel is rendered once.

There are four channel kinds, matching how the existing effects actually render:

| Kind | Holds | Rendered as | Used by today |
|---|---|---|---|
| `light` | emitted light (additive float) | bloom → tone-map through palette → colour + alpha | fire, energy, sparks, glints… |
| `smoke` | density (additive float) | flat smoke shading: shifted self-shadow, optional tint from light channels (`fire_light`), and optional relief shading from light direction | smoke, poison, explosion smoke, mist, portal eye |
| `volume` | density (additive float) | surface shading: normals from blurred density, wrap-Lambert, occlusion, rim, specular | dust cloud, water |
| `solid` | painted RGBA (painter's order) | flat/faceted opaque matter | ice shards, crystal fragments, rocks, cracks |

Channels are stacked bottom→top in the order the scene lists them. The default
blend is `over`. `add` and `screen` are available, and light channels default to
`over`, which matches today's light-over-smoke.

### 1.2 Transforms are applied to geometry, not pixels

Every atom draws in its own **local coordinates**. The canvas hands the atom
local coordinates for every scene pixel (inverse transform), so translation,
rotation, non-uniform scale and flips stay pixel-sharp and analytic. Shading
(smoke shadows, volume lighting, facet lighting) happens per channel in scene
space, so light direction stays consistent when a layer is rotated.

---

## 2. Scene file format

All units: pixels (scene space, origin top-left, y down), seconds and degrees.
Every field below is optional unless marked **required**.

```yaml
# yaml-language-server: $schema=../vfxgen.schema.json   (editor autocompletion)
vfxgen: 1                     # format version (required)
name: storm_strike            # output name (default: file stem)
description: Charge, strike, explode, dust trail.

size: [512, 512]              # [w, h] or "auto" (union of layer bounds + margin)
fps: 24
duration: 1.5                 # seconds; "auto" = latest layer end (one-shots) / longest loop
loop: false                   # true = output must loop seamlessly (checked)
seed: 7
background: transparent       # always transparent for sprites; colour only for previews

vars:                         # tweakables exposed by this scene (CLI --set, SDK kwargs)
  strike_x: {default: 256, doc: "x position of the strike"}
  power: {default: 1.0, min: 0, max: 3, doc: "overall intensity"}
  colors: {default: electric, doc: "main colour ramp"}

channels:                     # bottom → top; omitted channels are auto-created (see §4)
  dust:  {kind: volume, palette: dust, opacity: 1.1, bump: 40, shadow: 0.3}
  smoke: {kind: smoke, palette: explosion_smoke, opacity: 1.2, fire_light: 0.8}
  light: {kind: light, palette: "${colors}", bloom: [[1.5, 0.6], [5, 0.5], [14, 0.35]],
          exposure: "${power}", alpha_gamma: 0.85}

layers:
  - id: charge                # optional id (needed for attach/lit_by/masks)
    effect: charge_up         # atom name | library effect | path/to/scene.yaml
    at: ["${strike_x}", 120]
    start: 0.0
    params: {particles: 80}   # atom params, or vars of a referenced scene

  - id: bolt
    effect: bolt
    at: ["${strike_x}", 0]
    anchor: top               # which point of the effect sits on `at`
    rotate: 10
    start: 0.45
    channels: {light: light}  # map the atom's output roles → this scene's channels

  - effect: explosion         # library scene used as a layer (nested, depth ≤ 3)
    at: ["${strike_x}", 470]
    anchor: bottom
    start: 0.5
    scale: 0.8
    params: {radius: 1.2, smoke_amount: 0.8}

  - effect: fireball
    move: {path: {line: [[0, 256], [512, 256]]}, ease: in_out}
    face_motion: true         # rotate to point along the path
    speed: 1.5                # local time runs 1.5× faster

  - effect: dust_puff
    start: 0.55
    repeat:                   # copies over a course
      count: 5
      every: 0.08             # seconds between copies
      along: {line: [[150, 480], [360, 480]]}
      jitter: {scale: 0.2, rotate: 15, time: 0.03}

  - effect: scenes/impact_ring.yaml
    at: [256, 470]
    start: 0.5
```

### 2.1 Layer fields

| Field | Type | Default | Meaning |
|---|---|---|---|
| `id` | str | auto (`layer3`) | Reference name for other layers. |
| `effect` | str | **required** | Atom name, library effect name, or `.yaml` path (relative to this file, then search paths). |
| `params` | map | `{}` | Atom parameters, or a referenced scene's `vars`. |
| `colors` | colour spec | inherited | Shortcut for the main palette. For an atom, sets the palette of the channel it draws into (see §4.3). For a scene, sets its `colors` var. |
| `at` | position · animatable | scene centre | Where the layer's anchor point goes, in the parent's coordinates. `[x, y]` = exact pixels; `{uv: [0.25, 0.8]}` = fraction of the parent canvas. |
| `anchor` | anchor | `center` | Which point **of the effect** is pinned to `at`. Defaults to the middle of the effect for every layer: an atom's bounds centre, or a nested scene's canvas centre. See §2.1.1. |
| `offset` | [dx, dy] · animatable | [0, 0] | Extra nudge in parent pixels, applied after `at` / `move` / `attach`. |
| `rotate` | deg · animatable | 0 | Clockwise. |
| `scale` | num or [sx, sy] · animatable | 1 | Geometric scale (exact, no resampling). |
| `flip` | `x`, `y`, `xy` | none | Mirror. |
| `opacity` | 0..1 · animatable | 1 | Multiplies the layer's deposits. Light is scaled linearly, before the tone-map. |
| `start` | s | 0 | When the layer's local time 0 happens in the parent's time. |
| `length` | s | effect's natural length | Clip; the layer is inactive after start + length. |
| `speed` | num | 1 | Local time rate. |
| `time_mode` | `once`, `loop`, `hold`, `pingpong` | `loop` for loop effects, `once` otherwise | What happens past the effect's natural length. `hold` freezes the last state. |
| `time_warp` | {curve, exp, offset, span} | linear | Remaps local time over `span` s (default: natural length): `t' = span·(offset + (1−offset)·curve(t/span))`. `{curve: power, exp: 1.35}` front-loads an explosion. |
| `fade` | {in: s, out: s} | 0 | Opacity ramps at the clip ends. |
| `move` | path motion | none | Animates `at` along a path (see §2.4), optionally with `face_motion: true`. |
| `attach` | {layer: id, anchor, rotate: bool} | none | Follows another layer's (possibly moving) anchor instead of `at`; `rotate: true` also inherits that layer's rotation. Use `offset` to nudge. |
| `mask` | mask spec | none | Multiplies deposits by a scalar field (sweep, radial, linear, noise), e.g. the magic circle's sweep. |
| `channels` | {role: channel} | auto | Routes the atom's output roles to scene channels. |
| `z` | int | list order | Draw order within a `solid` channel (painter's order). |
| `seed` | int | derived | Per-layer seed for all of this layer's randomness. Derived from the scene seed and the layer id when omitted, so editing one layer never reshuffles another. See §3.1. |
| `seeds` | {stream: int} | derived | Pins or changes individual random parts of the effect, e.g. `{sparks: 12}` reshuffles only the sparks. Stream names are listed by `vfxgen describe`. See §3.1. |
| `repeat` | repeat spec | none | §2.5 |
| `spawn` | spawn spec | none | §2.6 |
| `enabled` | bool | true | Quick toggle while iterating. |

#### 2.1.1 Positioning: `at` + `anchor`

Placement is always "put **this point of the effect** (`anchor`) at **that point of the parent** (`at`)". Rotation and scale pivot around the anchor. This works the same for atoms, library effects and scenes pulled in from other YAML files.

`anchor` accepts:

| Form | Meaning | Example |
|---|---|---|
| `center` (default) | middle of the effect | |
| a name | a named point the effect defines: `top`, `bottom`, `left`, `right`, `top_left`, … for any effect; plus effect-specific ones like `base` (fire), `impact` (bolt), `emitter` (beam), `mid_arc` (slash). A scene YAML can define its own under `anchors:`. | `anchor: impact` |
| `{px: [x, y]}` | exact pixel in the effect's own coordinate space (an atom's local units, or a nested scene's canvas pixels) | `anchor: {px: [128, 300]}` |
| `{uv: [u, v]}` or `[u, v]` | fraction of the effect's bounds; `[0.5, 1]` = bottom centre | `anchor: [0.5, 1]` |

`at` accepts `[x, y]` (exact parent pixels) or `{uv: [u, v]}` (fraction of the parent canvas). `vfxgen describe <name>` prints an effect's size, bounds and named anchors.

Example: pin the bottom-centre of a nested scene at exactly (412, 380), then nudge it 6 px left:

```yaml
- effect: scenes/impact_ring.yaml
  at: [412, 380]
  anchor: [0.5, 1.0]
  offset: [-6, 0]
```

A scene file declares its own named anchors like this:

```yaml
size: [384, 384]
anchors: {ground: [192, 340], muzzle: {uv: [0.9, 0.5]}}
```

### 2.2 Animatable values

Any field marked *animatable* accepts any of these:
- **a constant**: `rotate: 30`
- **keyframes**, with t in the layer's local seconds and a per-segment ease (`linear`, `in`, `out`, `in_out`, `step`, `power:N`, `back`, `bounce`):
  ```yaml
  scale: {keys: [[0, 0.2], [0.3, 1.0, out], [1.2, 0.8]]}
  ```
- **an expression** using `t` (local seconds), `T` (scene seconds), `p` (local progress 0..1) and vars:
  ```yaml
  rotate: "${360 * t / 2}"
  ```

### 2.3 Variables and expressions

- `${...}` is evaluated by a **safe expression evaluator**. It is a whitelisted AST: numbers, + − × ÷ // % **, comparisons, `a if c else b`, and `min max abs clamp lerp round floor ceil sqrt sin cos tan pi tau smoothstep`. It has no attribute access, imports or calls outside the list.
- Expressions without `t`/`T`/`p` are resolved once at load. Expressions with them are evaluated every frame.
- A whole-string `${x}` keeps the value's type (number, string, list). `${x}` embedded in a longer string is substituted as text.
- Built-in vars: `seed`, `fps`, `duration`, `width`, `height`, `depth` (nesting/spawn generation).
- YAML anchors (`&name` / `*name`) work for reusing blocks (shared masks, shared params).

### 2.4 Paths

Paths are used by `move`, `repeat.along` and `spawn.at`.

| Path | Fields |
|---|---|
| `line` | `[[x0,y0],[x1,y1]]` |
| `polyline` | points, `closed` |
| `bezier` | cubic control points (chained) |
| `arc` | `center`, `radius` (or `[rx, ry]`), `from`, `to` (deg) |
| `circle` | `center`, `radius`; also `ellipse` with `[rx, ry]`, `rotate` |
| `spiral` | `center`, `r0`, `r1`, `turns` |
| `points` | explicit list |

Every path has `spacing: even` (equal arc length, the default) or `random`. Motion along it has `ease`, and `loop` for closed paths.

### 2.5 `repeat` — copies over a course

```yaml
repeat:
  count: 8
  every: 0.06              # time offset between copies (s); 0 = simultaneous
  along: {arc: {center: [256, 256], radius: 120, from: 0, to: 360}}
  align: true              # rotate each copy to the path tangent
  order: forward           # forward | reverse | random | pingpong
  jitter:                  # per-copy uniform random ±
    position: 6
    rotate: 15
    scale: 0.2
    time: 0.02
    speed: 0.1
    hue: 10                # degrees of hue shift on the copy's palette
  ramp:                    # linear from first copy → last copy
    scale: [1.0, 0.4]
    opacity: [1.0, 0.3]
  new_seed: true           # each copy gets its own seed (default true)
```

Without `along`, copies stack at `at`, so it is a pure time repeat (`count: 4, every: 0.25` = four strikes). Repeats nest: a repeated scene may itself contain repeats.

### 2.6 `spawn` — self-spawning, up to depth 3

```yaml
spawn:
  effect: self             # self (default) or any effect reference
  depth: 3                 # generations below this one (max 3)
  count: 4                 # children per parent
  at: {circle: {radius: 60}}       # placement relative to the parent's anchor
  when: 0.15               # child start, seconds after the parent's start (or [min, max])
  scale: 0.5               # multiplied per generation
  opacity: 0.8             # multiplied per generation
  jitter: {rotate: 30, position: 10, time: 0.05}
```

`effect: self` re-instantiates the same layer with `depth − 1`. Children inherit the parent's transform, then apply their own. `${depth}` is available inside params (e.g. `params: {branches: "${3 - depth}"}`).

### 2.7 References and limits

- `effect:` resolution order: a `.yaml` path (relative to the referencing file, then `VFXGEN_PATH` dirs, then the library), then the library effect name, then the atom name. Ambiguity is an error.
- **Nesting depth ≤ 3** for scene references and for spawn generations, counted separately. Cycles are detected and reported with the full chain (`a.yaml → b.yaml → a.yaml`).
- Hard limits (configurable): 20,000 instances, 4096 px per side, 1,000 frames. Exceeding a limit raises `LimitError` naming the culprit layer.
- A referenced scene's channels are **namespaced** under the layer (`explosion.light`), so the child keeps its exact look. Copies of the same child share its channels, so two overlapping explosions merge their glow. `isolate: true` gives each copy its own channels.

---

## 3. Time model

- **Only the final output has a frame rate.** Atoms and nested scenes have no fps of their own. They are functions of time, and the top-level scene's `fps` decides when they are sampled. A nested scene's own `fps` is ignored.
- Frame *i* samples scene time **`T_i = i / fps`**, so playback at `fps` is always real-time.
  - **Looping scenes:** N = `duration · fps` frames, covering 0 … duration − 1/fps. Frame N would equal frame 0, so the loop is seamless. `duration · fps` must be a whole number (validated).
  - **One-shots:** N = `floor(duration · fps) + 1` frames, so the last frame lands on `duration`.
- Each atom declares either `duration` (one-shot, seconds) or `period` (loop, seconds) as a normal parameter. Looping speeds are integer "cycles per period", so any value stays seamless.
- Local time for a layer = `time_warp( (T − start) · speed )`, then wrapped by `time_mode`.
- **No atom may depend on the frame index or frame count.** Flicker and re-jag style randomness (lightning, crackle, muzzle petals) uses a `tick_rate` parameter in Hz. The random state for time t is `hash(seed, floor(t · tick_rate))`, wrapped by the period for loops. Changing fps then changes only sampling, never the look.
- Everything is closed-form in t. The inventory confirmed no effect steps a simulation, so any frame renders independently. That is what makes parallel rendering, time remapping and caching possible.

### 3.1 Seeds and pixel-perfect re-renders

**Guarantee:** the same scene, vars and seeds produce byte-identical frames, every time, at any worker count, with or without the cache. This is tested.

Every random number comes from a seed you can set. There are three levels, and each level is derived from the one above unless you set it explicitly:

1. **Scene `seed`** (`--seed N` on the CLI): the root of everything.
2. **Layer `seed`**: this layer's base seed. Default = `hash(scene seed, layer id)`. Repeat copies and spawn children get `hash(layer seed, copy index)` and `hash(layer seed, generation, index)`, overridable with `repeat.seeds: [..]`.
3. **Stream seeds** (`seeds: {name: n}`). Each atom draws its randomness from **named streams**, one per random part. For example, the `fire_cloud` atom has `puffs`, `noise`, `sparks` and `debris` streams, and `bolt` has `skeleton`, `branches` and `flicker`. Default = `hash(layer seed, stream name)`. Pinning one stream reshuffles only that part and leaves the rest exactly as it was.

Consequences:
- Changing a count (`sparks: 40 → 60`) never changes the random values of the *other* parts, because each part has its own stream.
- Tick randomness (flicker) is `hash(stream seed, tick index)`, so it is reproducible too.
- The explicit YAML written with every render (§6) records the scene seed, every layer seed, and every stream seed actually used. Re-rendering it is byte-identical even if the default derivation scheme changed in a later version.
- The CLI accepts `--set layers.<id>.seed=5` and `--set layers.<id>.seeds.sparks=12`. `--variants N` varies only the scene seed.

(§5.4's "not bit-identical" refers only to the *old* renders in `out/`: the rebuilt effects use new seeds, so their random layouts differ from today's files. Re-renders of anything made with the new system are always exact.)

---

## 4. Channels and colour

### 4.1 Channel settings

| Kind | Settings (all optional) |
|---|---|
| all | `palette` (colour spec), `blend` (`over`, `add`, `screen`), `opacity`, `adjust` {hue, saturation, brightness, contrast} |
| `light` | `bloom` [[sigma_px, weight], …], `exposure`, `alpha_gamma`, `glow_from` {channel, amount, blur} (light derived from a density channel, e.g. poison glow) |
| `smoke` | `opacity`, `shadow`, `light_dir` (deg), `relief`, `ambient`, `lit_by` {channels: [light…], amount} (was `fire_light`) |
| `volume` | `opacity`, `bump`, `soft`, `shadow`, `wrap`, `spec`, `spec_power`, `rim`, `light` [x, y, z] |
| `solid` | (palette, blend, opacity, adjust only; facet lighting is computed by the atom, see §14) |

Lengths in settings (bloom sigma, blur radii, shadow offsets) are in scene pixels. They scale with the layer's scale when a scene is nested, so a nested explosion at `scale: 0.5` looks like a smaller explosion, not a blurrier one.

### 4.2 Colour specs

Used for `colors`, any `palette`, and any atom param whose name ends in `palette`:
- preset name: `fire`, `arcane`, `smoke`, `dust`, `water`, …
- one hex: `"#2a7fff"`, auto-ramped (deep, then the colour, then pale, then white)
- list: `"#300,#f30,#fc6,#fff"`, faint→hot for light, shadow→lit for smoke and volume
- explicit stops: `"0:#300,0.4:#f30,1:#fff"`

Palettes are also loadable from YAML (`palettes:` block in a scene, or a palette file on `VFXGEN_PATH`).

### 4.3 How atoms map to channels

Each atom declares **output roles** with a kind and default settings, e.g.
`fire_cloud: {light: light(palette=fire, bloom=…), smoke: smoke(palette=explosion_smoke, …)}`.

- If a layer doesn't route a role, it goes to the scene channel with the role's name. If that channel doesn't exist, it is auto-created from the atom's default settings. A scene containing a single atom therefore looks exactly like that atom's tuned look.
- A layer whose `colors` differs from its target channel's palette is moved to an auto-created sibling channel (`light#2`). Different colours can't share one tone-map, so they're rendered separately and composited.

---

## 5. Atoms

### 5.1 Authoring API

```python
from vfxgen.atom import Atom, atom, P, Light, Smoke, Volume, Solid

@atom("shockwave", tags=["ring", "burst"])
class Shockwave(Atom):
    """Expanding ring of light that fades as it grows."""

    bounds = (-140, -140, 140, 140)          # local box at scale 1 (for size: auto, trim, culling)
    anchors = {"center": (0, 0)}             # named points (may be methods of t)
    outputs = {"light": Light(palette="fire", bloom=[(1.5, .6), (5, .5)], alpha_gamma=.85)}

    class Params(Atom.Params):               # pydantic model → validation, docs, JSON schema
        duration: float = P(0.35, gt=0, doc="seconds until fully faded")
        radius: float = P(120, ge=0, doc="final radius (px)")
        width: float = P(3, gt=0, doc="ring thickness (px)")
        brightness: float = P(0.9, ge=0, doc="peak intensity")
        breakup: float = P(0.0, ge=0, le=1, doc="noise breakup of the ring")

    streams = ("noise",)                     # named random streams (seedable individually)

    def setup(self):                         # ALL randomness here (or via self.tick_rng in draw)
        self.noise = Noise(self.rng("noise"))

    def draw(self, ctx, t):                  # t: local seconds; ctx.light etc. are channel canvases
        p = t / self.p.duration
        R = self.p.radius * ease_out(p)
        ctx.light.ring(0, 0, R, self.p.width, self.p.brightness * (1 - p) ** 2)
```

Contract (enforced by the shared test suite):
- **Deterministic:** same params + seeds + t ⇒ identical deposits.
- **Named random streams:** randomness comes only from `self.rng("name")` (a NumPy Generator seeded from the stream seed, §3.1) inside `setup`, or from `self.tick_rng("name", t, rate)` inside `draw`. Atoms declare their stream names (`streams = ("puffs", "sparks", ...)`) so `describe` can list them and the explicit YAML can record them.
- **Time-pure:** `draw` depends only on t. It never uses frame index or frame count, never accumulates state, and only draws randomness via `setup(rng)` or `self.tick_rng(t, rate)`.
- One-shots are empty at `t ≥ duration`. Loops satisfy draw(t) == draw(t + period).
- Draws stay within `bounds`, so nothing hard-clips.
- Every parameter has a type, a default, a doc string and sensible min/max.
- A parameter ending in `palette` is a colour spec.
- Atom code never hard-codes canvas size. It works in local units, and scale comes from the layer transform.

### 5.2 Drawing toolkit

These are available to every atom:
- **Canvas primitives** in local coordinates, transformed exactly:
  - `dot`, `ring` (squash, dashes, n-fold), `polyline` (taper), `polygon`, `capsule`, `star`
  - `field(fn)`: an arbitrary function of local X/Y, evaluated only inside the atom's transformed bounds
- **Solid painting:** `solid.polygon / capsule / stroke` with facet colour from a palette position and scene light.
- **Noise:** periodic 3D Perlin, `fbm`, domain warp, tileable textures, low-res evaluation with upsampling.
- **`refine`** (midpoint displacement), easing curves, `smoothstep`, closed-form ballistics (drag + gravity), and a ground-plane helper (projection, bounce).
- **Particle system** (`vfxgen.particles`): closed-form emitters, evaluated at any t with no stepping.
  - Shapes: point, ring, disk, line, box, ellipse, cone, path.
  - Burst or continuous (looping) emission.
  - Life, speed, direction/spread, gravity, drag, swirl, inflow (spiral to a point), sway, twinkle.
  - Size, alpha and colour-position over life.
  - Drawn as dot, streak, glint, plus, shard or puff.

  It is also exposed directly as the `particles` atom, so many new effects need **only YAML**.

### 5.3 Atom catalogue (decomposition of today's effects)

Strongly coupled parts stay inside one atom. Examples: lightning branches grow from the bolt's own skeleton, water drops pinch off the crown's fingers, and the explosion's fire and smoke share each puff's heat. Everything else is split.

| Atom | Roles | Covers (today) |
|---|---|---|
| `glow` | light | flashes, cores, halos, afterglows, impact flares (pulse / flicker / decay envelopes) |
| `ring` | light / smoke / volume | shockwaves, ground rings, dash and tick rings, ripples, converge rings (expand curve, squash, dashes, noise breakup, inner haze) |
| `ticks` | light | magic-circle tick band |
| `star_shape` | light | hexagram, inner hexagon, 8-point star, tip circles |
| `star_burst` | light | hit-impact spiky star |
| `glints` | light | sparkle field, single glints, cross flash, flying glints |
| `particles` | light / smoke / volume / solid | sparks, embers, motes, bits, pop sparks, dust kicks, ice dust, pebbles, inflow, plus-sparkles |
| `speed_lines` | light | radial speed lines |
| `smoke_puffs` | smoke / volume | smoke column, poison gas, dust puff, muzzle wisp, cold mist (modes: column, roll, burst, ring) |
| `bubbles` | light | poison bubbles |
| `flame` | light | fire, torch (teardrop noise flame) |
| `flame_trail` | light | fireball head + trail |
| `fire_cloud` | light + smoke | explosion fireball cooling into smoke |
| `debris_trails` | light + smoke | burning debris with smoke trails |
| `muzzle_burst` | light | petals + side spikes + core |
| `bolt` | light | lightning (branches, restrike, impact anchor) |
| `arc_field` | light | electric crackle |
| `wobble_shells` | light | orb shells |
| `orbiters` | light | orb motes, magic-circle sparks |
| `rune_ring` | light | procedural glyph bands |
| `vortex` | light + smoke | portal body, licks, in-spiralling particles, eye |
| `noisy_rim` | light | portal rim |
| `beam_line` | light | beam core, sheath, filaments, surge (endpoint anchors `emitter` / `impact`) |
| `spike_flare` | light | beam emitter flare (6-fold seam bug fixed) |
| `splash_fan` | light | beam impact fan + wall flare |
| `sphere_shell` | light | shield (grid, fresnel, wave and twinkle modulators; seam fixed) |
| `light_column` | light | heal column, holy pillar body (width curve, head, ground clip) |
| `anamorphic_streak` | light | pillar impact streak |
| `wind_funnel` | light | whirlwind bands, streaks, base specks |
| `crescent_slash` | light | slash crescent + arc speed lines + edge sparks (anchors `mid_arc`, `head`) |
| `ice_shards` | solid + light | frost-nova shards |
| `crystal_shatter` | solid + light | ice shatter (crack phase in seconds) |
| `ground_cracks` | solid + light | cracks, crater, hot glow (ground plane) |
| `dust_cloud` | volume | ground-slam billows |
| `rocks` | solid | ground-slam rocks + pebbles with bounce |
| `water_crown` | volume + light | crown sheet, fingers, beads, pinched drops, glints |
| `water_jet` | volume + light | central jet + bead |
| `masks`: `sweep`, `radial`, `linear`, `noise` | mask | magic-circle sweep, fades, wipes |

Examples of library scenes built from these:
- `explosion` = `glow`(flash) + `ring`(shockwave, breakup) + `fire_cloud` + `particles`(sparks) + `debris_trails`. Channels: smoke, light.
- `slash_combo` = 3 × `crescent_slash` with staggered starts + `glints` attached to each slash's `mid_arc`.
- `water_splash` = `water_crown` + `water_jet` + `particles`(spray, routed to the shared `water` volume channel) + `ring`×2 (ripples into the volume) + `smoke_puffs`(mist).

### 5.4 Faithful but not bit-identical

The rebuilt effects will match today's look closely, with some expected differences:
- **Different random layouts versus the old files only:** each layer and random part now has its own seed (§3.1), so the positions of sparks, puffs and so on differ from today's `out/` renders. Re-renders within the new system are always pixel-perfect.
- **Seam fixes:** the beam emitter and shield seams are fixed.
- **Timing now in seconds:** effects whose timing depended on frame count (holy pillar slam, ice-shatter crack phase, crackle, lightning flicker) now use seconds and `tick_rate`.

Each library scene exposes the same tweakable variables the effect has today (`radius`, `smoke_amount`, `debris`, …), wired through `vars` to the atoms. A **fidelity sheet** (`scripts/compare_legacy.py`) renders old and new side by side for review.

---

## 6. Output: the gallery folder

`vfxgen render` writes one **generation** per render into the gallery folder (default `./gallery`):

```
gallery/
  explosion/
    v001/
      explosion.yaml          # fully explicit settings (see below); render it to reproduce
      explosion_sheet.png     # transparent sprite sheet
      explosion_sheet.json    # atlas: frame rects, times, fps, loop, pivot, trim offsets
      explosion.webp          # animated preview, transparent
      frames/explosion_000.png …
    v002/ …
    latest -> v002
```

- **Explicit YAML:**
  - All vars are substituted, and every atom parameter is written out, defaults included.
  - Per-layer seeds are written out.
  - Referenced scenes are inlined, so later edits to them can't change what this file reproduces. The original reference is kept as a comment.
  - A `meta:` block records the vfxgen version, source file, a hash of the source, render time and the check results.
  - Every parameter carries its doc as a comment.
  - **Re-rendering that file produces byte-identical frames.** This is tested.
- **Versioning:** each render is a new `vNNN`. `watch` re-renders into the same version so you aren't flooded with versions; `--overwrite` does the same from the CLI.
- **Formats** are controlled by `--formats` (default `sheet,atlas,frames,webp`; also `gif`, `apng`).
- **Export options:**
  - `--trim`: crop to the union of non-empty pixels and record the offset and pivot.
  - `--alpha straight|premultiplied|additive` (additive = glow on black).
  - `--pad`, `--pot` (power-of-two sheet), `--columns`.
- Sprites are always transparent RGBA. Straight alpha is the default.
- **No HTML gallery generation.** The current `out/` folder and its `gallery.html` are left untouched as the legacy reference.

---

## 7. Python SDK

```python
import vfxgen as vfx

scene = vfx.load("scenes/storm_strike.yaml", vars={"power": 1.4})
result = vfx.render(scene, draft=False, workers=None)  # -> RenderResult
result.frames            # list[np.ndarray] HxWx4 uint8, straight alpha
result.fps, result.size, result.loop
result.resolved          # Scene (fully explicit)
gen = result.save("gallery")          # -> Generation (paths, version)

vfx.render(vfx.effect("explosion", radius=1.2, colors="#4af")).save("gallery")

s = vfx.Scene(size=(512, 512), fps=24, duration=1.2)
s.add("smoke", at=(256, 400)).add("bolt", at=(256, 0), anchor="top", start=0.3)
vfx.render(s).sheet()    # -> (np.ndarray, atlas dict)

vfx.list_atoms()         # [AtomInfo(name, doc, params schema, roles, anchors, kind)]
vfx.list_effects()       # library scenes + their vars
vfx.describe("explosion")
vfx.json_schema()        # scene JSON Schema (for FastAPI / editors)
vfx.check(result)        # CheckReport: loop seam, edge clipping, empty end, empty frames
```

- **Typed models:** pydantic models for `Scene`, `Layer`, `Channel`, paths, repeat, spawn and output options. YAML and JSON load into these same models, which is what makes the FastAPI service a thin wrapper.
- **Errors carry the spec location:**
  - `SpecError` (validation), e.g. `layers[2].repeat.count: must be ≥ 1`
  - `ResolveError` (missing ref, cycle, depth)
  - `LimitError`
  - `RenderError`
- **No global mutable state** besides the registries and the cache. `render` is a pure function of (scene, vars, seed, options).
- **Future FastAPI (not built now):** `POST /render` (scene JSON or YAML → zip or sheet + atlas), `GET /atoms`, `GET /effects`, `GET /schema`. This is a thin `vfxgen[api]` extra on top of the SDK.

---

## 8. CLI

```
vfxgen render  <scene.yaml | effect | atom> …  [options]
vfxgen watch   <scene.yaml> [render options]       # re-render on save
vfxgen list    [atoms|effects|palettes]
vfxgen describe <name>                              # params/vars, docs, defaults, anchors, roles
vfxgen new     atom <name> | scene <name> [--from explosion]
vfxgen check   <scene.yaml | gallery/x/vNNN>
vfxgen schema  [-o vfxgen.schema.json]
vfxgen export  <gallery/x/vNNN> --trim --alpha premultiplied --fps 12 --size 128x128
vfxgen cache   clear | info
```

Render options:

| Option | What it does |
|---|---|
| `--set name=value` | Sets a var, or an atom param when rendering an atom directly. Repeatable. Values are typed by the target's schema. |
| `--set layers.bolt.params.branches=8` | Dotted path into any field of the scene. |
| `--colors SPEC` | Main colour ramp. |
| `--seed N`, `--variants N` | Seeds N..N+k. |
| `--sweep name=a,b,c` | Renders the grid of combinations and also writes a comparison sheet. Repeatable. |
| `--draft` | Half resolution, half fps, no supersampling: about 8× faster. |
| `--fps`, `--duration`, `--size` | Override the scene's timing and canvas. |
| `--out DIR`, `--name NAME`, `--overwrite` | Output location and naming. |
| `--formats`, `--trim`, `--alpha` | Output formats and export options. |
| `--workers N` | Parallel frame rendering; default cores − 1. |
| `--no-cache` | Skip the cache. |

**Watch mode:**
- Watches the scene, every scene it references, and atom source files (including user atoms on `VFXGEN_PATH`).
- Each render runs in a fresh worker process, so Python edits apply without restarting.
- Draft quality by default; `--final` for full quality.
- Prints timing and check results, and rewrites `gallery/<name>/latest`.

`vfxgen new scene my_fx --from explosion` copies a library scene into an editable file. This is the starting point for iterating.

---

## 9. Performance

- **Parallel:** frames are independent, so they render in worker subprocesses (`python -m vfxgen._worker`). Each worker sets up instances once and renders a chunk of frames. Calling scripts need no `if __name__ == "__main__":` guard.
- **Culling:** atoms evaluate only inside their transformed bounds. Inactive layers, and instances outside the canvas, are skipped.
- **Cache:**
  - **Render cache:** key = hash of (explicit scene, draft flag, version, source of every atom module) → frames in `~/.cache/vfxgen` (LRU, 2 GB; `VFXGEN_CACHE`, `VFXGEN_CACHE_MB`). An identical re-render is a disk read. Editing any atom's code invalidates it.
  - The per-layer cache from the draft was **not built**; see §14.
- **Draft mode:** as above.

---

## 10. Testing

`pytest` with `pytest-xdist`. The fast suite targets under 2 minutes; `-m slow` runs full-quality renders.

| Area | Tests |
|---|---|
| Spec | Loading YAML/JSON into the models; every validation error carries its path; defaults; schema export |
| Expressions | Arithmetic and functions; type preservation; **rejection of unsafe constructs** (attributes, calls, names) |
| Resolution | Atom / library / path resolution order; relative paths; depth-3 limit; cycle detection with the chain; ambiguity errors |
| Time | `once`/`loop`/`hold`/`pingpong`; speed, start, length, fade; `time_warp`; loop seam on frame N; tick_rate determinism and fps-independence |
| Transforms | A probe atom with an asymmetric shape checks anchor placement, rotate direction, scale, flip, `move` and `face_motion`, and `attach` |
| Channels | Blend math (over/add/screen, premultiplied); summing before the tone-map; namespacing of nested scenes; auto-created channels for palette conflicts; `lit_by`, `glow_from` |
| Repeat / spawn | Instance counts, placements along each path type, jitter bounds, ramps, `every` offsets, spawn generations and scale decay, depth limit, instance limit |
| Determinism | Same spec ⇒ identical bytes; editing layer A doesn't change layer B; explicit YAML re-render is byte-identical |
| Atom contract | Parametrized over **every** atom, at draft size: renders at t = 0, mid, end; RGBA; transparent border; one-shot empty at end; loop seam; deterministic; every param documented; palette params accept hex |
| Library | Every library effect renders in draft; vars documented; recolour works; check passes |
| Output | Folder layout and versioning, atlas contents, trim offsets, alpha modes, formats |
| CLI | Each command via `main(argv)`: render, `--set`, sweep, variants, describe, new, check, schema, export; error messages |
| Watch | Saving a temp scene triggers one (debounced) re-render; a referenced-file change triggers it too |
| Cache | Hit/miss behaviour; invalidation on atom source change; LRU eviction |

---

## 11. Package layout

```
pyproject.toml              # vfxgen 1.0.0, Python ≥3.15, console script `vfxgen`
README.md                   # quickstart, CLI, SDK, scene format, writing atoms
SPEC.md                     # this document
src/vfxgen/
  __init__.py               # public SDK surface
  spec/      models.py  expr.py  loader.py  resolve.py  schema.py
  engine/    canvas.py  channels.py  noise.py  color.py  paths.py  timeline.py
             transform.py  compositor.py  particles.py  masks.py  cache.py  parallel.py
  atom.py                   # Atom base, @atom, P, role types, registry
  atoms/     basic.py (glow, ring, ticks, star_shape, star_burst, glints, speed_lines, masks)
             particles.py  fire.py  smoke.py  energy.py  arcane.py  combat.py  elemental.py
  library/   *.yaml         # the 30 effects
  output/    writer.py  atlas.py  export.py  formats.py
  check.py   cli.py  watch.py  scaffold.py
tests/       unit/  atoms/  library/  cli/  conftest.py
scripts/     compare_legacy.py
legacy/      vfxgen_legacy/  # today's code, frozen for the fidelity comparison
```

Dependencies: numpy, pillow, pydantic, ruamel.yaml (comment-preserving, writes the documented YAML), watchfiles. Dev: pytest, pytest-xdist. Future extra `api`: fastapi, uvicorn.

---

## 12. Build plan

1. **Engine (me):**
   - spec models, loader, expressions, resolution
   - canvas with exact transforms, the four channel kinds, compositor (timing, transforms, repeat, spawn, nesting, attach, masks)
   - atom API and particle system
   - output writer, CLI core
   - three probe atoms and the engine test suite
2. **Port atoms + library (parallel, three builders):**
   - fire/smoke, energy/arcane, combat/elemental
   - each ports its atoms to the new API with the contract tests passing, and writes its library YAMLs with the same tweakable vars
   - the engine is frozen while they work
3. **Iteration features (me):** watch, cache, parallel, draft, sweep, check, export/trim, `new`, schema, docs.
4. **Verification (me):**
   - full test suite
   - fidelity sheet of old vs new for all 30 effects
   - re-render of every explicit YAML to confirm byte-identical output
   - fix any gaps

## 13. Decisions taken (flag any you disagree with)

1. New renders go to `./gallery/`. `out/` is left untouched as the legacy reference.
2. Rebuilt effects match closely, not pixel-for-pixel (§5.4).
3. Smoke can't hide light within one scene; light channels draw over smoke channels by default. Channel order is fully controllable per scene, but there is no per-pixel depth.
4. A nested scene keeps its own channels (namespaced), so its look is preserved. Its glow can light parent smoke only via an explicit `lit_by`.
5. Old `--palette` and `--colors` and the effect params keep working: library effects expose the same names as vars.

---

## 14. Differences from the draft (as built)

**Added during the build:**
- **Exact placement for every layer** (§2.1.1): `anchor` accepts `{px: [x, y]}`, `{uv: [u, v]}`, `[u, v]` or a name; `offset` nudges; a scene file can declare its own `anchors:`.
- **Named random streams and per-stream seeds** (§3.1).
- **Time expressions can use scene vars**, e.g. `opacity: "${1 + pulse * sin(tau * t)}"`.
- **A top-level `defs:` block** holds YAML anchors (`&name`) for reuse across layers.
- **`attach.rotate`**, and **`time_warp` `offset` / `span`**.
- **A `profile: lobe` option on sweep masks**: a symmetric highlight instead of a trailing wedge. The `inner` radius fades the sweep toward the centre.
- **One-shot frame timing:** a one-shot always gets a frame exactly at `duration`, even when `duration × fps` is fractional or in draft mode.
- **`View.multiply(fn, lowres=...)`** for cheap multiplicative post-processing of an atom's own deposits. `View.map` is full-resolution only.

**Changed:**
- **SDK listing functions are `vfx.list_atoms()` and `vfx.list_effects()`.** The draft's `vfx.atoms()` clashed with the `vfxgen.atoms` subpackage.
- **Layer cache dropped:** only the whole-render cache was built. A per-layer cache is unsafe with painter's-order solid channels and per-channel bloom, and full-quality library renders already take 0.6–3.5 s with parallel workers.
- **Solid channels have no light setting.** Facet lighting is computed inside the atoms in their local frame, so a rotated or flipped solid layer keeps its baked lighting.

**Known limits:**
- **An atom is either a loop or a one-shot.** One-shot variants are siblings (`smoke_puffs` / `smoke_burst`). Loop atoms can be played once with `time_mode: once`.
- **Particle physics:** gravity in the closed-form particle emitter is coupled to drag (terminal velocity). Legacy effects that used uncoupled gravity were re-tuned in YAML.
- **`shockwave` / `ring` / `flash` / `glow` draw light only.** Density rings use `ripple` (volume).

---

## 15. v1.1 additions

Details and examples: `docs/AUTHORING.md` (authoring) and `docs/CONTRACT.md` (integration).

- **Python 3.12+, managed with uv.** Runtime dependencies: numpy, Pillow, pydantic, ruamel.yaml;
  `watchfiles` is an optional extra.
- **Integration API:** `validate()`, `render_frame()` (never starts a subprocess), `frames=` /
  `times=`, `on_progress`, `cancel` (`RenderCancelled`). Results carry `blend`, `pivot`, `events`,
  `tracks` and `aux` maps.
- **Scene fields:** `blend`, `pivot`, `events`, `track`, `tile`, `motion_blur`, `pixel`, `theme`,
  `rotate`, `ends_visible`. Render options of the same names are written into the explicit spec.
- **Layer fields:** `depth` (draw-order slices across channels), `feel` (hitstop, anticipation,
  squash, punch, shake), `events`, `targets` / `follow`.
- **Channels:**
  - any additive output can go to any additive channel kind;
  - `lit_by` takes `mode: tint | surface` and works on smoke, volume and solid channels;
  - `palette_keys` gives colour over time;
  - new kind `distort` produces a distortion map.
- **Drawing:**
  - image assets (`Atom.asset`, `SolidView.image` with tint / flash / mask / luma tint, `View.stamp`);
  - `SolidView.paint_field` for per-pixel colour;
  - solid layers fade as a group.
- **Motion:**
  - swirling flow (`CurlField`, `curl` on particles and smoke);
  - motion blur;
  - seamless tiling (wrap-around drawing and shading);
  - pixel-art render path;
  - 8- or 16-direction variants;
  - colour themes.
- **Quality:** look-regression snapshots for every library effect (`tests/regression`).
- **Not in scope** (owned by the Workbench): REST service, UI, Unity import, export manifest,
  sheet building, asset store.
