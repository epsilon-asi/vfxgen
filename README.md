# vfxgen

Procedural VFX sprite generator. Describe an effect as a **YAML scene** — layers of building
blocks placed anywhere, at any time, in any direction, repeated along paths or spawning copies of
themselves — and render transparent sprite sheets, frames and previews. Every render saves a fully
explicit YAML; rendering that file again gives byte-identical frames.

The full file format and design live in [SPEC.md](SPEC.md). This README is the practical guide.

## Install

The project is managed with [uv](https://docs.astral.sh/uv/) (Python ≥ 3.12):

```bash
uv sync                      # creates .venv with the package + dev tools
uv run vfxgen list           # run the CLI
uv run pytest -n 8           # run the tests
```

As a dependency of another project: `uv add "vfxgen @ git+https://github.com/epsilon-asi/vfxgen@v1.1.0"`.
Integrating it into a service or tool? Read [docs/CONTRACT.md](docs/CONTRACT.md).

## What's in the library

91 ready-made effects (`vfxgen list effects`), built from 79 building blocks (`vfxgen list atoms`):

| Category | Effects |
|---|---|
| Fire & smoke | fire, explosion, fireball, smoke, swirling_smoke, poison_cloud, torch_flame, dust_puff, embers, muzzle_flash |
| Magic & energy | lightning, orb, magic_circle, portal, beam, shield, heal_aura, holy_pillar, sparkle, electric_crackle, pickup_burst, charge_up |
| Combat & elemental | whirlwind, sparks, slash, slash_combo, sword_trail, hit_impact, frost_nova, ice_shatter, ground_slam, water_splash |
| Projectile families | fire_bolt, frost_shard, arcane_missile, lightning_orb, each as `_launch` / `_travel` / `_impact` that line up |
| Links | chain_lightning, tether, life_drain |
| Movement | dash, double_jump, landing_dust, footstep_dust, blink |
| Your sprites | hit_flash, dissolve_out, teleport_in, teleport_out, outline_glow, burning_sprite, electrified_sprite, frozen_sprite, coin_fountain, shard_burst |
| Status effects | poison_status, burn_status, freeze_status, stun_stars, sleep_zs, bleed_status, buff_aura, debuff_aura |
| UI | coin_pickup, button_shine, level_up, unlock_burst, damage_pop |
| Impacts per material | impact_metal, impact_wood, impact_stone, impact_blood, impact_energy |
| Environment | rain, snow, fireflies, falling_leaves, light_shafts, dust_motes, waterfall |
| Seamless tiles | fog_tile, water_tile, energy_tile, rain_tile |

Sprite effects take your own image (`--set image=path/to/sprite.png`); sample sprites live in
`src/vfxgen/library/assets/`.

## Quick start

```bash
vfxgen list                                        # building blocks (atoms) and library effects
vfxgen describe explosion                          # an effect's tweakable vars
vfxgen render explosion                            # -> gallery/explosion/v001/
vfxgen render explosion --set radius=1.3 --colors "#4af"
vfxgen render explosion --draft                    # ~8x faster preview quality
vfxgen render gallery/explosion/v001/explosion.yaml   # exact re-render
```

Each render writes a new version:

```
gallery/explosion/v001/
  explosion.yaml          every setting, default and seed, written out (with docs as comments)
  explosion_sheet.png     transparent sprite sheet
  explosion_sheet.json    atlas: frame rects, times, fps, loop, pivot, trim offset
  explosion.webp          animated preview (transparent)
  frames/explosion_000.png ...
gallery/explosion/latest -> v001
```

## Iterating on an effect

```bash
vfxgen new scene my_blast --from explosion   # editable copy (+ vfxgen.schema.json for editor autocomplete)
vfxgen watch my_blast.yaml                   # re-renders (draft) every time you save
```

`watch` also re-renders when a referenced scene file or any atom's Python code changes. Use
`--final` for full quality. Other loops:

```bash
vfxgen render my_blast.yaml --sweep radius=0.8,1.2,1.6 --sweep colors=fire,#4af   # grid of variants
vfxgen render my_blast.yaml --variants 6 --seed 10                                # 6 random layouts
vfxgen render my_blast.yaml --set layers.sparks.params.count=120                   # any field by path
vfxgen render my_blast.yaml --set layers.sparks.seeds.particles=7                  # reshuffle only the sparks
vfxgen check my_blast.yaml                    # loop seam, edge clipping, one-shot ends empty
```

Identical re-renders come from a cache (`vfxgen cache info|clear`, `--no-cache`).

More render options:

```bash
vfxgen render fire_bolt_travel --directions 8                  # 8-way rotated variants
vfxgen render explosion --pixel 4 --pixel-outline "#111"       # pixel-art version
vfxgen render frost_nova --theme "light=#310,#f60,#fc8,#fff"   # recolour by channel kind or name
vfxgen render slash --motion-blur 6                            # motion blur
vfxgen render water_splash --maps normal,emissive,intensity    # extra map sheets for engine shaders
```

## Scenes

```yaml
vfxgen: 1
name: storm_strike
size: [512, 512]
fps: 24
duration: 1.5            # seconds (all timing is in seconds; fps only decides sampling)
seed: 7
vars:                    # your knobs: --set power=2, or `params:` when this scene is used as a layer
  power: {default: 1.0, min: 0, max: 3, doc: overall intensity}
  colors: {default: electric, doc: main colour ramp}
channels:                # optional: shared render buffers (light / smoke / volume / solid), bottom -> top
  smoke: {kind: smoke, palette: dark_smoke, lit_by: {channels: [light], amount: 0.8}}
  light: {kind: light, palette: "${colors}", exposure: "${power}"}
layers:
  - id: charge
    effect: charge_up                 # atom | library effect | path/to/scene.yaml
    at: [256, 120]                    # exact position of the anchor (default: canvas centre)
    anchor: center                    # center | top | bottom | named | {px: [x, y]} | [u, v]
  - id: bolt
    effect: bolt
    at: [256, 0]
    anchor: top
    rotate: 10
    start: 0.45
  - effect: explosion                 # a whole library scene as one layer
    at: [256, 470]
    anchor: bottom
    start: 0.5
    scale: 0.8
    params: {radius: 1.2}             # that scene's vars
  - effect: scenes/impact_ring.yaml   # your own scene files nest too (up to 3 deep)
    start: 0.5
  - effect: dust_puff
    start: 0.55
    repeat:                           # copies over a course
      count: 5
      every: 0.08
      along: {line: [[150, 480], [360, 480]]}
      jitter: {scale: 0.2, rotate: 15, time: 0.03}
  - effect: flash
    at: [400, 120]
    spawn: {count: 4, depth: 2, radius: 60, when: 0.12, scale: 0.5}   # self-spawning copies
```

Scene-level extras: `blend` (alpha / additive), `pivot`, `events` (gameplay moments), `track`
(per-frame positions of anchors), `tile` (seamless tiles), `motion_blur`, `pixel`, `theme`, `rotate`.

Layer fields at a glance: `at`, `anchor`, `offset`, `rotate`, `scale`, `flip`, `opacity`
(constants, keyframes `{keys: [[t, v], [t, v, ease]]}` or time expressions `"${360 * t}"`),
`start`, `length`, `speed`, `time_mode` (once / loop / hold / pingpong), `time_warp`, `fade`,
`move` (path + ease + `face_motion`), `attach` (follow another layer's anchor), `mask`,
`colors`, `palettes`, `adjust`, `channels`, `depth`, `z`, `seed`, `seeds`, `repeat`, `spawn`, `feel`
(hitstop, anticipation, squash, punch, shake), `events`, `targets` / `follow` (links to other
layers for chains and trails), `enabled`.
`${...}` expressions use vars and safe math (`min max clamp lerp sin cos sqrt smoothstep pi ...`).

Colours anywhere: a preset (`vfxgen list palettes`), one hex (`"#2a7fff"`, auto ramp), a list
(`"#300,#f30,#fc6,#fff"`, faint→hot) or explicit stops (`"0:#300,0.4:#f30,1:#fff"`).

## Seeds and exact re-renders

Every random number comes from a seed you can set: the scene `seed`, a layer `seed`, or a single
random part of an atom via `seeds: {stream: n}` (`vfxgen describe <atom>` lists the streams).
The explicit YAML written with each render records them all; rendering it reproduces the frames
byte for byte (tested, also across worker counts and with/without the cache).

## Python SDK

```python
import vfxgen as vfx

r = vfx.render("explosion", {"radius": 1.2}, colors="#4af", draft=False)
r.frames            # list of HxWx4 uint8 arrays (straight alpha)
sheet, atlas = r.sheet(trim=True)
gen = r.save("gallery", formats=["sheet", "atlas", "frames", "webp"], trim=True)

scene = {"vfxgen": 1, "size": [256, 256], "duration": 1.0,
         "layers": [{"effect": "smoke_puffs"}, {"effect": "bolt", "anchor": "top", "at": [128, 0]}]}
vfx.render(scene, overrides={"layers.layer1.params.branches": 8}, seed=3).save()

vfx.validate(scene)                                 # full check, no rendering (~2 ms)
vfx.render_frame("explosion", 0.4, draft=True)      # one frame, in-process (live previews)
vfx.render(scene, frames=[0, 10], on_progress=print, cancel=threading.Event())
vfx.render(scene, maps=["normal"]).aux["normal"]    # extra maps; r.events, r.tracks, r.pivot, r.blend
vfx.render_directions("fire_bolt_travel", 8); vfx.render_themes("orb", {"ice": {"light": "#036,#4cf,#fff"}})

vfx.list_atoms(); vfx.list_effects(); vfx.describe("bolt"); vfx.json_schema()
```

Errors are typed and point at the spec location: `SpecError` (`layers[2].repeat.count: ...`),
`ResolveError` (missing reference, cycle, depth), `LimitError`, `RenderError`. `render()` is a
pure function of its inputs, which makes a FastAPI wrapper a thin layer (`POST /render` with the
scene JSON → sheet + atlas; `GET /atoms`, `/effects`, `/schema`).

## Writing a building block (atom)

```bash
vfxgen new atom my_pop --dir ~/vfx       # creates ~/vfx/atoms/my_pop.py
export VFXGEN_PATH=~/vfx                 # user atoms (and scene files) are found here
vfxgen render my_pop --draft
```

An atom draws itself at local time `t` (seconds) in its own coordinates; the engine handles
placement, rotation, scale, timing and channels. See `docs/PORTING.md` for the API cheat sheet
and `src/vfxgen/atoms/basic.py` for small complete examples. Every atom is checked by
`tests/atoms/test_atom_contract.py` (deterministic, time-pure, loops seamless, one-shots end
empty, params documented).

## Tests

```bash
uv run pytest -n 8                                   # full suite (includes look-regression snapshots)
uv run python scripts/update_snapshots.py [names]    # refresh reference images after an intended look change
uv run python scripts/verify_library.py /tmp/check   # render every effect, prove exact re-renders
```
