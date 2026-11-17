# Authoring atoms and library effects (v1.1 engine)

Read `docs/PORTING.md` first: it covers the atom API, views, channels, time rules, randomness
and library-scene conventions. This page adds the v1.1 engine features. Run everything with
`uv run` from the project root (`uv run python ...`, `uv run pytest ...`, `uv run vfxgen ...`).

## New engine capabilities

### Your own sprites / textures (image assets)
* `self.asset("path.png")` (in `setup`) loads an image param as **premultiplied** float RGBA
  (h, w, 4). Paths resolve relative to the scene file, then `VFXGEN_PATH`, then
  `src/vfxgen/library/assets/` (built-in sample assets live there; library effects must only use
  assets from that folder). Name image params like `image`, `sprite`, `texture` (strings).
* Drawing:
  * `ctx.solid.image(img, x, y, w=None, h=None, rot=0, a=1, anchor=(0.5, 0.5), tint=(r,g,b),
    tint_amount=0..1, flash=0..1, mask=fn(u, v) -> 0..1)`: paints the image in painter's order into
    a **solid** role (u, v = image UV for dissolves / wipes). Size in local units (default:
    image pixels).
  * `ctx.light.stamp(img_or_2d, x, y, w=None, h=None, rot=0, a=1, anchor=..., channel="alpha"|"luma"|i)`:
    deposits a greyscale image (alpha, edge mask, luminance) into a light / smoke / volume role
    (sprite glows, silhouette-shaped smoke).
* Helpers in `vfxgen.engine.assets`: `bilinear(img, px, py)`, `edge_mask(alpha, width)`,
  `glow_mask(alpha, radius)`, `sample_points(weight, n, rng)` (points proportional to a weight
  map: emit from silhouette interior or edges).

### Links between layers: targets and follow
* A layer may declare `targets: {a: {layer: hand, anchor: tip}, b: {...}}` and/or
  `follow: {layer: blade, anchor: head}`. Linked layers must appear **earlier** in the scene.
* In `draw`: `ctx.target("a", dt=0.0)` / `ctx.follow(dt=0.0)` give the linked anchor's position in
  **this atom's local coordinates** `dt` seconds ago, or `None` when that layer was inactive.
  Sample several `dt`s to build a trail (ribbons) or use two targets for a chain / beam.
  Place such atoms with `at`/`anchor: {px: [0, 0]}` at the scene origin or anywhere: positions
  come back already converted to local space.
* Atoms can expose moving anchors: `anchors = {"head": lambda self, t: (x, y)}`.

### Tileable output
* Scene `tile: true`: drawing wraps around the frame edges, blur and shading wrap, no edge fade.
  For a seamless tile your fields must also be periodic in space: sample `Noise` with a period
  that divides the tile (`Noise(rng, period=(px, py, pz))` and coordinates `x / (W / px)`), and
  place the atom with identity transform (`at: [0, 0]`, `anchor: {px: [0, 0]}`). Atoms that are
  tile-only should say so in their doc and take the tile size as params (`width`, `height`).
  Use `ctx.light.field(fn, (0, 0, W, H))` for full-tile fields.

### Swirling flow
* `vfxgen.engine.flow.CurlField(rng, scale, strength, speed, period)` — divergence-free swirl.
  `velocity(x, y, t)`; `displacement(base_fn, t_spawn, age)` integrates offsets for particles.
* The `particles` / `burst` atoms have `curl`, `curl_scale`, `curl_speed` params (0 = off).
  `EmitterConfig(curl=...)` in your own emitters. Loops stay seamless (time axis periodic).

### Channels: depth, cross-routing, lighting, colour over time, distortion
* Layer `depth: N` draws into a depth slice of its channels (higher = in front), e.g. smoke
  `depth: 1` passes in front of flames.
* Additive outputs route into any additive channel kind (light / smoke / volume / distort): e.g.
  `channels: {light: water}` sends a `shockwave` into a volume channel as a ripple.
* `lit_by: {channels: [light], amount, mode: tint|surface, radius}` on smoke / volume / solid
  channels. `surface` lights the side facing the light (needs normals: volume, or smoke with
  `relief > 0`).
* Channel `palette_keys: [[t_seconds, ramp], ...]` = colour over time (scene time).
* Channel kind `distort` (+ `blur`, `strength`): not drawn; becomes the distortion map
  (`maps=["distortion"]`) for heat haze / shockwave refraction. Route a light role into it.

### Timing, feel, events, tracks
* Layer `feel:` presets (pixels only): `hitstop: [{at, hold}]`, `anticipation: {duration, amount,
  offset}`, `squash: [{at, duration, amount}]` (+ `squash_axis`), `punch: [...]`, `shake: [...]`.
* Gameplay events: scene `events: [{name, t, data}]`, layer `events: [{name, t (local), data}]`.
  Add them where a game needs a moment: projectile `impact`, `spawn_projectile`, `hit`, `pop`,
  `land`, `footstep`, ... Name them in snake_case; put useful numbers in `data`.
* Tracked points: scene `track: [{name, layer, anchor}]` -> per-frame positions in
  `RenderResult.tracks` (projectile head, beam end, coin position...).
* Scene `pivot:` (default centre) — set it to the effect's natural attach point (feet for
  status/aura effects sitting on a character, muzzle/head for projectiles, impact point...).
* Scene `blend: alpha | additive` — how the effect is meant to be blended in-game. Use
  `additive` for pure light effects (glows, sparks, energy) and `alpha` for anything with smoke,
  dust, liquids, solids or dark colours.

### Rendering options (for checking your work)
`vfx.render(name, maps=[...], motion_blur=N, rotate=deg, pixel={size: 4}, theme={kind: ramp})`,
`vfx.render_directions(name, 8)`, `vfx.render_themes(name, {...})`, `vfx.render_frame(name, t)`.
CLI: `--maps`, `--motion-blur`, `--rotate`, `--directions`, `--pixel`, `--theme`.

## Library rules (v1.1)
* Every library scene: `description`, a `colors` var (main ramp; prefer hex defaults or the
  built-in presets), documented vars with min/max, sensible `pivot` and `blend`, events where a
  game would care, `track` entries for moving heads.
* Name channels by kind (`light`, `smoke`, `volume`, `solid`, `distort`) or descriptively
  (`gas`, `water`, `mist`) so themes (`{kind or name: ramp}`) recolour them.
* No effect may reference files outside the package; assets only from `library/assets/`.
* Quality bar: `uv run vfxgen check <name>` clean; looks good on dark and light backgrounds
  (`scripts/preview.py`); loops seamless; one-shots end empty; nothing clipped at the edges.
