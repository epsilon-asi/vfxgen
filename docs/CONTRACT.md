# vfxgen: integration contract (v1.1)

For anyone building on top of vfxgen (the Workbench). vfxgen is a library that hands over data:
it does not own a server, a UI, a Unity importer, a project manifest or an asset store.
Additions are always additive; existing fields don't change.

Install: `uv add "vfxgen @ git+https://github.com/epsilon-asi/vfxgen@<tag>"` (Python ≥ 3.12;
runtime deps: numpy, Pillow, pydantic v2, ruamel.yaml; optional extra `watch` for the CLI watcher).

**What it is:** a Python package (`import vfxgen`) that renders transparent 2D VFX sprite
animations from a **scene spec**. A scene is a list of layers: building blocks ("atoms"), library
effects, or other scenes, each placed, timed and coloured. Rendering is CPU-only, deterministic and
stateless. Full format: `SPEC.md`. Usage guide: `README.md`.

## Inputs

**Scene spec** (YAML or JSON; the same structure as a Python dict). The JSON Schema comes from
`vfxgen.json_schema()`.

```json
{"vfxgen": 1, "size": [512, 512], "fps": 24, "duration": 1.5, "loop": false, "seed": 7,
 "vars": {"power": {"default": 1.0, "min": 0, "max": 3, "doc": "brightness"}},
 "layers": [
   {"id": "boom", "effect": "explosion", "at": [256, 300], "params": {"radius": 1.2}, "colors": "#4af"},
   {"id": "bolt", "effect": "bolt", "at": [256, 0], "anchor": "top", "start": 0.4, "rotate": 10}]}
```

- **`effect`** is an atom name, a library effect name, a path to a `.yaml` file, or an inline scene
  `{"scene": {...}}`.
- **Colours** are a preset name (`"fire"`), one hex (`"#2a7fff"`), a hex list (`"#300,#f30,#fff"`),
  or stops (`"0:#300,1:#fff"`).

**Render call options:**

| Option | Meaning |
|---|---|
| `vars` | Values for the scene's declared vars. For a bare atom name, the atom's params. |
| `overrides` | Dotted-path edits applied before validation, e.g. `{"layers.bolt.params.branches": 8, "layers.bolt.seeds.flicker": 5}`. Layers are addressed by `id` or index. |
| `seed` | Replaces the scene seed. |
| `colors` | Main colour ramp. |
| `draft` | Half size, half fps, about 8× faster. |
| `workers` | Process count. Default is cores − 1. |
| `cache` | Reuse an identical previous full render. Default `True`. Partial renders are never cached. |
| `frames` | Render only these frame indices of the full sequence, e.g. `[0, 12]`. |
| `times` | Render only these scene times in seconds (overrides `frames`). |
| `on_progress` | `fn(done, total)` called as frames finish (serial and parallel). |
| `cancel` | A `threading.Event` (or zero-arg callable). When set, the render stops, worker processes are killed and `RenderCancelled` is raised. |
| `maps` | Extra maps returned in `r.aux`: any of `"normal"`, `"emissive"`, `"intensity"`, `"distortion"`. |
| `motion_blur` | Samples per frame (0 = off; overrides the scene's `motion_blur`). |
| `rotate` | Rotate the whole effect (degrees, clockwise) about the frame centre. |
| `pixel` | Pixel-art output: `{size, palette (hex list), colors, dither, outline, fps}`; `False` disables the scene's own. |
| `theme` | Recolour channels: `{channel name or kind ("light", "smoke", "volume", "solid"): hex ramp}`. |

`motion_blur`, `rotate`, `pixel` and `theme` are written into `r.explicit`, so re-rendering the
explicit spec reproduces them.

## Core calls

```python
r = vfxgen.render(source, vars=None, *, overrides=None, seed=None, colors=None,
                  draft=False, workers=None, cache=True)          # -> RenderResult
```

- `source` can be:
  - a scene dict
  - YAML text
  - a file path
  - a library or atom name
  - a `Scene`, or a `LoadedScene` from `vfxgen.prepare(...)`.
- `vfxgen.prepare(source, vars, overrides=, seed=, colors=)` validates the spec (schema, vars,
  expressions) and returns a `LoadedScene`, whose `.scene` is the typed `Scene`. Median 0.6 ms on
  the library.
- `vfxgen.validate(source, vars, overrides=, seed=, colors=, draft=)` does the **full** check
  without rendering: spec, effect references, every atom's params, and all limits. It raises the
  same `SpecError` / `ResolveError` / `LimitError` a render would, before any pixel work, and
  returns `SceneInfo{name, width, height, fps, duration, loop, frame_count, instances, channels,
  blend, pivot, explicit}`. Median ~2 ms, worst ~90 ms (explosion) on the library. Cheap enough
  to call on every edit.
- `vfxgen.render_frame(source, t, vars, overrides=, seed=, colors=, draft=)` returns one frame
  (H×W×4 uint8) at scene time `t`. It renders **in the calling process: no subprocess is ever
  started**, and nothing is cached. Measured on the library: draft frames median ~50 ms
  (worst ~150 ms, explosion); full-quality frames median ~175 ms (worst ~0.9 s). (`render(..., workers=1)`
  also never starts a subprocess.)
- `vfxgen.render_themes(source, {name: theme}, ...)` and `vfxgen.render_directions(source, 8, ...)`
  return `{name or degrees: RenderResult}`.
- `vfxgen.check.check_result(r)` returns a `CheckReport` with:
  - `.ok`
  - `.issues`, each with `level`, `code` and `message`. Codes are `empty`, `edge_clip`,
    `loop_seam`, `not_empty_at_end` and `blank_frames`.
  - `.stats`

**Discovery** (drives UI forms):
- `vfxgen.list_atoms()` returns one entry per atom: `{name, doc, kind: "oneshot"|"loop", tags,
  bounds, anchors[], outputs{role: channel-spec}, streams[], params{name: {default, doc, type}}}`.
- `vfxgen.list_effects()` returns one entry per library effect: `{name, description, loop, file,
  vars{name: {default, doc, min?, max?, choices?}}}`.
- `vfxgen.describe(name)` returns one atom or effect (`type: "atom"|"effect"`).
- `vfxgen.json_schema()` returns the full scene JSON Schema.

## Outputs

**`RenderResult`** fields:
- `frames`: list of H×W×4 `uint8` arrays, straight alpha, transparent background.
- `fps`, `width`, `height`, `loop`, `duration` (seconds), `name`.
- `times`: the time of each frame in seconds.
- `explicit`: dict of the fully explicit scene, with every default and seed written out.
  **Re-rendering it gives byte-identical frames.**
- `elapsed`, `draft`, `source`.
- `blend`: `"alpha"` or `"additive"`, how the effect was authored to be blended in-game (scene
  field `blend`, default `"alpha"`). Frames are always straight alpha either way.
- `pivot`: `(u, v)` normalised pivot in the frame, 0..1, y down (scene field `pivot`: a name,
  `[u, v]`, `{uv}` or `{px}`; default centre).
- `frame_indices`, `frame_count_total`: which frames of the full sequence this result holds.
- `events`: `[{name, t, frame, layer, data}]` gameplay moments (scene time, nearest frame).
- `tracks`: `{name: [[x, y] or None per frame]}` positions of tracked anchors, in frame pixels.
- `aux`: `{map name: [H×W×4 uint8 per frame]}` for the requested `maps`:
  - `normal`: tangent-space normal map (+y up), alpha = effect alpha;
  - `emissive`: only the light parts of the effect;
  - `intensity`: greyscale brightness (0 = faint, 1 = hottest) for runtime recolouring with a ramp;
  - `distortion`: RG = displacement direction (0.5 = none), A = strength, from `distort` channels.

**`RenderResult` methods:**
- `r.sheet(columns=None, trim=False)` returns `(sheet_rgba_ndarray, atlas_dict)`.
- `r.save(out_dir="gallery", name=None, formats=("sheet","atlas","frames","webp"),
  overwrite=False, trim=False, alpha="straight"|"premultiplied"|"additive", columns=None, pad=0,
  pot=False)` returns a `Generation` with `dir`, `name`, `version` and
  `files{yaml, sheet, atlas, frames, webp, gif?, apng?}`.

**Gallery layout on disk:**

```
<out>/<name>/vNNN/
  <name>.yaml
  <name>_sheet.png
  <name>_sheet.json
  <name>.webp
  frames/<name>_NNN.png
<out>/<name>/latest -> vNNN
```

`<name>.yaml` is the explicit spec, and it re-renders exactly.

**Atlas JSON fields** (CLI / `.save()` only; the Workbench builds its own sheets):
- `frames[{index, x, y, w, h, t}]`, `frame_count`, `fps`, `loop`
- `columns`, `rows`, `frame_size`, `source_size`, `trim_offset`
- `pivot`: normalised, `blend`, `alpha`, `sheet_size`, `image`
- `events`, `tracks` (as on `RenderResult`, tracks shifted by the trim offset)

## Behaviour guarantees

- **Deterministic:** the same spec, vars, overrides and seeds give the same bytes, regardless of
  worker count or cache.
- **Stateless:** no global mutable state besides the atom registry and an on-disk render cache.
  The cache lives at `~/.cache/vfxgen` (or `$VFXGEN_CACHE`); `VFXGEN_NO_CACHE=1` disables it.
- **Cost:** full-quality library effects take about 0.6–3.5 s each, and draft is much faster.
  Rendering is CPU-bound and starts worker subprocesses, so call it from a thread or job, **never
  on the async event loop**.
- **Limits:** 20,000 instances, 4096 px per side, 1,000 frames. Scene nesting and spawn depth are
  each limited to 3. All of them raise `LimitError` while the scene is built, before any
  rendering, so `validate()` reports them immediately.
- **Safety:** `${...}` expressions run in a whitelisted sandbox.
  - File references (`effect: x.yaml`, assets) resolve relative to the scene file, then
    `$VFXGEN_PATH`, then the built-in library.
  - The API should accept inline scenes, or restrict file paths to an allowed folder.
  - No library effect references a file outside the package. Effects that use sample sprites
    load them from the package's own `library/assets/` folder.

## Errors

All errors derive from `vfxgen.VfxError`:

| Error | Raised for |
|---|---|
| `SpecError` | Invalid input. Has `.path` (e.g. `layers[2].repeat.count`) and `.message`; map to 422. |
| `ResolveError` | Unknown effect or file, cycle, or nesting too deep (a subclass of `SpecError`); map to 422 or 404. |
| `LimitError` | A size, frame, instance or depth limit was exceeded (a subclass of `SpecError`); map to 413 or 422. |
| `RenderError` | Failure inside rendering; map to 500. |
| `RenderCancelled` | The render's `cancel` event was set. |

## Not part of vfxgen

The REST service, UI, Unity import, export manifest, sheet building and asset store belong to the
Workbench. vfxgen's `.save()`, gallery folders and CLI remain for command-line use only.
