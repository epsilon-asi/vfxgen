"""Command line interface: ``vfxgen <command> ...`` (see ``vfxgen -h``)."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__
from .errors import VfxError

DEFAULT_OUT = "gallery"


# ---------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------
def _target_info(target: str):
    """('atom' | 'effect' | 'file', declared var names or atom params)."""
    from .atom import ATOMS, ensure_atoms
    from .spec.loader import LIBRARY_DIR, read_yaml
    ensure_atoms()
    p = Path(target)
    if p.suffix in (".yaml", ".yml", ".json") or p.is_file() or "/" in target:
        raw = read_yaml(p.read_text()) if p.is_file() else {}
        return "file", set((raw or {}).get("vars") or {})
    lib = LIBRARY_DIR / f"{target}.yaml"
    if lib.is_file():
        return "effect", set((read_yaml(lib.read_text()) or {}).get("vars") or {})
    if target in ATOMS:
        return "atom", set(ATOMS[target].Params.model_fields)
    raise VfxError(f"{target!r} is not a scene file, library effect or atom (try `vfxgen list`)")


def split_sets(target: str, sets: Sequence[str]) -> tuple[dict, dict]:
    """--set items -> (vars or atom params, dotted overrides)."""
    from .render import _parse_scalar
    kind, names = _target_info(target)
    vars_: dict[str, Any] = {}
    overrides: dict[str, Any] = {}
    for item in sets:
        if "=" not in item:
            raise VfxError(f"--set expects NAME=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if kind == "atom" and k in names:             # an atom's own param wins (e.g. duration)
            vars_[k] = _parse_scalar(v)
        elif "." in k or k in ("fps", "duration", "size", "loop", "edge_fade", "seed"):
            overrides[k] = v
        elif k in names:
            vars_[k] = _parse_scalar(v)
        else:
            what = "parameter" if kind == "atom" else "var"
            raise VfxError(f"{target}: no {what} named {k!r}. Available: {', '.join(sorted(names)) or 'none'}. "
                           "Use a dotted path (e.g. layers.<id>.params.<name>=...) for anything else.")
    return vars_, overrides


def _print_gen(gen, result, quiet: bool) -> None:
    if quiet:
        print(gen.dir)
        return
    print(f"{gen.name} v{gen.version:03d}: {len(result.frames)} frames {result.width}x{result.height} "
          f"@ {result.fps:g} fps ({result.elapsed:.1f}s) -> {gen.dir}")


# ---------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------
def cmd_render(a) -> int:
    from .render import render
    sweeps = []
    for s in a.sweep or []:
        k, _, vals = s.partition("=")
        if not vals:
            raise VfxError(f"--sweep expects NAME=v1,v2,..., got {s!r}")
        sweeps.append([(k.strip(), v) for v in vals.split(",")])
    combos = list(itertools.product(*sweeps)) if sweeps else [()]
    extra_over: dict[str, Any] = {}
    if a.fps is not None:
        extra_over["fps"] = a.fps
    if a.duration is not None:
        extra_over["duration"] = a.duration
    if a.size is not None:
        w, _, h = a.size.lower().partition("x")
        extra_over["size"] = f"[{int(w)}, {int(h or w)}]"
    formats = a.formats.split(",") if a.formats else None
    for target in a.targets:
        for combo in combos:
            sets = list(a.set or []) + [f"{k}={v}" for k, v in combo]
            vars_, overrides = split_sets(target, sets)
            overrides.update(extra_over)
            seeds = [a.seed] if a.variants <= 1 else [(a.seed or 1) + i for i in range(a.variants)]
            theme = None
            if a.theme:
                theme = {}
                for item in a.theme:
                    k, _, v = item.partition("=")
                    if not v:
                        raise VfxError(f"--theme expects CHANNEL_OR_KIND=RAMP, got {item!r}")
                    theme[k.strip()] = v.strip()
            pixel = None
            if a.pixel:
                pixel = {"size": a.pixel}
                if a.pixel_palette:
                    pixel["palette"] = a.pixel_palette
                if a.pixel_outline:
                    pixel["outline"] = a.pixel_outline
            rotations = [None] if not a.directions else [round(360 * i / a.directions, 4) for i in range(a.directions)]
            if a.rotate is not None:
                rotations = [a.rotate]
            for sd, rot in itertools.product(seeds, rotations):
                result = render(target, vars_, overrides=overrides, seed=sd, colors=a.colors,
                                draft=a.draft, workers=a.workers, cache=not a.no_cache,
                                maps=a.maps.split(",") if a.maps else (), motion_blur=a.motion_blur,
                                rotate=rot or None, pixel=pixel, theme=theme)
                name = a.name or result.name
                if a.directions:
                    name += f"_dir{int(round(rot or 0)):03d}"
                if combo:
                    name += "__" + "_".join(f"{k.split('.')[-1]}-{v}" for k, v in combo)
                if a.variants > 1:
                    name += f"_s{sd}"
                kw = {"name": name, "overwrite": a.overwrite, "trim": a.trim, "alpha": a.alpha}
                if formats:
                    kw["formats"] = formats
                gen = result.save(a.out, **kw)
                if a.check:
                    from .check import check_result
                    rep = check_result(result)
                    print(rep.summary())
                _print_gen(gen, result, a.quiet)
    return 0


def cmd_list(a) -> int:
    import vfxgen as vfx
    what = a.what
    if what in ("atoms", "all"):
        print("ATOMS (building blocks)")
        for info in vfx.list_atoms():
            first = info["doc"].splitlines()[0] if info["doc"] else ""
            print(f"  {info['name']:20s} {info['kind']:8s} {first}")
    if what in ("effects", "all"):
        print("LIBRARY EFFECTS")
        for e in vfx.list_effects():
            print(f"  {e['name']:20s} {'loop' if e['loop'] else 'once':8s} {e['description']}")
    if what in ("palettes", "all"):
        from .engine.color import PALETTES
        print("PALETTES")
        print("  " + ", ".join(sorted(PALETTES)))
    return 0


def cmd_describe(a) -> int:
    import vfxgen as vfx
    info = vfx.describe(a.name)
    if a.json:
        print(json.dumps(info, indent=2, default=str))
        return 0
    if info["type"] == "atom":
        print(f"{info['name']}  (atom, {info['kind']})")
        if info["doc"]:
            print("  " + info["doc"].replace("\n", "\n  "))
        print(f"  bounds: {info['bounds']}   anchors: {', '.join(info['anchors'])}")
        print(f"  outputs: " + ", ".join(f"{r} -> {s['kind']}" for r, s in info["outputs"].items()))
        print(f"  random streams (seedable): {', '.join(info['streams']) or 'none'}")
        print("  params:")
        for k, f in info["params"].items():
            print(f"    {k:20s} = {f['default']!r:<14} {f['doc']}")
    else:
        print(f"{info['name']}  (library effect, {'loop' if info['loop'] else 'one-shot'})")
        if info["description"]:
            print("  " + info["description"])
        print("  vars:")
        for k, v in info["vars"].items():
            print(f"    {k:20s} = {v.get('default')!r:<14} {v.get('doc', '')}")
        print(f"  file: {info['file']}")
    return 0


def cmd_schema(a) -> int:
    import vfxgen as vfx
    text = json.dumps(vfx.json_schema(), indent=1)
    if a.output:
        Path(a.output).write_text(text)
        print(a.output)
    else:
        print(text)
    return 0


def cmd_check(a) -> int:
    from pathlib import Path as _P
    from .check import check_path
    vars_, overrides = ({}, {}) if _P(a.target).is_dir() else split_sets(a.target, a.set or [])
    rep = check_path(a.target, draft=a.draft, vars=vars_, overrides=overrides, colors=a.colors)
    print(rep.summary())
    return 0 if rep.ok else 1


def cmd_watch(a) -> int:
    from .watch import watch
    return watch(a)


def cmd_new(a) -> int:
    from .scaffold import new_atom, new_scene
    p = new_atom(a.name, a.dir) if a.kind == "atom" else new_scene(a.name, a.dir, a.from_)
    print(p)
    return 0


def cmd_export(a) -> int:
    from .output.export import export
    out = export(a.generation, out=a.out, trim=a.trim, alpha=a.alpha, fps=a.fps, size=a.size,
                 columns=a.columns, pad=a.pad, pot=a.pot)
    print(out)
    return 0


def cmd_cache(a) -> int:
    from .engine import cache
    if a.action == "clear":
        n = cache.clear()
        print(f"removed {n} cache entries")
    else:
        print(json.dumps(cache.info(), indent=1))
    return 0


# ---------------------------------------------------------------------------------
def _render_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument("--set", action="append", metavar="NAME=VALUE",
                   help="set a var (scenes/effects), a param (atoms), or a dotted path "
                        "(layers.<id>.params.<name>=..., layers.<id>.seeds.<stream>=...). Repeatable.")
    p.add_argument("--colors", "--palette", dest="colors", metavar="SPEC",
                   help="main colours: preset, '#hex', or '#a,#b,#c'")
    p.add_argument("--seed", type=int, help="scene seed (all randomness derives from it)")
    p.add_argument("--variants", type=int, default=1, help="render N seeds: seed, seed+1, ...")
    p.add_argument("--sweep", action="append", metavar="NAME=a,b,c",
                   help="render every combination of these values (repeatable)")
    p.add_argument("--draft", action="store_true", help="half resolution and fps, no supersampling (~8x faster)")
    p.add_argument("--fps", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--size", help="WxH")
    p.add_argument("--out", default=DEFAULT_OUT, help=f"gallery folder (default: {DEFAULT_OUT})")
    p.add_argument("--name", help="output name")
    p.add_argument("--overwrite", action="store_true", help="replace the latest version instead of adding one")
    p.add_argument("--formats", help="comma list: sheet,atlas,frames,webp,gif,apng")
    p.add_argument("--trim", action="store_true", help="crop to the non-transparent area")
    p.add_argument("--alpha", default="straight", choices=["straight", "premultiplied", "additive"])
    p.add_argument("--workers", type=int, help="processes (default: cores - 1)")
    p.add_argument("--check", action="store_true", help="also run quality checks")
    p.add_argument("--no-cache", action="store_true", help="always re-render (skip the render cache)")
    p.add_argument("--maps", help="extra maps: normal,emissive,intensity,distortion (saved as extra sheets)")
    p.add_argument("--motion-blur", type=int, metavar="SAMPLES", help="motion blur samples per frame (0 = off)")
    p.add_argument("--rotate", type=float, help="rotate the whole effect (degrees, clockwise)")
    p.add_argument("--directions", type=int, metavar="N", help="render N evenly spaced rotations (8, 16)")
    p.add_argument("--pixel", type=int, metavar="SIZE", help="pixel-art mode: scene pixels per art pixel")
    p.add_argument("--pixel-palette", help="pixel-art palette: hex list ('#000,#f80,#fff'); default automatic")
    p.add_argument("--pixel-outline", help="pixel-art 1 px outline colour (hex)")
    p.add_argument("--theme", action="append", metavar="CHANNEL=RAMP",
                   help="recolour a channel (by name or kind: light/smoke/volume/solid) with a hex ramp; repeatable")
    p.add_argument("-q", "--quiet", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vfxgen", description="Procedural VFX sprite generator")
    ap.add_argument("--version", action="version", version=f"vfxgen {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("render", help="render scenes / effects / atoms to the gallery")
    p.add_argument("targets", nargs="+", help="scene .yaml, library effect name, or atom name")
    _render_opts(p)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser("watch", help="re-render a scene every time it (or its references / atom code) changes")
    p.add_argument("targets", nargs=1)
    _render_opts(p)
    p.add_argument("--final", action="store_true", help="full quality instead of draft")
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("list", help="list atoms, library effects and palettes")
    p.add_argument("what", nargs="?", default="all", choices=["all", "atoms", "effects", "palettes"])
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("describe", help="parameters / vars of an atom or effect")
    p.add_argument("name")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_describe)

    p = sub.add_parser("check", help="quality checks on a scene or a generation folder")
    p.add_argument("target")
    p.add_argument("--draft", action="store_true")
    p.add_argument("--set", action="append", metavar="NAME=VALUE", help="as for render")
    p.add_argument("--colors", metavar="SPEC")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("schema", help="print the scene JSON Schema")
    p.add_argument("-o", "--output")
    p.set_defaults(fn=cmd_schema)

    p = sub.add_parser("new", help="scaffold an atom or a scene")
    p.add_argument("kind", choices=["atom", "scene"])
    p.add_argument("name")
    p.add_argument("--from", dest="from_", help="scene: start from this library effect / scene")
    p.add_argument("--dir", default=".", help="where to create it")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("export", help="re-export a generation (trim, alpha mode, fps, size)")
    p.add_argument("generation", help="gallery/<name>/vNNN (or .../latest)")
    p.add_argument("--out", help="output folder (default: <generation>/export)")
    p.add_argument("--trim", action="store_true")
    p.add_argument("--alpha", default="straight", choices=["straight", "premultiplied", "additive"])
    p.add_argument("--fps", type=float, help="resample to this fps")
    p.add_argument("--size", help="scale frames to WxH")
    p.add_argument("--columns", type=int)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--pot", action="store_true", help="power-of-two sheet")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("cache", help="layer cache maintenance")
    p.add_argument("action", choices=["info", "clear"])
    p.set_defaults(fn=cmd_cache)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return int(a.fn(a) or 0)
    except VfxError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
