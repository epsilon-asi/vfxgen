"""
CLI:  python -m vfxgen [effects...] [--seed N] [--variants N] [--colors SPEC]
                       [--set name=value ...] [--name TAG] [--scale S]
                       [--tier minor|major] [--list] [--params NAME] [--out DIR]

Examples
  python -m vfxgen --list                          what effects exist
  python -m vfxgen --params fire                   every tweakable setting of `fire`
  python -m vfxgen fire --colors "#2a7fff"         blue fire (one colour -> auto ramp)
  python -m vfxgen fire --colors "#102,#60f,#f6f,#fff" --name void_fire
  python -m vfxgen explosion --set smoke_palette=steam --set debris=0
  python -m vfxgen lightning --variants 6          six different bolts

Writes per effect:
  out/<name>/<name>_NN.png     transparent frames
  out/<name>_sheet.png         transparent sprite sheet (8 columns)
  out/<name>/meta.json         frame size, fps, loop flag, description
  out/preview/<name>.gif       preview on dark (not a sprite)
  out/preview/<name>.webp      animated preview WITH alpha (used by the gallery)
and always rebuilds:
  out/gallery.html             every effect in out/, animated, background switcher
  out/preview/contact.png      one key frame of every effect on dark
"""
import argparse
import base64
import html
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import EFFECTS, PALETTES
from .core import get_palette


def parse_value(key, raw, default):
    """Convert a --set string to the type of the parameter's default."""
    if isinstance(default, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{key}: expected true/false, got {raw!r}")
    if isinstance(default, int):
        return int(float(raw))
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, (tuple, list)):
        vals = [float(v) for v in raw.replace("(", "").replace(")", "").split(",")]
        return tuple(type(d)(v) for d, v in zip(default, vals)) if len(vals) == len(default) else tuple(vals)
    if key.endswith("palette"):
        get_palette(raw)  # validate early
    return raw


def print_params(name):
    e = EFFECTS[name]
    print(f"{name}  ({e['tier']}, {'loop' if e['loop'] else 'one-shot'})  {e['desc']}")
    print(f"  colors / palette = {e['palette']!r}   (--colors: preset or hex list)")
    for k, v in e["params"].items():
        doc = e["param_docs"].get(k, "")
        print(f"  {k:18s} = {v!r:14s} {doc}")


def checker(w, h, cell=16):
    yy, xx = np.mgrid[0:h, 0:w]
    v = np.where(((xx // cell) + (yy // cell)) % 2 == 0, 95, 125).astype(np.uint8)
    return np.dstack([v, v, v])


def over(rgba, bg):
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + bg * (1 - a)).astype(np.uint8)


def save(tag, frames, fps, info, palette, seed, outdir, params=None):
    d = outdir / tag
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob(f"{tag}_*.png"):
        old.unlink()
    for i, fr in enumerate(frames):
        Image.fromarray(fr, "RGBA").save(d / f"{tag}_{i:02d}.png", optimize=True)

    n = len(frames)
    cols = min(n, 8)
    rows = math.ceil(n / cols)
    fh, fw = frames[0].shape[:2]
    sheet = np.zeros((rows * fh, cols * fw, 4), np.uint8)
    for i, fr in enumerate(frames):
        r, c = divmod(i, cols)
        sheet[r * fh:(r + 1) * fh, c * fw:(c + 1) * fw] = fr
    Image.fromarray(sheet, "RGBA").save(outdir / f"{tag}_sheet.png", optimize=True)

    (d / "meta.json").write_text(json.dumps(dict(
        name=tag, effect=info["name"], tier=info["tier"], desc=info["desc"],
        loop=info["loop"], palette=palette, seed=seed, params=params or {}, fps=fps, frames=n,
        width=fw, height=fh, sheet=f"{tag}_sheet.png", sheet_cols=cols, sheet_rows=rows,
    ), indent=2))

    pv = outdir / "preview"
    pv.mkdir(exist_ok=True)
    dark = np.full((fh, fw, 3), (18, 20, 28), np.uint8)
    gif = [Image.fromarray(over(fr, dark)) for fr in frames]
    # one-shots hold the empty last frame a moment so the loop reads as repeats
    durs = [int(1000 / fps)] * n
    if not info["loop"]:
        durs[-1] += 500
    gif[0].save(pv / f"{tag}.gif", save_all=True, append_images=gif[1:], duration=durs, loop=0)
    wb = [Image.fromarray(fr, "RGBA") for fr in frames]
    wb[0].save(pv / f"{tag}.webp", save_all=True, append_images=wb[1:], duration=durs,
               loop=0, quality=88, method=4)
    Image.fromarray(over(sheet, checker(sheet.shape[1], sheet.shape[0]))).save(
        pv / f"{tag}_sheet_on_checker.png")


def build_gallery(outdir):
    metas = [json.loads(p.read_text()) for p in sorted(outdir.glob("*/meta.json"))]
    metas = [m for m in metas if (outdir / "preview" / f"{m['name']}.webp").exists()]
    order = {"major": 0, "minor": 1}
    metas.sort(key=lambda m: (order.get(m["tier"], 2), m["name"]))

    # contact sheet: brightest frame of each effect on dark, captioned
    cell = 256
    cols = 6
    rows = math.ceil(len(metas) / cols) or 1
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 18)), (18, 20, 28))
    dr = ImageDraw.Draw(sheet)
    for i, m in enumerate(metas):
        frames = sorted((outdir / m["name"]).glob(f"{m['name']}_*.png"))
        best = max(frames, key=lambda p: np.asarray(Image.open(p))[..., 3].astype(np.int64).sum())
        im = Image.open(best)
        im.thumbnail((cell, cell))
        r, c = divmod(i, cols)
        x, y = c * cell, r * (cell + 18)
        sheet.paste(im, (x + (cell - im.width) // 2, y + 18 + (cell - im.height) // 2), im)
        dr.text((x + 4, y + 3), f"{m['name']} ({m['tier']})", fill=(200, 200, 210))
    sheet.save(outdir / "preview" / "contact.png")

    cards = []
    for m in metas:
        b64 = base64.b64encode((outdir / "preview" / f"{m['name']}.webp").read_bytes()).decode()
        kind = "loop" if m["loop"] else "one-shot"
        cards.append(f"""
<figure class="card" data-tier="{m['tier']}">
  <div class="stage"><img alt="{html.escape(m['name'])}" src="data:image/webp;base64,{b64}"></div>
  <figcaption>
    <div class="row"><b>{html.escape(m['name'])}</b><span class="tag {m['tier']}">{m['tier']}</span></div>
    <p>{html.escape(m['desc'])}</p>
    <small>{m['width']}&times;{m['height']} &middot; {m['frames']} frames @ {m['fps']} fps &middot; {kind} &middot; {html.escape(m['palette'])}</small>
  </figcaption>
</figure>""")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VFX Sprite Gallery</title>
<style>
:root {{ --bg:#101218; --panel:#181b24; --line:#2a2f3c; --text:#e6e8ee; --muted:#9aa1b2;
        --stage:#12141b; --major:#e0a43a; --minor:#5aa9e6; }}
* {{ box-sizing:border-box }}
body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,-apple-system,sans-serif }}
header {{ position:sticky; top:0; z-index:2; background:rgba(16,18,24,.92); backdrop-filter:blur(6px);
         border-bottom:1px solid var(--line); padding:12px 16px; display:flex; flex-wrap:wrap; gap:10px 18px; align-items:center }}
h1 {{ font-size:16px; margin:0; font-weight:650 }}
.group {{ display:flex; gap:6px; flex-wrap:wrap; align-items:center }}
.group span {{ color:var(--muted); font-size:12px; margin-right:2px }}
button {{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px;
         padding:5px 10px; font:inherit; font-size:12px; cursor:pointer }}
button.on {{ border-color:#7c8aff; background:#232845 }}
main {{ padding:16px; display:grid; gap:14px; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)) }}
h2 {{ grid-column:1/-1; margin:10px 0 0; font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted) }}
.card {{ margin:0; background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden }}
.stage {{ height:280px; display:flex; align-items:center; justify-content:center; background:var(--stage) }}
.stage img {{ max-width:100%; max-height:100%; display:block }}
figcaption {{ padding:10px 12px 12px }}
.row {{ display:flex; justify-content:space-between; align-items:center; gap:8px }}
p {{ margin:4px 0 6px; color:#c9cdd8 }}
small {{ color:var(--muted) }}
.tag {{ font-size:11px; padding:1px 7px; border-radius:99px; border:1px solid }}
.tag.major {{ color:var(--major); border-color:var(--major) }}
.tag.minor {{ color:var(--minor); border-color:var(--minor) }}
body[data-bg=light] {{ --stage:#e9ebf0 }}
body[data-bg=grass] {{ --stage:#3c5a2e }}
body[data-bg=sky] {{ --stage:linear-gradient(#3b6fb0,#9cc6ea) }}
body[data-bg=checker] .stage {{ background:repeating-conic-gradient(#5f5f5f 0 25%,#7d7d7d 0 50%) 0 0/24px 24px }}
body[data-size=big] main {{ grid-template-columns:repeat(auto-fill,minmax(380px,1fr)) }}
body[data-size=big] .stage {{ height:440px }}
body[data-tier=major] .card[data-tier=minor], body[data-tier=minor] .card[data-tier=major] {{ display:none }}
body[data-tier=major] h2.minor, body[data-tier=minor] h2.major {{ display:none }}
@media (max-width:520px) {{ main {{ grid-template-columns:1fr }} }}
</style></head>
<body data-bg="dark">
<header>
  <h1>VFX Sprite Gallery <small style="color:var(--muted);font-weight:400">&middot; {len(metas)} effects &middot; transparent PNG sprites</small></h1>
  <div class="group" data-key="bg"><span>Background</span>
    <button class="on" data-v="dark">Dark</button><button data-v="light">Light</button>
    <button data-v="checker">Checker</button><button data-v="grass">Grass</button><button data-v="sky">Sky</button></div>
  <div class="group" data-key="tier"><span>Show</span>
    <button class="on" data-v="all">All</button><button data-v="major">Major</button><button data-v="minor">Minor</button></div>
  <div class="group" data-key="size"><span>Size</span>
    <button class="on" data-v="normal">Normal</button><button data-v="big">Big</button></div>
</header>
<main>
<h2 class="major">Major effects</h2>
{''.join(c for c, m in zip(cards, metas) if m['tier'] == 'major')}
<h2 class="minor">Minor effects</h2>
{''.join(c for c, m in zip(cards, metas) if m['tier'] == 'minor')}
</main>
<script>
document.querySelectorAll('.group').forEach(g => g.addEventListener('click', e => {{
  const b = e.target.closest('button'); if (!b) return;
  g.querySelectorAll('button').forEach(x => x.classList.toggle('on', x === b));
  document.body.dataset[g.dataset.key] = b.dataset.v;
}}));
</script>
</body></html>"""
    (outdir / "gallery.html").write_text(page)
    return len(metas)


def main():
    ap = argparse.ArgumentParser(prog="python -m vfxgen", description="Procedural VFX sprite generator")
    ap.add_argument("effects", nargs="*", help="effect names (default: all)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variants", type=int, default=1, help="render N seeds per effect")
    ap.add_argument("--colors", "--palette", dest="palette", metavar="SPEC",
                    help="main colour ramp: preset name (%s), one hex '#ff3300', or a list '#300,#f30,#fff'"
                    % ", ".join(sorted(PALETTES)))
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="override an effect parameter (repeatable); see --params NAME")
    ap.add_argument("--name", help="output name (default: effect name, plus seed for variants)")
    ap.add_argument("--params", metavar="NAME", nargs="?", const="*",
                    help="show tweakable parameters for an effect (or all)")
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 = 256px base size")
    ap.add_argument("--tier", choices=["minor", "major"])
    ap.add_argument("--list", action="store_true", help="list effects and exit")
    ap.add_argument("--out", default="out")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()

    if args.params:
        for n in (sorted(EFFECTS) if args.params == "*" else [args.params]):
            if n not in EFFECTS:
                ap.error(f"unknown effect {n!r}; try --list")
            print_params(n)
            print()
        return

    if args.list:
        for name, e in sorted(EFFECTS.items(), key=lambda kv: (kv[1]["tier"], kv[0])):
            print(f"{name:16s} {e['tier']:6s} {'loop' if e['loop'] else 'once':5s} {e['palette']:10s} {e['desc']}")
        return

    names = args.effects or sorted(EFFECTS)
    if args.tier:
        names = [n for n in names if EFFECTS[n]["tier"] == args.tier]
    for n in names:
        if n not in EFFECTS:
            ap.error(f"unknown effect {n!r}; try --list")

    if args.palette:
        try:
            get_palette(args.palette)
        except ValueError as e:
            ap.error(str(e))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for name in names:
        info = dict(EFFECTS[name], name=name)
        overrides = {}
        for item in args.set:
            k, sep, raw = item.partition("=")
            k = k.strip()
            if not sep:
                ap.error(f"--set expects NAME=VALUE, got {item!r}")
            if k not in info["params"]:
                if len(names) > 1:
                    continue  # parameter belongs to another effect in this run
                ap.error(f"{name} has no parameter {k!r}. Available: {', '.join(info['params'])}")
            try:
                overrides[k] = parse_value(k, raw.strip(), info["params"][k])
            except ValueError as e:
                ap.error(str(e))
        kwargs = {k: v for k, v in overrides.items() if k != "scale"}
        for v in range(args.variants):
            seed = args.seed + v
            pal = args.palette or info["palette"]
            t0 = time.time()
            frames, fps = info["fn"](np.random.default_rng(seed), pal,
                                     scale=overrides.get("scale", args.scale), **kwargs)
            base = args.name or (name if not (args.palette or overrides) else f"{name}_custom")
            tag = base if args.variants == 1 else f"{base}_s{seed}"
            save(tag, frames, fps, info, pal, seed, outdir, params=overrides)
            h, w = frames[0].shape[:2]
            print(f"{tag:24s} {len(frames):2d} frames @ {w}x{h}, {fps} fps  ({time.time() - t0:.1f}s)")
    if not args.no_gallery:
        n = build_gallery(outdir)
        print(f"gallery: {outdir / 'gallery.html'} ({n} effects)")


if __name__ == "__main__":
    main()
