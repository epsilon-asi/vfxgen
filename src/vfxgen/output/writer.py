"""Writing generations to the gallery folder.

::

    gallery/<name>/vNNN/<name>.yaml        fully explicit settings (render it to reproduce)
                        <name>_sheet.png   transparent sprite sheet
                        <name>_sheet.json  atlas
                        <name>.webp        animated preview (transparent)
                        frames/<name>_NNN.png
    gallery/<name>/latest -> vNNN
"""
from __future__ import annotations

import datetime as _dt
import io
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .. import __version__
from .atlas import apply_alpha, build_sheet

DEFAULT_FORMATS = ("sheet", "atlas", "frames", "webp")
ALL_FORMATS = ("sheet", "atlas", "frames", "webp", "gif", "apng")


@dataclass
class Generation:
    dir: Path
    name: str
    version: int
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def yaml(self) -> Path:
        return self.files["yaml"]


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", name).strip("._") or "scene"


def next_version_dir(root: Path, overwrite: bool) -> tuple[Path, int]:
    root.mkdir(parents=True, exist_ok=True)
    nums = sorted(int(p.name[1:]) for p in root.glob("v[0-9][0-9][0-9]*") if p.is_dir() and p.name[1:].isdigit())
    if overwrite and nums:
        v = nums[-1]
        d = root / f"v{v:03d}"
        shutil.rmtree(d)
    else:
        v = (nums[-1] + 1) if nums else 1
        d = root / f"v{v:03d}"
    d.mkdir(parents=True)
    return d, v


def _commented(data: Any) -> Any:
    """Plain data -> ruamel Commented* with atom parameter docs as end-of-line comments."""
    from ..atom import ATOMS
    if isinstance(data, dict):
        cm = CommentedMap()
        for k, v in data.items():
            cm[k] = _commented(v)
        eff = data.get("effect")
        params = cm.get("params")
        if isinstance(eff, str) and eff in ATOMS and isinstance(params, CommentedMap):
            fields = ATOMS[eff].Params.model_fields
            for k in params:
                doc = fields[k].description if k in fields else None
                if doc:
                    params.yaml_add_eol_comment(doc, k, column=40)
        return cm
    if isinstance(data, list):
        seq = CommentedSeq(_commented(v) for v in data)
        if all(isinstance(v, (int, float)) for v in data) and len(data) <= 4:
            seq.fa.set_flow_style()
        return seq
    if isinstance(data, (np.floating,)):
        return float(data)
    if isinstance(data, (np.integer,)):
        return int(data)
    if isinstance(data, tuple):
        return _commented(list(data))
    return data


def dump_yaml(data: dict, header: str | None = None) -> str:
    y = YAML()
    y.default_flow_style = False
    y.width = 110
    y.indent(mapping=2, sequence=4, offset=2)
    buf = io.StringIO()
    if header:
        buf.write("".join(f"# {line}\n" if line else "#\n" for line in header.splitlines()))
    y.dump(_commented(data), buf)
    return buf.getvalue()


def _update_latest(root: Path, d: Path) -> None:
    link = root / "latest"
    try:
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()
        os.symlink(d.name, link)
    except OSError:  # pragma: no cover - filesystems without symlinks
        (root / "LATEST").write_text(d.name)


def save(result, out_dir: str | os.PathLike = "gallery", *, name: str | None = None,
         formats: Iterable[str] = DEFAULT_FORMATS, overwrite: bool = False, trim: bool = False,
         alpha: str = "straight", columns: int | None = None, pad: int = 0, pot: bool = False,
         extra_meta: dict | None = None) -> Generation:
    """Write a RenderResult as a new generation (or overwrite the latest with overwrite=True)."""
    formats = tuple(formats)
    bad = set(formats) - set(ALL_FORMATS)
    if bad:
        raise ValueError(f"unknown format(s) {sorted(bad)}; use {', '.join(ALL_FORMATS)}")
    nm = _safe_name(name or result.name)
    root = Path(out_dir) / nm
    d, v = next_version_dir(root, overwrite)
    gen = Generation(dir=d, name=nm, version=v)

    explicit = dict(result.explicit)
    explicit["name"] = nm
    meta = dict(explicit.get("meta") or {})
    meta.update({
        "vfxgen_version": __version__,
        "rendered_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "source": result.source,
        "render_seconds": round(result.elapsed, 3),
        "frames": len(result.frames), "size": [result.width, result.height], "fps": result.fps,
        "draft": result.draft,
    })
    meta.update(extra_meta or {})
    explicit["meta"] = meta
    order = ["vfxgen", "name", "description", "size", "fps", "duration", "loop", "seed", "edge_fade",
             "vars", "palettes", "anchors", "channels", "layers", "meta"]
    explicit = {k: explicit[k] for k in order if k in explicit} | {k: v for k, v in explicit.items() if k not in order}
    header = (f"vfxgen generation {nm} v{v:03d}: fully explicit settings.\n"
              f"Re-render exactly with:  vfxgen render {d / (nm + '.yaml')}")
    yp = d / f"{nm}.yaml"
    yp.write_text(dump_yaml(explicit, header))
    gen.files["yaml"] = yp

    frames = result.frames
    if "sheet" in formats or "atlas" in formats:
        sheet, atlas = build_sheet(frames, result.fps, result.loop, columns=columns, trim=trim, pad=pad,
                                   pot=pot, alpha=alpha, times=result.times, events=result.events,
                                   tracks=result.tracks, pivot=result.pivot, blend=result.blend)
        if "sheet" in formats:
            p = d / f"{nm}_sheet.png"
            Image.fromarray(sheet, "RGBA").save(p, optimize=True)
            gen.files["sheet"] = p
        if "atlas" in formats:
            atlas["image"] = f"{nm}_sheet.png"
            p = d / f"{nm}_sheet.json"
            p.write_text(json.dumps(atlas, indent=1))
            gen.files["atlas"] = p
    if trim:
        from .atlas import union_bbox
        box = union_bbox(frames) or (0, 0, 1, 1)
        frames = [f[box[1]:box[3], box[0]:box[2]] for f in frames]
    frames_out = [apply_alpha(f, alpha) for f in frames]
    if "frames" in formats:
        fd = d / "frames"
        fd.mkdir()
        for i, f in enumerate(frames_out):
            Image.fromarray(f, "RGBA").save(fd / f"{nm}_{i:03d}.png", optimize=True)
        gen.files["frames"] = fd
    dur = [max(1, int(round(1000 / result.fps)))] * len(frames_out)
    if not result.loop and dur:
        dur[-1] += 400
    imgs = [Image.fromarray(f, "RGBA") for f in frames_out]
    if "webp" in formats and imgs:
        p = d / f"{nm}.webp"
        imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=dur, loop=0, quality=90, method=4)
        gen.files["webp"] = p
    if "apng" in formats and imgs:
        p = d / f"{nm}.apng.png"
        imgs[0].save(p, save_all=True, append_images=imgs[1:], duration=dur, loop=0)
        gen.files["apng"] = p
    if "gif" in formats and imgs:
        p = d / f"{nm}.gif"
        bg = np.array((18, 20, 28), np.float32)
        gif = []
        for f in frames:
            a = f[..., 3:4].astype(np.float32) / 255
            gif.append(Image.fromarray((f[..., :3] * a + bg * (1 - a)).astype(np.uint8)))
        gif[0].save(p, save_all=True, append_images=gif[1:], duration=dur, loop=0)
        gen.files["gif"] = p
    for mname, mframes in (getattr(result, "aux", None) or {}).items():
        msheet, _ = build_sheet(mframes, result.fps, result.loop, columns=columns, pad=pad, pot=pot)
        mp = d / f"{nm}_{mname}_sheet.png"
        Image.fromarray(msheet, "RGBA").save(mp, optimize=True)
        gen.files[f"{mname}_sheet"] = mp
    _update_latest(root, d)
    return gen
