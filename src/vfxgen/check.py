"""Quality checks for renders: loop seams, clipping at the frame edge, one-shots that should
end empty, empty / blank frames."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Issue:
    level: str        # "error" | "warning"
    code: str
    message: str


@dataclass
class CheckReport:
    name: str
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(i.level == "error" for i in self.issues)

    def summary(self) -> str:
        head = f"check {self.name}: {'OK' if self.ok else 'FAILED'}"
        lines = [head] + [f"  [{i.level}] {i.code}: {i.message}" for i in self.issues]
        return "\n".join(lines)


def _diff(a: np.ndarray, b: np.ndarray) -> float:
    fa = a.astype(np.float32) / 255
    fb = b.astype(np.float32) / 255
    pa = fa[..., :3] * fa[..., 3:]
    pb = fb[..., :3] * fb[..., 3:]
    return float(np.abs(pa - pb).mean() + np.abs(fa[..., 3] - fb[..., 3]).mean())


def check_frames(frames: list[np.ndarray], loop: bool, name: str = "render",
                 border: int = 2, border_alpha: int = 24, tile: bool = False,
                 ends_visible: bool = False) -> CheckReport:
    rep = CheckReport(name)
    n = len(frames)
    alphas = [f[..., 3] for f in frames]
    blank = [i for i, a in enumerate(alphas) if not a.any()]
    if len(blank) == n:
        rep.issues.append(Issue("error", "empty", "every frame is fully transparent"))
        return rep
    edge = 0
    for a in alphas:
        e = max(int(a[:border].max()), int(a[-border:].max()), int(a[:, :border].max()), int(a[:, -border:].max()))
        edge = max(edge, e)
    rep.stats["max_edge_alpha"] = edge
    if edge > border_alpha and not tile:
        rep.issues.append(Issue("warning", "edge_clip",
                                f"content reaches the frame border (alpha {edge}/255); enlarge `size` or "
                                "move layers inward"))
    if n >= 3:
        steps = [_diff(frames[i], frames[i + 1]) for i in range(n - 1)]
        rep.stats["max_step"] = max(steps)
        if loop:
            seam = _diff(frames[-1], frames[0])
            rep.stats["seam"] = seam
            if seam > 1.6 * max(steps) + 1e-4:
                rep.issues.append(Issue("warning", "loop_seam",
                                        f"last->first frame change ({seam:.4f}) is much larger than any "
                                        f"frame step ({max(steps):.4f}); the loop may pop"))
        elif not ends_visible:
            last = float(alphas[-1].max())
            rep.stats["last_frame_max_alpha"] = last
            if last > 16:
                rep.issues.append(Issue("warning", "not_empty_at_end",
                                        f"one-shot's last frame is not empty (max alpha {last:.0f}/255)"))
    if blank and loop:
        rep.issues.append(Issue("warning", "blank_frames", f"{len(blank)} blank frame(s) in a loop: {blank[:8]}"))
    return rep


def check_result(result) -> CheckReport:
    ex = result.explicit or {}
    return check_frames(result.frames, result.loop, result.name, tile=bool(ex.get("tile")),
                        ends_visible=bool(ex.get("ends_visible")))


def check_path(target: str, draft: bool = False, vars=None, overrides=None, colors=None) -> CheckReport:
    """Check a generation folder (reads its frames + YAML) or render a scene / effect and check it."""
    p = Path(target)
    if p.is_dir():
        from PIL import Image
        from .spec.loader import read_yaml
        ymls = [y for y in p.glob("*.yaml")]
        meta = (read_yaml(ymls[0].read_text()) or {}) if ymls else {}
        loop = bool(meta.get("loop"))
        frames = [np.asarray(Image.open(f).convert("RGBA")) for f in sorted((p / "frames").glob("*.png"))]
        if not frames:
            raise FileNotFoundError(f"no frames/ in {p}")
        return check_frames(frames, loop, p.parent.name if p.name.startswith("v") else p.name,
                            tile=bool(meta.get("tile")), ends_visible=bool(meta.get("ends_visible")))
    from .render import render
    return check_result(render(target, vars, overrides=overrides, colors=colors, draft=draft))
