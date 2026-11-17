"""Exception types. Every error raised for bad input names where in the spec it happened."""
from __future__ import annotations


class VfxError(Exception):
    """Base class for all vfxgen errors."""


class SpecError(VfxError, ValueError):
    """Invalid scene / layer / parameter. ``path`` is the location in the spec,
    e.g. ``layers[2].repeat.count``."""

    def __init__(self, message: str, path: str = ""):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}" if path else message)


class ResolveError(SpecError):
    """An effect reference could not be resolved (missing file, unknown name, cycle, depth)."""


class LimitError(SpecError):
    """A hard limit (instances, size, frames, depth) was exceeded."""


class RenderError(VfxError, RuntimeError):
    """Rendering failed inside an atom or channel."""


class RenderCancelled(VfxError):
    """The render was cancelled through its ``cancel`` event."""
