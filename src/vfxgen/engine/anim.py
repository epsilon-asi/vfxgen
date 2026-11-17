"""Animatable values: constants, keyframes and time expressions -> callables of time.

A compiled value is ``f(t, T, p) -> float | tuple`` where ``t`` = layer-local
seconds, ``T`` = parent (scene) seconds and ``p`` = local progress 0..1.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from ..errors import SpecError
from ..spec.expr import evaluate, full_expr
from ..spec.models import Keys
from .mathx import ease

Fn = Callable[[float, float, float], Any]


def _num(v):
    if isinstance(v, (list, tuple)):
        return tuple(float(x) for x in v)
    return float(v)


def compile_value(value: Any, variables: Mapping[str, Any] | None = None, path: str = "") -> Fn:
    variables = dict(variables or {})
    if isinstance(value, Keys):
        value = {"keys": value.keys}
    if isinstance(value, dict) and "keys" in value:
        keys = [(float(k[0]), _num(k[1]), (k[2] if len(k) > 2 else "linear")) for k in value["keys"]]
        keys.sort(key=lambda k: k[0])
        if any(isinstance(k[1], tuple) for k in keys):        # mixed scalar / [x, y]: promote scalars
            n = max(len(k[1]) for k in keys if isinstance(k[1], tuple))
            keys = [(t, v if isinstance(v, tuple) else (v,) * n, e) for t, v, e in keys]

        def kf(t, T=0.0, p=0.0):
            if t <= keys[0][0]:
                return keys[0][1]
            for (t0, v0, e0), (t1, v1, _) in zip(keys, keys[1:]):
                if t <= t1:
                    k = ease(e0, (t - t0) / (t1 - t0) if t1 > t0 else 1.0)
                    if isinstance(v0, tuple):
                        return tuple(a + (b - a) * k for a, b in zip(v0, v1))
                    return v0 + (v1 - v0) * k
            return keys[-1][1]
        return kf
    if isinstance(value, (list, tuple)) and any(isinstance(v, str) for v in value):
        parts = [compile_value(v, variables, path=f"{path}[{i}]") for i, v in enumerate(value)]
        return lambda t=0.0, T=0.0, p=0.0: tuple(float(f(t, T, p)) for f in parts)
    if isinstance(value, str):
        inner = full_expr(value)
        if inner is None:
            raise SpecError(f"expected a number, list, keyframes or '${{...}}' expression, got {value!r}",
                            path=path)

        def ex(t, T=0.0, p=0.0):
            try:
                return _num(evaluate(inner, {**variables, "t": t, "T": T, "p": p}))
            except SpecError as e:
                raise SpecError(e.message, path=path) from None
        return ex
    try:
        c = _num(value)
    except (TypeError, ValueError):
        raise SpecError(f"not an animatable value: {value!r}", path=path) from None
    return lambda t=0.0, T=0.0, p=0.0: c


def is_constant(value: Any) -> bool:
    return not (isinstance(value, (Keys, str)) or (isinstance(value, dict) and "keys" in value))
