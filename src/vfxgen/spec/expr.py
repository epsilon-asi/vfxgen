"""Safe ``${...}`` expressions.

A whitelisted subset of Python expressions: numbers, strings, lists, arithmetic,
comparisons, boolean ops, conditional expressions, variable names and a fixed
set of math functions. No attribute access, subscripts on arbitrary objects,
comprehensions, lambdas or imports.

* :func:`evaluate` evaluates one expression string (without ``${}``).
* :func:`substitute` walks a YAML tree and resolves every ``${...}`` that does not
  depend on time. Time-dependent expressions (using ``t``, ``T`` or ``p``) are left
  in place and evaluated per frame by :mod:`vfxgen.engine.anim`.
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Mapping

from ..errors import SpecError

TIME_NAMES = frozenset({"t", "T", "p"})
# names whose expressions are evaluated later (per frame, or per spawn generation)
DEFERRED_NAMES = TIME_NAMES | {"gen"}


def _smoothstep(e0, e1, x):
    if e1 == e0:
        return 0.0 if x < e0 else 1.0
    k = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return k * k * (3 - 2 * k)


FUNCS: dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "round": round,
    "floor": math.floor, "ceil": math.ceil, "sqrt": math.sqrt, "exp": math.exp, "log": math.log,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "atan2": math.atan2,
    "clamp": lambda x, lo, hi: min(max(x, lo), hi),
    "lerp": lambda a, b, k: a + (b - a) * k,
    "smoothstep": _smoothstep,
    "int": int, "float": float,
}
CONSTS = {"pi": math.pi, "tau": 2 * math.pi, "e": math.e, "true": True, "false": False}

_ALLOWED = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)

_cache: dict[str, Any] = {}
_PATTERN = re.compile(r"\$\{([^{}]*)\}")


def compile_expr(src: str):
    """Parse and validate an expression; returns a code object. Raises SpecError."""
    if src in _cache:
        return _cache[src]
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise SpecError(f"bad expression ${{{src}}}: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise SpecError(f"expression ${{{src}}} uses a disallowed construct "
                            f"({type(node).__name__})")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCS or node.keywords:
                raise SpecError(f"expression ${{{src}}}: only these functions are allowed: "
                                f"{', '.join(sorted(FUNCS))}")
    code = compile(tree, "<vfxgen-expr>", "eval")
    _cache[src] = (code, names_in(tree))
    return _cache[src]


def names_in(tree: ast.AST) -> frozenset[str]:
    return frozenset(n.id for n in ast.walk(tree) if isinstance(n, ast.Name)) - frozenset(FUNCS)


def evaluate(src: str, variables: Mapping[str, Any]) -> Any:
    code, names = compile_expr(src)
    env = dict(CONSTS)
    env.update(variables)
    missing = [n for n in names if n not in env]
    if missing:
        raise SpecError(f"expression ${{{src}}}: unknown variable(s) {', '.join(sorted(missing))}")
    try:
        return eval(code, {"__builtins__": {}}, {**FUNCS, **env})  # noqa: S307 - AST-whitelisted
    except Exception as e:  # ZeroDivisionError, TypeError, ...
        raise SpecError(f"expression ${{{src}}} failed: {e}") from None


def uses_time(src: str) -> bool:
    return bool(compile_expr(src)[1] & DEFERRED_NAMES)


def is_expr(value: Any) -> bool:
    return isinstance(value, str) and "${" in value


def full_expr(value: str) -> str | None:
    """Inner text if the whole string is a single ``${...}``, else None."""
    m = _PATTERN.fullmatch(value.strip())
    return m.group(1) if m else None


def substitute(node: Any, variables: Mapping[str, Any], path: str = "") -> Any:
    """Resolve all static ``${...}`` in a YAML tree. Time-dependent ones are kept verbatim
    (only allowed as a whole string)."""
    if isinstance(node, dict):
        return {k: substitute(v, variables, f"{path}.{k}" if path else str(k)) for k, v in node.items()}
    if isinstance(node, list):
        return [substitute(v, variables, f"{path}[{i}]") for i, v in enumerate(node)]
    if not isinstance(node, str) or "${" not in node:
        return node
    inner = full_expr(node)
    try:
        if inner is not None:
            if uses_time(inner):
                return node            # deferred: evaluated per frame
            return evaluate(inner, variables)

        def rep(m: re.Match) -> str:
            if uses_time(m.group(1)):
                raise SpecError("time-dependent expressions must be the whole value, "
                                f"not embedded in text: {node!r}")
            return str(evaluate(m.group(1), variables))
        return _PATTERN.sub(rep, node)
    except SpecError as e:
        raise SpecError(str(e), path=path) from None
