"""Splice a model's refactored function (plus new private helpers) into a module.

The model returns only function definitions. The target must keep its name; helpers must be
new, private (leading underscore) and module-level; imports and any other statements are
refused. Every returned function must fit the line limit.
"""

from __future__ import annotations

import ast
import re

_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class SpliceError(ValueError):
    pass


def extract_code(output: str) -> str:
    text = _THINK.sub("", output)
    blocks = _CODE_BLOCK.findall(text)
    return max(blocks, key=len) if blocks else text.strip()


def _parse_functions(code: str) -> list[ast.FunctionDef]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise SpliceError(f"returned code does not parse: {exc.msg}") from exc
    functions: list[ast.FunctionDef] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node)
        elif not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)):
            raise SpliceError(f"only function definitions are allowed, got {type(node).__name__}")
    return functions


def _check(
    functions: list[ast.FunctionDef], target: str, existing: set[str], max_lines: int
) -> None:
    names = [f.name for f in functions]
    if target not in names:
        raise SpliceError(f"target function {target!r} missing from the output")
    if len(names) != len(set(names)):
        raise SpliceError("a function is defined twice")
    for fn in functions:
        length = (fn.end_lineno or fn.lineno) - fn.lineno + 1
        if length > max_lines:
            raise SpliceError(f"{fn.name}() is {length} lines > {max_lines}")
        if fn.name == target:
            continue
        if not fn.name.startswith("_"):
            raise SpliceError(f"helper {fn.name!r} must be private (leading underscore)")
        if fn.name in existing:
            raise SpliceError(f"helper {fn.name!r} would overwrite an existing name")


def splice(source: str, target: str, model_code: str, max_lines: int) -> str:
    module = ast.parse(source)
    original = next(
        (n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == target), None
    )
    if original is None:
        raise SpliceError(f"{target!r} is not a module-level function")
    existing = {n.name for n in module.body if isinstance(n, ast.FunctionDef | ast.ClassDef)}
    existing |= {
        t.id for n in module.body if isinstance(n, ast.Assign) for t in n.targets
        if isinstance(t, ast.Name)
    }  # fmt: skip
    functions = _parse_functions(model_code)
    _check(functions, target, existing - {target}, max_lines)

    lines = source.splitlines(keepends=True)
    start = (original.decorator_list[0].lineno if original.decorator_list else original.lineno) - 1
    end = original.end_lineno or original.lineno
    replacement = model_code.strip("\n") + "\n"
    return "".join(lines[:start]) + replacement + "".join(lines[end:])
