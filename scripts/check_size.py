"""Enforce the size standards: at most 500 lines per file and 50 lines per function.

Usage: python scripts/check_size.py [paths...]   (defaults to src and tests)
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MAX_FILE_LINES = 500
MAX_FUNCTION_LINES = 50


def violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    found: list[str] = []
    lines = source.count("\n") + 1
    if lines > MAX_FILE_LINES:
        found.append(f"{path}: {lines} lines > {MAX_FILE_LINES}")
    for node in ast.walk(ast.parse(source, filename=str(path))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > MAX_FUNCTION_LINES:
                found.append(
                    f"{path}:{node.lineno} {node.name}() {length} lines > {MAX_FUNCTION_LINES}"
                )
    return found


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path("src"), Path("tests")]
    problems = [v for root in roots for f in sorted(root.rglob("*.py")) for v in violations(f)]
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
