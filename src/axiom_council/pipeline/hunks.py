"""Read the changed line ranges of a unified diff, so code can check what a finding cites.

Each range is the span of one hunk on the new side of the file, context lines included.
A deleted file keeps its old path and old-side ranges. A pure deletion inside a file gives
the one new-side line next to the removed lines. Text that is not a diff gives no ranges.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_FILE = "/dev/null"

Ranges = dict[str, list[tuple[int, int]]]


def _path(header: str) -> str:
    """The path in a `---` or `+++` header, without a timestamp or an `a/` or `b/` prefix."""
    name = header.split("\t", 1)[0].strip()
    if name == _NO_FILE:
        return name
    return name[2:] if name[:2] in {"a/", "b/"} else name


def _span(start: int, count: int) -> tuple[int, int]:
    first = max(start, 1)
    return (first, first + count - 1) if count > 0 else (first, first)


@dataclass
class _Reader:
    ranges: Ranges = field(default_factory=dict)
    old_path: str = _NO_FILE
    new_path: str = _NO_FILE
    old_left: int = 0
    new_left: int = 0

    def in_hunk(self) -> bool:
        return self.old_left > 0 or self.new_left > 0

    def hunk_line(self, line: str) -> None:
        """Count one body line; a body line is content even when it looks like a header."""
        marker = line[:1]
        if marker == "\\":
            return
        if marker in {" ", "", "-"}:
            self.old_left -= 1
        if marker in {" ", "", "+"}:
            self.new_left -= 1

    def header(self, line: str) -> None:
        if line.startswith("--- "):
            self.old_path = _path(line[4:])
        elif line.startswith("+++ "):
            self.new_path = _path(line[4:])
        elif match := _HUNK.match(line):
            self.hunk(*(int(g) if g is not None else 1 for g in match.groups()))

    def hunk(self, old_start: int, old_count: int, new_start: int, new_count: int) -> None:
        self.old_left, self.new_left = old_count, new_count
        if self.new_path != _NO_FILE:
            path, span = self.new_path, _span(new_start, new_count)
        elif self.old_path != _NO_FILE:
            path, span = self.old_path, _span(old_start, old_count)
        else:
            return
        self.ranges.setdefault(path, []).append(span)


def changed_ranges(patch: str) -> Ranges:
    """Map each file the patch touches to the line ranges of its hunks."""
    reader = _Reader()
    for line in patch.splitlines():
        if reader.in_hunk():
            reader.hunk_line(line)
        else:
            reader.header(line)
    return reader.ranges
