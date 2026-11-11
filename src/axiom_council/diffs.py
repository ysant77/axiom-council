"""The one unified-diff reader: a strict parse and the changed line ranges built on it.

`parse_diff` refuses any hunk whose body does not match its header counts, and any text
outside a hunk that is not a file header or a git extended header. Body lines are content
even when they look like headers. A removed line carries the new-file line it sits before,
so a fault made by a deletion still has a location.

`changed_ranges` maps each file to the span of each hunk, context lines included. It uses
the new side, except for a deleted file, which keeps its old path and old-side spans. A
diff that does not parse gives no ranges, so the pipeline treats it as unreadable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_NO_FILE = "/dev/null"
_GIT_HEADERS = (
    "diff ", "index ", "old mode ", "new mode ", "deleted file mode ", "new file mode ",
    "similarity index ", "dissimilarity index ", "rename from ", "rename to ", "copy from ",
    "copy to ", "Binary files ",
)  # fmt: skip

Op = Literal["+", "-", " "]
Ranges = dict[str, list[tuple[int, int]]]


class DiffError(ValueError):
    """The text is not a well-formed unified diff."""


@dataclass(frozen=True)
class DiffLine:
    op: Op
    text: str
    new_line: int  # for "-", the new-file line that follows the removal


@dataclass(frozen=True)
class Hunk:
    path: str  # the new path; the old path for a deleted file
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    deleted: bool
    lines: tuple[DiffLine, ...]

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_count, 1) - 1

    def covers(self, start: int, end: int) -> bool:
        return self.new_start <= start <= end <= self.new_end


def _header_path(header: str) -> str:
    """The path in a `---` or `+++` header, without a timestamp or an `a/` or `b/` prefix."""
    name = header.split("\t", 1)[0].strip()
    if name == _NO_FILE:
        return name
    return name[2:] if name[:2] in {"a/", "b/"} else name


@dataclass
class _Builder:
    path: str
    deleted: bool
    counts: tuple[int, int, int, int]  # old start, old count, new start, new count
    old_left: int
    new_left: int
    next_new: int
    lines: list[DiffLine]

    def add(self, raw: str) -> None:
        op, text = (" ", "") if raw == "" else (raw[0], raw[1:])
        if op == " ":
            self.old_left, self.new_left = self.old_left - 1, self.new_left - 1
            self.lines.append(DiffLine(" ", text, self.next_new))
            self.next_new += 1
        elif op == "-":
            self.old_left -= 1
            self.lines.append(DiffLine("-", text, self.next_new))
        elif op == "+":
            self.new_left -= 1
            self.lines.append(DiffLine("+", text, self.next_new))
            self.next_new += 1
        else:
            raise DiffError(f"unknown diff line marker {op!r} in {self.path}")
        if self.old_left < 0 or self.new_left < 0:
            raise DiffError(f"hunk body longer than its header in {self.path}")

    @property
    def done(self) -> bool:
        return self.old_left == 0 and self.new_left == 0

    def build(self) -> Hunk:
        if not self.done:
            raise DiffError(f"hunk body shorter than its header in {self.path}")
        old_start, old_count, new_start, new_count = self.counts
        return Hunk(
            self.path, old_start, old_count, new_start, new_count, self.deleted,
            tuple(self.lines),
        )  # fmt: skip


@dataclass
class _Files:
    old: str = _NO_FILE
    new: str | None = None


def _start(header: str, files: _Files) -> _Builder:
    match = _HUNK.match(header)
    if match is None:
        raise DiffError(f"malformed hunk header: {header!r}")
    if files.new is None:
        raise DiffError("hunk before any file header")
    old_start, old_n, new_start, new_n = (1 if g is None else int(g) for g in match.groups())
    deleted = files.new == _NO_FILE
    path = files.old if deleted else files.new
    first = new_start if new_n else new_start + 1
    counts = (old_start, old_n, new_start, new_n)
    return _Builder(path, deleted, counts, old_n, new_n, first, [])


def _header(raw: str, files: _Files) -> _Builder | None:
    """Read one line outside a hunk; return a builder when it opens a hunk."""
    if raw.startswith("--- "):
        files.old, files.new = _header_path(raw[4:]), None
    elif raw.startswith("+++ "):
        files.new = _header_path(raw[4:])
    elif raw.startswith("@@"):
        return _start(raw, files)
    elif not raw.startswith(_GIT_HEADERS):
        raise DiffError(f"unexpected line outside a hunk: {raw[:60]!r}")
    return None


def parse_diff(text: str) -> tuple[Hunk, ...]:
    """Every hunk in the diff, in order. Raise DiffError on anything malformed."""
    hunks: list[Hunk] = []
    files = _Files()
    current: _Builder | None = None
    for raw in text.splitlines():
        if current is not None and not current.done and not raw.startswith("\\"):
            current.add(raw)
            continue
        if current is not None and current.done:
            hunks.append(current.build())
            current = None
        if raw.startswith("\\") or (raw == "" and current is None):
            continue
        current = _header(raw, files) or current
    if current is not None:
        hunks.append(current.build())
    if not hunks:
        raise DiffError("no hunks in diff")
    return tuple(hunks)


def change_blocks(hunk: Hunk) -> list[tuple[DiffLine, ...]]:
    """Runs of consecutive added or removed lines inside one hunk."""
    blocks: list[tuple[DiffLine, ...]] = []
    run: list[DiffLine] = []
    for line in hunk.lines:
        if line.op == " ":
            if run:
                blocks.append(tuple(run))
            run = []
        else:
            run.append(line)
    if run:
        blocks.append(tuple(run))
    return blocks


def _span(start: int, count: int) -> tuple[int, int]:
    first = max(start, 1)
    return (first, first + count - 1) if count > 0 else (first, first)


def changed_ranges(patch: str) -> Ranges:
    """Map each file the patch touches to the line ranges of its hunks. A patch that does not
    parse strictly gives no ranges."""
    try:
        hunks = parse_diff(patch)
    except DiffError:
        return {}
    ranges: Ranges = {}
    for hunk in hunks:
        if hunk.path == _NO_FILE:
            continue
        if hunk.deleted:
            span = _span(hunk.old_start, hunk.old_count)
        else:
            span = _span(hunk.new_start, hunk.new_count)
        ranges.setdefault(hunk.path, []).append(span)
    return ranges
