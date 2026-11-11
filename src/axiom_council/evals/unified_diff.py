"""Strict parser for unified diffs: files, hunks and line numbers in the new file version.

Admission scoring locates faults by path and new-file line, so the parser refuses any hunk
whose body does not match its header counts. A removed line carries the new-file line it
sits before, so a fault made by a deletion still has a location.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_PREFIXES = ("a/", "b/", "./")

Op = Literal["+", "-", " "]


class DiffError(ValueError):
    """The text is not a well-formed unified diff."""


@dataclass(frozen=True)
class DiffLine:
    op: Op
    text: str
    new_line: int  # for "-", the new-file line that follows the removal


@dataclass(frozen=True)
class Hunk:
    path: str
    new_start: int
    new_count: int
    lines: tuple[DiffLine, ...]

    @property
    def new_end(self) -> int:
        return self.new_start + max(self.new_count, 1) - 1

    def covers(self, start: int, end: int) -> bool:
        return self.new_start <= start <= end <= self.new_end


def normalise_path(path: str) -> str:
    """Drop the `a/`, `b/` or `./` prefix a diff or a model may put before a path."""
    path = path.strip()
    for prefix in _PREFIXES:
        if path.startswith(prefix):
            return path[len(prefix) :]
    return path


def _file_path(old: str, new: str) -> str:
    target = new if new != "/dev/null" else old
    return normalise_path(target.split("\t")[0])


@dataclass
class _Builder:
    path: str
    new_start: int
    new_count: int
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
        return Hunk(self.path, self.new_start, self.new_count, tuple(self.lines))


def _start(header: str, path: str | None) -> _Builder:
    match = _HUNK.match(header)
    if match is None:
        raise DiffError(f"malformed hunk header: {header!r}")
    if path is None:
        raise DiffError("hunk before any file header")
    _, old_count, new_start, new_count = match.groups()
    old_n = 1 if old_count is None else int(old_count)
    new_n = 1 if new_count is None else int(new_count)
    first = int(new_start) if new_n else int(new_start) + 1
    return _Builder(path, int(new_start), new_n, old_n, new_n, first, [])


def parse_diff(text: str) -> tuple[Hunk, ...]:
    """Every hunk in the diff, in order. Raise DiffError on anything malformed."""
    hunks: list[Hunk] = []
    path: str | None = None
    old_name = ""
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
        if raw.startswith("--- "):
            old_name = raw[4:]
        elif raw.startswith("+++ "):
            path = _file_path(old_name, raw[4:])
        elif raw.startswith("@@"):
            current = _start(raw, path)
        else:
            raise DiffError(f"unexpected line outside a hunk: {raw[:60]!r}")
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
