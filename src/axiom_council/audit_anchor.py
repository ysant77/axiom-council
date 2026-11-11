"""Audit anchors: the head hash and record count of an audit log, kept outside the log.

A hash chain alone cannot show that records were cut from its end, or that the whole file was
swapped for another valid chain. An anchor records what the log looked like after the last
append, so :meth:`AuditLog.verify` can compare the log with it. Each anchor store keeps one
anchor; the audit log writes every store it has after each append.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from axiom_council.contracts import utcnow

_HASH = re.compile(r"^[0-9a-f]{64}$")


class AnchorError(RuntimeError):
    """An anchor store holds data that is not a valid anchor."""


@dataclass(frozen=True)
class Anchor:
    count: int
    head: str

    def __post_init__(self) -> None:
        if self.count < 0 or not _HASH.match(self.head):
            raise AnchorError(f"invalid anchor (count={self.count!r}, head={self.head!r})")

    def as_dict(self) -> dict[str, int | str]:
        return {"count": self.count, "head": self.head}


class AnchorStore(Protocol):
    """One place that holds the anchor of one audit log."""

    name: str

    def read(self) -> Anchor | None: ...

    def write(self, anchor: Anchor) -> None: ...


def fsync_dir(path: Path) -> None:
    """Make a rename or a new file in this directory durable."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class FileAnchor:
    """A small JSON file next to the log, replaced atomically on every write."""

    name = "file"

    def __init__(self, path: Path):
        self.path = path

    def read(self) -> Anchor | None:
        if not self.path.exists():
            return None
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return Anchor(int(data["count"]), str(data["head"]))
        except (ValueError, KeyError, TypeError) as exc:
            raise AnchorError(f"unreadable file anchor {self.path}") from exc

    def write(self, anchor: Anchor) -> None:
        body = {**anchor.as_dict(), "updated_at": utcnow().isoformat()}
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(body, sort_keys=True))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(self.path)
        fsync_dir(self.path.parent)


_TABLE = """
CREATE TABLE IF NOT EXISTS audit_anchor (
    log TEXT PRIMARY KEY,
    count INTEGER NOT NULL,
    head TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class SqliteAnchor:
    """A row in a SQLite database (the task store), keyed by the log file name."""

    name = "database"

    def __init__(self, db: Path, log_name: str):
        self.db = db
        self.log_name = log_name

    def read(self) -> Anchor | None:
        if not self.db.exists():
            return None
        conn = sqlite3.connect(self.db.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
        try:
            row = conn.execute(
                "SELECT count, head FROM audit_anchor WHERE log = ?", (self.log_name,)
            ).fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" in str(exc):
                return None
            raise AnchorError(f"unreadable database anchor in {self.db}: {exc}") from exc
        finally:
            conn.close()
        if row is None:
            return None
        try:
            return Anchor(int(row[0]), str(row[1]))
        except AnchorError as exc:
            raise AnchorError(f"unreadable database anchor in {self.db}") from exc

    def write(self, anchor: Anchor) -> None:
        conn = sqlite3.connect(self.db, timeout=30)
        try:
            conn.executescript(_TABLE)
            with conn:
                conn.execute(
                    "INSERT INTO audit_anchor (log, count, head, updated_at) VALUES (?, ?, ?, ?)"
                    " ON CONFLICT(log) DO UPDATE SET count = excluded.count,"
                    " head = excluded.head, updated_at = excluded.updated_at",
                    (self.log_name, anchor.count, anchor.head, utcnow().isoformat()),
                )
        finally:
            conn.close()
