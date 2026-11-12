"""Append-only, hash-chained audit ledger stored as JSON lines, with anchors outside the log.

Each record carries the hash of the previous record. Anchors (:mod:`axiom_council.audit_anchor`)
keep the head hash and the record count after each append: one in a file next to the log and,
through :func:`open_audit`, one in the task database.

:meth:`AuditLog.verify` detects:

- an edited, removed or reordered record (the chain breaks);
- records cut from the end, or a log swapped for another valid chain, while one anchor is intact;
- a missing anchor while the log has records;
- a partial last line, for example from a crash during an append.

The anchors do not stop these attacks:

- A writer with write access to the log and to every anchor store can cut or rewrite history.
  That writer can recompute a valid chain and write matching anchors. Nothing is signed.
- Deleting the log and every anchor together looks like a new, empty log.
- Restoring the whole home directory from an older copy gives an older, consistent state.
- A process that can call :meth:`AuditLog.append` can add false records.

Protection against these needs a witness outside this host, for example a head hash that the
director keeps or a remote append-only store. The files stay writable; this design only detects.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from axiom_council.audit_anchor import (
    Anchor,
    AnchorError,
    AnchorStore,
    FileAnchor,
    SqliteAnchor,
    fsync_dir,
)
from axiom_council.contracts import content_hash, utcnow
from axiom_council.filelock import locked

GENESIS = "0" * 64
MIN_REASON_CHARS = 5
_CHUNK = 4096
_ADOPT_HINT = "; check the log, then run `council audit adopt --reason ...`"


class AuditTamperError(RuntimeError):
    pass


def open_audit(home: Path) -> AuditLog:
    """The council's audit log, anchored in a file and in the task database."""
    return AuditLog(home / "audit.jsonl", anchor_db=home / "tasks.sqlite3")


def _check_chain(records: list[dict[str, Any]]) -> list[str]:
    """Return the hash of every record; raise on the first broken link."""
    prev, hashes = GENESIS, []
    for index, record in enumerate(records):
        body = dict(record)
        claimed = body.pop("hash", None)
        if body.get("prev") != prev:
            raise AuditTamperError(f"broken chain at seq {body.get('seq')}")
        if body.get("seq") != index:
            raise AuditTamperError(f"out-of-order sequence at record {index + 1}")
        if content_hash(body) != claimed:
            raise AuditTamperError(f"hash mismatch at seq {body.get('seq')}")
        hashes.append(claimed)
        prev = claimed
    return hashes


def _anchor_problem(
    name: str, anchor: Anchor | None, count: int, hash_at: Callable[[int], str | None]
) -> str | None:
    """Why the log does not match one anchor; None when it matches.

    hash_at(n) gives the head hash after n records, or None when it is not known.
    """
    if anchor is None:
        return f"{name} anchor missing while the log has {count} records" if count else None
    if anchor.count > count:
        return f"log has {count} records, shorter than the {name} anchor ({anchor.count})"
    at = hash_at(anchor.count)
    if at is not None and at != anchor.head:
        if anchor.count == count:
            return f"log head differs from the {name} anchor (log replaced?)"
        return f"log diverges from the {name} anchor at record {anchor.count} (log replaced?)"
    if anchor.count < count:
        return f"log has {count - anchor.count} records past the {name} anchor"
    return None


def _hint(problem: str) -> str:
    """Point to adopt only for the states that adopt can accept."""
    return _ADOPT_HINT if "missing" in problem or "past" in problem else ""


def _partial_error(line_no: int | None = None) -> AuditTamperError:
    where = f" at line {line_no}" if line_no else ""
    return AuditTamperError(
        f"partial last record{where}: the log does not end with a newline"
        f" (crash during an append?){_ADOPT_HINT}"
    )


class AuditLog:
    def __init__(self, path: Path, *, anchor_db: Path | None = None):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = path.with_name(path.name + ".lock")
        file_anchor = FileAnchor(path.with_name(path.name + ".anchor"))
        db_anchor = () if anchor_db is None else (SqliteAnchor(anchor_db, path.name),)
        self.anchors: tuple[AnchorStore, ...] = (file_anchor, *db_anchor)

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        """An advisory lock shared by every process that uses this log."""
        with locked(self._lock_path, exclusive=exclusive):
            yield

    def _read(self) -> tuple[list[dict[str, Any]], bytes]:
        """Every complete record, plus any bytes after the last newline."""
        if not self.path.exists():
            return [], b""
        body, _, partial = self.path.read_bytes().rpartition(b"\n")
        records: list[dict[str, Any]] = []
        for line_no, line in enumerate(body.split(b"\n"), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AuditTamperError(f"corrupt audit line {line_no}") from exc
            if not isinstance(record, dict):
                raise AuditTamperError(f"corrupt audit line {line_no}")
            records.append(record)
        return records, partial

    def records(self) -> list[dict[str, Any]]:
        """Every complete record in order; raises on a partial last line."""
        with self._locked(exclusive=False):
            records, partial = self._read()
        if partial:
            raise _partial_error(len(records) + 1)
        return records

    @staticmethod
    def _read_anchor(store: AnchorStore) -> Anchor | None:
        try:
            return store.read()
        except AnchorError as exc:
            raise AuditTamperError(f"{store.name} anchor unreadable: {exc}") from exc

    def _last_line(self) -> bytes | None:
        """The last complete line without reading the whole file; None for an empty log."""
        if not self.path.exists():
            return None
        with self.path.open("rb") as fh:
            end = fh.seek(0, os.SEEK_END) - 1
            if end < 0:
                return None
            fh.seek(end)
            if fh.read(1) != b"\n":
                raise _partial_error()
            pos, buf = end, b""
            while pos > 0:
                step = min(_CHUNK, pos)
                pos -= step
                fh.seek(pos)
                buf = fh.read(step) + buf
                cut = buf.rfind(b"\n")
                if cut != -1:
                    return buf[cut + 1 :]
            return buf

    def _tail(self) -> tuple[int, str]:
        """Record count and head hash, read from the last line only."""
        line = self._last_line()
        if line is None:
            return 0, GENESIS
        try:
            body = json.loads(line)
            claimed = body.pop("hash")
            seq = body["seq"]
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as exc:
            raise AuditTamperError("corrupt last audit line") from exc
        if not isinstance(seq, int) or content_hash(body) != claimed:
            raise AuditTamperError(f"hash mismatch at the last record (seq {seq})")
        return seq + 1, str(claimed)

    def _write(
        self, count: int, head: str, event: str, task_id: str, actor: str, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Append one record after (count, head), then move every anchor to it. Hold the lock."""
        body = {
            "seq": count,
            "ts": utcnow().isoformat(),
            "event": event,
            "task_id": task_id,
            "actor": actor,
            "data": data,
            "prev": head,
        }
        digest = content_hash(body)
        record = {**body, "hash": digest}
        created = not self.path.exists()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if created:
            fsync_dir(self.path.parent)
        anchor = Anchor(count + 1, digest)
        for store in self.anchors:
            store.write(anchor)
        return record

    def append(self, event: str, task_id: str, actor: str, **data: Any) -> dict[str, Any]:
        """Add one record. Refuse when the log does not match its anchors, so that an append
        never makes a cut or replaced log look valid."""
        with self._locked(exclusive=True):
            count, head = self._tail()
            for store in self.anchors:
                problem = _anchor_problem(
                    store.name, self._read_anchor(store), count,
                    lambda n: head if n == count else (GENESIS if n == 0 else None),
                )  # fmt: skip
                if problem:
                    raise AuditTamperError(problem + _hint(problem))
            return self._write(count, head, event, task_id, actor, data)

    def _verified_hashes(self) -> list[str]:
        """Check the chain, the tail and every anchor under a shared lock."""
        with self._locked(exclusive=False):
            records, partial = self._read()
            hashes = _check_chain(records)
            if partial:
                raise _partial_error(len(records) + 1)
            for store in self.anchors:
                problem = _anchor_problem(
                    store.name, self._read_anchor(store), len(hashes),
                    lambda n: hashes[n - 1] if n else GENESIS,
                )  # fmt: skip
                if problem:
                    raise AuditTamperError(problem + _hint(problem))
        return hashes

    def verify(self) -> int:
        """Return the number of records; raise when the chain, the tail or an anchor is wrong."""
        return len(self._verified_hashes())

    def head(self) -> Anchor:
        """The verified record count and head hash."""
        hashes = self._verified_hashes()
        return Anchor(len(hashes), hashes[-1] if hashes else GENESIS)

    def for_task(self, task_id: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r["task_id"] == task_id]

    def adopt(self, actor: str, reason: str) -> dict[str, Any]:
        """Trust the current log once, explicitly, and record that decision in the log itself.

        Use it to anchor a log written before anchors existed, or after a crash between the
        log write and the anchor write. A partial last line moves to a quarantine file. It
        refuses a log that is shorter than an anchor or that diverges from one.
        """
        if len(reason.strip()) < MIN_REASON_CHARS:
            raise ValueError("adopting the log needs a reason of at least five characters")
        with self._locked(exclusive=True):
            records, partial = self._read()
            hashes = _check_chain(records)
            previous = {store.name: self._read_anchor(store) for store in self.anchors}
            self._check_adoptable(previous, hashes, partial)
            dropped = self._quarantine(partial) if partial else {}
            head = hashes[-1] if hashes else GENESIS
            data = {
                "previous": {k: v.as_dict() if v else None for k, v in previous.items()},
                "records": len(hashes),
                "head": head,
                "reason": reason.strip(),
                **dropped,
            }
            return self._write(len(hashes), head, "audit_adopted", "audit", actor, data)

    @staticmethod
    def _check_adoptable(
        previous: dict[str, Anchor | None], hashes: list[str], partial: bytes
    ) -> None:
        behind = False
        for name, anchor in previous.items():
            if anchor is not None and anchor.count > len(hashes):
                raise AuditTamperError(f"cannot adopt: the log is shorter than the {name} anchor")
            if anchor is not None and anchor.head != (
                hashes[anchor.count - 1] if anchor.count else GENESIS
            ):
                raise AuditTamperError(f"cannot adopt: the log diverges from the {name} anchor")
            behind = behind or anchor is None or anchor.count < len(hashes)
        if not behind and not partial:
            raise ValueError("nothing to adopt: the log matches every anchor")

    def _quarantine(self, partial: bytes) -> dict[str, Any]:
        """Move the partial last line to its own file, then cut it from the log."""
        stamp = utcnow().strftime("%Y%m%dT%H%M%S%f")
        target = self.path.with_name(f"{self.path.name}.partial-{stamp}")
        with target.open("wb") as fh:
            fh.write(partial)
            fh.flush()
            os.fsync(fh.fileno())
        size = self.path.stat().st_size
        with self.path.open("r+b") as fh:
            fh.truncate(size - len(partial))
            fh.flush()
            os.fsync(fh.fileno())
        fsync_dir(self.path.parent)
        return {
            "quarantined": str(target),
            "dropped_bytes": len(partial),
            "dropped_sha256": hashlib.sha256(partial).hexdigest(),
        }
