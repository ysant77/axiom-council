"""Append-only, hash-chained audit ledger stored as JSON lines.

Each record carries the hash of the previous record, so any edit, deletion or reordering
is detected by :meth:`AuditLog.verify`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from axiom_council.contracts import content_hash, utcnow

GENESIS = "0" * 64


class AuditTamperError(RuntimeError):
    pass


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        last = GENESIS
        for record in self._records():
            last = record["hash"]
        return last

    def _records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AuditTamperError(f"corrupt audit line {line_no}") from exc

    def append(self, event: str, task_id: str, actor: str, **data: Any) -> dict[str, Any]:
        body = {
            "seq": sum(1 for _ in self._records()),
            "ts": utcnow().isoformat(),
            "event": event,
            "task_id": task_id,
            "actor": actor,
            "data": data,
            "prev": self._last_hash(),
        }
        record = {**body, "hash": content_hash(body)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return record

    def verify(self) -> int:
        """Return the number of valid records; raise on the first broken link."""
        prev = GENESIS
        count = 0
        for count, record in enumerate(self._records(), start=1):
            claimed = record.pop("hash", None)
            if record.get("prev") != prev:
                raise AuditTamperError(f"broken chain at seq {record.get('seq')}")
            if record.get("seq") != count - 1:
                raise AuditTamperError(f"out-of-order sequence at record {count}")
            if content_hash(record) != claimed:
                raise AuditTamperError(f"hash mismatch at seq {record.get('seq')}")
            prev = claimed
        return count

    def for_task(self, task_id: str) -> list[dict[str, Any]]:
        return [r for r in self._records() if r["task_id"] == task_id]
