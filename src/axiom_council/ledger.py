"""Capability ledger: measured outcomes per (model, task class), stored in SQLite.

Only outcomes confirmed by a verifier (deterministic check or the director) belong here;
a model grading itself is not evidence.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from axiom_council.contracts import TaskClass, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    task_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    task_class TEXT NOT NULL,
    success INTEGER NOT NULL CHECK (success IN (0, 1)),
    verified_by TEXT NOT NULL,
    wall_seconds REAL NOT NULL,
    total_tokens INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_outcomes_model_class ON outcomes (model_id, task_class);
"""


@dataclass(frozen=True)
class SkillStats:
    successes: int
    failures: int
    mean_wall_seconds: float | None
    mean_tokens: float | None

    @property
    def trials(self) -> int:
        return self.successes + self.failures


class CapabilityLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    def record(
        self,
        *,
        task_id: str,
        model_id: str,
        node_id: str,
        task_class: TaskClass,
        success: bool,
        verified_by: str,
        wall_seconds: float,
        total_tokens: int,
    ) -> None:
        if verified_by in {"", model_id}:
            raise ValueError("an outcome must be verified by someone other than the model")
        with self._db:
            self._db.execute(
                "INSERT INTO outcomes (ts, task_id, model_id, node_id, task_class, success,"
                " verified_by, wall_seconds, total_tokens) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    utcnow().isoformat(),
                    task_id,
                    model_id,
                    node_id,
                    str(task_class),
                    int(success),
                    verified_by,
                    wall_seconds,
                    total_tokens,
                ),
            )

    def stats(self, model_id: str, task_class: TaskClass) -> SkillStats:
        row = self._db.execute(
            "SELECT COALESCE(SUM(success), 0), COUNT(*) - COALESCE(SUM(success), 0),"
            " AVG(wall_seconds), AVG(total_tokens)"
            " FROM outcomes WHERE model_id = ? AND task_class = ?",
            (model_id, str(task_class)),
        ).fetchone()
        return SkillStats(int(row[0]), int(row[1]), row[2], row[3])
