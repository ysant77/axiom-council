"""Capability ledger: measured outcomes per (placement, task class), stored in SQLite.

Only outcomes confirmed by a verifier (deterministic check or the director) belong here;
a model grading itself is not evidence.

Evidence rules:
- Every outcome has a split. Runtime and calibration outcomes inform routing; development
  outcomes are stored for debugging but never route; holdout outcomes are refused outright.
- A benchmark outcome carries a run ID and an item key. Routing counts only the latest outcome
  per (placement, item), so re-running a benchmark never multiplies one item's weight.
- Evidence belongs to one placement: model build (digest), quantisation, runtime, node and
  reasoning setting (D-04). A row with unknown provenance matches only a placement whose
  same fields are unknown. The harness hash is stored and can be required by the caller.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from axiom_council.contracts import TaskClass, utcnow
from axiom_council.provenance import Placement

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
# Columns added after the first schema; older ledgers are migrated in place.
_ADDED_COLUMNS = {
    "split": "TEXT NOT NULL DEFAULT 'runtime'",
    "run_id": "TEXT",
    "item_key": "TEXT",
    "runtime": "TEXT",
    "model_digest": "TEXT",
    "quantization": "TEXT",
    "reasoning": "TEXT",
    "harness_hash": "TEXT",
}
_MIGRATE = """
UPDATE outcomes SET item_key = task_id, run_id = 'legacy-' || id, split = 'calibration'
    WHERE item_key IS NULL AND task_id LIKE 'bakeoff-%';
DROP INDEX IF EXISTS ux_outcomes_run_item;
CREATE UNIQUE INDEX IF NOT EXISTS ux_outcomes_run_item_placement
    ON outcomes (run_id, item_key, model_id, node_id, COALESCE(runtime, ''),
                 COALESCE(model_digest, ''), COALESCE(quantization, ''), COALESCE(reasoning, ''))
    WHERE item_key IS NOT NULL;
"""
SPLITS = frozenset({"runtime", "development", "calibration", "holdout"})
ROUTING_SPLITS = ("runtime", "calibration")
_STATS = """
WITH eligible AS (
    SELECT * FROM outcomes
    WHERE model_id = ? AND node_id = ? AND runtime IS ? AND model_digest IS ?
      AND quantization IS ? AND reasoning IS ? AND task_class = ?
      AND split IN ('runtime', 'calibration') AND (? IS NULL OR harness_hash = ?)
), latest AS (
    SELECT id FROM eligible WHERE item_key IS NULL
    UNION ALL
    SELECT MAX(id) FROM eligible WHERE item_key IS NOT NULL GROUP BY item_key
)
SELECT COALESCE(SUM(success), 0), COUNT(*) - COALESCE(SUM(success), 0),
       AVG(wall_seconds), AVG(total_tokens)
FROM outcomes WHERE id IN (SELECT id FROM latest)
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


def _check_outcome(
    model_id: str, verified_by: str, split: str, run_id: str | None, item_key: str | None
) -> None:
    if verified_by in {"", model_id}:
        raise ValueError("an outcome must be verified by someone other than the model")
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLITS)}")
    if split == "holdout":
        raise ValueError("holdout results never enter the routing ledger")
    if item_key is not None and not run_id:
        raise ValueError("a benchmark item needs a run_id")


class CapabilityLedger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(outcomes)")}
        with self._db:
            for name, kind in _ADDED_COLUMNS.items():
                if name not in existing:
                    self._db.execute(f"ALTER TABLE outcomes ADD COLUMN {name} {kind}")
        self._db.executescript(_MIGRATE)

    def close(self) -> None:
        self._db.close()

    def record(
        self,
        *,
        task_id: str,
        placement: Placement,
        task_class: TaskClass,
        success: bool,
        verified_by: str,
        wall_seconds: float,
        total_tokens: int,
        split: str = "runtime",
        run_id: str | None = None,
        item_key: str | None = None,
        harness_hash: str | None = None,
    ) -> None:
        p = placement
        _check_outcome(p.model_id, verified_by, split, run_id, item_key)
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO outcomes (ts, task_id, model_id, node_id, task_class, success,"
                    " verified_by, wall_seconds, total_tokens, split, run_id, item_key, runtime,"
                    " model_digest, quantization, reasoning, harness_hash)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        utcnow().isoformat(), task_id, p.model_id, p.node_id, str(task_class),
                        int(success), verified_by, wall_seconds, total_tokens, split, run_id,
                        item_key, p.runtime, p.model_digest, p.quantization, p.reasoning,
                        harness_hash,
                    ),
                )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"{item_key} is already recorded for {p} in {run_id}") from exc

    def stats(
        self, placement: Placement, task_class: TaskClass, *, harness_hash: str | None = None
    ) -> SkillStats:
        """Verified evidence for exactly this placement; `harness_hash` narrows it further."""
        p = placement
        row = self._db.execute(
            _STATS,
            (p.model_id, p.node_id, p.runtime, p.model_digest, p.quantization, p.reasoning,
             str(task_class), harness_hash, harness_hash),
        ).fetchone()  # fmt: skip
        return SkillStats(int(row[0]), int(row[1]), row[2], row[3])
