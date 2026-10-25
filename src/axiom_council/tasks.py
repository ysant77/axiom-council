"""Durable task records: spec, inputs and state survive a process restart.

State changes go through an explicit transition table, so a task can never jump from
"awaiting approval" to "running" without passing "approved". The stored spec is re-validated
on load and its digest must still match what the director approved.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from axiom_council.contracts import TaskSpec, TaskState, utcnow

S = TaskState
TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    S.DRAFT: frozenset({S.POLICY_REJECTED, S.ROUTED}),
    S.ROUTED: frozenset({S.AWAITING_APPROVAL, S.AWAITING_CLARIFICATION, S.RUNNING}),
    S.AWAITING_APPROVAL: frozenset({S.APPROVED, S.REJECTED}),
    S.AWAITING_CLARIFICATION: frozenset({S.REJECTED, S.ROUTED}),
    S.APPROVED: frozenset({S.RUNNING, S.POLICY_REJECTED, S.AWAITING_CLARIFICATION}),
    S.RUNNING: frozenset({S.COMPLETED, S.FAILED}),
}
TERMINAL = frozenset({S.COMPLETED, S.FAILED, S.REJECTED, S.POLICY_REJECTED, S.VERIFIED})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    spec TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    user_input TEXT NOT NULL,
    role TEXT NOT NULL,
    output_tokens INTEGER NOT NULL,
    state TEXT NOT NULL,
    inbox_item TEXT,
    reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    pinned_model TEXT,
    pinned_node TEXT
);
"""
# Columns added after the first schema; opened stores are migrated in place.
_ADDED_COLUMNS = {"pinned_model": "TEXT", "pinned_node": "TEXT"}


class TaskStateError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRecord:
    spec: TaskSpec
    user_input: str
    role: str
    output_tokens: int
    state: TaskState
    inbox_item: str | None
    reason: str | None
    pinned_model: str | None = None
    pinned_node: str | None = None


class TaskStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.executescript(_SCHEMA)
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(tasks)")}
        with self._db:
            for name, kind in _ADDED_COLUMNS.items():
                if name not in existing:
                    self._db.execute(f"ALTER TABLE tasks ADD COLUMN {name} {kind}")

    def close(self) -> None:
        self._db.close()

    def create(self, spec: TaskSpec, user_input: str, role: str, output_tokens: int) -> None:
        now = utcnow().isoformat()
        try:
            with self._db:
                self._db.execute(
                    "INSERT INTO tasks (task_id, idempotency_key, spec, spec_digest, user_input,"
                    " role, output_tokens, state, created_at, updated_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        spec.task_id, spec.idempotency_key, spec.model_dump_json(),
                        spec.digest(), user_input, role, output_tokens, str(S.DRAFT), now, now,
                    ),
                )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise TaskStateError(f"task {spec.task_id} or its idempotency key exists") from exc

    def get(self, task_id: str) -> TaskRecord:
        row = self._db.execute(
            "SELECT spec, spec_digest, user_input, role, output_tokens, state, inbox_item, reason,"
            " pinned_model, pinned_node FROM tasks WHERE task_id = ?",
            (task_id,),
        ).fetchone()
        if row is None:
            raise TaskStateError(f"unknown task {task_id!r}")
        spec = TaskSpec.model_validate_json(row[0])
        if spec.digest() != row[1]:
            raise TaskStateError(f"stored spec for {task_id} no longer matches its digest")
        return TaskRecord(spec, row[2], row[3], row[4], S(row[5]), row[6], row[7], row[8], row[9])

    def pin(self, task_id: str, model_id: str, node_id: str) -> None:
        """Record the placement shown to the director; only while the task is being routed."""
        if self.get(task_id).state is not S.ROUTED:
            raise TaskStateError(f"{task_id}: a placement can only be pinned while routed")
        with self._db:
            self._db.execute(
                "UPDATE tasks SET pinned_model = ?, pinned_node = ?, updated_at = ?"
                " WHERE task_id = ?",
                (model_id, node_id, utcnow().isoformat(), task_id),
            )

    def transition(
        self,
        task_id: str,
        new: TaskState,
        *,
        inbox_item: str | None = None,
        reason: str | None = None,
    ) -> None:
        current = self.get(task_id).state
        if new not in TRANSITIONS.get(current, frozenset()):
            raise TaskStateError(f"{task_id}: illegal transition {current} -> {new}")
        with self._db:
            self._db.execute(
                "UPDATE tasks SET state = ?, inbox_item = COALESCE(?, inbox_item),"
                " reason = COALESCE(?, reason), updated_at = ? WHERE task_id = ?",
                (str(new), inbox_item, reason, utcnow().isoformat(), task_id),
            )

    def in_state(self, state: TaskState) -> list[str]:
        rows = self._db.execute(
            "SELECT task_id FROM tasks WHERE state = ? ORDER BY created_at", (str(state),)
        ).fetchall()
        return [r[0] for r in rows]

    def summary(self, limit: int = 50) -> list[dict[str, str]]:
        rows = self._db.execute(
            "SELECT task_id, state, updated_at, spec FROM tasks ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"task_id": r[0], "state": r[1], "updated_at": r[2],
             "objective": json.loads(r[3])["objective"][:80]}
            for r in rows
        ]  # fmt: skip
