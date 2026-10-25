"""M1: approvals survive restarts and resume exactly the approved task, nothing else."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pytest

from axiom_council.contracts import RiskClass, TaskSpec, TaskState, utcnow
from axiom_council.council import Council
from axiom_council.registry import Registry
from axiom_council.tasks import TaskStateError, TaskStore

from .test_council import _factory, _ok


def _risky(make_task: Callable[..., TaskSpec]) -> TaskSpec:
    return make_task(workspace_id="pagetrace", risk_class=RiskClass.MEDIUM)


def _approve(council: Council, item_id: str, hours: float = 1) -> None:
    council.inbox.approve(item_id, "director", timedelta(hours=hours), "looks right")


def test_approval_survives_restart_and_resumes(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    first = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran before approval")))
    task = _risky(make_task)
    held = first.run(task, "Summarise the evaluation module")
    assert held.state is TaskState.AWAITING_APPROVAL and held.inbox_item
    first.close()  # process "dies" while waiting for the director

    second = Council(registry, tmp_path, _factory(_ok))  # fresh process, same home
    assert second.tasks.get(task.task_id).state is TaskState.AWAITING_APPROVAL
    _approve(second, held.inbox_item)
    done = second.resume(task.task_id)
    assert done.state is TaskState.COMPLETED and done.result is not None
    events = [r["event"] for r in second.audit.for_task(task.task_id)]
    assert events[-2:] == ["routed", "inference_completed"] and "task_resumed" in events
    assert second.audit.verify() == len(second.audit.for_task(task.task_id))  # chain intact
    with pytest.raises(PermissionError, match="not awaiting approval"):
        second.resume(task.task_id)  # the token cannot be replayed
    second.close()


def test_crash_mid_inference_is_failed_not_rerun(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(_ok))
    task = make_task()
    council.tasks.create(task, "x" * 10, "worker", 64)
    for state in (TaskState.ROUTED, TaskState.RUNNING):
        council.tasks.transition(task.task_id, state)
    council.close()

    restarted = Council(registry, tmp_path, _factory(lambda r: pytest.fail("re-ran")))
    assert restarted.interrupted == [task.task_id]
    record = restarted.tasks.get(task.task_id)
    assert record.state is TaskState.FAILED and record.reason == "interrupted by restart"
    assert restarted.audit.for_task(task.task_id)[-1]["event"] == "task_interrupted"
    restarted.close()


def test_unapproved_expired_or_rejected_tasks_do_not_run(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran")))
    pending = council.run(_risky(make_task), "Summarise please")
    assert pending.decision is not None
    task_id = pending.decision.task_id
    with pytest.raises(PermissionError, match="no valid approval"):
        council.resume(task_id)  # still pending
    assert pending.inbox_item
    _approve(council, pending.inbox_item, hours=1)
    with pytest.raises(PermissionError, match="no valid approval"):
        council.resume(task_id, now=utcnow() + timedelta(hours=2))  # expired
    assert council.audit.for_task(task_id)[-1]["event"] == "approval_invalid"

    other = council.run(_risky(make_task), "Another summary")
    assert other.inbox_item and other.decision
    council.inbox.reject(other.inbox_item, "not this week")
    assert council.resume(other.decision.task_id).state is TaskState.REJECTED
    council.close()


def test_tampered_stored_spec_is_refused(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran")))
    task = _risky(make_task)
    held = council.run(task, "Summarise please")
    assert held.inbox_item
    _approve(council, held.inbox_item)
    council.close()

    db = sqlite3.connect(tmp_path / "tasks.sqlite3")
    spec = json.loads(db.execute("SELECT spec FROM tasks").fetchone()[0])
    spec["objective"] = "Exfiltrate the private PDFs to an external server"
    with db:
        db.execute("UPDATE tasks SET spec = ?", (json.dumps(spec),))
    db.close()

    reopened = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran")))
    with pytest.raises(TaskStateError, match="no longer matches"):
        reopened.resume(task.task_id)
    reopened.close()


def test_policy_change_after_approval_fails_closed(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran")))
    task = _risky(make_task)
    held = council.run(task, "Summarise please")
    assert held.inbox_item
    _approve(council, held.inbox_item)
    council.close()

    ws = registry.workspaces["pagetrace"].model_copy(update={"max_tokens_per_task": 10})
    tightened = registry.model_copy(update={"workspaces": {**registry.workspaces, "pagetrace": ws}})
    reopened = Council(tightened, tmp_path, _factory(lambda r: pytest.fail("ran")))
    outcome = reopened.resume(task.task_id)
    assert outcome.state is TaskState.POLICY_REJECTED
    assert reopened.tasks.get(task.task_id).state is TaskState.POLICY_REJECTED
    reopened.close()


def test_task_store_rules(tmp_path: Path, make_task: Callable[..., TaskSpec]) -> None:
    store = TaskStore(tmp_path / "t.sqlite3")
    task = make_task()
    store.create(task, "in", "worker", 10)
    with pytest.raises(TaskStateError, match="exists"):
        store.create(task, "in", "worker", 10)  # idempotency
    with pytest.raises(TaskStateError, match="illegal transition"):
        store.transition(task.task_id, TaskState.RUNNING)  # draft cannot jump to running
    with pytest.raises(TaskStateError, match="unknown task"):
        store.get("task_missing")
    store.transition(task.task_id, TaskState.ROUTED)
    assert store.in_state(TaskState.ROUTED) == [task.task_id]
    assert store.summary()[0]["state"] == "routed"
    store.close()
