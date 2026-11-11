"""Offline checks for tasks that ran after a director approval.

For each such task, the approval check shows that an approved inbox item holds a token for this
task and action. The token must bind the artifact hash that the run used, and be valid at the
time of the resume. The placement check shows that every record of the run names the pinned
placement. Both checks read the audit log, the task store and the inbox only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import ValidationError

from axiom_council.contracts import TaskState
from axiom_council.council import approval_artifact
from axiom_council.inbox import Inbox, InboxItem
from axiom_council.tasks import TaskRecord, TaskStateError, TaskStore
from axiom_council.verify_checks import Check, ConfigHistory, Status, failed, passed

INFERENCE = frozenset({"inference_completed", "inference_failed"})
_RAN = (TaskState.RUNNING, TaskState.COMPLETED, TaskState.FAILED)
Event = dict[str, Any]


@dataclass(frozen=True)
class Evidence:
    tasks: TaskStore | None
    inbox: Inbox | None
    configs: ConfigHistory


def _item(inbox: Inbox | None, item_id: str | None) -> InboxItem | str:
    """The inbox item, or the reason it cannot be read."""
    if inbox is None or item_id is None:
        return "the task has no approval item in the inbox"
    try:
        return inbox.get(item_id)
    except (OSError, ValueError, ValidationError) as exc:
        return f"inbox item {item_id} is unreadable: {exc}"


def _is_execute_request(inbox: Inbox | None, event: Event) -> bool:
    """True for a director request whose item is an execute approval, or cannot be read."""
    if event["event"] != "director_requested":
        return False
    item = _item(inbox, event.get("data", {}).get("item"))
    return isinstance(item, str) or (item.kind == "approval" and item.action == "execute")


def _store_gated(evidence: Evidence) -> set[str]:
    """Tasks the store shows as run (or running) after an execute approval was requested."""
    found: set[str] = set()
    if evidence.tasks is None:
        return found
    for state in _RAN:
        for task_id in evidence.tasks.in_state(state):
            try:
                record = evidence.tasks.get(task_id)
            except TaskStateError:
                found.add(task_id)  # an unreadable record is reported, never skipped
                continue
            if record.inbox_item is not None:
                item = _item(evidence.inbox, record.inbox_item)
                if isinstance(item, str) or item.action == "execute":
                    found.add(task_id)
    return found


def _gated_runs(events: dict[str, list[Event]], evidence: Evidence) -> set[str]:
    found = _store_gated(evidence)
    for task_id, task_events in events.items():
        names = {e["event"] for e in task_events}
        if not names & INFERENCE:
            continue
        if "task_resumed" in names or any(
            _is_execute_request(evidence.inbox, e) for e in task_events
        ):
            found.add(task_id)
    return found


def _resume_index(events: list[Event]) -> int | None:
    """Index of the last task_resumed event before the first inference event."""
    first_run = next((i for i, e in enumerate(events) if e["event"] in INFERENCE), len(events))
    resumes = [i for i, e in enumerate(events[:first_run]) if e["event"] == "task_resumed"]
    return resumes[-1] if resumes else None


def _token_problem(task_id: str, item: InboxItem, resumed: Event) -> str | None:
    token = item.token
    if item.kind != "approval" or item.action != "execute":
        return f"inbox item {item.item_id} is not an execute approval"
    if item.status != "approved" or token is None:
        return f"inbox item {item.item_id} is {item.status}, not approved"
    if token.task_id != task_id or token.action != "execute":
        return f"token {token.token_id} is for {token.task_id}/{token.action}"
    if token.artifact_hash != item.artifact_hash:
        return (
            f"token artifact {token.artifact_hash[:12]} differs from the requested artifact"
            f" {str(item.artifact_hash)[:12]}"
        )
    used = resumed.get("data", {}).get("token")
    if used != token.token_id:
        return f"the run used token {used}, the inbox holds {token.token_id}"
    at = datetime.fromisoformat(resumed["ts"])
    if not token.issued_at <= at < token.expires_at:
        return f"token {token.token_id} was not valid at the resume time {resumed['ts']}"
    return None


def _legacy_artifact(
    task_id: str, record: TaskRecord, resumed: Event, token_hash: str, configs: ConfigHistory
) -> Check:
    """For a resume record without an artifact: recompute it from the config of that time."""
    when = resumed["ts"]
    registry = configs.at(when)
    if registry is None or record.pinned_model is None or record.pinned_node is None:
        reason = f"no config or pinned placement is known for the resume at {when}"
        return failed("approval", task_id, reason, Status.CANNOT_RESOLVE)
    try:
        expected = approval_artifact(registry, record.spec, record.pinned_model, record.pinned_node)
    except KeyError:
        reason = f"the pinned placement is not in the config active at {when}"
        return failed("approval", task_id, reason, Status.CANNOT_RESOLVE)
    if expected != token_hash:
        reason = f"the token binds {token_hash[:12]}, the config at {when} gives {expected[:12]}"
        return failed("approval", task_id, reason)
    return passed("approval", task_id, f"token binds {token_hash[:12]}, recomputed from {when}")


def _approval(task_id: str, record: TaskRecord, resumed: Event | None, ev: Evidence) -> Check:
    if resumed is None:
        return failed("approval", task_id, "the task ran without a task_resumed event")
    item = _item(ev.inbox, record.inbox_item)
    if isinstance(item, str):
        return failed("approval", task_id, item)
    problem = _token_problem(task_id, item, resumed)
    if problem is not None or item.token is None:
        return failed("approval", task_id, problem or "no token")
    token = item.token
    used = resumed.get("data", {}).get("artifact")
    if used is None:
        return _legacy_artifact(task_id, record, resumed, token.artifact_hash, ev.configs)
    if used != token.artifact_hash:
        reason = (
            f"the run used artifact {str(used)[:12]}, the token binds {token.artifact_hash[:12]}"
        )
        return failed("approval", task_id, reason)
    reason = f"token {token.token_id} binds artifact {token.artifact_hash[:12]}, used by the run"
    return passed("approval", task_id, reason)


def _placements(events: list[Event]) -> list[tuple[str, str | None, str | None]]:
    """(source, model, node) for each record that names where the task ran."""
    found: list[tuple[str, str | None, str | None]] = []
    for e in events:
        data = e.get("data", {})
        if e["event"] == "task_resumed" and "node" in data:
            found.append(("task_resumed", data.get("model"), data.get("node")))
        elif e["event"] == "routed" and data.get("pinned") and data.get("chosen"):
            chosen = data["chosen"]
            found.append(("routed", chosen.get("model_id"), chosen.get("node_id")))
        elif e["event"] == "inference_completed":
            node, _, model = str(e["actor"]).partition("/")
            found.append(("inference_completed", model, node))
        elif e["event"] == "inference_failed":
            found.append(("inference_failed", None, str(e["actor"])))
    return found


def _placement(task_id: str, record: TaskRecord, events: list[Event], start: int) -> Check:
    pin_model, pin_node = record.pinned_model, record.pinned_node
    if pin_model is None or pin_node is None:
        return failed("placement", task_id, "the task has no pinned placement")
    seen = _placements(events[start:])
    if not any(source.startswith("inference") for source, _, _ in seen):
        reason = "no inference record after the approval (interrupted?)"
        return failed("placement", task_id, reason, Status.CANNOT_RESOLVE)
    for source, model, node in seen:
        if node != pin_node or (model is not None and model != pin_model):
            reason = f"{source} names {model} @ {node}, the pin is {pin_model} @ {pin_node}"
            return failed("placement", task_id, reason)
    return passed(
        "placement", task_id, f"{len(seen)} records name the pin {pin_model} @ {pin_node}"
    )


def _check_task(task_id: str, events: list[Event], ev: Evidence) -> list[Check]:
    try:
        record = ev.tasks.get(task_id) if ev.tasks is not None else None
    except TaskStateError as exc:
        record, missing = None, str(exc)
    else:
        missing = "the task store has no record of this task"
    if record is None:
        return [failed("approval", task_id, missing), failed("placement", task_id, missing)]
    index = _resume_index(events)
    resumed = None if index is None else events[index]
    return [
        _approval(task_id, record, resumed, ev),
        _placement(task_id, record, events, 0 if index is None else index),
    ]


def approval_checks(records: list[Event], evidence: Evidence) -> list[Check]:
    """An approval and a placement check for every task that ran after an approval."""
    events: dict[str, list[Event]] = defaultdict(list)
    for record in records:
        events[record["task_id"]].append(record)
    checks: list[Check] = []
    for task_id in sorted(_gated_runs(events, evidence)):
        checks += _check_task(task_id, events.get(task_id, []), evidence)
    return checks
