"""Offline run verification (backlog F14): `council verify` needs no model, network or GPU."""

from __future__ import annotations

import json
import socket
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from axiom_council import cli
from axiom_council.configstore import ConfigStore
from axiom_council.contracts import RiskClass, TaskSpec, TaskState
from axiom_council.council import Council, open_registry
from axiom_council.provenance import config_hash, harness_hash
from axiom_council.registry import Registry
from axiom_council.verify import Status, VerifyReport, verify_run

from .conftest import CONFIG_DIR
from .test_council import _factory, _ok


@dataclass(frozen=True)
class Run:
    home: Path
    task_id: str
    registry: Registry


def _gated_task(make_task: Callable[..., TaskSpec]) -> TaskSpec:
    return make_task(workspace_id="pagetrace", risk_class=RiskClass.MEDIUM)


def _hold(home: Path, make_task: Callable[..., TaskSpec]) -> tuple[Council, str, str]:
    """A task parked for approval; returns the open council, task id and inbox item."""
    council = Council(open_registry(home, CONFIG_DIR), home, _factory(_ok))
    outcome = council.run(_gated_task(make_task), "Answer from evidence", output_tokens=256)
    assert outcome.state is TaskState.AWAITING_APPROVAL and outcome.inbox_item
    assert outcome.decision is not None
    return council, outcome.decision.task_id, outcome.inbox_item


@pytest.fixture
def run(tmp_path: Path, make_task: Callable[..., TaskSpec]) -> Run:
    home = tmp_path / "home"
    council, task_id, item = _hold(home, make_task)
    council.inbox.approve(item, "director", timedelta(hours=1), "looks right")
    assert council.resume(task_id).state is TaskState.COMPLETED
    registry = council.registry
    council.close()
    return Run(home, task_id, registry)


def _by(report: VerifyReport, name: str) -> list[Any]:
    return [c for c in report.checks if c.name == name]


def test_approved_run_verifies(run: Run) -> None:
    report = verify_run(run.home)
    assert report.passed, report.checks
    assert [c.status for c in _by(report, "audit")] == [Status.PASS]
    assert [c.subject for c in _by(report, "approval")] == [run.task_id]
    assert [c.status for c in _by(report, "placement")] == [Status.PASS]


def test_missing_home_fails(tmp_path: Path) -> None:
    report = verify_run(tmp_path / "nowhere")
    assert not report.passed
    assert report.checks[0].status is Status.FAIL


def test_truncated_audit_fails(run: Run) -> None:
    log = run.home / "audit.jsonl"
    log.write_bytes(b"".join(log.read_bytes().splitlines(keepends=True)[:-1]))
    report = verify_run(run.home)
    assert not report.passed
    audit = _by(report, "audit")[0]
    assert audit.status is Status.FAIL and "shorter than" in audit.reason


def test_placement_that_differs_from_the_pin_fails(run: Run) -> None:
    with sqlite3.connect(run.home / "tasks.sqlite3") as db:
        db.execute("UPDATE tasks SET pinned_node = 'other-node' WHERE task_id = ?", (run.task_id,))
    report = verify_run(run.home)
    placement = _by(report, "placement")[0]
    assert placement.status is Status.FAIL and "other-node" in placement.reason


def test_token_bound_to_another_artifact_fails(run: Run) -> None:
    path = next((run.home / "inbox").glob("*.json"))
    item = json.loads(path.read_text())
    item["token"]["artifact_hash"] = "f" * 64
    path.write_text(json.dumps(item))
    approval = _by(verify_run(run.home), "approval")[0]
    assert approval.status is Status.FAIL and "artifact" in approval.reason


def test_run_without_resume_fails(tmp_path: Path, make_task: Callable[..., TaskSpec]) -> None:
    home = tmp_path / "home"
    council, task_id, _ = _hold(home, make_task)
    council.audit.append("inference_completed", task_id, "mac-m5/qwen3:4b")
    council.close()
    approval = _by(verify_run(home), "approval")[0]
    assert approval.status is Status.FAIL and "task_resumed" in approval.reason


def _legacy_resume(
    home: Path, make_task: Callable[..., TaskSpec], ttl: timedelta = timedelta(hours=1)
) -> str:
    """Resume the way the council did before it recorded the artifact in the audit log."""
    council, task_id, item = _hold(home, make_task)
    token = council.inbox.approve(item, "director", ttl, "looks right")
    record = council.tasks.get(task_id)
    council.tasks.transition(task_id, TaskState.APPROVED)
    council.audit.append("task_resumed", task_id, "director", token=token.token_id)
    chosen = {"model_id": record.pinned_model, "node_id": record.pinned_node}
    council.audit.append("routed", task_id, "router", pinned=True, chosen=chosen, rejected=[])
    council.tasks.transition(task_id, TaskState.RUNNING)
    actor = f"{record.pinned_node}/{record.pinned_model}"
    council.audit.append("inference_completed", task_id, actor)
    council.tasks.transition(task_id, TaskState.COMPLETED)
    council.close()
    return task_id


def test_legacy_resume_is_checked_against_the_config_of_its_time(
    tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    home = tmp_path / "home"
    _legacy_resume(home, make_task)
    store = ConfigStore(home / "config.sqlite3")
    card = store.get("model", "qwen3:4b")
    assert card is not None
    store.put("model", "qwen3:4b", {**card.body, "digest": "a" * 64})  # a later config change
    store.close()
    report = verify_run(home)
    assert _by(report, "approval")[0].status is Status.PASS, report.checks


def test_legacy_resume_without_config_cannot_resolve(
    tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    home = tmp_path / "home"
    _legacy_resume(home, make_task)
    (home / "config.sqlite3").unlink()
    report = verify_run(home)
    assert _by(report, "approval")[0].status is Status.CANNOT_RESOLVE
    assert not report.passed


def test_expired_token_fails(tmp_path: Path, make_task: Callable[..., TaskSpec]) -> None:
    home = tmp_path / "home"
    _legacy_resume(home, make_task, ttl=timedelta(seconds=-1))
    approval = _by(verify_run(home), "approval")[0]
    assert approval.status is Status.FAIL and "not valid" in approval.reason


def _row(registry: Registry, **prov: Any) -> dict[str, Any]:
    provenance = {
        "run_id": "results-x",
        "config_hash": config_hash(registry),
        "harness_hash": harness_hash(registry),
        "model_id": "qwen3:4b",
        "model_digest": registry.models["qwen3:4b"].digest,
        **prov,
    }
    return {"model_id": "qwen3:4b", "fixture_id": "f1", "passed": True, "provenance": provenance}


def _results(home: Path, rows: list[dict[str, Any] | str]) -> Path:
    path = home / "bakeoff" / "results-x.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = [r if isinstance(r, str) else json.dumps(r) for r in rows]
    path.write_text("\n".join(text) + "\n")
    return path


def test_results_with_provenance_verify(run: Run) -> None:
    path = _results(run.home, [_row(run.registry)])
    report = verify_run(run.home, [path])
    assert [c.status for c in _by(report, "provenance")] == [Status.PASS], report.checks
    assert report.passed


@pytest.mark.parametrize(
    ("prov", "status", "words"),
    [
        ({"config_hash": "abc"}, Status.FAIL, "config_hash"),
        ({"harness_hash": "Z" * 64}, Status.FAIL, "harness_hash"),
        ({"model_digest": "b" * 64}, Status.FAIL, "digest"),
        ({"config_hash": "c" * 64}, Status.CANNOT_RESOLVE, "config hash"),
        ({"model_id": "ghost:1b"}, Status.CANNOT_RESOLVE, "ghost:1b"),
    ],
)
def test_bad_provenance_is_reported(
    run: Run, prov: dict[str, Any], status: Status, words: str
) -> None:
    path = _results(run.home, [_row(run.registry, **prov)])
    check = _by(verify_run(run.home, [path]), "provenance")[0]
    assert check.status is status and words in check.reason


def test_harness_mismatch_fails(run: Run) -> None:
    store = ConfigStore(run.home / "config.sqlite3")
    old_harness = harness_hash(run.registry)
    store.put("setting", "router", {"name": "router", "min_success": 0.6})
    changed = Registry.from_store(store)
    store.close()
    path = _results(run.home, [_row(changed, harness_hash=old_harness)])
    check = _by(verify_run(run.home, [path]), "provenance")[0]
    assert check.status is Status.FAIL and "harness" in check.reason


def test_card_without_digest_cannot_resolve(run: Run) -> None:
    store = ConfigStore(run.home / "config.sqlite3")
    card = store.get("model", "qwen3:4b")
    assert card is not None
    store.put("model", "qwen3:4b", {k: v for k, v in card.body.items() if k != "digest"})
    changed = Registry.from_store(store)
    store.close()
    path = _results(run.home, [_row(changed, model_digest=None)])
    check = _by(verify_run(run.home, [path]), "provenance")[0]
    assert check.status is Status.CANNOT_RESOLVE and "pins no digest" in check.reason


def test_legacy_rows_fail_strict_and_warn_with_allow_legacy(run: Run) -> None:
    legacy = {"model_id": "qwen3:4b", "fixture_id": "f1", "passed": True}
    path = _results(run.home, [legacy])
    strict = verify_run(run.home, [path])
    check = _by(strict, "provenance")[0]
    assert check.status is Status.NO_PROVENANCE and check.blocking and not strict.passed
    lenient = verify_run(run.home, [path], allow_legacy=True)
    assert not _by(lenient, "provenance")[0].blocking and lenient.passed


def test_unreadable_results_fail(run: Run) -> None:
    path = _results(run.home, ["{broken", "[1]", json.dumps({"provenance": "nope"})])
    checks = _by(verify_run(run.home, [path, run.home / "missing.jsonl"]), "provenance")
    assert [c.status for c in checks] == [Status.FAIL] * 4


@pytest.fixture
def env(run: Run, monkeypatch: pytest.MonkeyPatch) -> Run:
    monkeypatch.setenv("AXIOM_COUNCIL_HOME", str(run.home))
    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(CONFIG_DIR))
    return run


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("verify must not open a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr("axiom_council.providers.provider_for", refuse)


def test_cli_verify_is_offline_and_finds_results(
    env: Run, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _results(env.home, [_row(env.registry)])
    _no_network(monkeypatch)
    assert cli.main(["verify"]) == 0
    out = capsys.readouterr().out
    assert "results-x.jsonl:1" in out and "PASSED" in out


def test_cli_verify_json_and_exit_codes(env: Run, capsys: pytest.CaptureFixture[str]) -> None:
    path = _results(env.home, [{"model_id": "qwen3:4b"}])
    assert cli.main(["verify", "--json", "--results", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is False
    assert {"name", "subject", "status", "blocking", "reason"} <= report["checks"][0].keys()
    assert cli.main(["verify", "--allow-legacy", "--results", str(path)]) == 0
    assert "warn" in capsys.readouterr().out


def _edit_item(home: Path, edit: Callable[[dict[str, Any]], None]) -> None:
    path = next((home / "inbox").glob("*.json"))
    item = json.loads(path.read_text())
    edit(item)
    path.write_text(json.dumps(item))


def _set_token(field: str, value: str) -> Callable[[dict[str, Any]], None]:
    return lambda item: item["token"].__setitem__(field, value)


def _rebind(item: dict[str, Any]) -> None:
    item["artifact_hash"] = item["token"]["artifact_hash"] = "f" * 64


@pytest.mark.parametrize(
    ("edit", "words"),
    [
        (lambda item: item.__setitem__("status", "pending"), "not approved"),
        (lambda item: item.__setitem__("action", "override_routing"), "not an execute"),
        (_set_token("task_id", "task_other"), "is for task_other"),
        (_set_token("token_id", "appr_other"), "the run used token"),
        (_rebind, "the run used artifact"),
        (lambda item: item.__setitem__("kind", "bogus"), "unreadable"),
    ],
)
def test_tampered_approval_fails(
    run: Run, edit: Callable[[dict[str, Any]], None], words: str
) -> None:
    _edit_item(run.home, edit)
    approval = _by(verify_run(run.home), "approval")[0]
    assert approval.status is Status.FAIL and words in approval.reason, approval


def test_legacy_resume_with_a_rebound_token_fails(
    tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    home = tmp_path / "home"
    _legacy_resume(home, make_task)
    _edit_item(home, _rebind)
    approval = _by(verify_run(home), "approval")[0]
    assert approval.status is Status.FAIL and "the config at" in approval.reason


def test_unreadable_task_record_fails_both_checks(run: Run) -> None:
    with sqlite3.connect(run.home / "tasks.sqlite3") as db:
        db.execute("UPDATE tasks SET spec_digest = 'x' WHERE task_id = ?", (run.task_id,))
    checks = [c for c in verify_run(run.home).checks if c.subject == run.task_id]
    assert [c.status for c in checks] == [Status.FAIL, Status.FAIL]
    assert "no longer matches" in checks[0].reason


def test_missing_pin_and_missing_inbox_fail(run: Run) -> None:
    with sqlite3.connect(run.home / "tasks.sqlite3") as db:
        db.execute("UPDATE tasks SET pinned_model = NULL WHERE task_id = ?", (run.task_id,))
    for path in (run.home / "inbox").glob("*.json"):
        path.unlink()
    report = verify_run(run.home)
    assert "no pinned placement" in _by(report, "placement")[0].reason
    assert "unreadable" in _by(report, "approval")[0].reason


def test_failed_run_after_approval_is_checked(
    tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    import httpx

    home = tmp_path / "home"
    council, task_id, item = _hold(home, make_task)
    council.inbox.approve(item, "director", timedelta(hours=1), "looks right")
    council.close()
    failing = Council(
        open_registry(home, CONFIG_DIR), home, _factory(lambda r: httpx.Response(500))
    )
    assert failing.resume(task_id).state is TaskState.FAILED
    failing.close()
    report = verify_run(home)
    assert report.passed, report.checks
    assert "records name the pin" in _by(report, "placement")[0].reason


def test_interrupted_run_cannot_resolve_its_placement(
    tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    home = tmp_path / "home"
    council, task_id, item = _hold(home, make_task)
    token = council.inbox.approve(item, "director", timedelta(hours=1), "looks right")
    council.tasks.transition(task_id, TaskState.APPROVED)
    council.audit.append("task_resumed", task_id, "director", token=token.token_id)
    council.tasks.transition(task_id, TaskState.RUNNING)  # the process dies here
    council.close()
    Council(open_registry(home, CONFIG_DIR), home, _factory(_ok)).close()  # recovery
    report = verify_run(home)
    placement = _by(report, "placement")[0]
    assert placement.status is Status.CANNOT_RESOLVE and "interrupted" in placement.reason
    assert _by(report, "approval")[0].status is Status.PASS
