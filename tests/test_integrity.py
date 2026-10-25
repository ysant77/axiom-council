"""Regression tests for the routing-evidence defects in the 2026-10-25 defect register.

D-01: an approved task runs on exactly the placement the director approved.
D-02: repeated benchmark runs do not multiply the weight of the same item.
D-03: holdout results never reach the routing ledger.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from axiom_council.contracts import RiskClass, TaskClass, TaskSpec, TaskState
from axiom_council.council import Council
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry

from .test_council import _factory, _ok


def _held(registry: Registry, home: Path, make_task: Callable[..., TaskSpec]) -> tuple[str, str]:
    council = Council(registry, home, _factory(lambda r: pytest.fail("ran before approval")))
    task = make_task(workspace_id="pagetrace", risk_class=RiskClass.MEDIUM)
    held = council.run(task, "Summarise the evaluation module")
    assert held.state is TaskState.AWAITING_APPROVAL and held.inbox_item and held.decision
    assert held.decision.chosen is not None
    council.inbox.approve(held.inbox_item, "director", timedelta(hours=1), "looks right")
    council.close()
    return task.task_id, held.decision.chosen.model_id


def _capture() -> tuple[list[str], Callable[[httpx.Request], httpx.Response]]:
    models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return _ok(request)

    return models, handler


def _fail_model(home: Path, model_id: str, task_class: TaskClass, count: int = 40) -> None:
    ledger = CapabilityLedger(home / "ledger.sqlite3")
    for i in range(count):
        ledger.record(task_id=f"fail-{i}", model_id=model_id, node_id="mac-m5",
                      task_class=task_class, success=False, verified_by="pytest",
                      wall_seconds=1.0, total_tokens=10)  # fmt: skip
    ledger.close()


# --- D-01 -------------------------------------------------------------------------------


def test_resume_runs_the_approved_model_even_if_routing_now_prefers_another(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    task_id, approved = _held(registry, tmp_path, make_task)
    _fail_model(tmp_path, approved, TaskClass.TRIAGE)  # routing would now reject it
    models, handler = _capture()
    council = Council(registry, tmp_path, _factory(handler))
    outcome = council.resume(task_id)
    council.close()
    assert outcome.state is TaskState.COMPLETED and models == [approved]


def test_resume_refuses_if_the_approved_model_config_changed(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    task_id, approved = _held(registry, tmp_path, make_task)
    card = registry.models[approved]
    changed = card.model_copy(update={"request_defaults": {"temperature_override": 1.5}})
    edited = registry.model_copy(update={"models": {**registry.models, approved: changed}})
    council = Council(edited, tmp_path, _factory(lambda r: pytest.fail("ran")))
    with pytest.raises(PermissionError, match="no valid approval"):
        council.resume(task_id)
    council.close()


def test_resume_refuses_if_the_approved_node_config_changed(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    task_id, _ = _held(registry, tmp_path, make_task)
    node = registry.nodes["mac-m5"].model_copy(update={"enabled": False})
    edited = registry.model_copy(update={"nodes": {**registry.nodes, "mac-m5": node}})
    council = Council(edited, tmp_path, _factory(lambda r: pytest.fail("ran")))
    with pytest.raises(PermissionError, match="no valid approval"):
        council.resume(task_id)
    council.close()


def test_resume_holds_instead_of_rerouting_when_the_placement_fails_its_recheck(
    registry: Registry,
    tmp_path: Path,
    make_task: Callable[..., TaskSpec],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id, approved = _held(registry, tmp_path, make_task)
    council = Council(registry, tmp_path, _factory(lambda r: pytest.fail("ran")))
    original = council.router.check_placement

    def too_big(*args: Any, **kwargs: Any) -> Any:
        return original(*args, **{**kwargs, "prompt_tokens": 10_000_000})

    monkeypatch.setattr(council.router, "check_placement", too_big)
    outcome = council.resume(task_id)
    council.close()
    assert outcome.state is TaskState.AWAITING_CLARIFICATION and outcome.inbox_item
    assert outcome.decision is not None and outcome.decision.rejected[0].model_id == approved


def test_unknown_or_unavailable_placement_is_rejected(
    registry: Registry, tmp_path: Path, make_task: Callable[..., TaskSpec]
) -> None:
    from axiom_council.router import Router

    router = Router(registry, CapabilityLedger(tmp_path / "l.sqlite3"))
    for model, node in (
        ("ghost:1", "mac-m5"),
        ("qwen3:4b", "ghost-node"),
        ("qwen3:4b", "win-4060"),
    ):
        decision = router.check_placement(make_task(), model, node, prompt_tokens=10)
        assert decision.chosen is None and decision.rejected[0].reason == "placement unavailable"
    router.ledger.close()


# --- D-02 -------------------------------------------------------------------------------


def _bench(ledger: CapabilityLedger, run_id: str, item: str, success: bool, **kw: Any) -> None:
    ledger.record(task_id=f"{run_id}-{item}", model_id="m", node_id="n",
                  task_class=TaskClass.PLAN, success=success, verified_by="scorer",
                  wall_seconds=1.0, total_tokens=10, run_id=run_id, item_key=item,
                  **kw)  # fmt: skip


def test_repeated_runs_count_each_benchmark_item_once(tmp_path: Path) -> None:
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    for run in ("run-1", "run-2", "run-3"):
        _bench(ledger, run, "plan-a", success=False, split="calibration")
    _bench(ledger, "run-4", "plan-a", success=True, split="calibration")
    _bench(ledger, "run-4", "plan-b", success=False, split="calibration")
    stats = ledger.stats("m", TaskClass.PLAN)
    assert (stats.successes, stats.failures) == (1, 1)  # latest result per item
    with pytest.raises(ValueError, match="already recorded"):
        _bench(ledger, "run-4", "plan-a", success=True, split="calibration")
    with pytest.raises(ValueError, match="run_id"):
        ledger.record(task_id="x", model_id="m", node_id="n", task_class=TaskClass.PLAN,
                      success=True, verified_by="scorer", wall_seconds=1.0, total_tokens=1,
                      item_key="plan-c", split="calibration")  # fmt: skip
    ledger.close()


def test_legacy_rows_are_migrated_to_one_vote_per_item(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "ledger.sqlite3"
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,"
        " task_id TEXT NOT NULL, model_id TEXT NOT NULL, node_id TEXT NOT NULL,"
        " task_class TEXT NOT NULL, success INTEGER NOT NULL, verified_by TEXT NOT NULL,"
        " wall_seconds REAL NOT NULL, total_tokens INTEGER NOT NULL)"
    )
    rows = [("bakeoff-plan-a", 0)] * 3 + [("bakeoff-plan-a", 1), ("live-task", 1)]
    with db:
        for task_id, ok in rows:
            db.execute(
                "INSERT INTO outcomes (ts, task_id, model_id, node_id, task_class, success,"
                " verified_by, wall_seconds, total_tokens) VALUES ('t', ?, 'm', 'n', 'plan',"
                " ?, 'scorer', 1.0, 1)",
                (task_id, ok),
            )
    db.close()
    ledger = CapabilityLedger(path)
    stats = ledger.stats("m", TaskClass.PLAN)
    ledger.close()
    assert (stats.successes, stats.failures) == (2, 0)  # latest bake-off vote + live task


# --- D-03 -------------------------------------------------------------------------------


def test_holdout_never_enters_the_ledger_and_development_does_not_route(tmp_path: Path) -> None:
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    with pytest.raises(ValueError, match="holdout"):
        _bench(ledger, "run-1", "plan-h", success=True, split="holdout")
    with pytest.raises(ValueError, match="split"):
        _bench(ledger, "run-1", "plan-x", success=True, split="production")
    _bench(ledger, "run-1", "plan-d", success=True, split="development")
    assert ledger.stats("m", TaskClass.PLAN).trials == 0  # development is for debugging only
    _bench(ledger, "run-1", "plan-c", success=True, split="calibration")
    assert ledger.stats("m", TaskClass.PLAN).trials == 1
    ledger.close()


def test_bakeoff_labels_runs_and_keeps_holdout_out_of_routing(
    registry: Registry, tmp_path: Path
) -> None:
    from axiom_council.energy import EstimateMeter
    from axiom_council.evals.bakeoff import Bakeoff, Contestant
    from axiom_council.evals.fixtures import load_fixtures

    from .test_bakeoff import GOOD_PLAN, REPO, FakeChat

    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    for run, split in (("results-a", "calibration"), ("results-b", "calibration"),
                       ("results-c", "holdout")):  # fmt: skip
        chat = FakeChat({"plan": json.dumps(GOOD_PLAN), "review": '{"verdict": "approve"}'})
        bake = Bakeoff(registry, fixtures, chat, ledger, tmp_path / f"{run}.jsonl", "mac-m5",  # type: ignore[arg-type]
                       EstimateMeter(30), split=split)  # fmt: skip
        bake.run([Contestant("qwen3:4b")])
    stats = ledger.stats("qwen3:4b", TaskClass.PLAN)
    ledger.close()
    assert stats.trials == len(fixtures.plan)  # two calibration runs, one vote per fixture
    rows = [json.loads(line) for line in (tmp_path / "results-c.jsonl").read_text().splitlines()]
    assert {r["split"] for r in rows} == {"holdout"} and rows[0]["run_id"] == "results-c"
