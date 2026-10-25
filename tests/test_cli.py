from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_council import cli

from .conftest import CONFIG_DIR


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXIOM_COUNCIL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(CONFIG_DIR))


def test_config_seed_and_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["config", "seed"]) == 0
    assert json.loads(capsys.readouterr().out)["model"] >= 3
    assert cli.main(["config", "list"]) == 0
    assert "gpt-oss:20b" in capsys.readouterr().out


def test_route_explains_choice(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["route", "Plan the PageTrace adapter", "--task-class", "plan"]) == 0
    out = capsys.readouterr().out
    assert "REJECTED  qwen3:4b" in out and "escalate" not in out.splitlines()[-1]


def test_ask_without_eligible_model_goes_to_inbox(capsys: pytest.CaptureFixture[str]) -> None:
    # A token budget smaller than the output allowance leaves no eligible placement.
    assert cli.main(["ask", "Draft release notes for R0", "--max-tokens", "100"]) == 0
    out = capsys.readouterr().out
    assert "awaiting_clarification" in out and "waiting on director" in out


def test_ask_rejected_by_policy(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["ask", "Summarise a paper", "--workspace", "nowhere"]) == 1
    assert "unknown workspace" in capsys.readouterr().out


def test_inbox_round_trip_and_audit(capsys: pytest.CaptureFixture[str]) -> None:
    from axiom_council.council import open_registry
    from axiom_council.inbox import Inbox

    home = Path(cli._home())
    open_registry(home, CONFIG_DIR)
    item = Inbox(home / "inbox").request_approval(
        task_id="t1", action="execute", artifact_hash="d" * 64, summary="demo"
    )
    assert cli.main(["inbox", "list"]) == 0
    assert item.item_id in capsys.readouterr().out
    assert cli.main(["inbox", "show", item.item_id]) == 0
    assert cli.main(["inbox", "approve", item.item_id, "--text", "fine by me"]) == 0
    assert "approved" in capsys.readouterr().out
    other = Inbox(home / "inbox").request_approval(
        task_id="t2", action="execute", artifact_hash="e" * 64, summary="demo"
    )
    assert cli.main(["inbox", "reject", other.item_id, "--text", "not now"]) == 0
    assert cli.main(["audit"]) == 0
    assert '"chain": "intact"' in capsys.readouterr().out.splitlines()[-1]
    assert cli.main(["inbox", "show"]) == 2


def test_nodes_reports_down_servers(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seed = tmp_path / "seed"
    seed.mkdir()
    for f in CONFIG_DIR.glob("*.yaml"):
        text = f.read_text().replace("http://127.0.0.1:11434", "http://127.0.0.1:9")
        (seed / f.name).write_text(text)
    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(seed))
    assert cli.main(["nodes"]) == 0
    assert "down" in capsys.readouterr().out


def test_tasks_and_resume_commands(capsys: pytest.CaptureFixture[str]) -> None:
    from axiom_council.contracts import Budget, Capability, RiskClass, TaskClass, TaskSpec
    from axiom_council.council import Council, open_registry

    home = Path(cli._home())
    council = Council(open_registry(home, CONFIG_DIR), home)
    task = TaskSpec(
        workspace_id="pagetrace",
        objective="Summarise the evaluation module",
        task_class=TaskClass.TRIAGE,
        actor="director",
        risk_class=RiskClass.MEDIUM,
        allowed_capabilities=frozenset({Capability.LLM_INFERENCE}),
        budget=Budget(max_tokens=16_000, max_wall_seconds=60),
        acceptance_checks=("review",),
    )
    held = council.run(task, "Summarise please")
    council.close()
    assert cli.main(["tasks"]) == 0
    assert "awaiting_approval" in capsys.readouterr().out
    assert cli.main(["resume", task.task_id]) == 1  # not approved yet
    assert "refused" in capsys.readouterr().err
    assert held.inbox_item
    assert cli.main(["inbox", "reject", held.inbox_item, "--text", "not now"]) == 0
    assert cli.main(["resume", task.task_id]) == 0
    assert "state: rejected" in capsys.readouterr().out
