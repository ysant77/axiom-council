from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from axiom_council import cli
from axiom_council.evals import session
from axiom_council.evals.bakeoff import Bakeoff, CodeContext, Contestant, summarise
from axiom_council.evals.code_task import verify_code
from axiom_council.evals.fixtures import CodeFixture, load_fixtures
from axiom_council.evals.ollama_metrics import ChatMetrics, ChatReply, OllamaChat
from axiom_council.evals.session import parse_contestants
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry
from axiom_council.sandbox import DockerSandbox, create_snapshot
from axiom_council.sandbox.runner import Completed

from .sandbox_helpers import FakeRunner, make_repo, profile

REPO = Path(__file__).resolve().parents[1]
GOOD_PLAN = {
    "milestones": [{"name": "a", "goal": "measure baseline latency, test, review behaviour"}],
    "team": [
        {"role": "dev", "model": "qwen3:4b"},
        {"role": "qa mlops design", "model": "qwen3:4b"},
    ],
    "risks": [
        "capability gap: speech, render, private tailscale, health, vram, citation,"
        " unsupported, eval, held-out, regress"
    ],
    "estimate_hours": 1,
    "clarifying_questions": ["which?"],
}


class FakeChat:
    def __init__(self, replies: dict[str, str]):
        self.replies = replies
        self.unloaded: list[str] = []
        self.calls: list[tuple[str, Any]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> ChatReply:
        self.calls.append((model, kwargs["think"]))
        text = messages[-1]["content"]
        content = self.replies["review"] if "UNTRUSTED DIFF" in text else self.replies["plan"]
        load = 5.0 if len(self.calls) == 1 else 0.0
        return ChatReply(content, "", ChatMetrics(load, 100, 0.5, 50, 2.0, 3.0 + load), "stop")

    def unload(self, model: str) -> None:
        self.unloaded.append(model)


def test_bakeoff_scores_records_and_summarises(registry: Registry, tmp_path: Path) -> None:
    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    chat = FakeChat({"plan": json.dumps(GOOD_PLAN), "review": '{"verdict": "approve"}'})
    ledger = CapabilityLedger(tmp_path / "ledger.sqlite3")
    bake = Bakeoff(registry, fixtures, chat, ledger, tmp_path / "r.jsonl")  # type: ignore[arg-type]
    records = bake.run([Contestant("qwen3:4b", "low")])
    assert len(records) == 12 and records[0].cold and not records[1].cold
    assert chat.calls[0] == ("qwen3:4b", "low") and chat.unloaded == ["qwen3:4b"]
    reviews = [r for r in records if r.kind == "review"]
    assert sum(r.passed for r in reviews) == 2  # only the two clean diffs are approvals
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == 12
    assert json.loads((tmp_path / "r.jsonl").read_text().splitlines()[0])["output"]
    from axiom_council.contracts import TaskClass

    assert ledger.stats("qwen3:4b", TaskClass.REVIEW).trials == 6
    row = summarise(records)[0]
    assert row["review"] == "2/6" and row["cold_load_s"] == 5.0 and row["median_ttft_s"] == 0.5
    ledger.close()


def test_code_verification_in_sandbox(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    prof = profile(repo, checks={"lint": ("ruff",), "tests": ("pytest",)})
    base = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    fixture = CodeFixture(
        id="c", module="src/demo/core.py", function="add", max_lines=5, difficulty="easy"
    )
    ok = (
        "```python\ndef add(a, b):\n    return _plus(a, b)\n\n"
        "def _plus(a, b):\n    return a + b\n```"
    )
    green = DockerSandbox(prof, run=FakeRunner())
    assert verify_code(fixture, ok, base, green, tmp_path / "work", "r1").passed
    red = DockerSandbox(prof, run=FakeRunner([Completed(0, "", ""), Completed(1, "1 failed", "")]))
    failed = verify_code(fixture, ok, base, red, tmp_path / "work", "r2")
    assert not failed.passed and failed.checks["tests"] is False and "1 failed" in failed.note
    bad = verify_code(fixture, "def other(): pass", base, green, tmp_path / "work", "r3")
    assert bad.checks == {"splice": False}
    frozen = CodeFixture(
        id="f", module="tests/test_core.py", function="test_add", max_lines=5, difficulty="easy"
    )
    code = "def test_add():\n    assert True\n"
    assert (
        verify_code(frozen, code, base, green, tmp_path / "work", "r4").checks["edit_policy"]
        is False
    )


def test_bakeoff_with_code_context(registry: Registry, tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    base = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    prof = profile(repo, checks={"lint": ("ruff",), "tests": ("pytest",)})
    code_ctx = CodeContext(DockerSandbox(prof, run=FakeRunner()), base, tmp_path / "w")
    fixtures = load_fixtures(REPO / "evals" / "fixtures").model_copy(
        update={"plan": (), "review": (), "code": (
            CodeFixture(id="c", module="src/demo/core.py", function="add", max_lines=5,
                        difficulty="easy"),
        )}
    )  # fmt: skip
    chat = FakeChat({"plan": "```python\ndef add(a, b):\n    return b + a\n```", "review": ""})
    ledger = CapabilityLedger(tmp_path / "l.sqlite3")
    bake = Bakeoff(registry, fixtures, chat, ledger, tmp_path / "r.jsonl", code=code_ctx)  # type: ignore[arg-type]
    assert bake.run([Contestant("qwen3:4b")])[0].passed
    ledger.close()


def test_ollama_client_parses_metrics() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={
            "message": {"content": "hi", "thinking": "hmm"}, "done_reason": "stop",
            "load_duration": 2e9, "prompt_eval_count": 10, "prompt_eval_duration": 5e8,
            "eval_count": 40, "eval_duration": 2e9, "total_duration": 4.5e9,
        })  # fmt: skip

    chat = OllamaChat(transport=httpx.MockTransport(handler))
    reply = chat.chat("m", [], think="low", num_ctx=1024, num_predict=10, temperature=0)
    assert reply.metrics.ttft_seconds == 2.5 and reply.metrics.decode_tps == 20
    assert seen[0]["think"] == "low" and seen[0]["options"]["num_ctx"] == 1024
    chat.chat("m", [], think=None, num_ctx=1, num_predict=1, temperature=0)
    assert "think" not in seen[1]
    chat.unload("m")
    assert seen[2] == {"model": "m", "keep_alive": 0}
    assert ChatMetrics(0, 0, 0, 0, 0, 0).decode_tps == 0.0
    chat.close()


def test_contestant_parsing_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    parsed = parse_contestants("a:1=low, b:2=false,c:3=true,d:4")
    assert [(c.model_id, c.think) for c in parsed] == [
        ("a:1", "low"),
        ("b:2", False),
        ("c:3", True),
        ("d:4", None),
    ]
    monkeypatch.setenv("AXIOM_COUNCIL_HOME", str(tmp_path / "home"))
    from .conftest import CONFIG_DIR

    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(CONFIG_DIR))
    captured: dict[str, Any] = {}

    def fake_run(registry: Registry, home: Path, fixtures: Path, contestants: Any, **kw: Any):  # type: ignore[no-untyped-def]
        captured.update(kw, contestants=contestants)
        return [], home / "results.jsonl"

    monkeypatch.setattr(cli, "run_bakeoff", fake_run)
    assert cli.main(["bakeoff", "--models", "qwen3:4b=low", "--skip-code"]) == 0
    assert captured["with_code"] is False and "results:" in capsys.readouterr().out


def test_session_rejects_unknown_models(registry: Registry, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model card"):
        session.run_bakeoff(
            registry, tmp_path, REPO / "evals/fixtures", [Contestant("ghost:1")], with_code=False
        )


def test_session_runs_without_code(
    registry: Registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chat = FakeChat({"plan": "{}", "review": "{}"})
    monkeypatch.setattr(session, "OllamaChat", lambda url: chat)
    monkeypatch.setattr(chat, "close", lambda: None, raising=False)
    records, path = session.run_bakeoff(
        registry, tmp_path, REPO / "evals/fixtures", [Contestant("qwen3:4b")], with_code=False
    )
    assert len(records) == 12 and path.exists()
