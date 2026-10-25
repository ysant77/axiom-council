"""Repair mode: public-check failures are fed back for bounded retries; the hidden rubric
never is, and a failure on hidden checks alone gets no retry signal at all."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from axiom_council.energy import EstimateMeter
from axiom_council.evals.bakeoff import Bakeoff, Contestant
from axiom_council.evals.fixtures import CodeFixture, FixtureSet, PlanFixture, ReviewFixture
from axiom_council.evals.ollama_metrics import ChatMetrics, ChatReply
from axiom_council.evals.repair import feedback
from axiom_council.evals.scoring import Score, score_plan
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry

PLAN = PlanFixture(
    id="plan-x",
    request="Speed up indexing",
    candidates=("qwen3:4b", "gemma3:4b"),
    min_milestones=1,
    max_milestones=3,
    hours_budget=10,
    required_roles=("dev",),
    must_mention=(("secret-concept",),),
)
PUBLIC = frozenset({"valid_json", "within_hours", "models_from_candidates"})


def _plan(hours: float, model: str = "qwen3:4b", mention: str = "secret-concept") -> str:
    return json.dumps(
        {
            "milestones": [{"name": "m", "goal": mention}],
            "team": [{"role": "dev", "model": model}],
            "estimate_hours": hours,
            "clarifying_questions": [],
        }
    )


class ScriptedChat:
    def __init__(self, replies: list[str]):
        self._replies = replies
        self.conversations: list[list[dict[str, str]]] = []

    def chat(self, model: str, messages: list[dict[str, str]], **kwargs: Any) -> ChatReply:
        self.conversations.append(list(messages))
        content = self._replies[min(len(self.conversations), len(self._replies)) - 1]
        return ChatReply(content, "", ChatMetrics(0.0, 100, 0.5, 40, 2.0, 3.0), "stop")

    def unload(self, model: str) -> None:
        return None


def _bake(registry: Registry, tmp_path: Path, chat: ScriptedChat, repair: int) -> Bakeoff:
    return Bakeoff(
        registry,
        FixtureSet(plan=(PLAN,)),
        chat,  # type: ignore[arg-type]
        CapabilityLedger(tmp_path / "ledger.sqlite3"),
        tmp_path / "r.jsonl",
        "mac-m5",
        EstimateMeter(36.0),
        repair_attempts=repair,
    )


def test_feedback_names_public_failures_only() -> None:
    over_budget = score_plan(PLAN, _plan(50, mention="nothing"))  # also misses hidden concept
    message = feedback(PLAN, over_budget, PUBLIC)
    assert message is not None and "at most 10" in message
    assert "secret-concept" not in message and "mentions_required" not in message
    hidden_only = score_plan(PLAN, _plan(5, mention="nothing"))
    assert not hidden_only.passed and feedback(PLAN, hidden_only, PUBLIC) is None
    wrong_model = feedback(PLAN, score_plan(PLAN, _plan(5, model="gpt-9")), PUBLIC)
    assert wrong_model is not None and "qwen3:4b, gemma3:4b" in wrong_model


def test_feedback_for_review_and_code() -> None:
    review = ReviewFixture(id="r", verdict="reject", diff="d", defect_terms=("x",))
    wrong_verdict = Score(False, {"valid_json": True, "verdict_correct": False})
    assert feedback(review, wrong_verdict, frozenset({"valid_json"})) is None
    bad_json = Score(False, {"valid_json": False}, "no JSON object")
    assert "JSON" in (feedback(review, bad_json, frozenset({"valid_json"})) or "")

    code = CodeFixture(id="c", module="m.py", function="main", max_lines=25, difficulty="easy")
    too_long = Score(False, {"splice": False}, "main() is 30 lines > 25")
    message = feedback(code, too_long, frozenset({"splice", "lint", "tests"}))
    assert message is not None and "main() is 30 lines > 25" in message


def test_repair_fixes_a_public_failure_and_counts_every_attempt(
    registry: Registry, tmp_path: Path
) -> None:
    chat = ScriptedChat([_plan(50), _plan(8)])
    bake = _bake(registry, tmp_path, chat, repair=2)
    record = bake.run([Contestant("qwen3:4b")])[0]
    bake.ledger.close()
    assert record.passed and not record.passed_first and record.attempts == 2
    assert record.total_seconds == 6.0 and record.output_tokens == 80  # both attempts
    assert record.energy_method == "estimate" and len(chat.conversations) == 2
    repair_turn = chat.conversations[1][-1]["content"]
    assert repair_turn.startswith("Your answer failed") and "secret-concept" not in repair_turn


def test_hidden_failure_gets_no_retry_and_repair_is_bounded(
    registry: Registry, tmp_path: Path
) -> None:
    hidden = ScriptedChat([_plan(5, mention="nothing")])
    bake = _bake(registry, tmp_path, hidden, repair=3)
    record = bake.run([Contestant("qwen3:4b")])[0]
    bake.ledger.close()
    assert not record.passed and record.attempts == 1  # no oracle signal

    stubborn = ScriptedChat([_plan(99)])
    bake = _bake(registry, tmp_path / "b", stubborn, repair=2)
    record = bake.run([Contestant("qwen3:4b")])[0]
    bake.ledger.close()
    assert not record.passed and record.attempts == 3  # 1 + 2 repairs, then stop


def test_repair_defaults_to_settings(registry: Registry, tmp_path: Path) -> None:
    chat = ScriptedChat([_plan(50), _plan(8)])
    bake = _bake(registry, tmp_path, chat, repair=0)
    bake.repair_attempts = None  # settings say 0 extra attempts
    assert not bake.run([Contestant("qwen3:4b")])[0].passed
    bake.ledger.close()
