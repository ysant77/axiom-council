"""The two critic kinds under admission: the model critic adapter and the rule baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from axiom_council.evals.critic_admission import score_case, score_critic
from axiom_council.evals.critic_contracts import CriticCase, load_cases
from axiom_council.evals.model_critic import ModelCritic, critic_output
from axiom_council.evals.rule_critic import RuleCritic
from axiom_council.registry import Registry
from axiom_council.settings import CriticAdmissionSettings

from .fake_provider import FakeProvider

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/critic_admission"
SETTINGS = CriticAdmissionSettings()
ITEM: dict[str, Any] = {
    "finding_id": "f1", "severity": "high", "fault_class": "injection", "path": "src/a.py",
    "start_line": 3, "end_line": 4, "claim": "c", "evidence": "e",
}  # fmt: skip


def _cases() -> tuple[CriticCase, ...]:
    return load_cases(FIXTURES, SETTINGS.fault_classes)


def test_critic_output_uses_the_pipeline_findings_contract() -> None:
    output = critic_output(json.dumps({"findings": [ITEM]}), limit=5)
    assert output.valid and len(output.findings) == 1
    assert output.findings[0].fault_class == "injection"
    empty = critic_output('{"findings": []}', limit=5)
    assert empty.valid and empty.findings == ()


@pytest.mark.parametrize(
    "text",
    [
        "no json at all",
        '{"findings": [',  # truncated
        '{"verdict": "reject"}',  # no findings key
        '{"findings": {"a": 1}}',  # not a list
        json.dumps([ITEM]),  # a bare array is not the findings object
        "```json\n" + json.dumps({"findings": [ITEM]}) + "\n```",  # fenced
        "<think>hmm</think>" + json.dumps({"findings": [ITEM]}),  # text around the object
        json.dumps({"findings": [{**ITEM, "confidence": 0.9}]}),  # unknown field
        json.dumps({"findings": [{**ITEM, "start_line": "3"}]}),  # string, not int
        json.dumps({"findings": [{**ITEM, "severity": "critical"}]}),  # unknown severity
        json.dumps({"findings": [ITEM, "free text"]}),  # one bad item spoils the output
        json.dumps({"findings": [ITEM], "notes": "extra key"}),
        json.dumps({"findings": [ITEM, ITEM]}),  # duplicate finding ids
        json.dumps({"findings": [{**ITEM, "finding_id": f"f{i}"} for i in range(6)]}),  # > limit
    ],
)
def test_critic_output_rejects_anything_not_strictly_valid(text: str) -> None:
    output = critic_output(text, limit=5)
    assert not output.valid and output.findings == ()


def test_model_critic_renders_the_registry_prompt_and_records_placement(
    registry: Registry,
) -> None:
    provider = FakeProvider([json.dumps({"findings": [ITEM]})])
    critic = ModelCritic(registry, provider, model_id="qwen3:4b", node_id="mac-m5")
    case = _cases()[0]
    output = critic.review(case)
    assert output.valid and output.findings[0].finding_id == "f1"
    assert critic.kind == "model" and critic.critic_id.startswith("model:qwen3:4b")
    assert critic.placement.model_id == "qwen3:4b" and critic.placement.node_id == "mac-m5"
    system, user = provider.conversations[0]
    prompt = registry.prompts[SETTINGS.prompt_id]
    assert system["content"] == prompt.system
    assert case.title in user["content"] and "UNTRUSTED DIFF" in user["content"]
    model_id, extra = provider.calls[0]
    assert model_id == "qwen3:4b" and extra["num_ctx"] == registry.settings.critic_admission.num_ctx


def test_model_critic_counts_a_reply_cut_at_the_token_limit_as_invalid(
    registry: Registry,
) -> None:
    provider = FakeProvider(['{"findings": ['], finish_reason="length")
    critic = ModelCritic(registry, provider, model_id="qwen3:4b", node_id="mac-m5")
    assert not critic.review(_cases()[0]).valid


def test_rule_critic_is_deterministic() -> None:
    critic = RuleCritic()
    cases = _cases()
    assert [critic.review(c) for c in cases] == [critic.review(c) for c in cases]
    assert critic.kind == "rule" and critic.critic_id == "rules"


def test_rule_critic_baseline_on_the_admission_fixtures() -> None:
    """Pin the baseline numbers so a rule change that moves them is a visible decision.

    The rules and the fixtures share an author, so these numbers are an optimistic bound.
    """
    critic = RuleCritic()
    scores = [
        score_case(c, critic.review(c), max_lines=SETTINGS.max_finding_lines) for c in _cases()
    ]
    summary = score_critic(critic.critic_id, critic.kind, None, scores)
    missed = sorted(s.case_id for s in scores if s.case_kind == "seeded" and not s.detected)
    assert missed == ["seeded-race-condition", "seeded-type-coercion"]
    assert summary.clean_with_high == 0
    assert summary.invalid_outputs == 0
    assert summary.high_precision is not None and summary.high_precision >= 0.5


def test_rule_critic_findings_stay_inside_the_diff() -> None:
    critic = RuleCritic()
    for case in _cases():
        for item in critic.review(case).findings:
            assert item.fault_class in SETTINGS.fault_classes
            assert item.path in case.diff and item.start_line <= item.end_line
