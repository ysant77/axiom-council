"""Critic admission: scoring by location and mechanism, the admission decision, and the run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from axiom_council.evals.bakeoff import Contestant
from axiom_council.evals.critic_admission import (
    admit,
    run_admission,
    score_case,
    score_critic,
    supports,
)
from axiom_council.evals.critic_contracts import (
    AdmissionSummary,
    CaseScore,
    CriticCase,
    CriticOutput,
    CriticScore,
    FaultKey,
    load_cases,
)
from axiom_council.ledger import CapabilityLedger
from axiom_council.pipeline.findings import Finding
from axiom_council.registry import Registry
from axiom_council.settings import CriticAdmissionSettings

from .fake_provider import FakeProvider

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/critic_admission"
KEY = FaultKey(fault_class="off_by_one", path="src/a.py", start_line=10, end_line=12)
DIFF = "--- a/src/a.py\n+++ b/src/a.py\n@@ -9,4 +9,4 @@\n a\n-b\n+B\n c\n d\n"
SEEDED = CriticCase(id="s", kind="seeded", title="t", diff=DIFF, key=KEY)
CLEAN = CriticCase(id="c", kind="clean", title="t", diff=DIFF)


def finding(**overrides: Any) -> Finding:
    fields: dict[str, Any] = {
        "finding_id": "f1", "severity": "high", "fault_class": "off_by_one",
        "path": "src/a.py", "start_line": 11, "end_line": 11, "claim": "c", "evidence": "e",
    }  # fmt: skip
    return Finding(**{**fields, **overrides})


def test_finding_has_exactly_the_shared_fields() -> None:
    assert list(Finding.model_fields) == [
        "finding_id", "severity", "fault_class", "path", "start_line", "end_line", "claim",
        "evidence",
    ]  # fmt: skip
    with pytest.raises(ValueError):
        Finding.model_validate({**finding().model_dump(), "confidence": 0.9})


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"path": "b/src/a.py"}, True),  # diff prefixes are normalised
        ({"start_line": 1, "end_line": 10}, True),  # overlap at the edge
        ({"fault_class": "Off-By-One"}, True),  # spelling of the same class id
        ({"path": "src/b.py"}, False),
        ({"start_line": 13, "end_line": 14}, False),
        ({"fault_class": "injection"}, False),  # right place, wrong mechanism
        ({"start_line": 1, "end_line": 100}, False),  # too wide to locate anything
        ({"start_line": 12, "end_line": 11}, False),  # inverted range
        ({"start_line": 0, "end_line": 11}, False),  # line numbers start at 1
        ({"path": "./src/a.py"}, True),  # the pipeline's path rules apply
        ({"path": "src/../src/a.py"}, True),
        ({"path": "../src/a.py"}, False),
        ({"path": "/src/a.py"}, False),
    ],
)
def test_support_needs_path_overlap_and_class(overrides: dict[str, Any], expected: bool) -> None:
    assert supports(finding(**overrides), KEY, max_lines=20) is expected


def test_keywords_in_the_claim_never_earn_support() -> None:
    noisy = finding(fault_class="injection", claim="off-by-one bug in range", evidence="- 1")
    assert not supports(noisy, KEY, max_lines=20)


def test_score_case_counts_supporting_and_unsupported_findings() -> None:
    output = CriticOutput(
        findings=(finding(), finding(finding_id="f2", severity="low", path="src/z.py"))
    )
    seeded = score_case(SEEDED, output, max_lines=20)
    assert seeded.detected and (seeded.supporting, seeded.supporting_high) == (1, 1)
    assert (seeded.unsupported, seeded.high) == (1, 1)
    clean = score_case(CLEAN, output, max_lines=20)
    assert not clean.detected and clean.unsupported == 2 and clean.high == 1
    invalid = score_case(SEEDED, CriticOutput(findings=(), valid=False), max_lines=20)
    assert not invalid.valid_output and not invalid.detected


def _scores() -> list[CaseScore]:
    hit = CriticOutput(findings=(finding(),))
    miss = CriticOutput(findings=(finding(path="src/z.py"),))
    medium = CriticOutput(findings=(finding(severity="medium"),))
    return [
        score_case(SEEDED, hit, max_lines=20),
        score_case(SEEDED, miss, max_lines=20),
        score_case(CLEAN, CriticOutput(findings=()), max_lines=20),
        score_case(CLEAN, medium, max_lines=20),
        score_case(CLEAN, CriticOutput(findings=(), valid=False), max_lines=20),
    ]


def test_score_critic_reports_recall_false_high_precision_and_unsupported() -> None:
    score = score_critic("m", "model", "qwen3:4b", _scores())
    assert (score.seeded_cases, score.clean_cases, score.detected) == (2, 3, 1)
    assert score.supported_recall == 0.5
    assert score.clean_with_high == 0 and score.clean_false_high_rate == 0.0
    assert (score.high_findings, score.supported_high, score.high_precision) == (2, 1, 0.5)
    assert score.unsupported_findings == 2
    assert score.invalid_outputs == 1 and score.invalid_output_rate == 0.2


def test_score_critic_without_high_findings_has_no_precision() -> None:
    score = score_critic("m", "model", None, _scores()[2:4])
    assert score.high_precision is None and score.supported_recall == 0.0


def _critic(critic_id: str, kind: str, **overrides: Any) -> CriticScore:
    fields: dict[str, Any] = {
        "critic_id": critic_id, "kind": kind, "model_id": None, "seeded_cases": 12,
        "clean_cases": 12, "detected": 9, "supported_recall": 0.75, "clean_with_high": 0,
        "clean_false_high_rate": 0.0, "high_findings": 10, "supported_high": 8,
        "high_precision": 0.8, "unsupported_findings": 3, "invalid_outputs": 0,
        "invalid_output_rate": 0.0,
    }  # fmt: skip
    return CriticScore(**{**fields, **overrides})


def _summary(*critics: CriticScore) -> AdmissionSummary:
    return AdmissionSummary(run_id="r", harness_hash="h", fixture_hash="f", critics=critics)


def test_summary_requires_the_rule_baseline_and_unique_ids() -> None:
    with pytest.raises(ValueError, match="rule"):
        _summary(_critic("m", "model"))
    with pytest.raises(ValueError, match="unique"):
        _summary(_critic("rules", "rule"), _critic("rules", "model"))


def test_admit_passes_a_model_critic_that_meets_every_threshold() -> None:
    decision = admit(_summary(_critic("rules", "rule"), _critic("m", "model")), _settings())
    assert decision.admitted == ("m",) and decision.fallback is None
    assert all(v.passed and not v.reasons for v in decision.verdicts)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"supported_recall": 0.083, "detected": 1}, "supported recall"),
        ({"clean_with_high": 11, "clean_false_high_rate": 0.917}, "clean cases with a high"),
        ({"high_precision": 0.1, "supported_high": 1}, "high precision"),
        ({"high_precision": None, "high_findings": 0, "supported_high": 0}, "no high findings"),
        ({"invalid_outputs": 6, "invalid_output_rate": 0.5}, "invalid output"),
        ({"seeded_cases": 3}, "seeded cases"),
        ({"clean_cases": 3}, "clean cases"),
    ],
)
def test_admit_rejects_with_a_reason_and_names_the_rule_fallback(
    overrides: dict[str, Any], reason: str
) -> None:
    summary = _summary(_critic("rules", "rule"), _critic("m", "model", **overrides))
    decision = admit(summary, _settings())
    assert decision.admitted == () and decision.fallback == "rules"
    model = next(v for v in decision.verdicts if v.critic_id == "m")
    assert not model.passed and any(reason in r for r in model.reasons)


def test_rule_fallback_is_named_even_when_the_rule_critic_misses_a_threshold() -> None:
    weak_rules = _critic("rules", "rule", supported_recall=0.25, detected=3)
    decision = admit(_summary(weak_rules), _settings())
    assert decision.admitted == () and decision.fallback == "rules"
    assert not decision.verdicts[0].passed


def _settings() -> CriticAdmissionSettings:
    return CriticAdmissionSettings(
        min_supported_recall=0.5, max_clean_false_high_rate=0.1, min_high_precision=0.5,
        max_invalid_output_rate=0.1, min_seeded_cases=12, min_clean_cases=12,
    )  # fmt: skip


def _perfect_replies(cases: tuple[CriticCase, ...]) -> list[str]:
    replies: list[str] = []
    for case in cases:
        found = []
        if case.key is not None:
            found.append(
                finding(
                    fault_class=case.key.fault_class, path=case.key.path,
                    start_line=case.key.start_line, end_line=case.key.end_line,
                ).model_dump()
            )  # fmt: skip
        replies.append(json.dumps({"findings": found}))
    return replies


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_run_admission_admits_a_perfect_model_and_keeps_the_rule_baseline(
    registry: Registry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_ledger(*_: Any, **__: Any) -> None:
        raise AssertionError("admission results must not reach the capability ledger")

    monkeypatch.setattr(CapabilityLedger, "record", no_ledger)
    cases = load_cases(FIXTURES, registry.settings.critic_admission.fault_classes)
    provider = FakeProvider(_perfect_replies(cases))
    results = tmp_path / "admission" / "run-1.jsonl"
    run = run_admission(
        registry, fixtures_dir=FIXTURES, results_path=results, provider=provider,
        node_id="mac-m5", models=[Contestant("qwen3:4b", "low")],
    )  # fmt: skip
    kinds = [c.kind for c in run.summary.critics]
    assert kinds == ["rule", "model"]
    model = run.summary.critics[1]
    assert model.supported_recall == 1.0 and model.clean_false_high_rate == 0.0
    assert run.decision.admitted == (model.critic_id,) and run.decision.fallback is None
    assert len(provider.conversations) == 24
    user = provider.conversations[0][-1]["content"]
    assert "UNTRUSTED DIFF" in user and "off_by_one" in user
    assert provider.calls[0][1]["think"] == "low"
    assert not (tmp_path / "ledger.sqlite3").exists()
    _check_rows(_rows(results), run.summary.harness_hash, model.critic_id)


def _check_rows(rows: list[dict[str, Any]], harness: str, model_id: str) -> None:
    assert {r["record"] for r in rows} == {"case", "critic", "decision"}
    assert len([r for r in rows if r["record"] == "case"]) == 48
    assert all(r["harness_hash"] == harness and r["run_id"] == "run-1" for r in rows)
    model_rows = [r for r in rows if r.get("critic_id") == model_id]
    assert all(r["placement"]["model_id"] == "qwen3:4b" for r in model_rows)
    assert all(r["placement"]["reasoning"] == "low" for r in model_rows)
    rule_rows = [r for r in rows if r.get("critic_id") == "rules"]
    assert rule_rows and all(r["placement"] is None for r in rule_rows)
    decision = next(r for r in rows if r["record"] == "decision")
    assert decision["decision"]["admitted"] == [model_id]
    assert decision["baseline_in_sample"] is True


def test_run_admission_counts_invalid_model_output_and_falls_back(
    registry: Registry, tmp_path: Path
) -> None:
    provider = FakeProvider(["I think line 3 looks wrong."])
    run = run_admission(
        registry, fixtures_dir=FIXTURES, results_path=tmp_path / "r.jsonl", provider=provider,
        node_id="mac-m5", models=[Contestant("qwen3:4b")],
    )  # fmt: skip
    model = run.summary.critics[1]
    assert model.invalid_outputs == 24 and model.detected == 0
    assert run.decision.admitted == () and run.decision.fallback == "rules"


def test_run_admission_refuses_reuse_and_missing_inputs(registry: Registry, tmp_path: Path) -> None:
    existing = tmp_path / "r.jsonl"
    existing.write_text("", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_admission(registry, fixtures_dir=FIXTURES, results_path=existing)
    with pytest.raises(ValueError, match="provider"):
        run_admission(
            registry, fixtures_dir=FIXTURES, results_path=tmp_path / "x.jsonl",
            models=[Contestant("qwen3:4b")],
        )  # fmt: skip
    with pytest.raises(ValueError, match="model card"):
        run_admission(
            registry, fixtures_dir=FIXTURES, results_path=tmp_path / "y.jsonl",
            provider=FakeProvider([]), node_id="mac-m5", models=[Contestant("ghost:1")],
        )  # fmt: skip


def test_run_admission_with_rules_only_writes_a_decision(
    registry: Registry, tmp_path: Path
) -> None:
    run = run_admission(registry, fixtures_dir=FIXTURES, results_path=tmp_path / "r.jsonl")
    assert [c.critic_id for c in run.summary.critics] == ["rules"]
    assert run.decision.fallback == "rules" and run.decision.admitted == ()
    rows = _rows(tmp_path / "r.jsonl")
    assert len(rows) == 24 + 2
