"""Critic admission test (backlog A8): a critic or verifier role runs in the pipeline only
after it passes this test on cases with known answers.

Scoring is by location and mechanism, never by keyword. A finding supports a seeded fault
only if its path matches, its line range overlaps the key range, and its fault class
matches. The deterministic rule critic is scored on the same cases as every model critic,
and it is the fallback when no model critic passes. Results go to a JSONL file only. They
are admission evidence, not routing evidence, so they never reach the capability ledger.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from axiom_council.contracts import Strict, utcnow
from axiom_council.evals.bakeoff import Contestant
from axiom_council.evals.critic_contracts import (
    AdmissionDecision,
    AdmissionSummary,
    CaseScore,
    Critic,
    CriticCase,
    CriticKind,
    CriticOutput,
    CriticScore,
    CriticVerdict,
    FaultKey,
    load_cases,
)
from axiom_council.evals.model_critic import ModelCritic
from axiom_council.evals.rule_critic import BASELINE_IN_SAMPLE, RuleCritic
from axiom_council.evals.run_provenance import run_provenance
from axiom_council.pipeline.findings import Finding, resolve_path
from axiom_council.provenance import Placement, fixture_hash, harness_hash, source_commit
from axiom_council.providers import ChatProvider
from axiom_council.registry import Registry
from axiom_council.settings import CriticAdmissionSettings


def _class_id(value: str) -> str:
    """One spelling per fault class id: `Off-By-One` and `off_by_one` are the same id."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def supports(finding: Finding, key: FaultKey, *, max_lines: int) -> bool:
    """Path, line overlap and fault class must all match; the claim text is never read.
    The path is resolved by the pipeline's own rule, so both judge a path the same way."""
    start, end = finding.start_line, finding.end_line
    if start < 1 or end < start or end - start + 1 > max_lines:
        return False
    return (
        resolve_path(finding.path, {key.path: [(key.start_line, key.end_line)]}) == key.path
        and start <= key.end_line
        and end >= key.start_line
        and _class_id(finding.fault_class) == _class_id(key.fault_class)
    )


def score_case(case: CriticCase, output: CriticOutput, *, max_lines: int) -> CaseScore:
    key = case.key
    backing = [
        f for f in output.findings if key is not None and supports(f, key, max_lines=max_lines)
    ]
    return CaseScore(
        case_id=case.id,
        case_kind=case.kind,
        valid_output=output.valid,
        findings=len(output.findings),
        high=sum(f.severity == "high" for f in output.findings),
        supporting=len(backing),
        supporting_high=sum(f.severity == "high" for f in backing),
        unsupported=len(output.findings) - len(backing),
        detected=bool(backing),
    )


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def score_critic(
    critic_id: str, kind: CriticKind, model_id: str | None, scores: Sequence[CaseScore]
) -> CriticScore:
    seeded = [s for s in scores if s.case_kind == "seeded"]
    clean = [s for s in scores if s.case_kind == "clean"]
    detected = sum(s.detected for s in seeded)
    clean_high = sum(s.high > 0 for s in clean)
    high = sum(s.high for s in scores)
    supported_high = sum(s.supporting_high for s in scores)
    invalid = sum(not s.valid_output for s in scores)
    return CriticScore(
        critic_id=critic_id,
        kind=kind,
        model_id=model_id,
        seeded_cases=len(seeded),
        clean_cases=len(clean),
        detected=detected,
        supported_recall=_rate(detected, len(seeded)),
        clean_with_high=clean_high,
        clean_false_high_rate=_rate(clean_high, len(clean)),
        high_findings=high,
        supported_high=supported_high,
        high_precision=_rate(supported_high, high) if high else None,
        unsupported_findings=sum(s.unsupported for s in scores),
        invalid_outputs=invalid,
        invalid_output_rate=_rate(invalid, len(scores)),
    )


def _failures(score: CriticScore, cfg: CriticAdmissionSettings) -> list[str]:
    reasons: list[str] = []
    if score.seeded_cases < cfg.min_seeded_cases:
        reasons.append(f"seeded cases {score.seeded_cases} < {cfg.min_seeded_cases}")
    if score.clean_cases < cfg.min_clean_cases:
        reasons.append(f"clean cases {score.clean_cases} < {cfg.min_clean_cases}")
    if score.supported_recall < cfg.min_supported_recall:
        reasons.append(
            f"supported recall {score.supported_recall:.3f} < {cfg.min_supported_recall:.3f}"
        )
    if score.clean_false_high_rate > cfg.max_clean_false_high_rate:
        reasons.append(
            f"clean cases with a high finding {score.clean_false_high_rate:.3f}"
            f" > {cfg.max_clean_false_high_rate:.3f}"
        )
    if score.high_precision is None:
        reasons.append("no high findings, so high precision is unknown")
    elif score.high_precision < cfg.min_high_precision:
        reasons.append(f"high precision {score.high_precision:.3f} < {cfg.min_high_precision:.3f}")
    if score.invalid_output_rate > cfg.max_invalid_output_rate:
        reasons.append(
            f"invalid output rate {score.invalid_output_rate:.3f}"
            f" > {cfg.max_invalid_output_rate:.3f}"
        )
    return reasons


def admit(summary: AdmissionSummary, settings: CriticAdmissionSettings) -> AdmissionDecision:
    """Admit each model critic that meets every threshold. If none does, name the rule
    critic as the fallback, whether or not the rule critic meets the thresholds itself."""
    verdicts = []
    for score in summary.critics:
        reasons = _failures(score, settings)
        verdicts.append(
            CriticVerdict(
                critic_id=score.critic_id, kind=score.kind, passed=not reasons,
                reasons=tuple(reasons),
            )
        )  # fmt: skip
    admitted = tuple(v.critic_id for v in verdicts if v.kind == "model" and v.passed)
    rules = next(v.critic_id for v in verdicts if v.kind == "rule")
    return AdmissionDecision(
        admitted=admitted, fallback=None if admitted else rules, verdicts=tuple(verdicts)
    )


@dataclass(frozen=True)
class AdmissionRun:
    summary: AdmissionSummary
    decision: AdmissionDecision
    results_path: Path


@dataclass(frozen=True)
class _Entrant:
    critic: Critic
    placement: Placement | None
    model_id: str | None


def _entrants(
    registry: Registry,
    provider: ChatProvider | None,
    node_id: str | None,
    models: Sequence[Contestant],
    run_id: str,
) -> list[_Entrant]:
    entrants = [_Entrant(RuleCritic(), None, None)]
    if not models:
        return entrants
    if provider is None or node_id is None:
        raise ValueError("model critics need a provider and a node_id")
    unknown = [m.model_id for m in models if m.model_id not in registry.models]
    if unknown:
        raise ValueError(f"models without a model card: {unknown}")
    for model in models:
        prov = run_provenance(
            registry, provider, model_id=model.model_id, think=model.think,
            node_id=node_id, run_id=run_id, target_commit=None,
        )  # fmt: skip
        critic = ModelCritic(registry, provider, model.model_id, node_id, model.think)
        entrants.append(_Entrant(critic, prov.placement, model.model_id))
    return entrants


def _case_row(
    entrant: _Entrant, case: CriticCase, out: CriticOutput, score: CaseScore
) -> dict[str, Any]:
    return {
        "record": "case",
        "critic_id": entrant.critic.critic_id,
        "critic_kind": entrant.critic.kind,
        "placement": asdict(entrant.placement) if entrant.placement else None,
        "case_id": case.id,
        "case_hash": fixture_hash(case),
        "valid_output": out.valid,
        "findings": [f.model_dump() for f in out.findings],
        "raw": out.raw,
        "score": score.model_dump(),
    }


def _score_entrant(
    entrant: _Entrant, cases: Sequence[CriticCase], max_lines: int
) -> tuple[CriticScore, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    scores: list[CaseScore] = []
    for case in cases:
        output = entrant.critic.review(case)
        scores.append(score_case(case, output, max_lines=max_lines))
        rows.append(_case_row(entrant, case, output, scores[-1]))
    critic = entrant.critic
    score = score_critic(critic.critic_id, critic.kind, entrant.model_id, scores)
    rows.append({
        "record": "critic", "critic_id": critic.critic_id,
        "placement": asdict(entrant.placement) if entrant.placement else None,
        "score": score.model_dump(),
    })  # fmt: skip
    return score, rows


class _CaseSet(Strict):
    cases: tuple[CriticCase, ...]


def _write(path: Path, rows: list[dict[str, Any]], common: dict[str, Any]) -> None:
    """Write every row at once through a temporary file, so a crash leaves no partial run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({**common, **row}) + "\n")
    os.replace(tmp, path)


def run_admission(
    registry: Registry,
    *,
    fixtures_dir: Path,
    results_path: Path,
    provider: ChatProvider | None = None,
    node_id: str | None = None,
    models: Sequence[Contestant] = (),
) -> AdmissionRun:
    """Score the rule critic and each model critic on every case, then decide admission.

    The caller builds the provider (through the provider factory) and closes it.
    """
    if results_path.exists():
        raise FileExistsError(f"{results_path} exists; each admission run needs a new file")
    cfg = registry.settings.critic_admission
    cases = load_cases(fixtures_dir, cfg.fault_classes)
    run_id = results_path.stem
    entrants = _entrants(registry, provider, node_id, models, run_id)
    rows: list[dict[str, Any]] = []
    critics: list[CriticScore] = []
    for entrant in entrants:
        score, entrant_rows = _score_entrant(entrant, cases, cfg.max_finding_lines)
        critics.append(score)
        rows += entrant_rows
    summary = AdmissionSummary(
        run_id=run_id, harness_hash=harness_hash(registry),
        fixture_hash=fixture_hash(_CaseSet(cases=cases)), critics=tuple(critics),
    )  # fmt: skip
    decision = admit(summary, cfg)
    rows.append({"record": "decision", "decision": decision.model_dump(),
                 "baseline_in_sample": BASELINE_IN_SAMPLE,
                 "settings": cfg.model_dump()})  # fmt: skip
    common = {"run_id": run_id, "harness_hash": summary.harness_hash,
              "fixture_hash": summary.fixture_hash, "source_commit": source_commit(),
              "ts": utcnow().isoformat()}  # fmt: skip
    _write(results_path, rows, common)
    return AdmissionRun(summary, decision, results_path)
