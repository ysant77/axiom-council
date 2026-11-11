"""Typed contracts for the critic admission test (backlog A8): cases with known answers,
critic outputs, scores, the summary and the decision. Also loads and checks the case
fixtures. Findings use the pipeline's `Finding` type, so an admitted critic speaks the same
contract in the pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import Field, model_validator

from axiom_council.contracts import Strict
from axiom_council.diffs import parse_diff
from axiom_council.pipeline.findings import Finding

CaseKind = Literal["seeded", "clean"]
CriticKind = Literal["rule", "model"]


class FaultKey(Strict):
    """Where the seeded fault is (new-file lines) and what mechanism it is."""

    fault_class: str
    path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class CriticCase(Strict):
    id: str
    kind: CaseKind
    title: str
    diff: str = Field(min_length=1)
    key: FaultKey | None = None

    @model_validator(mode="after")
    def _key_matches_kind(self) -> CriticCase:
        if (self.kind == "seeded") != (self.key is not None):
            raise ValueError(f"case {self.id}: a seeded case has a key, a clean case has none")
        return self


class CriticOutput(Strict):
    """What one critic said about one case. An output that failed strict parsing is
    `valid=False` with no findings, and it is counted."""

    findings: tuple[Finding, ...] = ()
    valid: bool = True
    raw: str = ""


class Critic(Protocol):
    @property
    def critic_id(self) -> str: ...

    @property
    def kind(self) -> CriticKind: ...

    def review(self, case: CriticCase) -> CriticOutput: ...


class CaseScore(Strict):
    case_id: str
    case_kind: CaseKind
    valid_output: bool
    findings: int = Field(ge=0)
    high: int = Field(ge=0)
    supporting: int = Field(ge=0)
    supporting_high: int = Field(ge=0)
    unsupported: int = Field(ge=0)
    detected: bool


class CriticScore(Strict):
    critic_id: str
    kind: CriticKind
    model_id: str | None
    seeded_cases: int = Field(ge=0)
    clean_cases: int = Field(ge=0)
    detected: int = Field(ge=0)
    supported_recall: float = Field(ge=0, le=1)
    clean_with_high: int = Field(ge=0)
    clean_false_high_rate: float = Field(ge=0, le=1)
    high_findings: int = Field(ge=0)
    supported_high: int = Field(ge=0)
    high_precision: float | None = Field(ge=0, le=1)
    unsupported_findings: int = Field(ge=0)
    invalid_outputs: int = Field(ge=0)
    invalid_output_rate: float = Field(ge=0, le=1)


class AdmissionSummary(Strict):
    """Scores for every critic on one case set. The rule baseline is always present."""

    run_id: str
    harness_hash: str
    fixture_hash: str
    critics: tuple[CriticScore, ...]

    @model_validator(mode="after")
    def _baseline_present(self) -> AdmissionSummary:
        if not any(c.kind == "rule" for c in self.critics):
            raise ValueError("an admission summary must include the rule critic baseline")
        ids = [c.critic_id for c in self.critics]
        if len(ids) != len(set(ids)):
            raise ValueError("critic ids must be unique")
        return self


class CriticVerdict(Strict):
    critic_id: str
    kind: CriticKind
    passed: bool
    reasons: tuple[str, ...] = ()


class AdmissionDecision(Strict):
    """Model critics admitted to run; if none, `fallback` names the rule critic to use."""

    admitted: tuple[str, ...]
    fallback: str | None
    verdicts: tuple[CriticVerdict, ...]


def _read(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        items: list[dict[str, object]] = (yaml.safe_load(fh) or {}).get("cases", [])
    return items


def _check_key(case: CriticCase, fault_classes: tuple[str, ...]) -> None:
    key = case.key
    if key is None:
        return
    if key.fault_class not in fault_classes:
        raise ValueError(f"case {case.id}: fault class {key.fault_class!r} is not in settings")
    hunks = [h for h in parse_diff(case.diff) if h.path == key.path]
    if not any(h.covers(key.start_line, key.end_line) for h in hunks):
        raise ValueError(f"case {case.id}: key location is not inside the diff")


def load_cases(directory: Path, fault_classes: tuple[str, ...]) -> tuple[CriticCase, ...]:
    """Seeded cases, then clean cases. Every diff must parse and every key must lie in it."""
    cases = [
        CriticCase.model_validate({**raw, "kind": kind})
        for kind in ("seeded", "clean")
        for raw in _read(directory / f"{kind}.yaml")
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate critic admission case ids")
    for case in cases:
        parse_diff(case.diff)
        _check_key(case, fault_classes)
    return tuple(cases)
