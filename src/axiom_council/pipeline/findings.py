"""Critic findings and arbiter answers: the typed contracts and the pure rules that check them.

A finding is supported only when code can resolve what it cites. Its path and line range
lie inside a changed hunk of the patch under review, or its evidence names a gate check
that failed in this run (`gate:<check>`). Every supported high finding needs an answer.
An accept counts with any non-blank reason. A reject counts only when it also names a gate
check that passed in this run. This module calls no model and writes no state.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from axiom_council.contracts import Strict
from axiom_council.pipeline.hunks import Ranges

GATE_EVIDENCE_PREFIX = "gate:"

Severity = Literal["high", "medium", "low"]
Decision = Literal["accept", "reject"]


class Finding(Strict):
    """One claim a critic makes about the patch. The field names are a shared contract."""

    finding_id: str
    severity: Severity
    fault_class: str
    path: str
    start_line: int
    end_line: int
    claim: str
    evidence: str


class FindingAnswer(Strict):
    """The arbiter's answer to one finding. `check` names the passing gate check a reject cites."""

    finding_id: str
    decision: Decision
    reason: str
    check: str = ""


class FindingsMalformed(ValueError):
    """The critic output is not a list of valid findings with unique ids within the limit."""


_FINDINGS = TypeAdapter(list[Finding])


def parse_findings(output: str, limit: int) -> list[Finding]:
    """Parse the critic's JSON array of findings. Any defect rejects the whole output."""
    try:
        findings = _FINDINGS.validate_json(output)
    except ValidationError as exc:
        raise FindingsMalformed(f"critic output malformed: {exc}") from exc
    if len(findings) > limit:
        raise FindingsMalformed(f"critic output malformed: {len(findings)} findings > {limit}")
    ids = [f.finding_id for f in findings]
    if len(set(ids)) != len(ids):
        raise FindingsMalformed("critic output malformed: duplicate finding ids")
    return findings


def cited_check(finding: Finding) -> str | None:
    evidence = finding.evidence.strip()
    if evidence.startswith(GATE_EVIDENCE_PREFIX):
        return evidence[len(GATE_EVIDENCE_PREFIX) :].strip()
    return None


def _location_problem(finding: Finding, ranges: Ranges) -> str | None:
    path = finding.path.removeprefix("./")
    if path not in ranges:
        return f"{finding.path} is not in the patch"
    start, end = finding.start_line, finding.end_line
    if any(first <= start and end <= last for first, last in ranges[path]):
        return None
    return f"lines {start}-{end} lie outside every changed hunk of {path}"


def support_problem(finding: Finding, ranges: Ranges, gate_checks: dict[str, bool]) -> str | None:
    """None when the finding is supported; otherwise the reason it is not."""
    if not finding.claim.strip():
        return "empty claim"
    if finding.start_line < 1 or finding.end_line < finding.start_line:
        return f"invalid line range {finding.start_line}-{finding.end_line}"
    location = _location_problem(finding, ranges)
    if location is None:
        return None
    check = cited_check(finding)
    if check is None:
        return location
    if gate_checks.get(check) is False:
        return None
    return f"{location}; gate check {check!r} did not fail in this run"


def answer_high_findings(
    findings: Iterable[Finding], gate_checks: dict[str, bool]
) -> list[FindingAnswer]:
    """The deterministic arbiter. It rejects a high finding only when a gate check named by
    its fault class passed in this run and the finding cites no failed gate check. It accepts
    every other high finding."""
    answers = []
    for f in findings:
        if f.severity != "high":
            continue
        cited = cited_check(f)
        cited_failed = cited is not None and gate_checks.get(cited) is False
        if gate_checks.get(f.fault_class) is True and not cited_failed:
            reason = f"gate check {f.fault_class!r} passed in this run"
            answers.append(
                FindingAnswer(
                    finding_id=f.finding_id, decision="reject", reason=reason, check=f.fault_class
                )
            )
        else:
            reason = (
                f"gate check {cited!r} failed in this run"
                if cited_failed
                else f"no passing gate check refutes this {f.fault_class} finding"
            )
            answers.append(FindingAnswer(finding_id=f.finding_id, decision="accept", reason=reason))
    return answers


def answer_counts(answer: FindingAnswer, gate_checks: dict[str, bool]) -> bool:
    if not answer.reason.strip():
        return False
    return answer.decision == "accept" or gate_checks.get(answer.check) is True


def unanswered_high(
    findings: Iterable[Finding], answers: Iterable[FindingAnswer], gate_checks: dict[str, bool]
) -> list[str]:
    """Ids of high findings with no answer that counts (the arbiter coverage check)."""
    answered = {a.finding_id for a in answers if answer_counts(a, gate_checks)}
    return sorted(
        f.finding_id for f in findings if f.severity == "high" and f.finding_id not in answered
    )


def accepted_high(
    findings: Iterable[Finding], answers: Iterable[FindingAnswer], gate_checks: dict[str, bool]
) -> list[str]:
    """Ids of high findings with an accept that counts. An accept wins over any reject."""
    accepts = {
        a.finding_id for a in answers if a.decision == "accept" and answer_counts(a, gate_checks)
    }
    return sorted(
        f.finding_id for f in findings if f.severity == "high" and f.finding_id in accepts
    )
