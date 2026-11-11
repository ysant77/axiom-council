"""Graph steps that screen critic findings and arbitrate over them (backlog A9 and A10).

`screen` keeps only supported findings; it counts, stores and audits the others with a reason.
Malformed critic output gives zero findings, is audited and blocks approval. A non-empty
patch that has no readable hunks also blocks approval: code cannot screen findings against it.
`answer_deterministic` is the default arbiter. A routed arbiter can replace it through
`AgentSteps.answer`; its answers pass the same coverage check.
`arbitrate` runs the coverage check and chooses the next node. The graph routes on its result.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from axiom_council.audit import AuditLog
from axiom_council.diffs import changed_ranges
from axiom_council.pipeline.findings import (
    Finding,
    FindingAnswer,
    FindingsMalformed,
    accepted_high,
    answer_high_findings,
    parse_findings,
    rejected_high,
    support_problem,
    unanswered_high,
)
from axiom_council.pipeline.guard import PipelineState

_ACTOR = "council"
_MAX_ERROR_CHARS = 1000

METRICS = ("unsupported_findings", "unanswered_high_findings")


def _findings(state: PipelineState) -> list[Finding]:
    return [Finding.model_validate(f) for f in state.get("findings", [])]


def _answers(raw: object) -> tuple[list[FindingAnswer], int]:
    """Valid answers, and the count of items that are not valid answers (they count as none)."""
    items = raw if isinstance(raw, list) else []
    answers, invalid = [], 0
    for item in items:
        try:
            answers.append(FindingAnswer.model_validate(item))
        except ValidationError:
            invalid += 1
    return answers, invalid


def answer_deterministic(state: PipelineState) -> dict[str, Any]:
    """Answer every supported high finding from this run's gate results; no model is called."""
    answers = answer_high_findings(_findings(state), state.get("gate_checks", {}))
    return {"answers": [a.model_dump() for a in answers]}


def decide(
    state: PipelineState, unanswered: list[str], accepted: list[str], max_attempts: int
) -> tuple[str, str]:
    """The next node and the reason. An unanswered high finding fails the run at once.
    A malformed critic output, an unreadable patch, a failed gate or an accepted high finding
    sends the run back to build while attempts remain, and fails it after that."""
    if unanswered:
        return "give_up", f"unanswered high findings: {unanswered}"
    problems = []
    if state.get("critic_malformed"):
        problems.append("critic output malformed")
    if state.get("patch_unreadable"):
        problems.append("patch unreadable")
    if not state.get("gate_passed"):
        problems.append("gate failed")
    if accepted:
        problems.append(f"accepted high findings: {accepted}")
    if not problems:
        return "request_approval", ""
    reason = "; ".join(problems)
    if state.get("attempts", 0) < max_attempts:
        return "build", reason
    return "give_up", reason


def route(state: PipelineState) -> str:
    """The node `arbitrate` chose. With no choice recorded, the run fails closed."""
    return state.get("route", "give_up")


class ReviewSteps:
    def __init__(self, audit: AuditLog, max_findings: int, max_attempts: int):
        self._audit = audit
        self._max_findings = max_findings
        self._max_attempts = max_attempts

    def _parse(self, raw: object) -> list[Finding]:
        if not isinstance(raw, str):
            raise FindingsMalformed("critic output malformed: the critic returned no text")
        return parse_findings(raw, self._max_findings)

    def screen(self, state: PipelineState) -> dict[str, Any]:
        """Drop every finding code cannot resolve. Answers from an earlier attempt are cleared."""
        task_id = state["task_id"]
        cleared: dict[str, Any] = {
            "findings": [],
            "unsupported": [],
            "answers": [],
            "critic_malformed": False,
            "patch_unreadable": False,
        }
        try:
            findings = self._parse(state.get("critic_output"))
        except FindingsMalformed as exc:
            error = str(exc)[:_MAX_ERROR_CHARS]
            self._audit.append("critic_output_malformed", task_id, _ACTOR, error=error)
            return {**cleared, "critic_malformed": True}
        patch = state.get("patch", "")
        ranges = changed_ranges(patch)
        if patch.strip() and not ranges:
            unscreened = [f.finding_id for f in findings]
            self._audit.append("patch_unreadable", task_id, _ACTOR, unscreened=unscreened)
            return {**cleared, "patch_unreadable": True}
        checks = state.get("gate_checks", {})
        supported, unsupported = [], []
        for f in findings:
            problem = support_problem(f, ranges, checks)
            if problem is None:
                supported.append(f.model_dump())
            else:
                unsupported.append({"finding_id": f.finding_id, "reason": problem})
        total = state.get("unsupported_findings", 0) + len(unsupported)
        self._audit.append(
            "findings_screened",
            task_id,
            _ACTOR,
            supported=[f["finding_id"] for f in supported],
            unsupported=unsupported,
            unsupported_findings=total,
        )
        return {
            **cleared,
            "findings": supported,
            "unsupported": unsupported,
            "unsupported_findings": total,
        }

    def arbitrate(self, state: PipelineState) -> dict[str, Any]:
        """Run the coverage check, choose the next node and record both metrics."""
        findings, checks = _findings(state), state.get("gate_checks", {})
        answers, invalid = _answers(state.get("answers"))
        unanswered = unanswered_high(findings, answers, checks)
        accepted = accepted_high(findings, answers, checks)
        rejected = [
            a.model_dump(include={"finding_id", "reason", "check"})
            for a in rejected_high(findings, answers, checks)
        ]
        chosen, reason = decide(state, unanswered, accepted, self._max_attempts)
        total = state.get("unanswered_high_findings", 0) + len(unanswered)
        known = {f.finding_id for f in findings}
        self._audit.append(
            "arbitration",
            state["task_id"],
            _ACTOR,
            route=chosen,
            reason=reason,
            accepted=accepted,
            rejected=[r["finding_id"] for r in rejected],
            unanswered=unanswered,
            invalid_answers=invalid,
            unknown_answers=sorted({a.finding_id for a in answers} - known),
            unsupported_findings=state.get("unsupported_findings", 0),
            unanswered_high_findings=total,
        )
        update: dict[str, Any] = {
            "route": chosen,
            "unanswered_high_findings": total,
            "rejected_high": rejected,
        }
        if chosen == "give_up":
            update["failure_reason"] = reason
        return update
