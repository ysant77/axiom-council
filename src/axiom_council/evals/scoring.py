"""Deterministic scorers for plan and review outputs. Each returns a pass flag and the
individual check results, so a failure says exactly which requirement was missed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from axiom_council.evals.fixtures import PlanFixture, ReviewFixture

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Score:
    passed: bool
    checks: dict[str, bool]
    note: str = ""


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = _FENCE.sub("", _THINK.sub("", text).strip())
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _roles_staffed(team: list[Any], required: tuple[str, ...]) -> bool:
    roles = " ".join(str(m.get("role", "")).lower() for m in team if isinstance(m, dict))
    return all(r.lower() in roles for r in required)


def score_plan(fixture: PlanFixture, output: str) -> Score:
    plan = extract_json(output)
    if plan is None:
        return Score(False, {"valid_json": False}, "no JSON object")
    milestones = _list(plan.get("milestones"))
    team = [m for m in _list(plan.get("team")) if isinstance(m, dict)]
    hours = plan.get("estimate_hours")
    text = json.dumps(plan).lower()
    checks = {
        "valid_json": True,
        "milestone_count": fixture.min_milestones <= len(milestones) <= fixture.max_milestones,
        "within_hours": isinstance(hours, int | float) and 0 < hours <= fixture.hours_budget,
        "models_from_candidates": bool(team)
        and all(str(m.get("model")) in fixture.candidates for m in team),
        "roles_staffed": _roles_staffed(team, fixture.required_roles),
        "mentions_required": all(
            any(term.lower() in text for term in group) for group in fixture.must_mention
        ),
        "avoids_forbidden": not any(term.lower() in text for term in fixture.forbid),
        "questions_when_ambiguous": (
            bool(_list(plan.get("clarifying_questions"))) if fixture.expect_questions else True
        ),
    }
    return Score(all(checks.values()), checks)


def score_review(fixture: ReviewFixture, output: str) -> Score:
    review = extract_json(output)
    if review is None:
        return Score(False, {"valid_json": False}, "no JSON object")
    verdict = str(review.get("verdict", "")).lower()
    findings = json.dumps(_list(review.get("findings"))).lower()
    checks = {"valid_json": True, "verdict_correct": verdict == fixture.verdict}
    if fixture.verdict == "reject":
        checks["names_defect"] = any(t.lower() in findings for t in fixture.defect_terms)
    return Score(all(checks.values()), checks)
