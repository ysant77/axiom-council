"""Repair feedback: tell a model which *public* checks its answer failed, never hidden ones.

Public checks are rules the task itself states (reply format, budget, candidate list, code
rules, lint and tests). Hidden checks are the scorer's rubric (required concepts, expected
verdict, milestone range). Feeding back a hidden check, or even signalling "try again" when
only hidden checks failed, would leak the answer key, so neither ever happens.
"""

from __future__ import annotations

from axiom_council.evals.fixtures import CodeFixture, PlanFixture, ReviewFixture
from axiom_council.evals.scoring import Score

Fixture = PlanFixture | ReviewFixture | CodeFixture

_FORMAT = "Your reply was not a single JSON object in the requested shape."


def _plan_message(check: str, fixture: PlanFixture) -> str:
    messages = {
        "valid_json": _FORMAT,
        "within_hours": (
            f"estimate_hours must be a number above 0 and at most {fixture.hours_budget:g}."
        ),
        "models_from_candidates": (
            "Every team member's model must be exactly one of: " + ", ".join(fixture.candidates)
        ),
    }
    return messages.get(check, f"Failed check: {check}.")


def public_failures(score: Score, public: frozenset[str]) -> list[str]:
    return [name for name, ok in score.checks.items() if not ok and name in public]


def feedback(fixture: Fixture, score: Score, public: frozenset[str]) -> str | None:
    """The repair message, or None when no public check failed (then no retry is allowed)."""
    failed = public_failures(score, public)
    if not failed:
        return None
    if isinstance(fixture, PlanFixture):
        lines = [_plan_message(name, fixture) for name in failed]
    elif isinstance(fixture, ReviewFixture):
        lines = [_FORMAT]
    else:
        lines = [f"Failed check: {name}." for name in failed]
        if score.note:
            lines.append("Details:\n" + score.note[-1200:])
    return (
        "Your answer failed these checks:\n- "
        + "\n- ".join(lines)
        + "\nReturn a corrected answer in the same format as before."
    )
