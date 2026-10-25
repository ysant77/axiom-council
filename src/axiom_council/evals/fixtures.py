"""Fixture schemas and loading. Fixtures are data with known answers, never model output."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from axiom_council.contracts import Strict


class PlanFixture(Strict):
    id: str
    request: str
    candidates: tuple[str, ...] = Field(min_length=1)
    min_milestones: int = Field(ge=1)
    max_milestones: int = Field(ge=1)
    hours_budget: float = Field(gt=0)
    required_roles: tuple[str, ...] = ()
    must_mention: tuple[tuple[str, ...], ...] = ()
    forbid: tuple[str, ...] = ()
    expect_questions: bool = False


class ReviewFixture(Strict):
    id: str
    verdict: Literal["approve", "reject"]
    defect_terms: tuple[str, ...] = ()
    diff: str = Field(min_length=1)


class CodeFixture(Strict):
    id: str
    module: str
    function: str
    max_lines: int = Field(ge=5, le=200)
    difficulty: Literal["easy", "medium", "hard"]


class FixtureSet(Strict):
    plan: tuple[PlanFixture, ...] = ()
    review: tuple[ReviewFixture, ...] = ()
    code: tuple[CodeFixture, ...] = ()

    def count(self) -> int:
        return len(self.plan) + len(self.review) + len(self.code)


def load_fixtures(directory: Path) -> FixtureSet:
    def read(name: str) -> list[dict[str, object]]:
        path = directory / f"{name}.yaml"
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            items: list[dict[str, object]] = (yaml.safe_load(fh) or {}).get("fixtures", [])
            return items

    fixtures = FixtureSet(
        plan=tuple(PlanFixture.model_validate(f) for f in read("plan")),
        review=tuple(ReviewFixture.model_validate(f) for f in read("review")),
        code=tuple(CodeFixture.model_validate(f) for f in read("code")),
    )
    ids = [f.id for group in (fixtures.plan, fixtures.review, fixtures.code) for f in group]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture ids")
    return fixtures
