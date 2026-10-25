from __future__ import annotations

import json
from pathlib import Path

import pytest

from axiom_council.evals.fixtures import PlanFixture, ReviewFixture, load_fixtures
from axiom_council.evals.scoring import extract_json, score_plan, score_review
from axiom_council.evals.splice import SpliceError, extract_code, splice

REPO = Path(__file__).resolve().parents[1]

PLAN = PlanFixture(
    id="p",
    request="refactor",
    candidates=("big:1", "small:1"),
    min_milestones=1,
    max_milestones=3,
    hours_budget=10,
    required_roles=("dev", "qa"),
    must_mention=(("test",), ("review", "approv")),
    forbid=("push to main",),
    expect_questions=True,
)


def _plan(**overrides: object) -> str:
    body: dict[str, object] = {
        "milestones": [{"name": "m1", "goal": "keep tests green", "acceptance": ["review"]}],
        "team": [{"role": "dev", "model": "big:1"}, {"role": "QA critic", "model": "small:1"}],
        "risks": [],
        "estimate_hours": 6,
        "clarifying_questions": ["Which module first?"],
    }
    body.update(overrides)
    return "<think>hmm</think>```json\n" + json.dumps(body) + "\n```"


def test_repo_fixtures_load() -> None:
    fixtures = load_fixtures(REPO / "evals" / "fixtures")
    assert len(fixtures.plan) == 6 and len(fixtures.review) == 6 and len(fixtures.code) == 6
    assert fixtures.count() == 18


def test_duplicate_fixture_ids_rejected(tmp_path: Path) -> None:
    item = "  - {id: x, verdict: approve, diff: d}\n"
    (tmp_path / "review.yaml").write_text("fixtures:\n" + item + item)
    with pytest.raises(ValueError, match="duplicate"):
        load_fixtures(tmp_path)


def test_good_plan_passes_every_check() -> None:
    score = score_plan(PLAN, _plan())
    assert score.passed, score.checks


@pytest.mark.parametrize(
    ("overrides", "failed"),
    [
        ({"milestones": []}, "milestone_count"),
        ({"estimate_hours": 40}, "within_hours"),
        ({"estimate_hours": "six"}, "within_hours"),
        (
            {"team": [{"role": "dev", "model": "gpt-9"}, {"role": "qa", "model": "big:1"}]},
            "models_from_candidates",
        ),
        ({"team": [{"role": "dev", "model": "big:1"}]}, "roles_staffed"),
        ({"risks": ["we will push to main"]}, "avoids_forbidden"),
        ({"clarifying_questions": []}, "questions_when_ambiguous"),
    ],
)
def test_plan_checks_catch_each_violation(overrides: dict[str, object], failed: str) -> None:
    score = score_plan(PLAN, _plan(**overrides))
    assert not score.passed and not score.checks[failed]


def test_unparseable_outputs_fail_cleanly() -> None:
    assert not score_plan(PLAN, "I think we should...").passed
    assert extract_json("[1, 2]") is None and extract_json("{bad json}") is None
    review = ReviewFixture(id="r", verdict="approve", diff="d")
    assert score_review(review, "no json").checks == {"valid_json": False}


def test_review_scoring() -> None:
    bug = ReviewFixture(id="r", verdict="reject", defect_terms=("off-by-one",), diff="d")
    hit = json.dumps({"verdict": "reject", "findings": [{"issue": "Off-by-one: last page"}]})
    vague = json.dumps({"verdict": "reject", "findings": [{"issue": "looks odd"}]})
    assert score_review(bug, hit).passed
    assert not score_review(bug, vague).checks["names_defect"]
    clean = ReviewFixture(id="c", verdict="approve", diff="d")
    assert not score_review(clean, hit).passed


SOURCE = """import json

LIMIT = 3


def keep():
    return 1


def target(values):
    total = 0
    for v in values:
        total += v
    return json.dumps(total)
"""


def test_splice_replaces_target_and_adds_helpers() -> None:
    code = (
        "def _sum(values):\n    return sum(values)\n\n\n"
        "def target(values):\n    return json.dumps(_sum(values))\n"
    )
    result = splice(SOURCE, "target", extract_code(f"Here:\n```python\n{code}```"), 5)
    assert "def keep():" in result and "_sum(values)" in result
    assert result.count("def target") == 1 and "total += v" not in result


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ("def other():\n    pass\n", "missing"),
        ("import os\ndef target(v):\n    pass\n", "only function definitions"),
        ("def target(v):\n" + "    x = 1\n" * 6, "lines >"),
        ("def target(v):\n    pass\ndef helper():\n    pass\n", "private"),
        ("def target(v):\n    pass\ndef _x():\n    pass\ndef _x():\n    pass\n", "twice"),
        ("def target(v):\n    pass\ndef keep():\n    pass\n", "private"),
        ("def target(v):\n    pass\ndef LIMIT():\n    pass\n", "private"),
        ("def target(v:\n", "does not parse"),
    ],
)
def test_splice_rejects_unsafe_output(code: str, reason: str) -> None:
    with pytest.raises(SpliceError, match=reason):
        splice(SOURCE, "target", code, 5)


def test_splice_refuses_overwriting_existing_private_names() -> None:
    source = SOURCE + "\n\ndef _existing():\n    return 2\n"
    with pytest.raises(SpliceError, match="overwrite"):
        splice(source, "target", "def target(v):\n    pass\ndef _existing():\n    pass\n", 5)
    with pytest.raises(SpliceError, match="module-level"):
        splice(SOURCE, "nope", "def nope():\n    pass\n", 5)
    assert extract_code("<think>x</think>def target(v):\n    pass") == "def target(v):\n    pass"
