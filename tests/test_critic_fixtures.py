"""Validity of the critic admission fixtures and of the diff parser they rely on."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from axiom_council.contracts import TaskClass
from axiom_council.diffs import DiffError, change_blocks, parse_diff
from axiom_council.evals.critic_contracts import CriticCase, load_cases
from axiom_council.registry import Registry
from axiom_council.settings import CriticAdmissionSettings

FIXTURES = Path(__file__).resolve().parents[1] / "evals/fixtures/critic_admission"
CLASSES = CriticAdmissionSettings().fault_classes


def _cases() -> tuple[CriticCase, ...]:
    return load_cases(FIXTURES, CLASSES)


def test_fixture_counts_ids_and_classes() -> None:
    cases = _cases()
    seeded = [c for c in cases if c.kind == "seeded"]
    clean = [c for c in cases if c.kind == "clean"]
    assert len(seeded) == 12 and len(clean) == 12
    assert len({c.id for c in cases}) == len(cases)
    classes = {c.key.fault_class for c in seeded if c.key is not None}
    assert len(classes) == 12 and classes <= set(CLASSES)
    assert all(c.key is None for c in clean)


def test_every_key_lies_inside_its_diff() -> None:
    for case in _cases():
        if case.key is None:
            continue
        key = case.key
        hunks = [h for h in parse_diff(case.diff) if h.path == key.path]
        assert hunks, case.id
        assert any(h.covers(key.start_line, key.end_line) for h in hunks), case.id


def test_every_diff_parses_with_consistent_hunk_counts() -> None:
    for case in _cases():
        hunks = parse_diff(case.diff)
        assert hunks, case.id
        assert all(any(line.op != " " for line in h.lines) for h in hunks), case.id


def _write(tmp: Path, seeded: list[dict[str, object]], clean: list[dict[str, object]]) -> None:
    (tmp / "seeded.yaml").write_text(yaml.safe_dump({"cases": seeded}), encoding="utf-8")
    (tmp / "clean.yaml").write_text(yaml.safe_dump({"cases": clean}), encoding="utf-8")


DIFF = "--- a/x.py\n+++ b/x.py\n@@ -10,2 +10,2 @@\n a = 1\n-b = 2\n+b = 3\n"


def _seeded(**key: object) -> dict[str, object]:
    base: dict[str, object] = {
        "fault_class": "off_by_one", "path": "x.py", "start_line": 11, "end_line": 11
    }  # fmt: skip
    return {"id": "s1", "title": "t", "diff": DIFF, "key": {**base, **key}}


def test_loader_accepts_a_valid_minimal_set(tmp_path: Path) -> None:
    _write(tmp_path, [_seeded()], [{"id": "c1", "title": "t", "diff": DIFF}])
    cases = load_cases(tmp_path, CLASSES)
    assert [c.kind for c in cases] == ["seeded", "clean"]


@pytest.mark.parametrize(
    "seeded",
    [
        _seeded(start_line=40, end_line=41),  # outside the hunk
        _seeded(path="y.py"),  # file not in the diff
        _seeded(fault_class="vibes"),  # not in the taxonomy
        {"id": "s1", "title": "t", "diff": DIFF},  # seeded case without a key
    ],
)
def test_loader_rejects_bad_keys(tmp_path: Path, seeded: dict[str, object]) -> None:
    _write(tmp_path, [seeded], [])
    with pytest.raises(ValueError):
        load_cases(tmp_path, CLASSES)


def test_loader_rejects_duplicate_ids_and_keyed_clean_cases(tmp_path: Path) -> None:
    _write(tmp_path, [_seeded()], [{"id": "s1", "title": "t", "diff": DIFF}])
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path, CLASSES)
    _write(tmp_path, [], [_seeded()])
    with pytest.raises(ValueError):
        load_cases(tmp_path, CLASSES)


@pytest.mark.parametrize(
    "diff",
    [
        "--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n a\n",  # body shorter than header
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n a\n+b\n",  # body longer than header
        "@@ -1,1 +1,1 @@\n a\n",  # hunk before any file header
        "--- a/x.py\n+++ b/x.py\n@@ bad @@\n a\n",  # malformed header
        "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,1 @@\n?a\n",  # unknown line marker
        "no diff here\n",
    ],
)
def test_parser_rejects_malformed_diffs(diff: str) -> None:
    with pytest.raises(DiffError):
        parse_diff(diff)


def test_parser_tracks_new_line_numbers_and_blocks() -> None:
    diff = (
        "--- a/src/m.py\n+++ b/src/m.py\n@@ -5,5 +5,4 @@ def f():\n"
        " a\n-b\n+B\n c\n-d\n\n\\ No newline at end of file\n"
        "--- a/gone.py\n+++ /dev/null\n@@ -1,1 +0,0 @@\n-x\n"
    )
    hunks = parse_diff(diff)
    first = hunks[0]
    assert first.path == "src/m.py" and (first.new_start, first.new_end) == (5, 8)
    assert [(ln.op, ln.new_line) for ln in first.lines] == [
        (" ", 5), ("-", 6), ("+", 6), (" ", 7), ("-", 8), (" ", 8)
    ]  # fmt: skip
    assert [[ln.text for ln in block] for block in change_blocks(first)] == [["b", "B"], ["d"]]
    assert hunks[1].path == "gone.py" and hunks[1].new_count == 0


def test_parser_accepts_git_extended_headers() -> None:
    diff = (
        "diff --git a/x.py b/x.py\nindex 83db48f..bf269f4 100644\n"
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/new.py b/new.py\nnew file mode 100644\nindex 0000000..e69de29\n"
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x\n"
    )
    assert [(h.path, h.new_start, h.new_end) for h in parse_diff(diff)] == [
        ("x.py", 1, 1), ("new.py", 1, 1)
    ]  # fmt: skip


def test_critic_prompt_does_not_shadow_the_reviewer_prompt(registry: Registry) -> None:
    review = registry.prompt_for("reviewer", TaskClass.REVIEW)
    assert review is not None and review.prompt_id == "reviewer.review"
    critic = registry.prompts[registry.settings.critic_admission.prompt_id]
    assert critic.role == "critic"
    assert set(registry.settings.critic_admission.fault_classes) == set(CLASSES)
