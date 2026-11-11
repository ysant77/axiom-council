"""A9 and A10: only supported findings reach arbitration, and every supported high finding
gets an explicit answer before a run may ask the director for approval."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from axiom_council.audit import AuditLog
from axiom_council.contracts import Capability, TaskSpec
from axiom_council.inbox import Inbox
from axiom_council.pipeline import (
    AgentSteps,
    Pipeline,
    PipelineDeps,
    PipelineRun,
    PipelineState,
)
from axiom_council.pipeline.findings import (
    Finding,
    FindingAnswer,
    FindingsMalformed,
    answer_high_findings,
    parse_findings,
    support_problem,
    unanswered_high,
)
from axiom_council.pipeline.hunks import changed_ranges
from axiom_council.policy import PolicyEngine
from axiom_council.registry import Registry
from axiom_council.settings import NodeGrant
from axiom_council.tasks import TaskStore

WORK = frozenset({Capability.LLM_INFERENCE, Capability.PROPOSE_PATCH, Capability.RUN_SANDBOX})

PATCH = """\
diff --git a/src/app.py b/src/app.py
--- a/src/app.py
+++ b/src/app.py
@@ -10,3 +10,5 @@ def main():
 keep = 1
--- removed line that looks like a header
+++ added line that looks like a header
+extra = 2
+more = 3
 tail = 4
@@ -40 +42,2 @@
-old = 5
+new = 5
+newer = 6
\\ No newline at end of file
diff --git a/docs/gone.md b/docs/gone.md
--- a/docs/gone.md
+++ /dev/null
@@ -1,3 +0,0 @@
-a
-b
-c
"""


def finding(**overrides: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "finding_id": "F1",
        "severity": "high",
        "fault_class": "logic",
        "path": "src/app.py",
        "start_line": 11,
        "end_line": 12,
        "claim": "extra is never read",
        "evidence": "line 11 assigns extra; nothing reads it",
    }
    body.update(overrides)
    return body


class Agents:
    """Scripted stand-ins: a builder that emits a real diff and a critic that emits JSON."""

    def __init__(self, critic_output: str | None, gate_checks: dict[str, bool] | None = None):
        self.calls = {"build": 0, "critique": 0}
        self._critic_output = critic_output
        self._gate_checks = gate_checks or {}
        self.answers: list[dict[str, Any]] | None = None
        self.patch = PATCH

    def steps(self) -> AgentSteps:
        answer = self.answer if self.answers is not None else None
        return AgentSteps(self.plan, self.build, self.gate, self.critique, answer)

    def plan(self, state: PipelineState) -> dict[str, Any]:
        return {"plan": "split the function"}

    def build(self, state: PipelineState) -> dict[str, Any]:
        self.calls["build"] += 1
        return {"patch": self.patch}

    def gate(self, state: PipelineState) -> dict[str, Any]:
        return {"gate_passed": all(self._gate_checks.values()), "gate_checks": self._gate_checks}

    def critique(self, state: PipelineState) -> dict[str, Any]:
        self.calls["critique"] += 1
        if self._critic_output is None:
            return {"critique": "no structured output", "critic_approves": True}
        return {"critic_output": self._critic_output}

    def answer(self, state: PipelineState) -> dict[str, Any]:
        return {"answers": self.answers}


@pytest.fixture
def deps(registry: Registry, tmp_path: Path) -> Iterator[PipelineDeps]:
    tasks = TaskStore(tmp_path / "tasks.sqlite3")
    yield PipelineDeps(
        policy=PolicyEngine(registry.workspaces, registry.roles),
        audit=AuditLog(tmp_path / "audit.jsonl"),
        inbox=Inbox(tmp_path / "inbox"),
        tasks=tasks,
        settings=registry.settings.pipeline,
    )
    tasks.close()


@pytest.fixture
def task(make_task: Callable[..., TaskSpec]) -> TaskSpec:
    return make_task(workspace_id="council", allowed_capabilities=WORK)


def _run(deps: PipelineDeps, agents: Agents, task: TaskSpec, tmp_path: Path) -> PipelineRun:
    pipe = Pipeline(deps, agents.steps(), tmp_path / "checkpoints.sqlite3")
    try:
        return pipe.start(task, "Split evaluate() into helpers")
    finally:
        pipe.close()


def _events(deps: PipelineDeps, task: TaskSpec, name: str) -> list[dict[str, Any]]:
    return [r["data"] for r in deps.audit.for_task(task.task_id) if r["event"] == name]


def _grant_arbiter(deps: PipelineDeps) -> None:
    grant = NodeGrant(role="reviewer", capability=Capability.LLM_INFERENCE)
    nodes = {**deps.settings.nodes, "answer_findings": grant}
    deps.settings = deps.settings.model_copy(update={"nodes": nodes})


def test_unsupported_findings_are_dropped_counted_and_audited(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    output = json.dumps(
        [
            finding(finding_id="F1", path="src/other.py"),  # file not in the patch
            finding(finding_id="F2", start_line=30, end_line=31),  # outside every hunk
            finding(finding_id="F3", path="src/x.py", evidence="gate:tests"),  # check passed
            finding(finding_id="F4", severity="low", start_line=43, end_line=43),
        ]
    )
    run = _run(deps, Agents(output, {"tests": True}), task, tmp_path)
    assert run.paused and "inbox_item" in run.state  # no supported high finding blocks it
    assert [f["finding_id"] for f in run.state["findings"]] == ["F4"]
    assert run.metrics == {"unsupported_findings": 3, "unanswered_high_findings": 0}
    reasons = {u["finding_id"]: u["reason"] for u in run.state["unsupported"]}
    assert "not in the patch" in reasons["F1"] and "outside" in reasons["F2"]
    assert "gate check 'tests' did not fail" in reasons["F3"]
    [screened] = _events(deps, task, "findings_screened")
    assert screened["unsupported"] == run.state["unsupported"] and screened["supported"] == ["F4"]
    [arbitration] = _events(deps, task, "arbitration")
    assert arbitration["unsupported_findings"] == 3 and arbitration["route"] == "request_approval"


def test_finding_citing_a_failed_gate_check_is_supported_and_accepted(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    output = json.dumps([finding(path="tests/test_app.py", evidence="gate:tests")])
    agents = Agents(output, {"tests": False, "lint": True})
    run = _run(deps, agents, task, tmp_path)
    assert run.metrics["unsupported_findings"] == 0 and len(run.state["findings"]) == 1
    [answer] = run.state["answers"]
    assert answer["decision"] == "accept" and "tests" in answer["reason"]
    assert run.state["outcome"] == "failed" and agents.calls["build"] == 2


def test_supported_high_finding_without_an_answer_fails_closed_before_approval(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    _grant_arbiter(deps)
    agents = Agents(json.dumps([finding(), finding(finding_id="F2", severity="medium")]))
    agents.answers = []  # a routed arbiter that answers nothing
    run = _run(deps, agents, task, tmp_path)
    assert not run.paused and "inbox_item" not in run.state
    assert run.state["outcome"] == "failed" and agents.calls["build"] == 1  # no retry
    assert "unanswered high findings" in run.state["failure_reason"]
    assert "F1" in run.state["failure_reason"] and "F2" not in run.state["failure_reason"]
    assert run.metrics == {"unsupported_findings": 0, "unanswered_high_findings": 1}
    [arbitration] = _events(deps, task, "arbitration")
    assert arbitration["unanswered"] == ["F1"] and arbitration["route"] == "give_up"
    assert not _events(deps, task, "approval_verified")


@pytest.mark.parametrize(
    "answer",
    [
        {"finding_id": "F1", "decision": "accept", "reason": "  "},
        {"finding_id": "F1", "decision": "reject", "reason": "fine", "check": "lint"},
        {"finding_id": "F1", "decision": "reject", "reason": "fine", "check": "unknown"},
        {"finding_id": "F1", "decision": "maybe", "reason": "not a decision"},
        {"finding_id": "F9", "decision": "accept", "reason": "answers another finding"},
    ],
)
def test_answers_that_do_not_count_leave_the_finding_unanswered(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path, answer: dict[str, Any]
) -> None:
    _grant_arbiter(deps)
    agents = Agents(json.dumps([finding()]), {"tests": True, "lint": False})
    agents.answers = [answer]
    run = _run(deps, agents, task, tmp_path)
    assert run.state["outcome"] == "failed" and run.metrics["unanswered_high_findings"] == 1


def test_routed_arbiter_rejection_citing_a_passing_check_reaches_approval(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    _grant_arbiter(deps)
    agents = Agents(json.dumps([finding()]), {"tests": True})
    agents.answers = [
        {"finding_id": "F1", "decision": "reject", "reason": "tests cover it", "check": "tests"}
    ]
    run = _run(deps, agents, task, tmp_path)
    assert run.paused and run.metrics["unanswered_high_findings"] == 0


def test_routed_arbiter_needs_a_node_grant(deps: PipelineDeps, tmp_path: Path) -> None:
    agents = Agents("[]")
    agents.answers = []
    with pytest.raises(ValueError, match="no grant"):
        Pipeline(deps, agents.steps(), tmp_path / "checkpoints.sqlite3")


def test_accepted_high_finding_loops_to_build_and_stops_at_the_attempt_limit(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    agents = Agents(json.dumps([finding()]), {"tests": True})
    run = _run(deps, agents, task, tmp_path)
    limit = deps.settings.max_build_attempts
    assert agents.calls == {"build": limit, "critique": limit} and not run.paused
    assert (
        run.state["outcome"] == "failed" and "accepted high findings" in run.state["failure_reason"]
    )
    routes = [a["route"] for a in _events(deps, task, "arbitration")]
    assert routes == ["build"] * (limit - 1) + ["give_up"]


def test_deterministic_arbiter_rejects_a_finding_whose_cited_gate_check_passed(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    output = json.dumps([finding(fault_class="logic", evidence="gate:tests")])
    run = _run(deps, Agents(output, {"tests": True}), task, tmp_path)
    [answer] = run.state["answers"]
    assert answer["decision"] == "reject" and answer["check"] == "tests"
    assert run.paused and run.metrics["unanswered_high_findings"] == 0


@pytest.mark.parametrize(
    "output",
    [
        None,  # the critic returned a verdict only
        "the patch looks fine",
        json.dumps({"findings": []}),
        json.dumps([finding(severity="critical")]),
        json.dumps([finding(extra_field=1)]),
        json.dumps([finding(), finding()]),  # duplicate finding ids
        json.dumps([finding(finding_id=f"F{i}") for i in range(51)]),  # above max_findings
    ],
)
def test_malformed_critic_output_yields_no_findings_and_never_approves(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path, output: str | None
) -> None:
    agents = Agents(output, {"tests": True})
    run = _run(deps, agents, task, tmp_path)
    assert run.state["outcome"] == "failed" and not run.paused and "inbox_item" not in run.state
    assert run.state["findings"] == [] and run.state["critic_malformed"] is True
    assert "critic output malformed" in run.state["failure_reason"]
    assert agents.calls["build"] == deps.settings.max_build_attempts
    assert len(_events(deps, task, "critic_output_malformed")) == agents.calls["critique"]
    assert deps.audit.verify() == len(deps.audit.for_task(task.task_id))


def test_empty_findings_list_passes_with_zero_metrics(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    run = _run(deps, Agents("[]", {"tests": True}), task, tmp_path)
    assert run.paused and run.metrics == {"unsupported_findings": 0, "unanswered_high_findings": 0}


def test_failing_gate_without_findings_retries_then_fails_with_a_reason(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    run = _run(deps, Agents("[]", {"tests": False}), task, tmp_path)
    assert run.state["outcome"] == "failed" and "gate failed" in run.state["failure_reason"]


def test_verdict_only_critic_still_works_behind_the_switch(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    deps.settings = deps.settings.model_copy(update={"gated_findings": False})
    run = _run(deps, Agents(None, {"tests": True}), task, tmp_path)
    assert run.paused  # the verdict and the gate decide, as before
    assert not _events(deps, task, "findings_screened")
    assert run.metrics == {"unsupported_findings": 0, "unanswered_high_findings": 0}


def test_changed_ranges_reads_new_side_hunks_and_deleted_files() -> None:
    assert changed_ranges(PATCH) == {"src/app.py": [(10, 14), (42, 43)], "docs/gone.md": [(1, 3)]}
    assert changed_ranges("diff attempt 1") == {}
    pure_delete = "--- a/x.py\n+++ b/x.py\n@@ -5,2 +4,0 @@\n-a\n-b\n"
    assert changed_ranges(pure_delete) == {"x.py": [(4, 4)]}
    new_file = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+x = 1\n"
    assert changed_ranges(new_file) == {"new.py": [(1, 1)]}
    marker_mid_hunk = "--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-a\n\\ No newline\n+b\n@@ -9 +9 @@\n"
    assert changed_ranges(marker_mid_hunk) == {"y.py": [(1, 1), (9, 9)]}
    assert changed_ranges("--- /dev/null\n+++ /dev/null\n@@ -0,0 +0,0 @@\n") == {}


def test_support_problem_rules() -> None:
    ranges = changed_ranges(PATCH)
    ok = Finding.model_validate(finding())
    assert support_problem(ok, ranges, {}) is None
    dotted = ok.model_copy(update={"path": "./src/app.py"})
    assert support_problem(dotted, ranges, {}) is None
    inverted = ok.model_copy(update={"start_line": 12, "end_line": 11})
    assert "invalid line range" in str(support_problem(inverted, ranges, {}))
    blank = ok.model_copy(update={"claim": " "})
    assert "empty claim" in str(support_problem(blank, ranges, {}))
    straddles = ok.model_copy(update={"start_line": 14, "end_line": 42})
    assert "outside" in str(support_problem(straddles, ranges, {}))


def test_pure_functions_cover_and_answer() -> None:
    with pytest.raises(FindingsMalformed):
        parse_findings("[1]", limit=5)
    high = Finding.model_validate(finding())
    low = Finding.model_validate(finding(finding_id="F2", severity="low"))
    [answer] = answer_high_findings([high, low], {})
    assert answer == FindingAnswer(
        finding_id="F1", decision="accept", reason=answer.reason, check=""
    )
    assert unanswered_high([high, low], [answer], {}) == []
    assert unanswered_high([high, low], [], {}) == ["F1"]


@pytest.mark.parametrize("output", ["[]", json.dumps([finding()])])
def test_unreadable_patch_never_reaches_approval(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path, output: str
) -> None:
    agents = Agents(output, {"tests": True})
    agents.patch = "def f(): pass"  # not a diff, so no finding can be located in it
    run = _run(deps, agents, task, tmp_path)
    limit = deps.settings.max_build_attempts
    assert not run.paused and "inbox_item" not in run.state and run.state["outcome"] == "failed"
    assert run.state["patch_unreadable"] is True
    assert "patch unreadable" in run.state["failure_reason"] and agents.calls["build"] == limit
    events = _events(deps, task, "patch_unreadable")
    assert len(events) == limit and events[0]["unscreened"] == [
        f["finding_id"] for f in json.loads(output)
    ]
    assert run.metrics["unsupported_findings"] == 0  # the critic is not blamed for the patch


def test_fault_class_is_never_read_as_a_gate_check_name() -> None:
    labelled = Finding.model_validate(finding(fault_class="tests"))
    [answer] = answer_high_findings([labelled], {"tests": True, "lint": True})
    assert answer.decision == "accept" and answer.check == ""
    cites_passing = labelled.model_copy(update={"evidence": "gate:lint"})
    [answer] = answer_high_findings([cites_passing], {"tests": True, "lint": True})
    assert answer.decision == "reject" and answer.check == "lint"


@pytest.mark.parametrize(
    ("path", "problem"),
    [
        ("src/app.py", None),
        ("./src/app.py", None),
        ("src/./app.py", None),
        ("src/../src/app.py", None),
        ("b/src/app.py", None),
        ("a/src/app.py", None),
        ("c/src/app.py", "not in the patch"),
        ("/src/app.py", "absolute path"),
        ("../src/app.py", "outside the repository root"),
        ("src/../../app.py", "outside the repository root"),
        ("", "not in the patch"),
    ],
)
def test_finding_paths_are_normalised_before_lookup(path: str, problem: str | None) -> None:
    located = Finding.model_validate(finding(path=path))
    found = support_problem(located, changed_ranges(PATCH), {})
    if problem is None:
        assert found is None
    else:
        assert found is not None and problem in found


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.py"])
def test_paths_outside_the_root_stay_unsupported_even_with_a_failed_check(path: str) -> None:
    cited = Finding.model_validate(finding(path=path, evidence="gate:tests"))
    assert support_problem(cited, changed_ranges(PATCH), {"tests": False}) is not None


def test_prefix_is_stripped_only_when_the_full_path_is_not_in_the_patch() -> None:
    ranges = {"b/x.py": [(1, 5)], "x.py": [(100, 100)]}
    inside_b = Finding.model_validate(finding(path="b/x.py", start_line=1, end_line=2))
    assert support_problem(inside_b, ranges, {}) is None
    stripped = Finding.model_validate(finding(path="a/x.py", start_line=100, end_line=100))
    assert support_problem(stripped, ranges, {}) is None
