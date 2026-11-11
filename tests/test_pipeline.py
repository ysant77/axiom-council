"""ADR 0004 spike: LangGraph runs the pipeline only behind our policy, audit and approvals.

Each test is one kill criterion (K1-K4). If any fails, LangGraph is dropped.
"""

from __future__ import annotations

import socket
from collections.abc import Callable, Iterator
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from axiom_council.audit import AuditLog
from axiom_council.contracts import Capability, TaskSpec, utcnow
from axiom_council.inbox import Inbox
from axiom_council.pipeline import (
    AgentSteps,
    ApprovalRefused,
    GraphBuilder,
    Pipeline,
    PipelineDeps,
    PipelineState,
    PolicyDenied,
    arbitrate,
)
from axiom_council.policy import PolicyEngine
from axiom_council.registry import Registry
from axiom_council.tasks import TaskStore

WORK = frozenset({Capability.LLM_INFERENCE, Capability.PROPOSE_PATCH, Capability.RUN_SANDBOX})


class Agents:
    """Scripted stand-ins for routed agents; counts calls so re-runs are visible."""

    def __init__(self, gate_results: list[bool], critic: bool = True):
        self.calls: dict[str, int] = {"plan": 0, "build": 0, "gate": 0, "critique": 0}
        self._gates = gate_results
        self._critic = critic
        self.crash_gate_once = False

    def steps(self) -> AgentSteps:
        return AgentSteps(self.plan, self.build, self.gate, self.critique)

    def plan(self, state: PipelineState) -> dict[str, Any]:
        self.calls["plan"] += 1
        return {"plan": f"plan for {state['objective']}"}

    def build(self, state: PipelineState) -> dict[str, Any]:
        self.calls["build"] += 1
        return {"patch": f"diff attempt {self.calls['build']}"}

    def gate(self, state: PipelineState) -> dict[str, Any]:
        self.calls["gate"] += 1
        if self.crash_gate_once:
            self.crash_gate_once = False
            raise RuntimeError("power cut")
        return {"gate_passed": self._gates[min(self.calls["gate"], len(self._gates)) - 1]}

    def critique(self, state: PipelineState) -> dict[str, Any]:
        self.calls["critique"] += 1
        return {"critic_approves": self._critic, "critique": "looks fine", "critic_output": "[]"}


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


def _pipeline(deps: PipelineDeps, agents: Agents, tmp_path: Path) -> Pipeline:
    return Pipeline(deps, agents.steps(), tmp_path / "checkpoints.sqlite3")


def _approve(deps: PipelineDeps, item_id: str, hours: float = 1) -> None:
    deps.inbox.approve(item_id, "director", timedelta(hours=hours), "patch reviewed")


def test_happy_path_pauses_for_director_then_completes(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    agents = Agents(gate_results=[False, True])  # first patch fails the gate, retry passes
    pipe = _pipeline(deps, agents, tmp_path)
    run = pipe.start(task, "Split evaluate() into helpers")
    assert run.paused and run.state["attempts"] == 2 and "inbox_item" in run.state
    _approve(deps, run.state["inbox_item"])
    done = pipe.resume(task.task_id)
    pipe.close()
    assert not done.paused and done.state["outcome"] == "approved"
    events = [r["event"] for r in deps.audit.for_task(task.task_id)]
    assert events.count("node_started") >= 8 and events[-1] == "node_completed"
    assert "approval_verified" in events and deps.audit.verify() == len(events)


def test_k1_every_node_is_policy_checked_and_unguarded_nodes_are_refused(
    deps: PipelineDeps, make_task: Callable[..., TaskSpec], tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="unguarded"):
        GraphBuilder().add(lambda s: {})  # type: ignore[arg-type]
    narrow = make_task(  # the task does not allow the builder's capability
        workspace_id="council", allowed_capabilities=frozenset({Capability.LLM_INFERENCE})
    )
    agents = Agents(gate_results=[True])
    pipe = _pipeline(deps, agents, tmp_path)
    with pytest.raises(PolicyDenied, match="build"):
        pipe.start(narrow, "Refactor")
    pipe.close()
    assert agents.calls["build"] == 0  # denied before the step ran
    assert deps.audit.for_task(narrow.task_id)[-1]["event"] == "node_denied"


def test_k1_node_without_a_configured_grant_cannot_be_built(
    deps: PipelineDeps, tmp_path: Path
) -> None:
    nodes = {k: v for k, v in deps.settings.nodes.items() if k != "critique"}
    deps.settings = deps.settings.model_copy(update={"nodes": nodes})
    with pytest.raises(ValueError, match="no grant"):
        _pipeline(deps, Agents([True]), tmp_path)


def test_k2_approval_is_bound_to_the_exact_patch(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    pipe = _pipeline(deps, Agents([True]), tmp_path)
    run = pipe.start(task, "Refactor")
    with pytest.raises(ApprovalRefused):
        pipe.resume(task.task_id)  # no decision yet
    _approve(deps, run.state["inbox_item"])
    config: Any = {"configurable": {"thread_id": task.task_id}}
    pipe._graph.update_state(config, {"patch": "diff swapped after approval"})
    # Editing checkpoint state clears LangGraph's pending interrupt, so the graph pauses
    # again instead of finishing; the next resume then fails the hash check.
    assert pipe.resume(task.task_id).paused
    with pytest.raises(ApprovalRefused, match="no valid approval"):
        pipe.resume(task.task_id)
    pipe.close()
    events = [r["event"] for r in deps.audit.for_task(task.task_id)]
    assert events.count("approval_invalid") == 2 and "approval_verified" not in events


def test_k2_expired_and_rejected_decisions(
    deps: PipelineDeps, make_task: Callable[..., TaskSpec], tmp_path: Path
) -> None:
    later = utcnow() + timedelta(hours=3)
    deps.clock = lambda: later
    pipe = _pipeline(deps, Agents([True]), tmp_path)
    first = make_task(workspace_id="council", allowed_capabilities=WORK)
    run = pipe.start(first, "Refactor")
    _approve(deps, run.state["inbox_item"], hours=1)
    with pytest.raises(ApprovalRefused):
        pipe.resume(first.task_id)  # token expired by the pipeline's clock
    second = make_task(workspace_id="council", allowed_capabilities=WORK)
    held = pipe.start(second, "Refactor again")
    deps.inbox.reject(held.state["inbox_item"], "not this approach")
    assert pipe.resume(second.task_id).state["outcome"] == "rejected"
    pipe.close()


def test_k3_runs_fully_offline(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    attempts: list[Any] = []
    real_connect = socket.socket.connect

    def loopback_only(self: socket.socket, address: Any) -> Any:
        attempts.append(address)
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise OSError(f"egress blocked: {address}")
        return real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", loopback_only)
    pipe = _pipeline(deps, Agents([True]), tmp_path)
    run = pipe.start(task, "Refactor")
    _approve(deps, run.state["inbox_item"])
    assert pipe.resume(task.task_id).state["outcome"] == "approved"
    pipe.close()
    assert attempts == []  # not even a loopback connection was attempted


def test_k4_crash_resumes_from_checkpoint_without_rerunning_finished_nodes(
    deps: PipelineDeps, task: TaskSpec, tmp_path: Path
) -> None:
    agents = Agents([True])
    agents.crash_gate_once = True
    first = _pipeline(deps, agents, tmp_path)
    with pytest.raises(RuntimeError, match="power cut"):
        first.start(task, "Refactor")
    first.close()  # the process dies after build, inside the gate

    second = _pipeline(deps, agents, tmp_path)  # fresh process, same checkpoint file
    run = second.resume(task.task_id)
    second.close()
    assert run.paused and agents.calls == {"plan": 1, "build": 1, "gate": 2, "critique": 1}
    assert deps.audit.verify() == len(deps.audit.for_task(task.task_id))


def test_arbitration_is_deterministic() -> None:
    passing = PipelineState(gate_passed=True, critic_approves=True, attempts=1)
    assert arbitrate(passing, 2) == "request_approval"
    assert (
        arbitrate(PipelineState(gate_passed=True, critic_approves=False, attempts=1), 2) == "build"
    )
    assert arbitrate(PipelineState(gate_passed=False, attempts=2), 2) == "give_up"
