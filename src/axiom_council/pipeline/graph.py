"""The M4 pipeline as a LangGraph state graph: plan → build → gate → critique → arbitrate,
with a bounded builder retry loop and a director approval pause.

LangGraph only runs the graph and checkpoints its state. Every node is a `GuardedNode`;
arbitration is deterministic code over gate results and the critic's verdict.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from axiom_council.audit import AuditLog
from axiom_council.contracts import TaskSpec, utcnow
from axiom_council.inbox import Inbox
from axiom_council.pipeline.approval import ApprovalSteps, patch_hash
from axiom_council.pipeline.guard import Guard, GuardedNode, PipelineState, Step
from axiom_council.policy import PolicyEngine
from axiom_council.settings import PipelineSettings
from axiom_council.tasks import TaskStore

# Tracing would send prompts to a hosted service; the council never enables it.
_NO_TRACING = {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}


@dataclass(frozen=True)
class AgentSteps:
    """The model- and sandbox-backed work; M4 fills these with routed agents and gates."""

    plan: Step
    build: Step
    gate: Step
    critique: Step


@dataclass(frozen=True)
class PipelineRun:
    task_id: str
    state: PipelineState
    paused: bool


def _build_with_hash(step: Step) -> Step:
    """Council code, not the builder, hashes the patch and counts attempts."""

    def build(state: PipelineState) -> dict[str, Any]:
        update = step(state)
        patch = str(update.get("patch", ""))
        return {
            **update,
            "artifact_hash": patch_hash(patch),
            "attempts": state.get("attempts", 0) + 1,
        }

    return build


def arbitrate(state: PipelineState, max_attempts: int) -> str:
    """Gates are authoritative; the critic must also approve. Otherwise retry or stop."""
    if state.get("gate_passed") and state.get("critic_approves"):
        return "request_approval"
    if state.get("attempts", 0) < max_attempts:
        return "build"
    return "give_up"


class GraphBuilder:
    """Accepts only guarded nodes, so nothing can join the graph around the policy check."""

    def __init__(self) -> None:
        self._graph: StateGraph[PipelineState] = StateGraph(PipelineState)

    def add(self, node: GuardedNode) -> None:
        if not isinstance(node, GuardedNode):
            raise TypeError(f"refusing unguarded node {node!r}")
        self._graph.add_node(node.name, node)

    @property
    def graph(self) -> StateGraph[PipelineState]:
        return self._graph


@dataclass
class PipelineDeps:
    policy: PolicyEngine
    audit: AuditLog
    inbox: Inbox
    tasks: TaskStore
    settings: PipelineSettings
    clock: Callable[[], datetime] = utcnow


def _wire(builder: GraphBuilder, max_attempts: int) -> None:
    g = builder.graph
    g.add_edge(START, "plan")
    g.add_edge("plan", "build")
    g.add_edge("build", "gate")
    g.add_edge("gate", "critique")
    g.add_conditional_edges(
        "critique",
        lambda s: arbitrate(s, max_attempts),
        {"request_approval": "request_approval", "build": "build", "give_up": "give_up"},
    )
    g.add_edge("request_approval", "await_decision")
    g.add_edge("await_decision", END)
    g.add_edge("give_up", END)


class Pipeline:
    def __init__(self, deps: PipelineDeps, steps: AgentSteps, checkpoint_path: Path):
        os.environ.update(_NO_TRACING)
        self._deps = deps
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
        self._graph = self._compile(steps, SqliteSaver(self._conn))

    def close(self) -> None:
        self._conn.close()

    def _compile(self, steps: AgentSteps, saver: SqliteSaver) -> CompiledStateGraph[Any]:
        d = self._deps
        guard = Guard(d.policy, d.audit, lambda tid: d.tasks.get(tid).spec, d.settings.nodes)
        approval = ApprovalSteps(d.inbox, d.audit, d.settings.approval_action, d.clock)
        builder = GraphBuilder()
        builder.add(guard.node("plan", steps.plan))
        builder.add(guard.node("build", _build_with_hash(steps.build)))
        builder.add(guard.node("gate", steps.gate))
        builder.add(guard.node("critique", steps.critique))
        builder.add(guard.internal("request_approval", approval.request))
        builder.add(guard.internal("await_decision", approval.await_decision))
        builder.add(guard.internal("give_up", lambda s: {"outcome": "failed"}))
        _wire(builder, d.settings.max_build_attempts)
        return builder.graph.compile(checkpointer=saver)

    @staticmethod
    def _config(task_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": task_id}}

    def _run(self, task_id: str, payload: Any) -> PipelineRun:
        config = self._config(task_id)
        self._graph.invoke(payload, config)
        snapshot = self._graph.get_state(config)
        return PipelineRun(task_id, cast(PipelineState, snapshot.values), bool(snapshot.next))

    def start(self, task: TaskSpec, user_input: str) -> PipelineRun:
        self._deps.tasks.create(task, user_input, "dispatcher", 0)
        self._deps.audit.append("pipeline_started", task.task_id, task.actor)
        return self._run(task.task_id, PipelineState(task_id=task.task_id, objective=user_input))

    def resume(self, task_id: str) -> PipelineRun:
        """Continue after a director decision (interrupt) or after a crash (last checkpoint)."""
        snapshot = self._graph.get_state(self._config(task_id))
        paused = any(task.interrupts for task in snapshot.tasks)
        return self._run(task_id, Command(resume=True) if paused else None)
