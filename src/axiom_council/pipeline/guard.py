"""Guarded nodes: the only way work enters a pipeline graph.

Every node is built here. Before a node's step runs, the policy engine checks the node's
configured role and capability against the task; every start, pause, denial, failure and
completion is written to the audit log. The graph runner never sees an unguarded callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypedDict

from langgraph.errors import GraphInterrupt

from axiom_council.audit import AuditLog
from axiom_council.contracts import Capability, TaskSpec
from axiom_council.policy import PolicyEngine
from axiom_council.settings import NodeGrant


class PipelineState(TypedDict, total=False):
    task_id: str
    objective: str
    plan: str
    patch: str
    artifact_hash: str
    attempts: int
    gate_passed: bool
    gate_report: str
    gate_checks: dict[str, bool]
    critique: str
    critic_approves: bool
    critic_output: str | None
    critic_malformed: bool
    findings: list[dict[str, Any]]
    unsupported: list[dict[str, str]]
    answers: list[dict[str, Any]]
    unsupported_findings: int
    unanswered_high_findings: int
    route: str
    failure_reason: str
    inbox_item: str
    approved_by: str
    outcome: str


Step = Callable[[PipelineState], dict[str, Any]]


class PolicyDenied(PermissionError):
    pass


@dataclass(frozen=True)
class GuardedNode:
    """A step bound to its role and capability. capability=None marks council-internal code
    (approval bookkeeping) that calls no model and touches no sandbox; it is still audited."""

    name: str
    role: str
    capability: Capability | None
    step: Step
    guard: Guard

    def __call__(self, state: PipelineState) -> dict[str, Any]:
        return self.guard.run(self, state)


class Guard:
    def __init__(
        self,
        policy: PolicyEngine,
        audit: AuditLog,
        task_for: Callable[[str], TaskSpec],
        grants: dict[str, NodeGrant],
    ):
        self._policy = policy
        self._audit = audit
        self._task_for = task_for
        self._grants = grants

    def node(self, name: str, step: Step) -> GuardedNode:
        """A work node; its role and capability come from config, never from the caller."""
        grant = self._grants.get(name)
        if grant is None:
            raise ValueError(f"pipeline node {name!r} has no grant in the pipeline settings")
        return GuardedNode(name, grant.role, grant.capability, step, self)

    def internal(self, name: str, step: Step) -> GuardedNode:
        return GuardedNode(name, "council", None, step, self)

    def run(self, node: GuardedNode, state: PipelineState) -> dict[str, Any]:
        task_id = state["task_id"]
        if node.capability is not None:
            try:
                self._policy.check_action(self._task_for(task_id), node.role, node.capability)
            except PermissionError as exc:
                self._audit.append(
                    "node_denied", task_id, node.role, node=node.name, error=str(exc)
                )
                raise PolicyDenied(f"{node.name}: {exc}") from exc
        self._audit.append("node_started", task_id, node.role, node=node.name)
        try:
            update = node.step(state)
        except GraphInterrupt:
            self._audit.append("node_paused", task_id, node.role, node=node.name)
            raise
        except Exception as exc:
            self._audit.append("node_failed", task_id, node.role, node=node.name, error=str(exc))
            raise
        self._audit.append(
            "node_completed", task_id, node.role, node=node.name, updated=sorted(update)
        )
        return update
