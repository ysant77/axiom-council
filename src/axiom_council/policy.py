"""Deterministic policy engine. No model output can widen what is allowed here."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import Field

from axiom_council.contracts import Capability, DataClass, RiskClass, Strict, TaskSpec

# Capabilities that always need a human approval token, whatever the workspace says.
ALWAYS_HUMAN: frozenset[Capability] = frozenset(
    {
        Capability.WRITE_WORKSPACE,
        Capability.EXTERNAL_MESSAGE,
        Capability.MERGE,
        Capability.SPEND_MONEY,
        Capability.NETWORK_EGRESS,
    }
)

_RISK_ORDER = [RiskClass.LOW, RiskClass.MEDIUM, RiskClass.HIGH, RiskClass.CRITICAL]


class RolePolicy(Strict):
    """What one council role may ever do. A role never grants itself more."""

    name: str
    capabilities: frozenset[Capability]
    can_ask: tuple[str, ...] = ("dispatcher", "director")
    description: str = ""


class WorkspacePolicy(Strict):
    workspace_id: str
    max_capabilities: frozenset[Capability]
    max_data_class: DataClass = DataClass.INTERNAL
    max_tokens_per_task: int = Field(default=200_000, gt=0)
    max_usd_per_task: float = Field(default=0.0, ge=0)
    approval_at_or_above: RiskClass = RiskClass.MEDIUM


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    needs_approval: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


_DATA_ORDER = [DataClass.PUBLIC, DataClass.INTERNAL, DataClass.PRIVATE]


class PolicyEngine:
    def __init__(self, workspaces: dict[str, WorkspacePolicy], roles: dict[str, RolePolicy]):
        self._workspaces = workspaces
        self._roles = roles

    def role(self, name: str) -> RolePolicy:
        try:
            return self._roles[name]
        except KeyError:
            raise PermissionError(f"unknown role {name!r}") from None

    def check_task(self, task: TaskSpec) -> PolicyDecision:
        """Fail closed: any unknown or out-of-scope request is denied with reasons."""
        ws = self._workspaces.get(task.workspace_id)
        if ws is None:
            return PolicyDecision(False, False, (f"unknown workspace {task.workspace_id!r}",))

        reasons: list[str] = []
        excess = task.allowed_capabilities - ws.max_capabilities
        if excess:
            reasons.append(f"capabilities not permitted in workspace: {sorted(excess)}")
        if _DATA_ORDER.index(task.data_classification) > _DATA_ORDER.index(ws.max_data_class):
            reasons.append(
                f"data class {task.data_classification} exceeds workspace limit {ws.max_data_class}"
            )
        if task.budget.max_tokens > ws.max_tokens_per_task:
            reasons.append(
                f"token budget {task.budget.max_tokens} exceeds limit {ws.max_tokens_per_task}"
            )
        if task.budget.max_usd > ws.max_usd_per_task:
            reasons.append(f"usd budget {task.budget.max_usd} exceeds {ws.max_usd_per_task}")
        if reasons:
            return PolicyDecision(False, False, tuple(reasons))

        needs_approval = bool(task.allowed_capabilities & ALWAYS_HUMAN) or _RISK_ORDER.index(
            task.risk_class
        ) >= _RISK_ORDER.index(ws.approval_at_or_above)
        return PolicyDecision(True, needs_approval, ())

    def check_action(self, task: TaskSpec, role_name: str, capability: Capability) -> None:
        """Raise unless the role, the task and the workspace all permit the capability."""
        role = self.role(role_name)
        if capability not in role.capabilities:
            raise PermissionError(f"role {role_name!r} lacks {capability}")
        if capability in task.forbidden_capabilities:
            raise PermissionError(f"task forbids {capability}")
        if capability not in task.allowed_capabilities:
            raise PermissionError(f"task does not allow {capability}")
        ws = self._workspaces.get(task.workspace_id)
        if ws is None or capability not in ws.max_capabilities:
            raise PermissionError(f"workspace does not allow {capability}")
