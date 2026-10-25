from __future__ import annotations

from collections.abc import Callable

import pytest

from axiom_council.contracts import Budget, Capability, DataClass, RiskClass, TaskSpec
from axiom_council.policy import PolicyEngine
from axiom_council.registry import Registry


@pytest.fixture
def engine(registry: Registry) -> PolicyEngine:
    return PolicyEngine(registry.workspaces, registry.roles)


def test_unknown_workspace_is_denied(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    decision = engine.check_task(make_task(workspace_id="nowhere"))
    assert not decision.allowed
    assert "unknown workspace" in decision.reasons[0]


def test_all_violations_are_reported(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    task = make_task(
        allowed_capabilities=frozenset({Capability.LLM_INFERENCE, Capability.MERGE}),
        data_classification=DataClass.PRIVATE,
        budget=Budget(max_tokens=500_000, max_wall_seconds=60, max_usd=5),
    )
    decision = engine.check_task(task)
    assert not decision.allowed
    assert len(decision.reasons) == 4


def test_low_risk_sandbox_task_runs_without_approval(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    decision = engine.check_task(make_task())
    assert decision.allowed and not decision.needs_approval


def test_risk_threshold_triggers_approval(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    decision = engine.check_task(make_task(workspace_id="pagetrace", risk_class=RiskClass.MEDIUM))
    assert decision.allowed and decision.needs_approval


def test_role_cannot_exceed_its_grants(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    task = make_task(workspace_id="council")
    with pytest.raises(PermissionError, match="lacks"):
        engine.check_action(task, "worker", Capability.PROPOSE_PATCH)
    with pytest.raises(PermissionError, match="unknown role"):
        engine.check_action(task, "ghost", Capability.LLM_INFERENCE)


def test_task_and_workspace_limits_apply_to_actions(
    engine: PolicyEngine, make_task: Callable[..., TaskSpec]
) -> None:
    with pytest.raises(PermissionError, match="does not allow"):
        engine.check_action(make_task(), "reviewer", Capability.READ_WORKSPACE)
    forbidding = make_task(forbidden_capabilities=frozenset({Capability.READ_EVIDENCE}))
    with pytest.raises(PermissionError, match="forbids"):
        engine.check_action(forbidding, "worker", Capability.READ_EVIDENCE)
    elsewhere = make_task(
        workspace_id="nowhere",
        allowed_capabilities=frozenset({Capability.LLM_INFERENCE}),
    )
    with pytest.raises(PermissionError, match="workspace"):
        engine.check_action(elsewhere, "worker", Capability.LLM_INFERENCE)
    engine.check_action(make_task(), "worker", Capability.LLM_INFERENCE)
