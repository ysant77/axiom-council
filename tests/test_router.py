from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from axiom_council.contracts import Budget, DataClass, TaskClass, TaskSpec
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import Registry
from axiom_council.router import Router

SMALL_MODELS = {"qwen3:4b", "gemma3:4b"}


@pytest.fixture
def ledger(tmp_path: Path) -> Iterator[CapabilityLedger]:
    lg = CapabilityLedger(tmp_path / "ledger.sqlite3")
    yield lg
    lg.close()


def _record(ledger: CapabilityLedger, model: str, cls: TaskClass, ok: int, bad: int) -> None:
    for i in range(ok + bad):
        ledger.record(
            task_id=f"t{i}",
            model_id=model,
            node_id="mac-m5",
            task_class=cls,
            success=i < ok,
            verified_by="pytest",
            wall_seconds=2.0,
            total_tokens=500,
        )


def test_small_model_wins_triage_on_speed(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    decision = Router(registry, ledger).route(make_task(), prompt_tokens=400)
    assert decision.chosen is not None
    assert decision.chosen.model_id in {"qwen3:4b", "gemma3:4b"}
    assert not decision.escalate
    assert all(c.node_id == "mac-m5" for c in decision.ranked)  # windows node disabled


def test_weak_prior_is_gated_out_for_hard_tasks(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    decision = Router(registry, ledger).route(
        make_task(task_class=TaskClass.PLAN), prompt_tokens=400
    )
    assert decision.chosen is not None and decision.chosen.model_id not in SMALL_MODELS
    gated = {r.model_id for r in decision.rejected if "skill lower bound" in r.reason}
    assert gated >= SMALL_MODELS


def test_measured_failures_demote_a_model(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    _record(ledger, "qwen3:4b", TaskClass.TRIAGE, ok=1, bad=12)
    _record(ledger, "gemma3:4b", TaskClass.TRIAGE, ok=1, bad=12)
    decision = Router(registry, ledger).route(make_task(), prompt_tokens=400)
    assert decision.chosen is not None and decision.chosen.model_id not in SMALL_MODELS


def test_measured_successes_promote_a_model(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    _record(ledger, "qwen3:4b", TaskClass.PLAN, ok=20, bad=0)
    decision = Router(registry, ledger).route(
        make_task(task_class=TaskClass.PLAN), prompt_tokens=400
    )
    chosen = decision.chosen
    assert chosen is not None and chosen.model_id == "qwen3:4b" and chosen.trials == 20


def test_hard_constraints_reject_with_reasons(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    router = Router(registry, ledger)
    image = router.route(make_task(), prompt_tokens=400, modalities=frozenset({"text", "image"}))
    assert image.chosen is not None and image.chosen.model_id == "gemma3:4b"
    long = router.route(
        make_task(budget=Budget(max_tokens=64_000, max_wall_seconds=60)), prompt_tokens=40_000
    )
    assert any("context" in r.reason for r in long.rejected)
    over = router.route(make_task(), prompt_tokens=15_500)
    assert over.escalate and over.chosen is None
    assert any("token budget" in r.reason for r in over.rejected)


def test_untrusted_node_never_sees_private_data(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    mac = registry.nodes["mac-m5"].model_copy(update={"trusted_for_private": False})
    reg = registry.model_copy(update={"nodes": {"mac-m5": mac}})
    decision = Router(reg, ledger).route(
        make_task(data_classification=DataClass.PRIVATE), prompt_tokens=100
    )
    assert decision.escalate
    assert all("private" in r.reason for r in decision.rejected)


def test_latency_ceiling_and_memory_budget(
    registry: Registry, ledger: CapabilityLedger, make_task: Callable[..., TaskSpec]
) -> None:
    slow = Router(
        registry, ledger, registry.settings.router.model_copy(update={"max_seconds": 0.001})
    ).route(make_task(), prompt_tokens=100)
    assert slow.escalate and all("ceiling" in r.reason for r in slow.rejected)
    tiny = registry.nodes["mac-m5"].model_copy(update={"memory_budget_gb": 3.8})
    reg = registry.model_copy(update={"nodes": {"mac-m5": tiny}})
    decision = Router(reg, ledger).route(make_task(), prompt_tokens=100)
    assert decision.chosen is not None and decision.chosen.model_id == "qwen3:4b"


def test_self_verification_is_refused(ledger: CapabilityLedger) -> None:
    with pytest.raises(ValueError, match="verified by someone other"):
        ledger.record(
            task_id="t",
            model_id="qwen3:4b",
            node_id="mac-m5",
            task_class=TaskClass.TRIAGE,
            success=True,
            verified_by="qwen3:4b",
            wall_seconds=1,
            total_tokens=1,
        )
