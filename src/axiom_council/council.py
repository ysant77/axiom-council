"""The council run loop for R0: policy → route → (approval) → inference → audit.

Multi-role planning, review and arbitration arrive in later releases on top of this loop;
every step they add must go through the same policy check and audit log.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from axiom_council.audit import AuditLog
from axiom_council.configstore import ConfigStore
from axiom_council.contracts import Capability, InferenceResult, TaskSpec, TaskState
from axiom_council.inbox import Inbox
from axiom_council.ledger import CapabilityLedger
from axiom_council.policy import PolicyEngine
from axiom_council.providers import OpenAICompatProvider
from axiom_council.registry import NodeSpec, Registry
from axiom_council.router import Router, RoutingDecision


@dataclass(frozen=True)
class RunOutcome:
    state: TaskState
    decision: RoutingDecision | None = None
    result: InferenceResult | None = None
    reasons: tuple[str, ...] = ()
    inbox_item: str | None = None


def open_registry(home: Path, seed_dir: Path) -> Registry:
    """Load config from the store under home, seeding it from YAML on first use."""
    store = ConfigStore(home / "config.sqlite3")
    try:
        if store.is_empty():
            store.seed(seed_dir)
        return Registry.from_store(store)
    finally:
        store.close()


ProviderFactory = Callable[[NodeSpec], OpenAICompatProvider]


class Council:
    def __init__(
        self,
        registry: Registry,
        home: Path,
        provider_factory: ProviderFactory = OpenAICompatProvider,
    ):
        self.registry = registry
        self._provider_factory = provider_factory
        self.home = home
        self.policy = PolicyEngine(registry.workspaces, registry.roles)
        self.audit = AuditLog(home / "audit.jsonl")
        self.ledger = CapabilityLedger(home / "ledger.sqlite3")
        self.inbox = Inbox(home / "inbox")
        self.router = Router(registry, self.ledger)

    def close(self) -> None:
        self.ledger.close()

    def _admit(self, task: TaskSpec, role: str) -> tuple[str, ...]:
        """Policy gate; returns rejection reasons (empty when admitted)."""
        verdict = self.policy.check_task(task)
        reasons = verdict.reasons
        if verdict.allowed:
            try:
                self.policy.check_action(task, role, Capability.LLM_INFERENCE)
            except PermissionError as exc:
                reasons = (str(exc),)
        if reasons:
            self.audit.append("policy_rejected", task.task_id, "policy", reasons=list(reasons))
        return reasons

    def _hold(self, task: TaskSpec, decision: RoutingDecision, action: str) -> RunOutcome:
        """Park the task in the director's inbox instead of running it."""
        if decision.chosen is None:
            summary = "No eligible model placement; choose a model or revise the task."
            state = TaskState.AWAITING_CLARIFICATION
        else:
            summary = f"{task.objective[:120]} -> {decision.chosen.model_id}"
            state = TaskState.AWAITING_APPROVAL
        item = self.inbox.request_approval(
            task_id=task.task_id, action=action, artifact_hash=task.digest(), summary=summary
        )
        self.audit.append("director_requested", task.task_id, "council", item=item.item_id)
        return RunOutcome(state, decision, inbox_item=item.item_id)

    def _infer(
        self,
        task: TaskSpec,
        decision: RoutingDecision,
        messages: list[dict[str, str]],
        output_tokens: int,
    ) -> RunOutcome:
        assert decision.chosen is not None
        node = self.registry.nodes[decision.chosen.node_id]
        model = self.registry.models[decision.chosen.model_id]
        provider = self._provider_factory(node)
        try:
            result = provider.chat(
                task_id=task.task_id,
                model_id=model.model_id,
                messages=messages,
                max_tokens=output_tokens + model.reasoning_overhead_tokens,
                extra=model.request_defaults,
            )
        except Exception as exc:
            self.audit.append("inference_failed", task.task_id, node.node_id, error=str(exc))
            return RunOutcome(TaskState.FAILED, decision, reasons=(str(exc),))
        finally:
            provider.close()
        self.audit.append(
            "inference_completed",
            task.task_id,
            f"{node.node_id}/{result.model_id}",
            usage=result.usage.model_dump(),
            output_chars=len(result.output),
        )
        return RunOutcome(TaskState.RUNNING, decision, result)

    def run(
        self, task: TaskSpec, user_input: str, *, role: str = "worker", output_tokens: int = 1024
    ) -> RunOutcome:
        self.audit.append("task_submitted", task.task_id, task.actor, spec_hash=task.digest())
        reasons = self._admit(task, role)
        if reasons:
            return RunOutcome(TaskState.POLICY_REJECTED, reasons=reasons)

        prompt = self.registry.prompt_for(role, task.task_class)
        messages = (
            prompt.render(input=user_input) if prompt else [{"role": "user", "content": user_input}]
        )
        prompt_tokens = max(1, sum(len(m["content"]) for m in messages) // 4)
        decision = self.router.route(task, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        self.audit.append(
            "routed",
            task.task_id,
            "router",
            prompt_id=prompt.prompt_id if prompt else None,
            chosen=asdict(decision.chosen) if decision.chosen else None,
            rejected=[asdict(r) for r in decision.rejected],
        )
        if decision.chosen is None:
            return self._hold(task, decision, "override_routing")
        if self.policy.check_task(task).needs_approval:
            return self._hold(task, decision, "execute")
        return self._infer(task, decision, messages, output_tokens)
