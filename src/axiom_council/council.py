"""The council run loop: policy → route → (approval) → inference → audit, with every task
persisted so an approval survives a restart and resumes exactly the task that was approved.

Multi-role planning, review and arbitration build on this loop; every step they add must go
through the same policy check, task state machine and audit log.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from axiom_council.audit import open_audit
from axiom_council.configstore import ConfigStore
from axiom_council.contracts import (
    Capability,
    InferenceResult,
    TaskSpec,
    TaskState,
    content_hash,
    utcnow,
)
from axiom_council.inbox import Inbox
from axiom_council.ledger import CapabilityLedger
from axiom_council.policy import PolicyEngine
from axiom_council.provenance import think_value
from axiom_council.providers import ChatProvider, ProviderError, provider_for
from axiom_council.registry import ModelCard, NodeSpec, Registry
from axiom_council.router import Router, RoutingDecision
from axiom_council.tasks import TaskRecord, TaskStore

S = TaskState


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


ProviderFactory = Callable[[NodeSpec], ChatProvider]


def _request_extra(model: ModelCard) -> dict[str, Any]:
    """The card's request fields plus its reasoning control, as the bake-off sends them."""
    think = think_value(model)
    return {**model.request_defaults, **({"think": think} if think is not None else {})}


def approval_artifact(registry: Registry, spec: TaskSpec, model_id: str, node_id: str) -> str:
    """What an execute approval covers: the exact task plus the exact placement and its config.

    Any change to the spec, the model card (incl. request defaults and reasoning settings) or
    the node spec (incl. URL and runtime) changes the hash, so an old approval stops working.
    Raises KeyError if the model or node no longer exists.
    """
    return content_hash(
        {
            "spec": spec.digest(),
            "model": registry.models[model_id].model_dump(mode="json"),
            "node": registry.nodes[node_id].model_dump(mode="json"),
        }
    )


class Council:
    def __init__(
        self,
        registry: Registry,
        home: Path,
        provider_factory: ProviderFactory = provider_for,
    ):
        self.registry = registry
        self._provider_factory = provider_factory
        self.home = home
        self.policy = PolicyEngine(registry.workspaces, registry.roles)
        self.audit = open_audit(home)
        self.ledger = CapabilityLedger(home / "ledger.sqlite3")
        self.inbox = Inbox(home / "inbox")
        self.tasks = TaskStore(home / "tasks.sqlite3")
        self.router = Router(registry, self.ledger)
        self.interrupted = self.recover()

    def close(self) -> None:
        self.ledger.close()
        self.tasks.close()

    def recover(self) -> list[str]:
        """Tasks caught mid-inference by a crash are failed, never silently re-run."""
        interrupted = self.tasks.in_state(S.RUNNING)
        for task_id in interrupted:
            self.tasks.transition(task_id, S.FAILED, reason="interrupted by restart")
            self.audit.append("task_interrupted", task_id, "council")
        return interrupted

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
            self.tasks.transition(task.task_id, S.POLICY_REJECTED, reason="; ".join(reasons))
        return reasons

    def _messages(self, record: TaskRecord) -> list[dict[str, str]]:
        prompt = self.registry.prompt_for(record.role, record.spec.task_class)
        if prompt is None:
            return [{"role": "user", "content": record.user_input}]
        return prompt.render(input=record.user_input)

    def _route(self, record: TaskRecord, messages: list[dict[str, str]]) -> RoutingDecision:
        prompt_tokens = max(1, sum(len(m["content"]) for m in messages) // 4)
        decision = self.router.route(
            record.spec, prompt_tokens=prompt_tokens, output_tokens=record.output_tokens
        )
        self.audit.append(
            "routed",
            record.spec.task_id,
            "router",
            chosen=asdict(decision.chosen) if decision.chosen else None,
            rejected=[asdict(r) for r in decision.rejected],
        )
        return decision

    def _hold(self, task: TaskSpec, decision: RoutingDecision) -> RunOutcome:
        """Park the task in the director's inbox instead of running it."""
        chosen = decision.chosen
        if chosen is None:
            action, state, artifact = "override_routing", S.AWAITING_CLARIFICATION, task.digest()
            summary = "No eligible model placement; choose a model or revise the task."
        else:
            action, state = "execute", S.AWAITING_APPROVAL
            self.tasks.pin(task.task_id, chosen.model_id, chosen.node_id)
            artifact = approval_artifact(self.registry, task, chosen.model_id, chosen.node_id)
            summary = f"{task.objective[:120]} -> {chosen.model_id} @ {chosen.node_id}"
        item = self.inbox.request_approval(
            task_id=task.task_id, action=action, artifact_hash=artifact, summary=summary
        )
        self.tasks.transition(task.task_id, state, inbox_item=item.item_id)
        self.audit.append("director_requested", task.task_id, "council", item=item.item_id)
        return RunOutcome(state, decision, inbox_item=item.item_id)

    def _infer(
        self, task: TaskSpec, decision: RoutingDecision, messages: list[dict[str, str]],
        output_tokens: int,
    ) -> RunOutcome:  # fmt: skip
        assert decision.chosen is not None
        node = self.registry.nodes[decision.chosen.node_id]
        model = self.registry.models[decision.chosen.model_id]
        self.tasks.transition(task.task_id, S.RUNNING)
        provider = self._provider_factory(node)
        try:
            result = provider.chat(
                task_id=task.task_id,
                model_id=model.model_id,
                messages=messages,
                max_tokens=output_tokens + model.reasoning_overhead_tokens,
                extra=_request_extra(model),
            )
            if not result.output.strip() and result.finish_reason == "length":
                raise ProviderError(
                    f"{model.model_id} exhausted its output budget before answering"
                    " (likely hidden reasoning); raise reasoning_overhead_tokens"
                )
        except Exception as exc:
            self.audit.append("inference_failed", task.task_id, node.node_id, error=str(exc))
            self.tasks.transition(task.task_id, S.FAILED, reason=str(exc)[:500])
            return RunOutcome(S.FAILED, decision, reasons=(str(exc),))
        finally:
            provider.close()
        self.audit.append(
            "inference_completed", task.task_id, f"{node.node_id}/{result.model_id}",
            usage=result.usage.model_dump(), output_chars=len(result.output),
        )  # fmt: skip
        self.tasks.transition(task.task_id, S.COMPLETED)
        return RunOutcome(S.COMPLETED, decision, result)

    def _dispatch(self, record: TaskRecord) -> RunOutcome:
        """Route an admitted task, then hold it for the director or execute it."""
        messages = self._messages(record)
        decision = self._route(record, messages)
        task = record.spec
        self.tasks.transition(task.task_id, S.ROUTED)
        if decision.chosen is None or self.policy.check_task(task).needs_approval:
            return self._hold(task, decision)
        return self._infer(task, decision, messages, record.output_tokens)

    def _run_pinned(self, record: TaskRecord) -> RunOutcome:
        """Execute an approved task on its approved placement; never route it again."""
        assert record.pinned_model is not None
        assert record.pinned_node is not None
        messages = self._messages(record)
        decision = self.router.check_placement(
            record.spec, record.pinned_model, record.pinned_node,
            prompt_tokens=max(1, sum(len(m["content"]) for m in messages) // 4),
            output_tokens=record.output_tokens,
        )  # fmt: skip
        self.audit.append(
            "routed", record.spec.task_id, "router", pinned=True,
            chosen=asdict(decision.chosen) if decision.chosen else None,
            rejected=[asdict(r) for r in decision.rejected],
        )  # fmt: skip
        if decision.chosen is None:
            return self._hold(record.spec, decision)
        return self._infer(record.spec, decision, messages, record.output_tokens)

    def _approved_artifact(self, record: TaskRecord) -> str | None:
        if record.pinned_model is None or record.pinned_node is None:
            return None
        try:
            return approval_artifact(
                self.registry, record.spec, record.pinned_model, record.pinned_node
            )
        except KeyError:
            return None

    def run(
        self, task: TaskSpec, user_input: str, *, role: str = "worker", output_tokens: int = 1024
    ) -> RunOutcome:
        self.audit.append("task_submitted", task.task_id, task.actor, spec_hash=task.digest())
        self.tasks.create(task, user_input, role, output_tokens)
        reasons = self._admit(task, role)
        if reasons:
            return RunOutcome(S.POLICY_REJECTED, reasons=reasons)
        return self._dispatch(self.tasks.get(task.task_id))

    def resume(self, task_id: str, *, now: datetime | None = None) -> RunOutcome:
        """Continue a task the director decided on, verifying the stored approval token."""
        record = self.tasks.get(task_id)
        if record.state is not S.AWAITING_APPROVAL or record.inbox_item is None:
            raise PermissionError(f"{task_id} is {record.state}, not awaiting approval")
        item = self.inbox.get(record.inbox_item)
        if item.status == "rejected":
            self.tasks.transition(task_id, S.REJECTED, reason=item.response)
            return RunOutcome(S.REJECTED, reasons=(item.response or "rejected",))
        token, artifact = item.token, self._approved_artifact(record)
        moment = now or utcnow()
        if (
            token is None
            or artifact is None
            or not token.authorises(task_id, "execute", artifact, moment)
        ):
            self.audit.append("approval_invalid", task_id, "council", item=item.item_id)
            raise PermissionError(f"{task_id} has no valid approval for this exact task")
        self.tasks.transition(task_id, S.APPROVED)
        self.audit.append(
            "task_resumed", task_id, token.approver, token=token.token_id, artifact=artifact,
            model=record.pinned_model, node=record.pinned_node,
        )  # fmt: skip
        if self._admit(record.spec, record.role):
            return RunOutcome(S.POLICY_REJECTED, reasons=("policy changed since approval",))
        return self._run_pinned(record)
