"""Cost-aware, evidence-gated routing of a task to a (model, node) placement.

Order of authority:
1. Hard eligibility (modality, context, memory, data trust) — deterministic, never overridden.
2. Skill gate — a conservative lower bound on measured success probability.
3. Ranking — memory-seconds per *verified* success: the node's scarce resource is memory
   held over time, so a large model must be proportionally more reliable or faster to win,
   and failures count against every model.

A dispatcher model may later pick among the eligible set and must justify its pick, but it
cannot add a candidate that this module rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from axiom_council.contracts import DataClass, TaskSpec
from axiom_council.ledger import CapabilityLedger
from axiom_council.registry import ModelCard, NodeSpec, Registry
from axiom_council.settings import RouterSettings


@dataclass(frozen=True)
class Candidate:
    model_id: str
    node_id: str
    p_success: float
    p_lower: float
    trials: int
    expected_seconds: float
    energy_usd: float
    shadow_api_usd: float
    cost_per_success_usd: float
    seconds_per_success: float
    gb_seconds_per_success: float


@dataclass(frozen=True)
class Rejection:
    model_id: str
    node_id: str
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    task_id: str
    chosen: Candidate | None
    ranked: tuple[Candidate, ...]
    rejected: tuple[Rejection, ...]
    escalate: bool
    notes: tuple[str, ...] = field(default_factory=tuple)


def _posterior(prior: float, successes: int, failures: int, strength: float) -> tuple[float, float]:
    a = prior * strength + successes
    b = (1 - prior) * strength + failures
    mean = a / (a + b)
    var = (a * b) / ((a + b) ** 2 * (a + b + 1))
    return mean, math.sqrt(var)


def _estimated_tps(model: ModelCard) -> float:
    """Rough decode speed until measurements exist; replaced by ledger means once recorded."""
    active = model.active_params_b or model.params_b
    return max(5.0, 120.0 / max(active, 0.5))


class Router:
    def __init__(
        self,
        registry: Registry,
        ledger: CapabilityLedger,
        settings: RouterSettings | None = None,
    ):
        self.registry = registry
        self.ledger = ledger
        self.settings = settings or registry.settings.router

    def _hard_check(
        self,
        task: TaskSpec,
        model: ModelCard,
        node: NodeSpec,
        modalities: frozenset[str],
        prompt_tokens: int,
        output_tokens: int,
    ) -> str | None:
        missing = modalities - model.modalities
        if missing:
            return f"missing modalities {sorted(missing)}"
        needed = prompt_tokens + output_tokens + model.reasoning_overhead_tokens
        if needed > model.context_tokens:
            return f"needs {needed} tokens > context {model.context_tokens}"
        if model.memory_gb > node.memory_budget_gb:
            return f"needs {model.memory_gb} GB > node budget {node.memory_budget_gb} GB"
        if task.data_classification is DataClass.PRIVATE and not node.trusted_for_private:
            return "node not trusted for private data"
        if needed > task.budget.max_tokens:
            return "exceeds task token budget"
        return None

    def _score(
        self,
        task: TaskSpec,
        model: ModelCard,
        node: NodeSpec,
        prompt_tokens: int,
        output_tokens: int,
        *,
        skill_gate: bool = True,
    ) -> Candidate | Rejection:
        """Apply the skill gate and latency ceiling, then price one eligible placement."""
        stats = self.ledger.stats(model.model_id, task.task_class)
        cfg = self.settings
        prior = model.prior_skill.get(task.task_class, cfg.default_prior)
        p, sd = _posterior(prior, stats.successes, stats.failures, cfg.prior_strength)
        p_lower = max(0.0, p - cfg.conservatism_sd * sd)
        if skill_gate and p_lower < cfg.min_success:
            return Rejection(
                model.model_id,
                node.node_id,
                f"skill lower bound {p_lower:.2f} < {cfg.min_success:.2f}"
                f" ({stats.trials} verified trials)",
            )
        seconds = (
            stats.mean_wall_seconds
            if stats.mean_wall_seconds is not None
            else (output_tokens + model.reasoning_overhead_tokens) / _estimated_tps(model)
        )
        if cfg.max_seconds is not None and seconds > cfg.max_seconds:
            return Rejection(model.model_id, node.node_id, f"expected {seconds:.0f}s > ceiling")
        energy = seconds * node.watts_under_load / 3_600_000 * node.usd_per_kwh
        return Candidate(
            model_id=model.model_id,
            node_id=node.node_id,
            p_success=p,
            p_lower=p_lower,
            trials=stats.trials,
            expected_seconds=seconds,
            energy_usd=energy,
            shadow_api_usd=(prompt_tokens + output_tokens + model.reasoning_overhead_tokens)
            * model.shadow_usd_per_mtok
            / 1e6,
            cost_per_success_usd=energy / p,
            seconds_per_success=seconds / p,
            gb_seconds_per_success=seconds * model.memory_gb / p,
        )

    def check_placement(
        self,
        task: TaskSpec,
        model_id: str,
        node_id: str,
        *,
        prompt_tokens: int,
        output_tokens: int = 1024,
        modalities: frozenset[str] = frozenset({"text"}),
    ) -> RoutingDecision:
        """Re-check one fixed placement (an approved one) without choosing another.

        Hard eligibility still applies; the skill gate does not, because the director has
        already approved this placement. No alternative is ever proposed.
        """
        node = self.registry.nodes.get(node_id)
        model = self.registry.models.get(model_id)
        if node is None or model is None or not node.enabled or model_id not in node.serves:
            outcome: Candidate | Rejection = Rejection(model_id, node_id, "placement unavailable")
        else:
            reason = self._hard_check(task, model, node, modalities, prompt_tokens, output_tokens)
            outcome = (
                Rejection(model_id, node_id, reason)
                if reason
                else self._score(task, model, node, prompt_tokens, output_tokens, skill_gate=False)
            )
        chosen = outcome if isinstance(outcome, Candidate) else None
        rejected = () if chosen else (outcome,)
        return RoutingDecision(
            task_id=task.task_id, chosen=chosen, ranked=(chosen,) if chosen else (),
            rejected=rejected, escalate=chosen is None,  # type: ignore[arg-type]
            notes=("approved placement re-checked",),
        )  # fmt: skip

    def route(
        self,
        task: TaskSpec,
        *,
        prompt_tokens: int,
        output_tokens: int = 1024,
        modalities: frozenset[str] = frozenset({"text"}),
    ) -> RoutingDecision:
        ranked: list[Candidate] = []
        rejected: list[Rejection] = []
        for node in self.registry.nodes.values():
            if not node.enabled:
                continue
            for model_id in node.serves:
                model = self.registry.models[model_id]
                reason = self._hard_check(
                    task, model, node, modalities, prompt_tokens, output_tokens
                )
                outcome = (
                    Rejection(model_id, node.node_id, reason)
                    if reason
                    else self._score(task, model, node, prompt_tokens, output_tokens)
                )
                if isinstance(outcome, Candidate):
                    ranked.append(outcome)
                else:
                    rejected.append(outcome)

        ranked.sort(key=lambda c: (c.gb_seconds_per_success, c.seconds_per_success, -c.p_success))
        return RoutingDecision(
            task_id=task.task_id,
            chosen=ranked[0] if ranked else None,
            ranked=tuple(ranked),
            rejected=tuple(rejected),
            escalate=not ranked,
            notes=() if ranked else ("no eligible placement; escalate to director",),
        )
