"""Model critic adapter: asks one placement for located findings through a `ChatProvider`.

The prompt comes from the registry (`critic_admission.prompt_id`). The reply is read by the
pipeline's own parser (`pipeline.findings.parse_findings`) with the pipeline's finding
limit, so a critic that passes admission produces output the pipeline accepts. Output the
parser refuses gives zero findings and `valid=False`, which the admission test counts.
"""

from __future__ import annotations

from typing import Any

from axiom_council.bus.spotlight import fence
from axiom_council.evals.critic_contracts import CriticCase, CriticKind, CriticOutput
from axiom_council.pipeline.findings import FindingsMalformed, parse_findings
from axiom_council.provenance import Placement, placement_for, think_value
from axiom_council.providers import ChatProvider
from axiom_council.registry import Registry

_RAW_LIMIT = 4000  # characters of the raw reply kept in results


def critic_output(text: str, limit: int) -> CriticOutput:
    """The pipeline parser's verdict on one reply, as an admission record."""
    raw = text[:_RAW_LIMIT]
    try:
        findings = parse_findings(text, limit)
    except FindingsMalformed:
        return CriticOutput(valid=False, raw=raw)
    return CriticOutput(findings=tuple(findings), raw=raw)


class ModelCritic:
    """One model at one placement acting as a critic."""

    def __init__(
        self,
        registry: Registry,
        provider: ChatProvider,
        model_id: str,
        node_id: str,
        think: bool | str | None = None,
    ):
        self._registry = registry
        self._provider = provider
        self._model_id = model_id
        self._think = think
        self.placement: Placement = placement_for(registry, model_id, node_id, think=think)

    @property
    def critic_id(self) -> str:
        return f"model:{self._model_id}:{self.placement.reasoning or 'default'}"

    @property
    def kind(self) -> CriticKind:
        return "model"

    def _extra(self) -> dict[str, Any]:
        cfg = self._registry.settings.critic_admission
        card = self._registry.models[self._model_id]
        extra: dict[str, Any] = {**card.request_defaults, "num_ctx": cfg.num_ctx}
        think = think_value(card, self._think)
        if think is not None:
            extra["think"] = think
        if cfg.seed is not None:
            extra["seed"] = cfg.seed
        return extra

    def review(self, case: CriticCase) -> CriticOutput:
        cfg = self._registry.settings.critic_admission
        prompt = self._registry.prompts[cfg.prompt_id]
        messages = prompt.render(
            title=case.title,
            fault_classes=", ".join(cfg.fault_classes),
            diff=fence(case.diff, "DIFF"),
        )
        result = self._provider.chat(
            task_id=f"admission-{case.id}",
            model_id=self._model_id,
            messages=messages,
            max_tokens=cfg.num_predict,
            temperature=cfg.temperature,
            extra=self._extra(),
        )
        return critic_output(result.output, self._registry.settings.pipeline.max_findings)
