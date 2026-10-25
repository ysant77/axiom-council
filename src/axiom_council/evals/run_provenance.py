"""Provenance shared by every record of one contestant in one bake-off run (D-12).

The model build is checked before any inference: if the runtime reports a digest or a
quantisation that differs from the model card, the run stops. Evidence for an unpinned or
different build would otherwise be filed under the wrong placement.
"""

from __future__ import annotations

from dataclasses import dataclass

from axiom_council.contracts import ModelProvenance
from axiom_council.provenance import (
    Placement,
    config_hash,
    harness_hash,
    placement_for,
    source_commit,
)
from axiom_council.providers import ChatProvider
from axiom_council.registry import ModelCard, Registry


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    source_commit: str | None
    config_hash: str
    harness_hash: str
    model_id: str
    model_digest: str | None  # pinned on the model card; the routing key
    runtime_digest: str | None  # reported by the runtime; None if it cannot say
    quantization: str | None
    runtime: str
    runtime_version: str | None
    node_id: str
    hardware: str
    reasoning: str | None
    target_commit: str | None

    @property
    def placement(self) -> Placement:
        return Placement(
            model_id=self.model_id,
            node_id=self.node_id,
            runtime=self.runtime,
            model_digest=self.model_digest,
            quantization=self.quantization,
            reasoning=self.reasoning,
        )


def check_build(card: ModelCard, reported: ModelProvenance) -> None:
    """Refuse a build the model card does not pin, or one that differs from it."""
    if reported.digest is not None:
        if card.digest is None:
            raise ValueError(
                f"{card.model_id}: pin digest {reported.digest} in the model card before a run"
            )
        if not reported.digest.startswith(card.digest):
            raise ValueError(
                f"{card.model_id}: runtime serves digest {reported.digest[:12]}, the model card"
                f" pins {card.digest[:12]}; update the card or pull the pinned build"
            )
    if (
        reported.quantization
        and card.quantization
        and reported.quantization.lower() != card.quantization.lower()
    ):
        raise ValueError(
            f"{card.model_id}: runtime quantisation {reported.quantization} differs from the"
            f" model card ({card.quantization})"
        )


def run_provenance(
    registry: Registry,
    provider: ChatProvider,
    *,
    model_id: str,
    think: bool | str | None,
    node_id: str,
    run_id: str,
    target_commit: str | None,
) -> RunProvenance:
    card, node = registry.models[model_id], registry.nodes[node_id]
    reported = provider.describe(model_id)
    check_build(card, reported)
    where = placement_for(registry, model_id, node_id, think=think)
    return RunProvenance(
        run_id=run_id,
        source_commit=source_commit(),
        config_hash=config_hash(registry),
        harness_hash=harness_hash(registry),
        model_id=model_id,
        model_digest=card.digest,
        runtime_digest=reported.digest,
        quantization=card.quantization,
        runtime=node.runtime,
        runtime_version=reported.runtime_version,
        node_id=node_id,
        hardware=node.hardware,
        reasoning=where.reasoning,
        target_commit=target_commit,
    )
