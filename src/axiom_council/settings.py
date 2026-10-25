"""Tunable settings held in the config store, so changing a number is a new config version,
not a code change. Defaults here only apply when the store has no entry for that setting."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from axiom_council.contracts import Capability, Strict, TaskClass


class RouterSettings(Strict):
    prior_strength: float = Field(
        default=10.0, gt=0, description="pseudo-observations behind a model card's prior_skill"
    )
    default_prior: float = Field(default=0.5, ge=0, le=1)
    conservatism_sd: float = Field(
        default=1.0, ge=0, description="posterior s.d. below the mean that the skill gate uses"
    )
    min_success: float = Field(default=0.55, ge=0, le=1)
    max_seconds: float | None = Field(default=None, gt=0)


class BakeoffKind(Strict):
    prompt_id: str
    task_class: TaskClass
    num_predict: int = Field(gt=0, description="max output tokens, reasoning included")


class BakeoffSettings(Strict):
    num_ctx: int = Field(default=16384, gt=0)
    temperature: float = Field(default=0.3, ge=0)
    sandbox_id: str = "pagetrace"
    kinds: dict[str, BakeoffKind] = Field(
        default_factory=lambda: {
            "plan": BakeoffKind(
                prompt_id="dispatcher.plan", task_class=TaskClass.PLAN, num_predict=4000
            ),
            "review": BakeoffKind(
                prompt_id="reviewer.review", task_class=TaskClass.REVIEW, num_predict=2500
            ),
            "code": BakeoffKind(
                prompt_id="builder.refactor", task_class=TaskClass.CODE_PATCH, num_predict=8000
            ),
        }
    )


class NodeGrant(Strict):
    """Which role a pipeline node acts as and the one capability it exercises."""

    role: str
    capability: Capability


class PipelineSettings(Strict):
    approval_action: str = "apply_patch"
    max_build_attempts: int = Field(default=2, ge=1, le=5)
    nodes: dict[str, NodeGrant] = Field(
        default_factory=lambda: {
            "plan": NodeGrant(role="dispatcher", capability=Capability.LLM_INFERENCE),
            "build": NodeGrant(role="builder", capability=Capability.PROPOSE_PATCH),
            "gate": NodeGrant(role="gatekeeper", capability=Capability.RUN_SANDBOX),
            "critique": NodeGrant(role="reviewer", capability=Capability.LLM_INFERENCE),
        }
    )


class Settings(Strict):
    """Every named setting; an unknown name in the store fails validation, never passes silently."""

    router: RouterSettings = Field(default_factory=RouterSettings)
    bakeoff: BakeoffSettings = Field(default_factory=BakeoffSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)

    @classmethod
    def from_entries(cls, raw: dict[str, dict[str, Any]]) -> Settings:
        """Build from config-store entries keyed by name (each body repeats its `name`)."""
        return cls(
            **{name: {k: v for k, v in body.items() if k != "name"} for name, body in raw.items()}
        )
