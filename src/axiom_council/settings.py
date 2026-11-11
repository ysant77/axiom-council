"""Tunable settings held in the config store, so changing a number is a new config version,
not a code change. Defaults here only apply when the store has no entry for that setting."""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

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
    match_harness: bool = Field(
        default=False,
        description="count only evidence gathered under the current harness hash (RFC 0005)",
    )


class BakeoffKind(Strict):
    prompt_id: str
    task_class: TaskClass
    num_predict: int = Field(gt=0, description="max output tokens, reasoning included")
    public_checks: frozenset[str] = Field(
        default_factory=frozenset,
        description="checks whose failures may be fed back for repair (rules the task states)",
    )


class BakeoffSettings(Strict):
    num_ctx: int = Field(default=16384, gt=0)
    temperature: float = Field(default=0.3, ge=0)
    seed: int | None = Field(default=None, description="engine sampling seed; None is unseeded")
    sandbox_id: str = "pagetrace"
    repair_attempts: int = Field(
        default=0, ge=0, le=3, description="extra attempts after public-check feedback"
    )
    kinds: dict[str, BakeoffKind] = Field(
        default_factory=lambda: {
            "plan": BakeoffKind(
                prompt_id="dispatcher.plan",
                task_class=TaskClass.PLAN,
                num_predict=4000,
                public_checks=frozenset({"valid_json", "within_hours", "models_from_candidates"}),
            ),
            "review": BakeoffKind(
                prompt_id="reviewer.review",
                task_class=TaskClass.REVIEW,
                num_predict=2500,
                public_checks=frozenset({"valid_json"}),
            ),
            "code": BakeoffKind(
                prompt_id="builder.refactor",
                task_class=TaskClass.CODE_PATCH,
                num_predict=8000,
                public_checks=frozenset({"splice", "edit_policy", "lint", "tests"}),
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
    gated_findings: bool = Field(
        default=True,
        description="screen critic findings and require answers to high findings (A9, A10);"
        " false restores the verdict-only critic",
    )
    max_findings: int = Field(
        default=50, ge=1, le=500, description="critic output with more findings is malformed"
    )
    nodes: dict[str, NodeGrant] = Field(
        default_factory=lambda: {
            "plan": NodeGrant(role="dispatcher", capability=Capability.LLM_INFERENCE),
            "build": NodeGrant(role="builder", capability=Capability.PROPOSE_PATCH),
            "gate": NodeGrant(role="gatekeeper", capability=Capability.RUN_SANDBOX),
            "critique": NodeGrant(role="reviewer", capability=Capability.LLM_INFERENCE),
        }
    )


class EnergySettings(Strict):
    sample_interval_s: float = Field(default=0.5, gt=0, le=10)
    grid_kg_co2e_per_kwh: float = Field(default=0.0, ge=0)
    grid_factor_source: str = "unset"


FAULT_CLASSES = (
    "off_by_one", "swallowed_exception", "path_traversal", "injection", "weakened_assertion",
    "changed_default", "resource_leak", "wrong_unit", "race_condition", "api_break",
    "missing_validation", "type_coercion",
)  # fmt: skip


class CriticAdmissionSettings(Strict):
    """Admission test for critic and verifier roles (backlog A8): inference parameters, the
    fault taxonomy that findings must use, and the thresholds a critic must meet."""

    prompt_id: str = "critic.admission"
    num_predict: int = Field(default=2000, gt=0)
    num_ctx: int = Field(default=8192, gt=0)
    temperature: float = Field(default=0.0, ge=0)
    seed: int | None = None
    fault_classes: tuple[str, ...] = Field(default=FAULT_CLASSES, min_length=1)
    min_supported_recall: float = Field(default=0.5, ge=0, le=1)
    max_clean_false_high_rate: float = Field(default=0.1, ge=0, le=1)
    min_high_precision: float = Field(default=0.5, ge=0, le=1)
    max_invalid_output_rate: float = Field(default=0.1, ge=0, le=1)
    min_seeded_cases: int = Field(default=12, ge=1)
    min_clean_cases: int = Field(default=12, ge=1)
    max_finding_lines: int = Field(
        default=20, ge=1, description="a wider finding cannot support a fault: it locates nothing"
    )

    @field_validator("fault_classes")
    @classmethod
    def _unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("fault_classes must be unique")
        return value


class GroundingSettings(Strict):
    """Rules for the number-grounding interceptor. A route opts in with a flag."""

    relative_tolerance: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="a number also matches an evidence value within this share of that value",
    )
    rounding_min_digits: int = Field(
        default=2,
        ge=1,
        le=12,
        description="a decimal shown with at least this many digits matches a rounded value",
    )
    percent_matches_fraction: bool = Field(
        default=False, description="a percent may match the evidence fraction (67% and 0.67)"
    )
    evidence_issuers: frozenset[str] = Field(
        default=frozenset({"gatekeeper"}),
        description="roles whose delivered messages issue evidence; others cannot cite their own",
    )
    identifier_allowlist: frozenset[str] = Field(
        default=frozenset({"p50", "p90", "p95", "p99", "p999"}),
        description="names with digits that are not measurements (whole token, any case)",
    )


class Settings(Strict):
    """Every named setting; an unknown name in the store fails validation, never passes silently."""

    router: RouterSettings = Field(default_factory=RouterSettings)
    bakeoff: BakeoffSettings = Field(default_factory=BakeoffSettings)
    pipeline: PipelineSettings = Field(default_factory=PipelineSettings)
    energy: EnergySettings = Field(default_factory=EnergySettings)
    critic_admission: CriticAdmissionSettings = Field(default_factory=CriticAdmissionSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)

    @classmethod
    def from_entries(cls, raw: dict[str, dict[str, Any]]) -> Settings:
        """Build from config-store entries keyed by name (each body repeats its `name`)."""
        return cls(
            **{name: {k: v for k, v in body.items() if k != "name"} for name, body in raw.items()}
        )
