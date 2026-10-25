"""Versioned, strict data contracts shared by every council component.

All models forbid unknown fields so that a malformed or injected payload fails loudly
instead of being silently accepted.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION: Literal[1] = 1


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def content_hash(payload: Any) -> str:
    """Stable SHA-256 over a JSON-serialisable payload (keys sorted)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Capability(StrEnum):
    """Things a role or task may be allowed to do. Checked in code, never by a model."""

    READ_WORKSPACE = "read_workspace"
    READ_EVIDENCE = "read_evidence"
    LLM_INFERENCE = "llm_inference"
    PROPOSE_PATCH = "propose_patch"
    RUN_SANDBOX = "run_sandbox"
    NETWORK_EGRESS = "network_egress"
    WRITE_WORKSPACE = "write_workspace"
    EXTERNAL_MESSAGE = "external_message"
    MERGE = "merge"
    SPEND_MONEY = "spend_money"


class RiskClass(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DataClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    PRIVATE = "private"


class TaskClass(StrEnum):
    """Kinds of work the router keeps separate capability statistics for."""

    TRIAGE = "triage"
    EXTRACT = "extract"
    GROUNDED_ANSWER = "grounded_answer"
    PLAN = "plan"
    REVIEW = "review"
    CODE_PATCH = "code_patch"
    DRAFT = "draft"


class TaskState(StrEnum):
    DRAFT = "draft"
    POLICY_REJECTED = "policy_rejected"
    ROUTED = "routed"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"
    REJECTED = "rejected"


class Budget(Strict):
    max_tokens: int = Field(gt=0, le=2_000_000)
    max_wall_seconds: float = Field(gt=0, le=86_400)
    max_attempts: int = Field(default=2, ge=1, le=10)
    max_usd: float = Field(default=0.0, ge=0)


class TaskSpec(Strict):
    schema_version: Literal[1] = SCHEMA_VERSION
    task_id: str = Field(default_factory=lambda: new_id("task"))
    workspace_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,40}$")
    objective: str = Field(min_length=8, max_length=4000)
    task_class: TaskClass
    actor: str = Field(min_length=1)
    data_classification: DataClass = DataClass.INTERNAL
    allowed_capabilities: frozenset[Capability]
    forbidden_capabilities: frozenset[Capability] = frozenset()
    network_allowed: bool = False
    budget: Budget
    acceptance_checks: tuple[str, ...] = Field(min_length=1)
    risk_class: RiskClass = RiskClass.LOW
    input_artifact_hashes: tuple[str, ...] = ()
    repo_ref: str | None = None
    base_commit_sha: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,40}$")
    idempotency_key: str = Field(default_factory=lambda: uuid.uuid4().hex)

    @model_validator(mode="after")
    def _consistent(self) -> TaskSpec:
        overlap = self.allowed_capabilities & self.forbidden_capabilities
        if overlap:
            raise ValueError(f"capabilities both allowed and forbidden: {sorted(overlap)}")
        if Capability.NETWORK_EGRESS in self.allowed_capabilities and not self.network_allowed:
            raise ValueError("network_egress capability requires network_allowed=true")
        if self.task_class is TaskClass.CODE_PATCH and not (self.repo_ref and self.base_commit_sha):
            raise ValueError("code_patch tasks must pin repo_ref and base_commit_sha")
        return self

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"))


class Clarification(Strict):
    """A question a role raises instead of guessing. Escalates up the authority chain."""

    clarification_id: str = Field(default_factory=lambda: new_id("clar"))
    task_id: str
    asked_by: str
    question: str = Field(min_length=4)
    options: tuple[str, ...] = ()
    escalate_to: Literal["dispatcher", "director"] = "dispatcher"
    created_at: datetime = Field(default_factory=utcnow)


class ApprovalToken(Strict):
    """Bound to one task, one action and one artifact hash; a changed artifact needs a new token."""

    token_id: str = Field(default_factory=lambda: new_id("appr"))
    task_id: str
    action: str
    artifact_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    approver: str
    issued_at: datetime = Field(default_factory=utcnow)
    expires_at: datetime

    def authorises(self, task_id: str, action: str, artifact_hash: str, now: datetime) -> bool:
        return (
            self.task_id == task_id
            and self.action == action
            and self.artifact_hash == artifact_hash
            and now < self.expires_at
        )


class InferenceUsage(Strict):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    wall_seconds: float = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class InferenceResult(Strict):
    task_id: str
    model_id: str
    node_id: str
    output: str
    usage: InferenceUsage
    finish_reason: str | None = None
