"""Envelope: the typed, hashed unit of communication between agents."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import Field, model_validator

from axiom_council.contracts import Strict, content_hash, new_id, utcnow


class MessageKind(StrEnum):
    TASK = "task"
    PLAN = "plan"
    EDIT = "edit"
    REVIEW = "review"
    CLARIFY = "clarify"
    ANSWER = "answer"
    VERDICT = "verdict"
    EVIDENCE = "evidence"


class Trust(StrEnum):
    """`trusted`: produced by council code or the director. `untrusted`: anything a model
    wrote or anything read from a repository, document or the network."""

    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class Envelope(Strict):
    msg_id: str = Field(default_factory=lambda: new_id("msg"))
    task_id: str = Field(min_length=1)
    parent_id: str | None = None
    sender: str = Field(min_length=1)
    recipient: str = Field(min_length=1)
    kind: MessageKind
    trust: Trust
    payload: dict[str, Any]
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=utcnow)

    @model_validator(mode="after")
    def _hash_matches(self) -> Envelope:
        if content_hash(self.payload) != self.payload_hash:
            raise ValueError("payload hash does not match payload")
        return self

    def text(self) -> str:
        """All string content in the payload, for content inspection."""
        return "\n".join(_strings(self.payload))


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _strings(v)]
    return []


def seal(
    *,
    task_id: str,
    sender: str,
    recipient: str,
    kind: MessageKind,
    payload: dict[str, Any],
    trust: Trust,
    parent: Envelope | None = None,
) -> Envelope:
    """Build an envelope with its payload hash. Untrusted parents taint their children."""
    if parent is not None and parent.trust is Trust.UNTRUSTED:
        trust = Trust.UNTRUSTED
    return Envelope(
        task_id=task_id,
        parent_id=parent.msg_id if parent else None,
        sender=sender,
        recipient=recipient,
        kind=kind,
        trust=trust,
        payload=payload,
        payload_hash=content_hash(payload),
    )
