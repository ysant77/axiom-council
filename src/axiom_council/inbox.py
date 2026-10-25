"""Director inbox: approvals and clarifications waiting on the human, stored as JSON files.

The CLI is the first channel; Telegram and mobile clients will read the same store.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from axiom_council.contracts import ApprovalToken, Clarification, Strict, new_id, utcnow

ItemKind = Literal["approval", "clarification"]


MIN_REASON_CHARS = 5


def _require_reason(reason: str) -> None:
    if len(reason.strip()) < MIN_REASON_CHARS:
        raise ValueError("a decision needs a reason of at least five characters")


class InboxItem(Strict):
    item_id: str = Field(default_factory=lambda: new_id("inbox"))
    kind: ItemKind
    task_id: str
    summary: str
    action: str | None = None
    artifact_hash: str | None = None
    clarification: Clarification | None = None
    status: Literal["pending", "approved", "rejected", "answered"] = "pending"
    response: str | None = None
    token: ApprovalToken | None = None


class Inbox:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, item_id: str) -> Path:
        if not item_id.replace("_", "").isalnum():
            raise ValueError(f"invalid inbox id {item_id!r}")
        return self.root / f"{item_id}.json"

    def _save(self, item: InboxItem) -> None:
        path = self._path(item.item_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(item.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX and Windows

    def get(self, item_id: str) -> InboxItem:
        data: dict[str, Any] = json.loads(self._path(item_id).read_text(encoding="utf-8"))
        return InboxItem.model_validate(data)

    def request_approval(
        self, *, task_id: str, action: str, artifact_hash: str, summary: str
    ) -> InboxItem:
        item = InboxItem(
            kind="approval",
            task_id=task_id,
            action=action,
            artifact_hash=artifact_hash,
            summary=summary,
        )
        self._save(item)
        return item

    def ask(self, clarification: Clarification) -> InboxItem:
        item = InboxItem(
            kind="clarification",
            task_id=clarification.task_id,
            summary=clarification.question,
            clarification=clarification,
        )
        self._save(item)
        return item

    def pending(self) -> list[InboxItem]:
        items = [self.get(p.stem) for p in sorted(self.root.glob("*.json"))]
        return [i for i in items if i.status == "pending"]

    def approve(self, item_id: str, approver: str, ttl: timedelta, reason: str) -> ApprovalToken:
        _require_reason(reason)
        item = self.get(item_id)
        if item.kind != "approval" or item.status != "pending":
            raise ValueError(f"{item_id} is not a pending approval")
        if item.action is None or item.artifact_hash is None:
            raise ValueError(f"{item_id} has no action or artifact to approve")
        token = ApprovalToken(
            task_id=item.task_id,
            action=item.action,
            artifact_hash=item.artifact_hash,
            approver=approver,
            expires_at=utcnow() + ttl,
        )
        response = f"{token.token_id}: {reason.strip()}"
        self._save(
            item.model_copy(update={"status": "approved", "response": response, "token": token})
        )
        return token

    def reject(self, item_id: str, reason: str) -> None:
        _require_reason(reason)
        item = self.get(item_id)
        if item.status != "pending":
            raise ValueError(f"{item_id} is not pending")
        self._save(item.model_copy(update={"status": "rejected", "response": reason}))

    def answer(self, item_id: str, answer: str) -> None:
        item = self.get(item_id)
        if item.kind != "clarification" or item.status != "pending":
            raise ValueError(f"{item_id} is not a pending clarification")
        self._save(item.model_copy(update={"status": "answered", "response": answer}))
