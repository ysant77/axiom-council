"""Director approval inside a graph: request an inbox item, pause, then verify the token.

The pause is a LangGraph interrupt; the decision lives in our inbox. Resuming succeeds only
with a token that authorises this task, this action and the hash of the patch as it stands
in the checkpoint now, so a patch changed after approval is refused.

When findings were screened, the request also tells the director which supported high
findings the arbiter rejected, with reason and cited check, and how many were unsupported.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from langgraph.types import interrupt

from axiom_council.audit import AuditLog
from axiom_council.contracts import content_hash
from axiom_council.inbox import Inbox
from axiom_council.pipeline.guard import PipelineState

_MAX_REASON_CHARS = 200


class ApprovalRefused(PermissionError):
    pass


def review_note(state: PipelineState) -> str:
    """The finding review for the director; empty when the run did not screen findings."""
    if "rejected_high" not in state:
        return ""
    rejected = state["rejected_high"]
    note = f"; rejected high findings: {len(rejected)}"
    if rejected:
        lines = [
            f"{r['finding_id']} (check {r['check']}): {r['reason'][:_MAX_REASON_CHARS]}"
            for r in rejected
        ]
        note += " [" + "; ".join(lines) + "]"
    return note + f"; unsupported findings in this run: {state.get('unsupported_findings', 0)}"


def patch_hash(patch: str) -> str:
    return content_hash({"patch": patch})


class ApprovalSteps:
    def __init__(self, inbox: Inbox, audit: AuditLog, action: str, clock: Callable[[], datetime]):
        self._inbox = inbox
        self._audit = audit
        self._action = action
        self._clock = clock

    def request(self, state: PipelineState) -> dict[str, Any]:
        """Idempotent: a re-run for the same artifact reuses the pending item."""
        if state.get("inbox_item"):
            item = self._inbox.get(state["inbox_item"])
            if item.artifact_hash == state["artifact_hash"]:
                return {}
        summary = f"{state['objective'][:100]} (patch {state['artifact_hash'][:12]})"
        item = self._inbox.request_approval(
            task_id=state["task_id"],
            action=self._action,
            artifact_hash=state["artifact_hash"],
            summary=summary + review_note(state),
        )
        self._audit.append(
            "approval_requested",
            state["task_id"],
            "council",
            item=item.item_id,
            artifact_hash=item.artifact_hash,
            summary=item.summary,
            rejected_high=state.get("rejected_high", []),
            unsupported_findings=state.get("unsupported_findings", 0),
        )
        return {"inbox_item": item.item_id}

    def await_decision(self, state: PipelineState) -> dict[str, Any]:
        interrupt({"inbox_item": state["inbox_item"], "artifact_hash": state["artifact_hash"]})
        task_id = state["task_id"]
        item = self._inbox.get(state["inbox_item"])
        if item.status == "rejected":
            return {"outcome": "rejected"}
        current = patch_hash(state.get("patch", ""))
        token = item.token
        if token is None or not token.authorises(task_id, self._action, current, self._clock()):
            self._audit.append("approval_invalid", task_id, "council", item=item.item_id)
            raise ApprovalRefused(f"{task_id}: no valid approval for patch {current[:12]}")
        self._audit.append("approval_verified", task_id, token.approver, token=token.token_id)
        return {"outcome": "approved", "approved_by": token.approver}
