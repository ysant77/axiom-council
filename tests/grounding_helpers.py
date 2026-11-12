"""Shared builders for the number-grounding tests."""

from __future__ import annotations

from typing import Any

from axiom_council.bus import Envelope, MessageKind, Trust, seal
from axiom_council.bus.envelope import EvidenceItem
from axiom_council.bus.evidence import EvidenceLedger
from axiom_council.bus.interceptors import Route

FLAGGED = Route(
    route_id="builder-gatekeeper-grounded",
    sender="builder",
    recipient="gatekeeper",
    kinds=frozenset({MessageKind.EDIT}),
    require_grounded_numbers=True,
)
PLAIN = Route(
    route_id="builder-dispatcher",
    sender="builder",
    recipient="dispatcher",
    kinds=frozenset({MessageKind.CLARIFY}),
)


def make_env(
    payload: dict[str, Any],
    *texts: str,
    recipient: str = "gatekeeper",
    kind: MessageKind = MessageKind.EDIT,
) -> Envelope:
    return seal(
        task_id="task_1",
        sender="builder",
        recipient=recipient,
        kind=kind,
        payload=payload,
        trust=Trust.UNTRUSTED,
        evidence=tuple(EvidenceItem.of(f"ev_{i}", t) for i, t in enumerate(texts)),
    )


def issue(ledger: EvidenceLedger, env: Envelope) -> None:
    for item in env.evidence:
        ledger.record(env.task_id, item.content_hash)
