"""Broker: the single choke point where every envelope is inspected, audited and delivered."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field

from axiom_council.audit import AuditLog
from axiom_council.bus.envelope import Envelope
from axiom_council.bus.evidence import EvidenceLedger
from axiom_council.bus.interceptors import Finding, Interceptor

Handler = Callable[[Envelope, tuple[Finding, ...]], None]
Tap = Callable[[Envelope, "Delivery"], None]


@dataclass(frozen=True)
class Delivery:
    envelope: Envelope
    delivered: bool
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def flagged(self) -> bool:
        return any(f.severity == "flag" for f in self.findings)


def _evidence_fields(envelope: Envelope) -> dict[str, list[str]]:
    """Audit the cited evidence by hash; envelopes without evidence keep the old record shape."""
    if not envelope.evidence:
        return {}
    return {"evidence_hashes": [item.content_hash for item in envelope.evidence]}


class Broker:
    def __init__(
        self,
        interceptors: Sequence[Interceptor],
        audit: AuditLog,
        *,
        ledger: EvidenceLedger | None = None,
        evidence_issuers: Collection[str] = (),
    ):
        self._interceptors = list(interceptors)
        self._audit = audit
        self._ledger = ledger
        self._issuers = frozenset(evidence_issuers)
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._taps: list[Tap] = []

    def subscribe(self, role: str, handler: Handler) -> None:
        self._handlers[role].append(handler)

    def tap(self, observer: Tap) -> None:
        """Observe every envelope and its outcome (director console, tracing)."""
        self._taps.append(observer)

    def _inspect(self, envelope: Envelope) -> tuple[Finding, ...]:
        """Run every interceptor, in chain order, and keep all findings.

        A reject does not stop the chain: a later interceptor still flags what it sees, so the
        audit record of a rejected message is complete and independent of the chain order.
        """
        return tuple(f for interceptor in self._interceptors for f in interceptor.inspect(envelope))

    def _issue_evidence(self, envelope: Envelope) -> None:
        """Record the evidence of a delivered message when its sender is an issuer role."""
        if self._ledger is not None and envelope.sender in self._issuers:
            for item in envelope.evidence:
                self._ledger.record(envelope.task_id, item.content_hash)

    def publish(self, envelope: Envelope) -> Delivery:
        findings = self._inspect(envelope)
        rejected = any(f.severity == "reject" for f in findings)
        if not rejected and not self._handlers.get(envelope.recipient):
            findings += (Finding("broker", "reject", f"no subscriber for {envelope.recipient}"),)
            rejected = True

        self._audit.append(
            "message_rejected" if rejected else "message_delivered",
            envelope.task_id,
            envelope.sender,
            msg_id=envelope.msg_id,
            parent_id=envelope.parent_id,
            recipient=envelope.recipient,
            kind=str(envelope.kind),
            trust=str(envelope.trust),
            payload_hash=envelope.payload_hash,
            findings=[f.__dict__ for f in findings],
            **_evidence_fields(envelope),
        )
        delivery = Delivery(envelope, not rejected, findings)
        for observer in self._taps:
            observer(envelope, delivery)
        if not rejected:
            self._issue_evidence(envelope)
            for handler in self._handlers[envelope.recipient]:
                handler(envelope, findings)
        return delivery
