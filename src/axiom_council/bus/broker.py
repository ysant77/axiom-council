"""Broker: the single choke point where every envelope is inspected, audited and delivered."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from axiom_council.audit import AuditLog
from axiom_council.bus.envelope import Envelope
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


class Broker:
    def __init__(self, interceptors: Sequence[Interceptor], audit: AuditLog):
        self._interceptors = list(interceptors)
        self._audit = audit
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._taps: list[Tap] = []

    def subscribe(self, role: str, handler: Handler) -> None:
        self._handlers[role].append(handler)

    def tap(self, observer: Tap) -> None:
        """Observe every envelope and its outcome (director console, tracing)."""
        self._taps.append(observer)

    def _inspect(self, envelope: Envelope) -> tuple[Finding, ...]:
        findings: list[Finding] = []
        for interceptor in self._interceptors:
            for finding in interceptor.inspect(envelope):
                findings.append(finding)
                if finding.severity == "reject":
                    return tuple(findings)
        return tuple(findings)

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
        )
        delivery = Delivery(envelope, not rejected, findings)
        for observer in self._taps:
            observer(envelope, delivery)
        if not rejected:
            for handler in self._handlers[envelope.recipient]:
                handler(envelope, findings)
        return delivery
