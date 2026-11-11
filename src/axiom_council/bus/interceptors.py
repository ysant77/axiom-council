"""Interceptors inspect every envelope on the broker. Each one can pass, flag or reject.

Flags are recorded and travel with the delivery so downstream gates and the director can
see them; rejections stop the envelope. Heuristic detectors (injection) flag by default,
because pattern matching cannot prove intent; exact detectors (secrets, canaries) reject.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Literal, Protocol

from axiom_council.bus.envelope import Envelope, MessageKind, Trust
from axiom_council.bus.evidence import EvidenceLedger, InMemoryEvidenceLedger
from axiom_council.bus.numbers import payload_text, ungrounded_numbers
from axiom_council.contracts import Strict
from axiom_council.settings import GroundingSettings


@dataclass(frozen=True)
class Finding:
    interceptor: str
    severity: Literal["flag", "reject"]
    detail: str


class Rejected(Exception):
    def __init__(self, finding: Finding):
        super().__init__(f"{finding.interceptor}: {finding.detail}")
        self.finding = finding


class Interceptor(Protocol):
    """An interceptor may also set `terminal = True` (read with `getattr`, default false).

    A reject from a terminal interceptor stops the chain; all other rejects do not.
    """

    name: str

    def inspect(self, envelope: Envelope) -> Iterable[Finding]: ...


class Route(Strict):
    """One allowed edge: sender role may send these kinds to recipient role."""

    route_id: str
    sender: str
    recipient: str
    kinds: frozenset[MessageKind]
    require_grounded_numbers: bool = False


class RoutePolicy:
    name = "route_policy"

    def __init__(self, routes: Iterable[Route]):
        self._allowed = {(r.sender, r.recipient, k) for r in routes for k in r.kinds}

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        if (envelope.sender, envelope.recipient, envelope.kind) not in self._allowed:
            yield Finding(
                self.name,
                "reject",
                f"{envelope.sender} may not send {envelope.kind} to {envelope.recipient}",
            )


class SizeLimit:
    """Rejects oversize text. Terminal: the broker runs no later interceptor on such a message."""

    name = "size_limit"
    terminal = True

    def __init__(self, max_chars: int = 200_000):
        self.max_chars = max_chars

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        size = len(envelope.text())
        if size > self.max_chars:
            yield Finding(self.name, "reject", f"payload text {size} chars > {self.max_chars}")


_INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules)",
    r"disregard (the |your )?(system|previous) (prompt|instructions)",
    r"you are now (in )?(developer|dan|jailbreak) mode",
    r"(reveal|print|show) (your |the )?(system prompt|hidden instructions)",
    r"(send|post|upload|exfiltrate) .{0,40}(token|secret|password|api key|credentials)",
    r"(curl|wget|invoke-webrequest) +https?://",
    r"approve (this|the) (patch|plan|request) automatically",
]


class InjectionHeuristics:
    """Flags instruction-like text in untrusted content. A tripwire, not a guarantee."""

    name = "injection_heuristics"

    def __init__(self, patterns: Iterable[str] = _INJECTION_PATTERNS, *, block: bool = False):
        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._severity: Literal["flag", "reject"] = "reject" if block else "flag"

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        if envelope.trust is Trust.TRUSTED:
            return
        text = envelope.text()
        for pattern in self._patterns:
            match = pattern.search(text)
            if match:
                yield Finding(self.name, self._severity, f"instruction-like text: {match[0]!r}")


_SECRET_PATTERNS = {
    "github_token": r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    "aws_access_key": r"\bAKIA[0-9A-Z]{16}\b",
    "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "slack_token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    "generic_assignment": (
        r"(?i)\b(api[_-]?key|secret|password|token)\b\s*[:=]\s*['\"][^'\"\s]{16,}['\"]"
    ),
}


class SecretScanner:
    name = "secret_scanner"

    def __init__(self) -> None:
        self._patterns = {k: re.compile(v) for k, v in _SECRET_PATTERNS.items()}

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        text = envelope.text()
        for label, pattern in self._patterns.items():
            if pattern.search(text):
                yield Finding(self.name, "reject", f"possible secret ({label}) in payload")


class CanaryGuard:
    """Rejects any message containing a canary planted in a system prompt (prompt leak)."""

    name = "canary_guard"

    def __init__(self, canaries: Iterable[str]):
        self._canaries = tuple(c for c in canaries if c)

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        text = envelope.text()
        if any(c in text for c in self._canaries):
            yield Finding(self.name, "reject", "system prompt canary leaked into a message")


_MAX_NAMED = 20


class NumberGrounding:
    """Rejects a message whose numbers are not in its cited evidence.

    Only edges that a route marks `require_grounded_numbers` are inspected. The number rules
    are in `axiom_council.bus.numbers`. If two routes cover one edge, the flag wins. Evidence
    counts only if the ledger knows its hash for the task (council code issued it). Evidence
    the ledger does not know is ignored and named in the rejection reason. Without a ledger,
    no evidence counts.
    """

    name = "number_grounding"

    def __init__(
        self,
        routes: Iterable[Route],
        rules: GroundingSettings | None = None,
        ledger: EvidenceLedger | None = None,
        identifiers: Collection[str] = (),
    ):
        self._rules = rules or GroundingSettings()
        self._identifiers = frozenset(identifiers)
        self._ledger = ledger or InMemoryEvidenceLedger()
        self._flagged = {
            (r.sender, r.recipient, kind)
            for r in routes
            if r.require_grounded_numbers
            for kind in r.kinds
        }

    def inspect(self, envelope: Envelope) -> Iterable[Finding]:
        if (envelope.sender, envelope.recipient, envelope.kind) not in self._flagged:
            return
        issued = [
            e for e in envelope.evidence if self._ledger.known(envelope.task_id, e.content_hash)
        ]
        unissued = [e.evidence_id for e in envelope.evidence if e not in issued]
        missing = ungrounded_numbers(
            payload_text(envelope.payload), [e.text for e in issued], self._rules, self._identifiers
        )
        if missing:
            yield Finding(self.name, "reject", _describe(missing, bool(issued), unissued))


def _describe(missing: list[str], has_evidence: bool, unissued: list[str]) -> str:
    named = ", ".join(missing[:_MAX_NAMED])
    if len(missing) > _MAX_NAMED:
        named += f" and {len(missing) - _MAX_NAMED} more"
    reason = "ungrounded numbers" if has_evidence else "numbers with no issued evidence"
    if unissued:
        reason += f" (unissued evidence ignored: {', '.join(unissued)})"
    return f"{reason}: {named}"


def default_chain(
    routes: Iterable[Route],
    canaries: Iterable[str] = (),
    grounding: GroundingSettings | None = None,
    ledger: EvidenceLedger | None = None,
    identifiers: Collection[str] = (),
) -> list[Interceptor]:
    route_list = list(routes)
    return [
        RoutePolicy(route_list),
        SizeLimit(),
        SecretScanner(),
        CanaryGuard(canaries),
        NumberGrounding(route_list, grounding, ledger, identifiers),
        InjectionHeuristics(),
    ]
