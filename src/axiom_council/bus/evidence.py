"""Evidence ledger: which evidence texts council code has issued for a task.

A sender can attach any text to its own envelope, so the text proves nothing. Only the content
hashes that this ledger holds for the task count as evidence. The broker records the hashes of
evidence items carried by delivered messages from an issuer role (a setting). The in-memory
ledger below loses its content when the process ends. A durable ledger is another class that
follows the same protocol.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol


class EvidenceLedger(Protocol):
    def record(self, task_id: str, content_hash: str) -> None: ...

    def known(self, task_id: str, content_hash: str) -> bool: ...


class InMemoryEvidenceLedger:
    def __init__(self) -> None:
        self._hashes: dict[str, set[str]] = defaultdict(set)

    def record(self, task_id: str, content_hash: str) -> None:
        self._hashes[task_id].add(content_hash)

    def known(self, task_id: str, content_hash: str) -> bool:
        return content_hash in self._hashes.get(task_id, ())
