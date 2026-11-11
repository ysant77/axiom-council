"""Shared parts of offline run verification: one check result, and the config history.

The config history reads the versioned config store only. It rebuilds the registry that was
active at a past time, or the one whose config hash a result row names.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from axiom_council.configstore import ConfigStore
from axiom_council.provenance import config_hash
from axiom_council.registry import Registry


class Status(StrEnum):
    PASS = "pass"  # noqa: S105 - a status, not a secret
    FAIL = "fail"
    CANNOT_RESOLVE = "cannot_resolve"  # the evidence needed to decide is not available
    NO_PROVENANCE = "no_provenance"  # a result row written before provenance was recorded


@dataclass(frozen=True)
class Check:
    name: str
    subject: str
    status: Status
    reason: str
    blocking: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "subject": self.subject,
            "status": str(self.status),
            "blocking": self.blocking,
            "reason": self.reason,
        }


def passed(name: str, subject: str, reason: str) -> Check:
    return Check(name, subject, Status.PASS, reason, blocking=False)


def failed(name: str, subject: str, reason: str, status: Status = Status.FAIL) -> Check:
    """A check that did not pass. Every status except PASS blocks unless stated otherwise."""
    return Check(name, subject, status, reason, blocking=True)


class ConfigHistory:
    """A read-only view of the config store at past times. A missing store resolves nothing."""

    def __init__(self, path: Path):
        self._path = path
        self._by_hash: dict[str, Registry] | None = None

    def _build(self, store: ConfigStore, moment: str) -> Registry | None:
        try:
            return Registry.from_store(store, as_of=moment)
        except (ValueError, KeyError, TypeError):
            return None  # an incomplete config, for example in the middle of a seed

    def at(self, moment: str) -> Registry | None:
        """The registry that was active at an ISO time."""
        if not self._path.is_file():
            return None
        store = ConfigStore(self._path)
        try:
            return self._build(store, moment)
        finally:
            store.close()

    def by_config_hash(self, digest: str) -> Registry | None:
        """The registry, at any past time, whose config hash equals digest."""
        if self._by_hash is None:
            self._by_hash = {}
            if self._path.is_file():
                store = ConfigStore(self._path)
                try:
                    for moment in store.version_times():
                        registry = self._build(store, moment)
                        if registry is not None:
                            self._by_hash.setdefault(config_hash(registry), registry)
                finally:
                    store.close()
        return self._by_hash.get(digest)
