"""Offline verification of a council home (backlog F14): `council verify`.

It reads the audit log and its anchors, the task store, the inbox, the config store and
bake-off results files. It needs no model, no network and no GPU. Each check reports pass,
fail, cannot_resolve or no_provenance with a reason. The report passes only when no check blocks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from axiom_council.audit import AuditTamperError, open_audit
from axiom_council.inbox import Inbox
from axiom_council.tasks import TaskStore
from axiom_council.verify_approvals import Evidence, approval_checks
from axiom_council.verify_checks import Check, ConfigHistory, Status, failed, passed
from axiom_council.verify_results import results_checks

__all__ = ["Check", "Status", "VerifyReport", "verify_run"]


@dataclass(frozen=True)
class VerifyReport:
    checks: tuple[Check, ...]
    allow_legacy: bool = False

    @property
    def passed(self) -> bool:
        return not any(check.blocking for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "allow_legacy": self.allow_legacy,
            "counts": dict(Counter(str(c.status) for c in self.checks)),
            "checks": [c.as_dict() for c in self.checks],
        }

    def lines(self) -> list[str]:
        out = []
        for c in self.checks:
            label = "warn" if c.status is not Status.PASS and not c.blocking else str(c.status)
            out.append(f"{label:<15} {c.name:<10} {c.subject}  {c.reason}")
        blocking = sum(c.blocking for c in self.checks)
        verdict = "PASSED" if self.passed else f"FAILED ({blocking} blocking)"
        out.append(f"verify {verdict}: {len(self.checks)} checks")
        return out


def _audit(home: Path) -> tuple[Check, list[dict[str, Any]]]:
    """The audit check, and the records when they can be read."""
    log = open_audit(home)
    try:
        head = log.head()
    except AuditTamperError as exc:
        try:
            records = log.records()
        except AuditTamperError:
            records = []
        return failed("audit", log.path.name, str(exc)), records
    reason = f"{head.count} records, head {head.head[:12]}; chain and anchors match"
    return passed("audit", log.path.name, reason), log.records()


def _results_paths(home: Path, results: Sequence[Path] | None) -> list[Path]:
    if results:
        return list(results)
    return sorted((home / "bakeoff").glob("results-*.jsonl"))


def verify_run(
    home: Path, results: Sequence[Path] | None = None, *, allow_legacy: bool = False
) -> VerifyReport:
    """Verify every recorded run under home; with no results given, every results file there."""
    if not home.is_dir():
        return VerifyReport((failed("home", str(home), "council home not found"),), allow_legacy)
    audit, records = _audit(home)
    checks = [audit]
    tasks_db = home / "tasks.sqlite3"
    tasks = TaskStore(tasks_db) if tasks_db.is_file() else None
    configs = ConfigHistory(home / "config.sqlite3")
    inbox = Inbox(home / "inbox") if (home / "inbox").is_dir() else None
    try:
        checks += approval_checks(records, Evidence(tasks, inbox, configs))
    finally:
        if tasks is not None:
            tasks.close()
    for path in _results_paths(home, results):
        checks += results_checks(path, configs, allow_legacy=allow_legacy)
    return VerifyReport(tuple(checks), allow_legacy)
