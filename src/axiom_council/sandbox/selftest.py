"""Sandbox self-test: the acceptance gate for a profile before any model touches the repo.

It snapshots the repository, builds the pre-edit image, proves the container has no
network, proves frozen paths reject edits, and requires the untouched code to pass every
allowlisted check (a baseline that fails cannot judge a refactor).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from axiom_council.audit import AuditLog
from axiom_council.sandbox.edits import EditRejected, FileEdit, apply_edits
from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.sandbox.runner import CheckResult, DockerSandbox
from axiom_council.sandbox.snapshot import GitRunner, Snapshot, create_snapshot, run_git


@dataclass
class SelftestReport:
    snapshot: Snapshot
    image: str
    isolation: CheckResult
    frozen_edit_rejected: bool
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return (
            self.isolation.passed
            and self.frozen_edit_rejected
            and all(c.passed for c in self.checks)
        )


def _frozen_probe(snapshot: Snapshot, profile: SandboxProfile, git: GitRunner) -> bool:
    if not profile.frozen_globs:
        return True
    probe = FileEdit(path="tests/test_injected.py", action="create", content="assert True\n")
    try:
        apply_edits(snapshot, [probe], profile, git=git)
    except EditRejected:
        return True
    return False


def selftest(
    profile: SandboxProfile,
    sandbox: DockerSandbox,
    home: Path,
    audit: AuditLog,
    *,
    commit: str = "HEAD",
    git: GitRunner = run_git,
) -> SelftestReport:
    task_id = f"selftest-{profile.sandbox_id}"
    snapshot = create_snapshot(profile.repo_path(), commit, home / "sandboxes", git=git)
    audit.append("sandbox_snapshot", task_id, "gatekeeper", commit=snapshot.commit)
    image = sandbox.prepare(snapshot)
    report = SelftestReport(
        snapshot=snapshot,
        image=image,
        isolation=sandbox.verify_isolation(snapshot),
        frozen_edit_rejected=_frozen_probe(snapshot, profile, git),
        checks=sandbox.run_all(snapshot),
    )
    audit.append(
        "sandbox_selftest",
        task_id,
        "gatekeeper",
        image=image,
        passed=report.passed,
        isolation=report.isolation.passed,
        frozen_edit_rejected=report.frozen_edit_rejected,
        checks={c.name: c.passed for c in report.checks},
    )
    return report
