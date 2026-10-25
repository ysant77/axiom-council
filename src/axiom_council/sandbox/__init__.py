"""Sandbox: where model-written code is applied and checked, and nowhere else.

A pinned snapshot of the target repository is prepared with network access from its own
lockfile *before* any model edit exists. Edits are applied under a path policy, and checks
run in a container with no network, a read-only root, no capabilities and resource limits.
Every command is chosen by council configuration, never by a model.
"""

from axiom_council.sandbox.edits import AppliedPatch, EditRejected, FileEdit, apply_edits
from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.sandbox.runner import CheckResult, DockerSandbox
from axiom_council.sandbox.snapshot import Snapshot, create_snapshot

__all__ = [
    "AppliedPatch",
    "CheckResult",
    "DockerSandbox",
    "EditRejected",
    "FileEdit",
    "SandboxProfile",
    "Snapshot",
    "apply_edits",
    "create_snapshot",
]
