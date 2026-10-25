"""Throwaway clones of a repository pinned to an exact commit, detached from any remote."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

GitRunner = Callable[[Sequence[str]], str]
_SHA = re.compile(r"^[0-9a-f]{40}$")


class SnapshotError(RuntimeError):
    pass


def run_git(args: Sequence[str]) -> str:
    git = shutil.which("git")
    if git is None:
        raise SnapshotError("git is not installed")
    result = subprocess.run(  # noqa: S603 - argv list, no shell
        [git, *args], capture_output=True, text=True, check=False, timeout=300
    )
    if result.returncode != 0:
        raise SnapshotError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


@dataclass(frozen=True)
class Snapshot:
    path: Path
    commit: str

    @property
    def short(self) -> str:
        return self.commit[:12]


def create_snapshot(
    source: Path, commit: str, dest_root: Path, *, git: GitRunner = run_git
) -> Snapshot:
    """Clone `source` into a fresh directory at exactly `commit`, with no remote left."""
    full = git(["-C", str(source), "rev-parse", "--verify", f"{commit}^{{commit}}"])
    if not _SHA.match(full):
        raise SnapshotError(f"could not resolve commit {commit!r}")
    dest = dest_root / f"{source.name.lower()}-{full[:12]}"
    if dest.exists():
        raise SnapshotError(f"snapshot directory already exists: {dest}")
    dest_root.mkdir(parents=True, exist_ok=True)
    git(["clone", "--quiet", "--no-hardlinks", str(source), str(dest)])
    git(["-C", str(dest), "checkout", "--quiet", "--detach", full])
    git(["-C", str(dest), "remote", "remove", "origin"])
    head = git(["-C", str(dest), "rev-parse", "HEAD"])
    if head != full:
        raise SnapshotError(f"snapshot HEAD {head} != requested {full}")
    return Snapshot(dest, full)
