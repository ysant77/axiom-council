"""Apply model-proposed whole-file edits to a snapshot under the profile's path policy.

All edits in a batch are validated before any file is written, so a rejected batch leaves
the snapshot untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from axiom_council.contracts import Strict, content_hash
from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.sandbox.snapshot import GitRunner, Snapshot, run_git


class EditRejected(ValueError):
    pass


class FileEdit(Strict):
    path: str = Field(min_length=1, max_length=300)
    action: Literal["create", "replace", "delete"]
    content: str = ""


@dataclass(frozen=True)
class AppliedPatch:
    changed: tuple[str, ...]
    diff: str
    patch_hash: str


def _normalise(raw: str) -> str:
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "\\" in raw or raw != str(path):
        raise EditRejected(f"unsafe path {raw!r}")
    return str(path)


def _target(root: Path, relative: str) -> Path:
    target = root / relative
    for parent in [target, *target.parents]:
        if parent == root:
            break
        if parent.is_symlink():
            raise EditRejected(f"{relative} goes through a symlink")
    if not target.resolve().is_relative_to(root.resolve()):
        raise EditRejected(f"{relative} escapes the snapshot")
    return target


def _validate(snapshot: Snapshot, edits: list[FileEdit], profile: SandboxProfile) -> list[Path]:
    if not edits:
        raise EditRejected("no edits supplied")
    if len(edits) > profile.max_files:
        raise EditRejected(f"{len(edits)} files > limit {profile.max_files}")
    paths = [_normalise(e.path) for e in edits]
    if len(set(paths)) != len(paths):
        raise EditRejected("the same path is edited twice")
    targets: list[Path] = []
    for edit, relative in zip(edits, paths, strict=True):
        if reason := profile.path_verdict(relative):
            raise EditRejected(reason)
        if len(edit.content.encode()) > profile.max_file_bytes:
            raise EditRejected(f"{relative} exceeds {profile.max_file_bytes} bytes")
        target = _target(snapshot.path, relative)
        if edit.action == "create" and target.exists():
            raise EditRejected(f"{relative} already exists; use replace")
        if edit.action != "create" and not target.is_file():
            raise EditRejected(f"{relative} does not exist")
        targets.append(target)
    return targets


def apply_edits(
    snapshot: Snapshot,
    edits: list[FileEdit],
    profile: SandboxProfile,
    *,
    git: GitRunner = run_git,
) -> AppliedPatch:
    targets = _validate(snapshot, edits, profile)
    for edit, target in zip(edits, targets, strict=True):
        if edit.action == "delete":
            target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.content, encoding="utf-8")
    # Stage only what this batch touched: stray files (e.g. written by a check run) never
    # leak into the patch.
    relative = [str(t.relative_to(snapshot.path)) for t in targets]
    git(["-C", str(snapshot.path), "add", "--all", "--", *relative])
    diff = git(["-C", str(snapshot.path), "diff", "--cached", "--no-color", snapshot.commit])
    changed = git(["-C", str(snapshot.path), "diff", "--cached", "--name-only", snapshot.commit])
    return AppliedPatch(tuple(changed.splitlines()), diff, content_hash(diff))
