from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from axiom_council.sandbox import EditRejected, FileEdit, apply_edits, create_snapshot
from axiom_council.sandbox.snapshot import SnapshotError

from .sandbox_helpers import git, make_repo, profile


def test_profile_validation_and_path_policy(tmp_path: Path) -> None:
    p = profile(tmp_path)
    assert p.path_verdict("src/demo/core.py") is None
    assert "frozen" in (p.path_verdict("tests/test_core.py") or "")
    assert "outside" in (p.path_verdict("setup.py") or "")
    with pytest.raises(ValidationError):
        profile(tmp_path, base_image="python:3.12")  # must be pinned by digest
    with pytest.raises(ValidationError, match="invalid check name"):
        profile(tmp_path, checks={"Bad Name": ("x",)})
    with pytest.raises(ValidationError, match="no command"):
        profile(tmp_path, checks={"tests": ()})
    assert profile(tmp_path, source_repo="~/x").repo_path() == Path.home() / "x"


def test_snapshot_is_pinned_and_detached(tmp_path: Path) -> None:
    repo, sha = make_repo(tmp_path)
    (repo / "src" / "demo" / "core.py").write_text("# uncommitted change\n")
    snap = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    assert snap.commit == sha and snap.path.name == f"demo-{sha[:12]}"
    assert "uncommitted" not in (snap.path / "src" / "demo" / "core.py").read_text()
    assert git(snap.path, "remote") == ""
    with pytest.raises(SnapshotError, match="already exists"):
        create_snapshot(repo, sha, tmp_path / "boxes")
    with pytest.raises(SnapshotError):
        create_snapshot(repo, "no-such-ref", tmp_path / "other")


@pytest.fixture
def snap_and_profile(tmp_path: Path):  # type: ignore[no-untyped-def]
    repo, _ = make_repo(tmp_path)
    return create_snapshot(repo, "HEAD", tmp_path / "boxes"), profile(repo)


def test_edits_apply_with_diff_and_hash(snap_and_profile) -> None:  # type: ignore[no-untyped-def]
    snap, prof = snap_and_profile
    patch = apply_edits(
        snap,
        [
            FileEdit(
                path="src/demo/core.py",
                action="replace",
                content="def add(a, b):\n    return b + a\n",
            ),
            FileEdit(path="src/demo/util.py", action="create", content="X = 1\n"),
        ],
        prof,
    )
    assert patch.changed == ("src/demo/core.py", "src/demo/util.py")
    assert "+    return b + a" in patch.diff and len(patch.patch_hash) == 64


@pytest.mark.parametrize(
    ("edit", "reason"),
    [
        (FileEdit(path="tests/test_core.py", action="replace", content=""), "frozen"),
        (FileEdit(path="pyproject.toml", action="create", content=""), "outside"),
        (FileEdit(path="../escape.py", action="create", content=""), "unsafe"),
        (FileEdit(path="/etc/passwd", action="replace", content=""), "unsafe"),
        (FileEdit(path="src/demo/./core.py", action="replace", content=""), "unsafe"),
        (FileEdit(path="src/demo/core.py", action="create", content=""), "already exists"),
        (FileEdit(path="src/demo/missing.py", action="delete"), "does not exist"),
        (FileEdit(path="src/demo/big.py", action="create", content="x" * 300_000), "exceeds"),
    ],
)
def test_bad_edits_are_rejected(snap_and_profile, edit: FileEdit, reason: str) -> None:  # type: ignore[no-untyped-def]
    snap, prof = snap_and_profile
    with pytest.raises(EditRejected, match=reason):
        apply_edits(snap, [edit], prof)


def test_batch_is_atomic_and_bounded(snap_and_profile) -> None:  # type: ignore[no-untyped-def]
    snap, prof = snap_and_profile
    good = FileEdit(path="src/demo/new.py", action="create", content="Y = 2\n")
    bad = FileEdit(path="tests/test_core.py", action="replace", content="")
    with pytest.raises(EditRejected):
        apply_edits(snap, [good, bad], prof)
    assert not (snap.path / "src/demo/new.py").exists()
    with pytest.raises(EditRejected, match="twice"):
        apply_edits(snap, [good, good], prof)
    with pytest.raises(EditRejected, match="no edits"):
        apply_edits(snap, [], prof)
    many = [FileEdit(path=f"src/demo/m{i}.py", action="create") for i in range(21)]
    with pytest.raises(EditRejected, match="limit"):
        apply_edits(snap, many, prof)


def test_symlinks_cannot_smuggle_writes(snap_and_profile, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    snap, prof = snap_and_profile
    outside = tmp_path / "outside.py"
    outside.write_text("secret = 1\n")
    (snap.path / "src/demo/link.py").symlink_to(outside)
    with pytest.raises(EditRejected, match="symlink"):
        apply_edits(snap, [FileEdit(path="src/demo/link.py", action="replace")], prof)
    assert outside.read_text() == "secret = 1\n"
    edit = FileEdit(path="src/demo/core.py", action="delete")
    assert apply_edits(snap, [edit], prof).changed == ("src/demo/core.py",)
