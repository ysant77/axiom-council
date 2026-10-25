from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.sandbox.runner import Completed

DIGEST = "ghcr.io/astral-sh/uv@sha256:" + "a" * 64


def git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def make_repo(root: Path) -> tuple[Path, str]:
    repo = root / "Demo"
    (repo / "src" / "demo").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "src" / "demo" / "core.py").write_text("def add(a, b):\n    return a + b\n")
    (repo / "tests" / "test_core.py").write_text("def test_add():\n    assert True\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "dev@example.com")
    git(repo, "config", "user.name", "dev")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo, git(repo, "rev-parse", "HEAD")


def profile(repo: Path, **overrides: Any) -> SandboxProfile:
    fields: dict[str, Any] = {
        "sandbox_id": "demo",
        "workspace_id": "sandbox",
        "source_repo": str(repo),
        "base_image": DIGEST,
        "setup": ("uv sync --locked",),
        "checks": {"tests": ("python", "-m", "pytest", "-q")},
        "editable_globs": ("src/demo/*.py",),
        "frozen_globs": ("tests/*",),
    }
    fields.update(overrides)
    return SandboxProfile(**fields)


class FakeRunner:
    """Records every command; answers from a queue of results (default: success)."""

    def __init__(self, results: Sequence[Completed] = ()):
        self.calls: list[tuple[list[str], str | None]] = []
        self._results = list(results)

    def __call__(self, argv: Sequence[str], stdin: str | None, timeout: float) -> Completed:
        self.calls.append((list(argv), stdin))
        return self._results.pop(0) if self._results else Completed(0, "ok", "")
