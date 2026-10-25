"""Sandbox profiles: per-repository build, check and path policy, stored in the config DB."""

from __future__ import annotations

import os
import re
from fnmatch import fnmatchcase
from pathlib import Path

from pydantic import Field, field_validator

from axiom_council.contracts import Strict

_CHECK_NAME = re.compile(r"^[a-z][a-z0-9_]{1,30}$")


class Limits(Strict):
    memory: str = Field(default="2g", pattern=r"^\d+[mg]$")
    cpus: float = Field(default=2.0, gt=0, le=16)
    pids: int = Field(default=256, ge=16, le=4096)
    timeout_seconds: int = Field(default=600, ge=10, le=3600)
    tmpfs_size: str = Field(default="256m", pattern=r"^\d+[mg]$")


class SandboxProfile(Strict):
    sandbox_id: str = Field(pattern=r"^[a-z][a-z0-9-]{1,40}$")
    workspace_id: str
    source_repo: str = Field(description="local git repository; $VARS and ~ are expanded")
    base_image: str = Field(pattern=r"^[\w./:-]+@sha256:[0-9a-f]{64}$")
    setup: tuple[str, ...] = Field(
        min_length=1, description="shell lines run at image build, with network, pre-edit"
    )
    checks: dict[str, tuple[str, ...]] = Field(min_length=1)
    editable_globs: tuple[str, ...] = Field(min_length=1)
    frozen_globs: tuple[str, ...] = ()
    max_files: int = Field(default=20, ge=1, le=200)
    max_file_bytes: int = Field(default=200_000, ge=1, le=2_000_000)
    limits: Limits = Field(default_factory=Limits)

    @field_validator("checks")
    @classmethod
    def _valid_checks(cls, checks: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
        for name, argv in checks.items():
            if not _CHECK_NAME.match(name):
                raise ValueError(f"invalid check name {name!r}")
            if not argv:
                raise ValueError(f"check {name!r} has no command")
        return checks

    def repo_path(self) -> Path:
        return Path(os.path.expandvars(self.source_repo)).expanduser()

    def path_verdict(self, relative: str) -> str | None:
        """None if an edit to `relative` is allowed, else the reason it is not."""
        if any(fnmatchcase(relative, g) for g in self.frozen_globs):
            return f"{relative} is frozen for this sandbox"
        if not any(fnmatchcase(relative, g) for g in self.editable_globs):
            return f"{relative} is outside the editable paths"
        return None
