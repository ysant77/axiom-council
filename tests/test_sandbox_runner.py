from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from axiom_council import cli
from axiom_council.audit import AuditLog
from axiom_council.sandbox import create_snapshot
from axiom_council.sandbox.runner import Completed, DockerSandbox, SandboxError, run_command
from axiom_council.sandbox.selftest import selftest

from .conftest import CONFIG_DIR
from .sandbox_helpers import FakeRunner, make_repo, profile

HARDENING = [
    ["--network", "none"],
    ["--read-only"],
    ["--cap-drop", "ALL"],
    ["--security-opt", "no-new-privileges"],
    ["--user", "10001:10001"],
]


def _contains(argv: list[str], flag: list[str]) -> bool:
    return any(argv[i : i + len(flag)] == flag for i in range(len(argv)))


def test_prepare_reuses_or_builds_image(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    snap = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    cached = FakeRunner()
    assert DockerSandbox(profile(repo), run=cached).prepare(snap).endswith(snap.short)
    assert len(cached.calls) == 1  # inspect only

    fresh = FakeRunner([Completed(1, "", "missing"), Completed(0, "sha", "")])
    DockerSandbox(profile(repo), run=fresh).prepare(snap)
    build_argv, dockerfile = fresh.calls[1]
    assert build_argv[:2] == ["docker", "build"] and dockerfile is not None
    assert "FROM ghcr.io/astral-sh/uv@sha256:" in dockerfile
    assert "RUN uv sync --locked" in dockerfile

    broken = FakeRunner([Completed(1, "", ""), Completed(1, "", "boom")])
    with pytest.raises(SandboxError, match="boom"):
        DockerSandbox(profile(repo), run=broken).prepare(snap)


def test_checks_run_hardened_and_allowlisted(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    snap = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    runner = FakeRunner([Completed(0, "3 passed", ""), Completed(1, "", "E1 lint")])
    box = DockerSandbox(profile(repo, checks={"tests": ("pytest",), "lint": ("ruff",)}), run=runner)
    results = box.run_all(snap)
    assert [(r.name, r.passed) for r in results] == [("tests", True), ("lint", False)]
    argv = runner.calls[0][0]
    assert all(_contains(argv, flag) for flag in HARDENING)
    assert argv[-1] == "pytest" and "E1 lint" in results[1].output_tail
    with pytest.raises(SandboxError, match="allowlist"):
        box.run_check(snap, "rm -rf /")


def test_timeout_kills_the_container(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    snap = create_snapshot(repo, "HEAD", tmp_path / "boxes")
    runner = FakeRunner([Completed(124, "", "timed out", timed_out=True)])
    result = DockerSandbox(profile(repo), run=runner).run_check(snap, "tests")
    assert result.timed_out and not result.passed
    assert runner.calls[1][0][:3] == ["docker", "rm", "--force"]


def test_run_command_handles_missing_and_timeouts() -> None:
    assert run_command(["definitely-not-a-binary-xyz"], None, 5).returncode == 127
    assert run_command(["git", "--version"], None, 30).returncode == 0
    slow = run_command(["sleep", "5"], None, 0.2)
    assert slow.timed_out and slow.returncode == 124


def test_selftest_requires_isolation_frozen_paths_and_green_baseline(tmp_path: Path) -> None:
    repo, _ = make_repo(tmp_path)
    audit = AuditLog(tmp_path / "audit.jsonl")
    ok = selftest(profile(repo), DockerSandbox(profile(repo), run=FakeRunner()), tmp_path, audit)
    assert ok.passed and ok.frozen_edit_rejected
    assert audit.for_task("selftest-demo")[-1]["data"]["passed"] is True

    leaky = FakeRunner([Completed(0, "", ""), Completed(1, "", "")])  # inspect ok, probe fails
    shutil.rmtree(tmp_path / "sandboxes")
    bad = selftest(profile(repo), DockerSandbox(profile(repo), run=leaky), tmp_path, audit)
    assert not bad.passed and not bad.isolation.passed

    shutil.rmtree(tmp_path / "sandboxes")
    open_profile = profile(repo, frozen_globs=(), editable_globs=("**",))
    assert selftest(
        open_profile, DockerSandbox(open_profile, run=FakeRunner()), tmp_path, audit
    ).frozen_edit_rejected  # nothing frozen: probe skipped


def test_cli_sandbox_selftest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, _ = make_repo(tmp_path)
    seed = tmp_path / "seed"
    shutil.copytree(CONFIG_DIR, seed)
    text = (seed / "sandboxes.yaml").read_text()
    (seed / "sandboxes.yaml").write_text(
        text.replace("${HOME}/Documents/personal/PageTrace", str(repo))
    )
    monkeypatch.setenv("AXIOM_COUNCIL_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AXIOM_COUNCIL_CONFIG", str(seed))
    fail = FakeRunner([Completed(0, "", ""), Completed(0, "", ""), Completed(1, "x", "")])
    monkeypatch.setattr(cli, "DockerSandbox", lambda p: DockerSandbox(p, run=fail))
    assert cli.main(["sandbox", "selftest", "pagetrace"]) == 1
    out = capsys.readouterr().out
    assert "network   denied" in out and "FAIL" in out and "selftest  FAILED" in out
    assert cli.main(["sandbox", "selftest", "nope"]) == 2


@pytest.mark.live
def test_live_pagetrace_selftest(tmp_path: Path) -> None:  # pragma: no cover - needs Docker
    if subprocess.run(["docker", "info"], capture_output=True, check=False).returncode != 0:
        pytest.skip("Docker daemon not running")
    from axiom_council.council import open_registry

    prof = open_registry(tmp_path, CONFIG_DIR).sandboxes["pagetrace"]
    report = selftest(prof, DockerSandbox(prof), tmp_path, AuditLog(tmp_path / "a.jsonl"))
    assert report.passed, [c.output_tail for c in report.checks if not c.passed]
