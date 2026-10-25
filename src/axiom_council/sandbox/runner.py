"""Container runner: builds a pre-edit environment image, then runs allowlisted checks on an
edited snapshot with no network, a read-only root, no capabilities and resource limits."""

from __future__ import annotations

import secrets
import shutil
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.sandbox.snapshot import Snapshot

OUTPUT_TAIL = 4000
SANDBOX_UID = "10001"

# Run inside the container by council code (never by a model) to prove isolation.
_NETWORK_PROBE = (
    "import socket,sys\n"
    "try:\n"
    "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
    "except OSError:\n"
    "    sys.exit(0)\n"
    "sys.exit(1)\n"
)


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False


CommandRunner = Callable[[Sequence[str], str | None, float], Completed]


def run_command(argv: Sequence[str], stdin: str | None, timeout: float) -> Completed:
    executable = shutil.which(argv[0])
    if executable is None:
        return Completed(127, "", f"{argv[0]} is not installed")
    try:
        result = subprocess.run(  # noqa: S603 - argv list, no shell
            [executable, *argv[1:]],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        return Completed(124, out, "timed out", timed_out=True)
    return Completed(result.returncode, result.stdout, result.stderr)


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    exit_code: int
    duration_seconds: float
    output_tail: str
    timed_out: bool = False


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    def __init__(self, profile: SandboxProfile, *, run: CommandRunner = run_command):
        self.profile = profile
        self._run = run

    def image_tag(self, snapshot: Snapshot) -> str:
        return f"axiom-sandbox/{self.profile.sandbox_id}:{snapshot.short}"

    def dockerfile(self) -> str:
        setup = "\n".join(f"RUN {line}" for line in self.profile.setup)
        return (
            f"FROM {self.profile.base_image}\n"
            "ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_LINK_MODE=copy UV_NO_CACHE=1 \\\n"
            "    PYTHONDONTWRITEBYTECODE=1 PATH=/opt/venv/bin:$PATH\n"
            "WORKDIR /work\n"
            "COPY . /work\n"
            f"{setup}\n"
            f"RUN useradd --uid {SANDBOX_UID} --no-create-home sandbox\n"
        )

    def prepare(self, snapshot: Snapshot) -> str:
        """Build the environment image from the untouched snapshot (network allowed)."""
        tag = self.image_tag(snapshot)
        exists = self._run(["docker", "image", "inspect", tag], None, 60)
        if exists.returncode == 0:
            return tag
        built = self._run(
            ["docker", "build", "--quiet", "-t", tag, "-f", "-", str(snapshot.path)],
            self.dockerfile(),
            1800,
        )
        if built.returncode != 0:
            raise SandboxError(f"image build failed: {built.stderr[-OUTPUT_TAIL:]}")
        return tag

    def _isolated(self, snapshot: Snapshot, name: str, argv: Sequence[str]) -> list[str]:
        limits = self.profile.limits
        return [
            "docker", "run", "--rm", "--name", name,
            "--network", "none",
            "--read-only",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,size={limits.tmpfs_size}",  # noqa: S108 - path inside the container
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--pids-limit", str(limits.pids),
            "--memory", limits.memory,
            "--cpus", str(limits.cpus),
            "--user", f"{SANDBOX_UID}:{SANDBOX_UID}",
            "--env", "HOME=/tmp",
            "--volume", f"{snapshot.path}:/work:rw",
            "--workdir", "/work",
            self.image_tag(snapshot),
            *argv,
        ]  # fmt: skip

    def _execute(self, snapshot: Snapshot, label: str, argv: Sequence[str]) -> CheckResult:
        name = f"axiom-{self.profile.sandbox_id}-{label}-{secrets.token_hex(4)}"
        started = time.perf_counter()
        done = self._run(
            self._isolated(snapshot, name, argv), None, self.profile.limits.timeout_seconds
        )
        if done.timed_out:
            self._run(["docker", "rm", "--force", name], None, 60)
        output = (done.stdout + done.stderr)[-OUTPUT_TAIL:]
        return CheckResult(
            name=label,
            passed=done.returncode == 0 and not done.timed_out,
            exit_code=done.returncode,
            duration_seconds=time.perf_counter() - started,
            output_tail=output,
            timed_out=done.timed_out,
        )

    def run_check(self, snapshot: Snapshot, check: str) -> CheckResult:
        try:
            argv = self.profile.checks[check]
        except KeyError:
            raise SandboxError(f"check {check!r} is not in the allowlist") from None
        return self._execute(snapshot, check, argv)

    def run_all(self, snapshot: Snapshot) -> list[CheckResult]:
        return [self.run_check(snapshot, name) for name in self.profile.checks]

    def verify_isolation(self, snapshot: Snapshot) -> CheckResult:
        """Passes only if an outbound connection from inside the container fails."""
        return self._execute(snapshot, "network_probe", ["python", "-c", _NETWORK_PROBE])
