"""The file lock picks its backend at import time and never imports the other platform's module."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from typing import Any

import pytest

from axiom_council import filelock

_SOURCE = Path(filelock.__file__)


def _load(name: str) -> Any:
    """Execute the helper's source again as a new module, under the current sys.platform."""
    spec = importlib.util.spec_from_file_location(name, _SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMsvcrt(types.ModuleType):
    LK_UNLCK, LK_LOCK, LK_NLCK, LK_RLCK, LK_NBLCK = 0, 1, 2, 3, 4

    def __init__(self, busy: int = 0) -> None:
        super().__init__("msvcrt")
        self.calls: list[tuple[int, int]] = []
        self.busy = busy

    def locking(self, fd: int, mode: int, nbytes: int) -> None:
        self.calls.append((mode, nbytes))
        if mode == self.LK_NBLCK and self.busy:
            self.busy -= 1
            raise OSError("locked by another process")


def test_posix_uses_flock() -> None:
    if sys.platform == "win32":
        pytest.skip("POSIX only")
    assert filelock.BACKEND == "fcntl"


def test_windows_uses_msvcrt_and_waits_for_the_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakeMsvcrt(busy=2)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    monkeypatch.setitem(sys.modules, "fcntl", None)  # `import fcntl` now raises ImportError
    module = _load("filelock_windows_probe")
    assert module.BACKEND == "msvcrt"
    with module.locked(tmp_path / "x.lock", exclusive=False):
        assert fake.calls == [(fake.LK_NBLCK, 1)] * 3
    assert fake.calls[-1] == (fake.LK_UNLCK, 1)


_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from axiom_council.filelock import locked
    with locked(Path(sys.argv[1]), exclusive=True):
        print("held", flush=True)
        time.sleep(1.0)
    """
)


def test_exclusive_lock_blocks_a_second_process(tmp_path: Path) -> None:
    import time

    path = tmp_path / "x.lock"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(path)], stdout=subprocess.PIPE, text=True
    )
    assert holder.stdout is not None and holder.stdout.readline().strip() == "held"
    start = time.monotonic()
    with filelock.locked(path, exclusive=True):
        waited = time.monotonic() - start
    assert holder.wait(timeout=30) == 0
    assert waited > 0.5


def test_fsync_dir_skips_directories_on_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from axiom_council import audit_anchor

    def refuse(*_: object) -> int:
        raise PermissionError("a directory cannot be opened on Windows")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(audit_anchor.os, "open", refuse)
    audit_anchor.fsync_dir(tmp_path)  # must not open the directory
