"""Advisory file locks shared between processes: `fcntl` on POSIX, `msvcrt` on Windows.

The backend is chosen once, at import time, from `sys.platform`. The module imports only the
backend of the current platform, so it imports on both platforms.

Windows notes:

- `msvcrt.locking` has no shared mode. A shared request takes an exclusive lock there, so
  readers wait for each other. This is slower but still correct.
- The lock covers the first byte of the lock file. The wait polls a non-blocking lock, because
  the blocking mode gives up after about ten seconds.
- The Windows branch has no test on a real Windows host yet. Tests on POSIX load it with a fake
  `msvcrt` module.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

_POLL_SECONDS = 0.05

if sys.platform == "win32":
    import msvcrt

    BACKEND = "msvcrt"

    def _acquire(fd: int, exclusive: bool) -> None:
        del exclusive  # msvcrt has no shared lock
        while True:
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                time.sleep(_POLL_SECONDS)

    def _release(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    BACKEND = "fcntl"

    def _acquire(fd: int, exclusive: bool) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def locked(path: Path, *, exclusive: bool) -> Iterator[None]:
    """Hold a lock on path (created if missing) for the body of the with-block."""
    with path.open("a+b") as fh:
        _acquire(fh.fileno(), exclusive)
        try:
            yield
        finally:
            _release(fh.fileno())
