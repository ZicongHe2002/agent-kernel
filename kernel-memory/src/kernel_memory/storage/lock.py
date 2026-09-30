"""Single cross-process locking protocol for mutation, recovery, and consistent reads.

P0 supports one machine and one coordinating writer. The lock uses ``fcntl.flock``
on POSIX and ``msvcrt.locking`` on Windows; it is re-entrant within one store
instance and thread-safe. Network filesystems are explicitly unsupported.
"""
from __future__ import annotations

import errno
import os
import threading
import time
from pathlib import Path

from ..domain.errors import ExecutionInfrastructureError

if os.name == "nt":  # pragma: no cover - exercised on Windows
    import msvcrt
else:  # pragma: no cover - the active branch is covered on POSIX
    import fcntl


def _ensure_lock_byte(fd: int) -> None:
    """Ensure the Windows lock file has a byte that can be locked.

    ``msvcrt.locking`` locks a byte at the current file position and fails on an
    empty file. The extra byte is runtime state and has no semantic content.
    """
    if os.name != "nt":
        return
    if os.fstat(fd).st_size == 0:
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, b"\0")
        os.fsync(fd)


def _lock_nonblocking(fd: int) -> None:
    if os.name == "nt":
        _ensure_lock_byte(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _is_busy_lock_error(exc: OSError) -> bool:
    """Return whether an OS error means another process currently owns the lock."""
    return exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK} or getattr(exc, "winerror", None) in {33, 36}


class StoreLock:
    def __init__(self, path: Path, *, timeout_seconds: float = 30.0, poll_interval: float = 0.02) -> None:
        self.path = Path(path)
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self._thread_lock = threading.RLock()
        self._fd: int | None = None
        self._depth = 0

    def acquire(self) -> None:
        self._thread_lock.acquire()
        if self._depth > 0:
            self._depth += 1
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                _lock_nonblocking(fd)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    self._thread_lock.release()
                    raise ExecutionInfrastructureError(
                        f"could not acquire store lock {self.path} within {self.timeout_seconds}s",
                        code="LOCK_TIMEOUT",
                    )
                time.sleep(self.poll_interval)
            except OSError as exc:
                if not _is_busy_lock_error(exc):
                    os.close(fd)
                    self._thread_lock.release()
                    raise ExecutionInfrastructureError(
                        f"could not acquire store lock {self.path}: {exc}",
                        code="LOCK_ERROR",
                    ) from exc
                if time.monotonic() >= deadline:
                    os.close(fd)
                    self._thread_lock.release()
                    raise ExecutionInfrastructureError(
                        f"could not acquire store lock {self.path} within {self.timeout_seconds}s",
                        code="LOCK_TIMEOUT",
                    ) from exc
                time.sleep(self.poll_interval)
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._depth == 0:
            raise RuntimeError("lock released more times than acquired")
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            try:
                _unlock(self._fd)
            finally:
                os.close(self._fd)
                self._fd = None
        self._thread_lock.release()

    def __enter__(self) -> "StoreLock":
        self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    @property
    def held(self) -> bool:
        return self._depth > 0
