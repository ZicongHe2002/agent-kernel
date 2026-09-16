"""Single cross-process locking protocol for mutation, recovery, and consistent reads.

P0 supports one machine and one coordinating writer. The lock is an exclusive
``fcntl.flock`` on ``.runtime/lock``; it is re-entrant within one store instance
and thread-safe. Network filesystems are explicitly unsupported.
"""
from __future__ import annotations

import fcntl
import os
import threading
import time
from pathlib import Path

from ..domain.errors import ExecutionInfrastructureError


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
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        self._fd = fd
        self._depth = 1

    def release(self) -> None:
        if self._depth == 0:
            raise RuntimeError("lock released more times than acquired")
        self._depth -= 1
        if self._depth == 0 and self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
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
