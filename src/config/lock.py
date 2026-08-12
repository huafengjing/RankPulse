from __future__ import annotations

import os
import sys
import time
from pathlib import Path


class FileLock:
    """Process-level mutual exclusion lock via atomic directory creation.

    Uses os.mkdir which is atomic on all platforms (Windows, macOS, Linux).
    Blocks for up to *timeout* seconds if another instance holds the lock,
    then raises RuntimeError.

    Usage:

        with FileLock("data/.lock"):
            ...  # critical section

    The lock is automatically released when the ``with`` block exits.
    If a process crashes without releasing, the lock directory persists
    and must be cleaned up manually.
    """

    def __init__(
        self,
        lock_path: str | Path,
        *,
        timeout: float = 120.0,
        poll_interval: float = 1.0,
    ) -> None:
        self.lock_path = Path(lock_path)
        self.timeout = timeout
        self.poll_interval = poll_interval

    def __enter__(self) -> None:
        start = time.monotonic()
        while True:
            try:
                self.lock_path.mkdir(parents=False, exist_ok=False)
                return
            except FileExistsError:
                elapsed = time.monotonic() - start
                if elapsed > self.timeout:
                    raise RuntimeError(
                        f"Could not acquire lock at {self.lock_path} "
                        f"after {elapsed:.0f}s. Another instance may still "
                        "be running, or the previous run crashed without "
                        "cleaning up. Delete the directory manually if the "
                        "previous process is definitely dead."
                    ) from None
                time.sleep(self.poll_interval)

    def __exit__(self, *args: object) -> None:
        try:
            self.lock_path.rmdir()
        except FileNotFoundError:
            pass
