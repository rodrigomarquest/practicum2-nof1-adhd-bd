#!/usr/bin/env python3
"""Snapshot lock implementation (moved into make_scripts.utils).

This file hosts the full SnapshotLock implementation so other modules can
import it from ``make_scripts.utils.snapshot_lock`` as the canonical location.
The top-level ``make_scripts.snapshot_lock`` will be a small shim that
delegates to this module for backward compatibility.
"""
from __future__ import annotations
import os
import time
import atexit
import json
from pathlib import Path
from typing import Optional


class SnapshotLockError(RuntimeError):
    pass


class SnapshotLock:
    """Simple cross-platform lockfile for a snapshot stage.

    Creates a file at <snapshot_root>/.lock.<stage>.<participant>.<snapshot>
    with JSON content {pid, ts, cmd} and removes it on context exit.
    If the lock exists and is recent (mtime <= timeout), acquisition fails.
    If it's stale (mtime > timeout) acquisition fails unless force=True.
    """

    def __init__(self, snapshot_root: Path, stage: str, participant: str, snapshot: str, timeout_sec: int = 3600, force: bool = False):
        self.snapshot_root = Path(snapshot_root)
        self.stage = stage
        self.participant = participant
        self.snapshot = snapshot
        self.timeout_sec = int(timeout_sec)
        self.force = bool(force)
        self.lock_path = self.snapshot_root / f".lock.{self.stage}.{self.participant}.{self.snapshot}"
        self.acquired = False

    def _write_lock(self):
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        content = {
            'pid': os.getpid(),
            'ts': time.time(),
            'stage': self.stage,
        }
        tmp = self.lock_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(content, fh)
        os.replace(tmp, self.lock_path)

    def _remove_lock(self):
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    def acquire(self):
        # check existing
        if self.lock_path.exists():
            try:
                mtime = self.lock_path.stat().st_mtime
                age = time.time() - mtime
            except Exception:
                age = float('inf')
            if age <= self.timeout_sec:
                raise SnapshotLockError(f"Lock exists for stage {self.stage} at {self.lock_path} (age {int(age)}s) - another process may be running. Use --force-lock to override if appropriate.")
            else:
                # stale
                if not self.force:
                    raise SnapshotLockError(f"Stale lock exists for stage {self.stage} at {self.lock_path} (age {int(age)}s). Use --force-lock to override.")
                # allow override: remove and proceed
                try:
                    self.lock_path.unlink()
                except Exception:
                    pass

        # create lock
        self._write_lock()
        # register removal
        atexit.register(self._remove_lock)
        self.acquired = True
        return self

    def release(self):
        if self.acquired:
            self._remove_lock()
            self.acquired = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc, tb):
        # always remove lock on exit (normal or exception)
        self.release()
        return False
