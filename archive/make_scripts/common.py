"""Small helpers used by make_scripts/* entrypoints.

They accept PID/SNAP from either CLI args or environment variables so Makefile
can invoke them simply as: python make_scripts/foo.py PID=... SNAP=...
"""
from __future__ import annotations
import argparse
import os
from typing import Tuple


def parse_pid_snap(argv: list[str] | None = None) -> Tuple[str, str, str]:
    """Parse pid, snapshot and optional policy from argv or environment.

    Returns (pid, snap, policy)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", "-p", dest="pid", default=None)
    parser.add_argument("--pid", dest="pid2", default=None)
    parser.add_argument("--snapshot", "-s", dest="snap", default=None)
    parser.add_argument("--snap", dest="snap2", default=None)
    parser.add_argument("--policy", dest="policy", default=None)
    ns = parser.parse_args(argv)
    pid = ns.pid or ns.pid2 or os.environ.get("PID") or os.environ.get("participant")
    snap = ns.snap or ns.snap2 or os.environ.get("SNAP") or os.environ.get("snapshot")
    policy = ns.policy or os.environ.get("POLICY", "best_of_day")
    if not pid:
        raise SystemExit("Missing PID (pass --participant or set PID env)")
    if not snap:
        raise SystemExit("Missing SNAP (pass --snapshot or set SNAP env)")
    return pid, snap, policy


def env_override(name: str, default: str | None = None) -> str | None:
    """Return the value from environment if set, else default."""
    return os.environ.get(name, default)


__all__ = ["parse_pid_snap", "env_override"]
