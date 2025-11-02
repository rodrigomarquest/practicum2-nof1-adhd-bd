#!/usr/bin/env python3
"""Deprecated shim for migrate_snapshots script.

This is a temporary wrapper that delegates to the new module location
`cli.migrate_snapshots`. Use `python -m cli.migrate_snapshots` instead.
"""
from __future__ import annotations
import runpy
import sys


def main() -> int:
    print("[DEPRECATION] Use `python -m cli.migrate_snapshots` instead of scripts/migrate_snapshots.py")
    runpy.run_module("cli.migrate_snapshots", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
