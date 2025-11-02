#!/usr/bin/env python3
"""Backward-compatible shim for the legacy scripts/ CLI.

This file remains temporarily to avoid breaking callers that still run
``python scripts/etl_runner.py``. It forwards the call to the new
module entrypoint `cli.etl_runner` and prints a short deprecation note.
"""
from __future__ import annotations
import sys
import runpy


def main() -> int:
    print("[DEPRECATION] Use `python -m cli.etl_runner` instead of scripts/etl_runner.py")
    # Execute the module entrypoint so behavior is identical to new location.
    # We import and run the module as __main__ to preserve argv handling.
    return runpy.run_module("cli.etl_runner", run_name="__main__") or 0


if __name__ == "__main__":
    raise SystemExit(main())
