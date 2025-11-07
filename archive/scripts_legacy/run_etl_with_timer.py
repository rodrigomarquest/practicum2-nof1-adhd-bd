#!/usr/bin/env python3
"""Backward-compatible shim for the legacy scripts/ CLI.

This file remains to avoid breaking callers that still run
``python scripts/run_etl_with_timer.py``. It forwards execution to
the new `cli.run_etl_with_timer` module and prints a brief deprecation
notice.
"""
from __future__ import annotations
import runpy
import sys


def main() -> int:
    print("[DEPRECATION] Use `python -m cli.run_etl_with_timer` instead of scripts/run_etl_with_timer.py")
    runpy.run_module("cli.run_etl_with_timer", run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
