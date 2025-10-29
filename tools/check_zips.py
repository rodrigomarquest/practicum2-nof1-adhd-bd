#!/usr/bin/env python3
"""Check for zip files under data/raw/<participant> and exit non-zero if none.

This small helper avoids complex quoting in Makefile - it's safe and cross-platform.
"""
import sys
import os
from pathlib import Path


def main(argv):
    if len(argv) < 2:
        print("usage: check_zips.py <participant>")
        return 2
    participant = argv[1]
    p = Path("data") / "raw" / participant
    files = list(p.rglob("*.zip")) if p.exists() else []
    if not files:
        print(f"ERROR: no zip files found under {p}")
        print(f"PWD={os.getcwd()}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
