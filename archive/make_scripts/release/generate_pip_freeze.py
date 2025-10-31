#!/usr/bin/env python3
"""Generate a pip freeze file under provenance/ (idempotent).

Usage:
  python make_scripts/release/generate_pip_freeze.py [--out-dir provenance]
"""
from __future__ import annotations
import argparse
import datetime
import os
import subprocess
import sys


def generate_pip_freeze(out_dir: str = "provenance") -> int:
    ts = datetime.date.today().isoformat()
    fname = os.path.join(out_dir, f"pip_freeze_{ts}.txt")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(fname):
        print(f"[provenance] reuse existing freeze -> {fname}")
        return 0
    try:
        with open(fname, "w", encoding="utf-8") as f:
            # Use the same python interpreter used to call the script for reproducibility
            py = sys.executable or "python"
            subprocess.run([py, "-m", "pip", "freeze"], stdout=f, check=True)
        print(f"[provenance] pip freeze written -> {fname}")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: pip freeze failed: {e}", file=sys.stderr)
        return 2


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="provenance")
    args = p.parse_args(argv)
    return generate_pip_freeze(args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
