#!/usr/bin/env python3
"""Simulate a partial atomic write by creating a temp file under the target dir.

Usage: make_scripts/simulate_partial_write.py --dir data/.../normalized/apple

Creates a file named like .tmp.<pid>.<ts> to simulate an interrupted write.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True)
    args = p.parse_args(argv)
    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    ts = int(time.time())
    tmp = d / f'.tmp.{pid}.{ts}'
    tmp.write_text('PARTIAL', encoding='utf-8')
    print(f'Created simulated partial write: {tmp}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
