#!/usr/bin/env python3
"""Create a snapshot of SHA256 for matching artifacts under normalized/processed/joined.

Usage: make_scripts/hash_snapshot.py --out provenance/hash_snapshot.json [--roots normalized processed joined]

This script scans the given roots under data/etl/<PID> and records sha256 for files with extensions .csv and .manifest.json.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from typing import List


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--pid', default='P000001')
    p.add_argument('--roots', nargs='+', default=['normalized', 'processed', 'joined'])
    p.add_argument('--out', required=True)
    args = p.parse_args(argv)

    base = Path('data/etl') / args.pid
    out = {}
    for r in args.roots:
        root = base / r
        if not root.exists():
            continue
        for f in sorted(root.rglob('*')):
            if f.is_file() and (f.suffix in ('.csv',) or f.name.endswith('.manifest.json') or f.suffix in ('.ok',)):
                try:
                    out[str(f)] = sha256_of_file(f)
                except Exception as e:
                    out[str(f)] = None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f'Wrote snapshot {args.out} with {len(out)} entries')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
