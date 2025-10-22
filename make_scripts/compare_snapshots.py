#!/usr/bin/env python3
"""Compare two JSON hash snapshot files and exit 0 if identical, non-zero otherwise.

Usage: make_scripts/compare_snapshots.py --before provenance/hash_snapshot_before.json --after provenance/hash_snapshot_after.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--before', required=True)
    p.add_argument('--after', required=True)
    args = p.parse_args(argv)

    b = json.load(open(args.before, 'r', encoding='utf-8'))
    a = json.load(open(args.after, 'r', encoding='utf-8'))
    if b == a:
        print('IDEMPOTENT ✅')
        return 0
    print('NOT IDEMPOTENT: differences found')
    bkeys = set(b.keys())
    akeys = set(a.keys())
    added = akeys - bkeys
    removed = bkeys - akeys
    changed = [k for k in (bkeys & akeys) if b.get(k) != a.get(k)]
    if added:
        print('added:', list(added)[:10])
    if removed:
        print('removed:', list(removed)[:10])
    if changed:
        print('changed:', list(changed)[:10])
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
