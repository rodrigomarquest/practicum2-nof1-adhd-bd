#!/usr/bin/env python3
"""Migrate existing etl/<PID> layout to the snapshots layout.

Usage: python make_scripts/migrate_to_snapshot_layout.py P000001 2025-09-29

This will create data/etl/<PID>/snapshots/<snapshot>/ and move the common
subdirs (extracted, normalized, processed, joined) into that snapshot if they
exist at the top-level under data/etl/<PID> and are not already under snapshots.
It will then write data/etl/<PID>/latest.txt with the snapshot id.

The operations are conservative: if a target already exists it won't overwrite
existing content and will instead print a warning. No deletions are performed.
"""
from pathlib import Path
import shutil
import sys
import re

ISO_RE = re.compile(r"^20\d{2}-[01]\d-[0-3]\d$")

MOVE_DIRS = ['extracted', 'normalized', 'processed', 'joined']


def migrate(pid: str, snapshot: str, root: Path = Path('data/etl')):
    if not ISO_RE.match(snapshot):
        raise SystemExit(f'invalid snapshot id {snapshot}')
    base = root / pid
    snapdir = base / 'snapshots' / snapshot
    snapdir.mkdir(parents=True, exist_ok=True)

    for d in MOVE_DIRS:
        src = base / d
        dst = snapdir / d
        if src.exists() and not dst.exists():
            print(f'Moving {src} -> {dst}')
            shutil.move(str(src), str(dst))
        elif src.exists() and dst.exists():
            print(f'SKIP {d}: target already exists at {dst}')
        else:
            print(f'No {src} to move')

    # write latest.txt
    latest_file = base / 'latest.txt'
    with open(latest_file, 'w', encoding='utf-8') as f:
        f.write(snapshot + '\n')
    print(f'Wrote {latest_file}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: migrate_to_snapshot_layout.py PID YYYY-MM-DD')
        raise SystemExit(2)
    migrate(sys.argv[1], sys.argv[2])
