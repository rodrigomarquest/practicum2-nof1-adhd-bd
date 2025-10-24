#!/usr/bin/env python3
"""Write the latest snapshot id into data/etl/<PID>/latest.txt

Usage: python make_scripts/update_latest_snapshot.py P000001

It will look under data/etl/<PID>/snapshots/ and pick the latest directory by
ISO name (YYYY-MM-DD). If none found, it will choose the most recently
modified child directory.
"""
import sys
from pathlib import Path
import re

ISO_RE = re.compile(r"^20\d{2}-[01]\d-[0-3]\d$")


def find_latest_snapshot(pid: str, root: Path = Path('data/etl')) -> str | None:
    base = root / pid / 'snapshots'
    if not base.exists():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    # Prefer ISO formatted names
    iso_dirs = [p for p in candidates if ISO_RE.match(p.name)]
    if iso_dirs:
        iso_sorted = sorted(iso_dirs, key=lambda p: p.name)
        return iso_sorted[-1].name
    # fallback: choose most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1].name


def write_latest(pid: str, latest: str, root: Path = Path('data/etl')) -> Path:
    path = root / pid / 'latest.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(latest + '\n')
    return path


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print('Usage: update_latest_snapshot.py PID')
        raise SystemExit(2)
    pid = argv[0]
    latest = find_latest_snapshot(pid)
    if latest is None:
        print(f'No snapshots found for {pid}')
        raise SystemExit(1)
    out = write_latest(pid, latest)
    print(f'Wrote {out} -> {latest}')

if __name__ == '__main__':
    main()
