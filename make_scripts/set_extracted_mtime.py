#!/usr/bin/env python3
"""Set extracted export.xml mtime to match canonical raw zip timestamp.

Usage: python make_scripts/set_extracted_mtime.py P000001

Find the latest canonical zip under data/raw/<pid>/apple/apple_health_export_*.zip,
extract the timestamp, and set mtime of data/etl/<pid>/snapshots/<snapshot>/extracted/apple/export.xml
to that timestamp (UTC).
"""
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

CANON_RE = re.compile(r'apple_health_export_(\d{8}T\d{6}Z)')


def find_latest_canonical(pid: str):
    raw_apple = Path('data') / 'raw' / pid / 'apple'
    if not raw_apple.exists():
        return None
    candidates = list(raw_apple.glob('apple_health_export_*.zip'))
    if not candidates:
        return None
    candidates.sort()
    sel = candidates[-1]
    m = CANON_RE.search(sel.name)
    if not m:
        return None
    ts = m.group(1)
    # parse like 20251022T061854Z
    dt = datetime.strptime(ts, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    return dt


def set_mtime_for_pid(pid: str):
    dt = find_latest_canonical(pid)
    if not dt:
        print('No canonical zip found')
        return 1
    snap = dt.strftime('%Y-%m-%d')
    target = Path('data') / 'etl' / pid / 'snapshots' / snap / 'extracted' / 'apple' / 'export.xml'
    if not target.exists():
        print('Target export.xml not found:', target)
        return 2
    secs = dt.timestamp()
    try:
        import os
        os.utime(target, (secs, secs))
        print('Set mtime for', target, '->', dt.isoformat())
        return 0
    except Exception as e:
        print('Failed to set mtime:', e)
        return 3


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: set_extracted_mtime.py PID')
        sys.exit(2)
    sys.exit(set_mtime_for_pid(sys.argv[1]))
