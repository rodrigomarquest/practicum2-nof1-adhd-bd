#!/usr/bin/env python3
"""Align data/etl/<PID>/snapshots with data/raw/<PID>/apple canonical zip date.

Behavior for PID:
- Inspect data/raw/<PID>/apple/apple_health_export_*.zip and pick the latest canonical by filename timestamp.
- Compute snapshot date YYYY-MM-DD from the canonical filename.
- Move or rename any existing snapshot directories (if a single snapshot exists and differs) into the canonical snapshot date.
- Update data/etl/<PID>/latest.txt accordingly.

This is conservative: it will not overwrite existing target snapshot directory; instead it will move individual subdirs when needed.
"""
from pathlib import Path
import re
import shutil
import sys

CANON_RE = re.compile(r'apple_health_export_(\d{8}T\d{6}Z)')


def find_latest_canonical_date(pid: str) -> str | None:
    raw_apple = Path('data') / 'raw' / pid / 'apple'
    if not raw_apple.exists():
        return None
    candidates = list(raw_apple.glob('apple_health_export_*.zip'))
    if not candidates:
        return None
    # pick latest by filename timestamp
    def key(p: Path):
        m = CANON_RE.search(p.name)
        if m:
            return m.group(1)
        return ''
    candidates.sort(key=key)
    sel = candidates[-1]
    m = CANON_RE.search(sel.name)
    if not m:
        return None
    ts = m.group(1)
    return f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"


def align(pid: str):
    snap_date = find_latest_canonical_date(pid)
    if not snap_date:
        print('No canonical raw zip found for', pid)
        return 1
    base = Path('data') / 'etl' / pid / 'snapshots'
    if not base.exists():
        print('No snapshots dir to align; creating', base / snap_date)
        (base / snap_date).mkdir(parents=True, exist_ok=True)
        (Path('data') / 'etl' / pid / 'latest.txt').write_text(snap_date + '\n')
        return 0
    existing = [p for p in base.iterdir() if p.is_dir()]
    if not existing:
        print('No existing snapshot subdirs; creating', snap_date)
        (base / snap_date).mkdir(parents=True, exist_ok=True)
        (Path('data') / 'etl' / pid / 'latest.txt').write_text(snap_date + '\n')
        return 0
    # If there's already a directory for snap_date, just set latest
    target = base / snap_date
    if target.exists():
        (Path('data') / 'etl' / pid / 'latest.txt').write_text(snap_date + '\n')
        print('Snapshot directory already correct; updated latest.txt')
        return 0
    # If only one existing snapshot and it's different, rename/move it
    if len(existing) == 1:
        src = existing[0]
        print(f'Renaming {src} -> {target}')
        try:
            shutil.move(str(src), str(target))
        except Exception:
            # fallback to moving children
            for child in src.iterdir():
                dst = target / child.name
                if not dst.exists():
                    shutil.move(str(child), str(dst))
            # attempt to remove src if empty
            try:
                src.rmdir()
            except Exception:
                pass
        (Path('data') / 'etl' / pid / 'latest.txt').write_text(snap_date + '\n')
        print('Updated latest.txt ->', snap_date)
        return 0
    # Multiple existing snapshot dirs: do not try to auto-merge
    print('Multiple snapshot dirs present; manual reconciliation needed:')
    for p in existing:
        print(' -', p)
    return 2


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: align_snapshot_with_raw.py PID')
        raise SystemExit(2)
    sys.exit(align(sys.argv[1]))
