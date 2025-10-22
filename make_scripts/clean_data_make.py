#!/usr/bin/env python3
"""Makefile-friendly cleanup helper.

Deletes:
 - ETL_DIR/runs/
 - ETL_DIR/logs/
 - AI_DIR/snapshots/<SNAPSHOT_DATE>

Preserves:
 - ETL_DIR/exports/

Writes audit to provenance/cleanup_log_<UTC_COMPACT>.txt
"""
from __future__ import annotations
import os
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
import sys

def confirm(prompt: str) -> bool:
    try:
        resp = input(prompt + ' [y/N]: ')
    except EOFError:
        return False
    return resp.strip().lower() == 'y'


def main():
    parser = argparse.ArgumentParser(description='Safe cleanup helper for Makefile')
    parser.add_argument('--prune-extras', action='store_true', help='Remove other top-level entries under ETL_DIR (except exports,runs,logs and dotfiles)')
    parser.add_argument('--prune-list', type=str, default='', help='Comma-separated list of specific top-level names to prune (overrides auto-detect)')
    args = parser.parse_args()

    etl = Path(os.environ.get('ETL_DIR', '.'))
    ai = Path(os.environ.get('AI_DIR', '.'))
    snapdate = os.environ.get('SNAPSHOT_DATE', '')
    provenance = Path('provenance')
    provenance.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    audit = provenance / f'cleanup_log_{ts}.txt'

    removed = []
    preserved = []

    exports = etl / 'exports'
    preserved.append(str(exports))

    # Ensure exports subdirectories exist before any deletions (create if missing)
    exports_apple = exports / 'apple'
    exports_zepp = exports / 'zepp'
    try:
        exports.mkdir(parents=True, exist_ok=True)
        exports_apple.mkdir(parents=True, exist_ok=True)
        exports_zepp.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Failed creating exports dirs: {e}", file=sys.stderr)

    print(f"ETL_DIR={etl}")
    print(f"AI_DIR={ai}")
    print(f"Snapshot date={snapdate}")
    print(f"Exports (will be preserved): {exports}")
    print('')

    # First confirmation: clean ETL runs/logs
    if confirm(f"Proceed to remove '{etl / 'runs'}' and '{etl / 'logs'}' under {etl}? This WILL NOT touch '{exports}'."):
        for sub in ('runs', 'logs'):
            p = etl / sub
            if p.exists():
                try:
                    shutil.rmtree(p)
                    removed.append(str(p))
                    print(f"Removed: {p}")
                except Exception as e:
                    print(f"Failed removing {p}: {e}")
            else:
                print(f"Not present: {p}")
    else:
        print('Skipping ETL runs/logs cleanup.')

    # Second confirmation: clean AI snapshot
    snap_path = ai / 'snapshots' / snapdate
    if confirm(f"Proceed to remove AI snapshot path '{snap_path}'?"):
        if snap_path.exists():
            try:
                shutil.rmtree(snap_path)
                removed.append(str(snap_path))
                print(f"Removed: {snap_path}")
            except Exception as e:
                print(f"Failed removing {snap_path}: {e}")
        else:
            print(f"Snapshot path not present: {snap_path}")
    else:
        print('Skipping AI snapshot cleanup.')

    # Optional pruning of other top-level ETL entries
    if args.prune_extras or args.prune_list:
        preserve_names = {'exports', 'runs', 'logs'}
        explicit = [n.strip() for n in args.prune_list.split(',') if n.strip()]
        if explicit:
            candidates = [etl / name for name in explicit]
        else:
            # Auto-detect: all non-dot, top-level entries except preserved names
            candidates = [p for p in etl.iterdir() if not p.name.startswith('.') and p.name not in preserve_names]

        if candidates:
            print('\nPrune candidates under ETL_DIR:')
            for c in candidates:
                print(' -', c)
            if confirm('Proceed to prune the listed items? This will permanently remove them.'):
                for p in candidates:
                    if p.exists():
                        try:
                            if p.is_dir():
                                shutil.rmtree(p)
                            else:
                                p.unlink()
                            removed.append(str(p))
                            print(f"Pruned: {p}")
                        except Exception as e:
                            print(f"Failed pruning {p}: {e}")
                    else:
                        print(f"Not present: {p}")
            else:
                print('Skipping pruning of extras.')
        else:
            print('\nNo prune candidates found under ETL_DIR.')

    # Ensure exports preserved
    if exports.exists():
        print(f"Preserved: {exports}")
    else:
        print(f"Exports path not present (preserved by rule): {exports}")

    summary = {
        'cleanup_run': ts,
        'etl_dir': str(etl),
        'ai_dir': str(ai),
        'snapshot_date': snapdate,
        'removed': removed,
        'preserved': preserved,
    }
    # Report exports status: existence flags and recursive regular-file counts
    exists_exports = exports.exists()
    exists_apple = exports_apple.exists()
    exists_zepp = exports_zepp.exists()
    files_apple = sum(1 for p in exports_apple.rglob('*') if p.is_file()) if exists_apple else 0
    files_zepp = sum(1 for p in exports_zepp.rglob('*') if p.is_file()) if exists_zepp else 0

    exports_status = {
        'exists_exports': bool(exists_exports),
        'exists_apple': bool(exists_apple),
        'exists_zepp': bool(exists_zepp),
        'files_apple': int(files_apple),
        'files_zepp': int(files_zepp),
    }

    # Add to summary and print a short summary line
    summary['exports_status'] = exports_status
    print(f"Preserved exports: {exports} (apple files={files_apple}, zepp files={files_zepp})")
    try:
        with audit.open('w', encoding='utf-8') as fh:
            fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        print('\nSummary written to:', audit)
    except Exception as e:
        print(f"Failed writing audit file {audit}: {e}", file=sys.stderr)
        sys.exit(2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
