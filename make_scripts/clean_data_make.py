#!/usr/bin/env python3
"""Makefile-friendly cleanup helper.

Stage-based cleaner that supports these stages:
 - raw: data/raw/<participant> (dangerous; must pass --i-understand-raw to actually delete)
 - extracted: ETL_DIR/extracted
 - normalized: ETL_DIR/normalized
 - processed: ETL_DIR/processed
 - joined: ETL_DIR/joined
 - ai_snapshot: AI_DIR/snapshots/<SNAPSHOT_DATE>

Default behavior (when invoked for a stage) is to remove the corresponding directory
unless DRY_RUN is set. The script writes an audit to provenance/cleanup_log_<UTC>.txt
with a 'stage' field and a list of removed entries.
"""
from __future__ import annotations
import os
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List
import sys


def confirm(prompt: str) -> bool:
    try:
        resp = input(prompt + ' [y/N]: ')
    except EOFError:
        return False
    return resp.strip().lower() == 'y'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage-based cleanup helper for Makefile')
    parser.add_argument('--stage', choices=['raw', 'extracted', 'normalized', 'processed', 'joined', 'ai_snapshot'], required=True, help='Which stage to clean')
    parser.add_argument('--i-understand-raw', action='store_true', help='Required to actually delete raw data')
    parser.add_argument('--prune-extras', action='store_true', help='Remove other top-level entries under ETL_DIR (except preserved names)')
    parser.add_argument('--prune-list', type=str, default='', help='Comma-separated list of specific top-level names to prune (overrides auto-detect)')
    return parser.parse_args()


def load_env():
    etl = Path(os.environ.get('ETL_DIR', 'data/etl'))
    raw = Path(os.environ.get('RAW_DIR', 'data/raw'))
    ai = Path(os.environ.get('AI_DIR', 'data/ai'))
    snapdate = os.environ.get('SNAPSHOT_DATE', '')
    dry_run = os.environ.get('DRY_RUN', '0') not in ('0', '', 'false', 'False') or os.environ.get('DRY_RUN') == '1'
    non_interactive = os.environ.get('NON_INTERACTIVE', '') not in ('', '0', 'false', 'False')

    provenance = Path('provenance')
    provenance.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit = provenance / f'cleanup_log_{ts}.txt'
    removed: List[str] = []

    return {
        'etl': etl,
        'raw': raw,
        'ai': ai,
        'snapdate': snapdate,
        'dry_run': dry_run,
        'non_interactive': non_interactive,
        'provenance': provenance,
        'ts': ts,
        'audit': audit,
        'removed': removed,
    }


def do_remove(path: Path, removed: List[str], dry_run: bool):
    if not path.exists():
        print(f"Not present: {path}")
        return
    try:
        if dry_run:
            print(f"DRY RUN - would remove: {path}")
        else:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
            print(f"Removed: {path}")
    except Exception as e:
        print(f"Failed removing {path}: {e}")


def choose_ai_snapshot(ai: Path, snapdate: str, non_interactive: bool) -> str | None:
    if snapdate:
        return snapdate
    snaps_dir = ai / 'snapshots'
    if non_interactive:
        print('NON_INTERACTIVE set and SNAPSHOT_DATE empty: skipping AI snapshot deletion.')
        return None
    if snaps_dir.exists() and snaps_dir.is_dir():
        candidates = [p.name for p in snaps_dir.iterdir() if p.is_dir()]
        if candidates:
            print('\nExisting snapshot folders under', snaps_dir)
            for name in sorted(candidates):
                print(' -', name)
            typed = input('Type exact snapshot folder name to delete (leave blank to skip): ').strip()
            return typed if typed else None
        print(f'No snapshot folders found under {snaps_dir}; skipping.')
        return None
    print(f'No snapshots directory at {snaps_dir}; skipping AI snapshot deletion.')
    return None


def prune_extras(etl: Path, args: argparse.Namespace, dry_run: bool, removed: List[str]):
    preserve_names = {'raw', 'extracted', 'normalized', 'processed', 'joined', 'logs'}
    explicit = [n.strip() for n in args.prune_list.split(',') if n.strip()]
    if explicit:
        candidates = [etl / name for name in explicit]
    else:
        candidates = [p for p in etl.iterdir() if not p.name.startswith('.') and p.name not in preserve_names]

    if candidates:
        print('\nPrune candidates under ETL_DIR:')
        for c in candidates:
            print(' -', c)
        if confirm('Proceed to prune the listed items? This will permanently remove them.'):
            for p in candidates:
                do_remove(p, removed, dry_run)
        else:
            print('Skipping pruning of extras.')
    else:
        print('\nNo prune candidates found under ETL_DIR.')


def handle_stage(stage: str, args: argparse.Namespace, cfg: dict):
    etl: Path = cfg['etl']
    raw: Path = cfg['raw']
    ai: Path = cfg['ai']
    snapdate: str = cfg['snapdate']
    dry_run: bool = cfg['dry_run']
    non_interactive: bool = cfg['non_interactive']
    removed: List[str] = cfg['removed']

    print(f"Stage={stage}")
    print(f"ETL_DIR={etl}")
    print(f"RAW_DIR={raw}")
    print(f"AI_DIR={ai}")
    print(f"SNAPSHOT_DATE={snapdate}")
    print('')

    if stage == 'raw':
        target = raw
        print('WARNING: raw data is precious. By default nothing will be removed.')
        if not args.i_understand_raw:
            print("To actually delete raw data pass the flag: --i-understand-raw")
            return
        if confirm(f"Proceed to remove RAW_DIR '{target}'? This is destructive and irreversible."):
            do_remove(target, removed, dry_run)
        else:
            print('Skipping raw cleanup.')
        return

    if stage in ('extracted', 'normalized', 'processed', 'joined'):
        target = etl / stage
        if confirm(f"Proceed to remove {stage} data at '{target}'?") or non_interactive:
            do_remove(target, removed, dry_run)
        else:
            print(f'Skipping {stage} cleanup.')
        return

    if stage == 'ai_snapshot':
        snapshot_target = choose_ai_snapshot(ai, snapdate, non_interactive)
        if snapshot_target:
            snap_path = ai / 'snapshots' / snapshot_target
            if confirm(f"Proceed to remove AI snapshot path '{snap_path}'?") or non_interactive:
                do_remove(snap_path, removed, dry_run)
            else:
                print('Skipping AI snapshot cleanup.')
        else:
            print('Skipping AI snapshot cleanup.')


def write_audit(cfg: dict, stage: str):
    summary = {
        'cleanup_run': cfg['ts'],
        'stage': stage,
        'etl_dir': str(cfg['etl']),
        'raw_dir': str(cfg['raw']),
        'ai_dir': str(cfg['ai']),
        'snapshot_date': cfg['snapdate'],
        'removed': cfg['removed'],
    }
    try:
        with cfg['audit'].open('w', encoding='utf-8') as fh:
            fh.write(json.dumps(summary, indent=2, ensure_ascii=False))
        print('\nSummary written to:', cfg['audit'])
    except Exception as e:
        print(f"Failed writing audit file {cfg['audit']}: {e}", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    cfg = load_env()
    handle_stage(args.stage, args, cfg)
    if args.prune_extras or args.prune_list:
        prune_extras(cfg['etl'], args, cfg['dry_run'], cfg['removed'])
    write_audit(cfg, args.stage)


if __name__ == '__main__':
    main()
