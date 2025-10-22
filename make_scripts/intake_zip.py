#!/usr/bin/env python3
"""Intake a ZIP export into the new data taxonomy.

Usage (Makefile will call via INTAKE_ARGS):
  RAW_DIR="$(RAW_DIR)" EXTRACTED_DIR="$(EXTRACTED_DIR)" PARTICIPANT="$(PARTICIPANT)" \
    $(VENV_PY) make_scripts/intake_zip.py --source apple --zip-path path/to.zip --stage

Behavior:
 - copy given ZIP into RAW_DIR/<source>/ with canonical name
 - detect apple in-app vs iTunes backup by peeking entries
 - if --stage: extract minimal files to EXTRACTED_DIR/<source>/
 - write JSON + CSV run logs under data/etl/<participant>/runs/<RUN_ID>/logs/

This script prefers CLI args over ENV variables (CLI > ENV > defaults).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone
import zipfile
import uuid
import csv
import os
from typing import Dict, Any, Optional


def now_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def detect_apple_zip_kind(z: zipfile.ZipFile) -> str:
    # in-app export often contains 'export.xml' or top-level apple_health_export/ dir
    names = [n for n in z.namelist()]
    if any('export.xml' in n for n in names):
        return 'apple_inapp'
    # iTunes/Finder backup: contains Manifest.db and AppDomain-* directories
    if any(n.endswith('Manifest.db') or n.startswith('AppDomain-') for n in names):
        return 'ios_backup'
    # Zepp exports have specific entries (fallback to zepp)
    if any(n.endswith('.csv') or 'zepp' in n.lower() for n in names):
        return 'zepp'
    return 'unknown'


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def copy_to_raw(src: Path, raw_dir: Path, source_label: str, dry_run: bool) -> Path:
    ts = now_utc_compact()
    if source_label == 'apple':
        name = f'apple_health_export_{ts}.zip'
    elif source_label == 'zepp':
        name = f'zepp_export_{ts}.zip'
    elif source_label == 'ios_backup':
        name = f'ios_backup_{ts}.zip'
    else:
        name = f'unknown_{ts}.zip'
    dest_dir = raw_dir / source_label
    ensure_dir(dest_dir)
    dest = dest_dir / name
    if dry_run:
        print(f"DRY RUN: would copy {src} -> {dest}")
        return dest
    shutil.copy2(src, dest)
    return dest


def write_run_logs(participant: str, run_id: str, log_json: Dict[str, Any], log_csv_row: Dict[str, Any]):
    base = Path('data') / 'etl' / participant / 'runs' / run_id / 'logs'
    ensure_dir(base)
    ts = now_utc_compact()
    json_path = base / f'intake_log_{ts}.json'
    csv_path = base / 'intake_log.csv'
    with json_path.open('w', encoding='utf8') as fh:
        json.dump(log_json, fh, indent=2, ensure_ascii=False)
    # append one-line CSV (if file doesn't exist, write header)
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(log_csv_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(log_csv_row)


def extract_minimal(zpath: Path, dest_root: Path, kind: str, dry_run: bool) -> None:
    ensure_dir(dest_root)
    if dry_run:
        print(f"DRY RUN: would extract minimal files from {zpath} into {dest_root} for kind={kind}")
        return
    with zipfile.ZipFile(zpath, 'r') as z:
        names = z.namelist()
        if kind == 'apple_inapp':
            # extract export.xml (may be nested)
            candidates = [n for n in names if n.endswith('export.xml')]
            if not candidates:
                print('No export.xml found to extract')
                return
            ensure_dir(dest_root)
            for n in candidates:
                target = dest_root / Path(n).name
                with z.open(n) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
        elif kind == 'ios_backup':
            # extract Manifest.db and AppDomain-* entries (best-effort)
            candidates = [n for n in names if n.endswith('Manifest.db') or n.startswith('AppDomain-')]
            if not candidates:
                print('No Manifest.db/AppDomain- entries found to extract')
                return
            for n in candidates:
                target = dest_root / Path(n).name
                with z.open(n) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
        elif kind == 'zepp':
            # for zepp we might extract csvs
            csvs = [n for n in names if n.endswith('.csv')]
            for n in csvs:
                target = dest_root / Path(n).name
                with z.open(n) as src, open(target, 'wb') as out:
                    shutil.copyfileobj(src, out)
        else:
            print('Unknown kind; nothing extracted')


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description='Intake ZIP into raw/ and optionally minimally extract into extracted/')
    p.add_argument('--source', choices=['apple', 'zepp', 'ios_backup'], required=True)
    p.add_argument('--zip-path', required=True, help='Path to source ZIP')
    p.add_argument('--participant', required=False)
    p.add_argument('--raw-dir', required=False)
    p.add_argument('--extracted-dir', required=False)
    p.add_argument('--stage', action='store_true', help='Also stage minimal extracted files into extracted/')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    # ENV fallback
    env_participant = os.environ.get('PARTICIPANT')
    participant = args.participant or env_participant or 'P000001'
    env_raw = os.environ.get('RAW_DIR')
    raw_dir = Path(args.raw_dir or env_raw or f'data/raw/{participant}')
    env_extracted = os.environ.get('EXTRACTED_DIR')
    extracted_dir = Path(args.extracted_dir or env_extracted or f'data/etl/{participant}/extracted')
    dry_run = args.dry_run or os.environ.get('DRY_RUN', '') in ('1', 'true', 'True')

    src = Path(args.zip_path)
    if not src.exists():
        print('ERROR: zip-path not found:', src)
        sys.exit(2)

    # ensure raw dir exists and copy
    dest = copy_to_raw(src, raw_dir, args.source, dry_run)

    # compute checksum and metadata
    metadata: Dict[str, Any] = {}
    if not dry_run:
        metadata['sha256'] = sha256_of_file(dest)
        metadata['size_bytes'] = dest.stat().st_size
    else:
        metadata['sha256'] = 'DRYRUN'
        metadata['size_bytes'] = None

    # peek into zip to detect kind
    kind = 'unknown'
    if not dry_run:
        try:
            with zipfile.ZipFile(dest, 'r') as z:
                kind = detect_apple_zip_kind(z)
        except Exception:
            kind = 'unknown'
    else:
        kind = args.source if args.source in ('apple', 'zepp', 'ios_backup') else 'unknown'

    # If staging requested, extract minimal files
    if args.stage:
        if args.source == 'apple':
            stage_dest = Path(extracted_dir) / 'apple'
            extract_minimal(dest, stage_dest, 'apple_inapp' if kind == 'apple_inapp' else kind, dry_run)
        elif args.source == 'ios_backup':
            stage_dest = Path(extracted_dir) / 'ios_backup'
            extract_minimal(dest, stage_dest, kind, dry_run)
        elif args.source == 'zepp':
            stage_dest = Path(extracted_dir) / 'zepp'
            extract_minimal(dest, stage_dest, kind, dry_run)

    # Build run log
    run_id = uuid.uuid4().hex
    log_json = {
        'run_id': run_id,
        'timestamp': now_utc_compact(),
        'participant': participant,
        'source': args.source,
        'raw_path': str(dest),
        'kind': kind,
        'metadata': metadata,
    }
    log_csv_row = {
        'run_id': run_id,
        'timestamp': log_json['timestamp'],
        'participant': participant,
        'source': args.source,
        'raw_path': str(dest),
        'kind': kind,
        'sha256': metadata.get('sha256'),
        'size_bytes': metadata.get('size_bytes'),
    }

    write_run_logs(participant, run_id, log_json, log_csv_row)
    print('Wrote run log for', run_id)


if __name__ == '__main__':
    main()
