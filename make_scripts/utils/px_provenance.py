#!/usr/bin/env python3
"""PX: Generate etl_provenance_report.csv listing artifacts and SHA256, row counts, timestamp ranges.

Searches common artifact folders (data_etl, data_ai, models, notebooks/eda_outputs) and writes a deterministic CSV.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple
import os
import math


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count_and_date_range(p: Path) -> Tuple[Optional[int], Optional[str]]:
    # returns (rows, "min->max") based on 'date' or 'date_utc' columns when present
    try:
        import csv as _csv
        with p.open(newline='', encoding='utf-8') as fh:
            rdr = _csv.DictReader(fh)
            rows = 0
            dates = []
            date_keys = [k for k in (rdr.fieldnames or []) if k.lower() in ('date','date_utc','timestamp','timestamp_utc')]
            for r in rdr:
                rows += 1
                if date_keys:
                    v = r.get(date_keys[0])
                    if v:
                        dates.append(v)
            if rows == 0:
                return 0, None
            if dates:
                return rows, f"{min(dates)}->{max(dates)}"
            return rows, None
    except Exception:
        return None, None


def find_artifacts(root: Path):
    # scan common locations
    patterns = [
        'data_etl/**/manifests/*.json',
        'data_etl/**/*.csv',
        'data_ai/**/snapshots/**/features_daily.csv',
        'models/**/*',
        'notebooks/eda_outputs/**/*',
        '**/*_manifest.json',
        '**/manifests/**/*.csv'
    ]
    seen = set()
    for pat in patterns:
        for p in root.glob(pat):
            if p.is_file():
                seen.add(p.resolve())
    return sorted(seen)


def atomic_write(path: Path, rows):
    tmp = path.with_suffix(path.suffix + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['path','source','schema_version','sha256','row_count','timestamp_range','upstream_stage'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)


def infer_source_and_stage(p: Path) -> Tuple[str,str]:
    parts = [s.lower() for s in p.parts]
    if 'zepp' in parts:
        return 'zepp', 'extract/parse'
    if 'apple' in parts or 'ios' in parts:
        return 'apple', 'extract/parse'
    if 'processed' in parts or 'zepp_processed' in parts:
        return 'processed', 'etl/processed'
    if 'models' in parts:
        return 'model', 'ai/train'
    if 'notebooks' in parts or 'eda_outputs' in parts:
        return 'notebook', 'analysis'
    return 'other', 'unknown'


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--repo-root', default='.', help='Repo root to scan')
    p.add_argument('--out-dir', required=True)
    args = p.parse_args(argv)

    root = Path(args.repo_root)
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    outcsv = outdir / 'etl_provenance_report.csv'

    artifacts = find_artifacts(root)
    rows = []
    for a in artifacts:
        sha = sha256_file(a)
        row_count, ts_range = (None, None)
        if a.suffix.lower() == '.csv':
            rc, tr = csv_row_count_and_date_range(a)
            row_count, ts_range = rc, tr
        source, stage = infer_source_and_stage(a)
        rows.append({'path': str(a.relative_to(root)), 'source': source, 'schema_version': '', 'sha256': sha, 'row_count': row_count if row_count is not None else '', 'timestamp_range': ts_range if ts_range else '', 'upstream_stage': stage})

    # deterministic order
    rows = sorted(rows, key=lambda r: r['path'])
    atomic_write(outcsv, rows)
    print('WROTE:', outcsv)


if __name__ == '__main__':
    main()
