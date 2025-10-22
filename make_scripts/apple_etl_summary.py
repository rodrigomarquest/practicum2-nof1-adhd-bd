#!/usr/bin/env python3
"""Write a run-level summary JSON for apple ETL outputs.

Writes data/etl/<PID>/runs/<RUN_ID>/apple_etl_summary.json with an entry
for each expected output file containing path, exists, sha256 and status.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--processed-root', default='data/etl')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    base = Path(args.processed_root) / args.participant
    out_dir = base / 'runs' / args.run_id
    files = [
        base / 'processed' / 'apple' / 'health_hr_daily.csv',
        base / 'processed' / 'apple' / 'health_hrv_sdnn_daily.csv',
        base / 'processed' / 'apple' / 'health_sleep_daily.csv',
    ]

    summary: Dict[str, Dict] = {}
    for f in files:
        entry = {'path': str(f), 'exists': f.exists(), 'sha256': None, 'status': 'missing'}
        if f.exists():
            try:
                entry['sha256'] = sha256_of_file(f)
                entry['status'] = 'ok'
            except Exception as e:
                entry['status'] = 'error'
                entry['error'] = str(e)
        summary[f.name] = entry

    if args.dry_run:
        print('DRY RUN: would write run summary to', out_dir / 'apple_etl_summary.json')
        for k, v in summary.items():
            print('-', k, v['status'], 'exists=' + str(v['exists']))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'apple_etl_summary.json'
    out_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Wrote run summary:', out_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
