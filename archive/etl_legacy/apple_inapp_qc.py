#!/usr/bin/env python3
"""Quality checks over normalized Apple CSVs.

Scans normalized/apple/*.csv (and fallback to normalized root) and computes per-file
metrics: rows, null %, duplicate timestamp %, min/max timestamp, sample-rate estimate (HR only).
Flags: duplicate_timestamps, mixed_tz_detected, gaps_hr_gt6h.

Writes `processed/apple/etl_qc_summary.csv` and updates `processed/apple/etl_report.md`.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import dateutil.parser as _dp
    from dateutil import tz
except Exception:
    _dp = None
    tz = None

from make_scripts import io_utils
from etl_modules.common.progress import progress_open, progress_bar


def parse_dt(s: str) -> datetime:
    if not s:
        raise ValueError('empty')
    if _dp:
        dt = _dp.parse(s)
    else:
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        if tz is not None:
            dt = dt.replace(tzinfo=tz.tzutc())
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def find_csv_files(normalized_dir: Path) -> List[Path]:
    out: List[Path] = []
    # prefer normalized/apple/*.csv
    app_dir = normalized_dir / 'apple'
    if app_dir.exists() and app_dir.is_dir():
        out.extend(sorted(app_dir.glob('*.csv'), key=lambda p: str(p)))
    # also include any top-level csvs that look like apple metrics
    for p in sorted(normalized_dir.glob('*.csv'), key=lambda p: str(p)):
        if 'apple' in p.name or p.name.startswith('apple_') or p.name.startswith('health_'):
            out.append(p)
        else:
            # still include files that look like metric names used in repo
            if p.name in ('apple_heart_rate.csv', 'apple_hrv_sdnn.csv', 'apple_sleep_intervals.csv'):
                out.append(p)
    # deduplicate preserve order
    seen = set()
    uniq = []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq


def inspect_file(p: Path) -> Dict:
    # default metrics
    rows = 0
    nulls = 0
    ts_list: List[str] = []
    tz_markers = set()
    value_cols: List[List[str]] = []

    # use progress_open to show read progress for large CSVs
    with progress_open(p, desc=f"Inspect {p.name}") as bf:
        # wrapped binary file -> decode lines
        text_iter = (line.decode('utf-8') for line in bf)
        r = csv.reader(text_iter)
        hdr = next(r, None)
        for row in r:
            rows += 1
            if not row:
                nulls += 1
                continue
            # timestamp in first column
            ts = row[0] if len(row) >= 1 else ''
            if ts:
                ts_list.append(ts)
                if ts.endswith('Z') or ts.endswith('z'):
                    tz_markers.add('Z')
                elif ('+' in ts[-6:]) or ('-' in ts[-6:]):
                    tz_markers.add('offset')
                else:
                    tz_markers.add('na')
            # consider null if any non-timestamp column is empty
            non_ts = row[1:]
            if any((c is None or c == '') for c in non_ts):
                nulls += 1
            value_cols.append(non_ts)

    dup_pct = 0.0
    dup_flag = False
    sample_rate_sec = None
    gaps_gt6h = False
    mixed_tz = len(tz_markers) > 1

    if rows > 0:
        # duplicates based on timestamp occurrences
        ctr = Counter(ts_list)
        dup_count = sum(c - 1 for c in ctr.values() if c > 1)
        dup_pct = dup_count / rows
        dup_flag = dup_count > 0

        # parse timestamps to datetimes for rate/gaps if possible
        dts: List[datetime] = []
        for s in sorted([t for t in ts_list if t]):
            try:
                dts.append(parse_dt(s))
            except Exception:
                pass
        if len(dts) >= 2:
            deltas = [ (dts[i+1] - dts[i]).total_seconds() for i in range(len(dts)-1) if (dts[i+1] - dts[i]).total_seconds() > 0 ]
            if deltas:
                # median sample interval
                sample_rate_sec = float(statistics.median(deltas))
                # gap > 6h
                gaps_gt6h = any(d > 6*3600 for d in deltas)

    # min/max timestamp (string) using lexical min/max which should hold for ISO timestamps
    date_min = min(ts_list) if ts_list else None
    date_max = max(ts_list) if ts_list else None

    null_pct = (nulls / rows) if rows > 0 else 0.0

    return {
        'path': str(p),
        'rows': rows,
        'null_pct': null_pct,
        'dup_pct': dup_pct,
        'date_min': date_min,
        'date_max': date_max,
        'sample_rate_sec': sample_rate_sec,
        'duplicate_timestamps': dup_flag,
        'mixed_tz_detected': mixed_tz,
        'gaps_hr_gt6h': gaps_gt6h,
    }


def write_summary(processed_dir: Path, qc_rows: List[Dict], dry_run: bool, run_id: str, participant: str) -> None:
    processed_dir = Path(processed_dir) / 'apple'
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_csv = processed_dir / 'etl_qc_summary.csv'
    manifest_path = processed_dir / 'etl_qc_summary_manifest.json'
    headers = ['file', 'rows', 'null_pct', 'dup_pct', 'date_min', 'date_max', 'sample_rate_sec', 'duplicate_timestamps', 'mixed_tz_detected', 'gaps_hr_gt6h']
    csv_rows = []
    for r in qc_rows:
        csv_rows.append([
            r['path'],
            str(r['rows']),
            f"{r['null_pct']:.6f}",
            f"{r['dup_pct']:.6f}",
            r['date_min'] or '',
            r['date_max'] or '',
            f"{r['sample_rate_sec']:.6f}" if r['sample_rate_sec'] is not None else '',
            str(bool(r['duplicate_timestamps'])),
            str(bool(r['mixed_tz_detected'])),
            str(bool(r['gaps_hr_gt6h'])),
        ])

    if dry_run:
        print(f'DRY RUN: would write {out_csv} rows={len(csv_rows)}')
        print(f'DRY RUN: would write {manifest_path}')
    else:
        io_utils.write_atomic_csv(out_csv, csv_rows, headers)
        meta = {
            'script': str(Path(__file__).resolve()),
            'run_id': run_id or '',
            'participant': participant or '',
            'schema_version': io_utils.SCHEMA_VERSION,
        }
        m = io_utils.manifest(out_csv, meta)
        print(f'Wrote {out_csv} ({m.get("rows")} rows)')

        # per-run manifest summarizing produced summary and files
        try:
            files_info = []
            # include the QC summary csv itself and any per-file manifests under processed/apple
            for p in sorted(processed_dir.glob('*')):
                if p.is_file():
                    try:
                        sha = io_utils.sha256_of_file(p)
                    except Exception:
                        sha = None
                    files_info.append({'path': str(p), 'name': p.name, 'sha256': sha})
            manifest_obj = {
                'schema_version': io_utils.SCHEMA_VERSION,
                'run_id': run_id or '',
                'participant': participant or '',
                'generated_at_utc': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat().replace('+00:00','Z'),
                'files': files_info,
            }
            io_utils.write_atomic_text(manifest_path, __import__('json').dumps(manifest_obj, indent=2, sort_keys=True, ensure_ascii=False))
            print(f'Wrote manifest: {manifest_path}')
        except Exception as e:
            print(f'Warning: failed to write QC manifest: {e}')

    # write human-readable report
    report_md = processed_dir / 'etl_report.md'
    lines = [f"# Apple ETL QC Report\n", f"Run: {run_id or ''}\n", f"Participant: {participant or ''}\n", "## Summary\n"]
    for r in qc_rows:
        flags = []
        if r['duplicate_timestamps']:
            flags.append('duplicate_timestamps')
        if r['mixed_tz_detected']:
            flags.append('mixed_tz_detected')
        if r['gaps_hr_gt6h']:
            flags.append('gaps_hr_gt6h')
        lines.append(f"- {Path(r['path']).name}: rows={r['rows']} date_min={r['date_min']} date_max={r['date_max']} flags={','.join(flags) or 'ok'}\n")

    if dry_run:
        print(f'DRY RUN: would write {report_md}')
    else:
        io_utils.write_atomic_text(report_md, '\n'.join(lines))
        print(f'Wrote {report_md}')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--normalized-dir', required=True)
    p.add_argument('--processed-dir', required=True)
    p.add_argument('--run-id', default='')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    normalized = Path(args.normalized_dir)
    processed = Path(args.processed_dir)
    files = find_csv_files(normalized)
    if not files:
        print(f'No apple CSVs found under {normalized} or {normalized/"apple"}')
        # still write empty summary and report
        if not args.dry_run:
            write_summary(processed, [], args.dry_run, args.run_id, args.participant)
        return

    qc_rows = []
    for f in files:
        try:
            r = inspect_file(f)
            qc_rows.append(r)
            if r['dup_pct'] > 0.001:
                print(f'Warning: {f} duplicate timestamp percentage {r["dup_pct"]:.6f} > 0.001 (0.1%)')
        except Exception as e:
            print(f'Failed to inspect {f}: {e}')

    write_summary(processed, qc_rows, args.dry_run, args.run_id, args.participant)


if __name__ == '__main__':
    main()
