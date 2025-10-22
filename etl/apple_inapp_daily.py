#!/usr/bin/env python3
"""Daily aggregation for Apple normalized metrics.

Produces three CSVs under processed/apple/:
 - health_hr_daily.csv -> date, hr_mean, hr_std, n_hr
 - health_hrv_sdnn_daily.csv -> date, hrv_sdnn_mean, hrv_sdnn_std, n_hrv_sdnn
 - health_sleep_daily.csv -> date, sleep_sum_h, sleep_mean, sleep_std, n_sleep

Date bucketing uses civil UTC day.
If an input metric is missing, writes an empty CSV with header and manifest.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:
    import dateutil.parser as _dp
    from dateutil import tz
except Exception:
    _dp = None
    tz = None

from make_scripts import io_utils
from etl_modules.common.progress import progress_open


def parse_dt(s: str) -> datetime:
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


def bucket_day_utc(dt: datetime) -> str:
    return dt.date().isoformat()


def read_metric(csv_path: Path) -> List[List[str]]:
    out = []
    if not csv_path.exists():
        return out
    if not csv_path.exists():
        return out
    with progress_open(csv_path, desc=f"Read {csv_path.name}") as bf:
        text_iter = (line.decode('utf-8') for line in bf)
        r = csv.reader(text_iter)
        hdr = next(r, None)
        for row in r:
            if not row:
                continue
            out.append(row)
    return out


def aggregate_hr(rows: List[List[str]]) -> Dict[str, List[float]]:
    by_day = defaultdict(list)
    for row in rows:
        ts = row[0]
        try:
            dt = parse_dt(ts)
        except Exception:
            continue
        day = bucket_day_utc(dt)
        try:
            val = float(row[1])
        except Exception:
            continue
        by_day[day].append(val)
    return by_day


def aggregate_sleep(rows: List[List[str]]) -> Dict[str, List[float]]:
    # rows: start,end,stage — compute duration hours and assign to start day bucket
    by_day = defaultdict(list)
    for row in rows:
        if len(row) < 2:
            continue
        start = row[0]
        end = row[1]
        try:
            ds = parse_dt(start)
            de = parse_dt(end)
        except Exception:
            continue
        dur_h = max(0.0, (de - ds).total_seconds() / 3600.0)
        day = bucket_day_utc(ds)
        by_day[day].append(dur_h)
    return by_day


def write_aggregate(processed_dir: Path, filename: str, headers: List[str], rows: List[List[str]], meta_extra: Optional[Dict] = None, dry_run: bool = False, run_id: str = '', participant: str = '') -> None:
    outp = processed_dir / 'apple' / filename
    processed_dir.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print(f'DRY RUN: would write {outp} rows={len(rows)}')
        return
    io_utils.write_atomic_csv(outp, rows, headers)
    meta = {
        'script': str(Path(__file__).resolve()),
        'run_id': run_id or '',
        'participant': participant or '',
        'schema_version': io_utils.SCHEMA_VERSION,
    }
    if meta_extra:
        meta.update(meta_extra)
    m = io_utils.manifest(outp, meta)
    print(f'Wrote {outp} ({m.get("rows")} rows)')


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

    # HR
    hr_rows = read_metric((normalized / 'apple' / 'apple_heart_rate.csv'))
    hr_by_day = aggregate_hr(hr_rows)
    hr_out = []
    for day in sorted(hr_by_day.keys()):
        vals = hr_by_day[day]
        hr_out.append([day, f"{statistics.mean(vals):.6f}", f"{statistics.pstdev(vals) if len(vals)>1 else 0.0:.6f}", str(len(vals))])
    write_aggregate(processed, 'health_hr_daily.csv', ['date', 'hr_mean', 'hr_std', 'n_hr'], hr_out, dry_run=args.dry_run, run_id=args.run_id, participant=args.participant)

    # HRV SDNN
    hrv_rows = read_metric((normalized / 'apple' / 'apple_hrv_sdnn.csv'))
    hrv_by_day = aggregate_hr(hrv_rows)
    hrv_out = []
    for day in sorted(hrv_by_day.keys()):
        vals = hrv_by_day[day]
        hrv_out.append([day, f"{statistics.mean(vals):.6f}", f"{statistics.pstdev(vals) if len(vals)>1 else 0.0:.6f}", str(len(vals))])
    write_aggregate(processed, 'health_hrv_sdnn_daily.csv', ['date', 'hrv_sdnn_mean', 'hrv_sdnn_std', 'n_hrv_sdnn'], hrv_out, dry_run=args.dry_run, run_id=args.run_id, participant=args.participant)

    # Sleep
    sleep_rows = read_metric((normalized / 'apple' / 'apple_sleep_intervals.csv'))
    sleep_by_day = aggregate_sleep(sleep_rows)
    sleep_out = []
    for day in sorted(sleep_by_day.keys()):
        vals = sleep_by_day[day]
        # convert durations already in hours
        sleep_out.append([day, f"{sum(vals):.6f}", f"{statistics.mean(vals):.6f}" if vals else '0.0', f"{statistics.pstdev(vals) if len(vals)>1 else 0.0:.6f}", str(len(vals))])
    write_aggregate(processed, 'health_sleep_daily.csv', ['date', 'sleep_sum_h', 'sleep_mean', 'sleep_std', 'n_sleep'], sleep_out, dry_run=args.dry_run, run_id=args.run_id, participant=args.participant)


if __name__ == '__main__':
    main()
