#!/usr/bin/env python3
"""Prompt A7: Apple In-App Daily Aggregate & Join (per-snapshot folders)

Reads normalized per-metric CSVs from a snapshot, aggregates to UTC daily
statistics (count,min,max,mean,median,std), writes per-metric daily CSVs,
joins them into a features table, and writes manifests and a modeling export.

Usage:
  make_scripts/apple/run_a7_apple.py --participant P000001 --snapshot 2025-09-29 [--dry-run]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import platform
import subprocess
from make_scripts.utils.snapshot_lock import SnapshotLock, SnapshotLockError

try:
    from etl_modules.common.progress import progress_bar
except Exception:
    from contextlib import contextmanager
    @contextmanager
    def progress_bar(total, desc: str = "", unit: str = "items"):
        class _B:
            def update(self, n=1):
                return None
            def close(self):
                return None
        yield _B()

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def atomic_write_text(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
    os.close(fd)
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(data)
    os.replace(tmp, str(path))

def atomic_write_csv_from_df(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.' + out_path.name + '.tmp.', dir=str(out_path.parent))
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.replace(tmp, str(out_path))

def load_normalized(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, parse_dates=['timestamp_utc'], keep_default_na=True)
    except Exception:
        # fall back to reading without parse and coerce
        df = pd.read_csv(path)
        if 'timestamp_utc' in df.columns:
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
        return df

def aggregate_daily(df: pd.DataFrame, value_col: str = 'value') -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        # return empty but deterministic columns
        return pd.DataFrame(columns=['date_utc','count','min','max','mean','median','std'])
    df = df.copy()
    # ensure timestamp_utc is datetime
    df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'], errors='coerce')
    df = df.dropna(subset=['timestamp_utc'])
    # date in UTC civil day
    df['date_utc'] = df['timestamp_utc'].dt.tz_convert('UTC').dt.date
    # coerce values to numeric
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    g = df.groupby('date_utc')[value_col]
    agg = g.agg(['count','min','max','mean','median','std']).reset_index()
    # ensure deterministic ordering and column types
    agg = agg[['date_utc','count','min','max','mean','median','std']]
    return agg

def make_manifest(run_id: str, participant: str, snapshot: str, inputs: list, outputs: list) -> dict:
    return {
        'schema_version': '1',
        'run_id': run_id,
        'participant': participant,
        'snapshot': snapshot,
        'inputs': inputs,
        'outputs': outputs,
    }

def normalize_and_join(pid: str, snap: str, dry_run: bool = False, hrv_method: str = 'both', lock_timeout: int = 3600, force_lock: bool = False) -> Dict[str, Any]:
    base = Path('data') / 'etl' / pid / 'snapshots' / snap
    norm_dir = base / 'normalized' / 'apple'
    processed = base / 'processed' / 'apple'
    joined_dir = base / 'joined'
    ai_out = Path('data') / 'ai' / pid / 'snapshots' / snap

    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

    # metric files we expect
    metrics = {
        'apple_heart_rate': ('health_hr_daily.csv', 'value'),
        'apple_hrv_sdnn': ('health_hrv_sdnn_daily.csv', 'value'),
        'apple_hrv_rmssd': ('health_hrv_rmssd_daily.csv', 'value'),
        'apple_sleep_intervals': ('health_sleep_daily.csv', 'value'),
    }

    inputs = []
    outputs = []

    # QC settings
    NAN_THRESHOLD_PCT = float(os.environ.get('NAN_THRESHOLD_PCT', os.environ.get('NAN_THRESHOLD', 5)))
    QC_MODE = os.environ.get('QC_MODE', 'flag')  # 'flag' or 'fail'

    # progress: total stages = metrics + join + export
    stages = list(metrics.keys()) + ['join','export']
    with progress_bar(total=len(stages), desc='A7 stages', unit='steps') as pbar:
        daily_frames = {}
        # aggregate each metric
        for m in metrics:
            pbar.update(1)
            in_path = norm_dir / f"{m}.csv"
            df = load_normalized(in_path)
            # inputs: path, sha256 if present, rows
            try:
                sha_in = sha256_of_file(in_path) if in_path.exists() and not dry_run else None
            except Exception:
                sha_in = None
            rows_in = len(df) if not df.empty else 0
            inputs.append({'path': str(in_path), 'sha256': sha_in, 'rows': rows_in})
            agg = aggregate_daily(df, value_col='value')
            # ensure date_utc is ISO string for CSV
            if not agg.empty:
                agg['date_utc'] = agg['date_utc'].astype(str)
            out_name, _vc = metrics[m]
            out_path = processed / out_name
            # write or dry-run
            if dry_run:
                print('DRY_RUN: would write', out_path, 'rows=', len(agg))
            else:
                atomic_write_csv_from_df(agg, out_path)
            # compute NaN% for the daily aggregated table (count refers to non-null counts)
            nan_pct_overall = None
            if agg.empty:
                nan_pct_overall = 100.0
            else:
                # determine expected rows (unique date_utc) and count NaNs in numeric cols
                # For daily table, NaNs appear in numeric agg columns when inputs were missing
                total_cells = len(agg) * (len(agg.columns) - 1)  # exclude date_utc
                if total_cells <= 0:
                    nan_pct_overall = 0.0
                else:
                    numnans = int(agg.iloc[:,1:].isna().sum().sum())
                    nan_pct_overall = (numnans / total_cells) * 100.0

            # manifest info will include sha256 and nan_pct_overall
            if dry_run:
                sha = None
            else:
                sha = sha256_of_file(out_path) if out_path.exists() else None
            out_meta = {'path': str(out_path), 'sha256': sha, 'nan_pct_overall': round(nan_pct_overall, 3), 'rows': len(agg)}
            # attach QC flag if threshold exceeded and QC_MODE=flag
            if nan_pct_overall is not None and nan_pct_overall > NAN_THRESHOLD_PCT:
                if QC_MODE == 'flag':
                    out_meta.setdefault('qc_flags', []).append('nan_rate_exceeds_threshold')
                elif QC_MODE == 'fail':
                    print(f"QC_FAIL: {m} nan_pct_overall={nan_pct_overall:.3f} > {NAN_THRESHOLD_PCT}", file=sys.stderr)
                    sys.exit(4)

            outputs.append(out_meta)
            daily_frames[m] = agg

        # join stage
        pbar.update(1)
        # start with union of all dates
        all_dates = None
        for m, df in daily_frames.items():
            if df.empty:
                continue
            d = df[['date_utc']].copy()
            if all_dates is None:
                all_dates = d
            else:
                all_dates = pd.concat([all_dates, d], ignore_index=True)
        if all_dates is None:
            # no data; create empty joined with date_utc column
            joined = pd.DataFrame(columns=['date_utc'])
        else:
            all_dates = all_dates.drop_duplicates().sort_values('date_utc')
            joined = all_dates.copy()
        # left-join metric columns
        for m, df in daily_frames.items():
            out_name, _vc = metrics[m]
            col_prefix = m.replace('apple_','')
            if df.empty:
                # add empty columns deterministic order
                cols = ['count','min','max','mean','median','std']
                for c in cols:
                    joined[f'{col_prefix}_{c}'] = pd.NA
            else:
                tmp = df.copy()
                # ensure date_utc string
                tmp['date_utc'] = tmp['date_utc'].astype(str)
                tmp = tmp.set_index('date_utc')
                cols = ['count','min','max','mean','median','std']
                for c in cols:
                    joined = joined.merge(tmp[[c]].rename(columns={c: f'{col_prefix}_{c}'}), how='left', left_on='date_utc', right_index=True)

    # optionally add version metadata from joined/version_log_enriched.csv
        ver_path = joined_dir / 'version_log_enriched.csv'
        if ver_path.exists():
            vdf = pd.read_csv(ver_path, parse_dates=['date'])
            # assume version log has a date column we can join on
            vdf['date_utc'] = vdf['date'].dt.date.astype(str)
            # pick relevant metadata columns (segment_id if exists)
            meta_cols = [c for c in vdf.columns if c not in ('date',)]
            meta = vdf[['date_utc'] + [c for c in meta_cols if c != 'date_utc']]
            joined = joined.merge(meta, how='left', on='date_utc')

        # write joined CSV
        joined_path = joined_dir / 'apple_features_join.csv'
        if dry_run:
            print('DRY_RUN: would write joined features', joined_path, 'rows=', len(joined))
        else:
            atomic_write_csv_from_df(joined, joined_path)
        # compute per-column NaN% for joined table
        col_nan = []
        joined_nan_overall = None
        if not joined.empty:
            # compute NaN% per column (exclude date_utc)
            cols = [c for c in joined.columns if c != 'date_utc']
            col_pct_list = []
            for c in cols:
                total = len(joined)
                if total == 0:
                    pct = 100.0
                else:
                    pct = (joined[c].isna().sum() / total) * 100.0
                col_pct_list.append((c, pct))
            # deterministic: sort by pct desc then name, compute top-10 for manifest
            col_nan = sorted([(c, round(p, 3)) for c, p in col_pct_list], key=lambda x: (-x[1], x[0]))[:10]
            # overall nan% for joined table: average of column NaN% (rounded)
            if col_pct_list:
                joined_nan_overall = round(sum(p for _, p in col_pct_list) / len(col_pct_list), 3)
            else:
                joined_nan_overall = 0.0
        else:
            col_nan = []
            joined_nan_overall = 100.0

        if dry_run:
            sha_joined = None
        else:
            sha_joined = sha256_of_file(joined_path) if joined_path.exists() else None

        # attach QC info for joined table based on column NaN rates
        joined_out_meta = {'path': str(joined_path), 'sha256': sha_joined, 'per_column_nan_pct_top10': col_nan, 'nan_pct_overall': joined_nan_overall, 'rows': len(joined)}
        # if any column exceeds threshold, flag or fail depending on QC_MODE
        if joined_nan_overall is not None and joined_nan_overall > NAN_THRESHOLD_PCT:
            # In the join-level check, use the max column pct to decide flag/fail
            max_col_pct = max((p for (_c, p) in col_nan), default=0)
            if max_col_pct > NAN_THRESHOLD_PCT:
                if QC_MODE == 'flag':
                    joined_out_meta.setdefault('qc_flags', []).append('nan_rate_exceeds_threshold')
                elif QC_MODE == 'fail':
                    print(f"QC_FAIL: joined table nan_pct_overall={joined_nan_overall:.3f} > {NAN_THRESHOLD_PCT}", file=sys.stderr)
                    sys.exit(4)

        outputs.append(joined_out_meta)

        # export modeling package
        pbar.update(1)
        ai_out.mkdir(parents=True, exist_ok=True)
        model_path = ai_out / 'features_daily.csv'
        if dry_run:
            print('DRY_RUN: would export modeling features to', model_path)
        else:
            atomic_write_csv_from_df(joined, model_path)
    outputs.append({'path': str(model_path), 'sha256': sha256_of_file(model_path) if model_path.exists() else None, 'rows': len(joined)})

    # include units in outputs where applicable for provenance
    # map known health outputs to units
    UNIT_MAP = {
        'health_hr_daily.csv': 'bpm',
        'health_hrv_sdnn_daily.csv': 'ms',
        'health_hrv_rmssd_daily.csv': 'ms',
        'health_sleep_daily.csv': 'minutes',
    }
    for o in outputs:
        p = o.get('path','')
        for k,v in UNIT_MAP.items():
            if p.endswith(k):
                o.setdefault('unit', v)
                break

    # write enriched manifest with provenance
    manifest_path = processed / 'a7_manifest.json'
    manifest = {
        'schema_version': '1.1',
        'producer': f"make_scripts/{Path(__file__).name}",
        'run_id': run_id,
        'participant': pid,
        'snapshot_date': snap,
        'git': None,
        'args': {'cli': sys.argv, 'env_overrides': {'NAN_THRESHOLD_PCT': NAN_THRESHOLD_PCT, 'QC_MODE': QC_MODE}},
        'system': {'os': platform.platform(), 'python_version': platform.python_version(), 'tz': datetime.now(timezone.utc).astimezone().tzname()},
        'inputs': inputs,
        'outputs': outputs,
        'nan_threshold_pct': NAN_THRESHOLD_PCT,
        'qc_mode': QC_MODE,
    }
    # try to populate git info
    try:
        commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'], stderr=subprocess.DEVNULL).decode().strip())
        manifest['git'] = {'commit': commit, 'branch': branch, 'dirty': dirty}
    except Exception:
        manifest['git'] = 'unknown'

    # sort inputs/outputs deterministically by path
    manifest['inputs'] = sorted(manifest['inputs'], key=lambda x: x.get('path',''))
    manifest['outputs'] = sorted(manifest['outputs'], key=lambda x: x.get('path',''))

    # attach producer with commit short if available (safe access)
    git = manifest.get('git')
    if isinstance(git, dict):
        manifest['producer'] = f"make_scripts/{Path(__file__).name} {git.get('commit','')[:7]}"

    # compute overall nan_pct across daily outputs
    nan_overall_list = [o.get('nan_pct_overall') for o in outputs if o.get('nan_pct_overall') is not None]
    manifest['nan_pct_overall'] = round(sum(nan_overall_list)/len(nan_overall_list),3) if nan_overall_list else None

    # Wrap non-dry-run writes with a snapshot lock to prevent concurrent runs
    snapshot_root = base
    lock = SnapshotLock(snapshot_root, 'processed', pid, snap, timeout_sec=lock_timeout, force=force_lock)
    if dry_run:
        print('DRY_RUN: would write manifest to', manifest_path)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        try:
            lock.acquire()
        except SnapshotLockError as e:
            print(str(e), file=sys.stderr)
            sys.exit(5)
        try:
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
        finally:
            lock.release()
            print('Released lock:', lock.lock_path)

    return {'inputs': inputs, 'outputs': outputs, 'manifest': str(manifest_path)}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--hrv-method', choices=['sdnn','rmssd','both'], default='both', help='Which HRV method(s) to include in aggregation')
    p.add_argument('--force-lock', action='store_true', help='Override stale snapshot lock')
    p.add_argument('--lock-timeout', type=int, default=3600, help='Seconds before a lock is considered stale')
    args = p.parse_args(argv)
    res = normalize_and_join(args.participant, args.snapshot, dry_run=args.dry_run, hrv_method=args.hrv_method, lock_timeout=args.lock_timeout, force_lock=args.force_lock)
    print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
