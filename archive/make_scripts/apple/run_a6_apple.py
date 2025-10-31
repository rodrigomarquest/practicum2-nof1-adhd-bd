#!/usr/bin/env python3
"""Run Prompt A6: Normalize Apple In-App extracted metrics + QC for a snapshot.

Usage:
  make_scripts/apple/run_a6_apple.py --participant P000001 --snapshot 2025-10-22 [--dry-run]

This script:
 - reads data/etl/<pid>/snapshots/<snap>/extracted/apple/export.xml
 - extracts records via iterparse and normalizes per metric into CSVs under
   data/etl/<pid>/snapshots/<snap>/normalized/apple/*.csv
 - writes QC summary to processed/apple/etl_qc_summary.csv and a manifest
 - supports DRY_RUN (no writes) and atomic writes when not dry-run
 - computes SHA256 for inputs and outputs
 - respects a simple on_progress(step, pct) callback

This is a compact implementation focused on acceptance criteria A6.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import xml.etree.ElementTree as ET

# reuse small helpers from project when possible
try:
    from make_scripts.io_utils import atomic_write_text
except Exception:
    def atomic_write_text(path: Path, data: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.tmp.', dir=str(path.parent))
        os.close(fd)
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(data)
        os.replace(tmp, str(path))

# minimal SHA256

def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

# types mapping (subset)
TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRate": ("apple_heart_rate", "bpm"),
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": ("apple_hrv_sdnn", "ms"),
    "HKCategoryTypeIdentifierSleepAnalysis": ("apple_sleep_intervals", "minutes"),
    "HKQuantityTypeIdentifierStepCount": ("apple_steps", "count"),
    # Potential RR / RMSSD types
    "HKQuantityTypeIdentifierHeartRateVariabilityRMSSD": ("apple_hrv_rmssd", "ms"),
    "HKQuantityTypeIdentifierRRInterval": ("apple_rr_interval", "s"),
}

# reverse mapping metric -> unit for manifest enrichment
METRIC_UNIT = {v[0]: v[1] for v in TYPE_MAP.values()}

# helper: parse ISO-ish datetime appearing in export.xml (various formats)
from dateutil.parser import parse as parse_dt  # requirements include python-dateutil
import platform
import math
from make_scripts.utils.snapshot_lock import SnapshotLock, SnapshotLockError

# Use the shared progress bar utility so we get a single-line tqdm bar
try:
    from etl_modules.common.progress import progress_bar
except Exception:
    # fallback: define a no-op context manager compatible with usage below
    from contextlib import contextmanager
    @contextmanager
    def progress_bar(total, desc: str = "", unit: str = "items"):
        class _B:
            def update(self, n=1):
                return None
            def close(self):
                return None
        yield _B()


def normalize_snapshot(pid: str, snap: str, dry_run: bool = False, hrv_method: str = 'both', lock_timeout: int = 3600, force_lock: bool = False) -> Dict[str, Any]:
    base = Path('data') / 'etl' / pid / 'snapshots' / snap
    extracted = base / 'extracted' / 'apple'
    normalized_dir = base / 'normalized' / 'apple'
    processed_dir = base / 'processed' / 'apple'
    # inputs
    xml_path = extracted / 'export.xml'
    if not xml_path.exists():
        raise FileNotFoundError(f'export.xml not found: {xml_path}')

    # prepare outputs (in-memory accumulation per metric)
    # each metric -> list of tuples (dt_obj, ts_iso, value, source, device)
    metrics: Dict[str, List[Tuple[datetime,str,Optional[float],str,str]]] = {}
    input_sha = sha256_of_file(xml_path)

    # first pass: count records to make progress estimation
    total = 0
    with xml_path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024*1024), b''):
            total += chunk.count(b'<Record')

    # stream parse and normalize with a single-line tqdm progress bar
    parsed = 0
    with xml_path.open('rb') as fh:
        context = ET.iterparse(fh, events=('end',))
        with progress_bar(total=total or None, desc='Parsing Records', unit='items') as _bar:
            for _, elem in context:
                if elem.tag != 'Record':
                    elem.clear()
                    continue
                parsed += 1
                # update the bar for every record (tqdm is efficient); if tqdm missing, _bar is no-op
                try:
                    _bar.update(1)
                except Exception:
                    pass
                typ = elem.attrib.get('type')
                if typ not in TYPE_MAP:
                    elem.clear()
                    continue
                metric_name, canonical_unit = TYPE_MAP[typ]
                ts = elem.attrib.get('startDate') or elem.attrib.get('value')
                if not ts:
                    elem.clear()
                    continue
                try:
                    dt = parse_dt(ts)
                    dt_utc = dt.astimezone(timezone.utc)
                    ts_s = dt_utc.isoformat().replace('+00:00','Z')
                except Exception:
                    elem.clear()
                    continue
                val = elem.attrib.get('value')
                try:
                    val_f = float(val) if val is not None else None
                except Exception:
                    val_f = None
                source = elem.attrib.get('sourceName','')
                device = elem.attrib.get('device','')
                metrics.setdefault(metric_name, []).append((dt_utc, ts_s, val_f, source, device))
                elem.clear()

    print('Parsing complete; parsed', parsed, 'records')
    # Ensure requested HRV metrics exist even if no direct records were present
    # hrv_method: 'sdnn'|'rmssd'|'both'
    if hrv_method in ('sdnn','both'):
        metrics.setdefault('apple_hrv_sdnn', [])
    if hrv_method in ('rmssd','both'):
        metrics.setdefault('apple_hrv_rmssd', [])

    # If RR-intervals are present and rmssd requested but rmssd records absent,
    # compute RMSSD per UTC date from RR intervals and populate apple_hrv_rmssd
    if 'apple_rr_interval' in metrics and (hrv_method in ('rmssd','both')):
        rr_rows = metrics.get('apple_rr_interval', [])
        # group by date and compute RMSSD for each date
        rr_rows.sort(key=lambda r: r[0])
        from collections import defaultdict
        by_date = defaultdict(list)
        for dt_obj, ts_iso, val, src, dev in rr_rows:
            if val is None:
                continue
            # value likely in seconds ('s') per TYPE_MAP; convert to milliseconds
            by_date[dt_obj.date()].append(float(val))
        # compute rmssd per date and append to apple_hrv_rmssd metric list
        for d, intervals in sorted(by_date.items()):
            if len(intervals) < 2:
                continue
            diffs = [intervals[i+1] - intervals[i] for i in range(len(intervals)-1)]
            # convert to ms and compute RMSSD
            diffs_ms = [(x * 1000.0) for x in diffs]
            msq = sum(x*x for x in diffs_ms) / len(diffs_ms)
            rmssd = math.sqrt(msq)
            # create a timestamp at UTC midnight for the day
            dt_mid = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            ts_iso = dt_mid.isoformat().replace('+00:00','Z')
            metrics.setdefault('apple_hrv_rmssd', []).append((dt_mid, ts_iso, float(f"{rmssd:.3f}"), '', ''))

    # write normalized CSVs
    outputs: Dict[str,str] = {}
    for mname, rows in metrics.items():
        rows.sort(key=lambda r: (r[0], r[3]))  # sort by dt, source
        out_path = normalized_dir / f"{mname}.csv"
        if dry_run:
            print('DRY_RUN: would write', out_path, 'rows=', len(rows))
            outputs[mname] = str(out_path)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.' + out_path.name + '.tmp.', dir=str(out_path.parent))
        os.close(fd)
        with open(tmp, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            # include unit in header for clarity
            unit = METRIC_UNIT.get(mname, '')
            writer.writerow(['timestamp_utc','timestamp_iso','value','unit','source','device'])
            for dt_obj, ts_iso, val, source, device in rows:
                writer.writerow([dt_obj.isoformat(), ts_iso, '' if val is None else f"{val}", unit, source, device])
        os.replace(tmp, str(out_path))
        outputs[mname] = str(out_path)

    print('Writing normalized outputs...')

    # compute QC summary
    qc_rows = []
    for mname, rows in metrics.items():
        n = len(rows)
        nulls = sum(1 for r in rows if r[2] is None)
        dup_ts = n - len({r[1] for r in rows})
        dates = [r[0] for r in rows]
        date_min = min(dates).isoformat() if dates else ''
        date_max = max(dates).isoformat() if dates else ''
        median_interval = ''
        if mname == 'apple_heart_rate' and len(dates) >= 2:
            intervals = sorted((dates[i+1]-dates[i]).total_seconds() for i in range(len(dates)-1))
            median = intervals[len(intervals)//2]
            median_interval = f"{median:.1f}"
        qc_rows.append((mname, n, nulls/n if n else 0.0, dup_ts/n if n else 0.0, date_min, date_max, median_interval))

    # write qc summary
    summary_path = processed_dir / 'etl_qc_summary.csv'
    manifest_path = processed_dir / 'etl_qc_summary_manifest.json'
    snapshot_root = base
    lock = SnapshotLock(snapshot_root, 'processed', pid, snap, timeout_sec=int(lock_timeout), force=bool(force_lock))

    if dry_run:
        print('DRY_RUN: would write QC summary to', summary_path)
        # build dry-run manifest for preview
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        manifest_preview = {
            'schema_version': '1.1',
            'producer': f"make_scripts/{Path(__file__).name}",
            'git': 'unknown',
            'args': {'cli': sys.argv, 'env_overrides': {}},
            'system': {'os': platform.platform(), 'python_version': platform.python_version(), 'tz': datetime.now(timezone.utc).astimezone().tzname()},
            'run_id': run_id,
            'participant': pid,
            'snapshot_date': snap,
            'inputs': [{'path': str(xml_path), 'sha256': input_sha, 'rows': parsed}],
            'outputs': [{'metric': k, 'path': p, 'sha256': None, 'rows': None} for k,p in outputs.items()] + [{'metric': 'etl_qc_summary', 'path': str(summary_path), 'sha256': None, 'rows': len(qc_rows)}]
        }
        print(json.dumps(manifest_preview, indent=2, sort_keys=True))
    else:
        try:
            lock.acquire()
        except SnapshotLockError as e:
            print(str(e), file=sys.stderr)
            sys.exit(5)
        try:
            print('Acquired lock:', lock.lock_path)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.' + summary_path.name + '.tmp.', dir=str(summary_path.parent))
            os.close(fd)
            with open(tmp, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(['metric','rows','null_pct','dup_pct','date_min','date_max','median_interval_seconds'])
                for r in qc_rows:
                    writer.writerow(r)
            os.replace(tmp, str(summary_path))

            # build enriched manifest with provenance
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')

            # git info
            try:
                import subprocess
                commit = subprocess.check_output(['git','rev-parse','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
                branch = subprocess.check_output(['git','rev-parse','--abbrev-ref','HEAD'], stderr=subprocess.DEVNULL).decode().strip()
                dirty = bool(subprocess.check_output(['git','status','--porcelain'], stderr=subprocess.DEVNULL).decode().strip())
                git_info = {'commit': commit, 'branch': branch, 'dirty': dirty}
            except Exception:
                git_info = 'unknown'

            system_info = {
                'os': platform.platform(),
                'python_version': platform.python_version(),
                'tz': datetime.now(timezone.utc).astimezone().tzname(),
            }

            # inputs/outputs with rows/sha when possible
            inputs_list = [{'path': str(xml_path), 'sha256': input_sha, 'rows': parsed}]
            outputs_list = []
            for k, p in outputs.items():
                pth = Path(p)
                sha = sha256_of_file(pth) if pth.exists() else None
                rows_count = None
                try:
                    if pth.exists():
                        with pth.open('r', encoding='utf-8') as fh:
                            rows_count = sum(1 for _ in fh) - 1
                            if rows_count < 0:
                                rows_count = 0
                except Exception:
                    rows_count = None
                outputs_list.append({'metric': k, 'path': str(pth), 'sha256': sha, 'rows': rows_count})
            # qc summary
            sha_summary = sha256_of_file(summary_path) if summary_path.exists() else None
            outputs_list.append({'metric': 'etl_qc_summary', 'path': str(summary_path), 'sha256': sha_summary, 'rows': len(qc_rows)})

            # deterministic ordering
            outputs_list = sorted(outputs_list, key=lambda x: x.get('path',''))

            # build producer string: prefer short commit if available
            if isinstance(git_info, dict):
                prod_suffix = git_info.get('commit','')[:7]
            else:
                prod_suffix = git_info
            manifest = {
                'schema_version': '1.1',
                'producer': f"make_scripts/{Path(__file__).name} {prod_suffix}",
                'git': git_info,
                'args': {'cli': sys.argv, 'env_overrides': {}},
                'system': system_info,
                'run_id': run_id,
                'participant': pid,
                'snapshot_date': snap,
                'inputs': inputs_list,
                'outputs': outputs_list,
            }

            fd, tmpm = tempfile.mkstemp(prefix='.' + manifest_path.name + '.tmp.', dir=str(manifest_path.parent))
            os.close(fd)
            with open(tmpm, 'w', encoding='utf-8') as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
            os.replace(tmpm, str(manifest_path))
        finally:
            lock.release()
            print('Released lock:', lock.lock_path)

    print('Done.')
    return {
        'inputs': {'export_xml': str(xml_path)},
        'normalized': outputs,
        'qc_summary': str(summary_path) if not dry_run else str(summary_path),
        'manifest': str(manifest_path) if not dry_run else str(manifest_path),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--hrv-method', choices=['sdnn','rmssd','both'], default='both', help='Which HRV method(s) to produce')
    p.add_argument('--force-lock', action='store_true', help='Override an existing stale lock')
    p.add_argument('--lock-timeout', type=int, default=3600, help='Seconds before a lock is considered stale')
    args = p.parse_args(argv)
    res = normalize_snapshot(args.participant, args.snapshot, dry_run=args.dry_run, hrv_method=args.hrv_method, lock_timeout=args.lock_timeout, force_lock=args.force_lock)
    print('Result:', json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
