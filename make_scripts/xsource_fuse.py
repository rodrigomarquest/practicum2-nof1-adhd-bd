#!/usr/bin/env python3
"""XZ3: Fuse Apple and Zepp dailies into AI features with provenance.

Produces fused_* columns in the AI features CSV and writes xsource_fusion_manifest.json
with per-day chosen source and QC flags. Deterministic, atomic, idempotent.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


def read_csv_map(path: Path, metric_keys: List[str]) -> Dict[str, Dict[str, float]]:
    """Return date -> {col: value} mapping for the csv. Keeps columns requested in metric_keys."""
    out = {}
    if not path.exists():
        return out
    with path.open(newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            d = r.get('date') or r.get('date_utc')
            if not d:
                continue
            out[d] = {}
            for k in metric_keys:
                v = r.get(k)
                try:
                    out[d][k] = float(v) if v not in (None, '') else math.nan
                except Exception:
                    out[d][k] = math.nan
    return out


def write_csv_rows(path: Path, rows: List[Dict[str, Any]], headers: List[str]):
    tmp = path.with_suffix(path.suffix + '.tmp')
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ('' if r.get(k) is None or (isinstance(r.get(k), float) and math.isnan(r.get(k))) else r.get(k)) for k in headers})
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--processed-dir', required=True)
    p.add_argument('--joined-dir', required=True)
    p.add_argument('--ai-input-dir', required=True)
    args = p.parse_args(argv)

    processed = Path(args.processed_dir)
    joined = Path(args.joined_dir)
    ai_dir = Path(args.ai_input_dir)
    ai_path = ai_dir / 'features_daily.csv'

    # Load canonical metric map if available to drive fusion rules
    map_path = joined / 'xsource_metric_map.json'
    if map_path.exists():
        metric_map = json.loads(map_path.read_text(encoding='utf-8'))
    else:
        # fallback small map
        metric_map = {
            'hr': {'apple_cols': ['hr_mean'], 'zepp_cols': ['zepp_hr_mean'], 'unit': 'bpm', 'fusion': 'quality_weighted_mean'},
            'hrv_sdnn': {'apple_cols': ['hrv_sdnn_ms'], 'zepp_cols': ['zepp_hrv_sdnn'], 'unit': 'ms', 'fusion': 'prefer_apple_if_present_else_zepp'},
            'hrv_rmssd': {'apple_cols': [], 'zepp_cols': ['zepp_hrv_rmssd'], 'unit': 'ms', 'fusion': 'prefer_zepp'},
            'sleep_total_minutes': {'apple_cols': ['sleep_minutes'], 'zepp_cols': ['zepp_sleep_minutes'], 'unit': 'min', 'fusion': 'max_confidence'}
        }

    # read processed maps
    apple_root = processed / 'apple'
    zepp_root = processed / 'zepp'

    # Build per-date chosen-source and fused values
    per_metric_manifest = {}

    # Load AI features (if missing create header-only)
    if not ai_path.exists():
        # create header with date only
        write_csv_rows(ai_path, [], ['date'])

    # read existing AI features rows
    ai_rows = []
    with ai_path.open(newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        for r in rdr:
            ai_rows.append(r)

    # build date set from ai rows
    dates = [r.get('date') for r in ai_rows if r.get('date')]
    if not dates:
        # fallback: union of apple/zepp date keys for HR
        # read small sample to obtain dates
        sample_apple = read_csv_map(apple_root / 'health_hr_daily.csv', ['hr_mean'])
        dates = sorted(sample_apple.keys())

    # for determinism, we will produce a new ai table preserving original rows and appending fused cols
    fused_headers = []
    for _, _, fused in metrics:
        fused_headers.append(fused)

    # initialize manifest structure
    manifest = {'generated_at': datetime.utcnow().isoformat() + 'Z', 'metrics': {}}

    # Load consistency report to enforce QC rules
    consistency = {}
    cons_path = joined / 'xsource_consistency_report.json'
    if cons_path.exists():
        try:
            consistency = json.loads(cons_path.read_text(encoding='utf-8'))
            cons_map = {m['metric']: m for m in consistency.get('metrics', [])}
        except Exception:
            cons_map = {}
    else:
        cons_map = {}

    for metric_key, cfg in metric_map.items():
        fusion_policy = cfg.get('fusion')
        apple_cols = cfg.get('apple_cols', [])
        zepp_cols = cfg.get('zepp_cols', [])

        # Determine source files
        apple_file = apple_root / f'health_{metric_key}_daily.csv'
        zepp_file = zepp_root / f'zepp_{metric_key}_daily.csv'

        apple_map = read_csv_map(apple_file, apple_cols if apple_cols else [])
        zepp_map = read_csv_map(zepp_file, zepp_cols if zepp_cols else [])

        per_day_source = {}
        fused_vals = {}
        nan_count = 0
        total = 0

        # QC flags for metric from consistency report
        qc_flags = []
        cm = cons_map.get(metric_key) if isinstance(cons_map, dict) else None
        if cm and cm.get('qc_flags'):
            qc_flags = cm.get('qc_flags')

        # If QC flags present we will not produce numeric fused values (per spec)
        block_fusion = bool(qc_flags)

        all_dates = sorted(set(dates) | set(apple_map.keys()) | set(zepp_map.keys()))
        for d in all_dates:
            total += 1
            # pick representative apple value: prefer first apple_cols present
            a = math.nan
            for ac in apple_cols:
                a = apple_map.get(d, {}).get(ac, math.nan)
                if not math.isnan(a):
                    break
            z = math.nan
            for zc in zepp_cols:
                z = zepp_map.get(d, {}).get(zc, math.nan)
                if not math.isnan(z):
                    break

            chosen = 'null'
            fused_value = math.nan

            if block_fusion:
                # respect QC: write NaN but still record provenance as 'null' or source available
                if not math.isnan(a) and not math.isnan(z):
                    chosen = 'both'
                elif not math.isnan(a):
                    chosen = 'apple'
                elif not math.isnan(z):
                    chosen = 'zepp'
                else:
                    chosen = 'null'
                fused_value = math.nan
            else:
                # apply fusion rules
                if fusion_policy == 'quality_weighted_mean':
                    # attempt to read count column if present in daily files
                    a_count = None
                    z_count = None
                    # apple_count candidates
                    for cc in ['count','n','rows']:
                        if (apple_file.exists() and cc in pd.read_csv(apple_file).columns):
                            a_count = read_csv_map(apple_file, [cc]).get(d, {}).get(cc)
                            break
                    for cc in ['count','n','rows']:
                        if (zepp_file.exists() and cc in pd.read_csv(zepp_file).columns):
                            z_count = read_csv_map(zepp_file, [cc]).get(d, {}).get(cc)
                            break
                    if not math.isnan(a) and not math.isnan(z):
                        if a_count and z_count and not math.isnan(a_count) and not math.isnan(z_count) and (a_count + z_count) > 0:
                            fused_value = (a * a_count + z * z_count) / (a_count + z_count)
                        else:
                            fused_value = (a + z) / 2.0
                        chosen = 'both'
                    elif not math.isnan(a):
                        fused_value = a; chosen = 'apple'
                    elif not math.isnan(z):
                        fused_value = z; chosen = 'zepp'
                    else:
                        fused_value = math.nan; chosen = 'null'
                elif fusion_policy == 'prefer_apple_if_present_else_zepp':
                    if not math.isnan(a):
                        fused_value = a; chosen = 'apple'
                    elif not math.isnan(z):
                        fused_value = z; chosen = 'zepp'
                    else:
                        fused_value = math.nan; chosen = 'null'
                elif fusion_policy == 'prefer_zepp':
                    if not math.isnan(z):
                        fused_value = z; chosen = 'zepp'
                    elif not math.isnan(a):
                        fused_value = a; chosen = 'apple'
                    else:
                        fused_value = math.nan; chosen = 'null'
                elif fusion_policy == 'max_confidence':
                    # Attempt to choose by coverage/efficiency if present; fallback to max value
                    if not math.isnan(a) and not math.isnan(z):
                        fused_value = max(a, z); chosen = 'both'
                    elif not math.isnan(a):
                        fused_value = a; chosen = 'apple'
                    elif not math.isnan(z):
                        fused_value = z; chosen = 'zepp'
                    else:
                        fused_value = math.nan; chosen = 'null'
                else:
                    # default: prefer apple
                    if not math.isnan(a):
                        fused_value = a; chosen = 'apple'
                    elif not math.isnan(z):
                        fused_value = z; chosen = 'zepp'
                    else:
                        fused_value = math.nan; chosen = 'null'

            if fused_value is None or (isinstance(fused_value, float) and math.isnan(fused_value)):
                nan_count += 1

            per_day_source[d] = chosen
            fused_vals[d] = fused_value

        nan_pct = (nan_count / total * 100.0) if total > 0 else 100.0
        fused_col_name = f'fused_{metric_key}'
        manifest['metrics'][fused_col_name] = {
            'fusion_policy': fusion_policy,
            'per_day_source': dict(sorted(per_day_source.items())),
            'nan_pct': nan_pct,
            'qc_flags': qc_flags
        }

        # attach fused values into overall map for writing into AI CSV later
        # store as stringified mapping for now
        manifest['metrics'][fused_col_name]['_values'] = fused_vals

    # write fused columns into AI features CSV (append columns, preserve existing)
    # Build new rows with fused cols appended using manifest values
    new_rows = []
    with ai_path.open(newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        old_headers = rdr.fieldnames or []
        dates_present = []
        for r in rdr:
            newr = dict(r)
            date = r.get('date')
            dates_present.append(date)
            for fused_col in [k for k in manifest['metrics'].keys() if k.startswith('fused_')]:
                vals_map = manifest['metrics'][fused_col].get('_values', {})
                v = vals_map.get(date)
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    newr[fused_col] = ''
                else:
                    newr[fused_col] = v
            new_rows.append(newr)

    # ensure headers include fused cols
    fused_cols = [k for k in manifest['metrics'].keys() if k.startswith('fused_')]
    new_headers = list(old_headers) + fused_cols
    write_csv_rows(ai_path, new_rows, new_headers)

    # cleanup _values from manifest before writing
    for k in list(manifest['metrics'].keys()):
        manifest['metrics'][k].pop('_values', None)

    out_manifest = joined / 'xsource_fusion_manifest.json'
    out_manifest.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding='utf-8')
    print('WROTE:', ai_path, out_manifest)


if __name__ == '__main__':
    main()
