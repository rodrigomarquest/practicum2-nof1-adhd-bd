#!/usr/bin/env python3
"""Aggregate iOS usage by UTC day and join with Apple features.

Writes:
 - {processed_dir}/ios/ios_usage_daily.csv
 - augmented AI input CSV (overwrites path passed in) with added ios_* columns
 - join manifest JSON under {joined_dir}/features_joined_ios_manifest.json

Behavior: atomic writes, deterministic ordering, QC on NaN% for new cols.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict, Counter
from datetime import datetime, timezone


def sha256_of_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, text, mode='w', encoding='utf-8'):
    tmp = path + '.tmp'
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(tmp, mode, encoding=encoding) as fh:
        fh.write(text)
    os.replace(tmp, path)


def parse_events(path):
    # returns list of dict rows and header list
    if not os.path.exists(path):
        return [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return rows, reader.fieldnames or []


def get_date_from_row(row):
    # prefer explicit date_utc column; else infer from timestamp_utc
    date_cols = ['date_utc', 'date']
    for c in date_cols:
        if c in row and row[c].strip():
            return row[c].strip()
    ts_cols = ['timestamp_utc', 'timestamp', 'time', 'ts']
    for c in ts_cols:
        if c in row and row[c].strip():
            s = row[c].strip()
            # try ISO parse
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%fZ"):
                try:
                    dt = datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
                    return dt.strftime('%Y-%m-%d')
                except Exception:
                    continue
            # fallback: date part of string
            try:
                return s.split('T')[0]
            except Exception:
                pass
    return None


def parse_usage_minutes(row):
    # look for usage in minutes or seconds
    if 'ios_usage_total_min' in row and row['ios_usage_total_min'].strip():
        try:
            return float(row['ios_usage_total_min'])
        except Exception:
            pass
    candidates_min = ['usage_total_min', 'usage_min', 'usage_minutes', 'minutes']
    for c in candidates_min:
        if c in row and str(row[c]).strip():
            try:
                return float(row[c])
            except Exception:
                pass
    candidates_sec = ['usage_seconds', 'duration_seconds', 'secs']
    for c in candidates_sec:
        if c in row and str(row[c]).strip():
            try:
                return float(row[c]) / 60.0
            except Exception:
                pass
    return 0.0


def detect_bundle_col(headers):
    for c in ('bundle_id', 'bundleId', 'app', 'app_bundle', 'app_identifier', 'app_id'):
        if c in headers:
            return c
    return None


def detect_category_col(headers):
    for c in ('category', 'top_category', 'app_category'):
        if c in headers:
            return c
    return None


def aggregate_events(rows, headers):
    # aggregate by date_utc (YYYY-MM-DD)
    agg = {}
    bundle_col = detect_bundle_col(headers)
    cat_col = detect_category_col(headers)
    for row in rows:
        date = get_date_from_row(row)
        if not date:
            continue
        if date not in agg:
            agg[date] = {'minutes': 0.0, 'apps': set(), 'cats': Counter()}
        minutes = parse_usage_minutes(row)
        agg[date]['minutes'] += minutes
        if bundle_col and row.get(bundle_col):
            agg[date]['apps'].add(row.get(bundle_col))
        if cat_col and row.get(cat_col):
            agg[date]['cats'][row.get(cat_col)] += 1
    # produce deterministic list sorted by date
    out = []
    for date in sorted(agg.keys()):
        minutes = round(agg[date]['minutes'], 6)
        n_apps = len(agg[date]['apps'])
        top_cat = None
        if agg[date]['cats']:
            top_cat = sorted(agg[date]['cats'].items(), key=lambda x: (-x[1], x[0]))[0][0]
        out.append({'date_utc': date, 'usage_total_min': minutes, 'n_apps': n_apps, 'top_category': top_cat or ''})
    return out


def write_csv(path, rows, headers):
    # deterministic column order per headers
    with open(path + '.tmp', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            # ensure keys exist
            out = {k: ('' if r.get(k) is None else r.get(k)) for k in headers}
            writer.writerow(out)
    os.replace(path + '.tmp', path)


def read_csv_rows(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames or []


def augment_ai_features(ai_path, agg_rows):
    # Read AI input CSV, append ios_* columns (ios_usage_total_min, ios_n_apps, ios_top_category)
    rows, headers = read_csv_rows(ai_path)
    if not headers:
        # create header-only augmented file with new columns
        new_headers = ['date_utc'] + ['ios_usage_total_min', 'ios_n_apps', 'ios_top_category']
        write_csv(ai_path, [], new_headers)
        return 0, {}, 0
    # map date -> agg
    agg_map = {r['date_utc']: r for r in agg_rows}
    new_cols = ['ios_usage_total_min', 'ios_n_apps', 'ios_top_category']
    out_rows = []
    for r in rows:
        date = r.get('date_utc','')
        a = agg_map.get(date)
        if a:
            r['ios_usage_total_min'] = a.get('usage_total_min', '')
            r['ios_n_apps'] = a.get('n_apps', '')
            r['ios_top_category'] = a.get('top_category', '')
        else:
            r['ios_usage_total_min'] = ''
            r['ios_n_apps'] = ''
            r['ios_top_category'] = ''
        out_rows.append(r)
    new_headers = list(headers) + new_cols
    write_csv(ai_path, out_rows, new_headers)
    # compute NaN% for new cols
    total = len(out_rows)
    nan_counts = dict.fromkeys(new_cols, 0)
    for r in out_rows:
        for c in new_cols:
            if r.get(c,'')=='' or r.get(c) is None:
                nan_counts[c] += 1
    nan_pct = {c: (nan_counts[c]/total*100 if total>0 else 0.0) for c in new_cols}
    return total, nan_pct, len([d for d in agg_rows if d['date_utc'] in {rr.get('date_utc') for rr in out_rows}])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--normalized-dir', required=True)
    p.add_argument('--ai-input', required=True)
    p.add_argument('--processed-dir', required=True)
    p.add_argument('--joined-dir', required=True)
    p.add_argument('--participant', required=False)
    p.add_argument('--snapshot', required=False)
    args = p.parse_args(argv)

    events_path = os.path.join(args.normalized_dir, 'ios_usage_events.csv')
    processed_out = os.path.join(args.processed_dir, 'ios', 'ios_usage_daily.csv')
    ai_input = args.ai_input
    manifest_out = os.path.join(args.joined_dir, 'features_joined_ios_manifest.json')

    # read events
    rows, headers = parse_events(events_path)
    events_count = len(rows)
    agg_rows = aggregate_events(rows, headers)
    processed_rows = []
    for r in agg_rows:
        processed_rows.append({'date_utc': r['date_utc'], 'usage_total_min': r['usage_total_min'], 'n_apps': r['n_apps'], 'top_category': r['top_category']})

    # write processed file (ensure deterministic header order)
    proc_headers = ['date_utc','usage_total_min','n_apps','top_category']
    # ensure dir
    proc_dir = os.path.dirname(processed_out)
    if proc_dir and not os.path.exists(proc_dir):
        os.makedirs(proc_dir, exist_ok=True)
    # atomic write
    write_csv(processed_out, processed_rows, proc_headers)

    # augment ai features file
    ai_rows_before, _ = read_csv_rows(ai_input)
    ai_sha_before = sha256_of_file(ai_input)
    agg_sha = sha256_of_file(events_path)
    # augment and get stats
    ai_total_rows, nan_pct_map, joined_count = augment_ai_features(ai_input, agg_rows)

    # compute inputs stats
    manifest = {
        'participant': args.participant or '',
        'snapshot': args.snapshot or '',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {
            'ios_events_csv': {'path': events_path, 'sha256': agg_sha, 'rows': events_count},
            'ai_input_csv': {'path': ai_input, 'sha256': ai_sha_before, 'rows': len(ai_rows_before)}
        },
        'outputs': {
            'ios_daily_csv': {'path': processed_out, 'rows': len(processed_rows)},
            'ai_input_augmented': {'path': ai_input, 'rows': ai_total_rows}
        },
        'join': {
            'joined_rows': joined_count,
            'coverage_pct': (joined_count / len(processed_rows) * 100.0) if len(processed_rows)>0 else 0.0
        },
        'qc': {
            'nan_pct': nan_pct_map,
            'qc_mode': os.environ.get('QC_MODE','flag'),
            'nan_threshold_pct': float(os.environ.get('NAN_THRESHOLD_PCT', '5'))
        },
        'schema_version': '1.0'
    }

    # QC enforcement
    max_nan = 0.0
    for v in nan_pct_map.values():
        if v > max_nan:
            max_nan = v
    qc_mode = os.environ.get('QC_MODE','flag')
    nan_threshold = float(os.environ.get('NAN_THRESHOLD_PCT','5'))
    manifest['qc']['max_nan_pct'] = max_nan
    manifest['qc']['qc_pass'] = True
    if qc_mode == 'fail' and max_nan > nan_threshold:
        manifest['qc']['qc_pass'] = False

    # write manifest atomically and deterministically
    if not os.path.exists(os.path.dirname(manifest_out)):
        os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    atomic_write(manifest_out, json.dumps(manifest, sort_keys=True, indent=2))

    # exit non-zero if QC fail
    if qc_mode == 'fail' and not manifest['qc']['qc_pass']:
        print('QC failed: max NaN% > threshold (', max_nan, '>', nan_threshold, ')')
        sys.exit(2)

    print('Wrote:', processed_out)
    print('Augmented:', ai_input)
    print('Manifest:', manifest_out)


if __name__ == '__main__':
    main()
