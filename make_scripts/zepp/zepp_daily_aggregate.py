#!/usr/bin/env python3
"""Aggregate Zepp normalized CSVs by UTC day and join with AI features_daily.csv.

Writes processed per-metric daily CSVs under processed_dir/zepp and augments AI input CSV.
Creates join manifest under joined_dir.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict, Counter
from statistics import mean, median, pstdev
from datetime import datetime, timezone


def sha256_of_file(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path, text, encoding='utf-8'):
    tmp = path + '.tmp'
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(tmp, 'w', encoding=encoding, newline='') as fh:
        fh.write(text)
    os.replace(tmp, path)


def read_csv(path):
    if not os.path.exists(path):
        return [], []
    with open(path, newline='', encoding='utf-8') as fh:
        reader = csv.DictReader(fh)
        return list(reader), reader.fieldnames or []


def write_csv_rows(path, rows, headers):
    with open(path + '.tmp', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in headers})
    os.replace(path + '.tmp', path)


def ts_to_date(ts):
    # expect ISO-like timestamp; return YYYY-MM-DD
    try:
        if not ts:
            return None
        # handle timezone aware
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.astimezone(timezone.utc).date().isoformat()
    except Exception:
        # fallback: split date
        try:
            return ts.split('T')[0]
        except Exception:
            return None


def stats_from_values(vals):
    if not vals:
        return {'count': 0, 'min': '', 'max': '', 'mean': '', 'median': '', 'std': ''}
    nums = [float(v) for v in vals]
    cnt = len(nums)
    mn = min(nums)
    mx = max(nums)
    avg = mean(nums)
    med = median(nums)
    st = 0.0
    if cnt > 1:
        try:
            st = pstdev(nums)
        except Exception:
            st = 0.0
    return {'count': cnt, 'min': mn, 'max': mx, 'mean': avg, 'median': med, 'std': st}


def aggregate_metric(path, value_col='value'):
    rows, headers = read_csv(path)
    by_date = defaultdict(list)
    for r in rows:
        date = ts_to_date(r.get('timestamp_utc',''))
        if not date:
            continue
        v = r.get(value_col,'')
        if v=='' or v is None:
            continue
        try:
            by_date[date].append(float(v))
        except Exception:
            continue
    out_rows = []
    for date in sorted(by_date.keys()):
        s = stats_from_values([str(x) for x in by_date[date]])
        out_rows.append({'date_utc': date, 'min': s['min'], 'max': s['max'], 'mean': s['mean'], 'median': s['median'], 'std': s['std'], 'count': s['count']})
    return out_rows


def augment_ai(ai_path, aggs, prefix='zepp_'):
    rows, headers = read_csv(ai_path)
    if not headers:
        # write header-only with date_utc + zepp columns
        new_cols = []
        for k,cols in aggs.items():
            for c in cols['cols']:
                new_cols.append(prefix + k + '_' + c)
        new_headers = ['date_utc'] + new_cols
        write_csv_rows(ai_path, [], new_headers)
        return 0, {c:100.0 for c in new_cols}, 0
    # build map date->values for each metric
    aggr_map = {}
    for k,v in aggs.items():
        m = {r['date_utc']: r for r in v['rows']}
        aggr_map[k] = m

    new_cols = []
    for k,cols in aggs.items():
        for c in cols['cols']:
            new_cols.append(prefix + k + '_' + c)

    out_rows = []
    for r in rows:
        date = r.get('date_utc','')
        for k,cols in aggs.items():
            m = aggr_map.get(k, {})
            rowvals = m.get(date)
            for c in cols['cols']:
                colname = prefix + k + '_' + c
                if rowvals:
                    outval = rowvals.get(c,'')
                else:
                    outval = ''
                r[colname] = outval
        out_rows.append(r)

    new_headers = list(headers) + new_cols
    write_csv_rows(ai_path, out_rows, new_headers)

    # compute nan pct
    total = len(out_rows)
    nan_counts = dict.fromkeys(new_cols, 0)
    for r in out_rows:
        for c in new_cols:
            if r.get(c,'')=='' or r.get(c) is None:
                nan_counts[c] += 1
    nan_pct = {c: (nan_counts[c]/total*100 if total>0 else 100.0) for c in new_cols}
    joined_count = sum(1 for r in out_rows if any(r.get(c) not in (None,'') for c in new_cols))
    return total, nan_pct, joined_count


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--normalized-dir', required=True)
    p.add_argument('--ai-input', required=True)
    p.add_argument('--processed-dir', required=True)
    p.add_argument('--joined-dir', required=True)
    p.add_argument('--participant', required=False)
    p.add_argument('--snapshot', required=False)
    args = p.parse_args(argv)

    norm = args.normalized_dir
    ai = args.ai_input
    proc = os.path.join(args.processed_dir, 'zepp')
    joined = args.joined_dir
    participant = args.participant or ''
    snapshot = args.snapshot or ''

    if not os.path.exists(proc):
        os.makedirs(proc, exist_ok=True)
    if not os.path.exists(joined):
        os.makedirs(joined, exist_ok=True)

    # metrics to aggregate and their output columns
    metrics = {
        'hr': {'file': os.path.join(norm,'zepp_hr.csv'), 'cols':['min','max','mean','median','std','count']},
        'hrv': {'file': os.path.join(norm,'zepp_hrv.csv'), 'cols':['min','max','mean','median','std','count']},
        'sleep': {'file': os.path.join(norm,'zepp_sleep.csv'), 'cols':['min','max','mean','median','std','count']},
        'emotion': {'file': os.path.join(norm,'zepp_emotion.csv'), 'cols':['label_mode','label_entropy','coverage']},
        'temperature': {'file': os.path.join(norm,'zepp_temperature.csv'), 'cols':['min','max','mean','count']}
    }

    aggs = {}
    inputs_sha = {}
    for k,v in metrics.items():
        path = v['file']
        if os.path.exists(path):
            rows, headers = read_csv(path)
            # aggregate
            if k == 'emotion':
                # simplistic: mode of value column
                by_date = defaultdict(list)
                for r in rows:
                    d = ts_to_date(r.get('timestamp_utc',''))
                    if not d:
                        continue
                    lbl = r.get('value','')
                    if lbl:
                        by_date[d].append(lbl)
                out = []
                for date in sorted(by_date.keys()):
                    cnts = Counter(by_date[date])
                    mode = sorted(cnts.items(), key=lambda x: (-x[1], x[0]))[0][0]
                    out.append({'date_utc': date, 'label_mode': mode, 'label_entropy': '', 'coverage': len(by_date[date])})
                aggs[k] = {'rows': out, 'cols': v['cols']}
            else:
                # numeric aggregate using value column
                by_date = defaultdict(list)
                for r in rows:
                    d = ts_to_date(r.get('timestamp_utc',''))
                    if not d:
                        continue
                    val = r.get('value','')
                    if val=='' or val is None:
                        continue
                    try:
                        by_date[d].append(float(val))
                    except Exception:
                        continue
                out = []
                for date in sorted(by_date.keys()):
                    s = stats_from_values(by_date[date]) if len(by_date[date])>0 else {'count':0,'min':'','max':'','mean':'','median':'','std':''}
                    out.append({'date_utc': date, 'min': s['min'], 'max': s['max'], 'mean': s['mean'], 'median': s['median'], 'std': s['std'], 'count': s['count']})
                aggs[k] = {'rows': out, 'cols': v['cols']}
        else:
            # write empty processed file
            outpath = os.path.join(proc, f'zepp_{k}_daily.csv')
            headers = ['date_utc'] + v['cols']
            write_csv_rows(outpath, [], headers)
            aggs[k] = {'rows': [], 'cols': v['cols']}
        inputs_sha[k] = sha256_of_file(v['file'])

    # write processed per-metric CSVs
    for k,v in aggs.items():
        outpath = os.path.join(proc, f'zepp_{k}_daily.csv')
        rows = v['rows']
        if rows:
            # map column names to appropriate headers
            if k == 'emotion':
                headers = ['date_utc','label_mode','label_entropy','coverage']
            elif k == 'temperature':
                headers = ['date_utc','min','max','mean','count']
            else:
                headers = ['date_utc','min','max','mean','median','std','count']
            write_csv_rows(outpath, rows, headers)
        else:
            # ensure header exists (already written above for missing files)
            if not os.path.exists(outpath):
                headers = ['date_utc'] + v['cols']
                write_csv_rows(outpath, [], headers)

    # augment AI features CSV
    total_rows, nan_pct_map, joined_count = augment_ai(ai, aggs, prefix='zepp_')

    # manifest
    manifest = {
        'participant': participant,
        'snapshot': snapshot,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'inputs': {k: {'path': metrics[k]['file'], 'sha256': inputs_sha.get(k)} for k in metrics.keys()},
        'ai_input': {'path': ai, 'sha256': sha256_of_file(ai), 'rows': total_rows},
        'outputs': {'processed_dir': proc},
        'join': {'joined_rows': joined_count, 'coverage_pct': (joined_count / max(1, len(aggs.get('hr',{}).get('rows',[]))) * 100.0) if aggs.get('hr') else 0.0},
        'qc': {'nan_pct': nan_pct_map, 'qc_mode': os.environ.get('QC_MODE','flag'), 'nan_threshold_pct': float(os.environ.get('NAN_THRESHOLD_PCT','5'))},
        'schema_version': '1.0'
    }

    manifest_out = os.path.join(joined, 'features_joined_zepp_manifest.json')
    atomic_write(manifest_out, json.dumps(manifest, sort_keys=True, indent=2))

    # QC enforcement
    max_nan = 0.0
    for v in nan_pct_map.values():
        if v > max_nan:
            max_nan = v
    qc_mode = os.environ.get('QC_MODE','flag')
    nan_threshold = float(os.environ.get('NAN_THRESHOLD_PCT','5'))
    if qc_mode == 'fail' and max_nan > nan_threshold:
        print('QC failed: max NaN% > threshold', max_nan, nan_threshold)
        sys.exit(2)

    print('Wrote processed zepp dailies in', proc)
    print('Augmented AI features file:', ai)
    print('Manifest:', manifest_out)


if __name__ == '__main__':
    main()
