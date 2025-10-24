#!/usr/bin/env python3
"""Normalize Zepp CSVs from archive into long per-metric CSVs with manifests.

Writes header-only CSVs if inputs missing; creates per-file manifests and overall outputs.
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from zipfile import ZipFile, is_zipfile, BadZipFile
try:
    import pyzipper
except Exception:
    pyzipper = None


def atomic_write(path, text, mode='w', encoding='utf-8'):
    tmp = path + '.tmp'
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(tmp, mode, encoding=encoding) as fh:
        fh.write(text)
    os.replace(tmp, path)


def sha256_bytes(b):
    import hashlib
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def sha256_fileobj(fobj):
    h = hashlib.sha256()
    for chunk in iter(lambda: fobj.read(8192), b''):
        h.update(chunk)
    fobj.seek(0)
    return h.hexdigest()


def parse_csv_from_bytes(b):
    try:
        s = b.decode('utf-8')
    except Exception:
        try:
            s = b.decode('latin1')
        except Exception:
            s = ''
    if not s:
        return [], []
    reader = csv.DictReader(io.StringIO(s))
    rows_raw = list(reader)
    # clean fieldnames (strip BOM and whitespace)
    fieldnames = [fn.lstrip('\ufeff').strip() if isinstance(fn, str) else fn for fn in (reader.fieldnames or [])]
    # normalize row keys to cleaned fieldnames
    rows = []
    for r in rows_raw:
        new = {}
        for k, v in r.items():
            if isinstance(k, str):
                kn = k.lstrip('\ufeff').strip()
            else:
                kn = k
            new[kn] = v
        rows.append(new)
    return rows, fieldnames


def normalize_rows_for_metric(rows, headers, metric):
    # Heuristic mapping: find timestamp and value-like columns
    # normalize header BOM if present
    headers = [(h.lstrip('\ufeff') if isinstance(h, str) else h) for h in headers]
    ts_cols = [c for c in headers if 'timestamp' in c.lower() or c.lower() in ('ts', 'datetime')]
    # include 'time' as candidate, but prefer full timestamp-like names
    ts_cols += [c for c in headers if 'time' in c.lower() and c not in ts_cols]
    val_cols = [c for c in headers if 'value' in c.lower() or 'bpm' in c.lower() or 'score' in c.lower() or 'temp' in c.lower() or 'hrv' in c.lower()]
    out = []
    for r in rows:
        ts = None
        # If both date and time exist, combine them
        date_key = next((c for c in headers if c.lower().lstrip('\ufeff') == 'date'), None)
        time_key = next((c for c in headers if c.lower() == 'time'), None)
        if date_key and time_key and r.get(date_key) and r.get(time_key):
            ts = f"{r.get(date_key)} {r.get(time_key)}"
        else:
            for c in ts_cols:
                v = r.get(c, '')
                if v:
                    ts = v
                    break
        val = None
        for c in val_cols:
            if c in r and r.get(c,'')!='':
                val = r.get(c)
                break
        # fallback: any numeric column
        if val is None:
            for c in headers:
                try:
                    if r.get(c) is None:
                        continue
                    float(r.get(c))
                    val = r.get(c)
                    break
                except Exception:
                    continue
        if not ts:
            continue
        # normalize timestamp; assume ISO-like or epoch
        tnorm = None
        try:
            # epoch seconds
            if ts.isdigit():
                tnorm = datetime.fromtimestamp(int(ts), tz=timezone.utc).isoformat()
            else:
                # try RFC/ISO
                tnorm = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(timezone.utc).isoformat()
        except Exception:
            try:
                # try common 'YYYY-MM-DD HH:MM[:SS][+ZZZZ]' formats
                # remove microseconds
                tnorm = datetime.strptime(ts.split('+')[0].strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                try:
                    tnorm = datetime.strptime(ts.split('+')[0].strip(), '%Y-%m-%d %H:%M').replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    # fallback parse date part
                    try:
                        tnorm = datetime.strptime(ts.split('.')[0], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=timezone.utc).isoformat()
                    except Exception:
                        tnorm = None
        if tnorm is None:
            continue
        out.append({'timestamp_utc': tnorm, 'value': val or '', 'unit': '', 'source': 'zepp', 'metric': metric})
    return out


def write_normalized(path, rows, headers):
    # deterministic column order
    with open(path + '.tmp', 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            row = {k: ('' if r.get(k) is None else r.get(k)) for k in headers}
            writer.writerow(row)
    os.replace(path + '.tmp', path)


def make_manifest(path, inputs, outputs, extra=None):
    manifest = {
        'schema_version': '1.0',
        'inputs': inputs,
        'outputs': outputs,
        'producer': 'make_scripts/zepp/zepp_normalize.py',
        'args': {},
        'system': {},
    }
    if extra:
        manifest.update(extra)
    return json.dumps(manifest, sort_keys=True, indent=2)


def process_metric_from_zip(z, member_path, metric, normalized_dir, participant, snapshot, dry_run=False):
    # read bytes
    try:
        with z.open(member_path) as fh:
            b = fh.read()
    except RuntimeError:
        # encrypted member; try pyzipper if available and password provided via env
        pwd = os.environ.get('ZEPP_ZIP_PASSWORD')
        if pyzipper is not None and pwd:
            try:
                with pyzipper.AESZipFile(z.filename, 'r') as az:
                    az.pwd = pwd.encode('utf-8') if isinstance(pwd, str) else pwd
                    b = az.read(member_path)
            except Exception:
                b = b''
        else:
            b = b''
    except Exception:
        b = b''
    rows, headers = parse_csv_from_bytes(b)
    normalized_rows = normalize_rows_for_metric(rows, headers, metric)
    out_path = os.path.join(normalized_dir, f'zepp_{metric}.csv')
    manifest_path = os.path.join(normalized_dir, f'zepp_{metric}_manifest.json')

    # QC fields
    # Normalize timestamps to always use Z suffix and deduplicate by (timestamp_utc, metric, source)
    for r in normalized_rows:
        # ensure UTC Z suffix
        t = r.get('timestamp_utc')
        if isinstance(t, str) and t.endswith('+00:00'):
            r['timestamp_utc'] = t.replace('+00:00', 'Z')
        elif isinstance(t, str) and t.endswith('+0000'):
            r['timestamp_utc'] = t[:-5] + 'Z'

    seen = set()
    deduped = []
    dup_count = 0
    for r in normalized_rows:
        key = (r.get('timestamp_utc'), r.get('metric'), r.get('source'))
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        deduped.append(r)

    # sort deterministically by timestamp_utc then metric then source
    try:
        deduped.sort(key=lambda x: (x.get('timestamp_utc') or '', x.get('metric') or '', x.get('source') or ''))
    except Exception:
        pass

    rows_count = len(deduped)
    null_pct = 0.0
    dup_pct = 0.0
    date_min = ''
    date_max = ''
    if rows_count > 0:
        dates = [r['timestamp_utc'] for r in deduped]
        date_min = min(dates)
        date_max = max(dates)
        nulls = sum(1 for r in deduped if r.get('value','') == '')
        null_pct = nulls / rows_count * 100.0
        dup_pct = dup_count / (rows_count + dup_count) * 100.0 if (rows_count + dup_count) > 0 else 0.0

    headers_out = ['timestamp_utc','value','unit','source','metric']
    if dry_run:
        print('DRY_RUN: would write', out_path, 'rows=', rows_count)
    else:
        write_normalized(out_path, deduped, headers_out)
    # compute input sha
    try:
        inp_sha = hashlib.sha256(b).hexdigest() if b else None
    except Exception:
        inp_sha = None
    out_sha = None
    if not dry_run and os.path.exists(out_path):
        with open(out_path,'rb') as fh:
            out_sha = hashlib.sha256(fh.read()).hexdigest()

    inputs = [{'path': member_path, 'sha256': inp_sha, 'rows': len(rows)}]
    outputs = [{'path': out_path, 'sha256': out_sha, 'rows': rows_count}]
    extra = {'rows': rows_count, 'null_pct': null_pct, 'dup_pct': dup_pct, 'date_min': date_min, 'date_max': date_max, 'participant': participant, 'snapshot_date': snapshot}
    manifest_text = make_manifest(os.path.basename(manifest_path), inputs, outputs, extra=extra)
    if dry_run:
        print('DRY_RUN: would write manifest', manifest_path)
    else:
        atomic_write(manifest_path, manifest_text)
    return out_path, manifest_path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--zip-path', required=True)
    p.add_argument('--filelist-tsv', required=False)
    p.add_argument('--normalized-dir', required=True)
    p.add_argument('--participant', required=False)
    p.add_argument('--snapshot', required=False)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    zpath = args.zip_path
    filelist = args.filelist_tsv
    normalized_dir = args.normalized_dir
    participant = args.participant or ''
    snapshot = args.snapshot or ''
    dry_run = args.dry_run or (os.environ.get('DRY_RUN','0')=='1')

    # ensure output dir
    if not os.path.exists(normalized_dir) and not dry_run:
        os.makedirs(normalized_dir, exist_ok=True)

    # expected metrics and their candidate filenames
    # keyword-based detection (case-insensitive substring match)
    metrics = {
        'hr': ['heartrate','heart_rate','heart-rate','hr'],
        'hrv': ['hrv','sdnn','rmssd'],
        'sleep': ['sleep','sleep_stage','sleep-stage'],
        'emotion': ['emotion','emoti'],
        'temperature': ['temperature','skin_temp','temp']
    }

    members = []
    if os.path.exists(filelist):
        with open(filelist,'r',encoding='utf-8') as fh:
            rdr = csv.reader(fh, delimiter='\t')
            hdr = next(rdr, None)
            for row in rdr:
                if row:
                    members.append(row[0])
    # if zip exists, prefer to stream from it
    if not os.path.exists(zpath):
        # no zip: create header-only outputs for each metric and manifests
        for metric in metrics.keys():
            out_path = os.path.join(normalized_dir, f'zepp_{metric}.csv')
            manifest_path = os.path.join(normalized_dir, f'zepp_{metric}_manifest.json')
            headers_out = ['timestamp_utc','value','unit','source','metric']
            if dry_run:
                print('DRY_RUN: would write', out_path)
            else:
                write_normalized(out_path, [], headers_out)
            manifest_text = make_manifest(manifest_path, [], [{'path': out_path, 'sha256': None, 'rows': 0}], extra={'participant': participant, 'snapshot_date': snapshot})
            if dry_run:
                print('DRY_RUN: would write manifest', manifest_path)
            else:
                atomic_write(manifest_path, manifest_text)
        print('No zip found; wrote header-only normalized files and manifests')
        return

    # open zip and process candidate files
    try:
        if not is_zipfile(zpath):
            print('Not a zip file:', zpath, file=sys.stderr)
            sys.exit(2)
        z = ZipFile(zpath, 'r')
    except BadZipFile:
        print('Bad zip file:', zpath, file=sys.stderr)
        sys.exit(2)

    # build mapping of available members
    avail = {info.filename: info for info in z.infolist()}

    for metric, candidates in metrics.items():
        found = None
        for kw in candidates:
            kw = kw.lower()
            for m in avail.keys():
                if kw in m.lower():
                    found = m
                    break
            if found:
                break
        if found:
            out_path, manifest_path = process_metric_from_zip(z, found, metric, normalized_dir, participant, snapshot, dry_run=dry_run)
        else:
            # write empty CSV + manifest
            out_path = os.path.join(normalized_dir, f'zepp_{metric}.csv')
            manifest_path = os.path.join(normalized_dir, f'zepp_{metric}_manifest.json')
            headers_out = ['timestamp_utc','value','unit','source','metric']
            if dry_run:
                print('DRY_RUN: would write empty', out_path)
            else:
                write_normalized(out_path, [], headers_out)
            manifest_text = make_manifest(manifest_path, [], [{'path': out_path, 'sha256': None, 'rows': 0}], extra={'participant': participant, 'snapshot_date': snapshot})
            if dry_run:
                print('DRY_RUN: would write manifest', manifest_path)
            else:
                atomic_write(manifest_path, manifest_text)

    print('Zepp normalization completed')


if __name__ == '__main__':
    main()
