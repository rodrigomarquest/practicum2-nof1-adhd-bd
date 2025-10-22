#!/usr/bin/env python3
"""Parse Apple Health export.xml into per-metric CSVs (timestamp,value).

Writes under the normalized dir:
 - apple_heart_rate.csv -> timestamp,bpm
 - apple_hrv_sdnn.csv -> timestamp,sdnn_ms
 - apple_sleep_intervals.csv -> start,end,stage
 - (optional) apple_state_of_mind.csv -> timestamp,mood_label,mood_score

Produces a small manifest JSON per file with count, min/max timestamps, sha256.

Usage: etl/apple_inapp_parse.py --participant P000001 --extracted-dir <path> --normalized-dir <path> --run-id <id> [--dry-run]
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

from make_scripts import io_utils
from etl_modules.common.progress import progress_bar

try:
    import dateutil.parser as _dp
    from dateutil import tz
except Exception:
    _dp = None
    tz = None


def _iso_z(dt: datetime) -> str:
    """Return ISO8601 Z-normalized string for datetime dt."""
    if tz is not None:
        return dt.astimezone(tz.tzutc()).isoformat().replace('+00:00', 'Z')
    return dt.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')


def parse_dt(s: str) -> datetime:
    """Parse datetime string into UTC-aware datetime (datetime object)."""
    if not s:
        raise ValueError('empty datetime')
    if _dp:
        dt = _dp.parse(s)
    else:
        dt = datetime.fromisoformat(s)
    # ensure tz-aware
    if dt.tzinfo is None:
        if tz is not None:
            dt = dt.replace(tzinfo=tz.tzutc())
        else:
            dt = dt.replace(tzinfo=timezone.utc)
    # return UTC-aware datetime
    if tz is not None:
        return dt.astimezone(tz.tzutc())
    return dt.astimezone(timezone.utc)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, rows: List[List[str]], headers: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(headers)
            for r in rows:
                w.writerow(r)
        # replace
        Path(tmp).replace(path)
    finally:
        if Path(tmp).exists():
            try:
                Path(tmp).unlink()
            except Exception:
                pass


def make_manifest(path: Path) -> Dict:
    sha = sha256_file(path)
    count = 0
    min_ts = None
    max_ts = None
    with path.open('r', encoding='utf-8') as f:
        r = csv.reader(f)
        try:
            hdr = next(r)
        except StopIteration:
            return {'count': 0, 'min_timestamp': None, 'max_timestamp': None, 'sha256': sha}
        # assume first column is timestamp or start
        for row in r:
            count += 1
            if not row:
                continue
            ts = row[0]
            try:
                # store as raw ISO strings for min/max
                if ts:
                    if min_ts is None or ts < min_ts:
                        min_ts = ts
                    if max_ts is None or ts > max_ts:
                        max_ts = ts
            except Exception:
                pass
    return {'count': count, 'min_timestamp': min_ts, 'max_timestamp': max_ts, 'sha256': sha}


def process_export(extracted_dir: Path, normalized_dir: Path, dry_run: bool = False, participant: str = '', run_id: str = '') -> None:
    # purge stale tmp files older than 24h in normalized_dir
    try:
        now = time.time()
        for f in normalized_dir.glob('.*.tmp.*'):
            try:
                mtime = f.stat().st_mtime
                if now - mtime > 24 * 3600:
                    f.unlink()
                    print(f'Removed stale tmp file: {f}')
            except Exception:
                pass
    except Exception:
        pass

    # find export.xml under extracted_dir/apple or similar (deterministic sort)
    candidates = sorted(extracted_dir.rglob('export.xml'), key=lambda p: str(p))
    if not candidates:
        print(f'No export.xml found under {extracted_dir}', file=sys.stderr)
        return
    xmlp = candidates[0]
    print(f'Parsing {xmlp} ...')
    tree = ET.parse(str(xmlp))
    root = tree.getroot()

    # compute canonical source sha
    src_sha = io_utils.sha256_of_file(xmlp)
    # build stable/canonical params (resolve paths and use posix separators)
    extracted_canon = Path(extracted_dir).resolve().as_posix()
    normalized_canon = Path(normalized_dir).resolve().as_posix()
    participant_id = participant or ''
    params = {
        'participant': participant_id,
        'extracted_dir': extracted_canon,
        'normalized_dir': normalized_canon,
    }
    # canonical script id (resolved path)
    script_id = Path(__file__).resolve().as_posix()
    # idempotency key for this run
    idempotency_key = io_utils.idempotency_key(src_sha, {'script': script_id, 'params': params, 'schema': io_utils.SCHEMA_VERSION})

    # stage marker path
    stage_name = 'parse'
    ok_path = normalized_dir / f'{stage_name}.ok'

    # If an .ok exists, validate its outputs and idempotency key. If the ok
    # record is identical (same idempotency_key and same outputs list with
    # matching sha256 and rows), consider the stage up-to-date and return.
    if ok_path.exists():
        try:
            okj = json.loads(ok_path.read_text(encoding='utf-8'))
            ok_key = okj.get('idempotency_key')
            ok_outputs = okj.get('outputs', [])
            # If the existing ok contains outputs that match the current file
            # content (sha256 and rows), consider the stage up-to-date even if
            # the idempotency_key changed due to e.g. canonicalization tweaks.
            valid = True
            for out in ok_outputs:
                p = Path(out.get('path'))
                sha = out.get('sha256')
                rows = out.get('rows')
                if not p.exists():
                    valid = False
                    break
                try:
                    cur = io_utils.sha256_of_file(p)
                    if cur != sha:
                        valid = False
                        break
                    # try to read manifest to get up-to-date rows if present
                    mpath = p.with_name(p.name + '.manifest.json')
                    if mpath.exists():
                        try:
                            m = json.loads(mpath.read_text(encoding='utf-8'))
                            if 'rows' in m and rows is not None and m.get('rows') != rows:
                                valid = False
                                break
                        except Exception:
                            # unreadable manifest -> not valid
                            valid = False
                            break
                except Exception:
                    valid = False
                    break
            if valid:
                print(f'STAGE UP-TO-DATE: {stage_name} (ok marker valid)')
                return
        except Exception:
            pass
        # otherwise, leave the existing ok in place for inspection but note
        # it's invalid and will be replaced after successful write of outputs.
        print(f'Existing stage marker invalid or changed: {ok_path}')

    # collect rows per file
    hr_rows: Dict[Tuple[str, str], Tuple[str, List[str]]] = {}  # key (timestamp,value) -> (sourceVersion, line)
    hrv_rows: Dict[Tuple[str, str], str] = {}
    sleep_rows: List[List[str]] = []
    state_rows: List[List[str]] = []
    # track units
    units_map: Dict[str, Optional[str]] = {
        'heart_rate': None,
        'hrv_sdnn': None,
        'sleep': None,
        'state': None,
    }

    records = list(root.findall('Record'))
    with progress_bar(total=len(records), desc='Parsing Records') as _bar:
        for rec in records:
            rtype = rec.get('type')
            start = rec.get('startDate')
            end = rec.get('endDate')
            val = rec.get('value')
            unit = rec.get('unit')
            src = rec.get('sourceName')
            srcver = rec.get('sourceVersion') or ''
            if not rtype or not start:
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue
            try:
                dt_start = parse_dt(start)
            except Exception:
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue
            if end:
                try:
                    dt_end = parse_dt(end)
                except Exception:
                    dt_end = dt_start
            else:
                dt_end = dt_start

            # timestamp for instants: midpoint
            if dt_start == dt_end:
                ts = _iso_z(dt_start)
            else:
                mid = dt_start + (dt_end - dt_start) / 2
                ts = _iso_z(mid)

            # heart rate
            if rtype and (rtype.endswith('heart_rate') or 'HeartRate' in rtype or rtype.lower().endswith('heart_rate')):
                if not val:
                    try:
                        _bar.update(1)
                    except Exception:
                        pass
                    continue
                key = (ts, val)
                # dedupe: keep earliest by sourceVersion (lexicographic)
                prev = hr_rows.get(key)
                if prev is None or (srcver and srcver < prev[0]):
                    hr_rows[key] = (srcver, [ts, val])
                    units_map['heart_rate'] = unit or units_map['heart_rate']
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue

            # hrv sdnn
            if rtype and ('SDNN' in rtype or rtype.lower().endswith('hrv') or 'sdnn' in rtype.lower()):
                if not val:
                    try:
                        _bar.update(1)
                    except Exception:
                        pass
                    continue
                hrv_rows[(ts, val)] = val
                units_map['hrv_sdnn'] = unit or units_map['hrv_sdnn']
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue

            # sleep
            if rtype and ('Sleep' in rtype or rtype.lower().startswith('sleep')):
                # rawValue might be '0' '1' etc in rec.attrib
                raw = rec.get('value') or rec.get('rawValue') or ''
                sleep_rows.append([_iso_z(dt_start), _iso_z(dt_end), raw])
                units_map['sleep'] = unit or units_map['sleep']
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue

            # state of mind / mood
            if rtype and ('StateOfMind' in rtype or 'mind' in rtype.lower()):
                # guess mapping
                mood = rec.get('value') or rec.get('rawValue') or ''
                state_rows.append([ts, mood, ''])
                units_map['state'] = unit or units_map['state']
                try:
                    _bar.update(1)
                except Exception:
                    pass
                continue

    # build output rows from dicts
    hr_out = [v for (_, v) in sorted(hr_rows.items(), key=lambda kv: kv[0])]  # deterministic order
    hr_out = [item[1] for item in hr_out]
    hrv_out = [[k[0], v] for k, v in sorted(hrv_rows.items(), key=lambda kv: kv[0])]

    # write files
    targets = []
    if hr_out:
        p = normalized_dir / 'apple_heart_rate.csv'
        targets.append((p, hr_out, ['timestamp', 'bpm']))
    if hrv_out:
        p = normalized_dir / 'apple_hrv_sdnn.csv'
        targets.append((p, hrv_out, ['timestamp', 'sdnn_ms']))
    if sleep_rows:
        p = normalized_dir / 'apple_sleep_intervals.csv'
        targets.append((p, sleep_rows, ['start', 'end', 'stage']))
    if state_rows:
        p = normalized_dir / 'apple_state_of_mind.csv'
        targets.append((p, state_rows, ['timestamp', 'mood_label', 'mood_score']))

    # short-circuit: if all outputs have manifest with matching idempotency_key -> up-to-date
    all_up_to_date = True
    need_write = []
    for path, rows, headers in targets:
        manifest_path = path.with_name(path.name + '.manifest.json')
        if manifest_path.exists():
            try:
                m = json.loads(manifest_path.read_text(encoding='utf-8'))
                if m.get('idempotency_key') == idempotency_key:
                    print(f'UP-TO-DATE: {path} (idempotency match)')
                    continue
            except Exception:
                pass
        all_up_to_date = False
        need_write.append((path, rows, headers))

    if all_up_to_date:
        print('All outputs up-to-date; nothing to do.')
        return

    # dry-run: just report which files would be written
    if dry_run:
        for t in need_write:
            print(f"DRY RUN: would write {t[0]} rows={len(t[1])}")
        return

    # write needed outputs with locks
    outputs_info: List[Dict[str, Optional[str]]] = []
    for path, rows, headers in need_write:
        lock_path = path.with_suffix(path.suffix + '.lock')
        with io_utils.with_lock(lock_path):
            # Ensure deterministic ordering and formatting before writing.
            # Sort rows by primary time key (first column) and format numeric
            # columns consistently using 6 decimal places where applicable.
            try:
                sorted_rows = sorted(rows, key=lambda r: (r[0] if r else ''))
            except Exception:
                sorted_rows = list(rows)

            # headers is an iterable; normalize to list for indexing
            hdrs = list(headers)
            # decide which headers should be formatted as floats
            float_headers = set(h for h in hdrs if h in ('bpm', 'sdnn_ms'))

            formatted_rows = []
            for r in sorted_rows:
                out_row = []
                for i, val in enumerate(r):
                    h = hdrs[i] if i < len(hdrs) else None
                    if h in float_headers:
                        try:
                            f = float(val)
                            out_row.append(f"{f:.6f}")
                        except Exception:
                            out_row.append(val if val is not None else '')
                    else:
                        out_row.append(val if val is not None else '')
                formatted_rows.append(out_row)

            # write atomically using io_utils
            io_utils.write_atomic_csv(path, formatted_rows, headers)
            # manifest metadata
            meta = {
                'source': str(xmlp),
                'source_sha256': src_sha,
                'idempotency_key': idempotency_key,
                'units_in': units_map.get('heart_rate') if 'heart_rate' in path.name else None,
                'units_out': None,
            }
            # add metric-specific units
            if 'heart_rate' in path.name:
                meta['units_in'] = units_map.get('heart_rate')
                meta['units_out'] = 'bpm'
            if 'hrv' in path.name:
                meta['units_in'] = units_map.get('hrv_sdnn')
                meta['units_out'] = 'ms'
            if 'sleep' in path.name:
                meta['units_in'] = units_map.get('sleep')
            if 'state_of_mind' in path.name:
                meta['units_in'] = units_map.get('state')
            # include schema version in meta so schema bumps participate in idempotency
            meta['schema_version'] = io_utils.SCHEMA_VERSION
            m = io_utils.manifest(path, meta)
            print(f'Wrote {path} ({m.get("rows")} rows)')
            # collect outputs info for stage marker
            try:
                outputs_info.append({
                    'path': str(path),
                    'sha256': m.get('sha256'),
                    'rows': m.get('rows'),
                })
            except Exception:
                outputs_info.append({'path': str(path), 'sha256': None, 'rows': None})
    
    # After successful writes, write atomic stage marker `<stage>.ok`
    if outputs_info:
        if not dry_run:
            ok_obj = {
                'run_id': run_id or '',
                'generated_at_utc': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
                'idempotency_key': idempotency_key,
                'outputs': outputs_info,
            }
            try:
                io_utils.write_atomic_text(ok_path, json.dumps(ok_obj, ensure_ascii=False, indent=2))
                print(f'Wrote stage marker: {ok_path}')
            except Exception as e:
                print(f'Warning: failed to write stage marker {ok_path}: {e}')
        else:
            print('DRY RUN: would write stage marker parse.ok')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--extracted-dir', required=True)
    p.add_argument('--normalized-dir', required=True)
    p.add_argument('--run-id', default='')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)

    extracted = Path(args.extracted_dir)
    normalized = Path(args.normalized_dir)
    process_export(extracted, normalized, dry_run=args.dry_run, participant=args.participant, run_id=args.run_id)


if __name__ == '__main__':
    sys.exit(main())
