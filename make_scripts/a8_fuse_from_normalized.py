#!/usr/bin/env python3
"""Build joined/features_daily.csv from normalized/ when processed/*_daily.csv are empty.

Lightweight, no new dependencies beyond pandas/numpy.
"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import json
from datetime import datetime
import sys


def find_col(df, candidates):
    for c in df.columns:
        lc = c.lower()
        for pat in candidates:
            if pat in lc:
                return c
    return None


def to_date_series(df, col, tz):
    # parse, coerce to UTC if possible then convert to tz and take date
    s = pd.to_datetime(df[col], utc=True, errors='coerce')
    # if all NaT, try without utc
    if s.isna().all():
        s = pd.to_datetime(df[col], errors='coerce')
        # if timezone-naive, localize to UTC then convert
        if s.dt.tz is None:
            try:
                s = s.dt.tz_localize('UTC')
            except Exception:
                pass
    try:
        s = s.dt.tz_convert(tz)
    except Exception:
        pass
    return s.dt.date


def agg_hr(path, tz):
    # chunked mean aggregation to avoid loading large files
    try:
        hdr = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    ts = find_col(hdr, ['timestamp', 'time', 'ts', 'date'])
    val = find_col(hdr, ['hr', 'heart', 'bpm', 'value', 'mean'])
    if ts is None or val is None:
        return None
    sums = {}
    counts = {}
    for chunk in pd.read_csv(path, usecols=[ts, val], chunksize=200000):
        dates = to_date_series(chunk, ts, tz)
        vals = pd.to_numeric(chunk[val], errors='coerce')
        chunk = pd.DataFrame({'date': dates, 'val': vals})
        grp = chunk.groupby('date', dropna=True)['val'].agg(['sum', 'count'])
        for d, row in grp.iterrows():
            if pd.isna(d):
                continue
            s = float(row['sum'])
            c = int(row['count'])
            sums[d] = sums.get(d, 0.0) + s
            counts[d] = counts.get(d, 0) + c
    if not sums:
        return None
    rows = []
    for d in sorted(sums.keys()):
        rows.append({'date': d, 'hr_mean': sums[d] / counts[d]})
    return pd.DataFrame(rows)


def agg_hrv(path, tz):
    try:
        hdr = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    ts = find_col(hdr, ['timestamp', 'time', 'ts', 'date'])
    val_sdnn = find_col(hdr, ['sdnn', 'sdn'])
    val_rmssd = find_col(hdr, ['rmssd'])
    val = val_sdnn or val_rmssd
    if ts is None or val is None:
        return None
    sums = {}
    counts = {}
    for chunk in pd.read_csv(path, usecols=[ts, val], chunksize=200000):
        dates = to_date_series(chunk, ts, tz)
        vals = pd.to_numeric(chunk[val], errors='coerce')
        chunk = pd.DataFrame({'date': dates, 'val': vals})
        grp = chunk.groupby('date', dropna=True)['val'].agg(['sum', 'count'])
        for d, row in grp.iterrows():
            if pd.isna(d):
                continue
            s = float(row['sum'])
            c = int(row['count'])
            sums[d] = sums.get(d, 0.0) + s
            counts[d] = counts.get(d, 0) + c
    if not sums:
        return None
    rows = []
    name = 'hrv_sdnn_mean' if val_sdnn else 'hrv_rmssd_mean'
    for d in sorted(sums.keys()):
        rows.append({'date': d, name: sums[d] / counts[d]})
    return pd.DataFrame(rows)


def agg_sleep(path, tz):
    # sleep sessions may be small; read whole file
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    # find duration column
    dur = find_col(df, ['duration', 'minutes', 'mins', 'min', 'seconds', 'secs'])
    start = find_col(df, ['start', 'begin', 'from'])
    end = find_col(df, ['end', 'stop', 'to'])
    if dur is not None:
        # assume minutes unless name suggests hours
        dcol = dur
        s = pd.to_numeric(df[dcol], errors='coerce')
        # if values small (<10) maybe hours; but we won't convert heuristically
        # try to group by date from start if present else use index
        if start is not None:
            dates = to_date_series(df, start, tz)
        else:
            # create a date index from position (not ideal)
            dates = pd.Series([pd.NaT]*len(df))
        res = pd.DataFrame({'date': dates, 'dur_min': s})
        g = res.groupby('date', dropna=True)['dur_min'].sum().reset_index()
        g = g.rename(columns={'dur_min': 'sleep_total_min'})
        return g
    elif start is not None and end is not None:
        starts = to_date_series(df, start, tz)
        ends = pd.to_datetime(df[end], utc=True, errors='coerce')
        if ends.dt.tz is None:
            try:
                ends = ends.dt.tz_localize('UTC')
            except Exception:
                pass
        try:
            ends = ends.dt.tz_convert(tz)
        except Exception:
            pass
        ends = ends.dt.date
        res = pd.DataFrame({'date': starts, 'end': ends})
        # can't compute duration without precise times; fallback to count of sessions as minutes=0
        # better: if start and end include time, compute diff
        try:
            s_ts = pd.to_datetime(df[start], utc=True, errors='coerce')
            e_ts = pd.to_datetime(df[end], utc=True, errors='coerce')
            if s_ts.dt.tz is None:
                try:
                    s_ts = s_ts.dt.tz_localize('UTC')
                    e_ts = e_ts.dt.tz_localize('UTC')
                except Exception:
                    pass
            try:
                s_ts = s_ts.dt.tz_convert(tz)
                e_ts = e_ts.dt.tz_convert(tz)
            except Exception:
                pass
            dur_min = (e_ts - s_ts).dt.total_seconds() / 60.0
            res = pd.DataFrame({'date': s_ts.dt.date, 'dur_min': dur_min})
            g = res.groupby('date', dropna=True)['dur_min'].sum().reset_index()
            g = g.rename(columns={'dur_min': 'sleep_total_min'})
            return g
        except Exception:
            return None
    return None


def agg_usage(path, tz):
    try:
        hdr = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    ts = find_col(hdr, ['timestamp', 'time', 'ts', 'date'])
    val = find_col(hdr, ['min', 'minutes', 'usage', 'duration'])
    if ts is None or val is None:
        return None
    sums = {}
    for chunk in pd.read_csv(path, usecols=[ts, val], chunksize=200000):
        dates = to_date_series(chunk, ts, tz)
        vals = pd.to_numeric(chunk[val], errors='coerce')
        chunk = pd.DataFrame({'date': dates, 'val': vals})
        grp = chunk.groupby('date', dropna=True)['val'].sum()
        for d, s in grp.items():
            if pd.isna(d):
                continue
            sums[d] = sums.get(d, 0.0) + float(s)
    if not sums:
        return None
    rows = [{'date': d, 'usage_total_min': sums[d]} for d in sorted(sums.keys())]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--out', required=False)
    parser.add_argument('--tz', default='Europe/Dublin')
    args = parser.parse_args()

    snap = Path(args.snapshot)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = snap / 'joined' / 'features_daily.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # possible normalized inputs
    norm = snap / 'normalized'
    candidates = {
        'apple_hr': norm / 'apple_heart_rate.csv',
        'apple_hrv_sdnn': norm / 'apple_hrv_sdnn.csv',
        'apple_hrv_rmssd': norm / 'apple_hrv_rmssd.csv',
        'apple_sleep': norm / 'apple_sleep_sessions.csv',
        'ios_usage': norm / 'ios_app_usage.csv',
        'zepp_hr': norm / 'zepp' / 'zepp_hr.csv',
        'zepp_hrv': norm / 'zepp' / 'zepp_hrv.csv',
        'zepp_sleep': norm / 'zepp' / 'zepp_sleep.csv',
    }

    sources_used = []
    dfs = []

    missing = [str(p) for p in candidates.values() if not p.exists()]

    # HR: prefer apple, else zepp
    if candidates['apple_hr'].exists():
        g = agg_hr(candidates['apple_hr'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['apple_hr']))
    elif candidates['zepp_hr'].exists():
        g = agg_hr(candidates['zepp_hr'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['zepp_hr']))

    # HRV: prefer sdnn then rmssd then zepp
    if candidates['apple_hrv_sdnn'].exists():
        g = agg_hrv(candidates['apple_hrv_sdnn'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['apple_hrv_sdnn']))
    elif candidates['apple_hrv_rmssd'].exists():
        g = agg_hrv(candidates['apple_hrv_rmssd'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['apple_hrv_rmssd']))
    elif candidates['zepp_hrv'].exists():
        g = agg_hrv(candidates['zepp_hrv'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['zepp_hrv']))

    # Sleep
    if candidates['apple_sleep'].exists():
        g = agg_sleep(candidates['apple_sleep'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['apple_sleep']))
    elif candidates['zepp_sleep'].exists():
        g = agg_sleep(candidates['zepp_sleep'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['zepp_sleep']))

    # Usage
    if candidates['ios_usage'].exists():
        g = agg_usage(candidates['ios_usage'], args.tz)
        if g is not None and not g.empty:
            dfs.append(g); sources_used.append(str(candidates['ios_usage']))

    if not dfs:
        # nothing could be derived
        print('NEED:')
        for p in candidates.values():
            if not p.exists():
                print(f'- {p}')
        sys.exit(0)

    # merge outer on date sequentially
    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on='date', how='outer')

    # collapse duplicate dates
    if out['date'].duplicated().any():
        agg = {}
        for c in out.columns:
            if c == 'date':
                continue
            if pd.api.types.is_numeric_dtype(out[c]):
                agg[c] = 'mean'
            else:
                agg[c] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else np.nan
        out = out.groupby('date', as_index=False).agg(agg)

    # drop sparse or constant columns
    cols = [c for c in out.columns if c != 'date']
    keep = []
    for c in cols:
        s = out[c]
        if s.isna().mean() > 0.9:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    out = out[['date'] + keep]

    # rename common variants to canonical names
    renmap = {}
    for c in out.columns:
        lc = c.lower()
        if 'hr_mean' in lc or lc.startswith('hr') or 'heart' in lc:
            renmap[c] = 'hr_mean'
        if 'hrv_sdnn' in lc:
            renmap[c] = 'hrv_sdnn_mean'
        if 'hrv_rmssd' in lc:
            renmap[c] = 'hrv_rmssd_mean'
        if 'sleep' in lc and 'min' in lc:
            renmap[c] = 'sleep_total_min'
        if 'sleep' in lc and 'sum' in lc:
            renmap[c] = 'sleep_total_min'
        if 'usage' in lc or 'minutes' in lc:
            renmap[c] = 'usage_total_min'
    if renmap:
        out = out.rename(columns=renmap)

    out = out.sort_values('date').reset_index(drop=True)
    # write date as ISO
    out['date'] = out['date'].apply(lambda d: pd.to_datetime(d).date().isoformat() if pd.notna(d) else '')
    out.to_csv(out_path, index=False)

    # provenance sidecar
    side = out_path.parent / 'features_daily_source.json'
    meta = {'sources_used': sources_used, 'generated_at': datetime.utcnow().isoformat()+'Z', 'tz': args.tz}
    side.write_text(json.dumps(meta, indent=None))

    rows = len(out)
    cols = len(out.columns)
    ready = (rows >= 2) and (cols >= 4)
    sources_json = json.dumps(sources_used)
    print(f'A8-BUILD: PATH="{out_path.resolve()}" rows={rows} cols={cols} sources_used={sources_json} READY={str(bool(ready)).lower()}')


if __name__ == '__main__':
    main()
