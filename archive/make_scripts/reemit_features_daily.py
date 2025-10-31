#!/usr/bin/env python3
"""Minimal re-emitter for features_daily.csv
Merges processed daily tables (apple/ios/zepp) by date and writes joined/features_daily.csv
"""
from pathlib import Path
import pandas as pd
import numpy as np
import sys


def read_if(path: Path):
    try:
        if path.exists():
            df = pd.read_csv(path)
            return df
    except Exception:
        return None
    return None


def main():
    repo = Path('.')
    snap = repo / 'data' / 'etl' / 'P000001' / 'snapshots' / '2025-10-22'
    processed = snap / 'processed'
    joined = snap / 'joined'
    joined.mkdir(parents=True, exist_ok=True)

    expected = [
        processed / 'apple' / 'health_hr_daily.csv',
        processed / 'apple' / 'health_hrv_sdnn_daily.csv',
        processed / 'apple' / 'health_sleep_daily.csv',
        processed / 'ios' / 'ios_usage_daily.csv',
        processed / 'zepp' / 'zepp_hr_daily.csv',
        processed / 'zepp' / 'zepp_hrv_daily.csv',
        processed / 'zepp' / 'zepp_sleep_daily.csv',
    ]

    present = []
    missing = []
    dfs = []
    # helper to map columns
    def map_columns(fname, df):
        fmap = {}
        lname = fname.lower()
        # HR
        if 'hr' in lname and 'hrv' not in lname and 'zepp' not in lname:
            # look for hr_mean or mean
            for c in df.columns:
                if 'mean' in c.lower() and 'hr' in c.lower():
                    fmap[c] = 'hr_mean'
                    return fmap
            for c in df.columns:
                if 'mean' in c.lower():
                    fmap[c] = 'hr_mean'; return fmap
        # HRV
        if 'hrv' in lname or 'hrv' in fname.lower():
            for c in df.columns:
                if 'sdn' in c.lower() or 'sdnn' in c.lower() or 'mean' in c.lower() and 'hrv' in c.lower():
                    fmap[c] = 'hrv_sdnn_mean'; return fmap
            for c in df.columns:
                if 'mean' in c.lower():
                    fmap[c] = 'hrv_sdnn_mean'; return fmap
        # Zepp HR (normalized naming)
        if 'zepp_hr' in fname.lower() or ('zepp' in fname.lower() and 'hr' in fname.lower()):
            for c in df.columns:
                if 'mean' in c.lower():
                    fmap[c] = 'hr_mean'; return fmap
        # Sleep
        if 'sleep' in fname.lower():
            for c in df.columns:
                if 'sum' in c.lower() or 'total' in c.lower():
                    # if hours, convert later
                    fmap[c] = 'sleep_total_min'; return fmap
            for c in df.columns:
                if 'mean' in c.lower():
                    fmap[c] = 'sleep_total_min'; return fmap
        # Usage
        if 'usage' in fname.lower() or 'activity' in fname.lower():
            for c in df.columns:
                if 'usage' in c.lower() or 'total' in c.lower() or 'min' in c.lower():
                    fmap[c] = 'usage_total_min'; return fmap
        # fallback: numeric columns map to themselves prefixed
        return {}

    for p in expected:
        if p.exists():
            present.append(str(p))
            df = read_if(p)
            if df is None or df.empty:
                # keep as present but empty
                continue
            # find date column
            date_col = None
            for c in df.columns:
                if c.lower() == 'date' or c.lower().endswith('_date') or c.lower().endswith('_utc') or c.lower() in ('start','end'):
                    date_col = c; break
            if date_col is None:
                date_col = df.columns[0]
            try:
                df['date'] = pd.to_datetime(df[date_col], utc=True, errors='coerce').dt.date
            except Exception:
                try:
                    df['date'] = pd.to_datetime(df[date_col], errors='coerce').dt.date
                except Exception:
                    continue
            # drop original date_col to avoid duplicates
            if date_col != 'date' and date_col in df.columns:
                try:
                    df = df.drop(columns=[date_col])
                except Exception:
                    pass
            # apply mapping heuristics
            fmap = map_columns(p.name, df)
            if fmap:
                df = df.rename(columns=fmap)
            else:
                # prefix numeric columns
                rename = {}
                for c in df.columns:
                    if c == 'date':
                        continue
                    if pd.api.types.is_numeric_dtype(df[c]):
                        rename[c] = f"{p.stem}_{c}"
                if rename:
                    df = df.rename(columns=rename)
            dfs.append(df[['date'] + [c for c in df.columns if c != 'date']])
        else:
            missing.append(str(p))

    # if no dataframes with rows
    non_empty = [d for d in dfs if not d.empty]
    if not non_empty:
        # return NEED listing missing inputs
        print('A8-REEMIT:')
        print('- PATH: MISSING')
        print('- rows=0 cols=0')
        print('- present_inputs:', present)
        print('- missing_inputs:', missing)
        print('- READY=false')
        sys.exit(0)

    # merge
    out = non_empty[0]
    for d in non_empty[1:]:
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

    # convert sleep hours to minutes if column names include '_h'
    for c in list(out.columns):
        if 'sleep' in c and ('_h' in c or c.endswith('_h')):
            try:
                out[c.replace('_h', '_min')] = out[c].astype(float) * 60.0
                out = out.drop(columns=[c])
            except Exception:
                pass

    # drop sparse or constant
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

    # ensure canonical names exist if found in columns mapping
    # rename variants to canonical
    renmap = {}
    for c in out.columns:
        lc = c.lower()
        if 'hr_mean' in lc or (lc.startswith('health_hr') and 'mean' in lc):
            renmap[c] = 'hr_mean'
        if 'hrv' in lc and ('mean' in lc or 'sdn' in lc):
            renmap[c] = 'hrv_sdnn_mean'
        if 'sleep' in lc and ('sum' in lc or 'total' in lc or 'sleep' in lc):
            renmap[c] = 'sleep_total_min'
        if 'usage' in lc and ('min' in lc or 'usage' in lc or 'total' in lc):
            renmap[c] = 'usage_total_min'
    if renmap:
        out = out.rename(columns=renmap)

    out = out.sort_values('date').reset_index(drop=True)
    out_path = joined / 'features_daily.csv'
    # write date as ISO
    out['date'] = out['date'].apply(lambda d: pd.to_datetime(d).date().isoformat() if pd.notna(d) else '')
    out.to_csv(out_path, index=False)

    rows = len(out)
    cols = len(out.columns)
    present_inputs = present
    missing_inputs = missing
    ready = (rows >= 2) and (cols >= 4)

    print('A8-REEMIT:')
    print(f'- PATH: {out_path.resolve()}')
    print(f'- rows={rows} cols={cols}')
    print(f'- present_inputs: {present_inputs}')
    print(f'- missing_inputs: {missing_inputs}')
    print(f'- READY={str(bool(ready)).lower()}')


if __name__ == '__main__':
    main()
