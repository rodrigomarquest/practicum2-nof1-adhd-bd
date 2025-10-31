#!/usr/bin/env python3
"""Join processed daily tables (apple, zepp, optional) into joined/features_daily.csv

Usage: python make_scripts/a8_join_processed.py --snapshot <snapshot_dir> [--out <out_csv>]
"""
from pathlib import Path
import argparse
import pandas as pd
import json
from datetime import datetime
import sys


def read_optional(path: Path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def normalize_date_col(df: pd.DataFrame):
    # find date-like column
    candidates = [c for c in df.columns if c.lower() in ('date', 'date_utc', 'timestamp') or 'date' in c.lower()]
    if candidates:
        col = candidates[0]
        try:
            df['date'] = pd.to_datetime(df[col], errors='coerce').dt.date
        except Exception:
            df['date'] = pd.to_datetime(df[col], utc=True, errors='coerce').dt.date
        df = df.drop(columns=[col]) if col != 'date' else df
        return df
    # fallback: if index-like, return unchanged but ensure date col exists
    return df


def prefix_columns(df: pd.DataFrame, prefix: str):
    cols = [c for c in df.columns if c != 'date']
    rename = {c: f"{prefix}{c}" for c in cols}
    return df.rename(columns=rename)


def drop_sparse_constant(df: pd.DataFrame):
    cols = [c for c in df.columns if c != 'date']
    keep = []
    for c in cols:
        s = df[c]
        if s.isna().mean() > 0.9:
            continue
        if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 1:
            continue
        keep.append(c)
    return df[['date'] + keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', required=True)
    p.add_argument('--out', required=False)
    p.add_argument('--tz', default='Europe/Dublin')
    args = p.parse_args()

    snap = Path(args.snapshot)
    processed = snap / 'processed'
    joined = snap / 'joined'
    joined.mkdir(parents=True, exist_ok=True)

    files = {
        'apple_hr': processed / 'apple' / 'health_hr_daily.csv',
        'zepp_hr': processed / 'zepp' / 'zepp_hr_daily.csv',
        'apple_sleep': processed / 'apple' / 'health_sleep_daily.csv',
        'ios_usage': processed / 'ios' / 'ios_usage_daily.csv',
    }

    dfs = []
    sources = []
    for k, path in files.items():
        df = read_optional(path)
        if df is None:
            continue
        df = normalize_date_col(df)
        # ensure date column and convert to ISO
        if 'date' in df.columns:
            df = df[df['date'].notna()]
            df['date'] = df['date'].apply(lambda d: pd.to_datetime(d).date().isoformat())
        # prefix metric columns to avoid collision
        pref = 'apple_' if k.startswith('apple') else ('zepp_' if k.startswith('zepp') else k + '_')
        df = prefix_columns(df, pref)
        dfs.append(df)
        sources.append(str(path))

    if not dfs:
        print('A8-JOIN:')
        print('- PATH: MISSING')
        print('- rows=0 cols=0')
        print('- features_head=[]')
        print('- READY=false')
        sys.exit(0)

    out = dfs[0]
    for d in dfs[1:]:
        out = out.merge(d, on='date', how='outer')

    # collapse duplicate dates by mean for numeric
    if out['date'].duplicated().any():
        agg = {}
        for c in out.columns:
            if c == 'date':
                continue
            if pd.api.types.is_numeric_dtype(out[c]):
                agg[c] = 'mean'
            else:
                agg[c] = lambda s: s.dropna().iloc[0] if not s.dropna().empty else None
        out = out.groupby('date', as_index=False).agg(agg)

    out = out.sort_values('date').reset_index(drop=True)
    out = drop_sparse_constant(out)

    out_path = Path(args.out) if args.out else joined / 'features_daily.csv'
    out.to_csv(out_path, index=False)

    side = out_path.parent / 'features_daily_source.json'
    meta = {'sources_used': sources, 'generated_at': datetime.utcnow().isoformat() + 'Z', 'tz': args.tz}
    side.write_text(json.dumps(meta))

    rows = len(out)
    cols = len(out.columns)
    features_head = [c for c in out.columns if c != 'date'][:6]
    ready = (rows >= 2) and (cols >= 4)

    print('A8-JOIN:')
    print(f'- PATH: {out_path.resolve()}')
    print(f'- rows={rows} cols={cols}')
    print(f'- features_head={features_head}')
    print(f'- READY={str(bool(ready)).lower()}')


if __name__ == '__main__':
    main()
