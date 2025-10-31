"""Create a deterministic, privacy-safe public subset from features_daily.csv

Writes:
 - data/ai/{pid}/snapshots/{snap}/public_subset/features_daily_public.csv
 - optional features_daily_labeled_public.csv (synthetic coarse labels)
 - manifest_public.json (schema, ranges, SHA256, anonymization notes)
 - README.md, LICENSE.txt

Determinism: sorted by date_utc, stable column order, JSON manifests with sorted keys.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime
import csv
import shutil

import pandas as pd

# Whitelist columns to keep (order matters)
WHITELIST = [
    'date_utc',
    'hr_min', 'hr_max', 'hr_mean', 'hr_median', 'hr_std', 'hr_count',
    'hrv_sdnn_mean', 'hrv_rmssd_mean',
    'sleep_total_minutes', 'sleep_efficiency',
    'segment_id_general',
]

DEFAULT_LICENSE = '''Copyright (c) 2025

Licensed for academic and non-commercial research use only. Redistribution or commercial use is prohibited without prior written permission from the data custodian.
'''

README_TEMPLATE = '''Public subset extracted from participant {pid} snapshot {snap}.

Contents:
- features_daily_public.csv: daily aggregated features with PII removed and columns whitelisted.
- features_daily_labeled_public.csv: OPTIONAL synthetic/coarse labels (if generated).
- manifest_public.json: self-describing manifest including schema, date ranges, SHA256 checksums, and anonymization notes.

Column whitelist: {cols}
Anonymization notes: minute-level timestamps, device identifiers and raw/ETL paths removed. Segment IDs are generalized.

Citation: If you use this dataset, cite the original study repository and contact the data steward.
'''

MANIFEST_TEMPLATE = {
    'schema_version': '1.0',
    'producer': 'package_public_subset.py',
    'generated_at': None,  # fill in
    'participant': None,
    'snapshot_date': None,
    'files': [],  # list of {path, sha256, rows, cols}
    'anonymization': 'Dropped device identifiers, exact tz info, minute-level timestamps, app-level data; segment_id generalized',
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_csv_deterministic(df: pd.DataFrame, path: Path) -> None:
    # Ensure stable column order and deterministic CSV writing
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use Unix newline for deterministic output
    df.to_csv(path, index=False, lineterminator='\n')


def synth_labels(df: pd.DataFrame) -> pd.DataFrame:
    # Create a coarse synthetic label: e.g., activity_level based on hr_mean
    out = df[['date_utc']].copy()
    mean_hr = df['hr_mean'].fillna(0)
    # simple bins: low (<60), normal (60-90), high (>90)
    bins = pd.cut(mean_hr, bins=[-1,59,90,9999], labels=['low','normal','high'])
    out['label'] = bins.astype(str)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('--participant', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--input', default=None, help='Path to features_daily.csv; if omitted defaults to data/ai/{pid}/snapshots/{snap}/features_daily.csv')
    p.add_argument('--outdir', default=None, help='Output public subset dir; defaults to data/ai/{pid}/snapshots/{snap}/public_subset/')
    p.add_argument('--with-labels', action='store_true', help='Generate synthetic coarse labels file')
    args = p.parse_args(argv)

    pid = args.participant
    snap = args.snapshot
    default_input = Path('data') / 'ai' / pid / 'snapshots' / snap / 'features_daily.csv'
    input_path = Path(args.input) if args.input else default_input
    outdir = Path(args.outdir) if args.outdir else Path('data') / 'ai' / pid / 'snapshots' / snap / 'public_subset'

    if not input_path.exists():
        print('ERROR: input features_daily.csv not found at', input_path, file=sys.stderr)
        return 2

    # Read input with pandas
    df = pd.read_csv(input_path)

    # Drop forbidden columns by only selecting whitelist intersection
    cols_present = [c for c in WHITELIST if c in df.columns]
    # Rename generalized segment id if necessary
    if 'segment_id_general' in cols_present and 'segment_id_general' not in df.columns and 'segment_id' in df.columns:
        df = df.rename(columns={'segment_id':'segment_id_general'})

    df_public = df.loc[:, cols_present].copy()

    # Ensure date_utc is present and sortable
    if 'date_utc' not in df_public.columns:
        print('ERROR: date_utc column required in input', file=sys.stderr)
        return 3

    # Sort rows by date_utc (parse if needed)
    try:
        df_public['date_utc'] = pd.to_datetime(df_public['date_utc'])
    except Exception:
        # if it's already a sorted string, leave as-is
        pass
    df_public = df_public.sort_values('date_utc')
    # Convert date_utc back to ISO date string for portability
    df_public['date_utc'] = df_public['date_utc'].dt.strftime('%Y-%m-%d')

    # Ensure canonical column order: as specified in cols_present
    df_public = df_public[[c for c in WHITELIST if c in df_public.columns]]

    # Write features_daily_public.csv
    out_features = outdir / 'features_daily_public.csv'
    write_csv_deterministic(df_public, out_features)

    manifest = MANIFEST_TEMPLATE.copy()
    # Make manifest deterministic: do not include wall-clock timestamps.
    manifest.pop('generated_at', None)
    manifest['participant'] = pid
    manifest['snapshot_date'] = snap

    # Add file entry list for features
    manifest['files'] = []
    h = sha256_file(out_features)
    manifest['files'].append({
        'path': str(Path('public_subset') / out_features.name),
        'sha256': h,
        'rows': int(len(df_public)),
        'cols': list(df_public.columns),
    })

    # Optional labels
    if args.with_labels:
        labels_df = synth_labels(df_public)
        out_labels = outdir / 'features_daily_labeled_public.csv'
        write_csv_deterministic(labels_df, out_labels)
        h2 = sha256_file(out_labels)
        manifest['files'].append({
            'path': str(Path('public_subset') / out_labels.name),
            'sha256': h2,
            'rows': int(len(labels_df)),
            'cols': list(labels_df.columns),
            'note': 'synthetic_coarse_labels',
        })

    # Optional labels already appended to manifest['files'] above

    # Compute deterministic manifest id from concatenated file shas
    concat = ''.join([entry['sha256'] for entry in manifest['files']])
    manifest['manifest_id'] = hashlib.sha256(concat.encode('utf-8')).hexdigest()

    # Write manifest_public.json with sorted keys
    manifest_path = outdir / 'manifest_public.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    # Additional files: README and LICENSE
    readme_path = outdir / 'README.md'
    readme_text = README_TEMPLATE.format(pid=pid, snap=snap, cols=', '.join(WHITELIST))
    readme_path.write_text(readme_text, encoding='utf-8')

    license_path = outdir / 'LICENSE.txt'
    license_path.write_text(DEFAULT_LICENSE, encoding='utf-8')

    # Recompute SHA for manifest and include it (optional)
    mhash = sha256_file(manifest_path)
    # update manifest with its own sha (not strictly necessary but useful)
    manifest['files'].append({
        'path': str(Path('public_subset') / manifest_path.name),
        'sha256': mhash,
        'rows': None,
        'cols': None,
        'note': 'manifest_self'
    })
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)

    print('Wrote public subset to', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
