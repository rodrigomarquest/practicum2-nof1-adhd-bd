#!/usr/bin/env python3
"""A8 diagnostic: inspect processed/*_daily.csv files and write a small markdown report.

Usage:
  python make_scripts/diag/a8_diag_processed.py --snapshot <snapshot_dir> --out reports/diag/a8_diag_processed.md

No new dependencies.
"""
from pathlib import Path
import argparse
import pandas as pd
import sys


def inspect_file(p: Path):
    try:
        df = pd.read_csv(p)
    except Exception as e:
        return dict(path=str(p), ok=False, error=str(e))
    rows = len(df)
    cols = len(df.columns)
    cols_list = list(df.columns)[:8]
    has_date = any(c.lower() == 'date' for c in df.columns)
    note = ' ⚠️ missing date' if not has_date else ''
    return dict(path=str(p), ok=True, rows=rows, cols=cols, columns=cols_list, note=note)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', required=False,
                        default='data/etl/P000001/snapshots/2025-10-22',
                        help='Snapshot root directory')
    parser.add_argument('--out', required=False,
                        default='reports/diag/a8_diag_processed.md',
                        help='Output markdown file')
    args = parser.parse_args()

    snap = Path(args.snapshot)
    processed = snap / 'processed'

    targets = [
        processed / 'apple' / 'health_hr_daily.csv',
        processed / 'apple' / 'health_hrv_sdnn_daily.csv',
        processed / 'apple' / 'health_sleep_daily.csv',
        processed / 'ios' / 'ios_usage_daily.csv',
        processed / 'zepp' / 'zepp_hr_daily.csv',
        processed / 'zepp' / 'zepp_hrv_daily.csv',
        processed / 'zepp' / 'zepp_sleep_daily.csv',
    ]

    report_lines = []
    all_ok = True

    for t in targets:
        if not t.exists():
            report_lines.append(f"- {t}: MISSING")
            all_ok = False
            continue
        info = inspect_file(t)
        if not info.get('ok'):
            report_lines.append(f"- {t}: ERROR reading ({info.get('error')})")
            all_ok = False
            continue
        cols_repr = ', '.join([f"'{c}'" for c in info['columns']])
        note = info.get('note','')
        report_lines.append(f"- {t}: rows={info['rows']}, cols={info['cols']}, columns=[{cols_repr}]{note}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open('w', encoding='utf-8') as fh:
        fh.write('# A8 DIAG: processed daily CSVs\n\n')
        fh.write('\n'.join(report_lines))
        fh.write('\n')

    print(f"DIAG: out={out_path.resolve()} READY={str(bool(all_ok)).lower()}")


if __name__ == '__main__':
    main()
