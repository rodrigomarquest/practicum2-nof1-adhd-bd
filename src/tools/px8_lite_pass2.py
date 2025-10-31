"""PX8-Lite Pass 2
Lightweight integrity + per-segment drift-prep using a daily features table.
Writes: reports/drift_hint_summary.csv, reports/drift_hint_summary.md

Rules:
- Prefer existing daily features files (joined/features_daily.csv -> joined/features_daily_updated.csv -> processed/features_daily.csv)
- Otherwise derive daily aggregates from minute-level normalized CSVs (lightweight mean/count)
- Europe/Dublin tz assumed for timestamps where needed
"""
from pathlib import Path
import hashlib, csv, json
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import math


def sha256_file(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_daily_source(snapshot: Path) -> tuple[Path, str]:
    """Search per the requested order and return (path, tag)"""
    joined = snapshot / 'joined'
    processed = snapshot / 'processed'
    candidates = [
        (joined / 'features_daily.csv', 'joined/features_daily.csv'),
        (joined / 'features_daily_updated.csv', 'joined/features_daily_updated.csv'),
        (processed / 'features_daily.csv', 'processed/features_daily.csv'),
    ]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    return None, ''


def derive_daily_from_normalized(snapshot: Path) -> pd.DataFrame:
    # lightweight: look for normalized per-minute CSVs for core metrics and aggregate to daily mean
    norm = snapshot / 'normalized'
    metrics = {
        'hr': ['hr', 'heart_rate', 'bpm'],
        'hrv': ['hrv', 'hrv_sdnn', 'sdnn'],
        'sleep': ['sleep'],
        'usage': ['usage', 'activity']
    }
    rows = {}
    for mname, keywords in metrics.items():
        # find any file under normalized matching keywords
        found = None
        for f in norm.rglob('*.csv'):
            n = f.name.lower()
            if any(k in n for k in keywords):
                found = f
                break
        if not found:
            continue
        try:
            df = pd.read_csv(found)
        except Exception:
            continue
        # find timestamp column
        ts_col = None
        for c in df.columns:
            if 'timestamp' in c.lower() or c.lower() in ('date','start','end'):
                ts_col = c; break
        # find value column (numeric)
        val_col = None
        for c in df.columns:
            if c==ts_col: continue
            if pd.api.types.is_numeric_dtype(df[c]):
                val_col = c; break
        if ts_col is None or val_col is None:
            continue
        try:
            df['date'] = pd.to_datetime(df[ts_col], utc=True, errors='coerce').dt.tz_convert('Europe/Dublin').dt.date
        except Exception:
            try:
                df['date'] = pd.to_datetime(df[ts_col], errors='coerce').dt.date
            except Exception:
                continue
        g = df.groupby('date')[val_col].agg(['mean','std','count']).reset_index()
        # melt into wide: mean/std/count columns prefixed with metric
        for _, r in g.iterrows():
            d = r['date']
            rows.setdefault(d, {})
            rows[d][f'{mname}__mean'] = r['mean']
            rows[d][f'{mname}__std'] = r['std']
            rows[d][f'{mname}__count'] = r['count']
    if not rows:
        return pd.DataFrame()
    out = []
    for d, vals in sorted(rows.items()):
        row = {'date': pd.to_datetime(d).date()}
        row.update(vals)
        out.append(row)
    return pd.DataFrame(out)


def main():
    repo = Path('.')
    snap = repo / 'data' / 'etl' / 'P000001' / 'snapshots' / '2025-10-22'
    outdir = repo / 'reports'
    outdir.mkdir(parents=True, exist_ok=True)

    # segments file
    ver = snap / 'joined' / 'version_log_enriched.csv'
    if not ver.exists():
        ver = snap / 'version_log_enriched.csv'
    if not ver.exists():
        raise SystemExit('NEED: version_log_enriched.csv')
    seg_df = pd.read_csv(ver)
    seg_df['start'] = pd.to_datetime(seg_df['start']).dt.date
    seg_df['end'] = pd.to_datetime(seg_df['end']).dt.date

    used_path_tag = ''
    used_path = None
    p, tag = find_daily_source(snap)
    if p:
        used_path = p
        used_path_tag = tag
        try:
            feat = pd.read_csv(p)
            if 'date' in feat.columns:
                feat['date'] = pd.to_datetime(feat['date']).dt.date
        except Exception:
            feat = pd.DataFrame()
    else:
        # derive
        feat = derive_daily_from_normalized(snap)
        used_path = Path('merged_processed')
        used_path_tag = 'merged_processed'

    # prune columns: drop >90% null or constant
    if feat.empty:
        rows_out = []
        flag_counts = {}
    else:
        cols = [c for c in feat.columns if c!='date']
        keep = []
        for c in cols:
            s = feat[c]
            if s.isna().mean() > 0.9:
                continue
            if pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) <= 1:
                continue
            keep.append(c)
        feat = feat[['date'] + keep]

        # per-segment comparisons
        rows_out = []
        from collections import Counter
        flag_counts = Counter()
        for _, seg in seg_df.iterrows():
            sid = int(seg['segment_id'])
            start = seg['start']; end = seg['end']
            for c in keep:
                ser = feat[['date', c]].dropna()
                seg_vals = ser.loc[(ser['date'] >= start) & (ser['date'] <= end), c].astype(float).values
                other_vals = ser.loc[~((ser['date'] >= start) & (ser['date'] <= end)), c].astype(float).values
                if len(seg_vals) < 2 or len(other_vals) < 2:
                    continue
                mean_a = float(np.nanmean(seg_vals))
                mean_b = float(np.nanmean(other_vals))
                delta = mean_a - mean_b
                delta_pct = (delta / (abs(mean_b) + 1e-9)) * 100.0
                try:
                    ks_stat, ks_p = ks_2samp(seg_vals, other_vals)
                except Exception:
                    ks_p = float('nan')
                flag = (abs(delta_pct) > 20.0) or (not math.isnan(ks_p) and ks_p < 0.05)
                rows_out.append({'feature': c, 'segment_id': f'S{sid}', 'n_seg': len(seg_vals), 'n_other': len(other_vals), 'delta_mean': delta, 'delta_pct': delta_pct, 'ks_p': ks_p, 'flag': flag})
                if flag:
                    flag_counts[f'S{sid}'] += 1

    # write drift_hint_summary.csv
    out_csv = outdir / 'drift_hint_summary.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as fh:
        fieldnames = ['feature','segment_id','n_seg','n_other','delta_mean','delta_pct','ks_p','flag']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_out:
            writer.writerow(r)

    # write md summary (<=10 lines) and append daily_source line
    out_md = outdir / 'drift_hint_summary.md'
    lines = ['# Drift hint summary (PX8-Lite)', '']
    if flag_counts:
        for sid, cnt in sorted(flag_counts.items()):
            lines.append(f'- {sid}: {cnt} features flagged')
    else:
        lines.append('- No drift hints flagged across segments')
    lines.append('')
    lines.append('Recommendation: review flagged features and check temporal coverage per-source (see etl_provenance_report.csv)')
    # append one-liner daily_source
    lines.append(f'daily_source: {used_path_tag}')
    with out_md.open('w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

    # READY: true if drift_hint_summary.csv exists AND at least one valid IN vs OUT comparison (rows_out > 0)
    ready = out_csv.exists() and (len(rows_out) > 0)

    # Final concise print
    print('rows:', len(rows_out))
    # flagged_per_segment dictionary
    fps = {k: int(v) for k, v in flag_counts.items()} if flag_counts else {}
    print('flagged_per_segment:', fps)
    print('daily_source:', str(used_path_tag))
    print('READY:', str(bool(ready)).lower())


if __name__ == '__main__':
    main()
