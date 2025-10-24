#!/usr/bin/env python3
"""XZ2: Compute cross-source consistency report between Apple and Zepp daily CSVs.

Outputs a JSON and a simple markdown summary. Deterministic and idempotent.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Any, List, Tuple


try:
    from scipy import stats
except Exception:
    stats = None


def read_daily_csv(path: Path, key: str) -> Dict[str, float]:
    """Read a daily CSV and return date->value map for the first matching key.
    Assumes CSV has a 'date' column and numeric metric columns.
    """
    if not path.exists():
        return {}
    out = {}
    with path.open(newline='', encoding='utf-8') as fh:
        rdr = csv.DictReader(fh)
        # pick the metric column that matches key or fallback to second column
        cols = [c for c in rdr.fieldnames if c != 'date'] if rdr.fieldnames else []
        col = None
        for c in cols:
            if key.lower() in c.lower():
                col = c
                break
        if col is None and cols:
            col = cols[0]
        for r in rdr:
            d = r.get('date')
            if not d:
                continue
            v = r.get(col) if col else None
            try:
                out[d] = float(v) if v not in (None, '') else math.nan
            except Exception:
                out[d] = math.nan
    return out


def paired_stats(a_map: Dict[str, float], b_map: Dict[str, float]) -> Tuple[List[float], List[float], List[str]]:
    common = sorted(set(a_map.keys()) & set(b_map.keys()))
    aseries, bseries, dates = [], [], []
    for d in common:
        av = a_map.get(d)
        bv = b_map.get(d)
        if av is None or bv is None:
            continue
        if math.isnan(av) or math.isnan(bv):
            continue
        aseries.append(av)
        bseries.append(bv)
        dates.append(d)
    return aseries, bseries, dates


def pearson_r(x: List[float], y: List[float]) -> float:
    if stats is not None:
        try:
            r, _ = stats.pearsonr(x, y)
            return float(r)
        except Exception:
            pass
    # simple fallback
    if not x or not y or len(x) < 2:
        return float('nan')
    mx, my = mean(x), mean(y)
    num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    den = math.sqrt(sum((xi - mx) ** 2 for xi in x) * sum((yi - my) ** 2 for yi in y))
    if den == 0:
        return float('nan')
    return num / den


def bland_altman_limits(a: List[float], b: List[float]) -> Tuple[float, float, float]:
    # returns bias_mean, lower_limit, upper_limit (95% LoA)
    diffs = [ai - bi for ai, bi in zip(a, b)]
    if not diffs:
        return float('nan'), float('nan'), float('nan')
    m = mean(diffs)
    s = pstdev(diffs) if len(diffs) > 1 else 0.0
    lo = m - 1.96 * s
    hi = m + 1.96 * s
    return m, lo, hi


def make_report_for_metric(name: str, apple_map: Dict[str, float], zepp_map: Dict[str, float], unit_threshold: float) -> Dict[str, Any]:
    a, b, dates = paired_stats(apple_map, zepp_map)
    n_overlap = len(a)
    r = pearson_r(a, b) if n_overlap > 1 else float('nan')
    bias_mean, lo, hi = bland_altman_limits(a, b) if n_overlap > 0 else (float('nan'), float('nan'), float('nan'))
    bias_std = pstdev([ai - bi for ai, bi in zip(a, b)]) if n_overlap > 1 else 0.0

    qc_flags = []
    if n_overlap < 30:
        qc_flags.append('low_overlap')
    if not math.isnan(r) and abs(r) < 0.3:
        qc_flags.append('low_correlation')
    if not math.isnan(bias_mean) and abs(bias_mean) > unit_threshold:
        qc_flags.append('systematic_bias')

    return {
        'metric': name,
        'n_overlap_days': n_overlap,
        'pearson_r': None if math.isnan(r) else r,
        'bias_mean': None if math.isnan(bias_mean) else bias_mean,
        'bias_std': None if math.isnan(bias_std) else bias_std,
        'bland_altman_limits': [None if math.isnan(lo) else lo, None if math.isnan(hi) else hi],
        'qc_flags': qc_flags
    }


def md_summary(metrics: List[Dict[str, Any]]) -> str:
    lines = ["# XSource consistency report\n"]
    for m in metrics:
        lines.append(f"## {m['metric']}\n")
        lines.append(f"- Overlap days: {m['n_overlap_days']}\n")
        lines.append(f"- Pearson r: {m.get('pearson_r')}\n")
        lines.append(f"- Bias mean: {m.get('bias_mean')} ; bias std: {m.get('bias_std')}\n")
        lines.append(f"- Bland–Altman 95% limits: {m.get('bland_altman_limits')}\n")
        if m.get('qc_flags'):
            lines.append(f"- QC flags: {', '.join(m.get('qc_flags'))}\n")
        lines.append('\n')
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--joined-dir', required=True)
    p.add_argument('--processed-dir', required=True, help='Processed dir containing apple/ and zepp/')
    p.add_argument('--unit-threshold-hr', type=float, default=3.0)
    p.add_argument('--unit-threshold-hrv', type=float, default=10.0)
    args = p.parse_args(argv)

    joined = Path(args.joined_dir)
    joined.mkdir(parents=True, exist_ok=True)

    # metric list and file-name heuristics (expects processed/zepp and processed/apple)
    metrics = [
        ('hr', 'hr_mean', 'zepp_hr_mean', args.unit_threshold_hr),
        ('hrv_sdnn', 'hrv_sdnn_ms', 'zepp_hrv_sdnn', args.unit_threshold_hrv),
        ('hrv_rmssd', 'hrv_rmssd_ms', 'zepp_hrv_rmssd', args.unit_threshold_hrv),
        ('sleep_total_minutes', 'sleep_minutes', 'zepp_sleep_minutes', 30.0)
    ]

    results = []
    for metric, apple_key, zepp_key, threshold in metrics:
        apple_map = read_daily_csv(Path(args.processed_dir) / 'apple' / f'health_{metric}_daily.csv', apple_key)
        zepp_map = read_daily_csv(Path(args.processed_dir) / 'zepp' / f'zepp_{metric}_daily.csv', zepp_key)
        rpt = make_report_for_metric(metric, apple_map, zepp_map, threshold)
        results.append(rpt)

    outjson = joined / 'xsource_consistency_report.json'
    atomic_text = json.dumps({'metrics': sorted(results, key=lambda x: x['metric'])}, sort_keys=True, indent=2)
    outjson.write_text(atomic_text, encoding='utf-8')

    outmd = joined / 'xsource_consistency_report.md'
    outmd.write_text(md_summary(sorted(results, key=lambda x: x['metric'])), encoding='utf-8')

    print('WROTE:', outjson, outmd)


if __name__ == '__main__':
    main()
