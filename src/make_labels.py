"""Labeling CLI skeleton for v4.

Usage:
  python -m src.make_labels --rules config/label_rules.yaml --in data/.../features_daily.csv --out data/.../features_daily_labeled.csv
"""

import argparse
from pathlib import Path
import yaml
import pandas as pd
from .utils import zscore_by_segment, write_csv


def load_rules(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--in", dest="inpath", required=True)
    ap.add_argument("--out", dest="outpath", required=True)
    args = ap.parse_args(argv)

    rules = load_rules(Path(args.rules))
    df = pd.read_csv(args.inpath, parse_dates=["date"])

    zcols = rules.get("zscore_columns", [])
    if zcols:
        df = zscore_by_segment(
            df,
            zcols,
            segment_col=rules.get("outputs", {}).get("segment_column", "segment_id"),
        )

    # Placeholder: apply heuristics here
    df["label"] = "unlabeled"
    df["label_confidence"] = 0.0
    df["label_source"] = rules.get("meta", {}).get("label_source", "heuristic_v1")

    write_csv(df, args.outpath)
    # Print coverage and class balance
    coverage = 100.0 * df["label"].notna().mean()
    print("Coverage_pct=", coverage)
    print(df["label"].value_counts(dropna=False).to_dict())

    min_cov = rules.get("outputs", {}).get("min_coverage_pct", 90)
    return 0 if coverage >= min_cov else 2


if __name__ == "__main__":
    raise SystemExit(main())
