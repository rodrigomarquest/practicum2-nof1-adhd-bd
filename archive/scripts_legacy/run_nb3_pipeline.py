#!/usr/bin/env python
"""
NB3 Pipeline Orchestrator

Runs:
  1. Logistic + SHAP + Drift Detection
  2. LSTM M1 + TFLite + Latency profiling
  
Usage:
  python scripts/run_nb3_pipeline.py \
    --csv data/etl/features_daily_labeled.csv \
    --outdir nb3 \
    --label_col label_3cls
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nb3_run import main as nb3_main


def main():
    """Run NB3 pipeline."""
    parser = argparse.ArgumentParser(
        description="NB3: SHAP + Drift + LSTM M1 + TFLite",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--csv",
        type=str,
        required=False,
        default="data/etl/features_daily_labeled.csv",
        help="Path to labeled CSV (default: data/etl/features_daily_labeled.csv)"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        default="nb3",
        help="Output directory (default: nb3)"
    )
    
    parser.add_argument(
        "--label_col",
        type=str,
        default="label_3cls",
        help="Label column name (default: label_3cls)"
    )
    
    parser.add_argument(
        "--date_col",
        type=str,
        default="date",
        help="Date column name (default: date)"
    )
    
    parser.add_argument(
        "--segment_col",
        type=str,
        default="segment_id",
        help="Segment column for drift detection (default: segment_id)"
    )
    
    parser.add_argument(
        "--seq_len",
        type=int,
        default=14,
        help="LSTM sequence length (default: 14)"
    )
    
    args = parser.parse_args()
    
    # Forward to nb3_main with updated sys.argv
    sys.argv = [
        sys.argv[0],
        "--csv", args.csv,
        "--outdir", args.outdir,
        "--label_col", args.label_col,
        "--date_col", args.date_col,
        "--segment_col", args.segment_col,
        "--seq_len", str(args.seq_len)
    ]
    
    nb3_main()


if __name__ == "__main__":
    main()
