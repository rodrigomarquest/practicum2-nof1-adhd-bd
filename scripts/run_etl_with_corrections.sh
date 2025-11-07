#!/bin/bash
# Pipeline de correção completa: ETL com agregação diária + filtro de 30 meses + labels heurísticos

set -e  # Exit on error

PID="P000001"
SNAPSHOT="2025-11-07"
ZEPP_DIR="data_etl/${PID}/zepp_processed/2025-09-28"

echo "================================================================"
echo "STEP 1: Extract (if not done)"
echo "================================================================"
python src/etl_pipeline.py extract \
  --participant $PID \
  --snapshot $SNAPSHOT \
  --cutover 2024-03-11 \
  --tz_before America/Sao_Paulo \
  --tz_after Europe/Dublin \
  || echo "Extract already done or skipped"

echo ""
echo "================================================================"
echo "STEP 2: Cardio features (if not done)"
echo "================================================================"
python src/etl_pipeline.py cardio \
  --participant $PID \
  --snapshot $SNAPSHOT \
  --zepp_dir "$ZEPP_DIR" \
  || echo "Cardio already done or skipped"

echo ""
echo "================================================================"
echo "STEP 3: Join with daily aggregation + 30-month filter"
echo "================================================================"
python src/etl_pipeline.py join \
  --snapshot "data/etl/${PID}/${SNAPSHOT}" \
  || echo "Join completed or failed"

echo ""
echo "================================================================"
echo "STEP 4: Verify joined output"
echo "================================================================"
python << 'VERIFY'
import pandas as pd
from pathlib import Path

df = pd.read_csv(f'data/etl/P000001/2025-11-07/joined/joined_features_daily.csv', parse_dates=['date'])
print(f"✓ Joined output verified:")
print(f"  Rows (daily): {len(df)}")
print(f"  Unique dates: {df['date'].nunique()}")
print(f"  One row per date: {len(df) == df['date'].nunique()}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"  Months span: {(df['date'].max() - df['date'].min()).days / 30:.1f}")

# Check for sleep data
if "zepp_slp_total_h" in df.columns:
    nval = df["zepp_slp_total_h"].notna().sum()
    print(f"  Sleep data: {nval} non-null rows ({nval/len(df)*100:.1f}%)")
else:
    print(f"  Sleep data: NOT FOUND")

VERIFY

echo ""
echo "================================================================"
echo "STEP 5: Build heuristic labels"
echo "================================================================"
python build_heuristic_labels.py \
  --pid $PID \
  --snapshot $SNAPSHOT \
  --verbose 1

echo ""
echo "================================================================"
echo "STEP 6: Verify labeled output"
echo "================================================================"
python << 'VERIFY2'
import pandas as pd

df = pd.read_csv(f'data/etl/P000001/2025-11-07/joined/features_daily_labeled.csv', parse_dates=['date'])
print(f"✓ Labeled output verified:")
print(f"  Rows: {len(df)}")
print(f"  Label distribution:")
print(df['label_final'].value_counts().to_string())
print(f"  Unlabeled rate: {(df['label_final'] == 'unlabeled').sum() / len(df) * 100:.1f}%")

VERIFY2

echo ""
echo "================================================================"
echo "PIPELINE COMPLETE - Ready for NB2"
echo "================================================================"
echo ""
echo "Next steps:"
echo "  python run_nb2_engage7.py --pid P000001 --snapshot 2025-11-07 --n-folds 6 --verbose 1"
