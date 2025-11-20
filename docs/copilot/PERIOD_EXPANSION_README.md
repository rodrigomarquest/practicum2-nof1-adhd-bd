# Period Expansion & Auto-Segmentation Pipeline

## Overview

Complete pipeline to expand ML6/ML7 analysis across all available data with automatic segmentation:

1. **ZIP Discovery & Extraction** — Recursive scan of `data/raw/` for Apple/Zepp archives
2. **Daily Unification** — Merge Apple+Zepp measurements per day
3. **Auto-Segmentation** — Detect context changes without `version_log_enriched.csv`
4. **PBSI Labels** — Compute stability index scores and 2/3-class labels
5. **ML6 Baselines** — Train 5 models across 6 calendar-based folds
6. **ML7 Analytics** — SHAP, Drift, LSTM, and TFLite export

## Quick Start

### Prerequisites

```bash
# Install dependencies (already done in .venv)
pip install scikit-learn shap river tensorflow matplotlib seaborn plotly pandas numpy

# Set Zepp password (if needed)
export ZEPP_ZIP_PASSWORD="your_password_here"
```

### Run Full Pipeline

```bash
# Dry run (no extraction, just scan)
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07 \
    --dry-run

# Full run (stages 1-4: extract, unify, segment, labels)
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07

# Include ML6/ML7 (stages 1-6)
python scripts/run_period_expansion.py \
    --participant P000001 \
    --snapshot 2025-11-07 \
    --skip-nb 0 \
    --n-folds 6
```

### Individual Steps

#### Step 1: Extract ZIPs

```bash
# Dry run
python src/io/zip_extractor.py \
    --participant P000001 \
    --dry-run

# Extract (requires ZEPP_ZIP_PASSWORD for encrypted ZIPs)
python src/io/zip_extractor.py \
    --participant P000001 \
    --zepp-password $ZEPP_ZIP_PASSWORD
```

**Output**: `data/extracted/{apple,zepp}/P000001/...`

#### Step 2: Unify Daily Data

```bash
# Manual unification (if not using full pipeline)
python -c "
from src.features.unify_daily import unify_apple_zepp
from pathlib import Path

apple_dir = Path('data/extracted/apple/P000001')
zepp_dir = Path('data/extracted/zepp/P000001')

df = unify_apple_zepp(apple_dir, zepp_dir)
df.to_csv('data/etl/features_daily_unified.csv', index=False)
print(f'Unified {len(df)} days')
"
```

**Output**: `data/etl/features_daily_unified.csv` (27 columns)

#### Step 3: Auto-Segmentation

```bash
# Run segmentation
python -c "
import pandas as pd
from src.labels.auto_segment import auto_segment
from pathlib import Path

df = pd.read_csv('data/etl/features_daily_unified.csv')
seg_df, decisions = auto_segment(
    df,
    output_csv=Path('data/etl/features_daily_with_segments.csv'),
    autolog_csv=Path('data/etl/segment_autolog.csv'),
)
print(f'Generated {seg_df[\"segment_id\"].max()} segments')
"
```

**Output**:

- `data/etl/features_daily_with_segments.csv` (28 columns, +segment_id)
- `data/etl/segment_autolog.csv` (transition log)

#### Step 4: PBSI Labels

```bash
# Compute labels (requires features_daily_with_segments.csv or features_daily_unified.csv)
python -c "
import pandas as pd
from src.labels.build_pbsi import compute_z_scores_by_segment, compute_pbsi_labels
from pathlib import Path

df = pd.read_csv('data/etl/features_daily_with_segments.csv')
df = compute_z_scores_by_segment(df)

pbsi_cols = []
for _, row in df.iterrows():
    pbsi_result = compute_pbsi_labels(row)
    pbsi_cols.append(pbsi_result)

pbsi_df = pd.DataFrame(pbsi_cols)
df = pd.concat([df, pbsi_df], axis=1)
df.to_csv('data/etl/features_daily_labeled.csv', index=False)
print(f'Created {len(df)} rows with PBSI labels')
"
```

**Output**: `data/etl/features_daily_labeled.csv` (35+ columns)

#### Step 5: ML6 Baselines

```bash
python run_ml6_beiwe.py \
    --pid P000001 \
    --snapshot 2025-11-07 \
    --n-folds 6 \
    --train-days 120 \
    --val-days 60 \
    --seed 42
```

**Output**:

- `ml6/baselines_label_3cls.csv` (6 folds × 5 models)
- `ml6/baselines_label_2cls.csv` (includes McNemar p-values)
- `ml6/confusion_matrices/*.png` (18 confusion matrices)

#### Step 6: ML7 Analytics

```bash
python scripts/run_nb3_pipeline.py \
    --participant P000001 \
    --snapshot 2025-11-07
```

**Output**:

- `ml7/shap_summary.md` (top-5 features per fold + global)
- `ml7/drift_report.md` (ADWIN + KS changepoints)
- `ml7/lstm_report.md` (best model metrics)
- `ml7/models/best_model.tflite` (quantized model)
- `ml7/plots/*.png` (SHAP + ADWIN visualizations)

## Auto-Segmentation Rules

Segment boundaries are triggered by (in order of priority):

### 1. Source Change (≥5 consecutive days)

When dominant source of heart rate changes (apple ↔ zepp)

```
Example: Days 1-90 from Apple → Days 91+ from Zepp → new segment at day 91
```

### 2. Signal Change (≥7-day sustained shift)

Abrupt biomarker changes:

- Heart rate: Δ ≥ 8 bpm
- HRV: Δ ≥ 10 ms
- Sleep efficiency: Δ ≥ 0.08

```
Example: HR mean 60 bpm (days 1-30) → 75 bpm (days 31+) → new segment at day 31
```

### 3. Gap Recovery (≥3 consecutive missing days)

After missing both cardio + sleep for ≥3 days, signal recovery starts new segment

```
Example: Days 45-47 missing → Day 48 signal returns → new segment at day 48
```

### 4. Temporal Fallback (~60-day windows)

If no other rule triggers, force new segment every ~60 days to maintain CV fold compatibility

```
Example: Day 60 of segment → new segment (ensures 4m train / 2m val compatibility)
```

**Decision Log**: See `data/etl/segment_autolog.csv` for all transitions with dates and reasons.

## Output Structure

```
data/
  etl/
    features_daily_unified.csv           # 27 cols: raw unified data
    features_daily_with_segments.csv      # +segment_id from auto-segmentation
    segment_autolog.csv                   # Transition log (date, reason, metric, old_seg, new_seg)
    features_daily_labeled.csv            # 35+ cols: final dataset with PBSI labels

ml6/
  baselines_label_3cls.csv                # 6 folds × 5 models, F1/Acc/Kappa/AUROC
  baselines_label_2cls.csv                # 2-class metrics + McNemar p-values
  confusion_matrices/
    fold_0_dummy.png
    fold_0_naive.png
    ...

ml7/
  shap_summary.md                         # Top-5 features per fold + global ranking
  drift_report.md                         # ADWIN changepoints + KS hits
  lstm_report.md                          # LSTM metrics + TFLite info
  latency_stats.json                      # 200-run latency measurement
  models/
    best_model.tflite                     # Quantized LSTM model (44 KB)
  plots/
    shap_summary_fold_0.png
    adwin_changepoints.png
    ...

logs/
  pipeline_expansion_*.log                # Detailed execution log
  pipeline_stats.json                     # JSON summary of all stages
```

## Monitoring Progress

### Real-time Log

```bash
# Watch log output (last 50 lines)
tail -50f logs/pipeline_expansion_*.log
```

### Check Extraction Progress

```bash
# Count extracted files
find data/extracted -type f | wc -l

# Show extraction summary
ls -lah data/extracted/{apple,zepp}/P000001/
```

### Verify Segmentation

```bash
# Inspect segment assignments
python -c "
import pandas as pd
df = pd.read_csv('data/etl/features_daily_with_segments.csv')
print(df[['date', 'segment_id', 'source_cardio']].groupby('segment_id').agg({
    'date': ['min', 'max', 'count'],
    'source_cardio': 'first'
}))
"
```

### Review Decisions

```bash
# Show segmentation decisions
python -c "
import pandas as pd
df = pd.read_csv('data/etl/segment_autolog.csv')
print(df.to_string())
"
```

## Troubleshooting

### Zepp ZIP extraction fails

```
Error: "Zepp ZIP password not provided and ZEPP_ZIP_PASSWORD not set"
```

**Solution**: Set environment variable before running:

```bash
export ZEPP_ZIP_PASSWORD="your_password"
python scripts/run_period_expansion.py --participant P000001
```

### No Apple data found

```
Error: "Apple directory not found: data/extracted/apple/..."
```

**Solution**: Check ZIP extraction succeeded:

```bash
python src/io/zip_extractor.py --participant P000001 --dry-run
```

### Segmentation creates too many segments

Adjust thresholds in `src/labels/auto_segment.py`:

```python
generate_segments(
    df,
    source_window=5,          # Days to detect source change
    signal_window=7,          # Days to detect signal change
    gap_min=3,                # Minimum gap days for recovery
    temporal_period=60,       # Force segment every N days
)
```

### PBSI labels all zeros

Check z-score computation by segment:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/etl/features_daily_with_segments.csv')
print('Segments:', df['segment_id'].unique())
print('Rows per segment:', df['segment_id'].value_counts().sort_index())
"
```

## Performance Notes

- **ZIP Extraction**: ~1 min for 1700+ files (Apple 120 MB)
- **Unification**: ~5 sec for 400 days
- **Auto-Segmentation**: ~2 sec for 400 days
- **PBSI Computation**: ~10 sec for 400 days
- **ML6 Training**: ~5 min for 6 folds × 5 models
- **ML7 SHAP+Drift+LSTM**: ~15 min for 6 folds

**Total time**: ~30 min for full pipeline (P000001)

## Next Steps

1. Extract all participants: `for p in P000001 P000002 P000003; do python src/io/zip_extractor.py --participant $p; done`
2. Compare segmentation across participants
3. Investigate SHAP feature importance by segment
4. Monitor drift metrics over time
5. Deploy TFLite models to mobile app

## References

- **PBSI Formula**: `src/labels/build_pbsi.py` (sleep_sub, cardio_sub, activity_sub weights)
- **CV Protocol**: Calendar-based (4m train, 2m val), non-overlapping folds
- **SHAP Method**: LinearExplainer for Logistic Regression (exact Shapley values)
- **Drift Detection**: ADWIN (δ=0.002) + KS tests (p<0.01)
- **LSTM**: LSTM(32) → Dense(32, ReLU) → Dropout(0.2) → softmax

---

**Created**: 2025-11-07  
**Last Updated**: 2025-11-07  
**Status**: Ready for production
