# Pipeline Smoke Test Guide

**Purpose**: Quick validation that the canonical v4.1.x ETL pipeline is working correctly after code changes or environment setup.

**Date**: 2025-11-16  
**Pipeline Version**: v4.1.3-dev

---

## Prerequisites

1. **Environment activated**:
   ```bash
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Data available**:
   - Participant data in `data/raw/P000001/`
   - At least one Apple Health export ZIP
   - (Optional) Zepp data for cardiovascular features

3. **Configuration**:
   - `config/participants.yaml` has P000001 entry
   - `config/label_rules.yaml` exists
   - `config/settings.yaml` configured

---

## Smoke Test Command

### Full Pipeline (Stages 0-9)

```bash
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --start-stage 0 \
  --end-stage 5
```

**Note**: Stages 0-5 cover the core ETL (Ingest → Aggregate → Unify → Label → Segment → Prep-NB2). Stages 6-9 (NB2 → NB3 → TFLite → Report) require more computational resources and are optional for smoke testing.

### Quick Test (Stages 1-3 only)

If data is already ingested (Stage 0 complete):

```bash
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --start-stage 1 \
  --end-stage 3
```

This runs:
- **Stage 1**: CSV Aggregation (per-metric files)
- **Stage 2**: Unify Daily (features_daily.csv)
- **Stage 3**: Apply Labels (features_daily_labeled.csv)

---

## Expected Outputs

### Stage 0: Ingest
- `data/etl/P000001/2025-09-29/apple_inapp/export.xml`
- `data/etl/P000001/2025-09-29/apple_inapp/qc/manifest.json`

### Stage 1: CSV Aggregation
- `data/etl/P000001/2025-09-29/per_metric/*.csv` (e.g., `steps.csv`, `heart_rate.csv`)

### Stage 2: Unify Daily
- `data/etl/P000001/2025-09-29/joined/features_daily.csv` (≈20-50 columns)

### Stage 3: Apply Labels
- `data/etl/P000001/2025-09-29/joined/features_daily_labeled.csv`
- Labels added: `episode_type`, `treatment_phase`, etc.

### Stage 4: Segment
- `data/etl/P000001/2025-09-29/joined/segments/*.csv` (by episode/phase)

### Stage 5: Prep-NB2
- `ai/local/P000001/2025-09-29/nb2_features_train.csv`
- `ai/local/P000001/2025-09-29/nb2_features_test.csv`
- Anti-leak preparation for modeling

---

## Approximate Runtime

| Stages | Description | Time (P000001) |
|--------|-------------|----------------|
| 0 | Ingest | ~30-60s |
| 1 | Aggregate | ~10-20s |
| 2 | Unify | ~5-10s |
| 3 | Labels | ~2-5s |
| 4 | Segment | ~2-5s |
| 5 | Prep-NB2 | ~5-10s |
| **0-5** | **Total** | **~1-2 min** |
| 6-7 | NB2 (Baselines + CV) | ~5-10 min |
| 8 | NB3 (LSTM + SHAP) | ~10-20 min |
| 9 | Report | ~1-2s |
| **0-9** | **Full** | **~15-30 min** |

*Times vary based on data size and hardware.*

---

## Validation Checks

### 1. No Errors in Logs
```bash
# Check for ERROR or CRITICAL messages
python -m scripts.run_full_pipeline ... 2>&1 | grep -i "error\|critical"
```

Should return **empty** or only expected warnings (e.g., "no Zepp data found" if not available).

### 2. Output Files Exist
```bash
# Check Stage 2 output
ls -lh data/etl/P000001/2025-09-29/joined/features_daily.csv

# Check Stage 3 output
ls -lh data/etl/P000001/2025-09-29/joined/features_daily_labeled.csv
```

### 3. Data Integrity
```bash
# Quick row count check
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-09-29/joined/features_daily.csv')
print(f'Rows: {len(df)}, Columns: {len(df.columns)}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
"
```

Expected:
- Rows: 100-500+ (depending on data span)
- Columns: 20-50
- Date range: Continuous dates within participant's data period

### 4. Labels Applied
```bash
# Check that labels were added
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-09-29/joined/features_daily_labeled.csv')
print('Label columns:', [c for c in df.columns if 'episode' in c or 'phase' in c or 'label' in c])
print('Unique episode types:', df['episode_type'].unique() if 'episode_type' in df.columns else 'N/A')
"
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError"
**Solution**: Ensure virtual environment is activated and requirements installed:
```bash
source .venv/bin/activate
pip install -r requirements/base.txt
```

### Issue: "FileNotFoundError: export.xml"
**Solution**: Stage 0 (Ingest) failed or data not available. Check:
```bash
ls data/raw/P000001/apple/
```

### Issue: "No rows in features_daily.csv"
**Solution**: 
- Check Stage 1 outputs: `ls data/etl/P000001/2025-09-29/per_metric/`
- Verify Apple Health export contains data for the snapshot date

### Issue: "ZEPP_ZIP_PASSWORD not set"
**Solution**: If using Zepp data, set environment variable:
```bash
export ZEPP_ZIP_PASSWORD="your_password"  # Linux/Mac
# or
set ZEPP_ZIP_PASSWORD=your_password       # Windows
```

### Issue: Slow performance
**Solution**: 
- Run stages 0-5 only for smoke test (skip NB2/NB3)
- Check available RAM (NB2/NB3 can use 4-8GB)
- Use `--start-stage` to skip completed stages

---

## After Smoke Test

### Success Checklist
- [ ] No ERROR/CRITICAL logs
- [ ] `features_daily.csv` created with expected rows
- [ ] `features_daily_labeled.csv` has label columns
- [ ] No Python exceptions
- [ ] Runtime within expected range

### Next Steps
- ✅ **Smoke test passed** → Proceed with full pipeline or development
- ❌ **Smoke test failed** → Check logs, verify data, review configuration

---

## Additional Commands

### Help
```bash
python -m scripts.run_full_pipeline --help
```

### Full Pipeline with All Stages
```bash
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --start-stage 0 \
  --end-stage 9
```

### Re-run Specific Stage
```bash
# Re-run only Stage 3 (Labels)
python -m scripts.run_full_pipeline \
  --participant P000001 \
  --snapshot 2025-09-29 \
  --start-stage 3 \
  --end-stage 3
```

---

**Last Updated**: 2025-11-16 (Finishing Pass - Step D)  
**Maintainer**: Rodrigo Marques Teixeira
