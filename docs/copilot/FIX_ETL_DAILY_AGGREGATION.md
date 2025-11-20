# ETL Data Flow Fix: Proper Daily Aggregation

## Problem

The `joined_features_daily.csv` file currently contains **multiple rows per day** (intra-day data is not aggregated). This breaks downstream ML6 baseline modeling which expects **one row per date**.

- Current state: `joined_features_daily.csv` has 284,049 rows (raw intra-day data)
- Expected state: `joined_features_daily.csv` should have ~2,700 rows (one per day)

## Root Cause

The `join_run()` function in `src/etl_pipeline.py` (line 3142):

1. Reads per-domain daily features (which may have multiple rows per day from concat of vendor/variants)
2. Concatenates domains without aggregating intra-day rows
3. Merges on date without first ensuring uniqueness

Result: Raw intra-day data propagates into joined output.

## Solution

Add **daily aggregation step** before merge in `join_run()`:

```python
# AFTER: concat per-domain rows (provenance preserved)
concat_df = pd.concat(parts, ignore_index=True, sort=False)

# NEW: Aggregate to one row per domain per date
# Group by (date, source_domain) and take mean of numeric columns
if "date" in concat_df.columns:
    concat_df["date"] = pd.to_datetime(concat_df["date"], errors="coerce")

    # Aggregate numeric columns by mean, keep first value for non-numeric
    numeric_cols = concat_df.select_dtypes(include=[np.number]).columns.tolist()
    agg_spec = {col: "mean" for col in numeric_cols}
    agg_spec["source_domain"] = "first"  # Keep domain indicator

    concat_df = concat_df.groupby(["date", "source_domain"], as_index=False).agg(agg_spec)
```

## Implementation Steps

1. **Fix `join_run()` in `src/etl_pipeline.py`**:

   - Add aggregation after per-domain concat
   - Ensure each (date, domain) pair appears once

2. **Verify daily uniqueness**:

   - After join completes, assert that final merged has unique dates
   - Log aggregation stats per domain

3. **Update `build_heuristic_labels.py`**:

   - Input: `joined_aggregate.csv` (from corrected join_run)
   - Output: `features_daily_labeled.csv` with heuristic labels
   - Preserves daily structure

4. **Update ML6**:
   - Reads `features_daily_labeled.csv` (guaranteed one row per date)
   - Label column: `label_final`

## Data Lineage

```
raw data (intra-day, multiple vendors)
    ↓
features/<domain>/<vendor>/<variant>/features_daily.csv (raw intra-day per domain)
    ↓
[NEW] Aggregate by (date, domain) to daily → concat
    ↓
joined/joined_aggregate.csv (one row per date, multiple domains)
    ↓
build_heuristic_labels.py
    ↓
joined/features_daily_labeled.csv (daily, labeled)
    ↓
NB2_Baseline.py
```

## Verification

After fix, check:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-11-07/joined/joined_features_daily.csv', parse_dates=['date'])
print(f'Rows: {len(df)}')
print(f'Unique dates: {df[\"date\"].nunique()}')
print(f'One row per date: {len(df) == df[\"date\"].nunique()}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
"
```

Expected: `One row per date: True`
