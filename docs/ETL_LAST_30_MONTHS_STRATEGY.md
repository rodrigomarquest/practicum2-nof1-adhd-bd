# ETL Data Window Strategy: Last 30 Months

## Decision

Use **only the last 30 months** of data to maximize data quality and assertiveness:

- Removes early sparse data (poor data coverage)
- Ensures sufficient daily density for meaningful baselines
- Improves label distribution quality
- Reduces temporal artifacts from data collection improvements

## Implementation

### Step 1: Add date filtering in join_run()

After aggregation, filter to last 30 months:

```python
# After daily aggregation in join_run()
cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=30)
concat_df = concat_df[concat_df['date'] >= cutoff_date]
print(f"INFO: filtered to last 30 months (since {cutoff_date.date()}): {len(concat_df)} records")
```

### Step 2: Add CLI parameter for flexibility

Update `etl_pipeline.py` join_run to accept optional `months` parameter:

```python
def join_run(snapshot_dir: Path | str, *, dry_run: bool = False, months: int = 30) -> int:
    ...
    if months and months > 0:
        cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months)
        concat_df = concat_df[concat_df['date'] >= cutoff_date]
```

### Step 3: Update etl_pipeline.py main to pass parameter

When calling join_run() from main pipeline, pass `months=30`.

### Step 4: Re-run pipeline

```bash
cd c:/dev/practicum2-nof1-adhd-bd
python src/etl_pipeline.py extract --participant P000001 --snapshot 2025-11-07
python src/etl_pipeline.py cardio --participant P000001 --snapshot 2025-11-07 --zepp_dir data_etl/P000001/zepp_processed/2025-11-07
python src/etl_pipeline.py join --snapshot data/etl/P000001/2025-11-07
python build_heuristic_labels.py --pid P000001 --snapshot 2025-11-07 --verbose 1
```

## Data Quality Expectations

- **Before**: 284K rows (raw intra-day), ~2.7K daily rows over ~7 years
- **After**: ~900 daily rows (~30 months), better temporal density, cleaner labels

## Benefits

1. ✅ Removes sparse early data with poor coverage
2. ✅ Recent data has better collection consistency
3. ✅ 30-month window = ~900 days sufficient for 6 temporal folds (120+60 day windows)
4. ✅ Cleaner label distribution (fewer "unlabeled")
5. ✅ More assertive baseline predictions (denser temporal patterns)

## Verification

After filtering:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-11-07/joined/joined_features_daily.csv', parse_dates=['date'])
print(f'Rows (daily): {len(df)}')
print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')
print(f'Months: {(df[\"date\"].max() - df[\"date\"].min()).days / 30:.1f}')
print(f'Unique dates: {df[\"date\"].nunique()}')
"
```

Expected: ~900-1000 rows, 30 months duration
