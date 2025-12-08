# HRV (Heart Rate Variability) Extension to Cardio Domain

**Date**: 2025-01-20  
**Scope**: Stage 1 (CSV Aggregation) and Stage 2 (Daily Unification)  
**Domain**: cardio

## Overview

This document describes the implementation of HRV (Heart Rate Variability, SDNN) extraction and aggregation into the existing cardio domain of the ETL pipeline.

## Key Constraints

- **Domain name**: `cardio` (not hr)
- **Strictly additive**: NO renaming of existing HR metrics
- **Backward compatible**: Existing HR columns unchanged
- **Stages affected**: 1 and 2 only (NO changes to QC, PBSI, or ML code)
- **Apple-only**: Zepp does NOT have HRV data

## New Columns Added

| Column            | Type  | Unit  | Description                        |
| ----------------- | ----- | ----- | ---------------------------------- |
| `hrv_sdnn_mean`   | float | ms    | Daily mean SDNN                    |
| `hrv_sdnn_median` | float | ms    | Daily median SDNN                  |
| `hrv_sdnn_min`    | float | ms    | Daily minimum SDNN                 |
| `hrv_sdnn_max`    | float | ms    | Daily maximum SDNN                 |
| `n_hrv_sdnn`      | int   | count | Number of HRV measurements per day |

**Note**: NaN (not 0) for days without HRV data.

## Implementation Details

### Stage 1: `src/etl/stage_csv_aggregation.py`

#### New Method: `AppleHealthAggregator.aggregate_hrv()`

- Parses Apple Health `HKQuantityTypeIdentifierHeartRateVariabilitySDNN` records
- Uses binary regex streaming (matching HR approach for performance)
- Applies outlier filtering: 5 ms ≤ SDNN ≤ 300 ms (biologically plausible range)
- Outputs daily statistics: mean, median, min, max, count

#### Cache Files Created

| File                                     | Purpose                     |
| ---------------------------------------- | --------------------------- |
| `.cache/export_apple_hrv_events.parquet` | Event-level HRV data for QC |
| `.cache/export_apple_hrv_daily.parquet`  | Daily aggregated HRV data   |

Cache columns use canonical `apple_` prefix (e.g., `apple_hrv_sdnn_mean`).

#### Modified: `AppleHealthAggregator.aggregate_all()`

- Calls both `aggregate_heartrate()` and `aggregate_hrv()`
- Merges HR + HRV into unified `daily_cardio` DataFrame (outer join on date)
- Ensures schema consistency (empty HRV columns if no data)

### Stage 2: `src/etl/stage_unify_daily.py`

#### Modified: `DailyUnifier.unify_cardio()`

- HRV columns pass through from Apple directly (no Zepp HRV exists)
- When merging Apple + Zepp:
  1. HR columns are averaged across sources
  2. HRV columns are preserved from Apple only (re-joined after merge)
- Ensures all cardio columns exist in output for schema consistency

## Output Files Affected

| File                         | New Columns                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------- |
| `daily_cardio.csv`           | `hrv_sdnn_mean`, `hrv_sdnn_median`, `hrv_sdnn_min`, `hrv_sdnn_max`, `n_hrv_sdnn` |
| `features_daily_unified.csv` | Same HRV columns                                                                 |

## HRV SDNN Background

SDNN (Standard Deviation of NN intervals) is a time-domain HRV metric measured in milliseconds. It reflects overall autonomic nervous system activity:

- **Higher SDNN**: Better autonomic regulation, lower stress
- **Lower SDNN**: Reduced heart rate variability, possible stress/fatigue

Apple Watch measures HRV during sleep and rest periods.

## Validation

After running ETL pipeline:

1. Check `daily_cardio.csv` contains HRV columns
2. Verify `features_daily_unified.csv` has HRV data
3. Confirm NaN (not 0) for days without HRV measurements
4. Verify existing HR columns unchanged

## Files Modified

1. `src/etl/stage_csv_aggregation.py`

   - Added `aggregate_hrv()` method (~200 lines)
   - Modified `aggregate_all()` to merge HR + HRV

2. `src/etl/stage_unify_daily.py`
   - Modified `unify_cardio()` to pass through HRV columns

## Future Considerations

- QC audit could be extended to validate HRV outlier filtering
- ML models could incorporate HRV as additional cardio features
- Consider HRV time-of-day analysis (morning vs. evening measurements)
