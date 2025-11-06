# Quick Reference: ETL Fixes Applied

## Files Modified

### 1. `src/domains/activity/activity_from_extracted.py`

- ✅ Added `import time` for QC timing
- ✅ FIX #1: Timezone normalization with home_tz parameter
- ✅ FIX #2: ActivitySummary dateComponents attribute support
- ✅ FIX #5: Enhanced QC metadata (8 fields: date_min, date_max, n_days, n_days_with_data, coverage_pct, source_files, source_summary, processing_time_sec, processed_at)

### 2. `src/domains/activity/zepp_activity.py`

- ✅ FIX #3: Added 4 new Zepp columns:
  - zepp_act_cal_total
  - zepp_act_sedentary_min
  - zepp_act_sport_sessions
  - zepp_act_score_daily
- ✅ Graceful fallbacks for missing columns (default to 0 or 0.0)
- ✅ Applied to both ACTIVITY and HEALTH_DATA paths

### 3. `src/domains/sleep/sleep_from_extracted.py`

- ✅ Added `import time` and `import json`
- ✅ FIX #4: Timezone normalization in \_agg_daily() and \_agg_intervals()
- ✅ Functions now accept home_tz parameter (default "UTC")
- ✅ main() reads home_tz from participant profile
- ✅ FIX #5: Enhanced QC with coverage_pct, source_files, processed_at

### 4. `src/domains/cardiovascular/cardio_from_extracted.py`

- ✅ Added `import time` for potential future QC enhancements

### 5. `requirements/base.txt`

- ✅ FIX #6: Added `lxml>=4.9.0` for efficient XML streaming

---

## Key Changes Summary

| Fix # | Issue                               | Solution                                         | Impact                                          |
| ----- | ----------------------------------- | ------------------------------------------------ | ----------------------------------------------- |
| #1    | Dates using UTC instead of local TZ | Pass home_tz to all date extraction functions    | Correct day boundaries per participant timezone |
| #2    | ActivitySummary entries missed      | Check dateComponents attribute first             | Capture all Apple health summary data           |
| #3    | Only 5 Zepp activity columns        | Added 4 new metrics with fallbacks               | Richer activity diagnostics from Zepp           |
| #4    | Sleep intervals in wrong timezone   | Use home_tz for sleep start_time→date conversion | Correct sleep date assignment per local TZ      |
| #5    | Minimal QC data (4 fields)          | Expanded to 8-9 fields with timing & coverage    | Better diagnostics for ETL troubleshooting      |
| #6    | No efficient XML parser available   | Add lxml dependency                              | Faster, memory-safe large export.xml parsing    |

---

## Testing Commands

### Verify Syntax

```bash
cd /c/dev/practicum2-nof1-adhd-bd
python -m py_compile \
  src/domains/activity/activity_from_extracted.py \
  src/domains/activity/zepp_activity.py \
  src/domains/sleep/sleep_from_extracted.py
```

### Run Activity Seed (via Makefile)

```bash
cd /c/dev/practicum2-nof1-adhd-bd
make etl activity  # dry-run default; add DRY_RUN=0 for real run
```

### Run Sleep Seed (via Makefile)

```bash
cd /c/dev/practicum2-nof1-adhd-bd
make etl sleep
```

### Check QC Output

```bash
cat data/etl/P000001/2025-11-06/qc/activity_seed_qc.csv
cat data/etl/P000001/2025-11-06/qc/sleep_seed_qc.csv
```

### Verify Timezone in Features

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/etl/P000001/2025-11-06/features/activity/features_daily.csv')
print(f'Date range: {df[\"date\"].min()} → {df[\"date\"].max()}')
print(f'Apple active_kcal populated: {df[\"apple_active_kcal\"].notna().sum()}/{len(df)}')
"
```

---

## Backward Compatibility

✅ **All changes are non-breaking:**

- New QC fields are additive (not breaking existing readers)
- New Zepp columns gracefully default to 0 when absent
- CLI interfaces unchanged (--pid, --snapshot, --dry-run, --allow-empty work as before)
- Timezone changes transparent to callers (only affects internal date computation)
- ActivitySummary parsing is backward compatible (still handles date/startDate fallbacks)

---

## Participant Profile Configuration

For timezone fixes to work correctly, ensure `data/config/<PID>_profile.json` exists with:

```json
{
  "home_tz": "America/Sao_Paulo",  // or appropriate IANA timezone
  ...other fields...
}
```

**Fallback:** If file missing or field absent, defaults to "UTC".

---

## Generated Documentation

Full details available in: **`FIXES_APPLIED.md`**

This document contains:

- Detailed before/after code samples for each fix
- Impact analysis per fix
- Testing & validation checklist
- Deployment steps
- Compatibility notes

---

## Next Steps

1. Run integration tests with test snapshot
2. Verify QC CSV outputs have all 8-9 fields
3. Confirm timezone correctness with known participant profile
4. Validate Zepp activity 9-column output
5. Check ActivitySummary capture completeness
6. Monitor for any downstream effects (join/enrich/labels/aggregate)
