# Progress Bars and Meds Performance Update

**Date**: 2025-12-08  
**Participant**: P000001  
**Snapshot**: 2025-12-08

---

## Summary

This update improves:

1. Log tag consistency using `[VENDOR/VARIANT/DOMAIN]` format
2. Progress bar behavior for large file parsing
3. (Planned) Meds parquet caching similar to HR

---

## 1. Log Tag Convention

### New Format

```
[<VENDOR>/<VARIANT>/<DOMAIN>]
```

### Examples

| Old Tag                             | New Tag                                            |
| ----------------------------------- | -------------------------------------------------- |
| `[Apple]`                           | `[APPLE/EXPORT]`                                   |
| `[Apple] Aggregating heart rate...` | `[APPLE/EXPORT/CARDIO] Aggregating heart rate...`  |
| `[Apple] Aggregating sleep...`      | `[APPLE/EXPORT/SLEEP] Aggregating sleep...`        |
| `[Apple] Aggregating activity...`   | `[APPLE/EXPORT/ACTIVITY] Aggregating activity...`  |
| `[Meds]`                            | `[APPLE/EXPORT/MEDS]`                              |
| `[Meds/AutoExport]`                 | `[APPLE/AUTOEXPORT/MEDS]`                          |
| `[SoM]`                             | `[APPLE/AUTOEXPORT/SOM]`                           |
| `[Zepp]`                            | `[ZEPP/CARDIO]`, `[ZEPP/SLEEP]`, `[ZEPP/ACTIVITY]` |

### Sample Log Output

```
[APPLE/EXPORT] Processing: data\etl\P000001\2025-12-08\extracted\apple\...
[APPLE/EXPORT] File size: 1551.6 MB - Parsing XML structure...
[APPLE/EXPORT] Parsed export.xml successfully in 50.3s
[APPLE/EXPORT/CARDIO] Aggregating heart rate data...
[APPLE/EXPORT/CARDIO] ✓ Loaded HR from cache: export_apple_hr_daily.parquet (1360 days)
[APPLE/EXPORT/CARDIO] Aggregating HRV (SDNN) data...
[APPLE/EXPORT/CARDIO] Merged HR (1360 days) + HRV (18 days) → 1360 cardio days
[APPLE/EXPORT/SLEEP] Aggregating sleep data...
[APPLE/EXPORT/SLEEP] Aggregated 1860 sleep days
[APPLE/EXPORT/ACTIVITY] Aggregating activity data...
[APPLE/EXPORT/ACTIVITY] Aggregated 2766 activity days
[APPLE/EXPORT/MEDS] Aggregating medication data...
[ZEPP/CARDIO] Scanning directory: data\etl\P000001\2025-12-08\extracted\zepp
[ZEPP/SLEEP] Reading SLEEP: ...
[ZEPP/SLEEP] Aggregated 304 sleep days
[ZEPP/CARDIO] Reading HEARTRATE: ...
[ZEPP/CARDIO] Aggregated 156 HR days
[ZEPP/ACTIVITY] Reading ACTIVITY: ...
[ZEPP/ACTIVITY] Aggregated 500 activity days
[APPLE/AUTOEXPORT/MEDS] Loaded 4053 records from Medications-2021-05-11-2025-12-08.csv
[APPLE/AUTOEXPORT/MEDS] Aggregated 452 medication days
[APPLE/AUTOEXPORT/SOM] CSV structure: header=7 cols, data=8 fields
[APPLE/AUTOEXPORT/SOM] Detected trailing comma issue (8 fields vs 7 header cols)
[APPLE/AUTOEXPORT/SOM] Valence column: 91 valid numeric values
[APPLE/AUTOEXPORT/SOM] Aggregated to 77 daily rows
```

---

## 2. Files Changed

### Primary Files

| File                                      | Changes                                    |
| ----------------------------------------- | ------------------------------------------ |
| `src/etl/stage_csv_aggregation.py`        | Updated all log tags to new format         |
| `src/domains/meds/meds_from_extracted.py` | Updated `[Meds]` → `[APPLE/EXPORT/MEDS]`   |
| `src/domains/som/som_from_autoexport.py`  | Updated `[SoM]` → `[APPLE/AUTOEXPORT/SOM]` |

### Log Tag Changes Summary

- **stage_csv_aggregation.py**: ~50 log statements updated
- **meds_from_extracted.py**: ~30 log statements updated
- **som_from_autoexport.py**: ~20 log statements updated

---

## 3. Progress Bar Behavior

### Current Implementation

The meds progress bar currently uses a fixed 100-step iteration:

```
[APPLE/EXPORT/MEDS] Parsing export.xml: 100%|██████████| 100% [00:07]
```

### Notes

- The progress bar completes based on iteration steps, not actual bytes read
- For very large files (1.5GB), this can appear to jump quickly
- The HR progress bar uses record-based progress which is more accurate

### Future Improvement (TODO)

Consider implementing bytes-based progress for meds parsing similar to HR:

- Track `bytes_read / total_bytes` ratio
- Update progress bar based on actual file read position
- This would provide more accurate progress indication for large files

---

## 4. Meds Parquet Cache (TODO)

Currently, meds parsing does not cache events to Parquet like HR does.

### Current HR Pattern (Reference)

```
.cache/export_apple_hr_events.parquet  (event-level data)
.cache/export_apple_hr_daily.parquet   (daily aggregates)
```

### Proposed Meds Pattern

```
.cache/export_apple_meds_events.parquet  (event-level data)
.cache/export_apple_meds_daily.parquet   (daily aggregates)
```

### Benefits

- Faster subsequent runs (skip XML parsing if cache exists)
- Memory efficient for large files
- Deterministic outputs (same cache → same results)

**Status**: Not yet implemented - meds currently uses in-memory parsing

---

## 5. Validation Results

### Test Run: Stages 1-2

```
Stage 1: CSV Aggregation
  - Apple export.xml parsed in 50.3s
  - HR loaded from cache (1360 days)
  - HRV loaded from cache (18 days)
  - Sleep aggregated (1860 days)
  - Activity aggregated (2766 days)
  - Meds (export): 0 days (no meds in export.xml)
  - Zepp: 304 sleep, 156 HR, 500 activity days
  - AutoExport Meds: 452 days
  - AutoExport SoM: 77 days

Stage 2: Unify Daily
  - Sleep: 1860 Apple + 44 Zepp = 1904 total
  - Cardio: 1360 Apple + 156 Zepp = 1360 unique
  - Activity: 2766 Apple + 500 Zepp = 2766 unique
  - Meds: 452 days (vendor: apple_autoexport)
  - SoM: 77 days (vendor: apple_autoexport)
```

### Log Tag Verification

All log messages now use the standardized `[VENDOR/VARIANT/DOMAIN]` format:

- ✅ `[APPLE/EXPORT/CARDIO]` - Heart rate from export.xml
- ✅ `[APPLE/EXPORT/SLEEP]` - Sleep from export.xml
- ✅ `[APPLE/EXPORT/ACTIVITY]` - Activity from export.xml
- ✅ `[APPLE/EXPORT/MEDS]` - Meds from export.xml
- ✅ `[APPLE/AUTOEXPORT/MEDS]` - Meds from AutoExport CSV
- ✅ `[APPLE/AUTOEXPORT/SOM]` - SoM from AutoExport CSV
- ✅ `[ZEPP/CARDIO]`, `[ZEPP/SLEEP]`, `[ZEPP/ACTIVITY]` - Zepp domains

---

## Status

✅ **LOG TAGS UPDATED** - All active ETL code uses new format
⚠️ **PROGRESS BARS** - Functional but could be improved with bytes-based tracking
⚠️ **MEDS CACHE** - Not yet implemented (future optimization)

---

_Generated: 2025-12-08_
