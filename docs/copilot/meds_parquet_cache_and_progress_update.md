# MEDS Domain: Parquet Caching and Progress Bar Update

**Date:** 2024-12-08  
**Author:** Copilot  
**Related:** HR/CARDIO domain caching pattern in `src/etl/stage_csv_aggregation.py`

## Summary

Updated `src/domains/meds/meds_from_extracted.py` to use:

1. **Parquet caching** (same pattern as HR/CARDIO)
2. **Byte-based progress bar** (accurate file read tracking)

## Changes Made

### 1. Parquet Caching Strategy

Added layered cache loading (same as HR domain):

```
1. Try daily cache (.cache/export_apple_meds_daily.parquet) - fastest
2. Else try events cache (.cache/export_apple_meds_events.parquet) + re-aggregate
3. Else parse XML with binary regex and save both caches
```

**Cache location:** Same `.cache/` directory as HR/HRV:

```
data/etl/<PID>/<SNAPSHOT>/extracted/apple/apple_health_export/.cache/
├── export_apple_hr_daily.parquet      # HR daily (pre-existing)
├── export_apple_hr_events.parquet     # HR events (pre-existing)
├── export_apple_hrv_daily.parquet     # HRV daily (pre-existing)
├── export_apple_hrv_events.parquet    # HRV events (pre-existing)
├── export_apple_meds_daily.parquet    # NEW: MEDS daily
└── export_apple_meds_events.parquet   # NEW: MEDS events
```

### 2. Byte-Based Progress Bar

Replaced simulated threading-based progress bar with actual byte tracking:

**Before (simulated):**

```python
# Threading-based fake progress (estimated time)
progress_thread = threading.Thread(target=show_scan_progress, daemon=True)
progress_thread.start()
with open(self.xml_path, 'rb') as f:
    content = f.read()  # Blocking, no progress
```

**After (actual bytes):**

```python
# Byte-based progress bar (accurate tracking)
with _make_bytes_pbar(self.file_size_bytes, "[APPLE/EXPORT/MEDS] Reading XML") as pbar:
    with open(self.xml_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)  # 64MB chunks
            if not chunk:
                break
            content_chunks.append(chunk)
            pbar.update(len(chunk))
```

**Progress bar output:**

```
[APPLE/EXPORT/MEDS] Reading XML: 100%|██████████████████| 1.52G/1.52G [00:02<00:00, 802MB/s]
```

### 3. New Helper Functions

Added progress bar helpers (consistent with `stage_csv_aggregation.py`):

```python
def _make_pbar(total: int, desc: str):
    """Create a tqdm progress bar for record counting."""
    return tqdm(total=total, desc=desc, unit="rec", ...)

def _make_bytes_pbar(total_bytes: int, desc: str):
    """Create a byte-based tqdm progress bar."""
    return tqdm(total=total_bytes, desc=desc, unit="B", unit_scale=True, ...)
```

### 4. MedsAggregator Class Updates

**New attributes:**

```python
self.file_size_bytes = self.xml_path.stat().st_size
self.use_cache = use_cache
self.cache_dir = self.xml_path.parent / ".cache"
self.cache_file_daily = self.cache_dir / f"{self.xml_path.stem}_apple_meds_daily.parquet"
self.cache_file_events = self.cache_dir / f"{self.xml_path.stem}_apple_meds_events.parquet"
```

**New methods:**

- `_load_from_cache() -> Optional[pd.DataFrame]`
- `_aggregate_events_to_daily(df_events) -> pd.DataFrame`
- `_save_events_cache(events: List[Dict]) -> None`
- `_save_daily_cache(df_daily: pd.DataFrame) -> None`

### 5. Public API Update

Updated `load_apple_meds_daily()` signature:

```python
def load_apple_meds_daily(
    export_xml_path: Path | str,
    home_tz: str = "UTC",
    max_records: int | None = None,
    use_cache: bool = True,  # NEW parameter
) -> pd.DataFrame:
```

**Backward compatible:** Default `use_cache=True` enables caching automatically.

## Performance Impact

| Scenario                | Before                    | After                                    |
| ----------------------- | ------------------------- | ---------------------------------------- |
| First parse (1.5GB XML) | ~2-3s (file read) + regex | ~2s (chunked read with progress) + regex |
| Cached load             | N/A                       | <0.1s (Parquet read)                     |
| Progress accuracy       | Simulated (fake %)        | Actual bytes/second                      |

## Schema (Unchanged)

Public MEDS schema remains identical:

```
date            - YYYY-MM-DD
med_any         - 0 or 1 (binary indicator)
med_event_count - integer count
med_dose_total  - float (NaN for export.xml)
med_names       - comma-separated strings
med_sources     - comma-separated strings
```

## Testing

```bash
# Test import
python -c "from src.domains.meds.meds_from_extracted import load_apple_meds_daily; print('OK')"

# Test cache behavior
python -c "
from src.domains.meds.meds_from_extracted import load_apple_meds_daily
from pathlib import Path
xml = Path('data/etl/P000001/2025-12-08/extracted/apple/apple_health_export/export.xml')
df = load_apple_meds_daily(xml)  # Uses cache if exists
print(f'Loaded: {len(df)} rows')
"
```

## Files Modified

- `src/domains/meds/meds_from_extracted.py`:
  - Added `_make_pbar()`, `_make_bytes_pbar()` helpers
  - Added cache methods to `MedsAggregator` class
  - Updated `aggregate_medications_binary_regex()` with caching
  - Updated `load_apple_meds_daily()` with `use_cache` parameter
