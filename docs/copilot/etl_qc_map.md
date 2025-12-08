# ETL QC / Audit Layer Documentation

> **Document type**: Technical reference  
> **Last Updated**: 2025-12-08  
> **Author**: Copilot  
> **Version**: v4.4.0 (Domain naming consistency fix)

---

## A) Overview

### Main Goal

The ETL QC/Audit layer provides **regression testing for data integrity** across the ETL pipeline. It verifies that:

1. Raw files exist and are being consumed
2. Extracted daily CSVs are consistent (no duplicates, expected coverage)
3. Unified join is valid (no duplicate dates, monotonic ordering)
4. Domain-specific data quality metrics meet thresholds

### Signals Currently Covered

| Domain                  | QC Module                                     | Makefile Target       | Status  |
| ----------------------- | --------------------------------------------- | --------------------- | ------- |
| **Cardio (HR)**         | `etl_audit.py::run_cardio_audit()`            | `make qc-cardio`      | ✅ Full |
| **Activity (Steps)**    | `etl_audit.py::run_activity_audit()`          | `make qc-activity`    | ✅ Full |
| **Sleep**               | `etl_audit.py::run_sleep_audit()`             | `make qc-sleep`       | ✅ Full |
| **Meds**                | `etl_audit.py::run_meds_audit()`              | `make qc-meds`        | ✅ Full |
| **SoM (State of Mind)** | `etl_audit.py::run_som_audit()`               | `make qc-som`         | ✅ Full |
| **Unified Extension**   | `etl_audit.py::run_unified_extension_audit()` | `make qc-unified-ext` | ✅ Full |
| **Labels**              | `etl_audit.py::run_labels_audit()`            | `make qc-labels`      | ✅ Full |

### File Locations

```
src/etl/
├── etl_audit.py                              # Main QC module (~1800 lines)
├── hr_daily_aggregation_consistency_check.py # Specialized HR consistency QC (587 lines)
└── sleep_hourly_audit.py                     # Specialized sleep hourly audit (890 lines)
```

### Output Locations

QC outputs are written to:

```
data/etl/<PID>/<SNAPSHOT>/qc/
├── hr_feature_audit.csv         # Per-day HR QC results
├── hr_audit_summary.json        # HR audit summary (PASS/FAIL + issues)
├── steps_feature_audit.csv      # Per-day Steps QC results
├── steps_audit_summary.json     # Steps audit summary
├── sleep_feature_audit.csv      # Per-day Sleep QC results
├── sleep_audit_summary.json     # Sleep audit summary
├── meds_feature_audit.csv       # Per-day Meds QC results (NEW)
├── meds_audit_summary.json      # Meds audit summary (NEW)
├── som_feature_audit.csv        # Per-day SoM QC results (NEW)
├── som_audit_summary.json       # SoM audit summary (NEW)
├── unified_ext_audit_summary.json # Unified extension summary (NEW)
└── labels_audit_summary.json    # Labels layer summary (NEW)
```

Additional specialized QC outputs:

```
data/ai/<PID>/<SNAPSHOT>/qc/
├── hr_daily_aggregation_diff.csv                 # HR consistency diff
├── hr_daily_aggregation_consistency_report.md    # HR consistency report
├── sleep_hourly_audit.md                         # Sleep hourly audit report
└── sleep_classification.csv                      # Sleep night classification
```

Human-readable Markdown reports (NEW):

```
docs/reports/qc/
├── QC_latest.md                                    # Pointer to latest QC reports
├── QC_<PID>_<SNAPSHOT>_meds_<TIMESTAMP>.md        # Meds QC report
├── QC_<PID>_<SNAPSHOT>_som_<TIMESTAMP>.md         # SoM QC report
├── QC_<PID>_<SNAPSHOT>_unified_ext_<TIMESTAMP>.md # Unified extension report
└── QC_<PID>_<SNAPSHOT>_labels_<TIMESTAMP>.md      # Labels layer report
```

---

## B) Checks per Signal

### B.1) Cardio (Heart Rate) Domain

**Module**: `etl_audit.py::run_cardio_audit()`

#### Input Files

| File                                                                        | Layer     | Purpose                                  |
| --------------------------------------------------------------------------- | --------- | ---------------------------------------- |
| `extracted/apple/apple_health_export/.cache/export_apple_hr_events.parquet` | Extracted | Event-level HR records (source of truth) |
| `extracted/apple/apple_health_export/.cache/export_apple_hr_daily.parquet`  | Extracted | Daily aggregated HR cache                |
| `extracted/apple/daily_cardio.csv`                                          | Extracted | Daily HR CSV used by pipeline            |
| `joined/features_daily_unified.csv`                                         | Joined    | Unified dataset                          |
| `joined/features_daily_labeled.csv`                                         | Joined    | Labeled dataset                          |

#### Tests Performed

| Test                       | Description                                                                                                    | Severity |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- | -------- |
| **Parquet Event Exists**   | Event-level Parquet must exist                                                                                 | CRITICAL |
| **Parquet Daily Exists**   | Daily cache Parquet must exist                                                                                 | CRITICAL |
| **Schema Validation**      | Daily cache must have 5 metrics: `apple_hr_mean`, `apple_hr_min`, `apple_hr_max`, `apple_hr_std`, `apple_n_hr` | CRITICAL |
| **Fabrication Rate (min)** | % of days where `hr_min == hr_mean` must be < 10%                                                              | CRITICAL |
| **Fabrication Rate (std)** | % of days where `hr_std == 0` must be < 10%                                                                    | CRITICAL |
| **CSV Consistency**        | Row count in cache must match daily_cardio.csv                                                                 | WARNING  |
| **Unified HR Columns**     | HR columns present in unified CSV                                                                              | WARNING  |

#### Metrics Collected

```python
# Per-day CSV output (hr_feature_audit.csv)
{
    'date': str,
    'hr_mean': float,
    'hr_min': float,
    'hr_max': float,
    'hr_std': float,
    'hr_samples': int,
    'is_fabricated_min': bool,   # hr_min == hr_mean
    'is_fabricated_std': bool,   # hr_std == 0.0
    'is_single_sample': bool     # n_hr == 1
}
```

#### Pass/Fail Criteria

- **PASS**: No CRITICAL issues found
- **FAIL**: Any CRITICAL issue (Parquet missing, schema invalid, fabrication > 10%)

---

### B.2) Activity (Steps) Domain

**Module**: `etl_audit.py::run_activity_audit()`

#### Input Files

| File                                 | Layer     | Purpose         |
| ------------------------------------ | --------- | --------------- |
| `extracted/apple/daily_activity.csv` | Extracted | Daily steps CSV |
| `joined/features_daily_unified.csv`  | Joined    | Unified dataset |
| `joined/features_daily_labeled.csv`  | Joined    | Labeled dataset |

#### Tests Performed

| Test                    | Description                                      | Severity |
| ----------------------- | ------------------------------------------------ | -------- |
| **Activity CSV Exists** | `daily_activity.csv` must exist                  | CRITICAL |
| **Steps Column Exists** | Must have column with "step" in name             | CRITICAL |
| **All-NaN Check**       | No steps column should be all-NaN in unified CSV | CRITICAL |
| **Unified Coverage**    | Steps columns present in unified CSV             | WARNING  |

#### Metrics Collected

```python
# Per-day CSV output (steps_feature_audit.csv)
{
    'date': str,
    'steps': int,
    'has_data': bool
}
```

#### Pass/Fail Criteria

- **PASS**: CSV exists, steps column found, no all-NaN columns
- **FAIL**: Missing CSV, missing column, or all-NaN column

---

### B.3) Sleep Domain

**Module**: `etl_audit.py::run_sleep_audit()`

#### Input Files

| File                                | Layer     | Purpose         |
| ----------------------------------- | --------- | --------------- |
| `extracted/apple/daily_sleep.csv`   | Extracted | Daily sleep CSV |
| `joined/features_daily_unified.csv` | Joined    | Unified dataset |
| `joined/features_daily_labeled.csv` | Joined    | Labeled dataset |

#### Tests Performed

| Test                       | Description                                        | Severity |
| -------------------------- | -------------------------------------------------- | -------- |
| **Sleep CSV Exists**       | `daily_sleep.csv` must exist                       | CRITICAL |
| **Sleep Hours Range**      | All values must be in [0, 24]                      | CRITICAL |
| **Sleep Efficiency Range** | If efficiency column exists, values in [0, 1]      | WARNING  |
| **Sleep Hours Column**     | Must have column with "hour" or "duration" in name | WARNING  |
| **Unified Coverage**       | Sleep columns present in unified CSV               | WARNING  |

#### Metrics Collected

```python
# Per-day CSV output (sleep_feature_audit.csv)
{
    'date': str,
    'sleep_hours': float,
    'has_sleep': bool,           # sleep_hours > 0
    'is_valid_range': bool,      # 0 <= sleep_hours <= 24
    'sleep_efficiency': float    # Optional, if column exists
}
```

#### Pass/Fail Criteria

- **PASS**: CSV exists, sleep hours in valid range
- **FAIL**: Missing CSV or invalid sleep hours (< 0 or > 24)

---

### B.4) Meds Domain (NEW in v4.3.1)

**Module**: `etl_audit.py::run_meds_audit()`

#### Input Files

| File                                        | Layer     | Purpose           |
| ------------------------------------------- | --------- | ----------------- |
| `extracted/apple/daily_meds_apple.csv`      | Extracted | Apple export meds |
| `extracted/apple/daily_meds_autoexport.csv` | Extracted | AutoExport meds   |
| `joined/features_daily_unified.csv`         | Joined    | Unified dataset   |

#### Tests Performed

| Test                           | Description                                                  | Severity       |
| ------------------------------ | ------------------------------------------------------------ | -------------- |
| **Source Existence**           | At least one meds CSV must exist                             | CRITICAL       |
| **Schema Validation**          | Required columns: date, med_any, med_event_count, etc.       | CRITICAL       |
| **Unified Columns**            | med_any, med_event_count, med_names, med_sources, med_vendor | CRITICAL       |
| **med_any ∈ {0,1}**            | Binary values only                                           | CRITICAL (>1%) |
| **med_event_count >= 0**       | Non-negative counts                                          | CRITICAL (>1%) |
| **med_dose_total >= 0**        | Non-negative doses (if present)                              | CRITICAL (>1%) |
| **Consistency when med_any=1** | event_count > 0, med_names not empty                         | CRITICAL (>1%) |
| **Valid med_vendor**           | ∈ {apple_export, apple_autoexport, fallback}                 | CRITICAL       |

#### Metrics Collected

```python
# Per-day CSV output (meds_feature_audit.csv)
{
    'date': str,
    'med_any': int,
    'med_event_count': int,
    'med_dose_total': float,  # if exists
    'med_names': str,
    'med_vendor': str,
    'has_meds': bool,
    'is_valid': bool
}
```

#### Pass/Fail Criteria

- **PASS**: At least one source exists, schema valid, data quality issues < 1%
- **FAIL**: No sources, missing columns, or data quality issues > 1%

---

### B.5) SoM Domain (NEW in v4.3.1)

**Module**: `etl_audit.py::run_som_audit()`

#### Input Files

| File                                       | Layer     | Purpose         |
| ------------------------------------------ | --------- | --------------- |
| `extracted/apple/daily_som_autoexport.csv` | Extracted | AutoExport SoM  |
| `joined/features_daily_unified.csv`        | Joined    | Unified dataset |

#### Tests Performed

| Test                               | Description                                                       | Severity       |
| ---------------------------------- | ----------------------------------------------------------------- | -------------- |
| **Source/Unified Existence**       | SoM source or unified columns must exist                          | CRITICAL       |
| **Schema Validation**              | Required columns: date, som_mean_score, som_category_3class, etc. | CRITICAL       |
| **som_category_3class ∈ {-1,0,1}** | Valid category values                                             | CRITICAL (>1%) |
| **som_n_entries >= 1**             | At least one entry per day                                        | CRITICAL (>1%) |
| **Valid som_vendor**               | ∈ {apple_autoexport, fallback}                                    | CRITICAL       |
| **Score ranges**                   | som_mean_score, som_last_score ∈ [-1, 1]                          | WARNING        |
| **Temporal gap check**             | Max gap > 90 days                                                 | WARNING        |

#### Metrics Collected

```python
# Per-day CSV output (som_feature_audit.csv)
{
    'date': str,
    'som_mean_score': float,
    'som_last_score': float,
    'som_n_entries': int,
    'som_category_3class': int,
    'som_vendor': str,
    'has_som': bool,
    'is_valid': bool
}
```

#### Pass/Fail Criteria

- **PASS**: Source/unified exists, schema valid, category values valid
- **FAIL**: No source or unified columns, invalid categories

---

### B.6) Unified Extension (NEW in v4.3.1)

**Module**: `etl_audit.py::run_unified_extension_audit()`

#### Input Files

| File                                | Layer  | Purpose         |
| ----------------------------------- | ------ | --------------- |
| `joined/features_daily_unified.csv` | Joined | Unified dataset |

#### Tests Performed

| Test                   | Description                                  | Severity |
| ---------------------- | -------------------------------------------- | -------- |
| **File Existence**     | features_daily_unified.csv must exist        | CRITICAL |
| **No Duplicate Dates** | All dates unique                             | CRITICAL |
| **Monotonic Dates**    | Dates in ascending order                     | CRITICAL |
| **Valid med_vendor**   | ∈ {apple_export, apple_autoexport, fallback} | CRITICAL |
| **Valid som_vendor**   | ∈ {apple_autoexport, fallback}               | CRITICAL |
| **No all-NaN columns** | Key columns have at least some data          | WARNING  |

#### Metrics Collected (Phase 2)

- Vendor distribution for med_vendor and som_vendor
- Domain overlap histogram:
  - Days with meds only
  - Days with SoM only
  - Days with both
  - Days with neither

---

### B.7) Labels Layer (NEW in v4.3.1)

**Module**: `etl_audit.py::run_labels_audit()`

#### Input Files

| File                                | Layer  | Purpose         |
| ----------------------------------- | ------ | --------------- |
| `joined/features_daily_labeled.csv` | Joined | Labeled dataset |

#### Tests Performed

| Test                   | Description                                         | Severity |
| ---------------------- | --------------------------------------------------- | -------- |
| **File Existence**     | features_daily_labeled.csv (optional, not critical) | WARNING  |
| **Label Columns**      | pbsi_score, label_3cls, segment_id presence         | WARNING  |
| **Class Distribution** | Descriptive stats for label_3cls                    | INFO     |

#### Metrics Collected

- Total days, labeled days, unlabeled days
- Label ratio (% labeled)
- Class distribution for label_3cls (-1=Low, 0=Mid, 1=High)
- pbsi_score statistics (mean, std, min, max)
- segment_id count

---

### B.8) Generic Cross-Domain Checks

**Module**: `etl_audit.py::run_full_audit()` (includes these sections)

#### Section 1: Raw File Coverage (`audit_raw_files()`)

Checks:

- Apple ZIP files exist in `data/raw/<PID>/apple/export/`
- Zepp ZIP files exist in `data/raw/<PID>/zepp/`

Output: Logs file names and sizes

#### Section 2: Extracted Layer (`audit_extracted_layer()`)

For both Apple and Zepp vendors, checks:

- `daily_sleep.csv`, `daily_cardio.csv`, `daily_activity.csv` exist
- Row counts, date ranges, unique dates
- Duplicate dates detection
- Missing days in date range

#### Section 3: Unified Join (`audit_unified_join()`)

Checks:

- `features_daily_unified.csv` exists
- No duplicate dates (CRITICAL)
- No missing days in date range (WARNING)
- Monotonic date ordering (CRITICAL)
- Domain coverage histogram (Sleep, Cardio, Activity)
- Days with no data in any domain (WARNING)

#### Section 4: Sample Day Cross-Check (`audit_sample_days()`)

- Picks 3 random days from unified CSV
- Logs sleep, cardio, activity values for each
- (TODO: Cross-check against raw/extracted layers)

---

## C) How It Is Invoked

### C.1) Makefile Targets

```makefile
# -------- QC / Audits --------
qc-cardio:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain cardio

qc-activity:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain activity

qc-sleep:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain sleep

qc-meds:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain meds

qc-som:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain som

qc-unified-ext:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain unified_ext

qc-labels:
    $(PYTHON) -m src.etl.etl_audit --participant $(PID) --snapshot $(SNAPSHOT_RESOLVED) --domain labels

qc-all: qc-cardio qc-activity qc-sleep qc-meds qc-som qc-unified-ext qc-labels
    @echo "All domain audits complete"

# Alias for backward compatibility
qc-etl: qc-cardio
```

### C.2) CLI Usage

```bash
# Domain audits (canonical names)
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain cardio
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain activity
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain sleep
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain meds
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain som
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain unified_ext
python -m src.etl.etl_audit --participant P000001 --snapshot 2025-12-08 --domain labels

# Using Makefile
make qc-cardio PID=P000001 SNAPSHOT=2025-12-08
make qc-activity PID=P000001 SNAPSHOT=2025-12-08
make qc-meds PID=P000001 SNAPSHOT=2025-12-08
make qc-som PID=P000001 SNAPSHOT=2025-12-08
make qc-all PID=P000001 SNAPSHOT=2025-12-08
```

### C.3) Pipeline Integration

**The QC layer is NOT automatically invoked by `run_full_pipeline.py`.**

The pipeline creates the `qc/` directory structure but does not run audits:

```python
# scripts/run_full_pipeline.py (line 73)
self.qc_dir = self.snapshot_dir / "qc"
```

QC is designed to be run **manually after pipeline completion** as a regression/validation step.

### C.4) Exit Codes

| Code | Meaning                                  |
| ---- | ---------------------------------------- |
| `0`  | PASS - All checks passed                 |
| `1`  | FAIL - At least one CRITICAL issue found |

This enables CI/CD integration:

```bash
make qc-all && echo "QC PASSED" || echo "QC FAILED"
```

---

## D) Limitations / Gaps

### D.1) Domains NOT Covered by QC

| Domain                  | Gap Description                                                                                                                                  |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Meds**                | No `run_meds_audit()` function. No checks for `daily_meds_apple.csv` or `daily_meds_autoexport.csv`. No validation of med_vendor in unified CSV. |
| **SoM (State of Mind)** | No `run_som_audit()` function. No checks for `daily_som_autoexport.csv`. No validation of som_vendor in unified CSV.                             |

### D.2) Missing Cross-Vendor Consistency Checks

- No comparison between Apple and Zepp data for overlapping date ranges
- No validation that source prioritizer chose the correct vendor
- No check that vendor columns (`hr_vendor`, `sleep_vendor`, etc.) are populated correctly

### D.3) Missing Label Layer Validation

- Checks existence of `features_daily_labeled.csv` but does not validate:
  - Label column presence (`mood_label`, `energy_label`, etc.)
  - Label value distributions
  - Missing labels vs missing data correlation

### D.4) Missing AutoExport Validation

The new AutoExport sources added in Stage 1/2 refactoring are not covered:

### D.4) ~~Missing AutoExport Validation~~ ✅ RESOLVED

~~The new AutoExport sources added in Stage 1/2 refactoring are not covered:~~

- ~~`daily_meds_autoexport.csv` - not validated~~
- ~~`daily_som_autoexport.csv` - not validated~~
- ~~AutoExport-specific schema validation missing~~

**Resolution (v4.3.1)**: Added `run_meds_audit()` and `run_som_audit()` with full AutoExport schema validation.

### D.5) Incomplete Sample Day Cross-Check

The `audit_sample_days()` function has TODO items:

```python
# TODO: Check extracted Apple/Zepp CSVs for this date
# TODO: Check raw XML/CSV for this date (requires parsing)
```

### D.6) Zepp-Specific HR QC Missing

- `hr_daily_aggregation_consistency_check.py` only checks Apple HR data
- No equivalent consistency check for Zepp HR aggregation
- Zepp HR data quality thresholds not defined

### D.7) Provenance Layer Not Integrated

The `provenance/` directory contains audit reports but:

- Not generated by `etl_audit.py`
- Separate from QC directory structure
- Format differs from QC output (CSV files vs JSON summaries)

### D.8) Sleep Hourly Audit Partially Standalone

`sleep_hourly_audit.py` is a specialized module that:

- Has its own classification logic (real sleepless vs sensor missing)
- Outputs to `data/ai/<PID>/<SNAPSHOT>/qc/` (different from main QC)
- Not integrated into `qc-sleep` Makefile target
- Must be run separately

---

## Summary Table

| Component               | Cardio | Activity | Sleep | Meds | SoM | Unified Ext | Labels |
| ----------------------- | :----: | :------: | :---: | :--: | :-: | :---------: | :----: |
| Extraction audit        |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Schema validation       |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Data quality checks     |   ✅   |    ⚠️    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Unified join validation |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Per-day CSV output      |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |      —      |   —    |
| JSON summary output     |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Markdown report         |   —    |    —     |   —   |  ✅  | ✅  |     ✅      |   ✅   |
| Makefile target         |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |
| Exit code for CI/CD     |   ✅   |    ✅    |  ✅   |  ✅  | ✅  |     ✅      |   ✅   |

Legend:

- ✅ = Fully implemented
- ⚠️ = Partially implemented
- — = Not applicable

---

## Changelog

### v4.4.0 (2025-12-08)

**Domain Naming Consistency Fix:**

- Renamed `run_hr_audit()` → `run_cardio_audit()` to match canonical domain name
- Renamed `run_steps_audit()` → `run_activity_audit()` to match canonical domain name
- Renamed Makefile target `qc-hr` → `qc-cardio`
- Renamed Makefile target `qc-steps` → `qc-activity`
- Updated CLI choices from `hr`/`steps` to `cardio`/`activity`
- Updated `qc-all` and `qc-etl` dependencies accordingly
- **Note**: Internal HR/steps metrics and file names remain unchanged for backward compatibility

### v4.3.1 (2025-12-08)

**Phase 1 - Minimal QC Extensions:**

- Added `run_meds_audit()` with existence, schema, and data quality checks
- Added `run_som_audit()` with temporal coverage analysis
- Added `run_unified_extension_audit()` with vendor coverage stats
- Added Makefile targets: `qc-meds`, `qc-som`, `qc-unified-ext`
- Updated `qc-all` to include all new domains

**Phase 2 - Advanced Extensions:**

- Added `run_labels_audit()` for label layer QC
- Added Markdown report generation under `docs/reports/qc/`
- Added `QC_latest.md` auto-update functionality
- Added vendor coverage and domain overlap statistics
