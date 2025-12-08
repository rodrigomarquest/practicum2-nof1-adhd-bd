# Stage 0–2 Refactoring Analysis and Plan

**Created:** 2025-01-08  
**Last Updated:** 2025-01-08  
**Status:** DRAFT (pending user approval)

---

## 1. Current Behavior Summary

### Stage 0: Ingest (`stage_0_ingest()`)

**Location:** `scripts/run_full_pipeline.py` lines 194–310

| Vendor           | Source Directory                   | ZIP Date Selection            | Files Extracted                          |
| ---------------- | ---------------------------------- | ----------------------------- | ---------------------------------------- |
| Apple Export     | `raw/{PID}/apple/export/*.zip`     | **ALL ZIPs** (no date filter) | `export.xml`, clinical records           |
| Apple AutoExport | `raw/{PID}/apple/autoexport/*.zip` | **date <= snapshot** ✅       | `*.csv` (Medications, StateOfMind, etc.) |
| Zepp             | `raw/{PID}/zepp/*.zip`             | **ALL ZIPs** (no date filter) | `SLEEP*.csv`, `HEARTRATE*.csv`, etc.     |

**Key Finding:** Only AutoExport uses deterministic ZIP selection (`_select_autoexport_zip_for_snapshot()`). Apple export and Zepp extract ALL ZIPs regardless of snapshot date.

---

### Stage 1: Aggregate (`stage_1_aggregate()`)

**Location:** `scripts/run_full_pipeline.py` lines 322–351  
**Delegated to:** `src/etl/stage_csv_aggregation.py::run_csv_aggregation()`

| Domain   | Apple (export.xml)                                             | Zepp                                            | AutoExport            | Output File                             |
| -------- | -------------------------------------------------------------- | ----------------------------------------------- | --------------------- | --------------------------------------- |
| Sleep    | ✅ `AppleHealthAggregator.aggregate_sleep()`                   | ✅ `ZeppHealthAggregator.aggregate_sleep()`     | ❌                    | `daily_sleep.csv`                       |
| Cardio   | ✅ `AppleHealthAggregator.aggregate_heartrate()`               | ✅ `ZeppHealthAggregator.aggregate_heartrate()` | ❌                    | `daily_cardio.csv`                      |
| Activity | ✅ `AppleHealthAggregator.aggregate_activity()`                | ✅ `ZeppHealthAggregator.aggregate_activity()`  | ❌                    | `daily_activity.csv`                    |
| Meds     | ✅ `AppleHealthAggregator.aggregate_meds()` → `MedsAggregator` | ❌                                              | ❌ **NOT IN STAGE 1** | `daily_meds.csv` (from export.xml only) |
| SoM      | ❌                                                             | ❌                                              | ❌ **NOT IN STAGE 1** | N/A                                     |

**Key Finding:** Stage 1 produces `daily_meds.csv` from `export.xml` only. AutoExport meds and SoM are NOT handled in Stage 1.

---

### Stage 2: Unify (`stage_2_unify()`)

**Location:** `scripts/run_full_pipeline.py` lines 355–378  
**Delegated to:** `src/etl/stage_unify_daily.py::run_unify_daily()`

| Domain   | Input Files                                           | Merging Strategy                                   | Vendor Column               |
| -------- | ----------------------------------------------------- | -------------------------------------------------- | --------------------------- |
| Sleep    | `apple/daily_sleep.csv`, `zepp/daily_sleep.csv`       | Apple-preferred fill                               | ❌ None                     |
| Cardio   | `apple/daily_cardio.csv`, `zepp/daily_cardio.csv`     | Average both                                       | ❌ None                     |
| Activity | `apple/daily_activity.csv`, `zepp/daily_activity.csv` | Sum both                                           | ❌ None                     |
| Meds     | `apple/daily_meds.csv`                                | Pass-through (expects `med_vendor` column)         | ✅ `med_vendor`             |
| SoM      | `apple/daily_som.csv`                                 | Pass-through, adds `som_vendor='apple_autoexport'` | ✅ `som_vendor` (hardcoded) |

**Key Finding:**

- `unify_meds()` loads `daily_meds.csv` and expects `med_vendor` column to already exist
- `unify_som()` loads `daily_som.csv` and hardcodes `som_vendor='apple_autoexport'`
- **Problem:** Neither `daily_meds.csv` (with AutoExport integration) nor `daily_som.csv` are created in Stage 1

---

### Domain Module Integration (Currently NOT Wired)

| Module                                    | Function                 | Purpose                                   | Current Call Point        |
| ----------------------------------------- | ------------------------ | ----------------------------------------- | ------------------------- |
| `src/domains/meds/meds_from_extracted.py` | `run_meds_aggregation()` | Multi-vendor meds with source prioritizer | **Standalone only** (CLI) |
| `src/domains/som/som_from_autoexport.py`  | `run_som_aggregation()`  | AutoExport SoM aggregation                | **Standalone only** (CLI) |

**Key Finding:** Both domain modules use `source_prioritizer` and produce daily CSVs with vendor columns, but they are NOT called from the pipeline stages.

---

## 2. Identified Inconsistencies

### Issue 1: Non-Deterministic ZIP Selection for Apple Export and Zepp

- **Current:** Stage 0 extracts ALL ZIPs from `apple/export/` and `zepp/`
- **Impact:** Different pipeline runs on same snapshot may process different files if ZIPs are added/removed
- **AutoExport:** Already uses `_select_autoexport_zip_for_snapshot()` ✅

### Issue 2: Meds Aggregation Split Between Two Stages

- **Stage 1:** Creates `daily_meds.csv` from `export.xml` only (via `MedsAggregator`)
- **Domain Module:** `run_meds_aggregation()` handles both `apple_export` AND `apple_autoexport` with prioritization
- **Impact:** Stage 1's `daily_meds.csv` does NOT contain AutoExport meds data

### Issue 3: SoM Not Aggregated in Stage 1

- **Stage 1:** No SoM aggregation at all
- **Domain Module:** `run_som_aggregation()` creates `daily_som.csv` from AutoExport
- **Stage 2:** `unify_som()` expects `daily_som.csv` to exist but it was never created

### Issue 4: Vendor Columns Inconsistent

| Domain   | Vendor Column | How Set                             |
| -------- | ------------- | ----------------------------------- |
| Sleep    | ❌ None       | N/A                                 |
| Cardio   | ❌ None       | N/A                                 |
| Activity | ❌ None       | N/A                                 |
| Meds     | `med_vendor`  | Set by source prioritizer (if used) |
| SoM      | `som_vendor`  | Hardcoded in `unify_som()`          |

---

## 3. Proposed Refactoring Plan

### Goal

Create a consistent pattern where:

1. **Stage 0** deterministically selects ZIPs for ALL vendors based on snapshot date
2. **Stage 1** aggregates ALL domains (including meds and SoM) from ALL sources
3. **Stage 2** consumes Stage 1 outputs, applies source prioritization, and preserves vendor metadata

### Stage 0 Changes

**Objective:** Ensure ALL vendors follow deterministic ingest rule: **ZIP date <= SNAPSHOT**

| Task | Description                                                         | Files to Modify                |
| ---- | ------------------------------------------------------------------- | ------------------------------ |
| 0.1  | Add `_select_apple_export_zip_for_snapshot()` for Apple export ZIPs | `scripts/run_full_pipeline.py` |
| 0.2  | Add `_select_zepp_zip_for_snapshot()` for Zepp ZIPs                 | `scripts/run_full_pipeline.py` |
| 0.3  | Apply deterministic selection to all vendors                        | `scripts/run_full_pipeline.py` |

**Implementation Notes:**

- For Apple Export: Use ZIP filename timestamp (if available) or file modification date
- For Zepp: Use ZIP filename date or file modification date
- Selection rule: `zip_date <= snapshot`

---

### Stage 1 Changes

**Objective:** Move meds and SoM aggregation to Stage 1 to match the pattern used for other domains

| Task | Description                                                                                           | Files to Modify                    |
| ---- | ----------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 1.1  | Add `aggregate_meds_multi_vendor()` to produce `daily_meds_apple.csv` and `daily_meds_autoexport.csv` | `src/etl/stage_csv_aggregation.py` |
| 1.2  | Add `aggregate_som()` to produce `daily_som_autoexport.csv`                                           | `src/etl/stage_csv_aggregation.py` |
| 1.3  | Call meds/SoM aggregation in `run_csv_aggregation()`                                                  | `src/etl/stage_csv_aggregation.py` |

**Output Files After Stage 1:**

```
extracted/
├── apple/
│   ├── daily_sleep.csv
│   ├── daily_cardio.csv
│   ├── daily_activity.csv
│   ├── daily_meds_apple.csv      # NEW: from export.xml
│   └── daily_meds_autoexport.csv # NEW: from Medications-*.csv
├── apple_autoexport/              # OR: Keep inside apple/
│   └── daily_som_autoexport.csv  # NEW: from StateOfMind-*.csv
└── zepp/
    ├── daily_sleep.csv
    ├── daily_cardio.csv
    └── daily_activity.csv
```

**Alternative (Simpler):** Keep existing structure but add:

- `apple/daily_meds.csv` (from export.xml, as now)
- `apple/daily_meds_autoexport.csv` (NEW: from AutoExport)
- `apple/daily_som.csv` (NEW: from AutoExport)

---

### Stage 2 Changes

**Objective:** Consume daily\_\* CSVs from Stage 1, use source_prioritizer, keep vendor metadata columns

| Task | Description                                                                                              | Files to Modify                |
| ---- | -------------------------------------------------------------------------------------------------------- | ------------------------------ |
| 2.1  | Update `unify_meds()` to use source_prioritizer on both `daily_meds.csv` and `daily_meds_autoexport.csv` | `src/etl/stage_unify_daily.py` |
| 2.2  | Update `unify_som()` to load `daily_som.csv` (already exists)                                            | `src/etl/stage_unify_daily.py` |
| 2.3  | Add vendor columns for sleep/cardio/activity (optional)                                                  | `src/etl/stage_unify_daily.py` |

**Source Prioritizer Integration:**

```python
# In unify_meds():
vendor_dataframes = {
    "apple_export": df_meds_apple,
    "apple_autoexport": df_meds_autoexport,
}
df_meds, selected_vendor = prioritize_source(vendor_dataframes, ...)
df_meds["med_vendor"] = selected_vendor
```

---

## 4. Constraints Preserved

| Constraint                                                | Status                      |
| --------------------------------------------------------- | --------------------------- |
| Stage 3 (PBSI labels) MUST NOT be removed                 | ✅ No changes to Stage 3    |
| Stages 3–9 SHOULD NOT be modified                         | ✅ No changes to Stages 3–9 |
| ML target change (PBSI → SoM) handled in Stages 5–8 later | ✅ Deferred                 |
| Existing cardio/sleep/activity flow unchanged             | ✅ Only additive changes    |

---

## 5. Implementation Order

1. **Stage 1 first:** Add meds_autoexport and som aggregation
2. **Stage 2 second:** Update unify methods to consume new files
3. **Stage 0 last:** Add deterministic ZIP selection (lower priority, can be done later)

**Rationale:** Stage 1 and 2 changes unblock SoM as ML target. Stage 0 deterministic selection is a correctness improvement but not blocking.

---

## 6. Open Questions

1. **File naming:** Should AutoExport outputs be:

   - `daily_meds_autoexport.csv` (explicit vendor suffix)
   - OR `autoexport/daily_meds.csv` (subdirectory)

2. **Vendor columns for cardio/sleep/activity:** Should we add vendor tracking for these domains too?

   - Currently: No vendor columns
   - With source prioritizer: Could add `cardio_vendor`, `sleep_vendor`, `activity_vendor`

3. **ZIP date parsing:** For Apple export ZIPs without date in filename, use:
   - File modification date?
   - Content date (latest record in export.xml)?
   - Error/skip?

---

## 7. Next Steps

- [ ] User approval of this plan
- [ ] Implement Stage 1 changes (Task 1.1–1.3)
- [ ] Implement Stage 2 changes (Task 2.1–2.2)
- [ ] Test with P000001/2025-12-08
- [ ] Implement Stage 0 changes (Task 0.1–0.3) - optional/deferred

---

_Document auto-generated by analysis of codebase._
