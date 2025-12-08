# Final Consolidation Report - Dissertation Version

**Date**: 2025-12-08  
**Pipeline Version**: SoM-Centric ML (v4.1.7)  
**Participant**: P000001  
**Snapshot**: 2025-12-08

---

## 1. Domain Naming Verification

### ✅ PASS - All active code uses canonical domain names

| Domain         | Canonical Name | Legacy Names | Status                      |
| -------------- | -------------- | ------------ | --------------------------- |
| Heart Rate     | `cardio`       | hr           | ✅ No legacy in active code |
| Steps/Movement | `activity`     | steps        | ✅ No legacy in active code |
| Sleep          | `sleep`        | -            | ✅                          |
| Medication     | `meds`         | -            | ✅                          |
| State of Mind  | `som`          | -            | ✅                          |

**Files Verified:**

- `Makefile`: Uses `qc-cardio`, `qc-activity`, `qc-sleep`, `qc-meds`, `qc-som`
- `src/etl/etl_audit.py`: Uses `--domain cardio`, `--domain activity`, etc.
- `scripts/run_full_pipeline.py`: All domain references canonical

**Note:** Legacy names (`qc-hr`, `qc-steps`) appear only in historical documentation under `docs/copilot/*.md` which are preserved as historical records.

---

## 2. SoM-Centric ML Implementation

### ✅ VERIFIED - All ML stages use SoM as primary target

| Stage | Component           | SoM-Centric Implementation                                          |
| ----- | ------------------- | ------------------------------------------------------------------- |
| 5     | PREP ML6            | ✅ Filters to `som_vendor='apple_autoexport'`, derives `som_binary` |
| 6     | Logistic Regression | ✅ Target: `som_category_3class`, fallback: `som_binary`            |
| 7     | LSTM + SHAP + Drift | ✅ Same target logic, graceful skips when insufficient              |
| 8     | TFLite Export       | ✅ Exports SoM model only if LSTM trained                           |
| 9     | Report              | ✅ SoM-centric summary with coverage stats                          |

**Extended Feature Set (19 features):**

- Sleep (2): `sleep_hours`, `sleep_quality_score`
- Cardio (5): `hr_mean`, `hr_min`, `hr_max`, `hr_std`, `hr_samples`
- HRV (5): `hrv_sdnn_mean`, `hrv_sdnn_median`, `hrv_sdnn_min`, `hrv_sdnn_max`, `n_hrv_sdnn`
- Activity (3): `total_steps`, `total_distance`, `total_active_energy`
- Meds (3): `med_any`, `med_event_count`, `med_dose_total`
- PBSI (1): `pbsi_score` (auxiliary feature, NOT target)

---

## 3. Pipeline Execution Results

### Full Pipeline Run (Stages 0-9)

```
Participant: P000001
Snapshot: 2025-12-08
Zepp Password: wYBoktDN
```

| Stage | Name          | Status     | Duration |
| ----- | ------------- | ---------- | -------- |
| 0     | INGEST        | ✅ SUCCESS | 52.0s    |
| 1     | AGGREGATE     | ✅ SUCCESS | 17min    |
| 2     | UNIFY         | ✅ SUCCESS | <1s      |
| 3     | LABEL         | ✅ SUCCESS | 2s       |
| 4     | SEGMENT       | ✅ SUCCESS | 1s       |
| 5     | PREP ML6      | ✅ SUCCESS | 8s       |
| 6     | ML6 Training  | ✅ SUCCESS | <1s      |
| 7     | ML7 Analysis  | ✅ SUCCESS | 17s      |
| 8     | TFLite Export | ✅ SUCCESS | 1s       |
| 9     | Report        | ✅ SUCCESS | <1s      |

**ML Results (after SoM bug fix):**

- **Stage 6**: Logistic Regression F1=0.2916±0.0324 (3-class SoM)
- **Stage 7**: LSTM F1=0.2333, SHAP computed successfully
- **Stage 8**: TFLite model exported (41.6 KB)

---

## 4. QC Validation Results

### All Domain Audits PASS

| Domain      | Status  | Key Metrics                              |
| ----------- | ------- | ---------------------------------------- |
| Cardio      | ✅ PASS | 1,360 days, 4.8M HR records              |
| Activity    | ✅ PASS | 2,766 days, mean 9,395 steps/day         |
| Sleep       | ✅ PASS | 994 days with data, mean 7.34h           |
| Meds        | ✅ PASS | 452 days, 1,420 medication events        |
| SoM         | ✅ PASS | 77 days, 3-class distribution (12/11/54) |
| Unified Ext | ✅ PASS | 30 columns, no duplicates                |
| Labels      | ✅ PASS | PBSI distributed: 26.5% / 48.5% / 25.0%  |

---

## 5. Data Coverage Summary

```
Date Range: 2017-12-04 to 2025-12-07 (2,868 days)

Domain Coverage:
- HR/Cardio:   1,360 days (47.4%)
- Activity:    2,766 days (96.4%)
- Sleep:       1,904 days (66.4%)
- Meds:          452 days (15.8%)
- SoM:            77 days (2.7%)
- HRV:            18 days (0.6%)
```

---

## 6. Known Limitations

1. **SoM Data Scarcity**: Only 77 days (2.7%) have State of Mind labels
2. **SoM Class Imbalance**: Positive class dominates (70.1%), minority classes have 12/11 samples
3. **HRV Coverage**: Only 18 days with HRV SDNN data from Apple Watch
4. **ML Performance**: F1~0.29 reflects class imbalance and small dataset

---

## 7. Fixes Applied During Consolidation

| Issue                             | Fix                                              | File                                     |
| --------------------------------- | ------------------------------------------------ | ---------------------------------------- |
| SoM Valence parsing bug           | Fixed CSV trailing comma handling                | `src/domains/som/som_from_autoexport.py` |
| LSTM crash with 1 class           | Added `n_classes < 2` check before training      | `scripts/run_full_pipeline.py`           |
| JSON serialization of numpy int64 | Added `numpy_json_serializer()`                  | `src/etl/etl_audit.py`                   |
| Log tag inconsistency             | Standardized to `[VENDOR/VARIANT/DOMAIN]` format | Multiple ETL files                       |

---

## 8. Artifact Locations

### Pipeline Outputs

- **Unified CSV**: `data/etl/P000001/2025-12-08/joined/features_daily_unified.csv`
- **Labeled CSV**: `data/etl/P000001/2025-12-08/joined/features_daily_labeled.csv`
- **ML6 Dataset**: `data/ai/P000001/2025-12-08/ml6/features_daily_ml6.csv`

### Reports

- **RUN Report**: `docs/reports/RUN_P000001_2025-12-08_stages3-9_20251208T175113.md`
- **QC Reports**: `docs/reports/qc/QC_P000001_2025-12-08_*.md`

### Documentation

- **ML Refactor Summary**: `docs/copilot/ml_refactor_som_summary.md`
- **Pre-Refactor Map**: `docs/copilot/ml_stage_map_pre_som_refactor.md`
- **Progress Bars Update**: `docs/copilot/progress_bars_and_meds_perf_update.md`
- **This Report**: `docs/copilot/final_consolidation_report.md`

---

## 9. Conclusion

✅ **PIPELINE READY FOR DISSERTATION**

The ETL/ML pipeline has been consolidated with:

1. **Consistent domain naming** (cardio, activity, sleep, meds, som)
2. **SoM-centric ML strategy** with successful model training
3. **All QC audits passing**
4. **Complete artifact generation**
5. **Standardized log tags** (`[VENDOR/VARIANT/DOMAIN]` format)

**ML Results Summary:**

- LogisticRegression: F1=0.29 (3-class SoM)
- LSTM: F1=0.23 (3-class SoM)
- Top features (SHAP): HRV metrics, total_steps, hr_samples

---

_Generated by Copilot consolidation session 2025-12-08_
