# Experiment Suite: Stage 5 Integration

**Date**: 2025-12-08  
**Author**: Copilot (ML Refactor Session)  
**Status**: Design Specification

---

## 1. Overview

Stage 5 is being refactored from a monolithic "Prep ML6" function into a composite stage with explicit sub-stages:

| Sub-Stage | Name              | Purpose                                       |
| --------- | ----------------- | --------------------------------------------- |
| **5.1**   | Temporal Filter   | Filter to ML-ready period (>= 2021-05-11)     |
| **5.2**   | Anti-Leak Removal | Drop columns that would leak target info      |
| **5.3**   | MICE Imputation   | Fill missing values segment-aware             |
| **5.4**   | Feature Universe  | Save complete imputed feature matrix          |
| **5.5**   | Experiment Suite  | Run FS ablation, select best config per model |

---

## 2. Feature Universe

### Definition

The **Feature Universe** is a single imputed table containing:

- All candidate features (after anti-leak filtering)
- All SoM targets (for experiments)
- Metadata for CV (dates, segments)

### Structure

```
features_daily_ml_universe.csv
├── date (datetime)
├── segment_id (int)
├── [Feature Columns]
│   ├── Sleep: sleep_hours, sleep_quality_score
│   ├── Cardio: hr_mean, hr_min, hr_max, hr_std, hr_samples
│   ├── HRV: hrv_sdnn_mean, hrv_sdnn_median, hrv_sdnn_min, hrv_sdnn_max, n_hrv_sdnn
│   ├── Activity: total_steps, total_distance, total_active_energy
│   ├── Meds: med_any, med_event_count, med_dose_total
│   └── PBSI: pbsi_score (as candidate feature, NOT target)
├── [Target Columns]
│   ├── som_category_3class
│   └── som_binary
└── [Metadata]
    └── segment_id
```

### Feature Set Definitions

| FS ID    | Name     | Features                  | Count |
| -------- | -------- | ------------------------- | ----- |
| **FS-A** | Baseline | Sleep + Cardio + Activity | 10    |
| **FS-B** | + HRV    | FS-A + HRV features       | 15    |
| **FS-C** | + Meds   | FS-B + med\_\*            | 18    |
| **FS-D** | + PBSI   | FS-C + pbsi_score         | 19    |

### Anti-Leak Columns (MUST NOT be features)

```python
ANTI_LEAK_COLUMNS = [
    # PBSI intermediate outputs (leak target derivation)
    'pbsi_quality',
    'sleep_sub', 'cardio_sub', 'activity_sub',
    'label_3cls', 'label_2cls', 'label_clinical',
    # Direct target columns (obvious leak)
    'som_category_3class', 'som_binary',  # Targets - not features
]
```

**Note**: `pbsi_score` is NOT in anti-leak because it's a candidate feature (FS-D includes it). The science question is whether PBSI adds predictive value beyond raw features.

---

## 3. Experiment Suite Contract

### Input

The Experiment Suite receives:

- `df_universe`: DataFrame with all candidate features + targets (post-MICE)
- `participant`: e.g., "P000001"
- `snapshot`: e.g., "2025-12-08"

### Processing

For each model (ML6, ML7):

1. Run ablation across all feature sets (FS-A/B/C/D)
2. For ML6: Test both targets (3-class, binary)
3. For ML7: Test configurations (CFG-1/2/3) × targets
4. Use consistent CV scheme (6-fold temporal)
5. Compute metrics: F1-macro, balanced accuracy, Cohen's kappa

### Output

```json
// ai/local/<PID>/<SNAPSHOT>/model_selection.json
{
  "snapshot": "2025-12-08",
  "participant": "P000001",
  "generated_at": "2025-12-08T15:30:00",
  "experiment_suite_version": "1.0.0",

  "ml6": {
    "selected_fs": "FS-B",
    "selected_target": "som_binary",
    "features": [
      "sleep_hours", "sleep_quality_score",
      "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples",
      "hrv_sdnn_mean", "hrv_sdnn_median", "hrv_sdnn_min", "hrv_sdnn_max", "n_hrv_sdnn",
      "total_steps", "total_distance", "total_active_energy"
    ],
    "n_features": 15,
    "metrics": {
      "f1_macro": 0.4623,
      "balanced_accuracy": 0.51,
      "cohen_kappa": 0.10
    },
    "selection_reason": "Highest F1-macro among FS candidates",
    "all_results": {
      "FS-A_binary": {"f1_macro": 0.29, ...},
      "FS-B_binary": {"f1_macro": 0.46, ...},
      "FS-C_binary": {"f1_macro": 0.45, ...},
      "FS-D_binary": {"f1_macro": 0.46, ...}
    }
  },

  "ml7": {
    "selected_config": "CFG-3",
    "selected_fs": "FS-B",
    "selected_target": "som_binary",
    "features": [...],
    "n_features": 15,
    "config_params": {
      "seq_len": 14,
      "lstm_units": 32,
      "dense_units": 32,
      "dropout": 0.4,
      "early_stopping": true,
      "class_weights": true
    },
    "metrics": {
      "f1_macro": 0.5667,
      "balanced_accuracy": 0.61,
      "f1_weighted": 0.73
    },
    "selection_reason": "CFG-3 × FS-B × binary: highest F1-macro",
    "all_results": {...}
  }
}
```

### Selection Rules (Deterministic)

```python
def select_best_config(results: List[dict]) -> dict:
    """
    Deterministic selection rule:
    1. Highest F1-macro
    2. If tie: highest Cohen's kappa
    3. If tie: fewer features (simpler model)
    """
    # Sort by (f1_macro DESC, kappa DESC, n_features ASC)
    sorted_results = sorted(
        results,
        key=lambda r: (-r['f1_macro'], -r.get('kappa', 0), r['n_features'])
    )
    return sorted_results[0]
```

---

## 4. Stage 5 Refactored Flow

```
stage_5_prep_ml6(ctx, df) {

    // 5.1 TEMPORAL FILTER
    df_filtered = df[df['date'] >= '2021-05-11']
    df_filtered = df_filtered[som_valid_mask]  // som_vendor == 'apple_autoexport'

    // 5.2 ANTI-LEAK REMOVAL
    df_filtered = df_filtered.drop(columns=ANTI_LEAK_COLUMNS, errors='ignore')
    // Keep pbsi_score as candidate feature

    // 5.3 MICE IMPUTATION
    df_imputed = apply_mice_segment_aware(df_filtered, FEATURE_UNIVERSE_COLS)

    // 5.4 BUILD & SAVE FEATURE UNIVERSE
    df_universe = df_imputed[[date, segment_id, ...features..., som_3class, som_binary]]
    df_universe.to_csv(ml_universe_path)

    // 5.5 EXPERIMENT SUITE
    model_selection = run_experiment_suite(df_universe, participant, snapshot)
    save_json(model_selection, model_selection_path)

    // 5.6 SAVE DEFAULT VIEW (backward compat)
    selected_features = model_selection['ml6']['features']
    df_ml6 = df_universe[['date'] + selected_features + targets + ['segment_id']]
    df_ml6.to_csv(ml6_path)

    return df_ml6
}
```

---

## 5. Integration with Stage 6/7

### Current Behavior (Phase 1 - This PR)

- Stage 6/7 continue using `features_daily_ml6.csv` as before
- `model_selection.json` is created but NOT yet consumed
- RUN_REPORT logs that Experiment Suite ran and shows selected config

### Future Behavior (Phase 2 - Later PR)

- Stage 6/7 read `model_selection.json` as source of truth
- Features and config params come from selection, not hardcoded

---

## 6. Graceful Fallback

If Experiment Suite fails (not enough data, TensorFlow errors, etc.):

```python
try:
    model_selection = run_experiment_suite(df_universe, ...)
except Exception as e:
    logger.warning(f"Experiment Suite failed: {e}")
    logger.warning("Falling back to FS-B defaults")
    model_selection = {
        "ml6": {"selected_fs": "FS-B", "features": FS_B_FEATURES, ...},
        "ml7": {"selected_config": "CFG-3", "features": FS_B_FEATURES, ...},
        "fallback": True,
        "fallback_reason": str(e)
    }
```

This ensures pipeline stability even if ablation experiments fail.

---

## 7. File Outputs

| File              | Path                                                           | Description                         |
| ----------------- | -------------------------------------------------------------- | ----------------------------------- |
| Feature Universe  | `ai/local/<PID>/<SNAPSHOT>/ml6/features_daily_ml_universe.csv` | All features + targets (imputed)    |
| Model Selection   | `ai/local/<PID>/<SNAPSHOT>/model_selection.json`               | Experiment Suite results            |
| ML6 Features      | `ai/local/<PID>/<SNAPSHOT>/ml6/features_daily_ml6.csv`         | Selected features (backward compat) |
| Experiment Report | `ai/local/<PID>/<SNAPSHOT>/ml6/experiments/ablation_report.md` | Detailed ablation results           |
