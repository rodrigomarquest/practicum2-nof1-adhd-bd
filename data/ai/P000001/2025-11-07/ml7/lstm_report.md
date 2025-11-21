# LSTM M1 Training Report

## Dataset Summary

- **Total Days**: 1625 (after temporal filter and dropna)
- **Total Sequences**: 1612 (14-day windows)
- **Temporal Filter**: >= 2021-05-11 (Amazfit-only era)
- **Date Range**: 2021-05-11 to 2025-10-21
- **Rationale**: ML7 uses the same inter-device consistency filter as ML6
  (cardio data does not exist before 2021-05-11)

## Architecture

- Sequence Length: 14 days
- Input Features: 7 z-scored canonical features
  - z_sleep_total_h, z_sleep_efficiency, z_hr_mean, z_hrv_rmssd, z_hr_max, z_steps, z_exercise_min
- LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax
- Classes: 3

## Feature Set

**Z-scored Canonical Features** (computed from ML6 MICE-imputed data):

1. `z_sleep_total_h`
2. `z_sleep_efficiency`
3. `z_hr_mean`
4. `z_hrv_rmssd`
5. `z_hr_max`
6. `z_steps`
7. `z_exercise_min`

**Note**: Features are z-scored on the entire ML7 dataset to enable LSTM learning.

## Cross-Validation Results

### Fold 0

- **Macro-F1**: 0.4263
- **Val Loss**: 0.8238
- **Val Accuracy**: 0.6042

### Fold 1

- **Macro-F1**: 0.2929
- **Val Loss**: 0.6547
- **Val Accuracy**: 0.6250

### Fold 2

- **Macro-F1**: 0.4025
- **Val Loss**: 0.9989
- **Val Accuracy**: 0.5000

### Fold 3

- **Macro-F1**: 0.4182
- **Val Loss**: 0.8811
- **Val Accuracy**: 0.4792

### Fold 4

- **Macro-F1**: 0.4567
- **Val Loss**: 0.8707
- **Val Accuracy**: 0.6458

### Fold 5

- **Macro-F1**: 0.4607
- **Val Loss**: 0.3875
- **Val Accuracy**: 0.8542


**Mean Macro-F1**: 0.4095
