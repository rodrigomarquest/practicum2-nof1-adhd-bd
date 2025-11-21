# SHAP Feature Importance Summary

**Model Explained**: Logistic Regression (multinomial, class_weight='balanced')

**Feature Set**: Z-scored canonical features from PBSI pipeline
- 7 features: z_sleep_total_h, z_sleep_efficiency, z_hr_mean, z_hrv_rmssd, z_hr_max, z_steps, z_exercise_min
- Segment-wise normalized (119 temporal segments) to prevent leakage

**Note**: SHAP explains the LogisticRegression baseline, NOT the LSTM model.

---

## Global Top-10 Features

1. **z_sleep_efficiency**: 0.9016
2. **z_sleep_total_h**: 0.8577
3. **z_hrv_rmssd**: 0.5246
4. **z_steps**: 0.3860
5. **z_hr_mean**: 0.3768
6. **z_exercise_min**: 0.3097
7. **z_hr_max**: 0.2753

## Per-Fold Top-5


### Fold 0

1. z_sleep_total_h: 0.9564
2. z_sleep_efficiency: 0.7913
3. z_hr_mean: 0.3111
4. z_hrv_rmssd: 0.3000
5. z_hr_max: 0.1530

### Fold 1

1. z_sleep_total_h: 0.9557
2. z_sleep_efficiency: 0.8826
3. z_hrv_rmssd: 0.5519
4. z_steps: 0.1447
5. z_hr_max: 0.1204

### Fold 2

1. z_sleep_efficiency: 1.1334
2. z_sleep_total_h: 0.9894
3. z_hr_mean: 0.4185
4. z_hr_max: 0.1565
5. z_steps: 0.1414

### Fold 3

1. z_sleep_efficiency: 0.8429
2. z_steps: 0.8084
3. z_sleep_total_h: 0.7594
4. z_hrv_rmssd: 0.3928
5. z_hr_mean: 0.3492

### Fold 4

1. z_hrv_rmssd: 0.8359
2. z_sleep_total_h: 0.7588
3. z_hr_mean: 0.6782
4. z_steps: 0.6682
5. z_sleep_efficiency: 0.6659

### Fold 5

1. z_sleep_efficiency: 1.0938
2. z_exercise_min: 1.0358
3. z_hrv_rmssd: 0.9494
4. z_hr_max: 0.7776
5. z_sleep_total_h: 0.7266
