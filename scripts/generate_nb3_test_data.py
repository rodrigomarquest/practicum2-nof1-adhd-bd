#!/usr/bin/env python
"""
Generate a minimal test CSV for NB3 validation.
This creates a synthetic features_daily_labeled.csv with realistic structure.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set seed
np.random.seed(42)

# Date range: 200 days (enough for 6 calendar folds)
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=i) for i in range(200)]

# Feature names (27 canonical + 8 PBSI = 35 total)
canonical_features = [
    'sleep_total_h', 'sleep_efficiency',
    'apple_hr_mean', 'apple_hr_max', 'apple_hrv_rmssd',
    'steps', 'exercise_min', 'stand_hours', 'move_kcal',
    'missing_sleep', 'missing_cardio', 'missing_activity',
    'source_sleep', 'source_cardio', 'source_activity'
]

pbsi_features = [
    'sleep_sub', 'cardio_sub', 'activity_sub',
    'pbsi_score', 'label_3cls', 'label_2cls', 'label_clinical', 'pbsi_quality'
]

all_features = canonical_features + pbsi_features

# Generate synthetic data
np.random.seed(42)
data = {
    'date': dates,
    'segment_id': [i // 33 + 1 for i in range(len(dates))],  # 6 segments
}

# Canonical features (continuous)
for feat in canonical_features:
    if 'missing' in feat or 'source' in feat:
        # Binary or count features
        data[feat] = np.random.randint(0, 2, size=len(dates))
    else:
        # Continuous features
        data[feat] = np.random.normal(50, 15, size=len(dates)).clip(0, 200)

# PBSI features
for feat in ['sleep_sub', 'cardio_sub', 'activity_sub']:
    data[feat] = np.random.randint(-3, 4, size=len(dates))

data['pbsi_score'] = (
    data['sleep_sub'] + data['cardio_sub'] + data['activity_sub']
) / 3.0 + np.random.normal(0, 0.2, size=len(dates))

data['pbsi_quality'] = np.random.uniform(0.7, 1.0, size=len(dates))

# Labels (3-class: -1, 0, 1)
data['label_3cls'] = np.random.choice([-1, 0, 1], size=len(dates), p=[0.25, 0.5, 0.25])
data['label_2cls'] = (data['label_3cls'] >= 0).astype(int)
data['label_clinical'] = np.random.choice([0, 1, 2], size=len(dates))

# Create DataFrame
df = pd.DataFrame(data)

# Save
output_path = 'data/etl/features_daily_labeled_test.csv'
df.to_csv(output_path, index=False)

print(f"Created test CSV: {output_path}")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nFirst 5 rows:")
print(df.head())
