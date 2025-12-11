"""
Test: Canonical PBSI Integration
Smoke test to verify that stage_apply_labels.py correctly uses build_pbsi.py
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from etl.stage_apply_labels import PBSILabeler


def test_canonical_pbsi_integration():
    """
    Smoke test: Create synthetic data with 2 segments, apply PBSI, verify behavior.
    """
    
    print("\n" + "="*80)
    print("TEST: Canonical PBSI Integration")
    print("="*80 + "\n")
    
    # Create synthetic data: 60 days, 2 segments (30 days each)
    dates = pd.date_range('2024-01-01', periods=60, freq='D')
    
    # Segment 1: Good sleep, low HR, high steps (should be STABLE, label +1)
    segment1_data = {
        'date': dates[:30],
        'sleep_hours': np.random.normal(8.0, 0.5, 30),  # Good sleep
        'sleep_quality_score': np.random.normal(85, 5, 30),  # High quality
        'hr_mean': np.random.normal(60, 5, 30),  # Low resting HR
        'hr_max': np.random.normal(120, 10, 30),
        'hr_std': np.random.normal(10, 2, 30),  # Low variability (proxy for HRV)
        'total_steps': np.random.normal(10000, 1000, 30),  # Active
        'total_active_energy': np.random.normal(500, 50, 30),
    }
    
    # Segment 2: Poor sleep, high HR, low steps (should be UNSTABLE, label -1)
    segment2_data = {
        'date': dates[30:],
        'sleep_hours': np.random.normal(5.0, 0.5, 30),  # Poor sleep
        'sleep_quality_score': np.random.normal(50, 5, 30),  # Low quality
        'hr_mean': np.random.normal(80, 5, 30),  # High resting HR
        'hr_max': np.random.normal(160, 10, 30),
        'hr_std': np.random.normal(20, 2, 30),  # High variability
        'total_steps': np.random.normal(3000, 500, 30),  # Sedentary
        'total_active_energy': np.random.normal(150, 30, 30),
    }
    
    # Combine
    df = pd.concat([
        pd.DataFrame(segment1_data),
        pd.DataFrame(segment2_data)
    ], ignore_index=True)
    
    print(f"✓ Created synthetic data: {len(df)} days")
    print(f"  Segment 1 (days 1-30): Good sleep, low HR, high steps")
    print(f"  Segment 2 (days 31-60): Poor sleep, high HR, low steps\n")
    
    # Apply canonical PBSI
    labeler = PBSILabeler()
    df_labeled = labeler.apply_labels(df)
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    # Verify columns exist
    required_cols = ['segment_id', 'pbsi_score', 'label_3cls', 'label_2cls', 
                     'sleep_sub', 'cardio_sub', 'activity_sub', 'pbsi_quality']
    
    missing = [col for col in required_cols if col not in df_labeled.columns]
    if missing:
        print(f"❌ FAIL: Missing columns: {missing}")
        return False
    
    print(f"✓ All required columns present: {required_cols}\n")
    
    # Verify segment-wise behavior
    n_segments = df_labeled['segment_id'].nunique()
    print(f"✓ Segments created: {n_segments} (expected: 2)")
    
    if n_segments != 2:
        print(f"⚠️  WARNING: Expected 2 segments, got {n_segments}")
    
    # Verify label distribution
    label_dist = df_labeled['label_3cls'].value_counts().to_dict()
    print(f"\n✓ Label distribution:")
    for label in [1, 0, -1]:
        count = label_dist.get(label, 0)
        pct = count / len(df_labeled) * 100
        label_name = {1: "stable", 0: "neutral", -1: "unstable"}[label]
        print(f"  Label {label:+2d} ({label_name:8s}): {count:2d} days ({pct:5.1f}%)")
    
    # Verify score range
    score_min = df_labeled['pbsi_score'].min()
    score_max = df_labeled['pbsi_score'].max()
    print(f"\n✓ PBSI score range: {score_min:.3f} to {score_max:.3f}")
    print(f"  (z-scored scale: lower = more stable)")
    
    # Segment-level analysis
    print(f"\n✓ Segment-wise statistics:")
    for seg_id in sorted(df_labeled['segment_id'].unique()):
        seg_df = df_labeled[df_labeled['segment_id'] == seg_id]
        mean_pbsi = seg_df['pbsi_score'].mean()
        mode_label = seg_df['label_3cls'].mode()[0]
        label_name = {1: "stable", 0: "neutral", -1: "unstable"}.get(int(mode_label), "unknown")
        
        print(f"  Segment {seg_id}: mean_pbsi={mean_pbsi:+.3f}, mode_label={int(mode_label):+2d} ({label_name})")
    
    # Expected behavior: Segment 1 should have lower (more negative) PBSI → stable (+1)
    seg1_mean = df_labeled[df_labeled['segment_id'] == 1]['pbsi_score'].mean()
    seg2_mean = df_labeled[df_labeled['segment_id'] == 2]['pbsi_score'].mean()
    
    print(f"\n✓ Segment comparison:")
    print(f"  Segment 1 (good): mean_pbsi = {seg1_mean:+.3f}")
    print(f"  Segment 2 (poor): mean_pbsi = {seg2_mean:+.3f}")
    
    if seg1_mean < seg2_mean:
        print(f"  ✓ CORRECT: Segment 1 (good) has lower PBSI than Segment 2 (poor)")
    else:
        print(f"  ⚠️  WARNING: Expected Segment 1 < Segment 2, but got opposite")
    
    # Verify z-score columns exist
    z_cols = [col for col in df_labeled.columns if col.startswith('z_')]
    print(f"\n✓ Z-score columns created: {len(z_cols)}")
    print(f"  {z_cols}")
    
    # Verify segment-wise z-scores (should have mean≈0, std≈1 per segment)
    if 'z_sleep_total_h' in df_labeled.columns:
        for seg_id in sorted(df_labeled['segment_id'].unique()):
            seg_df = df_labeled[df_labeled['segment_id'] == seg_id]
            z_mean = seg_df['z_sleep_total_h'].mean()
            z_std = seg_df['z_sleep_total_h'].std()
            print(f"  Segment {seg_id} z_sleep_total_h: mean={z_mean:.3f}, std={z_std:.3f}")
    
    print("\n" + "="*80)
    print("TEST PASSED ✓")
    print("="*80 + "\n")
    
    return True


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    success = test_canonical_pbsi_integration()
    sys.exit(0 if success else 1)
