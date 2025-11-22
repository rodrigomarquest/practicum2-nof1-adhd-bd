"""
Generate PhD Dissertation Figures
==================================
Purpose: Generate publication-quality figures for LaTeX dissertation
Output: docs/latex/figures/
Snapshot: P000001/2025-11-07
Style: PhD-level academic (grayscale-friendly, high DPI)

Dataset: Uses full timeline (features_daily_labeled.csv) for visualization
ML6/ML7 Stats: Displays statistics for ML-filtered dataset (>= 2021-05-11)
Note: ML6 and ML7 use temporal filter >= 2021-05-11 (1,625 days, 1,612 sequences)

Generated Figures:
==================
Core Pipeline (1-9):
  - fig01_pbsi_timeline: PBSI behavioral labels over time
  - fig02_cardio_distributions: HR/HRV distributions by label
  - fig03_sleep_activity_boxplots: Sleep and activity patterns
  - fig04_segmentwise_normalization: Z-score normalization demo
  - fig05_missing_data_pattern: Cardiovascular data availability
  - fig06_label_distribution_timeline: Label counts over time
  - fig07_correlation_heatmap: Feature correlation matrix
  - fig08_adwin_drift: ADWIN drift detection events
  - fig09_shap_importance: SHAP feature importance

Extended Models (10-13):
  - fig10_extended_ml6_models: ML6 extended models comparison (RF, XGB, LGBM, SVM)
  - fig11_extended_ml7_models: ML7 extended sequence models (GRU, TCN, MLP)
  - fig12_instability_scores: Top-15 temporally unstable features
  - fig13_extended_models_combined: Side-by-side ML6 + ML7 comparison

Usage:
======
  python scripts/generate_dissertation_figures.py

All figures are saved as both PDF (vector) and PNG (raster) at 300 DPI.
Figures automatically overwrite existing files for reproducibility.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context('paper', font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.linewidth'] = 0.5

# Paths
PID = 'P000001'
SNAPSHOT = '2025-11-07'
DATA_DIR = Path(f'data/etl/{PID}/{SNAPSHOT}/joined')
AI_DIR = Path(f'data/ai/{PID}/{SNAPSHOT}')
ML6_EXT_DIR = AI_DIR / 'ml6_ext'
ML7_EXT_DIR = AI_DIR / 'ml7_ext'
FIG_DIR = Path('docs/latex/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Participant: {PID}")
print(f"Snapshot: {SNAPSHOT}")
print(f"Output directory: {FIG_DIR}")
print(f"Figure settings: 300 DPI, grayscale-friendly\n")

# Load data
df = pd.read_csv(DATA_DIR / 'features_daily_labeled.csv', parse_dates=['date'])
print(f"Total days (full dataset): {len(df)}")
print(f"Date range (full): {df['date'].min()} to {df['date'].max()}")

# Also load ML6 filtered dataset for accurate statistics
ML6_PATH = Path('data/ai/P000001/2025-11-07/ml6/features_daily_ml6.csv')
if ML6_PATH.exists():
    df_ml6 = pd.read_csv(ML6_PATH, parse_dates=['date'])
    print(f"\nML-filtered dataset (>= 2021-05-11):")
    print(f"  Total days: {len(df_ml6)}")
    print(f"  Date range: {df_ml6['date'].min()} to {df_ml6['date'].max()}")
    print(f"  Label distribution (post-MICE):")
    for label, count in df_ml6['label_3cls'].value_counts().sort_index().items():
        pct = (count / len(df_ml6)) * 100
        print(f"    {label:+.1f}: {count:4d} ({pct:4.1f}%)")
    print(f"  Missing values: {df_ml6.isna().sum().sum()}")
    print(f"  ML7 sequences (14-day windows): {len(df_ml6) - 14 + 1}")
else:
    print(f"\n[WARN] ML6 dataset not found at {ML6_PATH}")

print(f"\nFull dataset label distribution:")
print(df['label_3cls'].value_counts().sort_index())

# Color mapping (grayscale-friendly)
color_map = {-1: '#d62728', 0: '#7f7f7f', 1: '#2ca02c'}
label_names = {-1: 'Dysregulated', 0: 'Typical', 1: 'Regulated'}

# =============================================================================
# FIGURE 1: Timeline Overview with PBSI Labels
# =============================================================================
print("\n[1/7] Generating PBSI timeline...")
fig, ax = plt.subplots(figsize=(10, 4))

for label in [-1, 0, 1]:
    mask = df['label_3cls'] == label
    ax.scatter(df.loc[mask, 'date'], 
               df.loc[mask, 'pbsi_score'],
               c=color_map[label], 
               label=label_names[label],
               alpha=0.6, 
               s=10, 
               edgecolors='none')

ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('PBSI Score', fontsize=11)
ax.set_title('Behavioral Stability Index (PBSI) Timeline (2017-2025)', fontsize=12, fontweight='bold')
ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='black')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig01_pbsi_timeline.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig01_pbsi_timeline.png', bbox_inches='tight')
plt.close()
print("  [OK] fig01_pbsi_timeline.{pdf,png}")

# =============================================================================
# FIGURE 2: Cardiovascular Feature Distributions
# =============================================================================
print("[2/7] Generating cardiovascular distributions...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

features = ['hr_mean', 'hr_max', 'hrv_rmssd', 'hr_std']
titles = ['Mean Heart Rate (bpm)', 'Max Heart Rate (bpm)', 'HRV RMSSD (ms)', 'HR Std Dev (bpm)']

for idx, (feat, title) in enumerate(zip(features, titles)):
    ax = axes[idx // 2, idx % 2]
    valid_data = df[df[feat].notna() & df['label_3cls'].notna()]
    
    if len(valid_data) > 0:
        for label in [-1, 0, 1]:
            data = valid_data[valid_data['label_3cls'] == label][feat]
            ax.hist(data, bins=30, alpha=0.5, label=label_names[label], 
                   color=color_map[label], edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel(title, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

plt.suptitle('Cardiovascular Feature Distributions by Behavioral Label', 
             fontsize=12, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig02_cardio_distributions.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig02_cardio_distributions.png', bbox_inches='tight')
plt.close()
print("  [OK] fig02_cardio_distributions.{pdf,png}")

# =============================================================================
# FIGURE 3: Sleep and Activity Patterns
# =============================================================================
print("[3/7] Generating sleep/activity boxplots...")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

features_3 = ['sleep_total_h', 'sleep_efficiency', 'steps']
titles_3 = ['Sleep Duration (hours)', 'Sleep Efficiency', 'Daily Steps']

for idx, (feat, title) in enumerate(zip(features_3, titles_3)):
    ax = axes[idx]
    valid_data = df[df[feat].notna() & df['label_3cls'].notna()]
    
    if len(valid_data) > 0:
        data_by_label = [valid_data[valid_data['label_3cls'] == label][feat].values 
                        for label in [-1, 0, 1]]
        
        bp = ax.boxplot(data_by_label, 
                       labels=[label_names[l] for l in [-1, 0, 1]],
                       patch_artist=True,
                       widths=0.6,
                       boxprops=dict(facecolor='lightgray', edgecolor='black', linewidth=0.8),
                       medianprops=dict(color='black', linewidth=1.5),
                       whiskerprops=dict(color='black', linewidth=0.8),
                       capprops=dict(color='black', linewidth=0.8))
        
        ax.set_ylabel(title, fontsize=10)
        ax.set_xticklabels([label_names[l] for l in [-1, 0, 1]], rotation=15, ha='right')
        ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Sleep and Activity Patterns by Behavioral Label', 
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig03_sleep_activity_boxplots.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig03_sleep_activity_boxplots.png', bbox_inches='tight')
plt.close()
print("  [OK] fig03_sleep_activity_boxplots.{pdf,png}")

# =============================================================================
# FIGURE 4: Segment-wise Normalization
# =============================================================================
print("[4/7] Generating segment-wise normalization demo...")
feature_raw = 'hr_mean'
feature_zscore = 'z_hr_mean'

demo_data = df[df[feature_raw].notna() & df[feature_zscore].notna() & df['segment_id'].notna()].copy()
demo_data = demo_data.sort_values('date')

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

# Top: Raw (BLUE)
ax1 = axes[0]
for seg in demo_data['segment_id'].unique()[:10]:
    seg_data = demo_data[demo_data['segment_id'] == seg]
    ax1.plot(seg_data['date'], seg_data[feature_raw], 
             color='steelblue',  # Blue for "before"
             marker='o', markersize=2, linewidth=0.8, alpha=0.7)

ax1.set_ylabel('Raw HR Mean (bpm)', fontsize=10)
ax1.set_title('Before Segment-wise Normalization', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Bottom: Z-scored (ORANGE)
ax2 = axes[1]
for seg in demo_data['segment_id'].unique()[:10]:
    seg_data = demo_data[demo_data['segment_id'] == seg]
    ax2.plot(seg_data['date'], seg_data[feature_zscore], 
             color='darkorange',  # Orange for "after"
             marker='o', markersize=2, linewidth=0.8, alpha=0.7)

ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.2, alpha=0.8)  # Red reference line
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Z-score (segment-wise)', fontsize=10)
ax2.set_title('After Segment-wise Normalization', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig04_segmentwise_normalization.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig04_segmentwise_normalization.png', bbox_inches='tight')
plt.close()
print("  [OK] fig04_segmentwise_normalization.{pdf,png}")

# =============================================================================
# FIGURE 5: Missing Data Pattern
# =============================================================================
print("[5/7] Generating missing data pattern...")
df_missing = df.copy()
df_missing['has_cardio'] = df_missing['hr_mean'].notna().astype(int)
df_missing['year'] = df_missing['date'].dt.year

coverage_by_year = df_missing.groupby('year')['has_cardio'].agg(['sum', 'count'])
coverage_by_year['coverage_pct'] = (coverage_by_year['sum'] / coverage_by_year['count']) * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: Coverage by year
ax1 = axes[0]
ax1.bar(coverage_by_year.index, coverage_by_year['coverage_pct'], 
       color='steelblue', edgecolor='black', linewidth=0.8)
ax1.axhline(y=50, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='50% threshold')
ax1.set_xlabel('Year', fontsize=10)
ax1.set_ylabel('Cardiovascular Coverage (%)', fontsize=10)
ax1.set_title('HR/HRV Data Availability by Year', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Right: Timeline
ax2 = axes[1]
ax2.fill_between(df_missing['date'], 0, df_missing['has_cardio'], 
                 color='steelblue', alpha=0.5, label='Data available')
ax2.set_xlabel('Date', fontsize=10)
ax2.set_ylabel('Data Availability', fontsize=10)
ax2.set_title('Cardiovascular Data Timeline', fontsize=11, fontweight='bold')
ax2.set_ylim(-0.1, 1.1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIG_DIR / 'fig05_missing_data_pattern.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig05_missing_data_pattern.png', bbox_inches='tight')
plt.close()
print("  [OK] fig05_missing_data_pattern.{pdf,png}")

# =============================================================================
# FIGURE 6: Label Distribution Over Time
# =============================================================================
print("[6/7] Generating label distribution timeline...")
df_monthly = df.copy()
df_monthly['year_month'] = df_monthly['date'].dt.to_period('M')

label_counts = df_monthly.groupby(['year_month', 'label_3cls']).size().unstack(fill_value=0)
label_counts.index = label_counts.index.to_timestamp()

fig, ax = plt.subplots(figsize=(12, 5))

ax.fill_between(label_counts.index, 0, label_counts[-1], 
               color=color_map[-1], alpha=0.7, label=label_names[-1])
ax.fill_between(label_counts.index, label_counts[-1], 
               label_counts[-1] + label_counts[0], 
               color=color_map[0], alpha=0.7, label=label_names[0])
ax.fill_between(label_counts.index, label_counts[-1] + label_counts[0], 
               label_counts[-1] + label_counts[0] + label_counts[1], 
               color=color_map[1], alpha=0.7, label=label_names[1])

ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Days per Month', fontsize=11)
ax.set_title('Behavioral Label Distribution Over Time (Stacked)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', frameon=True, edgecolor='black')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig06_label_distribution_timeline.pdf', bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig06_label_distribution_timeline.png', bbox_inches='tight')
plt.close()
print("  [OK] fig06_label_distribution_timeline.{pdf,png}")

# =============================================================================
# FIGURE 7: Correlation Heatmap
# =============================================================================
print("[7/7] Generating correlation heatmap...")
corr_features = [
    'sleep_total_h', 'sleep_efficiency',
    'hr_mean', 'hrv_rmssd', 'hr_max',
    'steps', 'exercise_min',
    'pbsi_score'
]

available_features = [f for f in corr_features if f in df.columns]
corr_data = df[available_features].dropna()

if len(corr_data) > 0:
    corr_matrix = corr_data.corr()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
               cmap='RdBu_r', center=0, vmin=-1, vmax=1,
               square=True, linewidths=0.5, 
               cbar_kws={'label': 'Pearson Correlation'},
               ax=ax)
    
    ax.set_title('Feature Correlation Matrix (Complete Cases)', 
                fontsize=12, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig07_correlation_heatmap.pdf', bbox_inches='tight')
    plt.savefig(FIG_DIR / 'fig07_correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    print("  [OK] fig07_correlation_heatmap.{pdf,png}")

# =============================================================================
# FIGURE 8: ADWIN Drift Detection
# =============================================================================
print("[8/9] Generating ADWIN drift detection...")
AI_DIR = Path('data/ai/P000001/2025-11-07/ml7')

try:
    # Load drift data
    adwin_df = pd.read_csv(AI_DIR / 'drift' / 'adwin_changes.csv', parse_dates=['date'])
    
    if len(adwin_df) > 0:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot PBSI timeline as background
        ax.scatter(df['date'], df['pbsi_score'], c='lightgray', s=5, alpha=0.3, label='PBSI score')
        
        # Overlay drift points using date column
        for idx, row in adwin_df.iterrows():
            change_date = row['date']
            ax.axvline(x=change_date, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # Add one label for drift lines
        ax.axvline(x=adwin_df.iloc[0]['date'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
                  label=f'ADWIN drift ({len(adwin_df)} events)')
        
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('PBSI Score', fontsize=11)
        ax.set_title('ADWIN Drift Detection on PBSI Timeline', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig08_adwin_drift.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'fig08_adwin_drift.png', bbox_inches='tight')
        plt.close()
        print("  [OK] fig08_adwin_drift.{pdf,png}")
    else:
        print("  [WARN] No ADWIN drift events found")
except FileNotFoundError:
    print("  [WARN] ADWIN data not found, skipping fig08")

# =============================================================================
# FIGURE 9: SHAP Feature Importance (Combined Folds)
# =============================================================================
print("[9/9] Generating SHAP feature importance...")

try:
    # Load SHAP summary
    shap_summary_path = AI_DIR / 'shap_summary.md'
    
    if shap_summary_path.exists():
        # Parse SHAP summary to extract feature importance
        # For now, create a composite figure from existing fold PNGs
        from PIL import Image
        
        fold_images = []
        for fold in range(6):
            img_path = AI_DIR / 'shap' / f'fold_{fold}_top5.png'
            if img_path.exists():
                fold_images.append(Image.open(img_path))
        
        if fold_images:
            # Create 2x3 grid
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            for idx, (ax, img) in enumerate(zip(axes.flatten(), fold_images)):
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(f'Fold {idx}', fontsize=11, fontweight='bold')
            
            plt.suptitle('SHAP Feature Importance by Cross-Validation Fold', 
                        fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()
            plt.savefig(FIG_DIR / 'fig09_shap_importance.pdf', bbox_inches='tight', dpi=150)
            plt.savefig(FIG_DIR / 'fig09_shap_importance.png', bbox_inches='tight', dpi=150)
            plt.close()
            print("  [OK] fig09_shap_importance.{pdf,png}")
        else:
            print("  [WARN] No SHAP fold images found")
    else:
        print("  [WARN] SHAP summary not found, skipping fig09")
except Exception as e:
    print(f"  [WARN] Error generating SHAP figure: {e}")

# =============================================================================
# FIGURE 10: Extended ML6 Models Comparison
# =============================================================================
print("[10/13] Generating extended ML6 models comparison...")

try:
    ml6_ext_summary = pd.read_csv(ML6_EXT_DIR / 'ml6_extended_summary.csv')
    
    if len(ml6_ext_summary) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by f1_macro_mean descending
        ml6_ext_summary = ml6_ext_summary.sort_values('f1_macro_mean', ascending=False)
        
        # Create barplot with error bars
        x_pos = np.arange(len(ml6_ext_summary))
        bars = ax.bar(x_pos, ml6_ext_summary['f1_macro_mean'], 
                     yerr=ml6_ext_summary['f1_macro_std'],
                     color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8,
                     capsize=5, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(ml6_ext_summary.iterrows()):
            ax.text(i, row['f1_macro_mean'] + row['f1_macro_std'] + 0.02,
                   f"{row['f1_macro_mean']:.3f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ml6_ext_summary['model'].str.upper(), fontsize=10)
        ax.set_ylabel('Macro-F1 Score', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_title('Extended ML6 Models Performance (6-fold Temporal CV)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.4, label='Baseline (0.5)')
        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig10_extended_ml6_models.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'fig10_extended_ml6_models.png', bbox_inches='tight')
        plt.close()
        print("  [OK] fig10_extended_ml6_models.{pdf,png}")
    else:
        print("  [WARN] No extended ML6 data found")
except FileNotFoundError:
    print("  [WARN] Extended ML6 summary not found, skipping fig10")
except Exception as e:
    print(f"  [WARN] Error generating extended ML6 figure: {e}")

# =============================================================================
# FIGURE 11: Extended ML7 Models Comparison
# =============================================================================
print("[11/13] Generating extended ML7 models comparison...")

try:
    ml7_ext_summary = pd.read_csv(ML7_EXT_DIR / 'ml7_extended_summary.csv')
    
    if len(ml7_ext_summary) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Sort by f1_macro_mean descending
        ml7_ext_summary = ml7_ext_summary.sort_values('f1_macro_mean', ascending=False)
        
        # Create barplot with error bars
        x_pos = np.arange(len(ml7_ext_summary))
        bars = ax.bar(x_pos, ml7_ext_summary['f1_macro_mean'], 
                     yerr=ml7_ext_summary['f1_macro_std'],
                     color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.8,
                     capsize=5, error_kw={'linewidth': 1.5})
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(ml7_ext_summary.iterrows()):
            ax.text(i, row['f1_macro_mean'] + row['f1_macro_std'] + 0.02,
                   f"{row['f1_macro_mean']:.3f}",
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(ml7_ext_summary['model'].str.upper(), fontsize=10)
        ax.set_ylabel('Macro-F1 Score', fontsize=11)
        ax.set_xlabel('Model', fontsize=11)
        ax.set_title('Extended ML7 Sequence Models Performance (6-fold Temporal CV)', 
                    fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.axhline(y=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.4, label='Baseline (0.5)')
        ax.legend(loc='lower right', frameon=True, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig11_extended_ml7_models.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'fig11_extended_ml7_models.png', bbox_inches='tight')
        plt.close()
        print("  [OK] fig11_extended_ml7_models.{pdf,png}")
    else:
        print("  [WARN] No extended ML7 data found")
except FileNotFoundError:
    print("  [WARN] Extended ML7 summary not found, skipping fig11")
except Exception as e:
    print(f"  [WARN] Error generating extended ML7 figure: {e}")

# =============================================================================
# FIGURE 12: Temporal Instability Scores (Top-15 Features)
# =============================================================================
print("[12/13] Generating temporal instability scores...")

try:
    instability_df = pd.read_csv(ML6_EXT_DIR / 'instability_scores.csv')
    
    if len(instability_df) > 0:
        # Take top 15 most unstable features
        top_n = 15
        instability_top = instability_df.nlargest(top_n, 'instability_score')
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Horizontal barplot (features on y-axis)
        y_pos = np.arange(len(instability_top))
        bars = ax.barh(y_pos, instability_top['instability_score'],
                      color='crimson', alpha=0.7, edgecolor='black', linewidth=0.8)
        
        # Add value labels
        for i, (idx, row) in enumerate(instability_top.iterrows()):
            ax.text(row['instability_score'] + 0.02, i,
                   f"{row['instability_score']:.4f}",
                   va='center', fontsize=8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(instability_top['feature'], fontsize=9)
        ax.set_xlabel('Instability Score (Normalized)', fontsize=11)
        ax.set_ylabel('Feature', fontsize=11)
        ax.set_title(f'Top-{top_n} Most Temporally Unstable Features\n(Variance of segment-wise means)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig12_instability_scores.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'fig12_instability_scores.png', bbox_inches='tight')
        plt.close()
        print("  [OK] fig12_instability_scores.{pdf,png}")
        print(f"       Top-3 unstable: {', '.join(instability_top['feature'].head(3).tolist())}")
    else:
        print("  [WARN] No instability scores found")
except FileNotFoundError:
    print("  [WARN] Instability scores not found, skipping fig12")
except Exception as e:
    print(f"  [WARN] Error generating instability figure: {e}")

# =============================================================================
# FIGURE 13: Extended Models Combined Comparison (ML6 + ML7)
# =============================================================================
print("[13/13] Generating combined extended models comparison...")

try:
    ml6_ext_summary = pd.read_csv(ML6_EXT_DIR / 'ml6_extended_summary.csv')
    ml7_ext_summary = pd.read_csv(ML7_EXT_DIR / 'ml7_extended_summary.csv')
    
    if len(ml6_ext_summary) > 0 and len(ml7_ext_summary) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: ML6 Extended
        ax1 = axes[0]
        ml6_sorted = ml6_ext_summary.sort_values('f1_macro_mean', ascending=True)
        y_pos_ml6 = np.arange(len(ml6_sorted))
        bars1 = ax1.barh(y_pos_ml6, ml6_sorted['f1_macro_mean'],
                        xerr=ml6_sorted['f1_macro_std'],
                        color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.8,
                        capsize=4, error_kw={'linewidth': 1.2})
        
        for i, (idx, row) in enumerate(ml6_sorted.iterrows()):
            ax1.text(row['f1_macro_mean'] + row['f1_macro_std'] + 0.02, i,
                    f"{row['f1_macro_mean']:.3f}",
                    va='center', fontsize=9, fontweight='bold')
        
        ax1.set_yticks(y_pos_ml6)
        ax1.set_yticklabels(ml6_sorted['model'].str.upper(), fontsize=10)
        ax1.set_xlabel('Macro-F1 Score', fontsize=11)
        ax1.set_title('ML6 Extended Models\n(Tabular, 6-fold CV)', fontsize=11, fontweight='bold')
        ax1.set_xlim(0, 1.0)
        ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Right: ML7 Extended
        ax2 = axes[1]
        ml7_sorted = ml7_ext_summary.sort_values('f1_macro_mean', ascending=True)
        y_pos_ml7 = np.arange(len(ml7_sorted))
        bars2 = ax2.barh(y_pos_ml7, ml7_sorted['f1_macro_mean'],
                        xerr=ml7_sorted['f1_macro_std'],
                        color='darkorange', alpha=0.8, edgecolor='black', linewidth=0.8,
                        capsize=4, error_kw={'linewidth': 1.2})
        
        for i, (idx, row) in enumerate(ml7_sorted.iterrows()):
            ax2.text(row['f1_macro_mean'] + row['f1_macro_std'] + 0.02, i,
                    f"{row['f1_macro_mean']:.3f}",
                    va='center', fontsize=9, fontweight='bold')
        
        ax2.set_yticks(y_pos_ml7)
        ax2.set_yticklabels(ml7_sorted['model'].str.upper(), fontsize=10)
        ax2.set_xlabel('Macro-F1 Score', fontsize=11)
        ax2.set_title('ML7 Extended Models\n(Sequence, 6-fold CV)', fontsize=11, fontweight='bold')
        ax2.set_xlim(0, 1.0)
        ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=0.8, alpha=0.4)
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.suptitle('Extended Models Performance Comparison', fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'fig13_extended_models_combined.pdf', bbox_inches='tight')
        plt.savefig(FIG_DIR / 'fig13_extended_models_combined.png', bbox_inches='tight')
        plt.close()
        print("  [OK] fig13_extended_models_combined.{pdf,png}")
    else:
        print("  [WARN] Extended model data incomplete, skipping combined figure")
except FileNotFoundError:
    print("  [WARN] Extended model summaries not found, skipping fig13")
except Exception as e:
    print(f"  [WARN] Error generating combined extended figure: {e}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PhD DISSERTATION FIGURES GENERATION COMPLETE")
print("="*70)
print(f"\nParticipant: {PID}")
print(f"Snapshot: {SNAPSHOT}")
print(f"Output directory: {FIG_DIR.absolute()}")
print(f"\n--- Core Pipeline Figures (1-9) ---")
core_figs = [
    'fig01_pbsi_timeline',
    'fig02_cardio_distributions',
    'fig03_sleep_activity_boxplots',
    'fig04_segmentwise_normalization',
    'fig05_missing_data_pattern',
    'fig06_label_distribution_timeline',
    'fig07_correlation_heatmap',
    'fig08_adwin_drift',
    'fig09_shap_importance'
]
for fig_name in core_figs:
    pdf_path = FIG_DIR / f'{fig_name}.pdf'
    if pdf_path.exists():
        print(f"  [OK] {fig_name}.{{pdf,png}}")
    else:
        print(f"  [SKIP] {fig_name}.{{pdf,png}}")

print(f"\n--- Extended Models Figures (10-13) ---")
extended_figs = [
    'fig10_extended_ml6_models',
    'fig11_extended_ml7_models',
    'fig12_instability_scores',
    'fig13_extended_models_combined'
]
for fig_name in extended_figs:
    pdf_path = FIG_DIR / f'{fig_name}.pdf'
    if pdf_path.exists():
        print(f"  [OK] {fig_name}.{{pdf,png}}")
    else:
        print(f"  [SKIP] {fig_name}.{{pdf,png}}")

total_figs = len(list(FIG_DIR.glob('fig*.pdf')))
print(f"\n--- Summary ---")
print(f"Total figures: {total_figs} (PDF + PNG for each)")
print(f"Resolution: 300 DPI")
print(f"Style: Publication-ready, grayscale-friendly")
print(f"\n[OK] Ready for LaTeX inclusion with \\includegraphics{{}}")
print(f"\nTo regenerate all figures:")
print(f"  python scripts/generate_dissertation_figures.py")
print("="*70)
