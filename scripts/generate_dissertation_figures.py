"""
Generate PhD Dissertation Figures
==================================
Purpose: Generate publication-quality figures for LaTeX dissertation
Output: docs/latex/figures/
Snapshot: P000001/2025-11-07
Style: PhD-level academic (grayscale-friendly, high DPI)
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
DATA_DIR = Path('data/etl/P000001/2025-11-07/joined')
FIG_DIR = Path('docs/latex/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {FIG_DIR}")
print(f"Figure settings: 300 DPI, grayscale-friendly\n")

# Load data
df = pd.read_csv(DATA_DIR / 'features_daily_labeled.csv', parse_dates=['date'])
print(f"Total days: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nLabel distribution:")
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

# Top: Raw
ax1 = axes[0]
for seg in demo_data['segment_id'].unique()[:10]:
    seg_data = demo_data[demo_data['segment_id'] == seg]
    ax1.plot(seg_data['date'], seg_data[feature_raw], 
             marker='o', markersize=2, linewidth=0.8, alpha=0.7)

ax1.set_ylabel('Raw HR Mean (bpm)', fontsize=10)
ax1.set_title('Before Segment-wise Normalization', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Bottom: Z-scored
ax2 = axes[1]
for seg in demo_data['segment_id'].unique()[:10]:
    seg_data = demo_data[demo_data['segment_id'] == seg]
    ax2.plot(seg_data['date'], seg_data[feature_zscore], 
             marker='o', markersize=2, linewidth=0.8, alpha=0.7)

ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
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
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("PhD DISSERTATION FIGURES GENERATION COMPLETE")
print("="*70)
print(f"\nOutput directory: {FIG_DIR.absolute()}")
print(f"\nGenerated figures:")
for fig_file in sorted(FIG_DIR.glob('fig*.pdf')):
    print(f"  [OK] {fig_file.name}")
print(f"\nTotal figures: {len(list(FIG_DIR.glob('fig*.pdf')))} (PDF + PNG for each)")
print(f"Resolution: 300 DPI")
print(f"Style: Publication-ready, grayscale-friendly")
print(f"\n[OK] Ready for LaTeX inclusion with \\includegraphics{{}}")
