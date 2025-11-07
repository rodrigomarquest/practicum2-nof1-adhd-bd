#!/usr/bin/env python3
"""
Generate NB2 baselines summary markdown.

Reads the baseline CSVs and creates a side-by-side comparison table.
"""

import pandas as pd
from pathlib import Path
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Generate summary markdown from baseline CSVs."""
    results_dir = Path("nb2")
    
    # Load CSVs
    try:
        df_3cls = pd.read_csv(results_dir / "baselines_label_3cls.csv")
        df_2cls = pd.read_csv(results_dir / "baselines_label_2cls.csv")
    except FileNotFoundError as e:
        logger.error(f"CSV not found: {e}")
        sys.exit(1)
    
    # Extract MEAN rows
    mean_row_3cls = df_3cls[df_3cls['fold'] == 'MEAN'].iloc[0]
    mean_row_2cls = df_2cls[df_2cls['fold'] == 'MEAN'].iloc[0]
    
    # Build markdown
    md = []
    md.append("# NB2 Baselines Summary\n")
    
    md.append("## 3-Class Task (label_3cls)\n")
    md.append(
        "| Baseline | F1-Macro | F1-Weighted | Balanced-Acc | Kappa |\n"
        "|----------|----------|-------------|--------------|-------|\n"
    )
    
    for baseline in ['dummy', 'naive', 'ma7', 'rule', 'lr']:
        f1_macro = mean_row_3cls.get(f'{baseline}_f1_macro', np.nan)
        f1_weighted = mean_row_3cls.get(f'{baseline}_f1_weighted', np.nan)
        balanced_acc = mean_row_3cls.get(f'{baseline}_balanced_acc', np.nan)
        kappa = mean_row_3cls.get(f'{baseline}_kappa', np.nan)
        
        md.append(
            f"| {baseline:12s} | {f1_macro:8.4f} | {f1_weighted:11.4f} | "
            f"{balanced_acc:12.4f} | {kappa:7.4f} |\n"
        )
    
    md.append("\n## 2-Class Task (label_2cls)\n")
    md.append(
        "| Baseline | F1-Macro | F1-Weighted | Balanced-Acc | Kappa | ROC-AUC | McNemar-p |\n"
        "|----------|----------|-------------|--------------|-------|---------|----------|\n"
    )
    
    for baseline in ['dummy', 'naive', 'ma7', 'rule', 'lr']:
        f1_macro = mean_row_2cls.get(f'{baseline}_f1_macro', np.nan)
        f1_weighted = mean_row_2cls.get(f'{baseline}_f1_weighted', np.nan)
        balanced_acc = mean_row_2cls.get(f'{baseline}_balanced_acc', np.nan)
        kappa = mean_row_2cls.get(f'{baseline}_kappa', np.nan)
        roc_auc = mean_row_2cls.get(f'{baseline}_roc_auc', np.nan)
        mcnemar_p = mean_row_2cls.get(f'{baseline}_mcnemar_p', np.nan)
        
        roc_str = f"{roc_auc:.4f}" if not pd.isna(roc_auc) else "N/A"
        mcn_str = f"{mcnemar_p:.4f}" if not pd.isna(mcnemar_p) else "N/A"
        
        md.append(
            f"| {baseline:12s} | {f1_macro:8.4f} | {f1_weighted:11.4f} | "
            f"{balanced_acc:12.4f} | {kappa:7.4f} | {roc_str:7s} | {mcn_str:8s} |\n"
        )
    
    md.append("\n## Notes\n\n")
    md.append("- **Dummy**: Stratified random (baseline)\n")
    md.append("- **Naive**: Yesterday's label (7-day mode fallback)\n")
    md.append("- **MA7**: 7-day moving average, quantized at ±0.33\n")
    md.append("- **Rule**: Clinical pbsi_score thresholds (±0.5)\n")
    md.append("- **LR**: Logistic Regression (L2, C tuning, balanced class weight)\n")
    md.append("\n- **F1-Macro**: Average F1 across classes\n")
    md.append("- **F1-Weighted**: Weighted by class support\n")
    md.append("- **Balanced-Acc**: Mean recall per class\n")
    md.append("- **Kappa**: Cohen's kappa agreement\n")
    md.append("- **ROC-AUC**: Area under ROC curve (2-class only)\n")
    md.append("- **McNemar-p**: McNemar test p-value vs Dummy (2-class only)\n")
    md.append("\n**Seed**: 42 (deterministic)\n")
    md.append("\n**Folds**: 6 calendar-based (4 months train / 2 months validation)\n")
    md.append("\n**Confusion Matrices**: See `nb2/confusion_matrices/*.png`\n")
    
    # Write
    output_file = results_dir / "baselines_summary.md"
    with open(output_file, 'w') as f:
        f.writelines(md)
    
    logger.info(f"Saved: {output_file}")
    print("".join(md))


if __name__ == "__main__":
    import numpy as np
    main()
