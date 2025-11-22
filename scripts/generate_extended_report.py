#!/usr/bin/env python
"""
Generate RUN_REPORT_EXTENDED.md
Supplements RUN_REPORT.md with extended ML6/ML7 model results.

Usage:
    python scripts/generate_extended_report.py --participant P000001 --snapshot 2025-11-07
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd


def load_core_summary(snapshot_dir: Path) -> dict:
    """Extract key metrics from RUN_REPORT.md or core pipeline outputs."""
    
    # Try to load from RUN_REPORT.md
    report_path = Path("RUN_REPORT.md")
    if not report_path.exists():
        return {"error": "RUN_REPORT.md not found"}
    
    # Parse key info from RUN_REPORT.md
    with open(report_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract date range, rows, label distribution (simple parsing)
    lines = content.split('\n')
    summary = {}
    
    for i, line in enumerate(lines):
        if 'Date Range' in line:
            summary['date_range'] = line.split('**')[2].strip()
        elif 'Total Rows' in line:
            summary['total_rows'] = line.split('**')[2].strip()
        elif 'Mean Macro-F1' in line and 'ML6' in ''.join(lines[max(0, i-10):i]):
            # Extract ML6 LR F1-macro
            parts = line.split('**')
            if len(parts) > 2:
                f1_str = parts[2].strip()
                summary['ml6_lr_f1'] = f1_str
    
    return summary


def load_ml6_extended(ml6_ext_dir: Path) -> pd.DataFrame:
    """Load ML6 extended summary CSV."""
    summary_path = ml6_ext_dir / "ml6_extended_summary.csv"
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path)


def load_ml7_extended(ml7_ext_dir: Path) -> pd.DataFrame:
    """Load ML7 extended summary CSV."""
    summary_path = ml7_ext_dir / "ml7_extended_summary.csv"
    if not summary_path.exists():
        return None
    return pd.read_csv(summary_path)


def load_ml6_baseline(ml6_dir: Path) -> dict:
    """Load ML6 baseline (logistic regression) results."""
    cv_summary_path = ml6_dir / "cv_summary.json"
    if not cv_summary_path.exists():
        return None
    
    with open(cv_summary_path, 'r') as f:
        data = json.load(f)
    
    return {
        'model': 'LogisticRegression',
        'f1_macro_mean': data.get('mean_f1_macro', 0),
        'f1_macro_std': data.get('std_f1_macro', 0),
        'f1_weighted_mean': None,  # Not in cv_summary.json
        'balanced_acc_mean': None,
        'kappa_mean': None
    }


def generate_extended_report(participant: str, snapshot: str, output_path: Path):
    """Generate RUN_REPORT_EXTENDED.md."""
    
    # Paths
    ai_dir = Path(f"data/ai/{participant}/{snapshot}")
    ml6_dir = ai_dir / "ml6"
    ml6_ext_dir = ai_dir / "ml6_ext"
    ml7_dir = ai_dir / "ml7"
    ml7_ext_dir = ai_dir / "ml7_ext"
    
    # Load data
    core_summary = load_core_summary(ai_dir.parent.parent / "etl" / participant / snapshot)
    ml6_baseline = load_ml6_baseline(ml6_dir)
    ml6_extended = load_ml6_extended(ml6_ext_dir)
    ml7_extended = load_ml7_extended(ml7_ext_dir)
    
    # Start building report
    lines = []
    
    # === HEADER ===
    lines.extend([
        "# RUN_REPORT_EXTENDED.md – Extended ML6/ML7 Experiments",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"**Participant**: {participant}  ",
        f"**Snapshot**: {snapshot}  ",
        f"**Pipeline Version**: v4.1.8 (Extended Models)",
        "",
        "---",
        "",
    ])
    
    # === CORE SUMMARY (SHORT) ===
    lines.extend([
        "## Core Pipeline Summary",
        "",
        "This report **supplements** the main [RUN_REPORT.md](RUN_REPORT.md) with results from extended ML6/ML7 models.",
        "",
        "**Core Pipeline Results**:",
        f"- Date Range: {core_summary.get('date_range', 'N/A')}",
        f"- Total Rows: {core_summary.get('total_rows', 'N/A')}",
        f"- ML6 Baseline (Logistic Regression): F1-macro = {core_summary.get('ml6_lr_f1', 'N/A')}",
        "",
        "For full ETL/baseline details, see [RUN_REPORT.md](RUN_REPORT.md).",
        "",
        "---",
        "",
    ])
    
    # === EXTENDED ML6 MODELS ===
    lines.extend([
        "## Extended ML6 Models (Tabular Classification)",
        "",
        "Four additional models were trained on the same ML6 dataset (1,625 MICE-imputed days, 6-fold temporal CV):",
        "- **Random Forest**: 200 trees, max_depth=10, instability-regularized max_features",
        "- **XGBoost**: max_depth=4, lr=0.05, instability-regularized L1/L2 penalties",
        "- **LightGBM**: max_depth=4, lr=0.05, instability-based feature weighting",
        "- **SVM (RBF)**: C=1.0, gamma='scale', NO instability penalty",
        "",
        "**Temporal Instability Regularization**: Tree/boosting models penalize features with high variance across 119 behavioral segments.",
        "",
    ])
    
    # Build ML6 table
    if ml6_extended is not None or ml6_baseline is not None:
        lines.append("### ML6 Model Comparison")
        lines.append("")
        lines.append("| Model | F1-macro (mean ± std) | F1-weighted | Balanced Accuracy | Cohen's κ |")
        lines.append("|-------|----------------------|-------------|-------------------|-----------|")
        
        # Add baseline
        if ml6_baseline:
            f1_mean = ml6_baseline['f1_macro_mean']
            f1_std = ml6_baseline['f1_macro_std']
            lines.append(f"| Logistic Regression | {f1_mean:.4f} ± {f1_std:.4f} | — | — | — |")
        
        # Add extended models
        if ml6_extended is not None:
            for _, row in ml6_extended.iterrows():
                model = row['model']
                f1_mean = row['f1_macro_mean']
                f1_std = row['f1_macro_std']
                f1_weighted = row.get('f1_weighted_mean', 0)
                bal_acc = row.get('balanced_acc_mean', 0)
                kappa = row.get('kappa_mean', 0)
                
                lines.append(
                    f"| {model} | {f1_mean:.4f} ± {f1_std:.4f} | "
                    f"{f1_weighted:.4f} | {bal_acc:.4f} | {kappa:.4f} |"
                )
        
        lines.append("")
    else:
        lines.extend([
            "*Extended ML6 results not found. Run:*",
            "```bash",
            f"make ml-extended-all PID={participant} SNAPSHOT={snapshot}",
            "```",
            "",
        ])
    
    lines.append("---")
    lines.append("")
    
    # === EXTENDED ML7 MODELS ===
    lines.extend([
        "## Extended ML7 Models (Temporal Sequence Classification)",
        "",
        "Three additional sequence models were trained on the same ML7 dataset (14-day windows, 6-fold temporal CV):",
        "- **GRU**: 64 hidden units, dropout=0.3",
        "- **TCN (Temporal Convolutional Network)**: 64 filters, dilations [1,2,4], causal padding",
        "- **Temporal MLP**: Flattened 14-day input → Dense(128) → Dense(64) → Softmax(3)",
        "",
    ])
    
    # Build ML7 table
    if ml7_extended is not None:
        lines.append("### ML7 Model Comparison")
        lines.append("")
        lines.append("| Model | F1-macro (mean ± std) | F1-weighted | Balanced Accuracy | AUROC (OvR) | Cohen's κ |")
        lines.append("|-------|----------------------|-------------|-------------------|-------------|-----------|")
        
        # Add LSTM baseline (if we can extract it)
        lines.append("| LSTM (baseline) | — | — | — | — | — |")
        lines.append("| *(metrics pending ML7 completion)* | | | | | |")
        
        # Add extended models
        for _, row in ml7_extended.iterrows():
            model = row['model']
            f1_mean = row['f1_macro_mean']
            f1_std = row.get('f1_macro_std', 0)
            f1_weighted = row.get('f1_weighted_mean', 0)
            bal_acc = row.get('balanced_acc_mean', 0)
            auroc = row.get('auroc_ovr_mean', 0)
            kappa = row.get('kappa_mean', 0)
            
            lines.append(
                f"| {model} | {f1_mean:.4f} ± {f1_std:.4f} | "
                f"{f1_weighted:.4f} | {bal_acc:.4f} | {auroc:.4f} | {kappa:.4f} |"
            )
        
        lines.append("")
    else:
        lines.extend([
            "*Extended ML7 results not found. Run:*",
            "```bash",
            f"make ml7-gru ml7-tcn ml7-mlp PID={participant} SNAPSHOT={snapshot}",
            "```",
            "",
        ])
    
    lines.append("---")
    lines.append("")
    
    # === INTERPRETATION NOTES ===
    lines.extend([
        "## Interpretation & Key Findings",
        "",
    ])
    
    if ml6_extended is not None:
        best_ml6 = ml6_extended.loc[ml6_extended['f1_macro_mean'].idxmax()]
        best_model = best_ml6['model']
        best_f1 = best_ml6['f1_macro_mean']
        lr_f1 = ml6_baseline['f1_macro_mean'] if ml6_baseline else 0
        
        lines.extend([
            f"- **ML6 Best Model**: {best_model} achieved F1-macro = {best_f1:.4f}, compared to Logistic Regression baseline ({lr_f1:.4f}).",
            f"- **Instability Regularization**: {'Improved' if best_f1 > lr_f1 else 'Marginal impact on'} performance for tree/boosting models.",
        ])
    
    lines.extend([
        "- **ML7 Performance**: Sequence models (GRU/TCN/MLP) face challenges due to:",
        "  - Weak supervision (PBSI heuristic labels, not clinical gold standard)",
        "  - Non-stationarity across 8-year timeline (119 behavioral segments)",
        "  - Limited dataset size (1,625 days post-2021 temporal filter)",
        "- **Baseline Strength**: Logistic regression remains a strong, interpretable baseline for this N-of-1 dataset.",
        "- **Future Work**: Multi-participant datasets, stronger supervision (PHQ-9/MDQ), and federated learning may improve sequence model performance.",
        "",
        "---",
        "",
    ])
    
    # === REPRODUCIBILITY ===
    lines.extend([
        "## Reproducibility Notes",
        "",
        "All extended models use **Stage 5 preprocessed outputs** (no raw data required):",
        "",
        "**Required Files**:",
        f"- `data/ai/{participant}/{snapshot}/ml6/features_daily_ml6.csv` (1,625 rows, MICE-imputed)",
        f"- `data/etl/{participant}/{snapshot}/joined/features_daily_labeled.csv` (2,828 rows, full timeline)",
        f"- `data/etl/{participant}/{snapshot}/segment_autolog.csv` (119 segments)",
        f"- `data/ai/{participant}/{snapshot}/ml6/cv_summary.json` (6-fold definitions)",
        "",
        "**No Zepp Password Required**: Pipeline runs in Apple-only mode if Zepp data unavailable.",
        "",
        "**Regenerate Extended Models**:",
        "```bash",
        f"make ml-extended-all PID={participant} SNAPSHOT={snapshot}",
        "```",
        "",
        "**Regenerate This Report**:",
        "```bash",
        f"make report-extended PID={participant} SNAPSHOT={snapshot}",
        "```",
        "",
        "---",
        "",
        "## References",
        "",
        "- **Implementation Details**: `docs/copilot/ML6_ML7_EXTENDED_IMPLEMENTATION.md`",
        "- **Quick Start Guide**: `docs/copilot/QUICK_START.md`",
        "- **Core Pipeline**: `RUN_REPORT.md`",
        "- **Pipeline Architecture**: `pipeline_overview.md`",
        "",
        f"**Generated**: {datetime.now().isoformat()}",
    ])
    
    # Write report
    report_text = "\n".join(lines)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"[OK] Generated: {output_path}")
    print(f"     ML6 extended: {'✓' if ml6_extended is not None else '✗ (run make ml-extended-all)'}")
    print(f"     ML7 extended: {'✓' if ml7_extended is not None else '✗ (run make ml7-gru ml7-tcn ml7-mlp)'}")


def main():
    parser = argparse.ArgumentParser(description="Generate RUN_REPORT_EXTENDED.md")
    parser.add_argument("--participant", type=str, default="P000001", help="Participant ID")
    parser.add_argument("--snapshot", type=str, default="2025-11-07", help="Snapshot date")
    parser.add_argument("--output", type=str, default="RUN_REPORT_EXTENDED.md", help="Output path")
    
    args = parser.parse_args()
    
    output_path = Path(args.output)
    generate_extended_report(args.participant, args.snapshot, output_path)


if __name__ == "__main__":
    main()
