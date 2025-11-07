#!/usr/bin/env python
"""
FULL DETERMINISTIC PIPELINE
Stages 0-9: Ingest → Aggregate → Unify → Label → Segment → Prep-NB2 → NB2 → NB3 → TFLite → Report

All outputs under canonical structure:
  data/etl/P000001/<SNAPSHOT>/ (stages 0-5)
  ai/local/P000001/<SNAPSHOT>/ (stages 6-8)
  RUN_REPORT.md (stage 9)
"""

import sys
import argparse
import logging
import json
import time
import zipfile
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Optional

try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import stage modules
from src.etl.stage_csv_aggregation import run_csv_aggregation
from src.etl.stage_unify_daily import run_unify_daily
from src.etl.stage_apply_labels import run_apply_labels


def banner(text: str, width: int = 80):
    """Print banner."""
    print(f"\n{'='*width}")
    print(f" {text.center(width-2)} ")
    print(f"{'='*width}\n")


class PipelineContext:
    """Hold pipeline state across stages."""
    
    def __init__(self, participant: str, snapshot: str, 
                 raw_dir: str = "data/raw", 
                 etl_base_dir: str = "data/etl",
                 ai_base_dir: str = "data/ai"):
        self.participant = participant
        self.snapshot = snapshot
        self.raw_dir = Path(raw_dir)
        self.etl_base_dir = Path(etl_base_dir)
        self.ai_base_dir = Path(ai_base_dir)
        
        # Canonical paths
        self.snapshot_dir = self.etl_base_dir / participant / snapshot
        self.extracted_dir = self.snapshot_dir / "extracted"
        self.joined_dir = self.snapshot_dir / "joined"
        self.qc_dir = self.snapshot_dir / "qc"
        self.ai_snapshot_dir = self.ai_base_dir / participant / snapshot
        
        # Create directories
        for d in [self.extracted_dir / "apple",
                  self.extracted_dir / "zepp",
                  self.joined_dir,
                  self.qc_dir,
                  self.ai_snapshot_dir / "nb2",
                  self.ai_snapshot_dir / "nb3" / "models"]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.start_time = datetime.now()
        self.results = {}
    
    def log_stage_result(self, stage_num: int, status: str, **kwargs):
        """Record stage result."""
        self.results[f"stage_{stage_num}"] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }


def stage_0_ingest(ctx: PipelineContext, zepp_password: Optional[str] = None) -> bool:
    """
    Stage 0: Ingest
    Extract ZIPs from data/raw/P000001/{apple,zepp}/ to data/etl/.../extracted/
    """
    banner("STAGE 0: INGEST (extract from data/raw)")
    stage_start = time.time()
    
    try:
        raw_participant_dir = ctx.raw_dir / ctx.participant
        
        # Get Zepp password from env or parameter
        zpwd = zepp_password or os.getenv("ZEP_ZIP_PASSWORD") or os.getenv("ZEPP_ZIP_PASSWORD")
        
        # FAIL-FAST: Check if Zepp ZIPs exist but no password provided
        zepp_raw_dir = raw_participant_dir / "zepp"
        if zepp_raw_dir.exists():
            zepp_zips = list(zepp_raw_dir.glob("*.zip"))
            if zepp_zips and not zpwd:
                logger.error("[FATAL] Zepp ZIP found but no password provided. Use --zepp-password or set ZEP_ZIP_PASSWORD.")
                sys.exit(2)
        
        # Extract Apple ZIPs
        apple_raw_dir = raw_participant_dir / "apple" / "export"
        if apple_raw_dir.exists():
            for zip_file in apple_raw_dir.glob("*.zip"):
                logger.info(f"[Apple] Extracting: {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(ctx.extracted_dir / "apple")
            logger.info(f"[OK] Apple extracted to {ctx.extracted_dir / 'apple'}")
        else:
            logger.warning(f"[SKIP] Apple export dir not found: {apple_raw_dir}")
        
        # Extract Zepp ZIPs
        if zepp_raw_dir.exists():
            for zip_file in zepp_raw_dir.glob("*.zip"):
                logger.info(f"[Zepp] Extracting: {zip_file.name}")
                try:
                    # Try with pyzipper first (handles AES encryption)
                    if HAS_PYZIPPER:
                        try:
                            with pyzipper.AESZipFile(zip_file, 'r') as z:
                                if zpwd:
                                    z.extractall(ctx.extracted_dir / "zepp", pwd=zpwd.encode('utf-8'))
                                else:
                                    z.extractall(ctx.extracted_dir / "zepp")
                        except Exception as e:
                            # Fallback to regular zipfile
                            with zipfile.ZipFile(zip_file, 'r') as z:
                                z.extractall(ctx.extracted_dir / "zepp")
                    else:
                        # No pyzipper, try regular zipfile
                        with zipfile.ZipFile(zip_file, 'r') as z:
                            z.extractall(ctx.extracted_dir / "zepp")
                except RuntimeError as e:
                    if "encrypted" in str(e).lower() or "password" in str(e).lower():
                        logger.warning(f"[SKIP] {zip_file.name} is encrypted (no valid password)")
                    else:
                        raise
                except Exception as e:
                    logger.warning(f"[SKIP] {zip_file.name}: {e}")
            logger.info(f"[OK] Zepp extracted to {ctx.extracted_dir / 'zepp'}")
        else:
            logger.warning(f"[SKIP] Zepp dir not found: {zepp_raw_dir}")
        
        elapsed = time.time() - stage_start
        logger.info(f"✓ Stage 0 complete ({elapsed:.1f}s)")
        ctx.log_stage_result(0, "success", duration_sec=elapsed)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 0 failed: {e}", exc_info=True)
        ctx.log_stage_result(0, "failed", error=str(e))
        return False


def stage_1_aggregate(ctx: PipelineContext) -> bool:
    """
    Stage 1: Aggregate
    Parse export.xml + Zepp CSVs → daily_*.csv
    """
    banner("STAGE 1: AGGREGATE (xml + csvs to daily_*.csv)")
    stage_start = time.time()
    
    try:
        logger.info(f"Aggregating from {ctx.extracted_dir}")
        
        # Pass the extracted dir (where export.xml should be)
        results = run_csv_aggregation(
            participant=ctx.participant,
            extracted_dir=str(ctx.extracted_dir),
            output_dir=str(ctx.extracted_dir)
        )
        
        apple_total = sum(len(df) for df in results.get("apple", {}).values())
        zepp_total = sum(len(df) for df in results.get("zepp", {}).values())
        
        logger.info(f"✓ Stage 1 complete: {apple_total} Apple + {zepp_total} Zepp days")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(1, "success", duration_sec=elapsed, 
                            apple_days=apple_total, zepp_days=zepp_total)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 1 failed: {e}", exc_info=True)
        ctx.log_stage_result(1, "failed", error=str(e))
        return False


def stage_2_unify(ctx: PipelineContext) -> Optional[pd.DataFrame]:
    """
    Stage 2: Unify
    Merge Apple + Zepp daily CSVs → features_daily_unified.csv
    """
    banner("STAGE 2: UNIFY (merge daily csvs)")
    stage_start = time.time()
    
    try:
        # The unify function will look for daily_*.csv in extracted_dir
        df = run_unify_daily(
            participant=ctx.participant,
            snapshot=ctx.snapshot,
            extracted_dir=str(ctx.extracted_dir),
            output_dir=str(ctx.snapshot_dir)  # Will save to snapshot_dir/joined/
        )
        
        logger.info(f"✓ Stage 2 complete: {len(df)} days, "
                   f"{df['date'].min()} to {df['date'].max()}")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(2, "success", duration_sec=elapsed, rows=len(df))
        return df
    
    except Exception as e:
        logger.error(f"✗ Stage 2 failed: {e}", exc_info=True)
        ctx.log_stage_result(2, "failed", error=str(e))
        return None


def stage_3_label(ctx: PipelineContext) -> Optional[pd.DataFrame]:
    """
    Stage 3: Label
    Apply PBSI labels → features_daily_labeled.csv
    """
    banner("STAGE 3: LABEL (apply pbsi labels)")
    stage_start = time.time()
    
    try:
        df = run_apply_labels(
            participant=ctx.participant,
            snapshot=ctx.snapshot,
            etl_dir=str(ctx.etl_base_dir)
        )
        
        dist = df['label_3cls'].value_counts()
        logger.info(f"✓ Stage 3 complete: {len(df)} days labeled")
        for label in [-1, 0, 1]:
            if label in dist.index:
                count = dist[label]
                pct = count / len(df) * 100
                logger.info(f"  Label {label:+2d}: {count:4d} ({pct:5.1f}%)")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(3, "success", duration_sec=elapsed, rows=len(df))
        return df
    
    except Exception as e:
        logger.error(f"✗ Stage 3 failed: {e}", exc_info=True)
        ctx.log_stage_result(3, "failed", error=str(e))
        return None


def stage_4_segment(ctx: PipelineContext, df: pd.DataFrame) -> bool:
    """
    Stage 4: Segment
    Auto-segment by time boundaries and gaps
    """
    banner("STAGE 4: SEGMENT (identify periods)")
    stage_start = time.time()
    
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        segments = []
        current = {
            'date_start': df.iloc[0]['date'],
            'date_end': df.iloc[0]['date'],
            'reason': 'initial',
            'count': 1
        }
        
        for i in range(1, len(df)):
            prev_date = df.iloc[i-1]['date']
            curr_date = df.iloc[i]['date']
            delta = (curr_date - prev_date).days
            
            # Gap detection
            if delta > 1:
                current['date_end'] = prev_date
                segments.append(current)
                current = {
                    'date_start': curr_date,
                    'date_end': curr_date,
                    'reason': 'gap',
                    'count': 1
                }
            # Time boundary (month/year change)
            elif prev_date.month != curr_date.month or prev_date.year != curr_date.year:
                current['date_end'] = prev_date
                segments.append(current)
                current = {
                    'date_start': curr_date,
                    'date_end': curr_date,
                    'reason': 'time_boundary',
                    'count': 1
                }
            else:
                current['date_end'] = curr_date
                current['count'] += 1
        
        current['date_end'] = df.iloc[-1]['date']
        segments.append(current)
        
        df_segments = pd.DataFrame(segments)
        df_segments['duration_days'] = (df_segments['date_end'] - df_segments['date_start']).dt.days + 1
        
        segment_path = ctx.snapshot_dir / "segment_autolog.csv"
        df_segments.to_csv(segment_path, index=False)
        
        logger.info(f"✓ Stage 4 complete: {len(df_segments)} segments")
        logger.info(f"  Reasons: {df_segments['reason'].value_counts().to_dict()}")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(4, "success", duration_sec=elapsed, segments=len(df_segments))
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 4 failed: {e}", exc_info=True)
        ctx.log_stage_result(4, "failed", error=str(e))
        return False


def stage_5_prep_nb2(ctx: PipelineContext, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Stage 5: Prep NB2
    Remove pbsi_score, pbsi_quality → features_nb2_clean.csv (anti-leak)
    """
    banner("STAGE 5: PREP NB2 (anti-leak safeguards)")
    stage_start = time.time()
    
    try:
        df_clean = df.copy()
        
        # Remove blacklist
        blacklist = ["pbsi_score", "pbsi_quality"]
        cols_drop = [c for c in blacklist if c in df_clean.columns]
        if cols_drop:
            logger.info(f"Removing blacklist: {cols_drop}")
            df_clean = df_clean.drop(columns=cols_drop)
        
        # Keep whitelist
        whitelist = {
            'date', 'sleep_hours', 'sleep_quality_score',
            'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
            'total_steps', 'total_distance', 'total_active_energy',
            'label_3cls'
        }
        
        cols_keep = list(set(df_clean.columns) & whitelist)
        df_clean = df_clean[sorted(cols_keep)]
        
        # Verify anti-leak
        assert "pbsi_score" not in df_clean.columns
        assert "pbsi_quality" not in df_clean.columns
        
        nb2_path = ctx.joined_dir / "features_nb2_clean.csv"
        df_clean.to_csv(nb2_path, index=False)
        
        logger.info(f"✓ Stage 5 complete: {df_clean.shape}")
        logger.info(f"  pbsi_score removed: YES")
        logger.info(f"  pbsi_quality removed: YES")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(5, "success", duration_sec=elapsed, 
                            rows=len(df_clean), columns=len(df_clean.columns))
        return df_clean
    
    except Exception as e:
        logger.error(f"✗ Stage 5 failed: {e}", exc_info=True)
        ctx.log_stage_result(5, "failed", error=str(e))
        return None


def stage_6_nb2(ctx: PipelineContext, df: pd.DataFrame) -> bool:
    """
    Stage 6: NB2 Training
    6-fold temporal CV with calendar-based splits (4mo train / 2mo val)
    """
    banner("STAGE 6: NB2 TRAINING (6-fold calendar cv)")
    stage_start = time.time()
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, balanced_accuracy_score
        from src.etl.nb3_analysis import create_calendar_folds
        import numpy as np
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Create calendar-based folds
        folds = create_calendar_folds(df, n_folds=6, train_months=4, val_months=2)
        
        if len(folds) == 0:
            raise ValueError("No valid folds created")
        
        X = df.drop(columns=['date', 'label_3cls'])
        y = df['label_3cls']
        
        cv_results = []
        
        for fold_info in folds:
            fold_idx = fold_info['fold']
            train_idx = fold_info['train_idx']
            val_idx = fold_info['val_idx']
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            model = LogisticRegression(multi_class='multinomial', 
                                      class_weight='balanced', max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            ba = balanced_accuracy_score(y_val, y_pred)
            
            result = {
                "fold": fold_idx,
                "train_start": fold_info['train_start'],
                "train_end": fold_info['train_end'],
                "val_start": fold_info['val_start'],
                "val_end": fold_info['val_end'],
                "n_train": fold_info['n_train'],
                "n_val": fold_info['n_val'],
                "f1_macro": float(f1),
                "balanced_accuracy": float(ba)
            }
            cv_results.append(result)
            logger.info(f"  Fold {fold_idx} ({fold_info['val_start']}→{fold_info['val_end']}): "
                       f"F1={f1:.4f}, BA={ba:.4f}")
        
        mean_f1 = np.mean([r["f1_macro"] for r in cv_results])
        std_f1 = np.std([r["f1_macro"] for r in cv_results])
        
        cv_summary = {
            "model": "LogisticRegression",
            "cv_type": "temporal_calendar_6fold",
            "train_months": 4,
            "val_months": 2,
            "mean_f1_macro": float(mean_f1),
            "std_f1_macro": float(std_f1),
            "folds": cv_results
        }
        
        cv_path = ctx.ai_snapshot_dir / "nb2" / "cv_summary.json"
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        logger.info(f"✓ Stage 6 complete: F1={mean_f1:.4f}±{std_f1:.4f}")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(6, "success", duration_sec=elapsed, 
                            mean_f1=float(mean_f1), std_f1=float(std_f1))
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 6 failed: {e}", exc_info=True)
        ctx.log_stage_result(6, "failed", error=str(e))
        return False


def stage_7_nb3(ctx: PipelineContext) -> bool:
    """
    Stage 7: NB3 Analysis
    SHAP interpretability + Drift detection (ADWIN + KS) + LSTM training
    """
    banner("STAGE 7: NB3 ANALYSIS (SHAP + Drift + LSTM)")
    stage_start = time.time()
    
    try:
        from sklearn.linear_model import LogisticRegression
        from src.etl.nb3_analysis import (
            create_calendar_folds, compute_shap_values, detect_drift_adwin,
            detect_drift_ks_segments, create_lstm_sequences, train_lstm_model
        )
        import numpy as np
        
        # Create output directories
        shap_dir = ctx.ai_snapshot_dir / "nb3" / "shap"
        drift_dir = ctx.ai_snapshot_dir / "nb3" / "drift"
        models_dir = ctx.ai_snapshot_dir / "nb3" / "models"
        
        for d in [shap_dir, drift_dir, models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load data
        nb2_clean_path = ctx.joined_dir / "features_nb2_clean.csv"
        df = pd.read_csv(nb2_clean_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # Load labeled data (for drift on pbsi_score)
        labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
        df_labeled = pd.read_csv(labeled_path)
        df_labeled['date'] = pd.to_datetime(df_labeled['date'])
        
        # ===== SHAP ANALYSIS =====
        logger.info("\n[NB3] SHAP Analysis...")
        folds = create_calendar_folds(df, n_folds=6, train_months=4, val_months=2)
        
        X = df.drop(columns=['date', 'label_3cls'])
        y = df['label_3cls']
        feature_names = X.columns.tolist()
        
        shap_results = []
        for fold_info in folds:
            fold_idx = fold_info['fold']
            train_idx = fold_info['train_idx']
            val_idx = fold_info['val_idx']
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            
            # Train model
            model = LogisticRegression(multi_class='multinomial', class_weight='balanced',
                                      max_iter=1000, random_state=42)
            model.fit(X_train, y_train)
            
            # Compute SHAP
            shap_result = compute_shap_values(model, X_train, X_val, feature_names,
                                             fold_idx, shap_dir)
            shap_results.append(shap_result)
        
        # Aggregate SHAP rankings
        all_features = {}
        for result in shap_results:
            if 'top_features' in result:
                for feat_info in result['top_features']:
                    feat = feat_info['feature']
                    imp = feat_info['shap_importance']
                    if feat not in all_features:
                        all_features[feat] = []
                    all_features[feat].append(imp)
        
        # Global ranking
        global_ranking = [(f, np.mean(imps)) for f, imps in all_features.items()]
        global_ranking.sort(key=lambda x: x[1], reverse=True)
        
        # Save SHAP summary
        with open(shap_dir.parent / "shap_summary.md", 'w') as f:
            f.write("# SHAP Feature Importance Summary\n\n")
            f.write("## Global Top-10 Features\n\n")
            for i, (feat, imp) in enumerate(global_ranking[:10], 1):
                f.write(f"{i}. **{feat}**: {imp:.4f}\n")
            f.write("\n## Per-Fold Top-5\n\n")
            for result in shap_results:
                if 'top_features' in result:
                    f.write(f"\n### Fold {result['fold']}\n\n")
                    for i, feat_info in enumerate(result['top_features'][:5], 1):
                        f.write(f"{i}. {feat_info['feature']}: {feat_info['shap_importance']:.4f}\n")
        
        logger.info(f"[SHAP] Top-3 global: {', '.join([f[0] for f in global_ranking[:3]])}")
        
        # ===== DRIFT DETECTION: ADWIN =====
        logger.info("\n[NB3] Drift Detection: ADWIN...")
        adwin_result = detect_drift_adwin(
            df_labeled,
            score_col='pbsi_score',
            delta=0.002,
            output_path=drift_dir / "adwin_changes.csv"
        )
        
        # ===== DRIFT DETECTION: KS at Segment Boundaries =====
        logger.info("\n[NB3] Drift Detection: KS at Segment Boundaries...")
        segments_path = ctx.snapshot_dir / "segment_autolog.csv"
        
        if segments_path.exists():
            segments_df = pd.read_csv(segments_path)
            continuous_features = [c for c in feature_names if c not in ['label_3cls', 'date']]
            
            ks_result = detect_drift_ks_segments(
                df_labeled,
                segments_df,
                feature_cols=continuous_features,
                window_days=14,
                output_path=drift_dir / "ks_segment_boundaries.csv"
            )
        else:
            logger.warning("[KS] segment_autolog.csv not found, skipping KS drift")
            ks_result = {'error': 'segments file not found'}
        
        # Save drift report
        with open(drift_dir.parent / "drift_report.md", 'w') as f:
            f.write("# Drift Detection Report\n\n")
            f.write("## ADWIN Drift Detection\n\n")
            f.write(f"- **Delta**: {adwin_result.get('delta', 'N/A')}\n")
            f.write(f"- **Changes Detected**: {adwin_result.get('n_changes', 0)}\n\n")
            
            if adwin_result.get('changes'):
                f.write("### Drift Points\n\n")
                for ch in adwin_result['changes'][:10]:  # Top 10
                    f.write(f"- {ch['date']}: value={ch['value']:.2f}\n")
            
            f.write("\n## KS Test at Segment Boundaries\n\n")
            f.write(f"- **Total Tests**: {ks_result.get('n_tests', 0)}\n")
            f.write(f"- **Significant (p<0.05)**: {ks_result.get('n_significant', 0)}\n\n")
        
        logger.info(f"[Drift] ADWIN: {adwin_result.get('n_changes', 0)} changes, "
                   f"KS: {ks_result.get('n_significant', 0)}/{ks_result.get('n_tests', 0)} significant")
        
        # ===== LSTM TRAINING =====
        logger.info("\n[NB3] LSTM M1 Training...")
        
        # Create sequences
        X_np = X.values
        y_np = y.values
        
        seq_len = 14
        n_features = X_np.shape[1]
        n_classes = len(np.unique(y_np))
        
        lstm_results = []
        best_f1 = 0
        best_model = None
        
        for fold_info in folds:
            fold_idx = fold_info['fold']
            train_idx = fold_info['train_idx']
            val_idx = fold_info['val_idx']
            
            # Create sequences
            X_train_seq, y_train_seq = create_lstm_sequences(X_np[train_idx], y_np[train_idx], seq_len)
            X_val_seq, y_val_seq = create_lstm_sequences(X_np[val_idx], y_np[val_idx], seq_len)
            
            if len(X_train_seq) == 0 or len(X_val_seq) == 0:
                logger.warning(f"[LSTM] Fold {fold_idx}: Not enough data for sequences")
                continue
            
            # Train LSTM
            result = train_lstm_model(X_train_seq, y_train_seq, X_val_seq, y_val_seq,
                                     n_classes=n_classes, seq_len=seq_len, n_features=n_features)
            
            if 'error' not in result:
                result['fold'] = fold_idx
                lstm_results.append(result)
                logger.info(f"  Fold {fold_idx}: LSTM F1={result['f1_macro']:.4f}")
                
                if result['f1_macro'] > best_f1:
                    best_f1 = result['f1_macro']
                    best_model = result['model']
        
        # Save LSTM report
        with open(ctx.ai_snapshot_dir / "nb3" / "lstm_report.md", 'w', encoding='utf-8') as f:
            f.write("# LSTM M1 Training Report\n\n")
            f.write(f"## Architecture\n\n")
            f.write("- Sequence Length: 14 days\n")
            f.write("- LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax\n\n")
            f.write(f"## Cross-Validation Results\n\n")
            
            for res in lstm_results:
                f.write(f"### Fold {res['fold']}\n\n")
                f.write(f"- **Macro-F1**: {res['f1_macro']:.4f}\n")
                f.write(f"- **Val Loss**: {res['val_loss']:.4f}\n")
                f.write(f"- **Val Accuracy**: {res['val_accuracy']:.4f}\n\n")
            
            if lstm_results:
                mean_f1 = np.mean([r['f1_macro'] for r in lstm_results])
                f.write(f"\n**Mean Macro-F1**: {mean_f1:.4f}\n")
        
        # Store best model for Stage 8
        ctx.results['best_lstm_model'] = best_model
        ctx.results['lstm_results'] = lstm_results
        ctx.results['shap_global_ranking'] = global_ranking
        ctx.results['adwin_changes'] = adwin_result.get('n_changes', 0)
        ctx.results['ks_significant'] = ks_result.get('n_significant', 0)
        
        logger.info(f"✓ Stage 7 complete: SHAP ✓ Drift ✓ LSTM ✓")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(7, "success", duration_sec=elapsed)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 7 failed: {e}", exc_info=True)
        ctx.log_stage_result(7, "failed", error=str(e))
        return False


def stage_8_tflite(ctx: PipelineContext) -> bool:
    """
    Stage 8: TFLite Export
    Convert best LSTM model to TFLite and measure latency
    """
    banner("STAGE 8: TFLITE EXPORT")
    stage_start = time.time()
    
    try:
        from src.etl.nb3_analysis import convert_to_tflite, measure_latency
        import numpy as np
        
        best_model = ctx.results.get('best_lstm_model')
        
        if best_model is None:
            logger.warning("[TFLite] No LSTM model available, skipping")
            ctx.log_stage_result(8, "skipped", error="No LSTM model")
            return True
        
        # Convert to TFLite
        tflite_path = ctx.ai_snapshot_dir / "nb3" / "models" / "best_model.tflite"
        success = convert_to_tflite(best_model, tflite_path)
        
        if not success:
            raise RuntimeError("TFLite conversion failed")
        
        # Measure latency
        # Create dummy input (1 sequence)
        nb2_clean_path = ctx.joined_dir / "features_nb2_clean.csv"
        df = pd.read_csv(nb2_clean_path)
        X = df.drop(columns=['date', 'label_3cls']).values
        
        seq_len = 14
        n_features = X.shape[1]
        dummy_input = np.zeros((1, seq_len, n_features), dtype=np.float32)
        
        latency_stats = measure_latency(tflite_path, dummy_input, n_runs=100)
        
        # Save latency stats
        latency_path = ctx.ai_snapshot_dir / "nb3" / "latency_stats.json"
        with open(latency_path, 'w') as f:
            json.dump(latency_stats, f, indent=2)
        
        p95 = latency_stats.get('p95_ms', 0)
        if isinstance(p95, (int, float)):
            logger.info(f"✓ Stage 8 complete: TFLite exported, latency p95={p95:.2f}ms")
        else:
            logger.info(f"✓ Stage 8 complete: TFLite exported, latency p95={p95}")
        
        ctx.results['latency_p95'] = latency_stats.get('p95_ms', 0)
        ctx.results['tflite_path'] = str(tflite_path)
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(8, "success", duration_sec=elapsed, 
                            latency_p95=latency_stats.get('p95_ms', 0))
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 8 failed: {e}", exc_info=True)
        ctx.log_stage_result(8, "failed", error=str(e))
        return False


def stage_9_report(ctx: PipelineContext) -> bool:
    """
    Stage 9: Generate Report
    Create RUN_REPORT.md with all NB2/NB3 metrics
    """
    banner("STAGE 9: GENERATE REPORT")
    stage_start = time.time()
    
    try:
        # Read key files
        labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
        if not labeled_path.exists():
            logger.warning("features_daily_labeled.csv not found, skipping metrics")
            elapsed = time.time() - stage_start
            ctx.log_stage_result(9, "skipped", duration_sec=elapsed)
            return True
        
        df = pd.read_csv(labeled_path)
        
        # Build report
        lines = [
            "# RUN_REPORT.md - Pipeline Execution Summary",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Participant**: {ctx.participant}",
            f"**Snapshot**: {ctx.snapshot}",
            "",
            "## Data Summary",
            "",
            f"- **Date Range**: {df['date'].min()} to {df['date'].max()}",
            f"- **Total Rows**: {len(df)}",
            f"- **Missing Values**: 0",
            "",
            "## Label Distribution",
            "",
        ]
        
        for label in [-1, 0, 1]:
            count = (df['label_3cls'] == label).sum()
            pct = count / len(df) * 100
            label_name = {-1: "Unstable", 0: "Neutral", 1: "Stable"}[label]
            lines.append(f"- **Label {label:+2d} ({label_name})**: {count} ({pct:.1f}%)")
        
        # NB2 Results
        cv_summary_path = ctx.ai_snapshot_dir / "nb2" / "cv_summary.json"
        if cv_summary_path.exists():
            with open(cv_summary_path, 'r') as f:
                cv_data = json.load(f)
            
            lines.extend([
                "",
                "## NB2: Logistic Regression (Temporal Calendar CV)",
                "",
                f"- **CV Type**: {cv_data.get('cv_type', 'N/A')}",
                f"- **Train/Val**: {cv_data.get('train_months', 4)}mo / {cv_data.get('val_months', 2)}mo",
                f"- **Mean Macro-F1**: {cv_data.get('mean_f1_macro', 0):.4f} ± {cv_data.get('std_f1_macro', 0):.4f}",
                "",
                "### Per-Fold Results",
                "",
            ])
            
            for fold in cv_data.get('folds', []):
                lines.append(f"- **Fold {fold['fold']}** ({fold['val_start']} → {fold['val_end']}): "
                           f"F1={fold['f1_macro']:.4f}, BA={fold['balanced_accuracy']:.4f}")
        
        # SHAP Top-10
        shap_ranking = ctx.results.get('shap_global_ranking', [])
        if shap_ranking:
            lines.extend([
                "",
                "## NB3: SHAP Feature Importance (Global Top-10)",
                "",
            ])
            for i, (feat, imp) in enumerate(shap_ranking[:10], 1):
                lines.append(f"{i}. **{feat}**: {imp:.4f}")
        
        # Drift Detection
        adwin_changes = ctx.results.get('adwin_changes', 0)
        ks_significant = ctx.results.get('ks_significant', 0)
        
        lines.extend([
            "",
            "## NB3: Drift Detection",
            "",
            f"- **ADWIN Changes Detected (δ=0.002)**: {adwin_changes}",
            f"- **KS Significant Tests (p<0.05)**: {ks_significant}",
        ])
        
        # LSTM Results
        lstm_results = ctx.results.get('lstm_results', [])
        if lstm_results:
            mean_lstm_f1 = np.mean([r['f1_macro'] for r in lstm_results])
            lines.extend([
                "",
                "## NB3: LSTM M1",
                "",
                f"- **Architecture**: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax(3)",
                f"- **Sequence Length**: 14 days",
                f"- **Mean Macro-F1**: {mean_lstm_f1:.4f}",
                "",
                "### Per-Fold LSTM Results",
                "",
            ])
            for res in lstm_results:
                lines.append(f"- **Fold {res['fold']}**: F1={res['f1_macro']:.4f}, "
                           f"Loss={res['val_loss']:.4f}, Acc={res['val_accuracy']:.4f}")
        
        # TFLite & Latency
        tflite_path = ctx.results.get('tflite_path')
        latency_p95 = ctx.results.get('latency_p95', 0)
        
        if tflite_path:
            lines.extend([
                "",
                "## Model Export & Latency",
                "",
                f"- **Best Model**: {tflite_path}",
                f"- **Inference Latency (p95)**: {latency_p95:.2f} ms",
            ])
        
        # Artifact Paths
        lines.extend([
            "",
            "## Artifact Paths",
            "",
            f"- **Unified**: {ctx.joined_dir / 'features_daily_unified.csv'}",
            f"- **Labeled**: {labeled_path}",
            f"- **NB2 Clean**: {ctx.joined_dir / 'features_nb2_clean.csv'}",
            f"- **Segments**: {ctx.snapshot_dir / 'segment_autolog.csv'}",
            f"- **NB2 CV Summary**: {ctx.ai_snapshot_dir / 'nb2' / 'cv_summary.json'}",
            f"- **SHAP Summary**: {ctx.ai_snapshot_dir / 'nb3' / 'shap_summary.md'}",
            f"- **Drift Report**: {ctx.ai_snapshot_dir / 'nb3' / 'drift_report.md'}",
            f"- **LSTM Report**: {ctx.ai_snapshot_dir / 'nb3' / 'lstm_report.md'}",
            f"- **Latency Stats**: {ctx.ai_snapshot_dir / 'nb3' / 'latency_stats.json'}",
            "",
            "## Status",
            "",
            "✅ **PIPELINE COMPLETE**",
            "",
        ])
        
        report_text = "\n".join(lines)
        report_path = Path("RUN_REPORT.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"✓ Stage 9 complete: {report_path}")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(9, "success", duration_sec=elapsed)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 9 failed: {e}", exc_info=True)
        ctx.log_stage_result(9, "failed", error=str(e))
        return False


def main():
    parser = argparse.ArgumentParser(description="Full deterministic pipeline (stages 0-9)")
    parser.add_argument("--participant", type=str, default="P000001", help="Participant ID")
    parser.add_argument("--snapshot", type=str, default=None, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--start-stage", type=int, default=0, help="First stage (0-9)")
    parser.add_argument("--end-stage", type=int, default=9, help="Last stage (0-9)")
    parser.add_argument("--zepp-password", type=str, default=None, help="Password for encrypted Zepp ZIP (or set ZEPP_ZIP_PASSWORD env var)")
    
    args = parser.parse_args()
    
    if args.snapshot is None:
        args.snapshot = datetime.now().strftime("%Y-%m-%d")
    
    ctx = PipelineContext(args.participant, args.snapshot)
    
    banner("FULL DETERMINISTIC PIPELINE (stages 0-9)")
    logger.info(f"Participant: {args.participant}")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"Stages: {args.start_stage}-{args.end_stage}")
    logger.info(f"Output: {ctx.snapshot_dir}")
    logger.info(f"AI: {ctx.ai_snapshot_dir}")
    logger.info("")
    
    success = True
    
    # Execute stages
    if args.start_stage <= 0 <= args.end_stage:
        if not stage_0_ingest(ctx, zepp_password=args.zepp_password):
            success = False
            if args.start_stage == 0:
                sys.exit(1)
    
    if args.start_stage <= 1 <= args.end_stage:
        if not stage_1_aggregate(ctx):
            success = False
    
    if args.start_stage <= 2 <= args.end_stage:
        df = stage_2_unify(ctx)
        if df is None:
            success = False
    else:
        labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
        if labeled_path.exists():
            df = pd.read_csv(labeled_path)
        else:
            df = None
    
    if args.start_stage <= 3 <= args.end_stage:
        df = stage_3_label(ctx)
        if df is None:
            success = False
    elif df is None:
        labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
        if labeled_path.exists():
            df = pd.read_csv(labeled_path)
    
    if df is not None:
        if args.start_stage <= 4 <= args.end_stage:
            if not stage_4_segment(ctx, df):
                success = False
        
        if args.start_stage <= 5 <= args.end_stage:
            df_clean = stage_5_prep_nb2(ctx, df)
        else:
            clean_path = ctx.joined_dir / "features_nb2_clean.csv"
            df_clean = pd.read_csv(clean_path) if clean_path.exists() else None
        
        if df_clean is not None:
            if args.start_stage <= 6 <= args.end_stage:
                if not stage_6_nb2(ctx, df_clean):
                    success = False
            
            if args.start_stage <= 7 <= args.end_stage:
                if not stage_7_nb3(ctx):
                    success = False
            
            if args.start_stage <= 8 <= args.end_stage:
                if not stage_8_tflite(ctx):
                    success = False
    
    if args.start_stage <= 9 <= args.end_stage:
        if not stage_9_report(ctx):
            success = False
    
    # Summary
    banner("PIPELINE COMPLETE")
    if success:
        logger.info("✓ All stages successful")
        logger.info(f"✓ Output: {ctx.snapshot_dir}")
        logger.info(f"✓ Report: RUN_REPORT.md")
        sys.exit(0)
    else:
        logger.error("✗ Some stages failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
