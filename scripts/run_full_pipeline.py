#!/usr/bin/env python
"""
FULL DETERMINISTIC PIPELINE
Stages 0-9: Ingest → Aggregate → Unify → Label → Segment → Prep-ML6 → ML6 → ML7 → TFLite → Report

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
from tqdm import tqdm

try:
    import pyzipper
    HAS_PYZIPPER = True
except ImportError:
    HAS_PYZIPPER = False


def _should_show_progress() -> bool:
    """Determine if progress bars should be shown (Git Bash compatible)."""
    if os.getenv("ETL_TQDM") == "1":
        return True
    if os.getenv("ETL_TQDM") == "0":
        return False
    if os.getenv("CI"):
        return False
    try:
        if hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            return True
    except Exception:
        pass
    # Git Bash / MSYS2 detection
    if os.getenv("MSYSTEM") or os.getenv("TERM"):
        return True
    return False


def _make_pbar(total, desc, unit="files"):
    """Create a tqdm progress bar with Git Bash compatible settings."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=False,
        dynamic_ncols=True,
        disable=not _should_show_progress()
    )


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


def train_lstm_model_cfg3(
    X_train, y_train, X_val, y_val,
    n_classes: int, seq_len: int, n_features: int,
    lstm_units: int = 32, dense_units: int = 32, dropout: float = 0.4,
    use_early_stopping: bool = True, early_stopping_patience: int = 3,
    use_class_weight: bool = True, epochs: int = 50, batch_size: int = 16
) -> dict:
    """
    Train LSTM model with CFG-3 configuration (regularized).
    
    This is the optimized configuration from ML7 ablation study:
    - Higher dropout (0.4) for regularization
    - Early stopping to prevent overfitting
    - Class weights to handle imbalance
    
    Args:
        X_train, y_train: Training sequences
        X_val, y_val: Validation sequences
        n_classes: Number of classes
        seq_len: Sequence length
        n_features: Number of input features
        lstm_units: LSTM units
        dense_units: Dense layer units
        dropout: Dropout rate
        use_early_stopping: Whether to use early stopping
        early_stopping_patience: Patience for early stopping
        use_class_weight: Whether to use class weights
        epochs: Max training epochs
        batch_size: Batch size
    
    Returns:
        Dict with model, history, metrics
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from sklearn.metrics import f1_score, balanced_accuracy_score
        from sklearn.utils.class_weight import compute_class_weight
        
        # Map labels to 0-indexed for Keras
        unique_labels = sorted(set(y_train.tolist() + y_val.tolist()))
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        reverse_map = {idx: label for label, idx in label_map.items()}
        
        y_train_mapped = np.array([label_map[y] for y in y_train])
        y_val_mapped = np.array([label_map[y] for y in y_val])
        
        # Build model
        model = keras.Sequential([
            layers.LSTM(lstm_units, input_shape=(seq_len, n_features), return_sequences=False),
            layers.Dense(dense_units, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(n_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ))
        
        # Class weights
        class_weight = None
        if use_class_weight:
            weights = compute_class_weight(
                'balanced',
                classes=np.unique(y_train_mapped),
                y=y_train_mapped
            )
            class_weight = dict(enumerate(weights))
        
        # Train
        history = model.fit(
            X_train, y_train_mapped,
            validation_data=(X_val, y_val_mapped),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_val, verbose=0)
        y_pred_mapped = np.argmax(y_pred_proba, axis=1)
        y_pred = np.array([reverse_map[y] for y in y_pred_mapped])
        
        # Metrics
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(y_val, y_pred)
        
        # Training stats
        n_epochs_trained = len(history.history['loss'])
        early_stopped = n_epochs_trained < epochs if use_early_stopping else False
        
        return {
            'model': model,
            'history': history.history,
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'balanced_accuracy': float(balanced_acc),
            'val_loss': float(history.history['val_loss'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'n_epochs_trained': n_epochs_trained,
            'early_stopped': early_stopped,
            'config': {
                'lstm_units': lstm_units,
                'dense_units': dense_units,
                'dropout': dropout,
                'use_early_stopping': use_early_stopping,
                'use_class_weight': use_class_weight,
                'epochs': epochs,
                'batch_size': batch_size
            }
        }
        
    except ImportError:
        logger.warning("[LSTM] TensorFlow not available")
        return {'error': 'tensorflow not installed'}
    except Exception as e:
        logger.error(f"[LSTM] Training error: {e}")
        return {'error': str(e)}


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
                  self.ai_snapshot_dir / "ml6",
                  self.ai_snapshot_dir / "ml7" / "models"]:
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


def _parse_autoexport_zip_date(zip_path: Path) -> Optional[datetime]:
    """Extract date from AutoExport ZIP filename.
    
    Filename patterns:
    - HealthAutoExport_YYYYMMDDHHMMSS.zip (e.g., HealthAutoExport_20251208005855.zip)
    - HealthAutoExport-YYYY-MM-DD.zip
    
    Returns:
        datetime object or None if not parseable
    """
    import re
    name = zip_path.stem
    
    # Pattern 1: HealthAutoExport_YYYYMMDDHHMMSS
    match = re.search(r'(\d{14})$', name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
        except ValueError:
            pass
    
    # Pattern 2: HealthAutoExport_YYYYMMDD (just date)
    match = re.search(r'(\d{8})$', name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            pass
    
    # Pattern 3: YYYY-MM-DD in filename
    match = re.search(r'(\d{4}-\d{2}-\d{2})', name)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y-%m-%d")
        except ValueError:
            pass
    
    return None


def _select_autoexport_zip_for_snapshot(
    zip_files: list, 
    snapshot: str
) -> Optional[Path]:
    """Select AutoExport ZIP deterministically based on snapshot date.
    
    RULE: Select the ZIP with filename date <= snapshot date.
          If multiple match, choose the one with the most recent date.
          
    This ensures reproducible, deterministic ETL runs.
    
    Args:
        zip_files: List of Path objects to ZIP files
        snapshot: Snapshot date as YYYY-MM-DD string
        
    Returns:
        Path to selected ZIP, or None if no valid ZIP found
    """
    try:
        snapshot_dt = datetime.strptime(snapshot, "%Y-%m-%d")
        # Set to end of day for inclusive comparison
        snapshot_dt = snapshot_dt.replace(hour=23, minute=59, second=59)
    except ValueError:
        logger.error(f"[AutoExport] Invalid snapshot format: {snapshot}")
        return None
    
    # Parse dates from all ZIPs and log for traceability
    candidates = []
    logger.info(f"[AutoExport] Snapshot date: {snapshot}")
    logger.info(f"[AutoExport] Available ZIP files:")
    
    for zp in sorted(zip_files, key=lambda p: p.name):
        zip_date = _parse_autoexport_zip_date(zp)
        if zip_date:
            date_str = zip_date.strftime("%Y-%m-%d %H:%M:%S")
            is_valid = zip_date <= snapshot_dt
            status = "✓ VALID" if is_valid else "✗ FUTURE"
            logger.info(f"  - {zp.name} → date={date_str} [{status}]")
            
            if is_valid:
                candidates.append((zp, zip_date))
        else:
            logger.warning(f"  - {zp.name} → date=UNPARSEABLE [✗ SKIPPED]")
    
    if not candidates:
        logger.warning(f"[AutoExport] No ZIP with date <= {snapshot} found.")
        return None
    
    # Select the most recent valid ZIP
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[0][0]
    selected_date = candidates[0][1].strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"[AutoExport] Selected: {selected.name} (date={selected_date})")
    return selected


def _select_zepp_zip_for_snapshot(
    zip_files: list, 
    snapshot: str
) -> Optional[Path]:
    """Select Zepp ZIP deterministically based on snapshot date.
    
    RULE: Select the ZIP with filesystem mtime <= snapshot date.
          If multiple match, choose the one with the most recent mtime.
          
    Uses filesystem modification time (mtime) for date comparison,
    as Zepp ZIP filenames use opaque numeric IDs.
    
    Args:
        zip_files: List of Path objects to ZIP files
        snapshot: Snapshot date as YYYY-MM-DD string
        
    Returns:
        Path to selected ZIP, or None if no valid ZIP found
    """
    try:
        snapshot_dt = datetime.strptime(snapshot, "%Y-%m-%d")
        # Set to end of day for inclusive comparison
        snapshot_dt = snapshot_dt.replace(hour=23, minute=59, second=59)
    except ValueError:
        logger.error(f"[Zepp] Invalid snapshot format: {snapshot}")
        return None
    
    # Get mtime for all ZIPs and log for traceability
    candidates = []
    logger.info(f"[Zepp] Snapshot date: {snapshot}")
    logger.info(f"[Zepp] Available ZIP files:")
    
    for zp in sorted(zip_files, key=lambda p: p.name):
        try:
            mtime = datetime.fromtimestamp(zp.stat().st_mtime)
            date_str = mtime.strftime("%Y-%m-%d %H:%M:%S")
            is_valid = mtime <= snapshot_dt
            status = "✓ VALID" if is_valid else "✗ FUTURE"
            logger.info(f"  - {zp.name} → mtime={date_str} [{status}]")
            
            if is_valid:
                candidates.append((zp, mtime))
        except OSError as e:
            logger.warning(f"  - {zp.name} → mtime=ERROR [{e}]")
    
    if not candidates:
        logger.warning(f"[Zepp] No ZIP with mtime <= {snapshot} found.")
        return None
    
    # Select the most recent valid ZIP
    candidates.sort(key=lambda x: x[1], reverse=True)
    selected = candidates[0][0]
    selected_date = candidates[0][1].strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"[Zepp] Selected: {selected.name} (mtime={selected_date})")
    return selected


def stage_0_ingest(ctx: PipelineContext, zepp_password: Optional[str] = None) -> bool:
    """
    Stage 0: Ingest
    Extract ZIPs from data/raw/P000001/{apple,zepp}/ to data/etl/.../extracted/
    
    Zepp extraction is OPTIONAL:
    - If Zepp ZIP exists but no password provided: Skip with warning (non-fatal).
    - ML6/ML7 pipelines can run without Zepp data.
    """
    banner("STAGE 0: INGEST (extract from data/raw)")
    stage_start = time.time()
    
    try:
        raw_participant_dir = ctx.raw_dir / ctx.participant
        
        # Get Zepp password from env or parameter
        zpwd = zepp_password or os.getenv("ZEP_ZIP_PASSWORD") or os.getenv("ZEPP_ZIP_PASSWORD")
        
        # NON-FATAL: Warn if Zepp ZIPs exist but no password provided
        zepp_raw_dir = raw_participant_dir / "zepp"
        zepp_skip_warned = False
        if zepp_raw_dir.exists():
            zepp_zips = list(zepp_raw_dir.glob("*.zip"))
            if zepp_zips and not zpwd:
                logger.warning("[WARN] Zepp ZIP detected but no password provided. Skipping Zepp extraction.")
                logger.warning("[WARN] Pipeline will run in Apple-only mode. ML6/ML7 models remain reproducible.")
                zepp_skip_warned = True
        
        # Extract Apple ZIPs
        apple_raw_dir = raw_participant_dir / "apple" / "export"
        if apple_raw_dir.exists():
            apple_zips = list(apple_raw_dir.glob("*.zip"))
            for zip_file in apple_zips:
                logger.info(f"[Apple] Extracting: {zip_file.name}")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    members = z.namelist()
                    with _make_pbar(len(members), f"[Apple] {zip_file.name}") as pbar:
                        for member in members:
                            z.extract(member, ctx.extracted_dir / "apple")
                            pbar.update(1)
            logger.info(f"[OK] Apple extracted to {ctx.extracted_dir / 'apple'}")
        else:
            logger.warning(f"[SKIP] Apple export dir not found: {apple_raw_dir}")
        
        # Extract Apple Auto Export ZIPs (deterministic: select ZIP with date <= snapshot)
        autoexport_raw_dir = raw_participant_dir / "apple" / "autoexport"
        if autoexport_raw_dir.exists():
            autoexport_zips = list(autoexport_raw_dir.glob("*.zip"))
            if autoexport_zips:
                # Deterministic selection: find ZIP with filename date <= snapshot
                selected_zip = _select_autoexport_zip_for_snapshot(autoexport_zips, ctx.snapshot)
                if selected_zip:
                    logger.info(f"[AutoExport] Extracting: {selected_zip.name}")
                    autoexport_target = ctx.extracted_dir / "apple" / "autoexport"
                    autoexport_target.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(selected_zip, 'r') as z:
                        members = z.namelist()
                        with _make_pbar(len(members), f"[AutoExport] {selected_zip.name}") as pbar:
                            for member in members:
                                z.extract(member, autoexport_target)
                                pbar.update(1)
                    logger.info(f"[OK] AutoExport extracted to {autoexport_target}")
                else:
                    logger.warning(f"[SKIP] No AutoExport ZIP found with date <= {ctx.snapshot}")
            else:
                logger.info(f"[SKIP] No AutoExport ZIPs found in {autoexport_raw_dir}")
        else:
            logger.info(f"[SKIP] AutoExport dir not found: {autoexport_raw_dir}")
        
        # Extract Zepp ZIPs (deterministic: select ZIP with mtime <= snapshot)
        if zepp_raw_dir.exists() and not zepp_skip_warned:
            zepp_zips = list(zepp_raw_dir.glob("*.zip"))
            if zepp_zips:
                # Deterministic selection: find ZIP with mtime <= snapshot
                selected_zip = _select_zepp_zip_for_snapshot(zepp_zips, ctx.snapshot)
                if selected_zip:
                    logger.info(f"[Zepp] Extracting: {selected_zip.name}")
                    try:
                        # Try with pyzipper first (handles AES encryption)
                        if HAS_PYZIPPER:
                            try:
                                with pyzipper.AESZipFile(selected_zip, 'r') as z:
                                    members = z.namelist()
                                    with _make_pbar(len(members), f"[Zepp] {selected_zip.name}") as pbar:
                                        for member in members:
                                            if zpwd:
                                                z.extract(member, ctx.extracted_dir / "zepp", pwd=zpwd.encode('utf-8'))
                                            else:
                                                z.extract(member, ctx.extracted_dir / "zepp")
                                            pbar.update(1)
                            except Exception as e:
                                # Fallback to regular zipfile
                                logger.warning(f"[Zepp] AES extraction failed, trying standard ZIP: {e}")
                                with zipfile.ZipFile(selected_zip, 'r') as z:
                                    members = z.namelist()
                                    with _make_pbar(len(members), f"[Zepp] {selected_zip.name}") as pbar:
                                        for member in members:
                                            z.extract(member, ctx.extracted_dir / "zepp")
                                            pbar.update(1)
                        else:
                            # No pyzipper, try regular zipfile
                            with zipfile.ZipFile(selected_zip, 'r') as z:
                                z.extractall(ctx.extracted_dir / "zepp")
                        logger.info(f"[OK] Zepp extracted to {ctx.extracted_dir / 'zepp'}")
                    except RuntimeError as e:
                        if "encrypted" in str(e).lower() or "password" in str(e).lower():
                            logger.warning(f"[SKIP] {selected_zip.name} is encrypted (no valid password)")
                        else:
                            raise
                    except Exception as e:
                        logger.warning(f"[SKIP] {selected_zip.name}: {e}")
                else:
                    logger.warning(f"[SKIP] No Zepp ZIP found with mtime <= {ctx.snapshot}")
            else:
                logger.info(f"[SKIP] No Zepp ZIPs found in {zepp_raw_dir}")
        elif zepp_raw_dir.exists() and zepp_skip_warned:
            logger.info(f"[SKIP] Zepp extraction skipped (no password)")
        else:
            logger.warning(f"[SKIP] Zepp dir not found: {zepp_raw_dir}")
        
        elapsed = time.time() - stage_start
        logger.info(f"[OK] Stage 0 complete ({elapsed:.1f}s)")
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
    Also aggregates meds and SoM from AutoExport CSVs.
    """
    banner("STAGE 1: AGGREGATE (xml + csvs to daily_*.csv)")
    stage_start = time.time()
    
    try:
        logger.info(f"Aggregating from {ctx.extracted_dir}")
        
        # Pass the extracted dir (where export.xml should be)
        # Include snapshot for AutoExport filtering
        results = run_csv_aggregation(
            participant=ctx.participant,
            extracted_dir=str(ctx.extracted_dir),
            output_dir=str(ctx.extracted_dir),
            snapshot=ctx.snapshot
        )
        
        apple_total = sum(len(df) for df in results.get("apple", {}).values())
        zepp_total = sum(len(df) for df in results.get("zepp", {}).values())
        autoexport_total = sum(len(df) for df in results.get("apple_autoexport", {}).values())
        
        logger.info(f"[OK] Stage 1 complete: {apple_total} Apple + {zepp_total} Zepp + {autoexport_total} AutoExport days")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(1, "success", duration_sec=elapsed, 
                            apple_days=apple_total, zepp_days=zepp_total,
                            autoexport_days=autoexport_total)
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
        
        logger.info(f"[OK] Stage 2 complete: {len(df)} days, "
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
        logger.info(f"[OK] Stage 3 complete: {len(df)} days labeled")
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
        
        logger.info(f"[OK] Stage 4 complete: {len(df_segments)} segments")
        logger.info(f"  Reasons: {df_segments['reason'].value_counts().to_dict()}")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(4, "success", duration_sec=elapsed, segments=len(df_segments))
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 4 failed: {e}", exc_info=True)
        ctx.log_stage_result(4, "failed", error=str(e))
        return False


def stage_5_prep_ml6(ctx: PipelineContext, df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Stage 5: ML Data Preparation - SoM-Centric (Composite Stage)
    
    This stage prepares the Feature Universe for ML6/ML7 training using
    State of Mind (SoM) as the primary target.
    
    SUB-STAGES:
    -----------
    5.1 TEMPORAL FILTER
        - Filter to ML-ready period (>= 2021-05-11, Amazfit GTR 2 era)
        - Keep only days with valid SoM labels (som_vendor == 'apple_autoexport')
    
    5.2 ANTI-LEAK REMOVAL
        - Remove columns that would leak target information
        - Keep pbsi_score as candidate feature (not in anti-leak list)
    
    5.3 MICE IMPUTATION
        - Segment-aware iterative imputation on feature columns
        - Targets (som_*) are NOT imputed
    
    5.4 FEATURE UNIVERSE
        - Build complete imputed feature matrix with all candidates
        - Save as features_daily_ml_universe.csv
    
    5.5 EXPERIMENT SUITE
        - Run ML6/ML7 ablation to select best feature set & config
        - Save model_selection.json artifact
        - Export selected features as features_daily_ml6.csv
    
    OUTPUTS:
    --------
    - ai/local/<PID>/<SNAPSHOT>/ml6/features_daily_ml_universe.csv
    - ai/local/<PID>/<SNAPSHOT>/ml6/features_daily_ml6.csv
    - ai/local/<PID>/<SNAPSHOT>/model_selection.json
    
    Based on prior ablation studies:
    - FS-B (Baseline + HRV) is typically best for SoM prediction
    - Binary target (som_binary) outperforms 3-class
    - CFG-3 (regularized LSTM) works best for sequences
    """
    banner("STAGE 5: ML DATA PREPARATION (SoM-Centric, Composite)")
    stage_start = time.time()
    
    try:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        import numpy as np
        
        df_work = df.copy()
        df_work['date'] = pd.to_datetime(df_work['date'])
        
        # =================================================================
        # FEATURE UNIVERSE DEFINITION
        # =================================================================
        # All candidate features for ablation experiments
        FEATURE_UNIVERSE_COLS = [
            # Sleep (2)
            'sleep_hours', 'sleep_quality_score',
            # Cardio (5)
            'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
            # HRV (5)
            'hrv_sdnn_mean', 'hrv_sdnn_median', 'hrv_sdnn_min', 'hrv_sdnn_max', 'n_hrv_sdnn',
            # Activity (3)
            'total_steps', 'total_distance', 'total_active_energy',
            # MEDS (3) - candidate features
            'med_any', 'med_event_count', 'med_dose_total',
            # PBSI (1) - candidate feature (NOT target)
            'pbsi_score',
        ]
        
        # Anti-leak columns: MUST be removed before ML
        ANTI_LEAK_COLS = [
            # PBSI intermediate outputs (encode target derivation)
            'pbsi_quality', 'sleep_sub', 'cardio_sub', 'activity_sub',
            # PBSI labels (old targets - would leak)
            'label_3cls', 'label_2cls', 'label_clinical',
        ]
        # Note: pbsi_score is NOT in anti-leak - it's a candidate feature
        
        # Default feature set (FS-B) for fallback
        FS_B_FEATURE_COLS = [
            'sleep_hours', 'sleep_quality_score',
            'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_samples',
            'hrv_sdnn_mean', 'hrv_sdnn_median', 'hrv_sdnn_min', 'hrv_sdnn_max', 'n_hrv_sdnn',
            'total_steps', 'total_distance', 'total_active_energy',
        ]
        
        # =================================================================
        # 5.1 TEMPORAL FILTER
        # =================================================================
        logger.info("=" * 60)
        logger.info("5.1 TEMPORAL FILTER")
        logger.info("=" * 60)
        
        ml_cutoff = pd.Timestamp('2021-05-11')
        n_before = len(df_work)
        df_work = df_work[df_work['date'] >= ml_cutoff].copy()
        n_after_temporal = len(df_work)
        n_excluded_temporal = n_before - n_after_temporal
        
        logger.info(f"[5.1] ML cutoff: >= {ml_cutoff.strftime('%Y-%m-%d')}")
        logger.info(f"  Excluded: {n_excluded_temporal} days (pre-Amazfit era)")
        logger.info(f"  Retained: {n_after_temporal} days")
        
        # SoM validity filter
        if 'som_category_3class' not in df_work.columns:
            logger.error("[5.1] som_category_3class column not found!")
            raise ValueError("som_category_3class column required for ML")
        
        som_valid_mask = df_work['som_category_3class'].notna()
        if 'som_vendor' in df_work.columns:
            som_valid_mask &= (df_work['som_vendor'] == 'apple_autoexport')
        
        n_som_valid = som_valid_mask.sum()
        logger.info(f"[5.1] Days with valid SoM: {n_som_valid} / {n_after_temporal}")
        
        MIN_SOM_DAYS = 10
        if n_som_valid < MIN_SOM_DAYS:
            logger.warning(f"[5.1] Only {n_som_valid} days with SoM (min={MIN_SOM_DAYS})")
        
        df_work = df_work[som_valid_mask].copy()
        
        if len(df_work) == 0:
            logger.error("[5.1] No days with valid SoM after filtering!")
            ctx.log_stage_result(5, "skipped", error="No valid SoM data")
            return None
        
        # Log SoM class distribution
        som_dist = df_work['som_category_3class'].value_counts().sort_index()
        logger.info(f"[5.1] SoM class distribution:")
        for cls, count in som_dist.items():
            pct = 100 * count / len(df_work)
            label_name = {-1: "Negative/Unstable", 0: "Neutral", 1: "Positive/Stable"}.get(int(cls), "Unknown")
            logger.info(f"  Class {int(cls):+d} ({label_name}): {count} ({pct:.1f}%)")
        
        # Derive som_binary
        df_work['som_binary'] = (df_work['som_category_3class'] == -1).astype(int)
        n_unstable = df_work['som_binary'].sum()
        logger.info(f"[5.1] Derived som_binary: {n_unstable} unstable ({100*n_unstable/len(df_work):.1f}%)")
        
        # =================================================================
        # 5.2 ANTI-LEAK REMOVAL
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("5.2 ANTI-LEAK REMOVAL")
        logger.info("=" * 60)
        
        cols_to_drop = [c for c in ANTI_LEAK_COLS if c in df_work.columns]
        if cols_to_drop:
            logger.info(f"[5.2] Removing anti-leak columns: {cols_to_drop}")
            df_work = df_work.drop(columns=cols_to_drop)
        else:
            logger.info("[5.2] No anti-leak columns found to remove")
        
        # Verify pbsi_score kept as candidate feature
        if 'pbsi_score' in df_work.columns:
            logger.info("[5.2] pbsi_score retained as candidate feature (FS-D)")
        
        # =================================================================
        # 5.3 MICE IMPUTATION
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("5.3 MICE IMPUTATION")
        logger.info("=" * 60)
        
        # Get available feature columns
        feature_cols = [c for c in FEATURE_UNIVERSE_COLS if c in df_work.columns]
        missing_features = [c for c in FEATURE_UNIVERSE_COLS if c not in df_work.columns]
        
        if missing_features:
            logger.warning(f"[5.3] Missing feature columns: {missing_features}")
        
        logger.info(f"[5.3] Feature universe: {len(feature_cols)} columns")
        
        # Log feature coverage before imputation
        for fc in feature_cols:
            n_valid = df_work[fc].notna().sum()
            pct_valid = 100 * n_valid / len(df_work)
            logger.info(f"  {fc}: {n_valid}/{len(df_work)} ({pct_valid:.1f}%)")
        
        n_missing_before = df_work[feature_cols].isna().sum().sum()
        pct_missing = 100 * n_missing_before / (len(df_work) * len(feature_cols))
        logger.info(f"[5.3] Missing values: {n_missing_before} ({pct_missing:.1f}%)")
        
        if n_missing_before > 0:
            logger.info("[5.3] Running MICE imputation (segment-aware)...")
            
            if 'segment_id' in df_work.columns:
                df_imputed_list = []
                for segment in df_work['segment_id'].unique():
                    segment_mask = df_work['segment_id'] == segment
                    segment_df = df_work[segment_mask].copy()
                    
                    if len(segment_df) >= 5:
                        imputer = IterativeImputer(
                            max_iter=10, random_state=42, verbose=0,
                            n_nearest_features=None, sample_posterior=True
                        )
                        segment_df[feature_cols] = imputer.fit_transform(segment_df[feature_cols])
                    
                    df_imputed_list.append(segment_df)
                
                df_work = pd.concat(df_imputed_list, ignore_index=False).sort_values('date')
            else:
                imputer = IterativeImputer(
                    max_iter=10, random_state=42, verbose=0,
                    n_nearest_features=None, sample_posterior=True
                )
                df_work[feature_cols] = imputer.fit_transform(df_work[feature_cols])
            
            n_missing_after = df_work[feature_cols].isna().sum().sum()
            logger.info(f"[5.3] After MICE: {n_missing_after} missing values")
        else:
            logger.info("[5.3] No imputation needed - data complete")
        
        # =================================================================
        # 5.4 FEATURE UNIVERSE
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("5.4 FEATURE UNIVERSE")
        logger.info("=" * 60)
        
        # Build universe dataframe
        target_cols = ['som_category_3class', 'som_binary']
        meta_cols = ['segment_id'] if 'segment_id' in df_work.columns else []
        
        universe_cols = ['date'] + feature_cols + target_cols + meta_cols
        universe_cols = [c for c in universe_cols if c in df_work.columns]
        
        df_universe = df_work[universe_cols].copy()
        
        # Save Feature Universe
        universe_path = ctx.ai_snapshot_dir / "ml6" / "features_daily_ml_universe.csv"
        universe_path.parent.mkdir(parents=True, exist_ok=True)
        df_universe.to_csv(universe_path, index=False)
        
        logger.info(f"[5.4] Feature Universe: {df_universe.shape}")
        logger.info(f"  Saved: {universe_path}")
        
        # =================================================================
        # 5.5 EXPERIMENT SUITE
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("5.5 EXPERIMENT SUITE")
        logger.info("=" * 60)
        
        model_selection = None
        experiment_suite_ran = False
        selected_features = FS_B_FEATURE_COLS  # Default
        
        try:
            from src.etl.experiment_suite import run_experiment_suite
            
            logger.info("[5.5] Running Experiment Suite (ML6/ML7 ablation)...")
            
            model_selection = run_experiment_suite(
                df_universe=df_universe,
                participant=ctx.participant,
                snapshot=ctx.snapshot,
                output_dir=ctx.ai_snapshot_dir,
                skip_ml7_lstm=(len(df_universe) < 30)  # Skip LSTM if too few samples
            )
            
            experiment_suite_ran = True
            
            # Extract selected features from model_selection
            if model_selection and 'ml6' in model_selection:
                ml6_selection = model_selection['ml6']
                selected_features = ml6_selection.get('features', FS_B_FEATURE_COLS)
                selected_fs = ml6_selection.get('selected_fs', 'FS-B')
                selected_target = ml6_selection.get('selected_target', 'binary')
                
                logger.info(f"[5.5] ML6 Selection: {selected_fs} × {selected_target}")
                logger.info(f"  Features: {len(selected_features)}")
                if 'metrics' in ml6_selection:
                    f1 = ml6_selection['metrics'].get('f1_macro', 0)
                    logger.info(f"  Expected F1-macro: {f1:.4f}")
            
            if model_selection and 'ml7' in model_selection:
                ml7_selection = model_selection['ml7']
                ml7_config = ml7_selection.get('selected_config', 'CFG-3')
                ml7_status = ml7_selection.get('status', 'unknown')
                logger.info(f"[5.5] ML7 Selection: {ml7_config} (status={ml7_status})")
                
        except ImportError as e:
            logger.warning(f"[5.5] Experiment Suite not available: {e}")
            logger.warning("[5.5] Using default FS-B feature set")
            experiment_suite_ran = False
            
        except Exception as e:
            logger.warning(f"[5.5] Experiment Suite failed: {e}")
            logger.warning("[5.5] Falling back to FS-B defaults")
            experiment_suite_ran = False
            
            # Create fallback model_selection
            model_selection = {
                'snapshot': ctx.snapshot,
                'participant': ctx.participant,
                'generated_at': datetime.now().isoformat(),
                'experiment_suite_version': '1.0.0',
                'fallback': True,
                'fallback_reason': str(e),
                'ml6': {
                    'selected_fs': 'FS-B',
                    'selected_target': 'binary',
                    'features': [f for f in FS_B_FEATURE_COLS if f in df_universe.columns],
                    'n_features': len([f for f in FS_B_FEATURE_COLS if f in df_universe.columns]),
                    'selection_reason': 'Fallback to FS-B defaults',
                },
                'ml7': {
                    'selected_config': 'CFG-3',
                    'selected_target': 'som_binary',
                    'features': [f for f in FS_B_FEATURE_COLS if f in df_universe.columns],
                    'selection_reason': 'Fallback to CFG-3 defaults',
                }
            }
            
            # Save fallback model_selection
            ms_path = ctx.ai_snapshot_dir / 'model_selection.json'
            with open(ms_path, 'w') as f:
                json.dump(model_selection, f, indent=2, default=str)
        
        # =================================================================
        # 5.6 SAVE ML6 VIEW (backward compatibility)
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("5.6 SAVE ML6 VIEW")
        logger.info("=" * 60)
        
        # Build ML6 output with selected features
        selected_features = [f for f in selected_features if f in df_universe.columns]
        ml6_cols = ['date'] + selected_features + target_cols + meta_cols
        ml6_cols = [c for c in ml6_cols if c in df_universe.columns]
        
        df_ml6 = df_universe[ml6_cols].copy()
        
        ml6_path = ctx.ai_snapshot_dir / "ml6" / "features_daily_ml6.csv"
        df_ml6.to_csv(ml6_path, index=False)
        
        logger.info(f"[5.6] ML6 view: {df_ml6.shape}")
        logger.info(f"  Features: {len(selected_features)}")
        logger.info(f"  Saved: {ml6_path}")
        
        # =================================================================
        # STAGE 5 SUMMARY
        # =================================================================
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 5 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  ML period: >= {ml_cutoff.strftime('%Y-%m-%d')}")
        logger.info(f"  SoM days: {len(df_ml6)}")
        logger.info(f"  Feature Universe: {len(feature_cols)} columns")
        logger.info(f"  ML6 Features: {len(selected_features)} columns")
        logger.info(f"  Experiment Suite: {'ran' if experiment_suite_ran else 'fallback'}")
        logger.info(f"  Outputs:")
        logger.info(f"    - {universe_path}")
        logger.info(f"    - {ml6_path}")
        if model_selection:
            logger.info(f"    - {ctx.ai_snapshot_dir / 'model_selection.json'}")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(5, "success", duration_sec=elapsed,
                            rows=len(df_ml6), columns=len(df_ml6.columns),
                            n_som_days=len(df_ml6),
                            n_features_universe=len(feature_cols),
                            n_features_ml6=len(selected_features),
                            experiment_suite_ran=experiment_suite_ran,
                            som_distribution=som_dist.to_dict(),
                            model_selection_summary={
                                'ml6_fs': model_selection.get('ml6', {}).get('selected_fs') if model_selection else 'FS-B',
                                'ml6_target': model_selection.get('ml6', {}).get('selected_target') if model_selection else 'binary',
                                'ml7_config': model_selection.get('ml7', {}).get('selected_config') if model_selection else 'CFG-3',
                            })
        return df_ml6
    
    except Exception as e:
        logger.error(f"✗ Stage 5 failed: {e}", exc_info=True)
        ctx.log_stage_result(5, "failed", error=str(e))
        return None


def stage_6_ml6(ctx: PipelineContext, df: pd.DataFrame) -> bool:
    """
    Stage 6: ML6 Training (Logistic Regression) - SoM-Centric
    
    Based on ablation study (ml6_som_experiments.py):
    - **Best configuration**: FS-B × som_binary (F1=0.4623)
    - **Primary target**: som_binary (unstable vs rest)
    - **Fallback**: som_category_3class if binary fails
    
    Ablation study findings:
    - Binary target outperforms 3-class by +59% (0.46 vs 0.29 F1)
    - HRV features are critical (+0.14 F1 improvement)
    - MEDS and PBSI add minimal value
    
    Cross-Validation:
    - 6-fold temporal CV
    - Deterministic splits based on date
    - class_weight='balanced' to handle imbalance
    """
    banner("STAGE 6: ML6 TRAINING (SoM Binary, FS-B, LogisticRegression)")
    stage_start = time.time()
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, cohen_kappa_score
        from src.etl.ml7_analysis import create_calendar_folds
        import numpy as np
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        # =====================================================================
        # 1. SELECT TARGET (Binary preferred, 3-class fallback)
        # =====================================================================
        # Ablation study showed binary outperforms 3-class by +59%
        MIN_SAMPLES_PER_CLASS = 5
        
        # Check binary distribution first (preferred)
        y_binary = df['som_binary'].copy()
        class_dist_binary = y_binary.value_counts().sort_index()
        min_binary_count = class_dist_binary.min()
        
        logger.info("[SoM Target] Binary distribution (preferred per ablation study):")
        for cls, count in class_dist_binary.items():
            label = "Unstable" if cls == 1 else "Not Unstable"
            logger.info(f"  Class {cls} ({label}): {count}")
        
        # Check 3-class distribution (fallback)
        y_multi = df['som_category_3class'].copy()
        class_dist_multi = y_multi.value_counts().sort_index()
        
        logger.info("[SoM Target] 3-class distribution (fallback):")
        for cls, count in class_dist_multi.items():
            logger.info(f"  Class {cls:+.0f}: {count}")
        
        # Determine target: prefer binary per ablation study
        if min_binary_count >= MIN_SAMPLES_PER_CLASS:
            y = y_binary
            target_name = "som_binary"
            n_classes = 2
            class_dist = class_dist_binary.to_dict()
            logger.info(f"[SoM Target] Using: {target_name} (preferred - F1≈0.46 per ablation)")
        elif class_dist_multi.min() >= MIN_SAMPLES_PER_CLASS:
            y = y_multi
            target_name = "som_3class"
            n_classes = int(len(class_dist_multi))
            class_dist = class_dist_multi.to_dict()
            logger.info(f"[SoM Target] Using: {target_name} (fallback - binary too imbalanced)")
        else:
            logger.warning("[SoM Target] Both targets too imbalanced")
            logger.warning("[ML6] SKIPPED - Insufficient class balance")
            
            elapsed = time.time() - stage_start
            ctx.log_stage_result(6, "skipped", duration_sec=elapsed,
                                error="Insufficient class balance",
                                class_dist_3class=class_dist_multi.to_dict(),
                                class_dist_binary=class_dist_binary.to_dict())
            return True  # Return True to continue pipeline
        
        # =====================================================================
        # 2. PREPARE FEATURES (exclude date and target columns)
        # =====================================================================
        exclude_cols = ['date', 'som_category_3class', 'som_binary', 'segment_id']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].copy()
        
        logger.info(f"[Features] Using {len(feature_cols)} features")
        
        # =====================================================================
        # 3. CREATE CV FOLDS
        # =====================================================================
        # Note: create_calendar_folds uses label_3cls internally, need to adapt
        # For now, create simple temporal folds
        
        n_samples = len(df)
        n_folds = min(6, n_samples // 10)  # Ensure at least 10 samples per fold
        
        if n_folds < 2:
            logger.warning(f"[CV] Only {n_samples} samples - insufficient for CV")
            logger.warning("[ML6] SKIPPED - Need at least 20 samples for minimal CV")
            
            elapsed = time.time() - stage_start
            ctx.log_stage_result(6, "skipped", duration_sec=elapsed,
                                error=f"Only {n_samples} samples",
                                target_used=target_name,
                                class_dist=class_dist)
            return True
        
        # Simple temporal CV: split into folds by time
        fold_size = n_samples // n_folds
        cv_results = []
        
        logger.info(f"[CV] Creating {n_folds}-fold temporal CV (fold_size≈{fold_size})")
        
        for fold_idx in range(n_folds):
            # Use last portion as validation
            val_start = fold_idx * fold_size
            val_end = min((fold_idx + 1) * fold_size, n_samples)
            
            val_idx = list(range(val_start, val_end))
            train_idx = [i for i in range(n_samples) if i not in val_idx]
            
            if len(train_idx) < 5 or len(val_idx) < 2:
                continue
            
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            
            # Check if both train and val have all classes
            train_classes = set(y_train.unique())
            val_classes = set(y_val.unique())
            
            if len(train_classes) < 2:
                logger.warning(f"  Fold {fold_idx}: train has only {len(train_classes)} class(es) - skipping")
                continue
            
            # Train model
            model = LogisticRegression(
                multi_class='auto',
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
            ba = balanced_accuracy_score(y_val, y_pred)
            kappa = cohen_kappa_score(y_val, y_pred)
            
            val_start_date = df.iloc[val_start]['date'].strftime('%Y-%m-%d')
            val_end_date = df.iloc[val_end-1]['date'].strftime('%Y-%m-%d')
            
            result = {
                "fold": fold_idx,
                "val_start": val_start_date,
                "val_end": val_end_date,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "f1_macro": float(f1),
                "balanced_accuracy": float(ba),
                "cohen_kappa": float(kappa)
            }
            cv_results.append(result)
            logger.info(f"  Fold {fold_idx} ({val_start_date}→{val_end_date}): "
                       f"F1={f1:.4f}, BA={ba:.4f}, κ={kappa:.4f}")
        
        # =====================================================================
        # 4. AGGREGATE RESULTS
        # =====================================================================
        if len(cv_results) == 0:
            logger.warning("[ML6] No valid CV folds - all folds had class issues")
            elapsed = time.time() - stage_start
            ctx.log_stage_result(6, "skipped", duration_sec=elapsed,
                                error="No valid CV folds",
                                target_used=target_name,
                                class_dist=class_dist)
            return True
        
        mean_f1 = np.mean([r["f1_macro"] for r in cv_results])
        std_f1 = np.std([r["f1_macro"] for r in cv_results])
        mean_ba = np.mean([r["balanced_accuracy"] for r in cv_results])
        mean_kappa = np.mean([r["cohen_kappa"] for r in cv_results])
        
        # =====================================================================
        # 5. SAVE SUMMARY
        # =====================================================================
        cv_summary = {
            "model": "LogisticRegression",
            "feature_set": "FS-B (Baseline + HRV)",
            "target": target_name,
            "target_type": "som",
            "n_classes": n_classes,
            "class_distribution": class_dist,
            "cv_type": f"temporal_{len(cv_results)}fold",
            "mean_f1_macro": float(mean_f1),
            "std_f1_macro": float(std_f1),
            "mean_balanced_accuracy": float(mean_ba),
            "mean_cohen_kappa": float(mean_kappa),
            "n_samples": n_samples,
            "n_features": len(feature_cols),
            "folds": cv_results,
            "ablation_reference": "docs/reports/ML6_SOM_feature_ablation_P000001_2025-12-08.md",
            "warnings": []
        }
        
        # Add warning if had to use 3-class as fallback
        if target_name == "som_3class":
            cv_summary["warnings"].append("Used 3-class fallback (binary too imbalanced)")
        
        cv_path = ctx.ai_snapshot_dir / "ml6" / "cv_summary.json"
        with open(cv_path, 'w') as f:
            json.dump(cv_summary, f, indent=2)
        
        logger.info(f"[OK] Stage 6 complete: target={target_name}, F1={mean_f1:.4f}±{std_f1:.4f}, κ={mean_kappa:.4f}")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(6, "success", duration_sec=elapsed,
                            target_used=target_name,
                            mean_f1=float(mean_f1), std_f1=float(std_f1),
                            mean_kappa=float(mean_kappa),
                            n_folds=len(cv_results))
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 6 failed: {e}", exc_info=True)
        ctx.log_stage_result(6, "failed", error=str(e))
        return False


def stage_7_ml7(ctx: PipelineContext) -> bool:
    """
    Stage 7: ML7 Analysis (LSTM + SHAP + Drift) - SoM-Centric
    
    Uses SoM as the primary target for sequence classification.
    Configuration based on ablation study (CFG-3 × som_binary):
    - Seq len: 14 days
    - LSTM: 32 units
    - Dense: 32 units
    - Dropout: 0.4 (regularized)
    - Early stopping: patience=3
    - Class weights: balanced
    - Target: som_binary (preferred per ablation results)
    
    Components:
    1. SHAP: Feature importance on LogisticRegression
    2. Drift: SoM distribution changes over time
    3. LSTM: Sequence classifier for SoM prediction
    """
    banner("STAGE 7: ML7 ANALYSIS (SoM LSTM + SHAP)")
    stage_start = time.time()
    
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.utils.class_weight import compute_class_weight
        from src.etl.ml7_analysis import (
            compute_shap_values, create_lstm_sequences
        )
        import numpy as np
        
        # =================================================================
        # CFG-3 Configuration (from ablation study)
        # =================================================================
        SEQ_LEN = 14
        LSTM_UNITS = 32
        DENSE_UNITS = 32
        DROPOUT = 0.4
        USE_EARLY_STOPPING = True
        EARLY_STOPPING_PATIENCE = 3
        USE_CLASS_WEIGHT = True
        EPOCHS = 50
        BATCH_SIZE = 16
        MIN_DAYS_FOR_LSTM = 30
        
        # Create output directories
        shap_dir = ctx.ai_snapshot_dir / "ml7" / "shap"
        drift_dir = ctx.ai_snapshot_dir / "ml7" / "drift"
        models_dir = ctx.ai_snapshot_dir / "ml7" / "models"
        
        for d in [shap_dir, drift_dir, models_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # =====================================================================
        # LOAD ML6 DATA (already filtered for SoM)
        # =====================================================================
        ml6_path = ctx.ai_snapshot_dir / "ml6" / "features_daily_ml6.csv"
        
        if not ml6_path.exists():
            logger.error(f"[ML7] Required ML6 data not found: {ml6_path}")
            logger.warning("[ML7] SKIPPED - Run Stage 5 first")
            ctx.log_stage_result(7, "skipped", error="ML6 data not found")
            return True
        
        logger.info(f"[ML7] Loading SoM-filtered data from Stage 5: {ml6_path}")
        df = pd.read_csv(ml6_path)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        n_samples = len(df)
        logger.info(f"[ML7] Loaded {n_samples} days with SoM labels")
        
        # Check minimum data for LSTM
        if n_samples < MIN_DAYS_FOR_LSTM:
            logger.warning(f"[ML7] Only {n_samples} days (need {MIN_DAYS_FOR_LSTM} for LSTM)")
            logger.warning("[ML7] SKIPPED - Insufficient SoM data for sequence learning")
            
            # Save minimal reports
            with open(shap_dir.parent / "shap_summary.md", 'w') as f:
                f.write("# SHAP Feature Importance Summary\n\n")
                f.write(f"**Status**: SKIPPED - Only {n_samples} samples available\n\n")
                f.write("Minimum 30 samples required for meaningful analysis.\n")
            
            with open(drift_dir.parent / "drift_report.md", 'w') as f:
                f.write("# Drift Detection Report\n\n")
                f.write(f"**Status**: SKIPPED - Only {n_samples} samples available\n")
            
            with open(models_dir.parent / "lstm_report.md", 'w') as f:
                f.write("# LSTM M1 Training Report\n\n")
                f.write(f"**Status**: SKIPPED - Only {n_samples} samples available\n\n")
                f.write(f"Minimum {MIN_DAYS_FOR_LSTM} days required for sequence learning.\n")
            
            elapsed = time.time() - stage_start
            ctx.log_stage_result(7, "skipped", duration_sec=elapsed,
                                error=f"Only {n_samples} samples",
                                min_required=MIN_DAYS_FOR_LSTM)
            return True
        
        # =====================================================================
        # DETERMINE TARGET & FEATURES (prefer som_binary per ablation)
        # =====================================================================
        exclude_cols = ['date', 'som_category_3class', 'som_binary', 'segment_id']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # Prefer som_binary based on ML7 ablation study (CFG-3 × binary = F1 0.5667)
        # Fallback to 3-class only if binary has severe issues
        class_dist_binary = df['som_binary'].value_counts()
        class_dist_3 = df['som_category_3class'].value_counts()
        
        # Check if binary is viable (at least 3 samples in minority class)
        binary_viable = class_dist_binary.min() >= 3
        
        if binary_viable:
            y = df['som_binary'].values
            target_name = "som_binary"
            n_classes = 2
            logger.info(f"[ML7 Target] Using som_binary (preferred per ablation study)")
            logger.info(f"[ML7 Target]   Class 0: {class_dist_binary.get(0, 0)}")
            logger.info(f"[ML7 Target]   Class 1: {class_dist_binary.get(1, 0)}")
        else:
            # Fallback to 3-class
            y = df['som_category_3class'].values
            target_name = "som_3class"
            n_classes = len(class_dist_3)
            logger.info(f"[ML7 Target] Using som_3class (binary fallback not viable)")
        
        X = df[feature_cols].values
        
        logger.info(f"[ML7] Target: {target_name} ({n_classes} classes)")
        logger.info(f"[ML7] Features: {len(feature_cols)}")
        logger.info(f"[ML7] Config: CFG-3 (seq={SEQ_LEN}, lstm={LSTM_UNITS}, "
                   f"drop={DROPOUT}, early_stop={USE_EARLY_STOPPING}, class_wt={USE_CLASS_WEIGHT})")
        
        # =====================================================================
        # SHAP ANALYSIS (on LogisticRegression)
        # =====================================================================
        logger.info("\n[ML7] SHAP Analysis...")
        
        shap_results = []
        global_ranking = []
        
        try:
            # Simple train/test split for SHAP (80/20)
            split_idx = int(0.8 * len(df))
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            X_val = X[split_idx:]
            
            if len(set(y_train)) >= 2:  # Need at least 2 classes in train
                model = LogisticRegression(
                    multi_class='auto',
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Compute SHAP
                shap_result = compute_shap_values(
                    model,
                    pd.DataFrame(X_train, columns=feature_cols),
                    pd.DataFrame(X_val, columns=feature_cols),
                    feature_cols,
                    fold_idx=0,
                    output_dir=shap_dir
                )
                shap_results.append(shap_result)
                
                if 'top_features' in shap_result:
                    global_ranking = [(f['feature'], f['shap_importance']) 
                                     for f in shap_result['top_features']]
                
                logger.info(f"[SHAP] Computed successfully")
            else:
                logger.warning("[SHAP] Skipped - train set has only 1 class")
        
        except Exception as e:
            logger.warning(f"[SHAP] Failed: {e}")
        
        # Save SHAP summary
        with open(shap_dir.parent / "shap_summary.md", 'w') as f:
            f.write("# SHAP Feature Importance Summary (SoM-Centric)\n\n")
            f.write(f"**Target**: {target_name}\n")
            f.write(f"**Model**: Logistic Regression (class_weight='balanced')\n")
            f.write(f"**Samples**: {n_samples}\n\n")
            
            if global_ranking:
                f.write("## Global Feature Ranking\n\n")
                for i, (feat, imp) in enumerate(global_ranking[:10], 1):
                    f.write(f"{i}. **{feat}**: {imp:.4f}\n")
            else:
                f.write("*SHAP analysis skipped due to insufficient data*\n")
        
        # =====================================================================
        # DRIFT DETECTION (SoM distribution over time)
        # =====================================================================
        logger.info("\n[ML7] Drift Detection...")
        
        # Simple drift: check SoM distribution in first vs last half
        mid_idx = len(df) // 2
        first_half_dist = df.iloc[:mid_idx]['som_category_3class'].value_counts(normalize=True)
        second_half_dist = df.iloc[mid_idx:]['som_category_3class'].value_counts(normalize=True)
        
        # Save drift report
        with open(drift_dir.parent / "drift_report.md", 'w') as f:
            f.write("# Drift Detection Report (SoM-Centric)\n\n")
            f.write("## SoM Distribution Over Time\n\n")
            f.write("### First Half of Data\n")
            for cls, pct in first_half_dist.items():
                f.write(f"- Class {cls:+.0f}: {100*pct:.1f}%\n")
            f.write(f"\n### Second Half of Data\n")
            for cls, pct in second_half_dist.items():
                f.write(f"- Class {cls:+.0f}: {100*pct:.1f}%\n")
        
        logger.info("[Drift] SoM distribution analysis complete")
        
        # =====================================================================
        # LSTM TRAINING (CFG-3: regularized with class weights)
        # =====================================================================
        logger.info("\n[ML7] LSTM Training (CFG-3 Regularized)...")
        
        # Check if we have enough classes for classification
        if n_classes < 2:
            logger.warning(f"[LSTM] Only {n_classes} class(es) in target - need at least 2")
            logger.warning("[LSTM] Skipped - insufficient class variability")
            best_model = None
            lstm_results = []
        
        elif (n_sequences := n_samples - SEQ_LEN + 1) < 10:
            logger.warning(f"[LSTM] Only {n_sequences} sequences possible (need 10+)")
            logger.warning("[LSTM] Skipped - insufficient sequences")
            best_model = None
            lstm_results = []
        else:
            # Create sequences
            X_seq, y_seq = create_lstm_sequences(X, y, SEQ_LEN)
            
            if len(X_seq) < 10:
                logger.warning(f"[LSTM] Only {len(X_seq)} valid sequences - skipping")
                best_model = None
                lstm_results = []
            else:
                # Simple train/val split for LSTM (80/20)
                split_idx = int(0.8 * len(X_seq))
                X_train_seq = X_seq[:split_idx]
                y_train_seq = y_seq[:split_idx]
                X_val_seq = X_seq[split_idx:]
                y_val_seq = y_seq[split_idx:]
                
                logger.info(f"[LSTM] Train sequences: {len(X_train_seq)}")
                logger.info(f"[LSTM] Val sequences: {len(X_val_seq)}")
                
                # Train LSTM with CFG-3 configuration
                result = train_lstm_model_cfg3(
                    X_train_seq, y_train_seq,
                    X_val_seq, y_val_seq,
                    n_classes=n_classes,
                    seq_len=SEQ_LEN,
                    n_features=len(feature_cols),
                    lstm_units=LSTM_UNITS,
                    dense_units=DENSE_UNITS,
                    dropout=DROPOUT,
                    use_early_stopping=USE_EARLY_STOPPING,
                    early_stopping_patience=EARLY_STOPPING_PATIENCE,
                    use_class_weight=USE_CLASS_WEIGHT,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE
                )
                
                if 'error' not in result:
                    lstm_results = [result]
                    best_model = result.get('model')
                    logger.info(f"[LSTM] F1={result['f1_macro']:.4f}, "
                               f"epochs={result['n_epochs_trained']}, "
                               f"early_stopped={result['early_stopped']}")
                else:
                    logger.warning(f"[LSTM] Training failed: {result.get('error')}")
                    lstm_results = []
                    best_model = None
        
        # Save LSTM report (CFG-3 configuration)
        with open(models_dir.parent / "lstm_report.md", 'w', encoding='utf-8') as f:
            f.write("# LSTM M1 Training Report (SoM-Centric, CFG-3)\n\n")
            f.write(f"## Dataset Summary\n\n")
            f.write(f"- **Total Days**: {n_samples}\n")
            f.write(f"- **Target**: {target_name}\n")
            f.write(f"- **Sequence Length**: {SEQ_LEN} days\n")
            f.write(f"- **N Features**: {len(feature_cols)}\n")
            f.write(f"- **N Classes**: {n_classes}\n\n")
            
            f.write("## Configuration (CFG-3 Regularized)\n\n")
            f.write(f"- **LSTM Units**: {LSTM_UNITS}\n")
            f.write(f"- **Dense Units**: {DENSE_UNITS}\n")
            f.write(f"- **Dropout**: {DROPOUT}\n")
            f.write(f"- **Early Stopping**: {USE_EARLY_STOPPING} (patience={EARLY_STOPPING_PATIENCE})\n")
            f.write(f"- **Class Weights**: {USE_CLASS_WEIGHT}\n")
            f.write(f"- **Max Epochs**: {EPOCHS}\n")
            f.write(f"- **Batch Size**: {BATCH_SIZE}\n\n")
            
            if lstm_results:
                r = lstm_results[0]
                f.write("## Training Results\n\n")
                f.write(f"- **Architecture**: LSTM({LSTM_UNITS}) → Dense({DENSE_UNITS}) → Dropout({DROPOUT}) → Softmax\n")
                f.write(f"- **Macro-F1**: {r['f1_macro']:.4f}\n")
                f.write(f"- **F1-Weighted**: {r.get('f1_weighted', 0):.4f}\n")
                f.write(f"- **Balanced Accuracy**: {r.get('balanced_accuracy', 0):.4f}\n")
                f.write(f"- **Val Loss**: {r['val_loss']:.4f}\n")
                f.write(f"- **Val Accuracy**: {r['val_accuracy']:.4f}\n")
                f.write(f"- **Epochs Trained**: {r.get('n_epochs_trained', 'N/A')}\n")
                f.write(f"- **Early Stopped**: {r.get('early_stopped', False)}\n")
                f.write("\n### Ablation Reference\n\n")
                f.write("Configuration selected from ML7 ablation study: CFG-3 × som_binary achieved F1=0.5667\n")
            else:
                f.write("## Status\n\n")
                f.write("*LSTM training skipped due to insufficient data*\n")
        
        # Store results in context for Stage 8
        ctx.results['best_lstm_model'] = best_model
        ctx.results['lstm_results'] = lstm_results
        ctx.results['shap_global_ranking'] = global_ranking
        ctx.results['ml7_target'] = target_name
        ctx.results['ml7_n_samples'] = n_samples
        ctx.results['ml7_config'] = f"CFG-3 (seq={SEQ_LEN}, lstm={LSTM_UNITS}, drop={DROPOUT})"
        
        logger.info(f"[OK] Stage 7 complete: SHAP [{'OK' if global_ranking else 'SKIP'}] "
                   f"Drift [OK] LSTM [{'OK' if best_model else 'SKIP'}]")
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(7, "success", duration_sec=elapsed,
                            target=target_name,
                            n_samples=n_samples,
                            lstm_trained=best_model is not None)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 7 failed: {e}", exc_info=True)
        ctx.log_stage_result(7, "failed", error=str(e))
        return False


def stage_6_extended(ctx: PipelineContext) -> bool:
    """
    Stage 6-Extended: ML6-Extended Multi-Algorithm Pipeline
    
    Runs multiple classical ML algorithms on the selected feature set:
    - LogisticRegression
    - RandomForest
    - GradientBoosting
    - XGBoost (optional)
    - SVM (Linear, RBF)
    - GaussianNB
    - KNN (k=3, 5, 7)
    
    All models:
    - Load feature set and target from model_selection.json
    - Apply anti-leak column filtering
    - Fit scaler ONLY on training data per fold
    - Use 6-fold temporal CV
    
    Output: data/ai/<PID>/<SNAPSHOT>/ml6_extended/
    """
    banner("STAGE 6-EXTENDED: ML6 Multi-Algorithm Pipeline")
    stage_start = time.time()
    
    try:
        from src.etl.ml6_extended import run_ml6_extended
        
        results = run_ml6_extended(
            participant=ctx.participant,
            snapshot=ctx.snapshot,
            output_base=Path('data/ai'),
            skip_shap=False
        )
        
        if results.get('status') == 'success':
            best_model = results.get('best_model', {})
            logger.info(f"[OK] Stage 6-Extended complete: Best={best_model.get('name')} "
                       f"(F1={best_model.get('f1_macro', 0):.4f})")
            
            elapsed = time.time() - stage_start
            ctx.log_stage_result("6_extended", "success", duration_sec=elapsed,
                                best_model=best_model.get('name'),
                                f1_macro=best_model.get('f1_macro', 0),
                                n_models_tested=len(results.get('models', {})))
            
            # Store for reporting
            ctx.results['ml6_extended'] = results
            return True
        else:
            raise Exception(results.get('error', 'Unknown error'))
    
    except ImportError as e:
        logger.warning(f"[ML6-Ext] Skipped - Import error: {e}")
        ctx.log_stage_result("6_extended", "skipped", error=f"Import error: {e}")
        return True  # Continue pipeline
    
    except Exception as e:
        logger.error(f"✗ Stage 6-Extended failed: {e}", exc_info=True)
        ctx.log_stage_result("6_extended", "failed", error=str(e))
        return True  # Continue pipeline (non-critical)


def stage_7_extended(ctx: PipelineContext) -> bool:
    """
    Stage 7-Extended: ML7-Extended Multi-Architecture Pipeline
    
    Runs multiple sequence-based deep learning architectures:
    - LSTM (CFG-3 configuration)
    - GRU
    - 1D-CNN (Conv1D)
    - CNN-LSTM Hybrid
    
    All models:
    - Load feature set, target, and sequence length from model_selection.json
    - Apply anti-leak column filtering BEFORE sequence creation
    - Fit scaler ONLY on training data per fold
    - Use temporal CV for sequences
    
    Output: data/ai/<PID>/<SNAPSHOT>/ml7_extended/
    """
    banner("STAGE 7-EXTENDED: ML7 Multi-Architecture Pipeline")
    stage_start = time.time()
    
    try:
        from src.etl.ml7_extended import run_ml7_extended
        
        results = run_ml7_extended(
            participant=ctx.participant,
            snapshot=ctx.snapshot,
            output_base=Path('data/ai'),
            epochs=50,
            batch_size=16
        )
        
        if results.get('status') == 'success':
            best_model = results.get('best_model', {})
            logger.info(f"[OK] Stage 7-Extended complete: Best={best_model.get('name')} "
                       f"(F1={best_model.get('f1_macro', 0):.4f})")
            
            elapsed = time.time() - stage_start
            ctx.log_stage_result("7_extended", "success", duration_sec=elapsed,
                                best_model=best_model.get('name'),
                                f1_macro=best_model.get('f1_macro', 0),
                                n_models_tested=len(results.get('models', {})),
                                seq_len=results.get('seq_len'))
            
            # Store for reporting
            ctx.results['ml7_extended'] = results
            return True
        else:
            raise Exception(results.get('error', 'Unknown error'))
    
    except ImportError as e:
        logger.warning(f"[ML7-Ext] Skipped - Import error: {e}")
        logger.warning(f"[ML7-Ext] TensorFlow may not be installed")
        ctx.log_stage_result("7_extended", "skipped", error=f"Import error: {e}")
        return True  # Continue pipeline
    
    except Exception as e:
        logger.error(f"✗ Stage 7-Extended failed: {e}", exc_info=True)
        ctx.log_stage_result("7_extended", "failed", error=str(e))
        return True  # Continue pipeline (non-critical)


def stage_8_tflite(ctx: PipelineContext) -> bool:
    """
    Stage 8: TFLite Export (SoM-Centric)
    Convert best LSTM model (trained on SoM target) to TFLite and measure latency
    """
    banner("STAGE 8: TFLITE EXPORT (SoM Model)")
    stage_start = time.time()
    
    try:
        from src.etl.ml7_analysis import convert_to_tflite, measure_latency
        import numpy as np
        
        best_model = ctx.results.get('best_lstm_model')
        
        if best_model is None:
            logger.warning("[TFLite] No LSTM model available (SoM data insufficient)")
            logger.info("[TFLite] Skipped - Stage 7 did not produce a model")
            ctx.log_stage_result(8, "skipped", error="No LSTM model from Stage 7")
            return True
        
        # Convert to TFLite
        tflite_path = ctx.ai_snapshot_dir / "ml7" / "models" / "best_model.tflite"
        tflite_path.parent.mkdir(parents=True, exist_ok=True)
        success = convert_to_tflite(best_model, tflite_path)
        
        if not success:
            raise RuntimeError("TFLite conversion failed")
        
        # Measure latency
        # Use ML6 features from Stage 5 output (SoM-filtered data)
        ml6_path = ctx.ai_snapshot_dir / "ml6" / "features_daily_ml6.csv"
        
        if not ml6_path.exists():
            logger.warning(f"[TFLite] ML6 data not found: {ml6_path}, skipping latency test")
            logger.info(f"[OK] Stage 8 complete: TFLite exported (latency test skipped)")
            ctx.log_stage_result(8, "success", duration_sec=time.time() - stage_start)
            return True
        
        df = pd.read_csv(ml6_path)
        # Drop target columns and identifiers to get feature matrix
        drop_cols = ['date', 'som_category_3class', 'som_binary', 'segment_id']
        drop_cols = [c for c in drop_cols if c in df.columns]
        X = df.drop(columns=drop_cols).values
        
        seq_len = 14
        n_features = X.shape[1]
        dummy_input = np.zeros((1, seq_len, n_features), dtype=np.float32)
        
        latency_stats = measure_latency(tflite_path, dummy_input, n_runs=100)
        
        # Save latency stats
        latency_path = ctx.ai_snapshot_dir / "ml7" / "latency_stats.json"
        with open(latency_path, 'w') as f:
            json.dump(latency_stats, f, indent=2)
        
        p95 = latency_stats.get('p95_ms', 0)
        target_used = ctx.results.get('ml7_target', 'som_category_3class')
        
        if isinstance(p95, (int, float)):
            logger.info(f"[OK] Stage 8 complete: TFLite exported (target={target_used}), latency p95={p95:.2f}ms")
        else:
            logger.info(f"[OK] Stage 8 complete: TFLite exported (target={target_used}), latency p95={p95}")
        
        ctx.results['latency_p95'] = latency_stats.get('p95_ms', 0)
        ctx.results['tflite_path'] = str(tflite_path)
        
        elapsed = time.time() - stage_start
        ctx.log_stage_result(8, "success", duration_sec=elapsed, 
                            latency_p95=latency_stats.get('p95_ms', 0),
                            target=target_used)
        return True
    
    except Exception as e:
        logger.error(f"✗ Stage 8 failed: {e}", exc_info=True)
        ctx.log_stage_result(8, "failed", error=str(e))
        return False


def stage_9_report(ctx: PipelineContext, stages_executed: str = "0-9") -> bool:
    """
    Stage 9: Generate Report (SoM-Centric)
    
    Create RUN_<PID>_<SNAPSHOT>_<STAGES>_<TIMESTAMP>.md in docs/reports/
    
    Report content:
    - SoM (State of Mind) as primary ML target
    - PBSI as auxiliary feature (no longer target)
    - MEDS and HRV domain coverage
    - ML6/ML7 results with SoM target
    """
    banner("STAGE 9: GENERATE REPORT (SoM-Centric)")
    stage_start = time.time()
    
    try:
        import numpy as np
        
        # Read key files
        labeled_path = ctx.joined_dir / "features_daily_labeled.csv"
        if not labeled_path.exists():
            logger.warning("features_daily_labeled.csv not found, skipping metrics")
            elapsed = time.time() - stage_start
            ctx.log_stage_result(9, "skipped", duration_sec=elapsed)
            return True
        
        df = pd.read_csv(labeled_path)
        
        # =====================================================================
        # Build SoM-Centric Report
        # =====================================================================
        lines = [
            "# RUN_REPORT.md - Pipeline Execution Summary (SoM-Centric)",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Participant**: {ctx.participant}",
            f"**Snapshot**: {ctx.snapshot}",
            f"**Stages Executed**: {stages_executed}",
            "",
            "---",
            "",
            "## ML Strategy",
            "",
            "- **Primary ML Target**: `som_category_3class` (State of Mind)",
            "- **Secondary Target**: `som_binary` (1 if unstable, 0 otherwise)",
            "- **PBSI**: Used as auxiliary feature (`pbsi_score`), NOT as target",
            "- **Extended Features**: HR, HRV (SDNN), Sleep, Activity, Meds",
            "",
            "---",
            "",
            "## Data Summary",
            "",
            f"- **Date Range**: {df['date'].min()} to {df['date'].max()}",
            f"- **Total Days**: {len(df)}",
            "",
        ]
        
        # =====================================================================
        # SoM Coverage
        # =====================================================================
        n_som_valid = df['som_category_3class'].notna().sum() if 'som_category_3class' in df.columns else 0
        som_pct = 100 * n_som_valid / len(df) if len(df) > 0 else 0
        
        lines.extend([
            "### SoM (State of Mind) Coverage",
            "",
            f"- **Days with SoM labels**: {n_som_valid} / {len(df)} ({som_pct:.1f}%)",
        ])
        
        if 'som_category_3class' in df.columns and n_som_valid > 0:
            som_dist = df['som_category_3class'].value_counts().sort_index()
            lines.append("")
            lines.append("**SoM Distribution:**")
            for cls, count in som_dist.items():
                pct = 100 * count / n_som_valid if n_som_valid > 0 else 0
                label_name = {-1: "Negative/Unstable", 0: "Neutral", 1: "Positive/Stable"}.get(int(cls), "Unknown")
                lines.append(f"  - Class {int(cls):+d} ({label_name}): {count} ({pct:.1f}%)")
        
        lines.append("")
        
        # =====================================================================
        # MEDS Coverage
        # =====================================================================
        n_meds = df['med_any'].notna().sum() if 'med_any' in df.columns else 0
        n_meds_taken = (df['med_any'] == 1).sum() if 'med_any' in df.columns else 0
        meds_pct = 100 * n_meds / len(df) if len(df) > 0 else 0
        
        lines.extend([
            "### MEDS (Medication) Coverage",
            "",
            f"- **Days with meds data**: {n_meds} / {len(df)} ({meds_pct:.1f}%)",
            f"- **Days with med_any=1**: {n_meds_taken}",
        ])
        
        if 'med_event_count' in df.columns:
            med_events = df['med_event_count'].sum()
            lines.append(f"- **Total medication events**: {int(med_events)}")
        
        lines.append("")
        
        # =====================================================================
        # HRV Coverage
        # =====================================================================
        n_hrv = df['hrv_sdnn_mean'].notna().sum() if 'hrv_sdnn_mean' in df.columns else 0
        hrv_pct = 100 * n_hrv / len(df) if len(df) > 0 else 0
        
        lines.extend([
            "### HRV (Heart Rate Variability) Coverage",
            "",
            f"- **Days with HRV data**: {n_hrv} / {len(df)} ({hrv_pct:.1f}%)",
        ])
        
        if 'hrv_sdnn_mean' in df.columns and n_hrv > 0:
            hrv_mean = df['hrv_sdnn_mean'].mean()
            hrv_min = df['hrv_sdnn_mean'].min()
            hrv_max = df['hrv_sdnn_mean'].max()
            lines.append(f"- **HRV SDNN range**: {hrv_min:.1f} - {hrv_max:.1f} ms (mean={hrv_mean:.1f})")
        
        lines.append("")
        
        # =====================================================================
        # PBSI Distribution (auxiliary feature)
        # =====================================================================
        if 'label_3cls' in df.columns:
            lines.extend([
                "### PBSI Distribution (Auxiliary Feature)",
                "",
            ])
            for label in [-1, 0, 1]:
                count = (df['label_3cls'] == label).sum()
                pct = count / len(df) * 100
                label_name = {-1: "Dysregulated", 0: "Typical", 1: "Regulated"}[label]
                lines.append(f"- **PBSI {label:+2d} ({label_name})**: {count} ({pct:.1f}%)")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        # =====================================================================
        # ML6 Results (SoM target)
        # =====================================================================
        cv_summary_path = ctx.ai_snapshot_dir / "ml6" / "cv_summary.json"
        if cv_summary_path.exists():
            with open(cv_summary_path, 'r') as f:
                cv_data = json.load(f)
            
            target_used = cv_data.get('target', 'unknown')
            lines.extend([
                "## ML6: Logistic Regression (SoM Target)",
                "",
                f"- **Target Used**: `{target_used}`",
                f"- **N Classes**: {cv_data.get('n_classes', 'N/A')}",
                f"- **N Samples**: {cv_data.get('n_samples', 'N/A')}",
                f"- **N Features**: {cv_data.get('n_features', 'N/A')}",
                f"- **CV Type**: {cv_data.get('cv_type', 'N/A')}",
                f"- **Mean Macro-F1**: {cv_data.get('mean_f1_macro', 0):.4f} ± {cv_data.get('std_f1_macro', 0):.4f}",
                f"- **Mean Balanced Accuracy**: {cv_data.get('mean_balanced_accuracy', 0):.4f}",
                "",
            ])
            
            # Warnings
            warnings = cv_data.get('warnings', [])
            if warnings:
                lines.append("**Warnings:**")
                for w in warnings:
                    lines.append(f"- {w}")
                lines.append("")
            
            # Class distribution
            class_dist = cv_data.get('class_distribution', {})
            if class_dist:
                lines.append("**Class Distribution:**")
                for cls, count in sorted(class_dist.items(), key=lambda x: float(x[0])):
                    lines.append(f"- Class {cls}: {count}")
                lines.append("")
            
            # Per-fold results
            folds = cv_data.get('folds', [])
            if folds:
                lines.append("### Per-Fold Results")
                lines.append("")
                for fold in folds:
                    lines.append(f"- **Fold {fold['fold']}** ({fold.get('val_start', '?')} → {fold.get('val_end', '?')}): "
                               f"F1={fold.get('f1_macro', 0):.4f}, BA={fold.get('balanced_accuracy', 0):.4f}")
                lines.append("")
        else:
            lines.extend([
                "## ML6: Logistic Regression (SoM Target)",
                "",
                "*ML6 was skipped or not run*",
                "",
            ])
        
        lines.append("---")
        lines.append("")
        
        # =====================================================================
        # ML7 Results (SHAP, Drift, LSTM)
        # =====================================================================
        lines.append("## ML7: LSTM + SHAP + Drift (SoM Target)")
        lines.append("")
        
        # Target used
        ml7_target = ctx.results.get('ml7_target', 'unknown')
        ml7_n_samples = ctx.results.get('ml7_n_samples', 0)
        lines.append(f"- **Target Used**: `{ml7_target}`")
        lines.append(f"- **N Samples**: {ml7_n_samples}")
        lines.append("")
        
        # SHAP Top-10
        shap_ranking = ctx.results.get('shap_global_ranking', [])
        if shap_ranking:
            lines.extend([
                "### SHAP Feature Importance (Global Top-10)",
                "",
            ])
            for i, (feat, imp) in enumerate(shap_ranking[:10], 1):
                lines.append(f"{i}. **{feat}**: {imp:.4f}")
            lines.append("")
        else:
            lines.append("*SHAP analysis was skipped (insufficient data)*")
            lines.append("")
        
        # LSTM Results
        lstm_results = ctx.results.get('lstm_results', [])
        if lstm_results:
            mean_lstm_f1 = np.mean([r['f1_macro'] for r in lstm_results])
            lines.extend([
                "### LSTM Training",
                "",
                f"- **Architecture**: LSTM(32) -> Dense(32) -> Dropout(0.2) -> Softmax",
                f"- **Sequence Length**: 14 days",
                f"- **Mean Macro-F1**: {mean_lstm_f1:.4f}",
                f"- **Folds Trained**: {len(lstm_results)}",
                "",
            ])
        else:
            lines.extend([
                "### LSTM Training",
                "",
                "*LSTM was skipped (insufficient SoM sequences)*",
                "",
            ])
        
        lines.append("---")
        lines.append("")
        
        # =====================================================================
        # Stage 8: TFLite & Latency
        # =====================================================================
        tflite_path = ctx.results.get('tflite_path')
        latency_p95 = ctx.results.get('latency_p95', 0)
        
        lines.append("## Stage 8: TFLite Export")
        lines.append("")
        
        if tflite_path:
            lines.extend([
                f"- **Model Path**: `{tflite_path}`",
                f"- **Inference Latency (p95)**: {latency_p95:.2f} ms",
            ])
        else:
            lines.append("*TFLite export was skipped (no LSTM model available)*")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # =====================================================================
        # Artifact Paths
        # =====================================================================
        lines.extend([
            "## Artifact Paths",
            "",
            f"- **Unified CSV**: `{ctx.joined_dir / 'features_daily_unified.csv'}`",
            f"- **Labeled CSV**: `{labeled_path}`",
            f"- **ML6 Dataset**: `{ctx.ai_snapshot_dir / 'ml6' / 'features_daily_ml6.csv'}`",
            f"- **ML6 CV Summary**: `{ctx.ai_snapshot_dir / 'ml6' / 'cv_summary.json'}`",
            f"- **SHAP Summary**: `{ctx.ai_snapshot_dir / 'ml7' / 'shap_summary.md'}`",
            f"- **Drift Report**: `{ctx.ai_snapshot_dir / 'ml7' / 'drift_report.md'}`",
            f"- **LSTM Report**: `{ctx.ai_snapshot_dir / 'ml7' / 'lstm_report.md'}`",
            "",
            "---",
            "",
            "## Status",
            "",
            "✅ **PIPELINE COMPLETE (SoM-Centric ML)**",
            "",
        ])
        
        report_text = "\n".join(lines)
        
        # Generate report filename with PID, snapshot, stages, timestamp
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        report_dir = Path("docs/reports")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_filename = f"RUN_{ctx.participant}_{ctx.snapshot}_stages{stages_executed}_{timestamp}.md"
        report_path = report_dir / report_filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"[OK] Stage 9 complete: {report_path}")
        elapsed = time.time() - stage_start
        ctx.log_stage_result(9, "success", duration_sec=elapsed, report_path=str(report_path))
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
    parser.add_argument("--start-from-etl", action="store_true", 
                        help="Start from existing ETL snapshot (skip stages 0-2 that require raw data). "
                             "Use this when reproducing from public ETL snapshot without raw data access.")
    
    args = parser.parse_args()
    
    if args.snapshot is None:
        args.snapshot = datetime.now().strftime("%Y-%m-%d")
    
    ctx = PipelineContext(args.participant, args.snapshot)
    
    # Handle --start-from-etl mode
    if args.start_from_etl:
        if args.start_stage < 3:
            logger.info("[--start-from-etl] Adjusting start_stage to 3 (skip raw data extraction/aggregation)")
            args.start_stage = 3
        
        # Verify ETL snapshot exists
        etl_extracted_dir = ctx.extracted_dir
        etl_joined_dir = ctx.joined_dir
        if not etl_extracted_dir.exists() or not etl_joined_dir.exists():
            logger.error(f"[FATAL] --start-from-etl specified but ETL snapshot not found:")
            logger.error(f"  Expected: {ctx.snapshot_dir}")
            logger.error(f"  Missing: {etl_extracted_dir if not etl_extracted_dir.exists() else etl_joined_dir}")
            logger.error("")
            logger.error("To reproduce from public ETL snapshot, ensure you have:")
            logger.error(f"  data/etl/{args.participant}/{args.snapshot}/extracted/")
            logger.error(f"  data/etl/{args.participant}/{args.snapshot}/joined/")
            sys.exit(2)
        
        logger.info(f"[OK] ETL snapshot found: {ctx.snapshot_dir}")
    
    # Check raw data availability for stages 0-2
    if args.start_stage < 3 and not args.start_from_etl:
        raw_participant_dir = ctx.raw_dir / ctx.participant
        if not raw_participant_dir.exists():
            logger.error(f"[FATAL] Raw data required for stage {args.start_stage} but not found:")
            logger.error(f"  Expected: {raw_participant_dir}")
            logger.error("")
            logger.error("Raw data (Apple Health export, Zepp ZIPs) is not included in public repository.")
            logger.error("To reproduce from public ETL snapshot, use:")
            logger.error("")
            logger.error(f"  python -m scripts.run_full_pipeline \\")
            logger.error(f"    --participant {args.participant} \\")
            logger.error(f"    --snapshot {args.snapshot} \\")
            logger.error(f"    --start-from-etl --end-stage 9")
            logger.error("")
            logger.error("This will run stages 3-9 (unification → modeling → reporting)")
            sys.exit(2)
    
    banner("FULL DETERMINISTIC PIPELINE (stages 0-9)")
    logger.info(f"Participant: {args.participant}")
    logger.info(f"Snapshot: {args.snapshot}")
    logger.info(f"Stages: {args.start_stage}-{args.end_stage}")
    if args.start_from_etl:
        logger.info(f"Mode: ETL-ONLY (skip raw data extraction)")
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
            df_clean = stage_5_prep_ml6(ctx, df)
        else:
            clean_path = ctx.joined_dir / "features_ml6_clean.csv"
            df_clean = pd.read_csv(clean_path) if clean_path.exists() else None
        
        if df_clean is not None:
            if args.start_stage <= 6 <= args.end_stage:
                if not stage_6_ml6(ctx, df_clean):
                    success = False
                
                # Run ML6-Extended after standard ML6
                stage_6_extended(ctx)
            
            if args.start_stage <= 7 <= args.end_stage:
                if not stage_7_ml7(ctx):
                    success = False
                
                # Run ML7-Extended after standard ML7
                stage_7_extended(ctx)
            
            if args.start_stage <= 8 <= args.end_stage:
                if not stage_8_tflite(ctx):
                    success = False
    
    if args.start_stage <= 9 <= args.end_stage:
        stages_str = f"{args.start_stage}-{args.end_stage}"
        if not stage_9_report(ctx, stages_executed=stages_str):
            success = False
    
    # Summary
    banner("PIPELINE COMPLETE")
    if success:
        logger.info("[OK] All stages successful")
        logger.info(f"[OK] Output: {ctx.snapshot_dir}")
        report_info = ctx.results.get('stage_9', {}).get('report_path', 'docs/reports/')
        logger.info(f"[OK] Report: {report_info}")
        sys.exit(0)
    else:
        logger.error("✗ Some stages failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
