"""
Stage 3: Apply PBSI Labels
Apply PBSI mood labels to expanded daily dataset.

Input: features_daily_unified.csv (from Stage 2)
       config/label_rules.yaml (PBSI mapping rules)
Output: features_daily_labeled.csv with label_3cls (unstable=-1, neutral=0, stable=+1)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
import yaml
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PBSILabeler:
    """Apply PBSI mood labels to daily data."""
    
    def __init__(self, label_rules_path: str = "config/label_rules.yaml"):
        """Initialize with PBSI label rules."""
        self.label_rules_path = Path(label_rules_path)
        self.rules = self._load_rules()
        logger.info(f"[Labels] Loaded PBSI rules from {label_rules_path}")
    
    def _load_rules(self) -> Dict:
        """Load PBSI label mapping rules from YAML."""
        if not self.label_rules_path.exists():
            logger.warning(f"[Labels] Rules file not found: {self.label_rules_path}")
            # Return default simple rules
            return {
                "unstable": {"pbsi_score_min": 0, "pbsi_score_max": 33},
                "neutral": {"pbsi_score_min": 33, "pbsi_score_max": 66},
                "stable": {"pbsi_score_min": 66, "pbsi_score_max": 100}
            }
        
        with open(self.label_rules_path, "r") as f:
            rules = yaml.safe_load(f)
        
        return rules
    
    def _calculate_pbsi_score(self, row: pd.Series) -> float:
        """
        Calculate PBSI score from daily metrics.
        PBSI = Psychosocial Behavioral Stability Index
        
        Simple formula: weighted combination of sleep quality and activity
        - Sleep quality matters more for stability
        - Activity level also important
        - Heart rate variability (if available) indicates stress
        
        Scale: 0-100 (0=unstable, 100=stable)
        """
        
        sleep_quality = row.get("sleep_quality_score", 0) or 0
        sleep_hours = row.get("sleep_hours", 0) or 0
        
        # Normalize sleep hours (target: 7-8 hours)
        sleep_norm = min(sleep_hours / 8 * 100, 100) if sleep_hours > 0 else 0
        
        # Activity level (normalize steps)
        steps = row.get("total_steps", 0) or 0
        activity_norm = min(steps / 8000 * 100, 100) if steps > 0 else 0
        
        # Heart rate variability (lower stress = higher score)
        hr_std = row.get("hr_std", 0) or 0
        # Lower HR std is better (less stress), normalize inversely
        # Typical HR std: 0-50; we want std<20 to give high score
        hr_norm = max(100 - (hr_std * 2.5), 0) if hr_std >= 0 else 50
        
        # Weighted combination
        # Sleep quality: 40%, Sleep hours: 25%, Activity: 20%, HR stability: 15%
        pbsi_score = (
            sleep_quality * 0.40 +
            sleep_norm * 0.25 +
            activity_norm * 0.20 +
            hr_norm * 0.15
        )
        
        return max(0, min(pbsi_score, 100))  # Clamp to 0-100
    
    def apply_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PBSI labels to dataframe.
        
        Adds columns:
        - pbsi_score: 0-100 score
        - pbsi_quality: description of mood (unstable/neutral/stable)
        - label_3cls: -1 (unstable), 0 (neutral), +1 (stable)
        """
        logger.info(f"[Labels] Calculating PBSI scores for {len(df)} days...")
        
        # Calculate PBSI score
        df["pbsi_score"] = df.apply(self._calculate_pbsi_score, axis=1)
        
        # Map to categories
        def score_to_label(score):
            if score < 33:
                return -1  # unstable
            elif score < 66:
                return 0   # neutral
            else:
                return 1   # stable
        
        def score_to_quality(score):
            if score < 33:
                return "unstable"
            elif score < 66:
                return "neutral"
            else:
                return "stable"
        
        df["pbsi_quality"] = df["pbsi_score"].apply(score_to_quality)
        df["label_3cls"] = df["pbsi_score"].apply(score_to_label)
        
        logger.info(f"[Labels] PBSI scores applied")
        logger.info(f"[Labels] Score range: {df['pbsi_score'].min():.1f} to {df['pbsi_score'].max():.1f}")
        logger.info(f"[Labels] Distribution:")
        for label in [-1, 0, 1]:
            count = (df["label_3cls"] == label).sum()
            pct = count / len(df) * 100
            logger.info(f"  Label {label:+2d}: {count:4d} days ({pct:5.1f}%)")
        
        return df


def run_apply_labels(participant: str = "P000001",
                     snapshot: Optional[str] = None,
                     etl_dir: str = "data/etl",
                     label_rules: str = "config/label_rules.yaml") -> pd.DataFrame:
    """
    Execute Stage 3: Apply PBSI Labels
    
    Takes unified daily data and applies PBSI mood labels.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date; if None, uses today
        etl_dir: ETL data directory
        label_rules: Path to label rules YAML
    
    Returns:
        Labeled dataframe with columns: ..., pbsi_score, pbsi_quality, label_3cls
    """
    
    if snapshot is None:
        snapshot = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STAGE 3: APPLY PBSI LABELS")
    logger.info(f"Participant: {participant}, Snapshot: {snapshot}")
    logger.info(f"{'='*80}\n")
    
    # Load unified data
    input_path = Path(etl_dir) / participant / snapshot / "joined" / "features_daily_unified.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Unified data not found: {input_path}")
    
    logger.info(f"[Labels] Loading: {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"[Labels] Loaded {len(df)} days")
    
    # Apply labels
    labeler = PBSILabeler(label_rules_path=label_rules)
    df = labeler.apply_labels(df)
    
    # Save labeled data
    output_dir = Path(etl_dir) / participant / snapshot / "joined"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "features_daily_labeled.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n[OK] Saved labeled data: {output_path}")
    logger.info(f"[OK] Records: {len(df)}, Columns: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    participant = sys.argv[1] if len(sys.argv) > 1 else "P000001"
    snapshot = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = run_apply_labels(participant=participant, snapshot=snapshot)
    
    print(f"\n[DONE] Apply Labels complete: {len(df)} days with labels")
