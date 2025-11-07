"""
Stage 2: Unify Daily
Merge Apple + Zepp daily CSVs into a unified daily dataset with period expansion.

Input: data/extracted/{apple|zepp}/P000001/daily_{sleep|cardio|activity}.csv
Output: data/etl/P000001/SNAPSHOT/joined/features_daily_unified.csv (with expanded dates)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class DailyUnifier:
    """Merge Apple + Zepp daily metrics into unified daily dataset."""
    
    def __init__(self, extracted_dir: str = "data/extracted", participant: str = "P000001"):
        """Initialize with paths to extracted Apple/Zepp data."""
        self.extracted_dir = Path(extracted_dir)
        self.participant = participant
        self.apple_dir = self.extracted_dir / "apple" / participant
        self.zepp_dir = self.extracted_dir / "zepp" / participant
        
        logger.info(f"[Unify] Initializing for {participant}")
        logger.info(f"[Unify] Apple dir: {self.apple_dir}")
        logger.info(f"[Unify] Zepp dir: {self.zepp_dir}")
    
    def _load_metric(self, source: str, metric: str) -> pd.DataFrame:
        """Load daily metric from Apple or Zepp."""
        if source == "apple":
            path = self.apple_dir / f"daily_{metric}.csv"
        else:
            path = self.zepp_dir / f"daily_{metric}.csv"
        
        if not path.exists():
            logger.warning(f"[Unify] File not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["source"] = source.upper()
        logger.info(f"[Unify] Loaded {source} {metric}: {len(df)} days from {path}")
        return df
    
    def unify_sleep(self) -> pd.DataFrame:
        """Unify sleep data from Apple and Zepp."""
        logger.info("\n[Unify] === SLEEP DATA ===")
        
        df_apple = self._load_metric("apple", "sleep")
        df_zepp = self._load_metric("zepp", "sleep")
        
        # Merge on date, preferring Apple data when both exist
        if len(df_apple) > 0 and len(df_zepp) > 0:
            df_unified = df_apple.copy()
            # Fill missing dates with Zepp data
            missing_dates = set(df_zepp["date"]) - set(df_apple["date"])
            if missing_dates:
                df_zepp_missing = df_zepp[df_zepp["date"].isin(missing_dates)]
                df_unified = pd.concat([df_unified, df_zepp_missing], ignore_index=True)
            logger.info(f"[Unify] Merged sleep: {len(df_apple)} Apple + {len(df_zepp_missing)} Zepp = {len(df_unified)} total")
        elif len(df_apple) > 0:
            df_unified = df_apple
            logger.info(f"[Unify] Using Apple sleep only: {len(df_unified)} days")
        elif len(df_zepp) > 0:
            df_unified = df_zepp
            logger.info(f"[Unify] Using Zepp sleep only: {len(df_unified)} days")
        else:
            logger.warning("[Unify] No sleep data found!")
            return pd.DataFrame()
        
        # Rename columns for consistency
        df_unified = df_unified[["date", "sleep_hours", "sleep_quality_score"]].drop_duplicates(subset=["date"])
        
        return df_unified.sort_values("date").reset_index(drop=True)
    
    def unify_cardio(self) -> pd.DataFrame:
        """Unify heart rate data from Apple and Zepp."""
        logger.info("\n[Unify] === CARDIO DATA ===")
        
        df_apple = self._load_metric("apple", "cardio")
        df_zepp = self._load_metric("zepp", "cardio")
        
        # For cardio, we aggregate Apple + Zepp on same day
        if len(df_apple) > 0 and len(df_zepp) > 0:
            # Group by date and aggregate
            df_combined = pd.concat([df_apple, df_zepp], ignore_index=True)
            
            # Average the metrics for days with both sources
            agg_dict = {}
            for col in ["hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples"]:
                if col in df_combined.columns:
                    agg_dict[col] = "mean"
            
            df_unified = df_combined.groupby("date").agg(agg_dict).reset_index()
            logger.info(f"[Unify] Merged cardio: {len(df_apple)} Apple + {len(df_zepp)} Zepp = {len(df_unified)} unique days")
        
        elif len(df_apple) > 0:
            df_unified = df_apple[["date", "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples"]]
            logger.info(f"[Unify] Using Apple cardio only: {len(df_unified)} days")
        
        elif len(df_zepp) > 0:
            df_unified = df_zepp[["date", "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples"]]
            logger.info(f"[Unify] Using Zepp cardio only: {len(df_unified)} days")
        
        else:
            logger.warning("[Unify] No cardio data found!")
            return pd.DataFrame()
        
        return df_unified.sort_values("date").reset_index(drop=True)
    
    def unify_activity(self) -> pd.DataFrame:
        """Unify activity data from Apple and Zepp."""
        logger.info("\n[Unify] === ACTIVITY DATA ===")
        
        df_apple = self._load_metric("apple", "activity")
        df_zepp = self._load_metric("zepp", "activity")
        
        # For activity, sum the values from both sources
        if len(df_apple) > 0 and len(df_zepp) > 0:
            df_combined = pd.concat([df_apple, df_zepp], ignore_index=True)
            
            agg_dict = {
                "total_steps": "sum",
                "total_distance": "sum",
                "total_active_energy": "sum"
            }
            
            df_unified = df_combined.groupby("date").agg(agg_dict).reset_index()
            logger.info(f"[Unify] Merged activity: {len(df_apple)} Apple + {len(df_zepp)} Zepp = {len(df_unified)} unique days")
        
        elif len(df_apple) > 0:
            df_unified = df_apple[["date", "total_steps", "total_distance", "total_active_energy"]]
            logger.info(f"[Unify] Using Apple activity only: {len(df_unified)} days")
        
        elif len(df_zepp) > 0:
            df_unified = df_zepp[["date", "total_steps", "total_distance", "total_active_energy"]]
            logger.info(f"[Unify] Using Zepp activity only: {len(df_unified)} days")
        
        else:
            logger.warning("[Unify] No activity data found!")
            return pd.DataFrame()
        
        return df_unified.sort_values("date").reset_index(drop=True)
    
    def unify_all(self) -> pd.DataFrame:
        """
        Unify all metrics into a single daily dataframe.
        
        Output columns:
        - date
        - sleep_hours, sleep_quality_score
        - hr_mean, hr_min, hr_max, hr_std, hr_samples
        - total_steps, total_distance, total_active_energy
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 2: Unify Daily - {self.participant}")
        logger.info(f"{'='*80}\n")
        
        df_sleep = self.unify_sleep()
        df_cardio = self.unify_cardio()
        df_activity = self.unify_activity()
        
        # Merge all on date (outer join to preserve all dates)
        df_unified = pd.DataFrame(columns=["date"])
        
        all_dates = set()
        for df in [df_sleep, df_cardio, df_activity]:
            if len(df) > 0:
                all_dates.update(df["date"].unique())
        
        df_unified["date"] = sorted(list(all_dates))
        
        # Left merge with each metric
        if len(df_sleep) > 0:
            df_unified = df_unified.merge(df_sleep, on="date", how="left")
        if len(df_cardio) > 0:
            df_unified = df_unified.merge(df_cardio, on="date", how="left")
        if len(df_activity) > 0:
            df_unified = df_unified.merge(df_activity, on="date", how="left")
        
        # Forward fill NaN values (common in sensor data)
        numeric_cols = df_unified.select_dtypes(include=np.number).columns
        df_unified[numeric_cols] = df_unified[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        
        logger.info(f"\n[Unify] === UNIFIED DATASET ===")
        logger.info(f"[Unify] Total days: {len(df_unified)}")
        logger.info(f"[Unify] Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        logger.info(f"[Unify] Columns: {list(df_unified.columns)}")
        logger.info(f"[Unify] Missing values:\n{df_unified.isnull().sum()}")
        
        return df_unified


def run_unify_daily(participant: str = "P000001", 
                    snapshot: str = None,
                    extracted_dir: str = "data/extracted",
                    output_dir: str = "data/etl") -> pd.DataFrame:
    """
    Execute Stage 2: Unify Daily
    
    Merge daily aggregated CSVs (from Stage 1.5) into unified dataset.
    
    Args:
        participant: Participant ID (e.g., "P000001")
        snapshot: Snapshot date (e.g., "2025-11-07"); if None, uses today's date
        extracted_dir: Path to extracted data
        output_dir: Output directory for unified data
    
    Returns:
        Unified daily dataframe
    """
    
    if snapshot is None:
        snapshot = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"STAGE 2: UNIFY DAILY")
    logger.info(f"Participant: {participant}, Snapshot: {snapshot}")
    logger.info(f"{'='*80}\n")
    
    # Create unifier
    unifier = DailyUnifier(extracted_dir=extracted_dir, participant=participant)
    
    # Unify all metrics
    df_unified = unifier.unify_all()
    
    # Save to output
    output_path = Path(output_dir) / "joined"
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_path / "features_daily_unified.csv"
    df_unified.to_csv(csv_path, index=False)
    
    logger.info(f"\n[OK] Saved unified data: {csv_path}")
    logger.info(f"[OK] Records: {len(df_unified)}, Columns: {len(df_unified.columns)}")
    
    return df_unified


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
    
    df = run_unify_daily(participant=participant, snapshot=snapshot)
    
    print(f"\n[DONE] Unify Daily complete: {len(df)} days with {len(df.columns)} features")
