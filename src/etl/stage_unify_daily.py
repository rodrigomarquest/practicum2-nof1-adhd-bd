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
        """Initialize with paths to extracted Apple/Zepp data.
        
        CANONICAL PATHS (no intermediate PID directory):
        - data/etl/<PID>/<SNAPSHOT>/extracted/apple/daily_*.csv
        - data/etl/<PID>/<SNAPSHOT>/extracted/zepp/daily_*.csv
        """
        self.extracted_dir = Path(extracted_dir)
        self.participant = participant
        # CANONICAL paths - NO /participant/ subdirectory
        self.apple_dir = self.extracted_dir / "apple"
        self.zepp_dir = self.extracted_dir / "zepp"
        
        logger.info(f"[Unify] Initializing for {participant}")
        logger.info(f"[Unify] Apple dir: {self.apple_dir}")
        logger.info(f"[Unify] Zepp dir: {self.zepp_dir}")
    
    def _load_metric(self, source: str, metric: str) -> pd.DataFrame:
        """Load daily metric from Apple or Zepp.
        
        NO FALLBACK READING - only canonical paths.
        """
        if source == "apple":
            path = self.apple_dir / f"daily_{metric}.csv"
        else:
            path = self.zepp_dir / f"daily_{metric}.csv"
        
        if not path.exists():
            logger.warning(f"[WARN] Missing {source} daily_{metric} at canonical path: {path}")
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
    
    def unify_meds(self) -> pd.DataFrame:
        """Unify medication data from Apple Health with vendor tracking.
        
        Medication data comes from Apple Health exports:
        - apple_export: Official export.xml (ClinicalRecords)
        - apple_autoexport: Auto Export app (Medications-*.csv)
        
        The meds domain uses source prioritization to select the best available
        vendor. The selected vendor is recorded in the `med_vendor` column.
        
        Output columns:
        - date: YYYY-MM-DD
        - med_any: 0/1 (any medication event on that day)
        - med_event_count: integer count of medication events
        - med_dose_total: sum of dosages (float, may be NaN)
        - med_names: comma-separated unique medication names (optional)
        - med_sources: comma-separated unique source apps (optional)
        - med_vendor: vendor selected by prioritizer (e.g., "apple_autoexport")
        """
        logger.info("\n[Unify] === MEDS DATA ===")
        
        df_meds = self._load_metric("apple", "meds")
        
        if len(df_meds) > 0:
            # Keep essential columns only
            output_cols = ["date", "med_any", "med_event_count"]
            # Add optional columns if present
            for opt_col in ["med_dose_total", "med_names", "med_sources", "med_vendor"]:
                if opt_col in df_meds.columns:
                    output_cols.append(opt_col)
            
            df_meds = df_meds[output_cols].drop_duplicates(subset=["date"])
            
            # Log vendor info if available
            if "med_vendor" in df_meds.columns:
                vendor = df_meds["med_vendor"].iloc[0] if len(df_meds) > 0 else "unknown"
                logger.info(f"[Unify] Using meds: {len(df_meds)} days (vendor: {vendor})")
            else:
                logger.info(f"[Unify] Using meds: {len(df_meds)} days")
            
            return df_meds.sort_values("date").reset_index(drop=True)
        else:
            logger.info("[Unify] No meds data found (this is normal if no medication records)")
            return pd.DataFrame()
    
    def unify_som(self) -> pd.DataFrame:
        """Unify State of Mind (mood) data with vendor tracking.
        
        SoM data currently comes only from Apple Auto Export app 
        (apple/autoexport/StateOfMind-*.csv). The vendor is tracked
        for consistency with other domains that use source prioritization.
        
        Output columns:
        - date: YYYY-MM-DD
        - som_mean_score: mean valence [-1, +1] for the day
        - som_last_score: last recorded valence for the day
        - som_n_entries: count of mood entries
        - som_category_3class: discrete -1/0/+1 based on mean thresholds
        - som_kind_dominant: most frequent Kind (Daily Mood / Momentary Emotion)
        - som_labels: comma-separated emotion labels (optional)
        - som_associations: comma-separated context associations (optional)
        - som_vendor: data source vendor (currently always "apple_autoexport")
        """
        logger.info("\n[Unify] === SOM (STATE OF MIND) DATA ===")
        
        df_som = self._load_metric("apple", "som")
        
        if len(df_som) > 0:
            # Keep essential columns only
            output_cols = ["date", "som_mean_score", "som_last_score", "som_n_entries", "som_category_3class"]
            # Add optional columns if present
            for opt_col in ["som_kind_dominant", "som_labels", "som_associations"]:
                if opt_col in df_som.columns:
                    output_cols.append(opt_col)
            
            # Filter to existing columns only
            output_cols = [c for c in output_cols if c in df_som.columns]
            
            df_som = df_som[output_cols].drop_duplicates(subset=["date"])
            
            # Add vendor tracking - currently only apple_autoexport provides SoM
            df_som["som_vendor"] = "apple_autoexport"
            
            logger.info(f"[Unify] Using SoM: {len(df_som)} days (vendor: apple_autoexport)")
            return df_som.sort_values("date").reset_index(drop=True)
        else:
            logger.info("[Unify] No SoM data found (this is normal if no State of Mind records)")
            return pd.DataFrame()
    
    def unify_all(self) -> pd.DataFrame:
        """
        Unify all metrics into a single daily dataframe.
        
        Output columns:
        - date
        - sleep_hours, sleep_quality_score
        - hr_mean, hr_min, hr_max, hr_std, hr_samples
        - total_steps, total_distance, total_active_energy
        - med_any, med_event_count (if medication data available)
        - som_mean_score, som_last_score, som_n_entries, som_category_3class (if SoM data available)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 2: Unify Daily - {self.participant}")
        logger.info(f"{'='*80}\n")
        
        df_sleep = self.unify_sleep()
        df_cardio = self.unify_cardio()
        df_activity = self.unify_activity()
        df_meds = self.unify_meds()
        df_som = self.unify_som()
        
        # Merge all on date (outer join to preserve all dates)
        df_unified = pd.DataFrame(columns=["date"])
        
        all_dates = set()
        for df in [df_sleep, df_cardio, df_activity, df_meds, df_som]:
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
        if len(df_meds) > 0:
            df_unified = df_unified.merge(df_meds, on="date", how="left")
        if len(df_som) > 0:
            df_unified = df_unified.merge(df_som, on="date", how="left")
        
        # NOTE (v4.1.5): Forward-fill removed for scientific integrity.
        # Missing values are kept as NaN to avoid inventing sleep/cardio/activity data.
        # Previous behavior: df_unified[numeric_cols].fillna(method="ffill").fillna(method="bfill")
        # New behavior: Preserve NaN to represent true uncertainty/missingness.
        logger.info(f"[Unify] Preserving NaN values (no forward-fill)")
        
        # Log missing data statistics per column
        numeric_cols = df_unified.select_dtypes(include=np.number).columns
        missing_stats = []
        for col in numeric_cols:
            missing_count = df_unified[col].isna().sum()
            missing_pct = 100 * missing_count / len(df_unified)
            if missing_count > 0:
                missing_stats.append(f"  {col}: {missing_count}/{len(df_unified)} ({missing_pct:.1f}%)")
        
        if missing_stats:
            logger.info(f"[Unify] Missing value summary:")
            for stat in missing_stats:
                logger.info(stat)
        else:
            logger.info(f"[Unify] No missing values in numeric columns")
        
        logger.info(f"\n[Unify] === UNIFIED DATASET ===")
        logger.info(f"[Unify] Total days: {len(df_unified)}")
        logger.info(f"[Unify] Date range: {df_unified['date'].min()} to {df_unified['date'].max()}")
        logger.info(f"[Unify] Columns: {list(df_unified.columns)}")

        
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
