"""
Stage 1.5: CSV Aggregation
Convert raw extracted data (Apple export.xml + Zepp CSVs) → daily aggregated CSVs

Apple: parse export.xml → daily sleep/cardio/activity metrics
Zepp: read raw CSVs → daily aggregated metrics

Output: data/extracted/{apple|zepp}/P000001/daily_*.csv for unify_daily to consume
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class AppleHealthAggregator:
    """Parse Apple export.xml and aggregate to daily metrics."""
    
    # Apple HealthKit type identifiers (common sleep/cardio/activity types)
    TYPE_SLEEP = "HKCategoryTypeIdentifierSleepAnalysis"
    TYPE_HEARTRATE = "HKQuantityTypeIdentifierHeartRate"
    TYPE_STEPS = "HKQuantityTypeIdentifierStepCount"
    TYPE_DISTANCE = "HKQuantityTypeIdentifierDistanceWalkingRunning"
    TYPE_ACTIVE_ENERGY = "HKQuantityTypeIdentifierActiveEnergyBurned"
    TYPE_RESTING_HR = "HKQuantityTypeIdentifierRestingHeartRate"
    TYPE_HRV = "HKQuantityTypeIdentifierHeartRateVariabilitySDNN"
    TYPE_WORKOUTS = "HKWorkoutTypeIdentifier"
    
    def __init__(self, xml_path: str):
        """Initialize with path to export.xml."""
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"export.xml not found: {xml_path}")
        
        logger.info(f"[Apple] Loading export.xml: {xml_path}")
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        logger.info(f"[Apple] Parsed export.xml successfully")
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse Apple HealthKit datetime format (ISO 8601 with timezone)."""
        # Example: "2024-01-15 10:30:45 +0000"
        try:
            return datetime.fromisoformat(dt_str.replace(" +0000", "+00:00"))
        except:
            try:
                return pd.to_datetime(dt_str)
            except:
                logger.warning(f"Could not parse datetime: {dt_str}")
                return None
    
    def _get_date_from_dt(self, dt: datetime) -> str:
        """Extract date string (YYYY-MM-DD) from datetime."""
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d")
    
    def aggregate_sleep(self) -> pd.DataFrame:
        """
        Aggregate sleep data from HKCategoryTypeIdentifierSleepAnalysis.
        Records indicate "asleep" (value=0) or "in bed" (value=1).
        
        Daily aggregation: total minutes in each state.
        """
        logger.info("[Apple] Aggregating sleep data...")
        
        sleep_data = {}
        
        for record in self.root.findall(".//Record"):
            rec_type = record.get("type")
            if rec_type != self.TYPE_SLEEP:
                continue
            
            start_dt = self._parse_datetime(record.get("startDate"))
            end_dt = self._parse_datetime(record.get("endDate"))
            value = record.get("value")  # "HKCategoryValueSleepAnalysisAsleep" or "InBed"
            
            if start_dt is None or end_dt is None:
                continue
            
            date_str = self._get_date_from_dt(start_dt)
            duration_minutes = (end_dt - start_dt).total_seconds() / 60
            
            if date_str not in sleep_data:
                sleep_data[date_str] = {
                    "date": date_str,
                    "total_sleep_minutes": 0,
                    "in_bed_minutes": 0,
                    "sleep_records_count": 0
                }
            
            if "asleep" in value.lower():
                sleep_data[date_str]["total_sleep_minutes"] += duration_minutes
            elif "bed" in value.lower():
                sleep_data[date_str]["in_bed_minutes"] += duration_minutes
            
            sleep_data[date_str]["sleep_records_count"] += 1
        
        df_sleep = pd.DataFrame(list(sleep_data.values()))
        
        # Convert minutes to hours
        if len(df_sleep) > 0:
            df_sleep["sleep_hours"] = df_sleep["total_sleep_minutes"] / 60
            df_sleep["sleep_quality_score"] = np.where(
                df_sleep["total_sleep_minutes"] > 0,
                (df_sleep["total_sleep_minutes"] / df_sleep["in_bed_minutes"] * 100).clip(0, 100),
                0
            )
        
        logger.info(f"[Apple] Aggregated {len(df_sleep)} sleep days")
        return df_sleep[["date", "sleep_hours", "sleep_quality_score", "total_sleep_minutes"]]
    
    def aggregate_heartrate(self) -> pd.DataFrame:
        """
        Aggregate heart rate data (HKQuantityTypeIdentifierHeartRate).
        Daily: mean, min, max HR.
        """
        logger.info("[Apple] Aggregating heart rate data...")
        
        hr_data = {}
        
        for record in self.root.findall(".//Record"):
            rec_type = record.get("type")
            if rec_type != self.TYPE_HEARTRATE:
                continue
            
            start_dt = self._parse_datetime(record.get("startDate"))
            value = record.get("value")
            
            if start_dt is None or value is None:
                continue
            
            try:
                hr_value = float(value)
            except:
                continue
            
            date_str = self._get_date_from_dt(start_dt)
            
            if date_str not in hr_data:
                hr_data[date_str] = {
                    "date": date_str,
                    "hr_values": []
                }
            
            hr_data[date_str]["hr_values"].append(hr_value)
        
        rows = []
        for date_str, data in hr_data.items():
            if len(data["hr_values"]) > 0:
                rows.append({
                    "date": date_str,
                    "hr_mean": np.mean(data["hr_values"]),
                    "hr_min": np.min(data["hr_values"]),
                    "hr_max": np.max(data["hr_values"]),
                    "hr_std": np.std(data["hr_values"]),
                    "hr_samples": len(data["hr_values"])
                })
        
        df_hr = pd.DataFrame(rows)
        logger.info(f"[Apple] Aggregated {len(df_hr)} HR days")
        return df_hr
    
    def aggregate_activity(self) -> pd.DataFrame:
        """
        Aggregate activity data (steps, distance, active energy burned).
        Daily totals.
        """
        logger.info("[Apple] Aggregating activity data...")
        
        activity_data = {}
        
        for record in self.root.findall(".//Record"):
            rec_type = record.get("type")
            start_dt = self._parse_datetime(record.get("startDate"))
            value = record.get("value")
            
            if start_dt is None or value is None:
                continue
            
            try:
                val = float(value)
            except:
                continue
            
            date_str = self._get_date_from_dt(start_dt)
            
            if date_str not in activity_data:
                activity_data[date_str] = {
                    "date": date_str,
                    "total_steps": 0,
                    "total_distance": 0,
                    "total_active_energy": 0
                }
            
            if rec_type == self.TYPE_STEPS:
                activity_data[date_str]["total_steps"] += val
            elif rec_type == self.TYPE_DISTANCE:
                activity_data[date_str]["total_distance"] += val
            elif rec_type == self.TYPE_ACTIVE_ENERGY:
                activity_data[date_str]["total_active_energy"] += val
        
        df_activity = pd.DataFrame(list(activity_data.values()))
        logger.info(f"[Apple] Aggregated {len(df_activity)} activity days")
        return df_activity
    
    def aggregate_all(self) -> Dict[str, pd.DataFrame]:
        """Aggregate all metrics and return dict with daily CSVs ready to save."""
        return {
            "daily_sleep": self.aggregate_sleep(),
            "daily_cardio": self.aggregate_heartrate(),
            "daily_activity": self.aggregate_activity()
        }


class ZeppHealthAggregator:
    """Parse Zepp raw CSVs and aggregate to daily metrics."""
    
    def __init__(self, zepp_dir: str):
        """Initialize with path to Zepp extracted directory."""
        self.zepp_dir = Path(zepp_dir)
        if not self.zepp_dir.exists():
            raise FileNotFoundError(f"Zepp directory not found: {zepp_dir}")
        
        logger.info(f"[Zepp] Scanning directory: {zepp_dir}")
        
        # Locate main CSV files
        self.sleep_csv = self._find_csv("SLEEP")
        self.heartrate_csv = self._find_csv("HEARTRATE")
        self.activity_csv = self._find_csv("ACTIVITY")
        
        # If nothing found in primary dir, try parent "unknown" (fallback mechanism)
        if self.sleep_csv is None and self.heartrate_csv is None and self.activity_csv is None:
            unknown_dir = self.zepp_dir.parent / "unknown"
            if unknown_dir.exists() and unknown_dir != self.zepp_dir:
                logger.info(f"[Zepp] No data in {self.zepp_dir}, trying fallback: {unknown_dir}")
                self.zepp_dir = unknown_dir
                self.sleep_csv = self._find_csv("SLEEP")
                self.heartrate_csv = self._find_csv("HEARTRATE")
                self.activity_csv = self._find_csv("ACTIVITY")
        
        logger.info(f"[Zepp] Found files: SLEEP={self.sleep_csv is not None}, "
                   f"HR={self.heartrate_csv is not None}, ACT={self.activity_csv is not None}")
    
    def _find_csv(self, pattern: str) -> Optional[Path]:
        """Find CSV file matching pattern (e.g., 'SLEEP', 'HEARTRATE')."""
        for csv_path in self.zepp_dir.rglob(f"{pattern}*.csv"):
            return csv_path
        return None
    
    def _parse_zepp_datetime(self, dt_str: str) -> datetime:
        """Parse Zepp datetime format (e.g., '2025-07-28 13:22:57+0000')."""
        try:
            return pd.to_datetime(dt_str)
        except:
            logger.warning(f"Could not parse Zepp datetime: {dt_str}")
            return None
    
    def aggregate_sleep(self) -> pd.DataFrame:
        """
        Zepp SLEEP CSV contains: date, deepSleepTime, shallowSleepTime, wakeTime, start, stop, REMTime, naps
        Daily aggregation: total sleep in hours = (deep + shallow + REM) / 60
        Note: naps column may contain JSON; we read only first 8 columns with nrows parsing
        """
        if self.sleep_csv is None:
            logger.warning("[Zepp] SLEEP CSV not found")
            return pd.DataFrame()
        
        logger.info(f"[Zepp] Reading SLEEP: {self.sleep_csv}")
        
        try:
            # Read with on_bad_lines='skip' to skip malformed rows with embedded JSON
            # Also use engine='python' for more lenient parsing
            df = pd.read_csv(
                self.sleep_csv, 
                encoding='utf-8', 
                engine='python',
                on_bad_lines='skip',
                dtype={'date': str, 'deepSleepTime': float, 'shallowSleepTime': float, 
                       'wakeTime': float, 'REMTime': float}
            )
        except:
            try:
                # Fallback: try latin-1 encoding
                df = pd.read_csv(
                    self.sleep_csv, 
                    encoding='latin-1', 
                    engine='python',
                    on_bad_lines='skip',
                    dtype={'date': str, 'deepSleepTime': float, 'shallowSleepTime': float, 
                           'wakeTime': float, 'REMTime': float}
                )
            except Exception as e:
                logger.error(f"[Zepp] Could not read SLEEP CSV: {e}")
                return pd.DataFrame()
        
        # Zepp has pre-aggregated sleep per day
        if len(df) == 0 or "date" not in df.columns:
            logger.warning("[Zepp] SLEEP CSV empty or missing 'date' column")
            return pd.DataFrame()
        
        df_out = pd.DataFrame()
        df_out["date"] = df["date"].astype(str)
        
        # Total sleep = deep + shallow + REM (in minutes in Zepp, so convert to hours)
        deep = df.get("deepSleepTime", pd.Series([0]*len(df))).fillna(0).astype(float)
        shallow = df.get("shallowSleepTime", pd.Series([0]*len(df))).fillna(0).astype(float)
        rem = df.get("REMTime", pd.Series([0]*len(df))).fillna(0).astype(float)
        wake = df.get("wakeTime", pd.Series([0]*len(df))).fillna(0).astype(float)
        
        df_out["sleep_hours"] = (deep + shallow + rem) / 60
        
        # Avoid division by zero
        denominator = deep + wake
        df_out["sleep_quality_score"] = np.where(
            denominator > 0,
            np.clip((deep / denominator) * 100, 0, 100),
            0
        )
        
        logger.info(f"[Zepp] Aggregated {len(df_out)} sleep days")
        return df_out[["date", "sleep_hours", "sleep_quality_score"]].dropna(subset=["date"])
    
    def aggregate_heartrate(self) -> pd.DataFrame:
        """
        Zepp HEARTRATE CSV contains: time, heartRate
        Daily aggregation: mean, min, max.
        """
        if self.heartrate_csv is None:
            logger.warning("[Zepp] HEARTRATE CSV not found")
            return pd.DataFrame()
        
        logger.info(f"[Zepp] Reading HEARTRATE: {self.heartrate_csv}")
        try:
            df = pd.read_csv(self.heartrate_csv, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(self.heartrate_csv, encoding='latin-1')
            except:
                logger.error(f"[Zepp] Could not read HEARTRATE CSV")
                return pd.DataFrame()
        
        # Parse timestamp and extract date
        df["datetime"] = pd.to_datetime(df["time"])
        df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
        
        # Daily aggregation
        df_daily = df.groupby("date").agg({
            "heartRate": ["mean", "min", "max", "std", "count"]
        }).reset_index()
        
        df_daily.columns = ["date", "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples"]
        
        logger.info(f"[Zepp] Aggregated {len(df_daily)} HR days")
        return df_daily
    
    def aggregate_activity(self) -> pd.DataFrame:
        """
        Zepp ACTIVITY CSV contains: date and activity metrics.
        Daily aggregation: sum steps, calories, etc.
        """
        if self.activity_csv is None:
            logger.warning("[Zepp] ACTIVITY CSV not found")
            return pd.DataFrame()
        
        logger.info(f"[Zepp] Reading ACTIVITY: {self.activity_csv}")
        try:
            df = pd.read_csv(self.activity_csv, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(self.activity_csv, encoding='latin-1')
            except:
                logger.error(f"[Zepp] Could not read ACTIVITY CSV")
                return pd.DataFrame()
        
        # Zepp already has daily aggregation
        df_out = pd.DataFrame()
        df_out["date"] = df["date"] if "date" in df.columns else df.get("Date", None)
        
        # Rename columns if they exist
        if "steps" in df.columns or "stepCount" in df.columns:
            col = "steps" if "steps" in df.columns else "stepCount"
            df_out["total_steps"] = df[col]
        else:
            df_out["total_steps"] = 0
        
        if "calories" in df.columns or "caloriesBurned" in df.columns:
            col = "calories" if "calories" in df.columns else "caloriesBurned"
            df_out["total_active_energy"] = df[col]
        else:
            df_out["total_active_energy"] = 0
        
        # Distance (if available)
        if "distance" in df.columns:
            df_out["total_distance"] = df["distance"]
        else:
            df_out["total_distance"] = 0
        
        logger.info(f"[Zepp] Aggregated {len(df_out)} activity days")
        return df_out[["date", "total_steps", "total_distance", "total_active_energy"]].dropna(subset=["date"])
    
    def aggregate_all(self) -> Dict[str, pd.DataFrame]:
        """Aggregate all metrics."""
        return {
            "daily_sleep": self.aggregate_sleep(),
            "daily_cardio": self.aggregate_heartrate(),
            "daily_activity": self.aggregate_activity()
        }


def run_csv_aggregation(participant: str = "P000001", 
                        extracted_dir: str = "data/extracted",
                        output_dir: str = "data/extracted") -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Run full CSV aggregation pipeline for Apple + Zepp.
    
    Outputs:
    {
        "apple": {"daily_sleep": df, "daily_cardio": df, "daily_activity": df},
        "zepp": {"daily_sleep": df, "daily_cardio": df, "daily_activity": df}
    }
    """
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 1.5: CSV Aggregation - {participant}")
    logger.info(f"{'='*80}\n")
    
    results = {}
    
    # ========== APPLE ==========
    try:
        apple_dir = Path(extracted_dir) / "apple" / "apple_health_export"
        xml_path = apple_dir / "export.xml"
        
        if xml_path.exists():
            logger.info(f"\n[Apple] Processing: {xml_path}\n")
            agg_apple = AppleHealthAggregator(str(xml_path))
            results["apple"] = agg_apple.aggregate_all()
            
            # Save to CSV
            apple_output_dir = Path(output_dir) / "apple" / participant
            apple_output_dir.mkdir(parents=True, exist_ok=True)
            
            for metric_name, df in results["apple"].items():
                if len(df) > 0:
                    out_path = apple_output_dir / f"{metric_name}.csv"
                    df.to_csv(out_path, index=False)
                    logger.info(f"[OK] Saved {metric_name}: {out_path} ({len(df)} rows)")
                else:
                    logger.warning(f"[WARN] No data for {metric_name}")
        else:
            logger.warning(f"[SKIP] Apple export.xml not found: {xml_path}")
            results["apple"] = {}
    
    except Exception as e:
        logger.error(f"[ERROR] Apple aggregation failed: {e}", exc_info=True)
        results["apple"] = {}
    
    # ========== ZEPP ==========
    try:
        zepp_dir = Path(extracted_dir) / "zepp"
        
        if zepp_dir.exists():
            logger.info(f"\n[Zepp] Processing: {zepp_dir}\n")
            agg_zepp = ZeppHealthAggregator(str(zepp_dir))
            results["zepp"] = agg_zepp.aggregate_all()
            
            # Save to CSV (always to participant dir for consistency)
            zepp_output_dir = Path(output_dir) / "zepp" / participant
            zepp_output_dir.mkdir(parents=True, exist_ok=True)
            
            for metric_name, df in results["zepp"].items():
                if len(df) > 0:
                    out_path = zepp_output_dir / f"{metric_name}.csv"
                    df.to_csv(out_path, index=False)
                    logger.info(f"[OK] Saved {metric_name}: {out_path} ({len(df)} rows)")
                else:
                    logger.warning(f"[WARN] No data for {metric_name}")
        else:
            logger.warning(f"[SKIP] Zepp directory not found: {zepp_dir}")
            results["zepp"] = {}
    
    except Exception as e:
        logger.error(f"[ERROR] Zepp aggregation failed: {e}", exc_info=True)
        results["zepp"] = {}
    
    # ========== SUMMARY ==========
    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 1.5 Summary: CSV Aggregation Complete")
    logger.info(f"{'='*80}")
    
    for source in ["apple", "zepp"]:
        if source in results:
            for metric_name, df in results[source].items():
                logger.info(f"  {source}/{metric_name}: {len(df)} days")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    participant = sys.argv[1] if len(sys.argv) > 1 else "P000001"
    
    results = run_csv_aggregation(participant=participant)
    
    print(f"\n[DONE] CSV Aggregation complete for {participant}")
