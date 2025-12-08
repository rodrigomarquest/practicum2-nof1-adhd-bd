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
from tqdm import tqdm

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
        
        file_size_mb = self.xml_path.stat().st_size / (1024 * 1024)
        logger.info(f"[Apple] Loading export.xml: {xml_path}")
        logger.info(f"[Apple] File size: {file_size_mb:.1f} MB - Parsing XML structure...")
        
        import time
        import threading
        
        start_time = time.time()
        
        # Create a progress bar that updates while parsing
        # Estimate based on file size: ~1MB per second for XML parsing
        estimated_seconds = max(int(file_size_mb / 1.0), 10)
        
        parsing_done = threading.Event()
        
        def show_progress():
            """Show progress bar while XML is being parsed."""
            with tqdm(total=estimated_seconds, desc="[Apple] Parsing XML", 
                     unit="s", ncols=100, leave=False, 
                     bar_format='{l_bar}{bar}| {elapsed} elapsed') as pbar:
                while not parsing_done.is_set():
                    time.sleep(1)
                    if pbar.n < estimated_seconds:
                        pbar.update(1)
        
        # Start progress bar in background thread
        progress_thread = threading.Thread(target=show_progress, daemon=True)
        progress_thread.start()
        
        try:
            # Parse XML (blocking operation)
            self.tree = ET.parse(xml_path)
            self.root = self.tree.getroot()
        finally:
            # Stop progress bar
            parsing_done.set()
            progress_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        logger.info(f"[Apple] Parsed export.xml successfully in {elapsed:.1f}s")
    
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
        Daily: mean, min, max, std HR + sample count.
        
        Uses ultra-fast binary regex streaming with Parquet caching:
        - First run: Parse XML with binary regex (~1-2 min for 1GB)
        - Subsequent runs: Load from Parquet cache (~1-5 seconds)
        - 100-500x faster than standard ET.findall() approach
        
        **Caching Strategy:**
        - Saves TWO Parquet files:
          1. export_apple_hr_events.parquet - event-level (timestamp, hr_value) for QC
          2. export_apple_hr_daily.parquet - pre-aggregated daily (ALL metrics, canonical names)
        
        **CANONICAL COLUMNS** (stored in cache):
        - apple_hr_mean, apple_hr_min, apple_hr_max, apple_hr_std, apple_n_hr
        """
        logger.info("[Apple] Aggregating heart rate data...")
        
        # Setup cache paths
        xml_path = Path(self.xml_path)
        cache_dir = xml_path.parent / ".cache"
        cache_file_daily = cache_dir / f"{xml_path.stem}_apple_hr_daily.parquet"
        cache_file_events = cache_dir / f"{xml_path.stem}_apple_hr_events.parquet"
        
        # Try to load from daily cache first (fastest path)
        if cache_file_daily.exists():
            try:
                df_cached = pd.read_parquet(cache_file_daily)
                logger.info(f"[Apple] ✓ Loaded HR from cache: {cache_file_daily.name} ({len(df_cached)} days)")
                
                # FIXED: Cache now stores ALL metrics with canonical Apple names.
                # Map back to generic names for daily_cardio.csv output.
                if "apple_hr_mean" in df_cached.columns:
                    df_cached = df_cached.rename(columns={
                        "apple_hr_mean": "hr_mean",
                        "apple_hr_min": "hr_min",
                        "apple_hr_max": "hr_max",
                        "apple_hr_std": "hr_std",
                        "apple_n_hr": "hr_samples"
                    })
                
                return df_cached[["date", "hr_mean", "hr_min", "hr_max", "hr_std", "hr_samples"]]
            except Exception as e:
                logger.warning(f"[Apple] Failed to load HR cache: {e}")
                # Continue to parse XML
        
        # Parse XML with ultra-fast binary regex streaming
        logger.info(f"[Apple] Parsing {xml_path.name} with binary regex (100-500x faster than ET.findall)...")
        
        import re
        from datetime import datetime as dt
        
        # HR outlier thresholds (filter biologically implausible values)
        HR_MIN_VALID = 30   # bpm - below this is likely sensor error
        HR_MAX_VALID = 220  # bpm - above this is likely sensor error or extreme exercise
        
        hr_data = {}
        hr_events = []  # Store event-level data for QC
        hr_count = 0
        hr_filtered = 0
        file_size_mb = xml_path.stat().st_size / (1024 * 1024)
        
        # Binary regex patterns
        record_pattern = rb'<Record[^>]*?type="HKQuantityTypeIdentifierHeartRate"[^>]*?>'
        value_pattern = rb'value="([^"]+)"'
        date_pattern = rb'startDate="([^"]+)"'
        
        try:
            logger.info(f"[Apple]   Reading {file_size_mb:.1f} MB...")
            with open(xml_path, 'rb') as f:
                content = f.read()
            
            logger.info(f"[Apple]   Extracting HR records with binary regex...")
            
            # Find all HR record matches first to get total count for progress bar
            hr_matches = list(re.finditer(record_pattern, content))
            total_matches = len(hr_matches)
            logger.info(f"[Apple]   Found {total_matches:,} HR record tags to process...")
            
            # Process with progress bar
            with tqdm(total=total_matches, desc="[Apple] Parsing HR records", 
                     unit="records", ncols=100, leave=False) as pbar:
                for record_match in hr_matches:
                    record_tag = record_match.group(0)
                    
                    # Extract value
                    val_match = re.search(value_pattern, record_tag)
                    if not val_match:
                        pbar.update(1)
                        continue
                    
                    # Extract startDate
                    date_match = re.search(date_pattern, record_tag)
                    if not date_match:
                        pbar.update(1)
                        continue
                    
                    try:
                        hr_value = float(val_match.group(1))
                        timestamp_str = date_match.group(1).decode('utf-8', errors='ignore')
                        date_str = timestamp_str[:10]  # YYYY-MM-DD
                        
                        # TASK C: Filter outliers (biologically implausible HR values)
                        if not (HR_MIN_VALID <= hr_value <= HR_MAX_VALID):
                            hr_filtered += 1
                            logger.debug(f"Filtered outlier HR: {hr_value} bpm on {date_str}")
                            pbar.update(1)
                            continue
                        
                        # Store event-level data (for QC re-aggregation)
                        hr_events.append({
                            "timestamp": timestamp_str,
                            "date": date_str,
                            "hr_value": hr_value
                        })
                        
                        if date_str not in hr_data:
                            hr_data[date_str] = {
                                "date": date_str,
                                "hr_values": []
                            }
                        
                        hr_data[date_str]["hr_values"].append(hr_value)
                        hr_count += 1
                    except (ValueError, TypeError):
                        pass
                    
                    pbar.update(1)
            
            if hr_filtered > 0:
                logger.info(f"[Apple]   ✓ Filtered {hr_filtered} outlier HR values ({hr_filtered/(hr_count+hr_filtered)*100:.2f}%)")
            logger.info(f"[Apple]   ✓ Parsed {hr_count} valid HR records into {len(hr_data)} days")
        
        except Exception as e:
            logger.warning(f"[Apple] Binary regex failed: {e}, falling back to ET.findall()...")
            # Fallback to original method (with same outlier filtering)
            HR_MIN_VALID = 30
            HR_MAX_VALID = 220
            hr_filtered = 0
            
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
                    
                    # TASK C: Filter outliers
                    if not (HR_MIN_VALID <= hr_value <= HR_MAX_VALID):
                        hr_filtered += 1
                        continue
                    
                except:
                    continue
                
                date_str = self._get_date_from_dt(start_dt)
                timestamp_str = record.get("startDate") or ""
                
                # Store event-level data
                hr_events.append({
                    "timestamp": timestamp_str,
                    "date": date_str,
                    "hr_value": hr_value
                })
                
                if date_str not in hr_data:
                    hr_data[date_str] = {
                        "date": date_str,
                        "hr_values": []
                    }
                
                hr_data[date_str]["hr_values"].append(hr_value)
            
            if hr_filtered > 0:
                logger.info(f"[Apple] Filtered {hr_filtered} outlier HR values in fallback mode")
        
        # Aggregate to daily stats
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
        
        # Save BOTH caches for next run (if pyarrow available)
        if not df_hr.empty:
            try:
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                # 1. Save event-level cache (for QC re-aggregation)
                if hr_events:
                    df_events = pd.DataFrame(hr_events)
                    df_events.to_parquet(cache_file_events, index=False, compression="snappy")
                    logger.info(f"[Apple] ✓ Cached event-level HR to {cache_file_events.name} ({len(df_events)} records, {cache_file_events.stat().st_size / 1024:.1f} KB)")
                
                # 2. Save daily aggregated cache (for fast loading)
                df_cache = df_hr.copy()
                df_cache = df_cache.rename(columns={
                    "hr_mean": "apple_hr_mean",
                    "hr_min": "apple_hr_min",
                    "hr_max": "apple_hr_max",
                    "hr_std": "apple_hr_std",
                    "hr_samples": "apple_n_hr"
                })
                df_cache = df_cache[["date", "apple_hr_mean", "apple_hr_min", "apple_hr_max", "apple_hr_std", "apple_n_hr"]]
                
                df_cache.to_parquet(cache_file_daily, index=False, compression="snappy")
                logger.info(f"[Apple] ✓ Cached daily HR to {cache_file_daily.name} ({cache_file_daily.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                logger.warning(f"[Apple] Failed to save HR cache: {e}")
        
        return df_hr
    
    def aggregate_activity(self) -> pd.DataFrame:
        """
        Aggregate activity data (steps, distance, active energy burned).
        Daily totals.
        """
        logger.info("[Apple] Aggregating activity data...")
        
        activity_data = {}
        
        # Get all activity records first for progress bar
        activity_records = [
            r for r in self.root.findall(".//Record") 
            if r.get("type") in [self.TYPE_STEPS, self.TYPE_DISTANCE, self.TYPE_ACTIVE_ENERGY]
        ]
        
        logger.info(f"[Apple] Found {len(activity_records):,} activity records to process...")
        
        with tqdm(total=len(activity_records), desc="[Apple] Aggregating activity", 
                 unit="records", ncols=100, leave=False) as pbar:
            for record in activity_records:
                rec_type = record.get("type")
                start_dt = self._parse_datetime(record.get("startDate"))
                value = record.get("value")
                
                if start_dt is None or value is None:
                    pbar.update(1)
                    continue
                
                try:
                    val = float(value)
                except:
                    pbar.update(1)
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
                
                pbar.update(1)
        
        df_activity = pd.DataFrame(list(activity_data.values()))
        logger.info(f"[Apple] Aggregated {len(df_activity)} activity days")
        return df_activity
    
    def aggregate_meds(self) -> pd.DataFrame:
        """
        Aggregate medication records from Apple Health.
        
        Delegates to the meds domain module for actual parsing.
        Returns empty well-formed DataFrame if no medication data exists.
        
        Daily columns:
        - date (YYYY-MM-DD)
        - med_any (0/1)
        - med_event_count (int)
        - med_names (comma-separated)
        - med_sources (comma-separated)
        """
        logger.info("[Apple] Aggregating medication data...")
        
        try:
            # Import meds domain module
            from src.domains.meds.meds_from_extracted import MedsAggregator, _empty_meds_daily
        except ImportError:
            try:
                from domains.meds.meds_from_extracted import MedsAggregator, _empty_meds_daily
            except ImportError:
                logger.warning("[Apple] Meds domain module not found - skipping medication aggregation")
                # Return empty well-formed DataFrame
                return pd.DataFrame({
                    "date": pd.Series(dtype="str"),
                    "med_any": pd.Series(dtype="int"),
                    "med_event_count": pd.Series(dtype="int"),
                    "med_names": pd.Series(dtype="str"),
                    "med_sources": pd.Series(dtype="str"),
                })
        
        try:
            # Use the same XML path as this aggregator
            meds_aggregator = MedsAggregator(self.xml_path)
            df_meds = meds_aggregator.aggregate_medications_binary_regex()
            logger.info(f"[Apple] Aggregated {len(df_meds)} medication days")
            return df_meds
        except Exception as e:
            logger.warning(f"[Apple] Medication aggregation failed: {e}")
            return _empty_meds_daily()
    
    def aggregate_all(self) -> Dict[str, pd.DataFrame]:
        """Aggregate all metrics and return dict with daily CSVs ready to save."""
        return {
            "daily_sleep": self.aggregate_sleep(),
            "daily_cardio": self.aggregate_heartrate(),
            "daily_activity": self.aggregate_activity(),
            "daily_meds": self.aggregate_meds()
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
        
        # TASK C: Filter outliers (same as Apple for consistency)
        HR_MIN_VALID = 30
        HR_MAX_VALID = 220
        
        mask_valid = (df["heartRate"] >= HR_MIN_VALID) & (df["heartRate"] <= HR_MAX_VALID)
        hr_filtered = (~mask_valid).sum()
        
        if hr_filtered > 0:
            pct = 100 * hr_filtered / len(df)
            logger.info(f"[Zepp] Filtered {hr_filtered} outlier HR values ({pct:.2f}%)")
        
        df = df[mask_valid]
        
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
            
            # Save to CSV (canonical paths: data/etl/<PID>/<SNAPSHOT>/extracted/apple/daily_*.csv)
            # NO intermediate PID directory!
            apple_output_dir = Path(output_dir) / "apple"
            apple_output_dir.mkdir(parents=True, exist_ok=True)
            
            for metric_name, df in results["apple"].items():
                if len(df) > 0:
                    # Rename existing file to .prev.csv if it exists
                    out_path = apple_output_dir / f"{metric_name}.csv"
                    if out_path.exists():
                        prev_path = apple_output_dir / f"{metric_name}.prev.csv"
                        out_path.rename(prev_path)
                        logger.info(f"[RENAME] {out_path.name} -> {prev_path.name}")
                    
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
            
            # Save to CSV (canonical paths: data/etl/<PID>/<SNAPSHOT>/extracted/zepp/daily_*.csv)
            # NO intermediate PID directory!
            zepp_output_dir = Path(output_dir) / "zepp"
            zepp_output_dir.mkdir(parents=True, exist_ok=True)
            
            for metric_name, df in results["zepp"].items():
                if len(df) > 0:
                    # Rename existing file to .prev.csv if it exists
                    out_path = zepp_output_dir / f"{metric_name}.csv"
                    if out_path.exists():
                        prev_path = zepp_output_dir / f"{metric_name}.prev.csv"
                        out_path.rename(prev_path)
                        logger.info(f"[RENAME] {out_path.name} -> {prev_path.name}")
                    
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
