"""Medication domain: Extract and aggregate medication records from Apple Health.

Reads medication-related records from Apple Health export.xml and aggregates
them to one row per calendar day.

Design principles:
1. DETERMINISTIC: Same input always produces same output (fixed seed, sorted outputs)
2. DEFENSIVE: Returns empty well-formed DataFrame if no medication data exists
3. NON-BREAKING: Never fails the global ETL pipeline; logs warnings and continues
4. CONSISTENT: Follows patterns from cardio/sleep/activity domains

Apple Health record types processed:
- HKClinicalTypeIdentifierMedicationRecord (iOS 15+)
- HKCategoryTypeIdentifierMindfulSession (wellness apps)
- Custom medication records from third-party apps

Output columns:
- date: YYYY-MM-DD format (same as other domains)
- med_any: 0 or 1 (binary indicator)
- med_event_count: integer count of events per day
- med_names: comma-separated unique medication names (if available)
- med_sources: comma-separated unique source apps (if available)
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import sys

# Optional tqdm for progress bars with safe fallback
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, *a, **k):
        return it


def _should_show_meds_progress() -> bool:
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


# Import IO guards (consistent with other domains)
try:
    from src.lib.io_guards import write_csv, atomic_backup_write
except ImportError:
    try:
        from lib.io_guards import write_csv, atomic_backup_write
    except ImportError:
        def write_csv(df, path, dry_run=False, backup_name=None):
            """Fallback write_csv if io_guards not available."""
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(p)
        atomic_backup_write = write_csv

# Import common IO utilities (consistent with cardio/sleep/activity domains)
try:
    from ..common.io import etl_snapshot_root
except ImportError:
    try:
        from src.domains.common.io import etl_snapshot_root
    except ImportError:
        def etl_snapshot_root(pid: str, snapshot: str) -> Path:
            """Fallback snapshot root builder."""
            return Path("data") / "etl" / pid / snapshot

# Import ETL pipeline for snapshot resolution
try:
    import src.etl_pipeline as etl
except Exception:
    try:
        import etl_pipeline as etl  # type: ignore
    except Exception:
        etl = None

logger = logging.getLogger("etl.meds")


# ============================================================================
# Progress Bar Helpers (consistent with stage_csv_aggregation.py)
# ============================================================================

def _make_pbar(total: int, desc: str):
    """Create a tqdm progress bar (Git Bash compatible).
    
    Args:
        total: Total number of items to iterate
        desc: Description for the progress bar
        
    Returns:
        tqdm progress bar or passthrough if disabled
    """
    return tqdm(
        total=total,
        desc=desc,
        unit="rec",
        leave=True,
        dynamic_ncols=True,
        disable=not _should_show_meds_progress(),
    )


def _make_bytes_pbar(total_bytes: int, desc: str):
    """Create a byte-based tqdm progress bar.
    
    Args:
        total_bytes: Total bytes to track
        desc: Description for the progress bar
        
    Returns:
        tqdm progress bar with byte formatting
    """
    return tqdm(
        total=total_bytes,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        leave=True,
        dynamic_ncols=True,
        disable=not _should_show_meds_progress(),
    )


# ============================================================================
# Apple Health Medication Record Types
# ============================================================================

# Primary medication record types from Apple Health
MEDICATION_RECORD_TYPES = [
    # Clinical records (iOS 15+)
    "HKClinicalTypeIdentifierMedicationRecord",
    "HKClinicalTypeIdentifierMedicationStatement",
    "HKClinicalTypeIdentifierMedicationOrder",
    "HKClinicalTypeIdentifierMedicationDispense",
    "HKClinicalTypeIdentifierMedicationAdministration",
    "HKClinicalTypeIdentifierImmunizationRecord",
    
    # Supplement/vitamin tracking (some apps use these)
    "HKQuantityTypeIdentifierDietaryVitaminA",
    "HKQuantityTypeIdentifierDietaryVitaminB6",
    "HKQuantityTypeIdentifierDietaryVitaminB12",
    "HKQuantityTypeIdentifierDietaryVitaminC",
    "HKQuantityTypeIdentifierDietaryVitaminD",
    "HKQuantityTypeIdentifierDietaryVitaminE",
    "HKQuantityTypeIdentifierDietaryVitaminK",
    "HKQuantityTypeIdentifierDietaryFolate",
    "HKQuantityTypeIdentifierDietaryBiotin",
    "HKQuantityTypeIdentifierDietaryIron",
    "HKQuantityTypeIdentifierDietaryZinc",
    "HKQuantityTypeIdentifierDietaryMagnesium",
    "HKQuantityTypeIdentifierDietaryCalcium",
    "HKQuantityTypeIdentifierDietaryPotassium",
    "HKQuantityTypeIdentifierDietaryCaffeine",
]

# Additional record types that might contain medication info in metadata
SECONDARY_MEDICATION_TYPES = [
    "HKCategoryTypeIdentifierMindfulSession",  # Meditation/wellness apps
    "HKWorkoutTypeIdentifier",  # Some workout apps log pre-workout supplements
]

# Binary regex patterns for fast XML parsing
CLINICAL_RECORD_PATTERN = rb'<ClinicalRecord[^>]*?type="([^"]*)"[^>]*?>'
RECORD_PATTERN = rb'<Record[^>]*?type="([^"]*)"[^>]*?>'
DATE_PATTERN = rb'(?:startDate|creationDate)="([^"]+)"'
SOURCE_PATTERN = rb'sourceName="([^"]*)"'
DEVICE_PATTERN = rb'device="([^"]*)"'
VALUE_PATTERN = rb'value="([^"]*)"'


# ============================================================================
# Empty DataFrame Schema (for defensive returns)
# ============================================================================

def _empty_meds_daily() -> pd.DataFrame:
    """Return empty DataFrame with canonical medication columns.
    
    This ensures downstream stages always receive a well-formed DataFrame,
    even when no medication data exists in the Apple Health export.
    """
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "med_any": pd.Series(dtype="int"),
        "med_event_count": pd.Series(dtype="int"),
        "med_dose_total": pd.Series(dtype="float"),
        "med_names": pd.Series(dtype="str"),
        "med_sources": pd.Series(dtype="str"),
    })


# ============================================================================
# AutoExport Medications Parser (from Auto Export iOS app)
# ============================================================================

# Import autoexport discovery and source prioritizer
try:
    from ..common.autoexport_discovery import (
        get_autoexport_csv_path,
        get_extracted_autoexport_dir,
    )
    from ..common.source_prioritizer import prioritize_source, log_source_summary
except ImportError:
    try:
        from src.domains.common.autoexport_discovery import (
            get_autoexport_csv_path,
            get_extracted_autoexport_dir,
        )
        from src.domains.common.source_prioritizer import prioritize_source, log_source_summary
    except ImportError:
        get_autoexport_csv_path = None
        get_extracted_autoexport_dir = None
        prioritize_source = None
        log_source_summary = None


def load_autoexport_meds_daily(
    csv_path: Path | str,
    snapshot_date: str,
) -> pd.DataFrame:
    """Load and aggregate medication records from Auto Export app CSV.
    
    Parses the Medications-*.csv exported by the Auto Export iOS app.
    
    Args:
        csv_path: Path to Medications-*.csv file
        snapshot_date: Snapshot date (YYYY-MM-DD) for deterministic filtering.
                       Only records with date <= snapshot_date are included.
        
    Returns:
        DataFrame with daily medication aggregations.
        If no data, returns empty well-formed DataFrame.
        
    Columns in input CSV:
        Date, Scheduled Date, Medication, Nickname, Dosage, 
        Scheduled Dosage, Unit, Status, Archived, Codings
        
    Status values:
        - "Taken": medication was taken
        - "Skipped": medication was explicitly skipped
        - "Not Interacted": no action recorded
        
    Only "Taken" records contribute to the output.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.warning(f"[APPLE/AUTOEXPORT/MEDS] CSV not found: {csv_path}")
        return _empty_meds_daily()
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"[APPLE/AUTOEXPORT/MEDS] Loaded {len(df)} records from {csv_path.name}")
    except Exception as e:
        logger.error(f"[APPLE/AUTOEXPORT/MEDS] Failed to read CSV: {e}")
        return _empty_meds_daily()
    
    if df.empty:
        logger.info(f"[APPLE/AUTOEXPORT/MEDS] CSV is empty")
        return _empty_meds_daily()
    
    # Verify required columns exist
    required_cols = {"Date", "Medication", "Status"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        logger.warning(f"[APPLE/AUTOEXPORT/MEDS] Missing required columns: {missing}")
        return _empty_meds_daily()
    
    # Parse dates and extract date-only string
    # Use utc=True to handle mixed timezone offsets, then convert to date string
    try:
        df["parsed_date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        df["date"] = df["parsed_date"].dt.strftime("%Y-%m-%d")
    except Exception:
        # Fallback: extract date portion directly from string
        logger.warning("[APPLE/AUTOEXPORT/MEDS] UTC parsing failed, using string extraction")
        df["date"] = df["Date"].astype(str).str[:10]
    
    # Filter by snapshot date (deterministic: date <= snapshot)
    # Use string comparison for robustness against timezone-aware/naive mixing
    df = df[df["date"] <= snapshot_date]
    
    if df.empty:
        logger.info(f"[APPLE/AUTOEXPORT/MEDS] No records with date <= {snapshot_date}")
        return _empty_meds_daily()
    
    # Filter to "Taken" status only
    df_taken = df[df["Status"] == "Taken"].copy()
    
    if df_taken.empty:
        logger.info(f"[APPLE/AUTOEXPORT/MEDS] No 'Taken' medication records")
        return _empty_meds_daily()
    
    # Parse Dosage as numeric (may be NaN)
    df_taken["Dosage"] = pd.to_numeric(df_taken["Dosage"], errors="coerce").fillna(0)
    
    # Aggregate daily
    daily = df_taken.groupby("date").agg(
        med_event_count=("Medication", "count"),  # COUNT of events
        med_dose_total=("Dosage", "sum"),  # SUM of dosages
        med_names=("Medication", lambda x: ", ".join(sorted(set(x)))),
    ).reset_index()
    
    # Compute med_any as (med_event_count > 0)
    daily["med_any"] = (daily["med_event_count"] > 0).astype(int)
    
    # Add source identifier
    daily["med_sources"] = "AutoExport"
    
    # Reorder columns to canonical order
    daily = daily[["date", "med_any", "med_event_count", "med_dose_total", "med_names", "med_sources"]]
    
    logger.info(f"[APPLE/AUTOEXPORT/MEDS] Aggregated {len(daily)} medication days")
    
    return daily.sort_values("date").reset_index(drop=True)


# ============================================================================
# Main Aggregator Class
# ============================================================================

class MedsAggregator:
    """Parse Apple export.xml and aggregate medication records to daily metrics.
    
    This class mirrors the pattern used by AppleHealthAggregator in
    stage_csv_aggregation.py for consistency across domains.
    
    Parquet Caching Strategy (same as HR/CARDIO domain):
    1. Check for daily cache (.cache/export_apple_meds_daily.parquet) - fastest
    2. If not found, check events cache and re-aggregate
    3. If neither exists, parse XML with binary regex and save both caches
    """
    
    def __init__(self, xml_path: str | Path, use_cache: bool = True):
        """Initialize with path to export.xml.
        
        Args:
            xml_path: Path to Apple Health export.xml file
            use_cache: Whether to use Parquet caching (default: True)
            
        Raises:
            FileNotFoundError: If export.xml does not exist
        """
        self.xml_path = Path(xml_path)
        if not self.xml_path.exists():
            raise FileNotFoundError(f"export.xml not found: {xml_path}")
        
        self.file_size_bytes = self.xml_path.stat().st_size
        self.file_size_mb = self.file_size_bytes / (1024 * 1024)
        self.use_cache = use_cache
        
        # Setup cache paths (same pattern as HR domain)
        self.cache_dir = self.xml_path.parent / ".cache"
        self.cache_file_daily = self.cache_dir / f"{self.xml_path.stem}_apple_meds_daily.parquet"
        self.cache_file_events = self.cache_dir / f"{self.xml_path.stem}_apple_meds_events.parquet"
        
        logger.info(f"[APPLE/EXPORT/MEDS] Loading export.xml: {xml_path}")
        logger.info(f"[APPLE/EXPORT/MEDS] File size: {self.file_size_mb:.1f} MB")
        
        # Lazy load XML tree (only if needed for fallback parsing)
        self._tree = None
        self._root = None
    
    @property
    def root(self):
        """Lazy load XML ElementTree root (only if binary regex fails)."""
        if self._root is None:
            logger.info(f"[APPLE/EXPORT/MEDS] Parsing XML tree (fallback mode)...")
            self._tree = ET.parse(self.xml_path)
            self._root = self._tree.getroot()
        return self._root
    
    def _parse_datetime(self, dt_str: str) -> Optional[datetime]:
        """Parse Apple HealthKit datetime format.
        
        Handles formats like: "2024-01-15 10:30:45 +0000"
        """
        if not dt_str:
            return None
        try:
            # Try ISO format first
            return datetime.fromisoformat(dt_str.replace(" +0000", "+00:00").replace(" -", "-"))
        except Exception:
            try:
                return pd.to_datetime(dt_str).to_pydatetime()
            except Exception:
                logger.debug(f"[APPLE/EXPORT/MEDS] Could not parse datetime: {dt_str}")
                return None
    
    def _get_date_str(self, dt: datetime) -> Optional[str]:
        """Extract date string (YYYY-MM-DD) from datetime."""
        if dt is None:
            return None
        return dt.strftime("%Y-%m-%d")
    
    def _load_from_cache(self) -> Optional[pd.DataFrame]:
        """Try to load daily MEDS data from Parquet cache.
        
        Caching strategy (same as HR/CARDIO domain):
        1. Try daily cache first (fastest: already aggregated)
        2. Else try events cache and re-aggregate
        3. Return None if no cache available
        
        Returns:
            DataFrame with daily medication data, or None if no cache
        """
        if not self.use_cache:
            return None
        
        # Strategy 1: Daily cache (fastest)
        if self.cache_file_daily.exists():
            try:
                df = pd.read_parquet(self.cache_file_daily)
                logger.info(f"[APPLE/EXPORT/MEDS] ✓ Loaded daily cache: {self.cache_file_daily.name} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"[APPLE/EXPORT/MEDS] Failed to load daily cache: {e}")
        
        # Strategy 2: Events cache -> re-aggregate
        if self.cache_file_events.exists():
            try:
                df_events = pd.read_parquet(self.cache_file_events)
                logger.info(f"[APPLE/EXPORT/MEDS] Loaded events cache: {self.cache_file_events.name} ({len(df_events)} rows)")
                df_daily = self._aggregate_events_to_daily(df_events)
                # Save daily cache for next time
                self._save_daily_cache(df_daily)
                return df_daily
            except Exception as e:
                logger.warning(f"[APPLE/EXPORT/MEDS] Failed to load events cache: {e}")
        
        return None
    
    def _aggregate_events_to_daily(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """Aggregate medication events DataFrame to daily stats.
        
        Args:
            df_events: DataFrame with columns [date, type, name, source]
            
        Returns:
            DataFrame with daily medication aggregations
        """
        if df_events.empty:
            return _empty_meds_daily()
        
        daily = df_events.groupby("date").agg(
            med_event_count=("type", "count"),
            med_names=("name", lambda x: ", ".join(sorted(set(x)))),
            med_sources=("source", lambda x: ", ".join(sorted(set(x)))),
        ).reset_index()
        
        daily["med_any"] = (daily["med_event_count"] > 0).astype(int)
        daily["med_dose_total"] = float("nan")  # Not available from export.xml
        
        # Reorder to canonical schema
        daily = daily[["date", "med_any", "med_event_count", "med_dose_total", "med_names", "med_sources"]]
        
        return daily.sort_values("date").reset_index(drop=True)
    
    def _save_events_cache(self, events: List[Dict]) -> None:
        """Save medication events to Parquet cache.
        
        Args:
            events: List of event dicts with keys [date, type, name, source]
        """
        if not self.use_cache or not events:
            return
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(events)
            df.to_parquet(self.cache_file_events, index=False)
            logger.info(f"[APPLE/EXPORT/MEDS] ✓ Saved events cache: {self.cache_file_events.name} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"[APPLE/EXPORT/MEDS] Failed to save events cache: {e}")
    
    def _save_daily_cache(self, df_daily: pd.DataFrame) -> None:
        """Save daily medication stats to Parquet cache.
        
        Args:
            df_daily: DataFrame with daily medication aggregations
        """
        if not self.use_cache or df_daily.empty:
            return
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            df_daily.to_parquet(self.cache_file_daily, index=False)
            logger.info(f"[APPLE/EXPORT/MEDS] ✓ Saved daily cache: {self.cache_file_daily.name} ({len(df_daily)} rows)")
        except Exception as e:
            logger.warning(f"[APPLE/EXPORT/MEDS] Failed to save daily cache: {e}")
    
    def aggregate_medications_binary_regex(self) -> pd.DataFrame:
        """Parse medication records using ultra-fast binary regex with Parquet caching.
        
        Caching strategy (same as HR/CARDIO domain):
        1. Try daily cache first (fastest)
        2. Else try events cache and re-aggregate
        3. Else parse XML with binary regex and save both caches
        
        Uses byte-based progress bar for accurate tracking during XML read.
        
        Returns:
            DataFrame with daily medication aggregations
        """
        # Try cache first
        cached_df = self._load_from_cache()
        if cached_df is not None:
            return cached_df
        
        logger.info(f"[APPLE/EXPORT/MEDS] Parsing with binary regex (fast mode)...")
        
        all_events: List[Dict] = []  # Store flat events for caching
        meds_data: Dict[str, Dict] = defaultdict(lambda: {
            "events": [],
            "names": set(),
            "sources": set(),
        })
        
        try:
            # Read file with byte-based progress bar (accurate tracking)
            logger.info(f"[APPLE/EXPORT/MEDS]   Reading {self.file_size_mb:.1f} MB...")
            
            chunk_size = 64 * 1024 * 1024  # 64MB chunks
            content_chunks = []
            
            with _make_bytes_pbar(self.file_size_bytes, "[APPLE/EXPORT/MEDS] Reading XML") as pbar:
                with open(self.xml_path, 'rb') as f:
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        content_chunks.append(chunk)
                        pbar.update(len(chunk))
            
            content = b''.join(content_chunks)
            del content_chunks  # Free memory
            
            records_found = 0
            
            # 1. Search for ClinicalRecord tags (most likely to contain meds)
            logger.info(f"[APPLE/EXPORT/MEDS]   Scanning for medication records...")
            clinical_matches = list(re.finditer(
                rb'<ClinicalRecord[^>]*?>.*?</ClinicalRecord>',
                content,
                re.DOTALL
            ))
            
            if clinical_matches:
                logger.info(f"[APPLE/EXPORT/MEDS]   Found {len(clinical_matches)} ClinicalRecord tags...")
                
                with _make_pbar(len(clinical_matches), "[APPLE/EXPORT/MEDS] Parsing ClinicalRecords") as pbar:
                    for match in clinical_matches:
                        pbar.update(1)
                        record_xml = match.group(0)
                        
                        # Check if this is a medication-related record
                        type_match = re.search(rb'type="([^"]*)"', record_xml)
                        if not type_match:
                            continue
                        
                        rec_type = type_match.group(1).decode('utf-8', errors='ignore')
                        
                        # Filter to medication-related types
                        is_med_type = any(
                            med_type.lower() in rec_type.lower() 
                            for med_type in ["medication", "immunization", "vaccine", "drug"]
                        )
                        if not is_med_type:
                            continue
                        
                        # Extract date
                        date_match = re.search(DATE_PATTERN, record_xml)
                        if not date_match:
                            continue
                        
                        date_str = date_match.group(1).decode('utf-8', errors='ignore')[:10]
                        
                        # Extract source
                        source_match = re.search(SOURCE_PATTERN, record_xml)
                        source = source_match.group(1).decode('utf-8', errors='ignore') if source_match else "Unknown"
                        
                        # Extract medication name (often in displayText or identifier)
                        name_match = re.search(rb'(?:displayText|identifier|name)="([^"]*)"', record_xml)
                        name = name_match.group(1).decode('utf-8', errors='ignore') if name_match else "Unknown"
                        
                        # Store for aggregation
                        meds_data[date_str]["events"].append({
                            "type": rec_type,
                            "name": name,
                            "source": source,
                        })
                        meds_data[date_str]["names"].add(name)
                        meds_data[date_str]["sources"].add(source)
                        
                        # Store flat event for caching
                        all_events.append({
                            "date": date_str,
                            "type": rec_type,
                            "name": name,
                            "source": source,
                        })
                        records_found += 1
            
            # 2. Search for Record tags with medication-related types
            for med_type in MEDICATION_RECORD_TYPES:
                type_pattern = f'<Record[^>]*?type="{med_type}"[^>]*?>'.encode()
                record_matches = list(re.finditer(type_pattern, content))
                
                if record_matches:
                    logger.info(f"[APPLE/EXPORT/MEDS]   Found {len(record_matches)} {med_type.split('Identifier')[-1]} records...")
                    
                    for match in record_matches:
                        record_tag = match.group(0)
                        
                        # Extract date
                        date_match = re.search(DATE_PATTERN, record_tag)
                        if not date_match:
                            continue
                        
                        date_str = date_match.group(1).decode('utf-8', errors='ignore')[:10]
                        
                        # Extract source
                        source_match = re.search(SOURCE_PATTERN, record_tag)
                        source = source_match.group(1).decode('utf-8', errors='ignore') if source_match else "Unknown"
                        
                        # Store for aggregation
                        short_type = med_type.split("Identifier")[-1]
                        meds_data[date_str]["events"].append({
                            "type": short_type,
                            "name": short_type,
                            "source": source,
                        })
                        meds_data[date_str]["names"].add(short_type)
                        meds_data[date_str]["sources"].add(source)
                        
                        # Store flat event for caching
                        all_events.append({
                            "date": date_str,
                            "type": short_type,
                            "name": short_type,
                            "source": source,
                        })
                        records_found += 1
            
            logger.info(f"[APPLE/EXPORT/MEDS] ✓ Found {records_found} medication-related records across {len(meds_data)} days")
            
            # Save events cache
            self._save_events_cache(all_events)
            
        except Exception as e:
            logger.warning(f"[APPLE/EXPORT/MEDS] Binary regex parsing failed: {e}")
            logger.info(f"[APPLE/EXPORT/MEDS] Falling back to ElementTree parsing...")
            return self.aggregate_medications_elementtree()
        
        # Build daily DataFrame and save cache
        df_daily = self._build_daily_dataframe(meds_data)
        self._save_daily_cache(df_daily)
        
        return df_daily
    
    def aggregate_medications_elementtree(self) -> pd.DataFrame:
        """Fallback: Parse medication records using ElementTree.
        
        This is slower but more robust for malformed XML or edge cases.
        
        Returns:
            DataFrame with daily medication aggregations
        """
        logger.info(f"[APPLE/EXPORT/MEDS] Parsing with ElementTree (fallback mode)...")
        
        meds_data: Dict[str, Dict] = defaultdict(lambda: {
            "events": [],
            "names": set(),
            "sources": set(),
        })
        
        records_found = 0
        
        # Search ClinicalRecord nodes
        for record in self.root.findall(".//ClinicalRecord"):
            rec_type = record.get("type", "")
            
            # Filter to medication-related
            is_med_type = any(
                kw in rec_type.lower() 
                for kw in ["medication", "immunization", "vaccine", "drug"]
            )
            if not is_med_type:
                continue
            
            # Parse date
            date_str = record.get("creationDate") or record.get("startDate")
            if not date_str:
                continue
            
            dt = self._parse_datetime(date_str)
            if dt is None:
                continue
            
            date_key = self._get_date_str(dt)
            source = record.get("sourceName", "Unknown")
            name = record.get("displayText") or record.get("identifier") or "Unknown"
            
            meds_data[date_key]["events"].append({
                "type": rec_type,
                "name": name,
                "source": source,
            })
            meds_data[date_key]["names"].add(name)
            meds_data[date_key]["sources"].add(source)
            records_found += 1
        
        # Search Record nodes for supplement/vitamin types
        for record in self.root.findall(".//Record"):
            rec_type = record.get("type", "")
            
            if rec_type not in MEDICATION_RECORD_TYPES:
                continue
            
            date_str = record.get("startDate")
            if not date_str:
                continue
            
            dt = self._parse_datetime(date_str)
            if dt is None:
                continue
            
            date_key = self._get_date_str(dt)
            source = record.get("sourceName", "Unknown")
            short_type = rec_type.split("Identifier")[-1]
            
            meds_data[date_key]["events"].append({
                "type": short_type,
                "name": short_type,
                "source": source,
            })
            meds_data[date_key]["names"].add(short_type)
            meds_data[date_key]["sources"].add(source)
            records_found += 1
        
        logger.info(f"[APPLE/EXPORT/MEDS] ✓ Found {records_found} medication records across {len(meds_data)} days")
        
        return self._build_daily_dataframe(meds_data)
    
    def _build_daily_dataframe(self, meds_data: Dict[str, Dict]) -> pd.DataFrame:
        """Convert aggregated medication data to DataFrame.
        
        Args:
            meds_data: Dict mapping date -> {events: [...], names: set, sources: set}
            
        Returns:
            DataFrame with canonical medication columns
        """
        if not meds_data:
            logger.info(f"[APPLE/EXPORT/MEDS] No medication data found - returning empty DataFrame")
            return _empty_meds_daily()
        
        rows = []
        for date_str in sorted(meds_data.keys()):
            data = meds_data[date_str]
            rows.append({
                "date": date_str,
                "med_any": 1 if len(data["events"]) > 0 else 0,
                "med_event_count": len(data["events"]),
                "med_dose_total": float("nan"),  # Not available from export.xml
                "med_names": ", ".join(sorted(data["names"])) if data["names"] else "",
                "med_sources": ", ".join(sorted(data["sources"])) if data["sources"] else "",
            })
        
        df = pd.DataFrame(rows)
        logger.info(f"[APPLE/EXPORT/MEDS] Aggregated {len(df)} medication days")
        
        return df
    
    def aggregate_all(self) -> Dict[str, pd.DataFrame]:
        """Aggregate all medication metrics and return dict.
        
        Returns:
            Dict with "daily_meds" key containing the aggregated DataFrame
        """
        return {
            "daily_meds": self.aggregate_medications_binary_regex()
        }


# ============================================================================
# Public API Functions
# ============================================================================

def load_apple_meds_daily(
    export_xml_path: Path | str,
    home_tz: str = "UTC",
    max_records: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load and aggregate Apple Health medication records.
    
    This is the primary entry point for loading medication data from an
    Apple Health export.xml file. Uses Parquet caching for fast subsequent loads.
    
    Args:
        export_xml_path: Path to export.xml file
        home_tz: User's home timezone (for date localization)
        max_records: Maximum records to process (for testing; None = all)
        use_cache: Whether to use Parquet caching (default: True)
        
    Returns:
        DataFrame with daily medication aggregations.
        If no medication data exists, returns empty well-formed DataFrame.
    """
    xml_path = Path(export_xml_path)
    
    if not xml_path.exists():
        logger.warning(f"[APPLE/EXPORT/MEDS] export.xml not found: {xml_path}")
        return _empty_meds_daily()
    
    try:
        aggregator = MedsAggregator(xml_path, use_cache=use_cache)
        df = aggregator.aggregate_medications_binary_regex()
        
        if df.empty:
            logger.info(f"[APPLE/EXPORT/MEDS] No medication records found in {xml_path.name}")
            return _empty_meds_daily()
        
        return df
    
    except Exception as e:
        logger.error(f"[APPLE/EXPORT/MEDS] Failed to parse medication data: {e}")
        logger.info(f"[APPLE/EXPORT/MEDS] Returning empty DataFrame to allow pipeline to continue")
        return _empty_meds_daily()


def find_latest_apple_zip(participant: str) -> Optional[Path]:
    """Find the most recent Apple Health export ZIP for a participant.
    
    Searches in data/raw/<PID>/apple/export/ and data/raw/<PID>/apple/autoexport/
    for ZIP files matching the Apple Health export pattern.
    
    Args:
        participant: Participant ID (e.g., "P000001")
        
    Returns:
        Path to the latest ZIP file, or None if not found
    """
    raw_base = Path("data") / "raw" / participant / "apple"
    
    if not raw_base.exists():
        logger.debug(f"[APPLE/EXPORT/MEDS] Raw Apple dir not found: {raw_base}")
        return None
    
    # Search for ZIPs in export/ and autoexport/ subdirs
    all_zips: List[Tuple[Path, datetime]] = []
    
    for subdir in ["export", "autoexport"]:
        search_dir = raw_base / subdir
        if not search_dir.exists():
            continue
        
        for zip_path in search_dir.glob("*.zip"):
            # Try to extract date from filename like "apple_health_export_20251207.zip"
            name = zip_path.stem.lower()
            date_match = re.search(r'(\d{8})', name)
            
            if date_match:
                try:
                    file_date = datetime.strptime(date_match.group(1), "%Y%m%d")
                    all_zips.append((zip_path, file_date))
                except ValueError:
                    # Use modification time as fallback
                    mtime = datetime.fromtimestamp(zip_path.stat().st_mtime)
                    all_zips.append((zip_path, mtime))
            else:
                # Use modification time
                mtime = datetime.fromtimestamp(zip_path.stat().st_mtime)
                all_zips.append((zip_path, mtime))
    
    if not all_zips:
        logger.debug(f"[APPLE/EXPORT/MEDS] No Apple ZIPs found in {raw_base}")
        return None
    
    # Sort by date descending and return the most recent
    all_zips.sort(key=lambda x: x[1], reverse=True)
    latest_zip = all_zips[0][0]
    
    logger.info(f"[APPLE/EXPORT/MEDS] Found latest Apple ZIP: {latest_zip}")
    return latest_zip


def extract_apple_zip_for_meds(
    zip_path: Path, 
    snapshot_dir: Path,
    dry_run: bool = False
) -> Optional[Path]:
    """Extract Apple Health ZIP to snapshot directory for medication parsing.
    
    Extracts to: <snapshot_dir>/extracted/apple/apple_health_export/
    
    Args:
        zip_path: Path to Apple Health ZIP file
        snapshot_dir: Target snapshot directory
        dry_run: If True, don't extract, just log
        
    Returns:
        Path to extracted export.xml, or None if extraction failed
    """
    import zipfile
    
    target_dir = snapshot_dir / "extracted" / "apple"
    
    if dry_run:
        logger.info(f"[APPLE/EXPORT/MEDS] DRY RUN: Would extract {zip_path} to {target_dir}")
        return None
    
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[APPLE/EXPORT/MEDS] Extracting {zip_path.name} to {target_dir}...")
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Extract all contents
            zf.extractall(target_dir)
        
        # Find the export.xml in the extracted contents
        export_xml = None
        for p in target_dir.rglob("export.xml"):
            export_xml = p
            break
        
        if export_xml:
            logger.info(f"[APPLE/EXPORT/MEDS] ✓ Extracted export.xml: {export_xml}")
            return export_xml
        else:
            logger.warning(f"[APPLE/EXPORT/MEDS] Extraction succeeded but export.xml not found in {target_dir}")
            return None
            
    except Exception as e:
        logger.error(f"[APPLE/EXPORT/MEDS] Failed to extract ZIP: {e}")
        return None


def discover_apple_meds(snapshot_dir: Path) -> List[Tuple[Path, str, str]]:
    """Discover Apple export.xml files for medication parsing.
    
    Follows the same pattern as discover_apple() in activity_from_extracted.py.
    Searches extracted/apple/*/apple_health_export/ for export.xml files.
    
    Args:
        snapshot_dir: Snapshot directory (e.g., data/etl/P000001/2025-12-07/)
        
    Returns:
        List of (Path, vendor, variant) tuples
    """
    base = snapshot_dir / "extracted" / "apple"
    if not base.exists():
        logger.debug(f"[APPLE/EXPORT/MEDS] Apple extracted dir not found: {base}")
        return []
    
    export_xmls: List[Tuple[Path, str, str]] = []
    
    # Check directly under apple/ for apple_health_export/export.xml
    direct_ah = base / "apple_health_export"
    if direct_ah.exists():
        ex = direct_ah / "export.xml"
        if ex.exists():
            export_xmls.append((ex, "apple", "default"))
            return export_xmls
    
    # Look under each variant folder (e.g., inapp/, export/, etc.)
    for variant_dir in [p for p in base.iterdir() if p.is_dir()]:
        variant = variant_dir.name
        ah_dir = variant_dir / "apple_health_export"
        if not ah_dir.exists():
            # Also accept exports directly under variant_dir
            ah_dir = variant_dir
        
        # Look for export.xml
        ex = ah_dir / "export.xml"
        if ex.exists():
            export_xmls.append((ex, "apple", variant))
            continue
        
        # Also search recursively for export.xml
        for p in ah_dir.rglob("export.xml"):
            export_xmls.append((p, "apple", variant))
    
    return export_xmls


def run_meds_aggregation(
    participant: str = "P000001",
    snapshot: str | None = None,
    extracted_dir: str | None = None,
    output_dir: str | None = None,
    dry_run: bool = False
) -> Dict[str, str]:
    """Execute medication aggregation stage with multi-vendor prioritization.
    
    This function collects medication data from multiple sources:
    1. apple_export: Official Apple Health export.xml (ClinicalRecords)
    2. apple_autoexport: Auto Export iOS app (Medications-*.csv)
    
    The source prioritizer selects the first non-empty source according to
    SOURCE_PRIORITY. The selected vendor is recorded in the `med_vendor` column.
    
    Args:
        participant: Participant ID (e.g., "P000001")
        snapshot: Snapshot date (e.g., "2025-12-08"); defaults to today
        extracted_dir: Base ETL directory (default: None, uses etl_snapshot_root)
        output_dir: Output directory; defaults to snapshot extracted/apple/
        dry_run: If True, don't write files (just log)
        
    Returns:
        Dict mapping output names to file paths
    """
    from datetime import datetime as dt
    
    # Default snapshot to today
    if snapshot is None:
        snapshot = dt.now().strftime("%Y-%m-%d")
    
    # Support "auto" resolution if etl module available
    if snapshot == "auto" and etl is not None:
        try:
            snapshot = etl.resolve_snapshot(snapshot, participant)
        except Exception as e:
            logger.warning(f"[APPLE/EXPORT/MEDS] Could not resolve 'auto' snapshot: {e}")
            snapshot = dt.now().strftime("%Y-%m-%d")
    
    # Ensure snapshot is a string for type safety
    snapshot = str(snapshot)
    
    # Build snapshot directory using canonical helper
    snapshot_dir = etl_snapshot_root(participant, snapshot)
    
    if output_dir is None:
        output_dir = str(snapshot_dir / "extracted" / "apple")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"MEDS AGGREGATION (multi-vendor) - {participant}/{snapshot}")
    logger.info(f"{'='*80}\n")
    
    # ========================================================================
    # Collect data from all available vendors
    # ========================================================================
    vendor_dataframes: Dict[str, pd.DataFrame] = {}
    
    # --- Source 1: apple_export (export.xml) ---
    export_xmls = discover_apple_meds(snapshot_dir)
    if export_xmls:
        xml_path = export_xmls[0][0]
        logger.info(f"[APPLE/EXPORT/MEDS] Loading apple_export: {xml_path}")
        df_apple_export = load_apple_meds_daily(xml_path)
        if len(df_apple_export) > 0:
            vendor_dataframes["apple_export"] = df_apple_export
            logger.info(f"[APPLE/EXPORT/MEDS] apple_export: {len(df_apple_export)} days")
        else:
            logger.info(f"[APPLE/EXPORT/MEDS] apple_export: 0 days (no medication records in export.xml)")
    else:
        logger.info(f"[APPLE/EXPORT/MEDS] apple_export: not available (no export.xml found)")
    
    # --- Source 2: apple_autoexport (Medications-*.csv) ---
    if get_autoexport_csv_path is not None:
        autoexport_csv = get_autoexport_csv_path(participant, snapshot, "Medications")
        if autoexport_csv and autoexport_csv.exists():
            logger.info(f"[APPLE/AUTOEXPORT/MEDS] Loading apple_autoexport: {autoexport_csv}")
            df_autoexport = load_autoexport_meds_daily(autoexport_csv, snapshot)
            if len(df_autoexport) > 0:
                vendor_dataframes["apple_autoexport"] = df_autoexport
                logger.info(f"[APPLE/AUTOEXPORT/MEDS] apple_autoexport: {len(df_autoexport)} days")
            else:
                logger.info(f"[APPLE/AUTOEXPORT/MEDS] apple_autoexport: 0 days")
        else:
            logger.info(f"[APPLE/AUTOEXPORT/MEDS] apple_autoexport: not available (no Medications CSV in extracted autoexport)")
    else:
        logger.info(f"[APPLE/AUTOEXPORT/MEDS] apple_autoexport: discovery module not available")
    
    # ========================================================================
    # Apply source prioritization
    # ========================================================================
    if prioritize_source is not None:
        if log_source_summary is not None:
            log_source_summary(vendor_dataframes, "meds")
        
        df_meds, selected_vendor = prioritize_source(
            vendor_dataframes=vendor_dataframes,
            fallback_df=_empty_meds_daily(),
            domain="meds"
        )
    else:
        # Fallback if prioritizer not available
        logger.warning(f"[APPLE/EXPORT/MEDS] Source prioritizer not available, using manual selection")
        if vendor_dataframes:
            # Manual priority: apple_export > apple_autoexport
            if "apple_export" in vendor_dataframes and len(vendor_dataframes["apple_export"]) > 0:
                df_meds = vendor_dataframes["apple_export"]
                selected_vendor = "apple_export"
            elif "apple_autoexport" in vendor_dataframes and len(vendor_dataframes["apple_autoexport"]) > 0:
                df_meds = vendor_dataframes["apple_autoexport"]
                selected_vendor = "apple_autoexport"
            else:
                df_meds = _empty_meds_daily()
                selected_vendor = "fallback"
        else:
            df_meds = _empty_meds_daily()
            selected_vendor = "fallback"
    
    # Add med_vendor column to indicate which source was selected
    df_meds = df_meds.copy()
    df_meds["med_vendor"] = selected_vendor
    
    # ========================================================================
    # Write output
    # ========================================================================
    out_path = Path(output_dir) / "daily_meds.csv"
    
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_meds.to_csv(out_path, index=False)
        logger.info(f"[APPLE/EXPORT/MEDS] ✓ Saved {len(df_meds)} medication days to {out_path}")
        logger.info(f"[APPLE/EXPORT/MEDS] ✓ Vendor: {selected_vendor}")
    else:
        logger.info(f"[APPLE/EXPORT/MEDS] DRY RUN: Would save {len(df_meds)} days to {out_path}")
    
    return {"daily_meds": str(out_path), "vendor": selected_vendor}


# ============================================================================
# CLI Entrypoint (follows cardio_from_extracted.py pattern)
# ============================================================================

def main(argv: list[str] | None = None) -> int:
    """Main entry point for medication aggregation.
    
    Follows the same CLI pattern as cardio_from_extracted.py.
    """
    if argv is None:
        import sys
        argv = sys.argv[1:]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(
        prog="meds_from_extracted",
        description="Aggregate medication records from Apple Health export.xml"
    )
    parser.add_argument(
        "--pid", 
        required=True,
        help="Participant ID (e.g., P000001)"
    )
    parser.add_argument(
        "--snapshot", 
        required=True,
        help="Snapshot date YYYY-MM-DD (e.g., 2025-11-07, 2025-12-07)"
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        type=int,
        default=1,
        help="Dry run mode (1=dry run, 0=real run; default: 1)"
    )
    parser.add_argument(
        "--export-xml",
        type=str,
        default=None,
        help="Direct path to export.xml (skips discovery)"
    )
    
    args = parser.parse_args(argv)
    
    pid = args.pid
    snap = args.snapshot
    dry_run = bool(args.dry_run)
    
    # Resolve "auto" snapshot if etl module available
    if snap == "auto":
        if etl is None:
            print("[error] cannot resolve 'auto' snapshot; etl resolver not available")
            return 3
        snap = etl.resolve_snapshot(snap, pid)
    
    if args.export_xml:
        # Direct mode: parse specific export.xml
        df = load_apple_meds_daily(args.export_xml)
        print(f"\n[DONE] Loaded {len(df)} medication days from {args.export_xml}")
        if not df.empty:
            print(f"\nSample output:")
            print(df.head(10).to_string(index=False))
        return 0
    
    # Discovery mode: find export.xml in ETL structure
    outputs = run_meds_aggregation(
        participant=pid,
        snapshot=snap,
        dry_run=dry_run
    )
    
    print(f"\n[DONE] Outputs: {outputs}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
