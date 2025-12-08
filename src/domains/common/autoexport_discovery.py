"""Auto Export discovery and snapshot-aware file selection.

This module provides deterministic file selection for Apple Health Auto Export CSVs.

Data flow:
1. Stage 0 extracts ZIP from data/raw/<PID>/apple/autoexport/ 
   to data/etl/<PID>/<SNAPSHOT>/extracted/apple/autoexport/
2. This module discovers CSVs from the extracted location
3. Individual domains (meds, som) use this to find their signal files

Filename conventions inside the ZIP:
    <SignalName>-<StartDate>-<EndDate>.csv
    e.g., StateOfMind-2025-11-08-2025-12-08.csv
          Medications-2025-11-08-2025-12-08.csv

Selection rule:
    For a given snapshot, CSVs have already been extracted by Stage 0
    using deterministic ZIP selection (date <= snapshot).
    This module simply discovers what's available in the extracted folder.

This ensures reproducible, deterministic ETL runs.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("etl.autoexport")


@dataclass
class AutoExportFile:
    """Metadata for an Auto Export CSV file."""
    path: Path
    signal_name: str
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    
    @property
    def date_range_days(self) -> int:
        """Number of days covered by this export."""
        if self.start_date and self.end_date:
            return (self.end_date - self.start_date).days
        return 0


# Regex pattern for Auto Export filenames: <Signal>-<YYYY-MM-DD>-<YYYY-MM-DD>.csv
AUTOEXPORT_FILENAME_PATTERN = re.compile(
    r'^(?P<signal>[A-Za-z]+)-'
    r'(?P<start>\d{4}-\d{2}-\d{2})-'
    r'(?P<end>\d{4}-\d{2}-\d{2})\.csv$'
)

# Signal name mappings (case-insensitive matching)
SIGNAL_NAME_MAP = {
    "stateofmind": "StateOfMind",
    "medications": "Medications",
    "workouts": "Workouts",
    "sleep": "Sleep",
    "heartrate": "HeartRate",
    "healthautoexport": "HealthAutoExport",
}


def parse_autoexport_filename(filename: str) -> Optional[Tuple[str, datetime, datetime]]:
    """Parse an Auto Export filename to extract signal name and date range.
    
    Args:
        filename: Filename like "StateOfMind-2025-11-08-2025-12-08.csv"
        
    Returns:
        Tuple of (signal_name, start_date, end_date) or None if not parseable
    """
    match = AUTOEXPORT_FILENAME_PATTERN.match(filename)
    if not match:
        return None
    
    try:
        signal = match.group("signal")
        # Normalize signal name
        signal = SIGNAL_NAME_MAP.get(signal.lower(), signal)
        start = datetime.strptime(match.group("start"), "%Y-%m-%d")
        end = datetime.strptime(match.group("end"), "%Y-%m-%d")
        return (signal, start, end)
    except ValueError:
        return None


def get_extracted_autoexport_dir(participant: str, snapshot: str) -> Path:
    """Get the canonical path for extracted Auto Export files.
    
    Args:
        participant: Participant ID (e.g., "P000001")
        snapshot: Snapshot date (e.g., "2025-12-08")
        
    Returns:
        Path to extracted/apple/autoexport directory
    """
    return Path("data") / "etl" / participant / snapshot / "extracted" / "apple" / "autoexport"


def list_extracted_autoexport_files(
    participant: str, 
    snapshot: str
) -> List[AutoExportFile]:
    """List all Auto Export CSV files extracted for a snapshot.
    
    Searches in: data/etl/<PID>/<SNAPSHOT>/extracted/apple/autoexport/
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date
        
    Returns:
        List of AutoExportFile objects with parsed metadata
    """
    autoexport_dir = get_extracted_autoexport_dir(participant, snapshot)
    
    if not autoexport_dir.exists():
        logger.debug(f"[AutoExport] Extracted directory not found: {autoexport_dir}")
        return []
    
    files: List[AutoExportFile] = []
    
    for csv_path in autoexport_dir.glob("*.csv"):
        parsed = parse_autoexport_filename(csv_path.name)
        if parsed:
            signal_name, start_date, end_date = parsed
            files.append(AutoExportFile(
                path=csv_path,
                signal_name=signal_name,
                start_date=start_date,
                end_date=end_date
            ))
        else:
            # Non-standard filename - still include but with unknown dates
            # Extract signal name from first part before hyphen or use whole name
            name_parts = csv_path.stem.split("-")
            signal_name = name_parts[0] if name_parts else csv_path.stem
            signal_name = SIGNAL_NAME_MAP.get(signal_name.lower(), signal_name)
            
            files.append(AutoExportFile(
                path=csv_path,
                signal_name=signal_name,
                start_date=None,
                end_date=None
            ))
            logger.debug(f"[AutoExport] Non-standard filename: {csv_path.name} → signal={signal_name}")
    
    logger.info(f"[AutoExport] Found {len(files)} CSV files in {autoexport_dir}")
    return files


def get_autoexport_csv_path(
    participant: str,
    snapshot: str,
    signal: str
) -> Optional[Path]:
    """Get the path to a specific Auto Export signal CSV for a snapshot.
    
    Searches in the extracted autoexport directory for a CSV matching
    the signal name.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date YYYY-MM-DD
        signal: Signal name (e.g., "StateOfMind", "Medications")
        
    Returns:
        Path to the CSV file, or None if not found
    """
    files = list_extracted_autoexport_files(participant, snapshot)
    
    # Normalize signal name for comparison
    signal_lower = signal.lower()
    
    for f in files:
        if f.signal_name.lower() == signal_lower:
            logger.info(f"[AutoExport] Found {signal}: {f.path.name}")
            return f.path
    
    logger.debug(f"[AutoExport] No {signal} CSV found for {participant}/{snapshot}")
    return None


def discover_extracted_signals(participant: str, snapshot: str) -> List[str]:
    """List all available signal types in the extracted Auto Export.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date
        
    Returns:
        List of unique signal names (e.g., ["StateOfMind", "Medications", ...])
    """
    files = list_extracted_autoexport_files(participant, snapshot)
    return sorted(set(f.signal_name for f in files))


def get_autoexport_signals_summary(participant: str, snapshot: str) -> Dict[str, dict]:
    """Get summary of all extracted Auto Export signals.
    
    Args:
        participant: Participant ID
        snapshot: Snapshot date
        
    Returns:
        Dict mapping signal_name -> {path, start_date, end_date, size_kb}
    """
    files = list_extracted_autoexport_files(participant, snapshot)
    
    summary = {}
    for f in files:
        summary[f.signal_name] = {
            "path": str(f.path),
            "start_date": f.start_date.strftime("%Y-%m-%d") if f.start_date else None,
            "end_date": f.end_date.strftime("%Y-%m-%d") if f.end_date else None,
            "size_kb": f.path.stat().st_size / 1024 if f.path.exists() else 0,
        }
    
    return summary


# ============================================================================
# Legacy support: Direct reading from data/raw (for backward compatibility)
# ============================================================================

def list_raw_autoexport_files(participant: str) -> List[AutoExportFile]:
    """List Auto Export CSV files directly in raw directory (legacy support).
    
    Searches in: data/raw/<PID>/apple/autoexport/
    
    Note: This is for backward compatibility. New code should use
    list_extracted_autoexport_files() after Stage 0 extraction.
    
    Args:
        participant: Participant ID
        
    Returns:
        List of AutoExportFile objects
    """
    raw_dir = Path("data") / "raw" / participant / "apple" / "autoexport"
    
    if not raw_dir.exists():
        return []
    
    files: List[AutoExportFile] = []
    
    for csv_path in raw_dir.glob("*.csv"):
        parsed = parse_autoexport_filename(csv_path.name)
        if parsed:
            signal_name, start_date, end_date = parsed
            files.append(AutoExportFile(
                path=csv_path,
                signal_name=signal_name,
                start_date=start_date,
                end_date=end_date
            ))
    
    return files


# Backward compatibility aliases
list_autoexport_files = list_raw_autoexport_files
get_autoexport_path = get_autoexport_csv_path
select_autoexport_for_snapshot = lambda p, s, **kw: {
    f.signal_name: f for f in list_extracted_autoexport_files(p, s)
}
discover_autoexport_signals = discover_extracted_signals
