"""State of Mind (SoM) domain: Extract and aggregate mood data from Apple Auto Export.

Reads State of Mind CSV exports from the Auto Export iOS app and aggregates
them to one row per calendar day.

Design principles:
1. DETERMINISTIC: Same input always produces same output (fixed seed, sorted outputs)
2. DEFENSIVE: Returns empty well-formed DataFrame if no SoM data exists
3. NON-BREAKING: Never fails the global ETL pipeline; logs warnings and continues
4. CONSISTENT: Follows patterns from cardio/sleep/activity/meds domains

Auto Export CSV columns expected:
- Start: timestamp with timezone (e.g., "2025-08-03 01:44:43 +0100")
- End: timestamp with timezone
- Kind: "Daily Mood" or "Momentary Emotion"
- Labels: pipe-separated emotion labels (e.g., "Happy | Content | Grateful")
- Associations: pipe-separated contexts (e.g., "Work | Education | Friends")
- Valence: numeric score in [-1, +1] range
- Valence Classification: text category (e.g., "Very Pleasant", "Unpleasant")

Output columns:
- date: YYYY-MM-DD format (same as other domains)
- som_mean_score: mean valence across all entries that day
- som_last_score: last entry's valence (chronologically)
- som_n_entries: count of mood entries per day
- som_category_3class: discrete -1/0/+1 based on mean thresholds
- som_kind_dominant: most frequent Kind for the day
- som_labels: unique emotion labels (comma-separated)
- som_associations: unique associations (comma-separated)
"""
from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Import common IO utilities
try:
    from ..common.io import etl_snapshot_root
    from ..common.autoexport_discovery import (
        get_autoexport_csv_path,
        list_extracted_autoexport_files,
        get_extracted_autoexport_dir,
        AutoExportFile
    )
except ImportError:
    try:
        from src.domains.common.io import etl_snapshot_root
        from src.domains.common.autoexport_discovery import (
            get_autoexport_csv_path,
            list_extracted_autoexport_files,
            get_extracted_autoexport_dir,
            AutoExportFile
        )
    except ImportError:
        def etl_snapshot_root(pid: str, snapshot: str) -> Path:
            return Path("data") / "etl" / pid / snapshot
        def get_extracted_autoexport_dir(pid: str, snapshot: str) -> Path:
            return Path("data") / "etl" / pid / snapshot / "extracted" / "apple" / "autoexport"
        get_autoexport_csv_path = None
        list_extracted_autoexport_files = None

# Import IO guards
try:
    from src.lib.io_guards import write_csv
except ImportError:
    try:
        from lib.io_guards import write_csv
    except ImportError:
        def write_csv(df, path, dry_run=False, backup_name=None):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(p)

logger = logging.getLogger("etl.som")


# ============================================================================
# Constants and Thresholds
# ============================================================================

# 3-class mapping thresholds (deterministic)
# Based on Valence scale [-1, +1] from Apple Health State of Mind
THRESHOLD_NEGATIVE = -0.25  # mean <= this → -1 (negative/unstable)
THRESHOLD_POSITIVE = 0.25   # mean >= this → +1 (positive/stable)
# between -0.25 and +0.25 → 0 (neutral)

# Expected column names in Auto Export CSV
EXPECTED_COLUMNS = {
    "Start", "End", "Kind", "Labels", "Associations", "Valence", "Valence Classification"
}

# Valence column name variants (Auto Export may vary)
VALENCE_COLUMN_VARIANTS = ["Valence", "valence", "Score", "score", "MoodScore"]


# ============================================================================
# Empty DataFrame Schema
# ============================================================================

def _empty_som_daily() -> pd.DataFrame:
    """Return empty DataFrame with canonical SoM columns.
    
    Ensures downstream stages always receive a well-formed DataFrame,
    even when no SoM data exists in Auto Export.
    """
    return pd.DataFrame({
        "date": pd.Series(dtype="str"),
        "som_mean_score": pd.Series(dtype="float"),
        "som_last_score": pd.Series(dtype="float"),
        "som_n_entries": pd.Series(dtype="int"),
        "som_category_3class": pd.Series(dtype="int"),
        "som_kind_dominant": pd.Series(dtype="str"),
        "som_labels": pd.Series(dtype="str"),
        "som_associations": pd.Series(dtype="str"),
    })


# ============================================================================
# Parsing Functions
# ============================================================================

def parse_timestamp(ts_str: str) -> Optional[datetime]:
    """Parse Auto Export timestamp format.
    
    Handles: "2025-08-03 01:44:43 +0100"
    
    Args:
        ts_str: Timestamp string with timezone offset
        
    Returns:
        datetime object (timezone-naive, converted to local) or None
    """
    if not ts_str or pd.isna(ts_str):
        return None
    
    ts_str = str(ts_str).strip()
    
    # Try common formats
    formats = [
        "%Y-%m-%d %H:%M:%S %z",  # 2025-08-03 01:44:43 +0100
        "%Y-%m-%d %H:%M:%S",     # Without timezone
        "%Y-%m-%dT%H:%M:%S%z",   # ISO format
        "%Y-%m-%dT%H:%M:%S",     # ISO without TZ
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(ts_str, fmt)
            # Return as naive datetime (date extraction doesn't need TZ)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    
    # Fallback: try pandas parser
    try:
        return pd.to_datetime(ts_str).to_pydatetime()
    except Exception:
        return None


def parse_pipe_separated(value: str) -> List[str]:
    """Parse pipe-separated values like 'Happy | Content | Grateful'.
    
    Args:
        value: Pipe-separated string
        
    Returns:
        List of trimmed values
    """
    if not value or pd.isna(value):
        return []
    
    return [v.strip() for v in str(value).split("|") if v.strip()]


def classify_valence_3class(mean_valence: float) -> int:
    """Convert mean valence to 3-class categorical.
    
    Args:
        mean_valence: Mean valence score in [-1, +1]
        
    Returns:
        -1 (negative), 0 (neutral), or +1 (positive)
    """
    if pd.isna(mean_valence):
        return 0  # Default to neutral if missing
    
    if mean_valence <= THRESHOLD_NEGATIVE:
        return -1
    elif mean_valence >= THRESHOLD_POSITIVE:
        return 1
    else:
        return 0


# ============================================================================
# Main Aggregator Class
# ============================================================================

class SoMAggregator:
    """Parse Auto Export State of Mind CSV and aggregate to daily metrics."""
    
    def __init__(self, csv_path: Path | str):
        """Initialize with path to Auto Export CSV.
        
        Args:
            csv_path: Path to StateOfMind-*.csv file
            
        Raises:
            FileNotFoundError: If CSV does not exist
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"SoM CSV not found: {csv_path}")
        
        logger.info(f"[APPLE/AUTOEXPORT/SOM] Loading: {csv_path}")
        
        # Load CSV with special handling for trailing comma issue
        # Auto Export CSVs sometimes have trailing commas that create phantom columns
        self.df_raw = self._load_csv_with_trailing_comma_fix(self.csv_path)
        logger.info(f"[APPLE/AUTOEXPORT/SOM] Loaded {len(self.df_raw)} rows, columns: {list(self.df_raw.columns)}")
        
        # Validate columns
        self._validate_columns()
    
    def _load_csv_with_trailing_comma_fix(self, csv_path: Path) -> pd.DataFrame:
        """Load CSV and fix trailing comma issue from Auto Export.
        
        Auto Export CSVs often have trailing commas that cause pandas to misalign
        columns. This method detects and fixes the issue.
        
        The expected format is:
        Header: Start,End,Kind,Labels,Associations,Valence,Valence Classification
        Data:   ...,Labels,Associations,0.5,Pleasant,
        
        The trailing comma creates 8 fields for 7 columns, shifting everything.
        """
        # First, read raw to detect the issue
        with open(csv_path, 'r', encoding='utf-8') as f:
            header_line = f.readline().strip()
            first_data_line = f.readline().strip()
        
        header_cols = header_line.split(',')
        data_cols = first_data_line.split(',')
        
        n_header = len(header_cols)
        n_data = len(data_cols)
        
        logger.info(f"[APPLE/AUTOEXPORT/SOM] CSV structure: header={n_header} cols, data={n_data} fields")
        
        if n_data > n_header:
            # Trailing comma detected - need to adjust
            # The actual Valence is at index 5, Valence Classification at index 6
            # But pandas shifts them because of trailing comma
            logger.warning(f"[APPLE/AUTOEXPORT/SOM] Detected trailing comma issue ({n_data} fields vs {n_header} header cols)")
            
            # Read with explicit column names, ignoring extra columns
            df = pd.read_csv(
                csv_path,
                names=header_cols + [f"_extra_{i}" for i in range(n_data - n_header)],
                header=0,  # Skip original header row
                dtype=str  # Read all as string first
            )
            
            # Drop extra columns
            extra_cols = [c for c in df.columns if c.startswith("_extra_")]
            if extra_cols:
                df = df.drop(columns=extra_cols)
                logger.info(f"[APPLE/AUTOEXPORT/SOM] Dropped {len(extra_cols)} phantom column(s)")
            
            # Now convert Valence to numeric
            if "Valence" in df.columns:
                df["Valence"] = pd.to_numeric(df["Valence"], errors="coerce")
                valid_valence = df["Valence"].notna().sum()
                logger.info(f"[APPLE/AUTOEXPORT/SOM] Valence column: {valid_valence} valid numeric values")
            
            return df
        else:
            # Normal CSV - read directly
            return pd.read_csv(csv_path)
    
    def _validate_columns(self):
        """Validate that expected columns are present."""
        missing = EXPECTED_COLUMNS - set(self.df_raw.columns)
        if missing:
            logger.warning(f"[APPLE/AUTOEXPORT/SOM] Missing expected columns: {missing}")
        
        # Find valence column
        self.valence_col = None
        for variant in VALENCE_COLUMN_VARIANTS:
            if variant in self.df_raw.columns:
                self.valence_col = variant
                break
        
        if self.valence_col is None:
            logger.warning("[APPLE/AUTOEXPORT/SOM] No valence/score column found!")
    
    def aggregate_daily(self, snapshot_date: Optional[str] = None) -> pd.DataFrame:
        """Aggregate SoM data to one row per calendar day.
        
        Args:
            snapshot_date: Optional YYYY-MM-DD to filter data up to this date
            
        Returns:
            DataFrame with daily SoM aggregations
        """
        if len(self.df_raw) == 0:
            logger.info("[APPLE/AUTOEXPORT/SOM] No data in CSV - returning empty DataFrame")
            return _empty_som_daily()
        
        # Parse timestamps and extract dates
        df = self.df_raw.copy()
        
        df["_parsed_dt"] = df["Start"].apply(parse_timestamp)
        df = df.dropna(subset=["_parsed_dt"])
        
        if len(df) == 0:
            logger.warning("[APPLE/AUTOEXPORT/SOM] No valid timestamps after parsing")
            return _empty_som_daily()
        
        df["date"] = df["_parsed_dt"].apply(lambda dt: dt.strftime("%Y-%m-%d"))
        
        # Filter by snapshot date if provided (deterministic cutoff)
        if snapshot_date:
            df = df[df["date"] <= snapshot_date]
            logger.info(f"[APPLE/AUTOEXPORT/SOM] Filtered to snapshot <= {snapshot_date}: {len(df)} rows")
        
        if len(df) == 0:
            logger.warning(f"[APPLE/AUTOEXPORT/SOM] No data before snapshot {snapshot_date}")
            return _empty_som_daily()
        
        # Ensure valence is numeric
        if self.valence_col and self.valence_col in df.columns:
            df["_valence"] = pd.to_numeric(df[self.valence_col], errors="coerce")
        else:
            df["_valence"] = np.nan
        
        # Sort by datetime for "last" aggregations
        df = df.sort_values("_parsed_dt")
        
        # Group by date and aggregate
        daily_rows = []
        
        for date_str, group in df.groupby("date", sort=True):
            # Valence aggregations
            valences = group["_valence"].dropna()
            mean_valence = valences.mean() if len(valences) > 0 else np.nan
            last_valence = valences.iloc[-1] if len(valences) > 0 else np.nan
            
            # Kind aggregation (most frequent)
            kinds = group["Kind"].dropna().tolist() if "Kind" in group.columns else []
            kind_dominant = Counter(kinds).most_common(1)[0][0] if kinds else ""
            
            # Labels aggregation (unique union)
            all_labels: Set[str] = set()
            if "Labels" in group.columns:
                for labels_str in group["Labels"].dropna():
                    all_labels.update(parse_pipe_separated(labels_str))
            
            # Associations aggregation (unique union)
            all_assocs: Set[str] = set()
            if "Associations" in group.columns:
                for assocs_str in group["Associations"].dropna():
                    all_assocs.update(parse_pipe_separated(assocs_str))
            
            daily_rows.append({
                "date": date_str,
                "som_mean_score": round(mean_valence, 6) if not pd.isna(mean_valence) else np.nan,
                "som_last_score": round(last_valence, 6) if not pd.isna(last_valence) else np.nan,
                "som_n_entries": len(group),
                "som_category_3class": classify_valence_3class(mean_valence),
                "som_kind_dominant": kind_dominant,
                "som_labels": ", ".join(sorted(all_labels)) if all_labels else "",
                "som_associations": ", ".join(sorted(all_assocs)) if all_assocs else "",
            })
        
        df_daily = pd.DataFrame(daily_rows)
        logger.info(f"[APPLE/AUTOEXPORT/SOM] Aggregated to {len(df_daily)} daily rows")
        
        return df_daily


# ============================================================================
# Public API Functions
# ============================================================================

def load_autoexport_som_daily(
    csv_path: Path | str,
    snapshot_date: Optional[str] = None
) -> pd.DataFrame:
    """Load and aggregate State of Mind data from Auto Export CSV.
    
    Primary entry point for loading SoM data.
    
    Args:
        csv_path: Path to StateOfMind-*.csv file
        snapshot_date: Optional date cutoff (YYYY-MM-DD)
        
    Returns:
        DataFrame with daily SoM aggregations.
        If no data exists, returns empty well-formed DataFrame.
    """
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.warning(f"[APPLE/AUTOEXPORT/SOM] CSV not found: {csv_path}")
        return _empty_som_daily()
    
    try:
        aggregator = SoMAggregator(csv_path)
        return aggregator.aggregate_daily(snapshot_date=snapshot_date)
    
    except Exception as e:
        logger.error(f"[APPLE/AUTOEXPORT/SOM] Failed to parse SoM data: {e}")
        return _empty_som_daily()


def run_som_aggregation(
    participant: str = "P000001",
    snapshot: Optional[str] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False
) -> Dict[str, str]:
    """Execute State of Mind aggregation stage.
    
    Uses the snapshot-aware Auto Export discovery to find the appropriate
    CSV file, then aggregates to daily metrics.
    
    Args:
        participant: Participant ID (e.g., "P000001")
        snapshot: Snapshot date YYYY-MM-DD
        output_dir: Output directory; defaults to snapshot extracted/apple/
        dry_run: If True, don't write files
        
    Returns:
        Dict mapping output names to file paths
    """
    if snapshot is None:
        snapshot = datetime.now().strftime("%Y-%m-%d")
    
    snapshot_dir = etl_snapshot_root(participant, snapshot)
    
    if output_dir is None:
        output_dir = str(snapshot_dir / "extracted" / "apple")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"SOM AGGREGATION - {participant}/{snapshot}")
    logger.info(f"{'='*80}\n")
    
    # Discover State of Mind Auto Export file from extracted directory
    som_csv_path = None
    
    # Primary: Look in extracted autoexport directory (populated by Stage 0)
    if get_autoexport_csv_path is not None:
        som_csv_path = get_autoexport_csv_path(participant, snapshot, "StateOfMind")
    
    if som_csv_path is None:
        # Fallback: direct search in extracted directory
        autoexport_dir = get_extracted_autoexport_dir(participant, snapshot)
        if autoexport_dir.exists():
            som_files = list(autoexport_dir.glob("StateOfMind*.csv"))
            if som_files:
                som_csv_path = som_files[0]
                logger.info(f"[APPLE/AUTOEXPORT/SOM] Found via fallback in extracted: {som_csv_path}")
    
    if som_csv_path is None or not som_csv_path.exists():
        logger.warning(f"[APPLE/AUTOEXPORT/SOM] No State of Mind data found for {participant}/{snapshot}")
        logger.info(f"[APPLE/AUTOEXPORT/SOM] Expected location: {get_extracted_autoexport_dir(participant, snapshot)}")
        logger.info("[APPLE/AUTOEXPORT/SOM] Tip: Ensure Stage 0 extracted an AutoExport ZIP with StateOfMind CSV")
        
        # Write empty CSV for consistency
        out_path = Path(output_dir) / "daily_som.csv"
        df_empty = _empty_som_daily()
        
        if not dry_run:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_empty.to_csv(out_path, index=False)
            logger.info(f"[APPLE/AUTOEXPORT/SOM] Wrote empty daily_som.csv to {out_path}")
        
        return {"daily_som": str(out_path)}
    
    # Load and aggregate
    logger.info(f"[APPLE/AUTOEXPORT/SOM] Processing: {som_csv_path}")
    df_som = load_autoexport_som_daily(som_csv_path, snapshot_date=snapshot)
    
    # Write output
    out_path = Path(output_dir) / "daily_som.csv"
    
    if not dry_run:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_som.to_csv(out_path, index=False)
        logger.info(f"[APPLE/AUTOEXPORT/SOM] ✓ Saved {len(df_som)} SoM days to {out_path}")
    else:
        logger.info(f"[APPLE/AUTOEXPORT/SOM] DRY RUN: Would save {len(df_som)} days to {out_path}")
    
    return {"daily_som": str(out_path)}


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for SoM aggregation."""
    if argv is None:
        import sys
        argv = sys.argv[1:]
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    parser = argparse.ArgumentParser(
        prog="som_from_autoexport",
        description="Aggregate State of Mind data from Apple Auto Export CSV"
    )
    parser.add_argument(
        "--pid",
        required=True,
        help="Participant ID (e.g., P000001)"
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot date YYYY-MM-DD"
    )
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=dry run (no writes), 0=normal"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory"
    )
    
    args = parser.parse_args(argv)
    
    results = run_som_aggregation(
        participant=args.pid,
        snapshot=args.snapshot,
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run)
    )
    
    print(f"\n[DONE] Outputs: {results}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
