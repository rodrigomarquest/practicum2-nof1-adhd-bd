"""Static deterministic source prioritizer for multi-vendor data fusion.

This module provides a simple, deterministic mechanism to select the best
available data source for each logical domain when multiple vendors may
provide overlapping signals.

Priority order is fixed and reproducible:
1. apple_export      — Official Apple Health export.xml (most complete)
2. apple_autoexport  — Auto Export iOS app (ZIP → CSVs)
3. zepp_cloud        — Zepp Cloud exports
4. fallback          — Empty well-formed DataFrame

Design principles:
- DETERMINISTIC: Same inputs always produce same output
- NON-BREAKING: Returns fallback if no data available
- TRACEABLE: Logs selected source for QC
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("etl.source_prioritizer")

# Static deterministic priority order
# First non-empty source wins
SOURCE_PRIORITY: List[str] = [
    "apple_export",       # Official Apple Health export.xml
    "apple_autoexport",   # Auto Export iOS app (ZIP → CSVs)
    "zepp_cloud",         # Zepp Cloud exports
    "fallback",           # Empty well-formed DataFrame (sentinel)
]


def prioritize_source(
    vendor_dataframes: Dict[str, pd.DataFrame],
    fallback_df: pd.DataFrame,
    domain: str,
) -> Tuple[pd.DataFrame, str]:
    """Select the first non-empty DataFrame according to SOURCE_PRIORITY.
    
    This function iterates through SOURCE_PRIORITY and returns the first
    vendor DataFrame that is non-empty. If all sources are empty or missing,
    returns the fallback DataFrame.
    
    Args:
        vendor_dataframes: Dict mapping vendor keys to DataFrames.
            Keys should match SOURCE_PRIORITY values (e.g., "apple_export").
            Missing keys are treated as empty/unavailable.
        fallback_df: Empty but well-formed DataFrame with canonical schema.
            Used if all vendor sources are empty.
        domain: Domain name for logging (e.g., "meds", "som", "cardio").
        
    Returns:
        Tuple of (selected_df, vendor_key):
        - selected_df: The chosen DataFrame (or fallback if all empty)
        - vendor_key: The source key that was selected (e.g., "apple_export", "fallback")
        
    Example:
        >>> vendor_dfs = {
        ...     "apple_export": df_from_xml,      # 0 rows
        ...     "apple_autoexport": df_from_csv,  # 50 rows
        ... }
        >>> df, vendor = prioritize_source(vendor_dfs, empty_df, "meds")
        >>> print(vendor)  # "apple_autoexport"
    """
    for source in SOURCE_PRIORITY:
        if source == "fallback":
            # Reached end of priority list
            break
            
        if source in vendor_dataframes:
            df = vendor_dataframes[source]
            if df is not None and len(df) > 0:
                logger.info(f"[{domain}] Selected vendor: {source} ({len(df)} rows)")
                return df.copy(), source
    
    # No non-empty source found
    logger.warning(f"[{domain}] No data from any vendor, using fallback (0 rows)")
    return fallback_df.copy(), "fallback"


def get_available_sources(
    vendor_dataframes: Dict[str, pd.DataFrame],
) -> Dict[str, int]:
    """Get summary of available sources and their row counts.
    
    Useful for QC and debugging to understand what data is available
    before prioritization.
    
    Args:
        vendor_dataframes: Dict mapping vendor keys to DataFrames.
        
    Returns:
        Dict mapping vendor keys to row counts.
        
    Example:
        >>> summary = get_available_sources(vendor_dfs)
        >>> print(summary)
        {"apple_export": 0, "apple_autoexport": 50, "zepp_cloud": 30}
    """
    summary = {}
    for source in SOURCE_PRIORITY:
        if source == "fallback":
            continue
        if source in vendor_dataframes:
            df = vendor_dataframes[source]
            summary[source] = len(df) if df is not None else 0
    return summary


def log_source_summary(
    vendor_dataframes: Dict[str, pd.DataFrame],
    domain: str,
) -> None:
    """Log a summary of available sources for a domain.
    
    Args:
        vendor_dataframes: Dict mapping vendor keys to DataFrames.
        domain: Domain name for logging.
    """
    summary = get_available_sources(vendor_dataframes)
    if summary:
        parts = [f"{k}={v}" for k, v in summary.items()]
        logger.info(f"[{domain}] Available sources: {', '.join(parts)}")
    else:
        logger.info(f"[{domain}] No sources available")
