"""
Device segmentation mapping S1-S6 based on firmware versions.

Segmentation definitions:
- S1 (2023-01-01 to 2023-12-04): iOS 16.3, GTR 2, no Ring
- S2 (2023-12-13 to 2024-08-29): iOS 17.0, GTR 4, no Ring
- S3 (2024-08-30 to 2024-11-02): iOS 17.0, GTR 4, Ring (pre-2.3.1)
- S4 (2024-11-03 to 2025-04-10): iOS 17.5, GTR 4, Ring (pre-2.3.1)
- S5 (2025-04-11 to 2025-07-25): iOS 17.5, GTR 4, Ring 2.3.1
- S6 (2025-07-26 to 2025-11-07): iOS 17.5, GTR 4, Ring 3.1.2.3

Use for:
1. Device-specific data quality assessment
2. Cross-device validation (Apple vs Zepp per segment)
3. Firmware effect analysis
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Device constants
GTR4_DEVICE = "GTR 4"
GTR2_DEVICE = "GTR 2"
IOS170 = "17.0"
IOS175 = "17.5"
IOS163 = "16.3"


class SegmentationConfig:
    """Device segmentation boundaries and metadata."""

    SEGMENTS = {
        "S1": {
            "start": pd.Timestamp("2023-01-01"),
            "end": pd.Timestamp("2023-12-12"),
            "ios": IOS163,
            "smartwatch": GTR2_DEVICE,
            "ring": False,
            "data_sources": ["apple"],
            "notes": "iPhone only period",
        },
        "S2": {
            "start": pd.Timestamp("2023-12-13"),
            "end": pd.Timestamp("2024-08-29"),
            "ios": IOS170,
            "smartwatch": GTR4_DEVICE,
            "ring": False,
            "data_sources": ["apple", "zepp"],
            "notes": "GTR2 → GTR4 upgrade, Zepp starts",
        },
        "S3": {
            "start": pd.Timestamp("2024-08-30"),
            "end": pd.Timestamp("2024-11-02"),
            "ios": IOS170,
            "smartwatch": GTR4_DEVICE,
            "ring": True,
            "ring_firmware": "pre-2.3.1",
            "data_sources": ["apple", "zepp", "ring"],
            "notes": "Ring debuts (firmware < 2.3.1)",
        },
        "S4": {
            "start": pd.Timestamp("2024-11-03"),
            "end": pd.Timestamp("2025-04-10"),
            "ios": IOS175,
            "smartwatch": GTR4_DEVICE,
            "ring": True,
            "ring_firmware": "pre-2.3.1",
            "data_sources": ["apple", "zepp", "ring"],
            "notes": "iOS upgrade, Ring stable",
        },
        "S5": {
            "start": pd.Timestamp("2025-04-11"),
            "end": pd.Timestamp("2025-07-25"),
            "ios": IOS175,
            "smartwatch": GTR4_DEVICE,
            "ring": True,
            "ring_firmware": "2.3.1",
            "data_sources": ["apple", "zepp", "ring"],
            "notes": "Ring firmware 2.3.1",
        },
        "S6": {
            "start": pd.Timestamp("2025-07-26"),
            "end": pd.Timestamp("2025-11-07"),
            "ios": IOS175,
            "smartwatch": GTR4_DEVICE,
            "ring": True,
            "ring_firmware": "3.1.2.3",
            "data_sources": ["apple", "zepp", "ring"],
            "notes": "Ring firmware upgrade to 3.1.2.3",
        },
    }

    @classmethod
    def get_segment_for_date(cls, date: pd.Timestamp) -> str:
        """Return segment name (S1-S6) for given date."""
        for seg_name, seg_config in cls.SEGMENTS.items():
            if seg_config["start"] <= date <= seg_config["end"]:
                return seg_name
        logger.warning(f"Date {date} not in any segment, returning 'UNKNOWN'")
        return "UNKNOWN"

    @classmethod
    def get_data_sources_for_segment(cls, segment: str) -> List[str]:
        """Return available data sources for segment."""
        if segment in cls.SEGMENTS:
            return cls.SEGMENTS[segment]["data_sources"]
        return []

    @classmethod
    def get_available_segments_for_source(cls, source: str) -> List[Tuple[str, pd.Timestamp, pd.Timestamp]]:
        """Return list of (segment_name, start, end) where source is available."""
        result = []
        for seg_name, seg_config in cls.SEGMENTS.items():
            if source in seg_config["data_sources"]:
                result.append((seg_name, seg_config["start"], seg_config["end"]))
        return result


def assign_segmentation(df: pd.DataFrame, date_col: str = "date", inplace: bool = False) -> pd.DataFrame:
    """
    Assign segment (S1-S6) to each row based on date.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column
    date_col : str
        Name of date column (must be datetime)
    inplace : bool
        If True, modify df; else return copy

    Returns
    -------
    pd.DataFrame
        DataFrame with new 'segment' column (S1-S6)
    """
    if not inplace:
        df = df.copy()

    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found. Available: {df.columns.tolist()}")

    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])

    df["segment"] = df[date_col].apply(SegmentationConfig.get_segment_for_date)

    segment_counts = df["segment"].value_counts()
    logger.info(f"Segmentation applied. Distribution:\n{segment_counts}")

    return df


def get_segment_date_range(segment: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Get (start, end) dates for segment."""
    if segment not in SegmentationConfig.SEGMENTS:
        raise ValueError(f"Unknown segment: {segment}. Valid: {list(SegmentationConfig.SEGMENTS.keys())}")
    cfg = SegmentationConfig.SEGMENTS[segment]
    return cfg["start"], cfg["end"]


def filter_by_segment(df: pd.DataFrame, segment: str, date_col: str = "date") -> pd.DataFrame:
    """Filter DataFrame to records within segment date range."""
    start, end = get_segment_date_range(segment)
    if date_col not in df.columns:
        raise ValueError(f"Column '{date_col}' not found")
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
    return df[(df[date_col] >= start) & (df[date_col] <= end)].copy()
