"""
Biomarkers module for ADHD/BD clinical feature extraction.

Tier 1: Core biomarcadores (HRV, Sleep %, HR CV, Activity variance, Sedentariness, Circadian)
Tier 2: Clinical biomarcadores (Sleep timing, HR trends, Body composition)
Tier X: Exploratory features (cross-device validation, device reliability, advanced metrics)

Architecture:
- segmentation.py: Map device/firmware per record (S1-S6)
- hrv.py: HRV SDNN/RMSSD/pNN50 extraction
- sleep.py: Sleep stages, latency, fragmentation (JSON naps parsing)
- activity.py: Activity variance, fragmentation, intensity detection
- circadian.py: Nocturnal activity ratio, sleep timing, peak hours
- validators.py: Cross-device correlation, data quality checks
- aggregate.py: Compile all metrics into daily features

All functions respect modular principle - no data duplication, input validation on all functions.
"""

from .segmentation import SegmentationConfig, assign_segmentation
from .hrv import compute_hrv_daily
from .sleep import parse_sleep_records, compute_sleep_metrics
from .activity import compute_activity_metrics
from .circadian import compute_circadian_metrics
from .validators import validate_cross_device, get_data_quality_flags
from .aggregate import aggregate_daily_biomarkers

__all__ = [
    "SegmentationConfig",
    "assign_segmentation",
    "compute_hrv_daily",
    "parse_sleep_records",
    "compute_sleep_metrics",
    "compute_activity_metrics",
    "compute_circadian_metrics",
    "validate_cross_device",
    "get_data_quality_flags",
    "aggregate_daily_biomarkers",
]

__version__ = "2.0.0"  # Clinical-grade biomarkers
