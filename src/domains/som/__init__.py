"""State of Mind (SoM) domain package.

This domain handles mood/emotion signals from Apple Health Auto Export.
"""
from .som_from_autoexport import (
    load_autoexport_som_daily,
    run_som_aggregation,
    SoMAggregator,
    _empty_som_daily,
)

__all__ = [
    "load_autoexport_som_daily",
    "run_som_aggregation",
    "SoMAggregator",
    "_empty_som_daily",
]
