"""Features module: data unification (Apple + Zepp)."""

from .unify_daily import (
    unify_daily,
    merge_apple_zepp,
    validate_and_report,
)

__all__ = [
    "unify_daily",
    "merge_apple_zepp",
    "validate_and_report",
]
