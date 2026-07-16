# init
"""Temporal cadence, monthly composite, and hydroyear adapter contracts."""

from hydrofragments.temporal.composites import (
    CompositeError,
    build_monthly_products,
)
from hydrofragments.temporal.hydroyear import (
    HyAnchorResult,
    HydroYearAdapterError,
    detect_hy_anchors,
    hydroseason_config_from_hydroconfig,
    hydroseason_config_to_mapping,
)

__all__ = [
    "CompositeError",
    "HyAnchorResult",
    "HydroYearAdapterError",
    "build_monthly_products",
    "detect_hy_anchors",
    "hydroseason_config_from_hydroconfig",
    "hydroseason_config_to_mapping",
]
