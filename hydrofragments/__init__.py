"""HydroFragments v1.2 — surface-water dynamics metrics for intermittent rivers."""

from hydrofragments._version import __version__
from hydrofragments.api import (
    HydroConfig,
    HydroResult,
    SCHEMA_VERSION,
    ValidationReport,
    WaterCube,
    analyze,
    compare_results,
    open_water_cube,
    validate_inputs,
)

__all__ = [
    "HydroConfig",
    "HydroResult",
    "SCHEMA_VERSION",
    "ValidationReport",
    "WaterCube",
    "__version__",
    "analyze",
    "compare_results",
    "open_water_cube",
    "validate_inputs",
]
