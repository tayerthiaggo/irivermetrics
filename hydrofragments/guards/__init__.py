"""Scientific and cross-run comparison guards."""

from hydrofragments.guards.comparison import (
    COMPARISON_FIELDS,
    ComparisonApproval,
    ComparisonGuardError,
    ComparisonMismatch,
    check_comparison_compatibility,
    compare_manifests,
    guard_comparison,
)
from hydrofragments.guards.quality import (
    WATERMASK_TSFILL_HINT,
    BaselineQualityReport,
    assess_baseline_quality,
)
from hydrofragments.guards.scientific import (
    ScientificGuardError,
    guard_aoi_comparability,
    guard_area_metric_crs,
    guard_persistence_zone,
)

__all__ = [
    "COMPARISON_FIELDS",
    "WATERMASK_TSFILL_HINT",
    "BaselineQualityReport",
    "ComparisonApproval",
    "ComparisonGuardError",
    "ComparisonMismatch",
    "ScientificGuardError",
    "assess_baseline_quality",
    "check_comparison_compatibility",
    "compare_manifests",
    "guard_aoi_comparability",
    "guard_area_metric_crs",
    "guard_comparison",
    "guard_persistence_zone",
]
