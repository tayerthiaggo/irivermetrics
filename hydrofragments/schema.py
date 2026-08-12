"""Canonical HydroFragments v1.2 output schema."""

from __future__ import annotations

from enum import Enum
import re


SCHEMA_VERSION = "1.1.0"


class SchemaError(ValueError):
    """Raised when a metric record violates the frozen output contract."""


class MetricFamily(str, Enum):
    EXTENT = "extent"
    PERSISTENCE = "persistence"
    MORPHOLOGY = "morphology"
    FRAGMENTATION = "fragmentation"
    CLUSTERING = "clustering"
    CONNECTIVITY = "connectivity"
    DYNAMICS = "dynamics"
    DIAGNOSTIC = "diagnostic"


class Statistic(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MAX = "max"
    CV = "cv"
    Q10 = "q10"
    Q90 = "q90"


class ValueType(str, Enum):
    MONTHLY = "monthly"
    HY_ANCHOR = "HY_anchor"
    HY_SUMMARY = "HY_summary"
    RASTER_SUMMARY = "raster_summary"
    DIAGNOSTIC = "diagnostic"


class EdgeFlag(str, Enum):
    N0 = "N0"
    N1 = "N1"
    N2_UNSTABLE = "N2_unstable"
    LOW_VALID_OBS = "low_valid_obs"
    MISSING_HY_ANCHOR = "missing_HY_anchor"
    NO_PREVIOUS_HY = "no_previous_HY"
    NONCONSECUTIVE_HY = "nonconsecutive_HY"
    LOW_COMMON_VALID_SUPPORT = "low_common_valid_support"
    EMPTY_REFUGE_UNION = "empty_refuge_union"
    NO_THRESHOLD_CROSSING = "no_threshold_crossing"


class WarningFlag(str, Enum):
    NO_CHANNEL = "no_channel"
    COMPOSITE_SENSITIVE = "composite_sensitive"
    PROXY_RECONNECTION = "proxy_reconnection"
    WIDTH_RESOLUTION_FLOOR = "width_resolution_floor"
    AOI_NOT_COMPARABLE = "aoi_not_comparable"
    LENGTH_CRS_CAVEAT = "length_crs_caveat"
    PROXY_CHANNEL = "proxy_channel"


class HydrologicalState(str, Enum):
    DRY = "dry"
    FRAGMENTED_WET = "fragmented_wet"
    CONNECTED_WET = "connected_wet"


class MetricDependency(str, Enum):
    NONE = "none"
    VALIDITY = "requires_validity"
    PATCHES = "requires_patches"
    CHANNEL = "requires_channel"
    FIXED_NODES = "requires_fixed_nodes"
    HY_ANCHOR = "requires_HY_anchor"
    GRAPH = "requires_graph"
    PROXY_ALLOWED = "proxy_allowed"
    DUAL_COMPOSITE = "requires_dual_composite"
    MESH_VALIDATION = "requires_mesh_validation"
    WIDTH_FLOOR = "requires_width_floor"


OUTPUT_COLUMNS = (
    "schema_version",
    "run_id",
    "config_hash",
    "package_version",
    "git_sha",
    "catchment_id",
    "aoi_id",
    "zone",
    "window_id",
    "date",
    "hy",
    "hy_anchor",
    "metric",
    "metric_family",
    "statistic",
    "value",
    "unit",
    "value_type",
    "state",
    "n_pools",
    "n_valid_pixels",
    "n_water_pixels",
    "valid_fraction_month",
    "min_valid_fraction_month",
    "low_coverage_flag",
    "edge_flag",
    "warning_flags",
    "is_reportable",
    "hy_confidence",
    "composite_sensitive",
    "source",
    "resolution_m",
    "crs",
    "area_unit",
    "length_unit",
    "monthly_composite",
    "water_threshold",
    "threshold_method",
    "min_patch_pixels",
    "min_patch_area_m2",
    "connectivity_rule",
    "metric_dependency",
    "proxy_channel",
    "awre_length_method",
    "node_source",
    "connected_wet_metric",
    "connected_wet_threshold",
    "reconnection_metric_used",
    "proxy_reconnection_flag",
)


OUTPUT_DTYPES = {
    "schema_version": "string",
    "run_id": "string",
    "config_hash": "string",
    "package_version": "string",
    "git_sha": "string",
    "catchment_id": "string",
    "aoi_id": "string",
    "zone": "string",
    "window_id": "string",
    "date": "datetime64[ns]",
    "hy": "Int64",
    "hy_anchor": "string",
    "metric": "string",
    "metric_family": "string",
    "statistic": "string",
    "value": "Float64",
    "unit": "string",
    "value_type": "string",
    "state": "string",
    "n_pools": "Int64",
    "n_valid_pixels": "Int64",
    "n_water_pixels": "Int64",
    "valid_fraction_month": "Float64",
    "min_valid_fraction_month": "Float64",
    "low_coverage_flag": "boolean",
    "edge_flag": "string",
    "warning_flags": "list[string]",
    "is_reportable": "boolean",
    "hy_confidence": "string",
    "composite_sensitive": "boolean",
    "source": "string",
    "resolution_m": "Float64",
    "crs": "string",
    "area_unit": "string",
    "length_unit": "string",
    "monthly_composite": "string",
    "water_threshold": "Float64",
    "threshold_method": "string",
    "min_patch_pixels": "Int64",
    "min_patch_area_m2": "Float64",
    "connectivity_rule": "Int64",
    "metric_dependency": "string",
    "proxy_channel": "boolean",
    "awre_length_method": "string",
    "node_source": "string",
    "connected_wet_metric": "string",
    "connected_wet_threshold": "Float64",
    "reconnection_metric_used": "string",
    "proxy_reconnection_flag": "boolean",
}


FORBIDDEN_METRIC_IDS = frozenset(
    {
        "pf",
        "plf",
        "awmpa",
        "awmpl",
        "awmpw",
        "pcf",
        "nni",
        "degree_centrality",
        "betweenness_centrality",
    }
)


def normalize_metric_id(metric_id: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", metric_id.lower())).strip(
        "_"
    )


def validate_metric_id(metric_id: str) -> str:
    normalized = normalize_metric_id(metric_id)
    if normalized in FORBIDDEN_METRIC_IDS:
        raise SchemaError(f"forbidden v1.2 metric id: {metric_id}")
    if metric_id != normalized or not re.fullmatch(r"[a-z][a-z0-9_]*", metric_id):
        raise SchemaError(f"metric id must be canonical snake case: {metric_id}")
    return metric_id


__all__ = [
    "EdgeFlag",
    "FORBIDDEN_METRIC_IDS",
    "HydrologicalState",
    "MetricDependency",
    "MetricFamily",
    "OUTPUT_COLUMNS",
    "OUTPUT_DTYPES",
    "SCHEMA_VERSION",
    "SchemaError",
    "Statistic",
    "ValueType",
    "WarningFlag",
    "normalize_metric_id",
    "validate_metric_id",
]
