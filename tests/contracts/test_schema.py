from __future__ import annotations

from datetime import datetime

import pytest


EXPECTED_COLUMNS = (
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


EXPECTED_DTYPES = {
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


def base_record(**changes: object):
    from hydrofragments.models import MetricRecord
    from hydrofragments.schema import (
        MetricDependency,
        MetricFamily,
        SCHEMA_VERSION,
        ValueType,
    )

    values: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "run-001",
        "config_hash": "a" * 64,
        "package_version": "0.1.0",
        "git_sha": "abc123",
        "catchment_id": "fitzroy",
        "aoi_id": "reach-01",
        "date": datetime(2026, 1, 1),
        "metric": "apsec",
        "metric_family": MetricFamily.EXTENT,
        "value": 12.5,
        "unit": "percent",
        "value_type": ValueType.MONTHLY,
        "is_reportable": True,
        "metric_dependency": MetricDependency.NONE,
    }
    values.update(changes)
    return MetricRecord(**values)


def test_output_schema_columns_and_dtypes_are_exact() -> None:
    from hydrofragments.schema import OUTPUT_COLUMNS, OUTPUT_DTYPES, SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.1.0"
    assert OUTPUT_COLUMNS == EXPECTED_COLUMNS
    assert OUTPUT_DTYPES == EXPECTED_DTYPES


def test_output_enums_are_exact() -> None:
    from hydrofragments.schema import (
        EdgeFlag,
        HydrologicalState,
        MetricDependency,
        MetricFamily,
        Statistic,
        ValueType,
        WarningFlag,
    )

    assert {item.value for item in MetricFamily} == {
        "extent",
        "persistence",
        "morphology",
        "fragmentation",
        "clustering",
        "connectivity",
        "dynamics",
        "diagnostic",
    }
    assert {item.value for item in Statistic} == {
        "mean",
        "median",
        "max",
        "cv",
        "q10",
        "q90",
    }
    assert {item.value for item in ValueType} == {
        "monthly",
        "HY_anchor",
        "HY_summary",
        "raster_summary",
        "diagnostic",
    }
    assert {item.value for item in EdgeFlag} == {
        "N0",
        "N1",
        "N2_unstable",
        "low_valid_obs",
        "missing_HY_anchor",
        "no_previous_HY",
        "nonconsecutive_HY",
        "low_common_valid_support",
        "empty_refuge_union",
        "no_threshold_crossing",
    }
    assert {item.value for item in WarningFlag} == {
        "no_channel",
        "composite_sensitive",
        "proxy_reconnection",
        "width_resolution_floor",
        "aoi_not_comparable",
        "length_crs_caveat",
        "proxy_channel",
    }
    assert {item.value for item in HydrologicalState} == {
        "dry",
        "fragmented_wet",
        "connected_wet",
    }
    assert {item.value for item in MetricDependency} == {
        "none",
        "requires_validity",
        "requires_patches",
        "requires_channel",
        "requires_fixed_nodes",
        "requires_HY_anchor",
        "requires_graph",
        "proxy_allowed",
        "requires_dual_composite",
        "requires_mesh_validation",
        "requires_width_floor",
    }


@pytest.mark.parametrize(
    "metric_id",
    [
        "PF",
        "PLF",
        "AWMPA",
        "AWMPL",
        "AWMPW",
        "PCF",
        "NNI",
        "degree_centrality",
        "betweenness_centrality",
    ],
)
def test_forbidden_metric_ids_are_rejected(metric_id: str) -> None:
    from hydrofragments.schema import SchemaError

    with pytest.raises(SchemaError, match="forbidden"):
        base_record(metric=metric_id)


def test_metric_record_serializes_in_schema_order_with_enum_values() -> None:
    record = base_record()

    row = record.to_mapping()

    assert tuple(row) == EXPECTED_COLUMNS
    assert row["metric_family"] == "extent"
    assert row["value_type"] == "monthly"
    assert row["metric_dependency"] == "none"
    assert row["warning_flags"] == []


def test_low_valid_suppression_clears_value_and_is_not_reportable() -> None:
    from hydrofragments.config import LowSupportBehavior
    from hydrofragments.schema import EdgeFlag

    record = base_record(value=42.0).with_low_valid(
        LowSupportBehavior.SUPPRESS_VALUE
    )

    assert record.value is None
    assert record.edge_flag is EdgeFlag.LOW_VALID_OBS
    assert record.is_reportable is False


def test_low_valid_flagged_value_is_retained_but_not_reportable() -> None:
    from hydrofragments.config import LowSupportBehavior
    from hydrofragments.schema import EdgeFlag

    record = base_record(value=42.0).with_low_valid(
        LowSupportBehavior.EMIT_FLAGGED_VALUE
    )

    assert record.value == 42.0
    assert record.edge_flag is EdgeFlag.LOW_VALID_OBS
    assert record.is_reportable is False


def test_n0_patch_record_is_present_with_nan_semantics() -> None:
    from hydrofragments.schema import EdgeFlag, HydrologicalState, MetricFamily

    record = base_record(
        metric="awre",
        metric_family=MetricFamily.MORPHOLOGY,
        value=0.0,
    ).with_patch_count(0)

    assert record.n_pools == 0
    assert record.value is None
    assert record.edge_flag is EdgeFlag.N0
    assert record.state is HydrologicalState.DRY
