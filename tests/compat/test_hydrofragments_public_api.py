"""M8 — hydrofragments public namespace imports cleanly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from datetime import datetime

import hydrofragments
from hydrofragments import (
    HydroConfig,
    HydroResult,
    ValidationReport,
    WaterCube,
    analyze,
    compare_results,
    open_water_cube,
    validate_inputs,
)
from hydrofragments.schema import SCHEMA_VERSION
from hydrofragments.metrics.extent import ApsecRecord


def test_package_version_exposed() -> None:
    assert hasattr(hydrofragments, "__version__")
    assert hydrofragments.__version__.startswith("1.2.")


def test_public_api_symbols_are_exported() -> None:
    assert hydrofragments.__all__ == [
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


def test_open_water_cube_from_generic_binary_array() -> None:
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    assert isinstance(cube, WaterCube)
    assert cube.water.dtype == bool
    assert cube.valid_obs.dtype == bool
    assert cube.cadence == "monthly"


def test_validate_inputs_reports_capabilities(tmp_path) -> None:
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.ones((2, 2, 2), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    report = validate_inputs(cube, aoi_id="demo", config=config)
    assert isinstance(report, ValidationReport)
    assert report.is_valid
    assert "contracts_core" in report.resolved_profiles


def test_analyze_returns_tidy_core_metrics_without_forbidden_ids(tmp_path) -> None:
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.array(
            [
                [[1, 1, 0], [0, 0, 0], [0, 0, 0]],
                [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
            ],
            dtype=bool,
        ),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )
    result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=30.0)
    assert isinstance(result, HydroResult)
    metrics = result.metrics_table
    assert len(metrics) > 0
    assert set(metrics["metric"]).isdisjoint(
        {"pf", "plf", "awmpa", "awmpl", "awmpw", "nni"}
    )
    assert (tmp_path / "run_manifest.json").exists()


def test_analyze_emits_pixel_temporal_profile_rows(tmp_path) -> None:
    times = pd.to_datetime(
        ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"]
    )
    water = xr.DataArray(
        np.array(
            [
                [[1, 0]], [[0, 0]],
                [[1, 1]], [[0, 0]],
            ],
            dtype=bool,
        ),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["pixel_temporal"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )

    result = analyze(cube, aoi_id="demo", config=config)

    rows = result.metrics_table[result.metrics_table["metric"].isin(
        ["recurrence", "hydroperiod"]
    )]
    assert set(rows["metric"]) == {"recurrence", "hydroperiod"}
    assert rows["is_reportable"].all()


def test_analyze_calls_hydroseason_when_hydroyear_extent_is_supplied(tmp_path) -> None:
    times = pd.date_range("2020-01-01", periods=36, freq="MS")
    water = xr.DataArray(
        np.ones((36, 1, 1), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )
    extent = pd.Series(np.tile([70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55], 3), index=times)

    analyze(cube, aoi_id="demo", config=config, hydroyear_extent=extent)

    manifest = __import__("json").loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["comparison"]["hydroseason_hy_count"] > 0


def test_analyze_emits_dual_composite_contraction_rows(tmp_path) -> None:
    times = pd.date_range("2001-01-01", periods=36, freq="MS")
    water = xr.DataArray(
        np.ones((36, 1, 1), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["dynamics"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "max_water",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )
    extent_values = np.tile([70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55], 3)
    extent = pd.Series(extent_values, index=times)
    max_records = [
        ApsecRecord(date=date.to_pydatetime(), value=float(value), n_water_pixels=0, a_ref_m2=1.0, cell_area_m2=1.0)
        for date, value in zip(times, extent_values)
    ]
    median_records = [
        ApsecRecord(date=date.to_pydatetime(), value=float(value - 1), n_water_pixels=0, a_ref_m2=1.0, cell_area_m2=1.0)
        for date, value in zip(times, extent_values)
    ]

    result = analyze(
        cube,
        aoi_id="demo",
        config=config,
        hydroyear_extent=extent,
        max_water_apsec=max_records,
        median_apsec=median_records,
    )

    rows = result.metrics_table[result.metrics_table["metric"] == "extent_contraction"]
    assert set(rows["monthly_composite"]) == {"max_water", "median"}
    assert rows["hy_confidence"].notna().all()


def test_compare_results_rejects_mismatched_validity_policy() -> None:
    from hydrofragments.guards.comparison import ComparisonGuardError

    left = {
        "run_id": "left",
        "comparison": {
            "aoi_id": "a",
            "source": "demo",
            "resolution_m": 30.0,
            "crs": "EPSG:3577",
            "validity_policy": "p_native_season_stratified_v1",
            "monthly_composite": "supplied",
        },
    }
    right = {
        "run_id": "right",
        "comparison": {
            "aoi_id": "a",
            "source": "demo",
            "resolution_m": 30.0,
            "crs": "EPSG:3577",
            "validity_policy": "other_policy",
            "monthly_composite": "supplied",
        },
    }
    with pytest.raises(ComparisonGuardError, match="validity"):
        compare_results(left, right)
