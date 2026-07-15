"""M8 — hydrofragments public namespace imports cleanly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

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
