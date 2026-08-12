"""M2: single per-month patch bundle must match the two separate legacy calls."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.section_analysis import analyze_section_rows
from hydrofragments.metrics import patches
from tests.fixtures.analytic_masks import (
    invalid_water_observation_fixture,
    patch_bundle_width_fixture,
)


def _mask():
    m = np.zeros((12, 12), dtype=bool)
    m[2:5, 2:5] = True
    m[7:10, 7:11] = True
    return m


def test_bundle_matches_separate_calls():
    mask = _mask()
    core_ref = patches.analyze_patch_metrics(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0, include_mesh=True
    )
    width_ref = patches.analyze_pool_width_distribution(
        mask, pixel_size_m=30.0, resolution_floor_pixels=2.0
    )
    core, width = patches.analyze_patch_bundle(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
        include_mesh=True, include_width=True, resolution_floor_pixels=2.0,
    )
    assert core == core_ref
    assert width == width_ref


def test_bundle_labels_once():
    mask = _mask()
    with mock.patch.object(
        patches, "label_components", wraps=patches.label_components
    ) as spy:
        patches.analyze_patch_bundle(
            mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
            include_mesh=False, include_width=True, resolution_floor_pixels=2.0,
        )
    assert spy.call_count == 1


def _core_plus_width_config(tmp_path) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core", "secondary"],
            "input": {"kind": "generic_binary"},
            "patches": {
                "connectivity_rule": 4,
                "min_patch_pixels": 3,
                "width_resolution_floor_pixels": 2.0,
            },
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )


def _single_month_bundle_cube():
    water_np, times = patch_bundle_width_fixture()
    water = xr.DataArray(water_np, dims=("time", "y", "x"), coords={"time": times})
    return open_water_cube(water, input_kind="generic_binary")


def test_open_water_cube_rejects_water_without_valid_observation():
    water_np, valid_np, times = invalid_water_observation_fixture()
    water = xr.DataArray(
        water_np,
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    valid_obs = xr.DataArray(
        valid_np,
        dims=("time", "y", "x"),
        coords={"time": times},
    )

    with pytest.raises(ValueError, match="water=True requires valid_obs=True"):
        open_water_cube(water, valid_obs=valid_obs, input_kind="generic_binary")


def test_analyze_section_rows_rejects_water_without_valid_observation():
    water_np, valid_np, times = invalid_water_observation_fixture()
    water = xr.DataArray(
        water_np.astype(bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    valid_obs = xr.DataArray(
        valid_np,
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    config = _core_plus_width_config("unused")

    with pytest.raises(ValueError, match="water=True requires valid_obs=True"):
        analyze_section_rows(
            water,
            section="demo",
            section_area_km2=0.0004,
            pixel_size_m=10.0,
            config=config,
            selected_ids={"number_of_pools", "pool_width"},
            valid_obs=valid_obs,
        )


def test_analyze_core_patches_and_pool_width_share_one_bundle(tmp_path):
    """Core patches and pool width share one measurement pass per month.

    Export-off analysis uses the lightweight ``_run_month_rows`` path, which
    labels through :func:`measure_patch_properties`. Spatial-export runs use
    ``label_and_measure_window`` via ``stream_section_month_rows``.
    """
    cube = _single_month_bundle_cube()
    config = _core_plus_width_config(tmp_path)

    with mock.patch.object(
        patches, "label_components", wraps=patches.label_components
    ) as spy:
        result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=10.0)

    assert spy.call_count == 1
    assert {"number_of_pools", "lpi", "awre", "awmsi", "pool_width"} <= set(
        result.metrics_table["metric"]
    )
