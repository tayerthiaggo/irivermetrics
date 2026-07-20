"""open_water_cube auto-detection + input_contract wiring (Section 1).

Covers: dropping the hardcoded zarr->TSFill assumption, input_kind
defaulting to None (auto-detect), variable_map actually being threaded
through, and WaterCube.provenance carrying the resolved adapter name plus
any auto-fixes applied.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
import rioxarray
import pytest

from hydrofragments.api import open_water_cube
from hydrofragments.io.input_contract import InputContractError


def _times(n):
    return np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")


def _projected_bool_da(shape=(4, 3, 3), crs="EPSG:3577"):
    t, y, x = shape
    data = np.zeros(shape, dtype=bool)
    data[:, 0, 0] = True
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0
    da_arr = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": _times(t), "y": ys, "x": xs},
    )
    return da_arr.rio.write_crs(crs)


# ---- TSFill parity (regression) --------------------------------------------


def test_tsfill_zarr_path_still_routes_through_watermask_tsfill_adapter(
    tmp_zarr_path,
):
    cube = open_water_cube(tmp_zarr_path)

    assert dict(cube.provenance).get("adapter") == "watermask_tsfill"


def test_tsfill_zarr_output_unchanged_with_explicit_override(tmp_zarr_path):
    """Auto-detect and an explicit override must produce identical output."""
    auto_cube = open_water_cube(tmp_zarr_path)
    explicit_cube = open_water_cube(tmp_zarr_path, input_kind="watermask_tsfill")

    np.testing.assert_array_equal(auto_cube.water.values, explicit_cube.water.values)
    np.testing.assert_array_equal(
        auto_cube.valid_obs.values, explicit_cube.valid_obs.values
    )


# ---- auto-detect default ----------------------------------------------------


def test_input_kind_defaults_to_none_and_auto_detects_generic_binary():
    water = _projected_bool_da()

    cube = open_water_cube(water)

    assert dict(cube.provenance).get("adapter") == "generic_binary"


def test_explicit_input_kind_override_skips_auto_detection():
    # A {0,1} float array would auto-detect as generic_binary; force raw_wofs
    # explicitly and confirm the override wins.
    data = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    cube = open_water_cube(da_in, input_kind="raw_wofs")

    assert dict(cube.provenance).get("adapter") == "raw_wofs"


# ---- raw_wofs end-to-end through open_water_cube ---------------------------


def test_raw_wofs_threshold_path_end_to_end():
    data = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    cube = open_water_cube(
        da_in,
        input_kind="raw_wofs",
        water_threshold=0.5,
    )

    np.testing.assert_array_equal(cube.water.values, [False, False, True, True])


def test_raw_wofs_binary_band_end_to_end_no_threshold_needed():
    data = np.array([0, 1, 1, 0], dtype=np.float32)
    ds = xr.Dataset({"water": (("time",), data)}, coords={"time": _times(4)})

    cube = open_water_cube(ds)  # auto-detects raw_wofs via 'water' variable

    assert dict(cube.provenance).get("adapter") == "raw_wofs"
    np.testing.assert_array_equal(cube.water.values, [False, True, True, False])


# ---- variable_map is actually used ------------------------------------------


def test_variable_map_renames_before_parsing():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset({"my_band": (("time",), data)}, coords={"time": _times(4)})

    cube = open_water_cube(ds, variable_map={"my_band": "water"})

    np.testing.assert_array_equal(cube.water.values, [False, True, True, False])
    fixes = dict(cube.provenance).get("auto_fixes", "")
    assert "my_band" in fixes


# ---- safe auto-fixes are applied and logged into provenance ---------------


def test_single_variable_dataset_auto_rename_is_logged():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset({"mask": (("time",), data)}, coords={"time": _times(4)})

    cube = open_water_cube(ds)

    fixes = dict(cube.provenance).get("auto_fixes", "")
    assert "mask" in fixes and "renamed" in fixes


def test_dtype_coercion_is_logged():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])

    cube = open_water_cube(da_in)

    fixes = dict(cube.provenance).get("auto_fixes", "")
    assert "coerced" in fixes


def test_dim_reorder_is_logged():
    # (y, time, x) -- wrong order relative to the expected (time, y, x).
    data = np.zeros((3, 4, 2), dtype=bool)
    da_in = xr.DataArray(data, dims=["y", "time", "x"])

    cube = open_water_cube(da_in)

    assert cube.water.dims == ("time", "y", "x")
    fixes = dict(cube.provenance).get("auto_fixes", "")
    assert "reorder" in fixes


# ---- grid mismatch / CRS refusal (never silently reprojects) --------------


def test_grid_mismatch_between_water_and_valid_obs_raises():
    water = _projected_bool_da(shape=(4, 3, 3))
    valid = _projected_bool_da(shape=(4, 4, 4))

    with pytest.raises(InputContractError, match="grid"):
        open_water_cube(water, valid_obs=valid)


def test_degrees_crs_input_is_refused_not_reprojected():
    water = _projected_bool_da(crs="EPSG:4326")

    with pytest.raises(InputContractError, match="degrees|geographic"):
        open_water_cube(water)


# ---- config test lives in tests/contracts/test_config.py (see brief) ------
