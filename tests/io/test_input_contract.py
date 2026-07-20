from __future__ import annotations

import numpy as np
import xarray as xr
import rioxarray
import pytest

from hydrofragments.io.input_contract import (
    InputContractError,
    check_crs_defined,
    check_grid_alignment,
    normalize_structure,
)


def _times(n):
    return np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")


def _projected_da(shape=(4, 3, 3), crs="EPSG:3577", *, x_offset=0.0, y_offset=0.0):
    t, y, x = shape
    data = np.zeros(shape, dtype=bool)
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0 + y_offset
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0 + x_offset
    da_arr = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": _times(t), "y": ys, "x": xs},
    )
    da_arr = da_arr.rio.write_crs(crs)
    return da_arr


# ---- check_grid_alignment -------------------------------------------------


def test_matching_grids_pass_alignment_check():
    water = _projected_da()
    valid = _projected_da()

    check_grid_alignment(water, valid)  # must not raise


def test_shape_mismatch_raises_actionable_error():
    water = _projected_da(shape=(4, 3, 3))
    valid = _projected_da(shape=(4, 4, 4))

    with pytest.raises(InputContractError, match="grid|shape|dims"):
        check_grid_alignment(water, valid)


def test_transform_mismatch_raises_actionable_error():
    water = _projected_da()
    valid = _projected_da(x_offset=30.0)

    with pytest.raises(InputContractError, match="grid|transform|coord"):
        check_grid_alignment(water, valid)


def test_crs_mismatch_between_layers_raises_actionable_error():
    water = _projected_da(crs="EPSG:3577")
    valid = _projected_da(crs="EPSG:32756")

    with pytest.raises(InputContractError, match="CRS"):
        check_grid_alignment(water, valid)


# ---- check_crs_defined -----------------------------------------------------


def test_projected_crs_passes():
    water = _projected_da(crs="EPSG:3577")

    check_crs_defined(water)  # must not raise


def test_degrees_crs_is_refused():
    water = _projected_da(crs="EPSG:4326")

    with pytest.raises(InputContractError, match="degrees|geographic"):
        check_crs_defined(water)


def test_undefined_crs_is_permitted_not_refused():
    """An unset CRS is not itself an error (spec guard targets *degrees*,

    not *absence*) -- this matches the existing geographic-CRS guard
    elsewhere in the codebase (spatial/crs.py::normalize_spatial_inputs),
    and many valid generic_binary/in-memory inputs carry no georeferencing.
    """
    t, y, x = 4, 3, 3
    data = np.zeros((t, y, x), dtype=bool)
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0
    water = xr.DataArray(
        data, dims=("time", "y", "x"), coords={"time": _times(t), "y": ys, "x": xs}
    )
    # No .rio.write_crs called -> CRS undefined.

    check_crs_defined(water)  # must not raise


def test_array_without_rio_accessor_concept_is_skipped_cleanly():
    # Pure in-memory fixture, no spatial coords/CRS concept at all.
    data = np.array([0, 1, 1, 0])
    water = xr.DataArray(data, dims=["time"])

    check_crs_defined(water)  # must not raise -- nothing to check


# ---- normalize_structure ---------------------------------------------------


def test_single_variable_dataset_renames_to_water_and_logs_fix():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset({"mask": (("time",), data)}, coords={"time": _times(4)})

    array, fixes = normalize_structure(ds, variable_map=None)

    assert "water" not in ds  # original untouched
    np.testing.assert_array_equal(array.values, [0, 1, 1, 0])
    assert any("mask" in fix and "renamed" in fix for fix in fixes)


def test_variable_map_rename_is_applied_and_logged():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset({"my_band": (("time",), data)}, coords={"time": _times(4)})

    array, fixes = normalize_structure(ds, variable_map={"my_band": "water"})

    np.testing.assert_array_equal(array.values, [0, 1, 1, 0])
    assert any("my_band" in fix and "water" in fix for fix in fixes)


def test_int_zero_one_array_is_coerced_to_bool_and_logged():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])

    array, fixes = normalize_structure(da_in, variable_map=None)

    assert array.dtype == bool
    assert any("coerced" in fix and "bool" in fix for fix in fixes)


def test_already_bool_array_has_no_fixes_applied():
    data = np.array([False, True, True, False])
    da_in = xr.DataArray(data, dims=["time"])

    array, fixes = normalize_structure(da_in, variable_map=None)

    assert array.dtype == bool
    assert fixes == ()


def test_dim_reorder_to_expected_order_is_applied_and_logged():
    data = np.zeros((3, 4, 2), dtype=bool)  # (y, time, x) -- wrong order
    da_in = xr.DataArray(data, dims=["y", "time", "x"])

    array, fixes = normalize_structure(da_in, variable_map=None)

    assert array.dims == ("time", "y", "x")
    assert any("reorder" in fix for fix in fixes)


def test_ambiguous_dataset_raises_actionable_error():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset(
        {
            "alpha": (("time",), data),
            "beta": (("time",), data),
        },
        coords={"time": _times(4)},
    )

    with pytest.raises(InputContractError, match="ambiguous|multiple"):
        normalize_structure(ds, variable_map=None)
