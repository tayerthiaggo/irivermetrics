import numpy as np
import xarray as xr
import pytest

from hydrofragments.io.adapters import ADAPTERS, parse_generic_binary


def test_bool_input_round_trips():
    data = np.array([False, True, True, False])
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_generic_binary(da_in)

    np.testing.assert_array_equal(water.values, [False, True, True, False])
    # No valid_obs supplied -> all-True fallback.
    np.testing.assert_array_equal(valid.values, [True, True, True, True])


def test_int_zero_one_input_coerces_to_bool():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_generic_binary(da_in)

    assert water.dtype == bool
    np.testing.assert_array_equal(water.values, [False, True, True, False])


def test_float_zero_one_input_coerces_to_bool():
    data = np.array([0.0, 1.0, 1.0, 0.0], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_generic_binary(da_in)

    assert water.dtype == bool
    np.testing.assert_array_equal(water.values, [False, True, True, False])


def test_paired_valid_obs_layer_is_respected():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])
    caller_valid = xr.DataArray(
        np.array([True, True, False, True]), dims=["time"]
    )

    water, valid = parse_generic_binary(da_in, valid_obs=caller_valid)

    np.testing.assert_array_equal(valid.values, [True, True, False, True])


def test_non_binary_value_raises_actionable_error():
    data = np.array([0, 1, 2, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])

    with pytest.raises(ValueError, match="generic_binary"):
        parse_generic_binary(da_in)


def test_explicit_nodata_excluded_from_valid():
    data = np.array([0, 1, -1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_generic_binary(da_in, nodata=-1)

    np.testing.assert_array_equal(valid.values, [True, True, False, True])
    # nodata pixel must never read as water.
    assert water.values[2] == False


def test_dask_backed_arrays_preserved():
    import dask.array as da

    data = da.from_array(np.array([0, 1, 1, 0], dtype=np.float32), chunks=(2,))
    dx = xr.DataArray(data, dims=["time"])

    water, valid = parse_generic_binary(dx)

    assert isinstance(water.data, da.Array)
    assert isinstance(valid.data, da.Array)


def test_registry_exposes_generic_binary():
    assert ADAPTERS["generic_binary"] is parse_generic_binary
