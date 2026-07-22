import numpy as np
import xarray as xr
import pytest

from hydrofragments.io.adapters import ADAPTERS, parse_raw_wofs


def test_binary_input_passes_through_without_threshold():
    # Raw WOfS water band already in {0, 1} form needs no threshold.
    data = np.array([0, 1, 1, 0], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_raw_wofs(da_in)

    np.testing.assert_array_equal(water.values, [False, True, True, False])
    # No bundled valid-obs layer by convention -> all-True fallback.
    np.testing.assert_array_equal(valid.values, [True, True, True, True])


def test_probability_input_is_thresholded_when_threshold_supplied():
    data = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    water, valid = parse_raw_wofs(da_in, water_threshold=0.5)

    np.testing.assert_array_equal(water.values, [False, False, True, True])
    np.testing.assert_array_equal(valid.values, [True, True, True, True])


def test_probability_input_without_threshold_raises_actionable_error():
    data = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])

    with pytest.raises(ValueError, match="water_threshold"):
        parse_raw_wofs(da_in)


def test_caller_supplied_valid_obs_is_respected():
    data = np.array([0, 1, 1, 0], dtype=np.float32)
    da_in = xr.DataArray(data, dims=["time"])
    caller_valid = xr.DataArray(
        np.array([True, True, False, True]), dims=["time"]
    )

    water, valid = parse_raw_wofs(da_in, valid_obs=caller_valid)

    np.testing.assert_array_equal(valid.values, [True, True, False, True])


def test_dask_backed_arrays_preserved():
    import dask.array as da

    data = da.from_array(np.array([0, 1, 1, 0], dtype=np.float32), chunks=(2,))
    dx = xr.DataArray(data, dims=["time"])

    water, valid = parse_raw_wofs(dx)

    assert isinstance(water.data, da.Array)
    assert isinstance(valid.data, da.Array)


def test_registry_exposes_watermask_tsfill_and_raw_wofs():
    from hydrofragments.io.adapters import parse_watermask_tsfill

    assert ADAPTERS["watermask_tsfill"] is parse_watermask_tsfill
    assert ADAPTERS["raw_wofs"] is parse_raw_wofs
