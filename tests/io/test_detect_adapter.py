import numpy as np
import xarray as xr
import pytest

from hydrofragments.io.adapters import detect_adapter


def _times(n):
    return np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")


def test_uint8_tsfill_sentinel_signature_is_detected():
    # 0/1/254/255 uint8 -- the WaterMask-TSFill sentinel convention.
    data = np.array([0, 1, 254, 255], dtype=np.uint8)
    da_in = xr.DataArray(data, dims=["time"], coords={"time": _times(4)})

    assert detect_adapter(da_in) == "watermask_tsfill"


def test_tsfill_dataset_with_water_mask_variable_is_detected():
    data = np.array([0, 1, 254, 255], dtype=np.uint8)
    ds = xr.Dataset(
        {"water_mask": (("time",), data)}, coords={"time": _times(4)}
    )

    assert detect_adapter(ds) == "watermask_tsfill"


def test_raw_wofs_named_water_band_is_detected():
    data = np.array([0, 1, 1, 0], dtype=np.int16)
    ds = xr.Dataset({"water": (("time",), data)}, coords={"time": _times(4)})

    assert detect_adapter(ds) == "raw_wofs"


def test_raw_wofs_frequency_variable_is_detected():
    data = np.array([0.1, 0.4, 0.6, 0.9], dtype=np.float32)
    ds = xr.Dataset(
        {"frequency": (("time",), data)}, coords={"time": _times(4)}
    )

    assert detect_adapter(ds) == "raw_wofs"


def test_generic_binary_bool_array_is_detected():
    data = np.array([False, True, True, False])
    da_in = xr.DataArray(data, dims=["time"], coords={"time": _times(4)})

    assert detect_adapter(da_in) == "generic_binary"


def test_generic_binary_zero_one_int_array_is_detected():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    da_in = xr.DataArray(data, dims=["time"], coords={"time": _times(4)})

    assert detect_adapter(da_in) == "generic_binary"


def test_generic_binary_single_unnamed_variable_dataset_is_detected():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset({"mask": (("time",), data)}, coords={"time": _times(4)})

    assert detect_adapter(ds) == "generic_binary"


def test_ambiguous_multi_variable_dataset_raises_actionable_error():
    data = np.array([0, 1, 1, 0], dtype=np.int32)
    ds = xr.Dataset(
        {
            "alpha": (("time",), data),
            "beta": (("time",), data),
        },
        coords={"time": _times(4)},
    )

    with pytest.raises(ValueError, match="ambiguous|multiple"):
        detect_adapter(ds)
