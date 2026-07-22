import numpy as np
import xarray as xr
import pytest

from hydrofragments.io.adapters import parse_watermask_tsfill
from hydrofragments.io.validity import apply_validity_policy

def test_decode_sentinels_before_signed_casts():
    # Create a synthetic DataArray representing watermask_tsfill output
    # 0: dry, 1: wet, 254: outside AOI, 255: unobserved
    data = np.array([0, 1, 254, 255], dtype=np.uint8)
    da = xr.DataArray(data, dims=["time"])
    
    # This should return a boolean water mask and a boolean valid mask, plus provenance
    # It must not treat 254 or 255 as dry (0) or water (1)
    water, valid = parse_watermask_tsfill(da)
    
    water_vals = water.values
    valid_vals = valid.values
    
    assert valid_vals[0] == True
    assert valid_vals[1] == True
    assert valid_vals[2] == False  # outside AOI is not valid
    assert valid_vals[3] == False  # unobserved is not valid
    
    # Where valid is False, water value doesn't strictly matter for metrics, but
    # it must definitely NOT be True if it's 254 or 255.
    assert water_vals[2] == False
    assert water_vals[3] == False

def test_dask_backed_arrays_preserved():
    import dask.array as da
    data = da.from_array(np.array([0, 1, 254, 255], dtype=np.uint8), chunks=(2,))
    dx = xr.DataArray(data, dims=["time"])
    
    water, valid = parse_watermask_tsfill(dx)
    
    assert isinstance(water.data, da.Array)
    assert isinstance(valid.data, da.Array)
