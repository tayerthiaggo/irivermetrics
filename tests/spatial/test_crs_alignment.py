import pytest
import xarray as xr
import numpy as np
import rioxarray
from hydrofragments.io.alignment import validate_alignment

def test_grid_transform_mismatch_raises():
    # Create two DataArrays with different coordinates
    da1 = xr.DataArray(np.zeros((2, 2)), coords={"x": [0, 1], "y": [0, 1]}, dims=["y", "x"])
    da2 = xr.DataArray(np.zeros((2, 2)), coords={"x": [0, 2], "y": [0, 1]}, dims=["y", "x"])
    
    with pytest.raises(ValueError, match="Grid mismatch"):
        validate_alignment(da1, da2)

def test_crs_mismatch_raises():
    da1 = xr.DataArray(np.zeros((2, 2)), coords={"x": [0, 1], "y": [0, 1]}, dims=["y", "x"])
    da1.rio.write_crs("EPSG:3577", inplace=True)
    
    da2 = xr.DataArray(np.zeros((2, 2)), coords={"x": [0, 1], "y": [0, 1]}, dims=["y", "x"])
    da2.rio.write_crs("EPSG:4326", inplace=True)
    
    with pytest.raises(ValueError, match="CRS mismatch"):
        validate_alignment(da1, da2)
