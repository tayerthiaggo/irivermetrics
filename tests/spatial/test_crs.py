import pytest
import xarray as xr
import numpy as np
import rioxarray
from hydrofragments.spatial.crs import normalize_spatial_inputs

def test_geographic_crs_rejection():
    # EPSG:4326 is geographic. Should be rejected unless explicitly configured to project
    da = xr.DataArray(np.zeros((2, 2)), coords={"x": [114, 115], "y": [-20, -21]}, dims=["y", "x"])
    da.rio.write_crs("EPSG:4326", inplace=True)
    
    with pytest.raises(ValueError, match="Geographic CRS not supported"):
        normalize_spatial_inputs(da, target_crs=None)

def test_equal_area_crs_accepted():
    da = xr.DataArray(np.zeros((2, 2)), coords={"x": [0, 1], "y": [0, 1]}, dims=["y", "x"])
    da.rio.write_crs("EPSG:3577", inplace=True)
    
    normalized = normalize_spatial_inputs(da, target_crs=None)
    assert "3577" in normalized.rio.crs.to_wkt()
