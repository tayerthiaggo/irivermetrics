import xarray as xr

def validate_alignment(da1, da2):
    for dim in da1.dims:
        if dim in da2.dims:
            if not da1.coords[dim].equals(da2.coords[dim]):
                raise ValueError("Grid mismatch")
    
    if hasattr(da1, "rio") and hasattr(da2, "rio"):
        if da1.rio.crs is not None and da2.rio.crs is not None:
            if da1.rio.crs != da2.rio.crs:
                raise ValueError("CRS mismatch")
