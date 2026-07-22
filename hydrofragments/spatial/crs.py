def normalize_spatial_inputs(da, target_crs=None):
    if hasattr(da, "rio") and da.rio.crs is not None:
        if da.rio.crs.is_geographic and target_crs is None:
            raise ValueError("Geographic CRS not supported unless explicitly configured to project")
    return da
