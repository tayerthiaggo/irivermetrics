import xarray as xr

def apply_validity_policy(da_in):
    # Valid = 0 or 1. Invalid = 254 (outside AOI) or 255 (unobserved)
    return (da_in == 0) | (da_in == 1)
