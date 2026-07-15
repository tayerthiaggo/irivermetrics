import xarray as xr
import dask.array as da
from hydrofragments.io.validity import apply_validity_policy

def parse_watermask_tsfill(da_in):
    valid = apply_validity_policy(da_in)
    water = (da_in == 1) & valid
    return water, valid
