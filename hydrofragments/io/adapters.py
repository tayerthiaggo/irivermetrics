from typing import Callable

import xarray as xr
import dask.array as da
from hydrofragments.io.validity import apply_validity_policy

def parse_watermask_tsfill(da_in):
    valid = apply_validity_policy(da_in)
    water = (da_in == 1) & valid
    return water, valid


def parse_raw_wofs(
    da_in,
    *,
    water_threshold: float | None = None,
    valid_obs: xr.DataArray | None = None,
):
    """Parse a raw DEA WOfS-style ``water`` band into (water, valid_obs).

    Values in ``{0, 1}`` are treated as already-binary water. Any other
    finite value is treated as a probability/frequency and requires
    ``water_threshold`` to binarize it. Raw WOfS does not bundle a
    separate valid-observation layer by convention, so if the caller does
    not supply ``valid_obs`` an all-True mask of the same shape is
    returned.
    """
    is_binary = bool(((da_in == 0) | (da_in == 1)).all())
    if is_binary:
        water = da_in == 1
    else:
        if water_threshold is None:
            raise ValueError(
                "parse_raw_wofs: input contains non-binary values but no "
                "water_threshold was supplied; pass water_threshold to "
                "binarize this WOfS probability/frequency band"
            )
        water = da_in >= water_threshold

    if valid_obs is None:
        valid = xr.ones_like(water, dtype=bool)
    else:
        valid = valid_obs.astype(bool)

    return water, valid


ADAPTERS: dict[str, Callable] = {
    "watermask_tsfill": parse_watermask_tsfill,
    "raw_wofs": parse_raw_wofs,
}
