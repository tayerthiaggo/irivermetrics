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


def parse_generic_binary(
    da_in,
    *,
    valid_obs: xr.DataArray | None = None,
    nodata: float | int | None = None,
):
    """Parse a generic ``{0, 1}``/bool water mask into (water, valid_obs).

    This is the fallback adapter for source-agnostic binary masks that carry
    no WOfS-specific or TSFill-specific sentinel convention: ``bool`` arrays
    pass straight through, ``{0, 1}`` int/float arrays are coerced to bool.
    Any other finite value is refused with an actionable error -- this
    adapter never guesses at a threshold (that is ``raw_wofs``'s/
    ``generic_probability``'s job).

    ``nodata``, when supplied, marks pixels equal to that sentinel value as
    invalid (excluded from both ``water`` and ``valid_obs``) -- this lets a
    single-band mask carry an explicit "no data" sentinel without a paired
    ``valid_obs`` layer. If ``valid_obs`` is also supplied, both are honored
    (a pixel is valid only if ``valid_obs`` says so AND it is not the
    ``nodata`` sentinel).

    If neither ``valid_obs`` nor ``nodata`` is supplied, an all-True valid
    mask of the same shape is returned (matching ``parse_raw_wofs``'s
    fallback behavior for missing ``valid_obs``).
    """
    if da_in.dtype != bool:
        checked = da_in if nodata is None else da_in.where(da_in != nodata, 0)
        is_binary = bool(((checked == 0) | (checked == 1)).all())
        if not is_binary:
            raise ValueError(
                "parse_generic_binary: input contains values outside {0, 1} "
                "and is not boolean; generic_binary requires an already-"
                "binary mask (use raw_wofs or generic_probability with a "
                "water_threshold for probability/frequency inputs)"
            )
    water = (da_in == 1) if da_in.dtype != bool else da_in.astype(bool)

    if valid_obs is None:
        valid = xr.ones_like(water, dtype=bool)
    else:
        valid = valid_obs.astype(bool)

    if nodata is not None:
        is_nodata = da_in == nodata
        valid = valid & ~is_nodata
        water = water & ~is_nodata

    return water, valid


ADAPTERS: dict[str, Callable] = {
    "watermask_tsfill": parse_watermask_tsfill,
    "raw_wofs": parse_raw_wofs,
    "generic_binary": parse_generic_binary,
}
