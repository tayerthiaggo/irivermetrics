from typing import Callable

import numpy as np
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


# WaterMask-TSFill's uint8 sentinel convention (see io/validity.py):
# 0 = dry, 1 = water, 254 = outside AOI, 255 = unobserved.
_TSFILL_SENTINELS = frozenset({0, 1, 254, 255})
_TSFILL_VARIABLE_NAMES = frozenset({"water_mask"})
_RAW_WOFS_VARIABLE_NAMES = frozenset({"water", "frequency"})


def _pick_data_array(source: "xr.DataArray | xr.Dataset") -> tuple[xr.DataArray, str | None]:
    """Return ``(array, variable_name)`` for detection purposes.

    ``variable_name`` is ``None`` for a bare ``DataArray``. Raises a
    ``ValueError`` naming the ambiguity when ``source`` is a ``Dataset``
    with more than one data variable and none of them look water-like
    (matches TSFill's or raw WOfS's known variable names, or is the sole
    variable in the dataset).
    """
    if isinstance(source, xr.DataArray):
        return source, None

    data_vars = list(source.data_vars)
    if len(data_vars) == 1:
        name = data_vars[0]
        return source[name], name

    for name in _TSFILL_VARIABLE_NAMES | _RAW_WOFS_VARIABLE_NAMES:
        if name in source:
            return source[name], name

    raise ValueError(
        "detect_adapter: ambiguous Dataset with multiple variables "
        f"({sorted(data_vars)}) and no water-like candidate (expected one "
        "of 'water_mask' [watermask_tsfill], 'water'/'frequency' "
        "[raw_wofs], or a single data variable); pass input_kind explicitly "
        "or use variable_map to disambiguate"
    )


def _looks_like_tsfill(array: xr.DataArray, variable_name: str | None) -> bool:
    if variable_name in _TSFILL_VARIABLE_NAMES:
        return True
    if array.dtype != "uint8":
        return False
    values = np.unique(array.values)
    if not set(values.tolist()).issubset(_TSFILL_SENTINELS):
        return False
    # Require at least one TSFill-specific sentinel (254/255) to distinguish
    # from a plain {0,1} generic_binary/raw_wofs mask that merely happens to
    # be uint8-typed.
    return bool({254, 255} & set(values.tolist()))


def _looks_like_raw_wofs(array: xr.DataArray, variable_name: str | None) -> bool:
    return variable_name in _RAW_WOFS_VARIABLE_NAMES


def detect_adapter(source: "xr.DataArray | xr.Dataset") -> str:
    """Pick the registry key (``ADAPTERS`` name) matching ``source``'s shape.

    Detection order (first match wins):

    1. ``watermask_tsfill`` -- variable named ``water_mask``, or a uint8
       array whose only values are a subset of ``{0, 1, 254, 255}`` and
       includes at least one of the TSFill-specific sentinels (254/255).
    2. ``raw_wofs`` -- variable named ``water`` or ``frequency`` (DEA WOfS
       band naming convention).
    3. ``generic_binary`` -- fallback: bool or ``{0, 1}`` values with no
       WOfS-specific band naming.

    For a ``Dataset`` with more than one data variable and no recognizable
    water-like candidate, raises ``ValueError`` rather than silently
    guessing (ambiguity must be surfaced, not resolved by chance).
    """
    array, variable_name = _pick_data_array(source)

    if _looks_like_tsfill(array, variable_name):
        return "watermask_tsfill"
    if _looks_like_raw_wofs(array, variable_name):
        return "raw_wofs"
    return "generic_binary"
