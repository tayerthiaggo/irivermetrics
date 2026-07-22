"""Lazy, validity-aware monthly composite graph construction."""

from __future__ import annotations

import numpy as np
import xarray as xr


class CompositeError(ValueError):
    """Raised when temporal inputs or composite provenance are incomplete."""


_COMPOSITES = {
    "max_water",
    "median",
    "mode",
    "end_of_month_nearest",
    "supplied",
}


def _validate_inputs(water: xr.DataArray, valid_obs: xr.DataArray) -> None:
    if water.dims != valid_obs.dims or water.sizes != valid_obs.sizes:
        raise CompositeError("water and valid_obs must have identical dimensions")
    if "time" not in water.dims:
        raise CompositeError("water and valid_obs require a time dimension")
    if not water.coords["time"].equals(valid_obs.coords["time"]):
        raise CompositeError("water and valid_obs time coordinates must align")
    if water.dtype.kind != "b" or valid_obs.dtype.kind != "b":
        raise CompositeError("water and valid_obs must be boolean arrays")


def _monthly_metadata_count(water: xr.DataArray) -> xr.DataArray:
    observations = xr.DataArray(
        np.ones(water.sizes["time"], dtype=np.int32),
        dims=("time",),
        coords={"time": water.coords["time"]},
    )
    return observations.resample(time="MS").sum(dim="time")


def build_monthly_products(
    water: xr.DataArray,
    valid_obs: xr.DataArray,
    *,
    input_cadence: str,
    monthly_composite: str | None,
    composite_owner: str | None,
) -> xr.Dataset:
    """Build monthly masks and diagnostic reductions without materializing data."""

    _validate_inputs(water, valid_obs)
    if monthly_composite is None or composite_owner is None:
        raise CompositeError("monthly input requires composite provenance")
    if monthly_composite not in _COMPOSITES:
        raise CompositeError(f"unsupported monthly composite: {monthly_composite}")

    if input_cadence == "monthly":
        valid_count = valid_obs.astype(np.int32)
        water_count = (water & valid_obs).astype(np.int32)
        result = xr.Dataset(
            {
                "water": water,
                "valid_obs": valid_obs,
                "valid_count": valid_count,
                "water_count": water_count,
                "valid_fraction": valid_obs.astype(np.float32),
            }
        )
    elif input_cadence == "submonthly":
        if monthly_composite == "supplied":
            raise CompositeError("submonthly input cannot use supplied composite")
        valid_count = valid_obs.astype(np.int32).resample(time="MS").sum(dim="time")
        water_count = (
            (water & valid_obs).astype(np.int32).resample(time="MS").sum(dim="time")
        )
        monthly_valid = valid_count > 0

        if monthly_composite == "max_water":
            monthly_water = water_count > 0
        elif monthly_composite == "median":
            median = (
                water.astype(np.float32)
                .where(valid_obs)
                .resample(time="MS")
                .median(dim="time", skipna=True)
            )
            monthly_water = median.fillna(0.0) >= 0.5
        elif monthly_composite == "mode":
            monthly_water = (water_count * 2) > valid_count
        else:
            nearest = water.where(valid_obs).resample(time="MS").last(skipna=True)
            monthly_water = nearest.fillna(False).astype(bool)

        observation_count = _monthly_metadata_count(water)
        result = xr.Dataset(
            {
                "water": monthly_water & monthly_valid,
                "valid_obs": monthly_valid,
                "valid_count": valid_count,
                "water_count": water_count,
                "valid_fraction": valid_count / observation_count,
            }
        )
    else:
        raise CompositeError(f"unsupported input cadence: {input_cadence}")

    result.attrs.update(
        {
            "composite_owner": composite_owner,
            "input_cadence": input_cadence,
            "monthly_composite": monthly_composite,
        }
    )
    return result


__all__ = ["CompositeError", "build_monthly_products"]
