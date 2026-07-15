"""Milestone 5 — occurrence persistence metric (season-stratified P-native).

Load-bearing contract (Decision Gate 0, U2/Q1, approved 2026-07-14):

- Occurrence uses **valid-observation counts** as the denominator, never total
  timesteps. Unobserved months must not dilute the ratio.
- Any temporal aggregate (occurrence included) uses a **season-stratified**
  estimator: per-calendar-month P-native ratio, equal-weighted across the 12
  calendar months present — NOT a naive pooled ``sum(water)/sum(valid)``.
- Per-pixel ``min_valid_obs`` floor suppresses occurrence where support is thin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import compute_occurrence


def _config(min_valid_obs: int = 1) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "upstream",
            },
            "validity": {"min_valid_obs": min_valid_obs},
        }
    )


def _monthly(
    water: np.ndarray, valid: np.ndarray, times: pd.DatetimeIndex
) -> xr.Dataset:
    """Build an M4-shaped monthly dataset from (time, y, x) boolean arrays."""
    dims = ("time", "y", "x")
    coords = {"time": times}
    water_da = xr.DataArray(water.astype(bool), dims=dims, coords=coords)
    valid_da = xr.DataArray(valid.astype(bool), dims=dims, coords=coords)
    return xr.Dataset({"water": water_da, "valid_obs": valid_da})


def test_occurrence_denominator_is_valid_obs_not_total_timesteps():
    # 1 pixel, 4 months. Wet in the 2 observed months, unobserved (not dry) in
    # the other 2. All 4 months are the SAME calendar month so season-strat and
    # pooled collapse to the same ratio, isolating the denominator question.
    times = pd.to_datetime(["2001-01-01", "2002-01-01", "2003-01-01", "2004-01-01"])
    water = np.array([[[1]], [[1]], [[0]], [[0]]])
    valid = np.array([[[1]], [[1]], [[0]], [[0]]])  # last two unobserved
    result = compute_occurrence(_monthly(water, valid, times), config=_config())

    # water_obs / valid_obs = 2/2 = 1.0 (100%), NOT 2/4 = 0.5 that total
    # timesteps would give.
    occ = result.occurrence.isel(y=0, x=0).item()
    assert occ == pytest.approx(100.0)
    assert result.valid_count.isel(y=0, x=0).item() == 2


def test_occurrence_is_season_stratified_not_naive_pooled():
    # MNAR construction on a single pixel: the wet calendar month (Feb) is
    # rarely observed; the dry calendar month (Aug) is observed often. A naive
    # pooled ratio under-weights the wet season; the season-stratified estimator
    # gives each calendar month equal weight.
    #
    # Feb: 1 observed month, wet        -> Feb ratio = 1/1 = 1.0
    # Aug: 4 observed months, all dry   -> Aug ratio = 0/4 = 0.0
    # Season-stratified = mean(1.0, 0.0) = 0.5 -> 50%
    # Naive pooled      = 1 wet / 5 valid = 0.2 -> 20%
    times = pd.to_datetime(
        [
            "2001-02-01",  # wet, observed
            "2001-08-01",  # dry, observed
            "2002-08-01",  # dry, observed
            "2003-08-01",  # dry, observed
            "2004-08-01",  # dry, observed
        ]
    )
    water = np.array([[[1]], [[0]], [[0]], [[0]], [[0]]])
    valid = np.ones_like(water)
    result = compute_occurrence(_monthly(water, valid, times), config=_config())

    occ = result.occurrence.isel(y=0, x=0).item()
    assert occ == pytest.approx(50.0)  # season-stratified
    assert occ != pytest.approx(20.0)  # would be naive pooled


def test_season_stratification_ignores_calendar_months_with_no_valid_obs():
    # A calendar month with zero valid observations across the whole record must
    # not contribute a 0/0 term; the estimator averages only the calendar months
    # that actually have support.
    times = pd.to_datetime(["2001-03-01", "2001-09-01"])
    water = np.array([[[1]], [[0]]])
    # March observed and wet; September never observed anywhere.
    valid = np.array([[[1]], [[0]]])
    result = compute_occurrence(_monthly(water, valid, times), config=_config())

    # Only March contributes: ratio 1/1 -> 100%, not mean(1.0, NaN) mishandled.
    assert result.occurrence.isel(y=0, x=0).item() == pytest.approx(100.0)


def test_occurrence_suppressed_below_min_valid_obs_floor():
    times = pd.to_datetime(["2001-01-01", "2002-01-01", "2003-01-01"])
    water = np.array([[[1]], [[1]], [[1]]])
    valid = np.array([[[1]], [[1]], [[1]]])  # 3 valid obs
    result = compute_occurrence(
        _monthly(water, valid, times), config=_config(min_valid_obs=5)
    )
    # 3 valid < floor of 5 -> occurrence suppressed to NaN for that pixel.
    assert np.isnan(result.occurrence.isel(y=0, x=0).item())


def test_occurrence_zero_valid_pixel_is_nan_not_zero():
    times = pd.to_datetime(["2001-01-01", "2002-01-01"])
    water = np.array([[[0]], [[0]]])
    valid = np.array([[[0]], [[0]]])  # never observed
    result = compute_occurrence(_monthly(water, valid, times), config=_config())
    assert np.isnan(result.occurrence.isel(y=0, x=0).item())
    assert result.valid_count.isel(y=0, x=0).item() == 0
