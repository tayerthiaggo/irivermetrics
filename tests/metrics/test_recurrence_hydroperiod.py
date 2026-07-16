"""Milestone 9 — pixel recurrence and hydroperiod (spec §6.12, JRC/DEA-style).

Load-bearing contract:

- Recurrence denominator is **valid years**, not total calendar years in the
  record. A year with zero valid (observed) months contributes to neither the
  numerator nor the denominator.
- Hydroperiod denominator is **valid observed months within a year**, not the
  full 12-month year length. Unobserved months do not count as dry.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import compute_hydroperiod, compute_recurrence
from tests.fixtures.analytic_masks import recurrence_temporal_fixture


def _config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "upstream",
            },
        }
    )


def _monthly(water: np.ndarray, valid: np.ndarray, times: pd.DatetimeIndex) -> xr.Dataset:
    dims = ("time", "y", "x")
    coords = {"time": times}
    water_da = xr.DataArray(water.astype(bool), dims=dims, coords=coords)
    valid_da = xr.DataArray(valid.astype(bool), dims=dims, coords=coords)
    return xr.Dataset({"water": water_da, "valid_obs": valid_da})


def test_recurrence_denominator_excludes_years_with_no_valid_observations():
    # 1 pixel, 3 calendar years. 2001: wet & observed. 2002: never observed
    # (must not count as a valid year at all). 2003: observed, dry.
    times = pd.to_datetime(["2001-06-01", "2002-06-01", "2003-06-01"])
    water = np.array([[[1]], [[0]], [[0]]])
    valid = np.array([[[1]], [[0]], [[1]]])
    result = compute_recurrence(_monthly(water, valid, times), config=_config())

    # Valid years = {2001, 2003} -> 2 valid years. Wet-at-least-once = {2001}.
    # REC = 1/2 * 100 = 50%, NOT 1/3 (naive total-years) = 33.3%.
    rec = result.recurrence.isel(y=0, x=0).item()
    assert rec == pytest.approx(50.0)
    assert result.valid_year_count.isel(y=0, x=0).item() == 2


def test_recurrence_equal_weights_supported_calendar_months():
    # January has 2/3 wet years; February has one observed dry year. The
    # approved season-stratified estimator gives each supported month equal
    # weight: (2/3 + 0) / 2 = 1/3, rather than pooled year-level 2/3.
    water, valid, times = recurrence_temporal_fixture()

    result = compute_recurrence(_monthly(water, valid, times), config=_config())

    assert result.recurrence.isel(y=0, x=0).item() == pytest.approx(100.0 / 3.0)
    assert result.valid_year_count.isel(y=0, x=0).item() == 3


def test_recurrence_all_years_invalid_is_nan():
    times = pd.to_datetime(["2001-06-01", "2002-06-01"])
    water = np.array([[[0]], [[0]]])
    valid = np.array([[[0]], [[0]]])
    result = compute_recurrence(_monthly(water, valid, times), config=_config())
    assert np.isnan(result.recurrence.isel(y=0, x=0).item())
    assert result.valid_year_count.isel(y=0, x=0).item() == 0


def test_hydroperiod_denominator_is_valid_observed_months_not_calendar_length():
    # One HY/calendar year, 4 months present in the record. Wet in 2 of the 3
    # OBSERVED months; the 4th month is unobserved and must not inflate the
    # denominator to 4.
    times = pd.to_datetime(["2001-01-01", "2001-02-01", "2001-03-01", "2001-04-01"])
    water = np.array([[[1]], [[1]], [[0]], [[0]]])
    valid = np.array([[[1]], [[1]], [[1]], [[0]]])  # April unobserved
    result = compute_hydroperiod(_monthly(water, valid, times), config=_config())

    # valid observed months = 3 (Jan, Feb, Mar); wet among those = 2 (Jan, Feb).
    # HP = 2/3, NOT 2/4 (which would treat the unobserved month as dry).
    hp = result.hydroperiod.sel(year=2001).isel(y=0, x=0).item()
    assert hp == pytest.approx(2.0 / 3.0)
    assert result.valid_observed_months.sel(year=2001).isel(y=0, x=0).item() == 3


def test_hydroperiod_year_with_zero_valid_observed_months_is_nan():
    times = pd.to_datetime(["2001-01-01", "2001-02-01"])
    water = np.array([[[0]], [[0]]])
    valid = np.array([[[0]], [[0]]])
    result = compute_hydroperiod(_monthly(water, valid, times), config=_config())
    hp = result.hydroperiod.sel(year=2001).isel(y=0, x=0).item()
    assert np.isnan(hp)
    assert result.valid_observed_months.sel(year=2001).isel(y=0, x=0).item() == 0
