"""Milestone 5 — APSEC (Areal Percent of Surface-water Extent Contraction... extent).

Spec §6.17: ``APSEC_t = WA_t / A_ref * 100`` where ``WA_t = cell_area * count(W_t=1)``.

The load-bearing property is the **fixed denominator**: ``A_ref`` is the AOI
reference area and must be identical across every month regardless of how much
water is present. Empty, all-wet, and spatially-clipped-AOI cases all divide by
the same ``A_ref``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec


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


def _monthly(water: np.ndarray, times: pd.DatetimeIndex) -> xr.Dataset:
    dims = ("time", "y", "x")
    valid = np.ones_like(water)
    return xr.Dataset(
        {
            "water": xr.DataArray(
                water.astype(bool), dims=dims, coords={"time": times}
            ),
            "valid_obs": xr.DataArray(
                valid.astype(bool), dims=dims, coords={"time": times}
            ),
        }
    )


def test_apsec_uses_fixed_aref_across_months():
    # 3x3 AOI (9 pixels), cell 900 m^2 -> A_ref = 8100 m^2.
    # Month 1: 3 wet pixels -> 2700/8100 = 33.33%
    # Month 2: 6 wet pixels -> 5400/8100 = 66.67%
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = np.zeros((2, 3, 3))
    water[0, 0, :] = 1  # 3 wet
    water[1, 0:2, :] = 1  # 6 wet
    a_ref_m2 = 8100.0

    records = compute_apsec(
        _monthly(water, times),
        a_ref_m2=a_ref_m2,
        cell_area_m2=900.0,
        config=_config(),
    )

    values = {r.date: r.value for r in records}
    assert values[times[0]] == pytest.approx(100.0 / 3.0)
    assert values[times[1]] == pytest.approx(200.0 / 3.0)


def test_apsec_all_dry_month_is_zero():
    times = pd.to_datetime(["2020-01-01"])
    water = np.zeros((1, 3, 3))
    records = compute_apsec(
        _monthly(water, times), a_ref_m2=8100.0, cell_area_m2=900.0, config=_config()
    )
    assert records[0].value == pytest.approx(0.0)
    assert records[0].n_water_pixels == 0


def test_apsec_all_wet_month_relative_to_aref_not_wet_extent():
    # AOI has 9 pixels but A_ref is deliberately LARGER than the wetted grid
    # (e.g. AOI clipped from a bigger catchment). All 9 pixels wet must divide
    # by the fixed A_ref, not by the wet-pixel count.
    times = pd.to_datetime(["2020-01-01"])
    water = np.ones((1, 3, 3))
    a_ref_m2 = 18000.0  # twice the 9 * 900 wetted area
    records = compute_apsec(
        _monthly(water, times),
        a_ref_m2=a_ref_m2,
        cell_area_m2=900.0,
        config=_config(),
    )
    # WA = 9 * 900 = 8100; 8100/18000 = 45%.
    assert records[0].value == pytest.approx(45.0)
    # Never clamps to 100 by redefining the denominator as the wet extent.
    assert records[0].value != pytest.approx(100.0)


def test_apsec_clipped_aoi_shares_same_denominator():
    # Two months with different wetted extents, one exceeding a small A_ref.
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = np.zeros((2, 3, 3))
    water[0, 0, 0] = 1  # 1 wet
    water[1, :, :] = 1  # 9 wet
    a_ref_m2 = 4500.0  # 5 pixels worth
    records = compute_apsec(
        _monthly(water, times),
        a_ref_m2=a_ref_m2,
        cell_area_m2=900.0,
        config=_config(),
    )
    values = {r.date: r.value for r in records}
    # 1 pixel: 900/4500 = 20%
    assert values[times[0]] == pytest.approx(20.0)
    # 9 pixels: 8100/4500 = 180% (extent can exceed A_ref; denominator fixed)
    assert values[times[1]] == pytest.approx(180.0)


def test_apsec_emits_one_record_per_month():
    times = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    water = np.zeros((3, 2, 2))
    records = compute_apsec(
        _monthly(water, times), a_ref_m2=3600.0, cell_area_m2=900.0, config=_config()
    )
    assert len(records) == 3
    assert [r.date for r in records] == list(times)
