"""Milestone 5 — Refuge Area (RA).

Spec §6.17: ``RA_theta = cell_area * count(OCC_p >= theta and valid_count_p >= min_valid_obs)``.

RA obeys the approved validity/refuge threshold semantics:

- The occurrence surface feeding RA is the season-stratified P-native occurrence
  (see ``test_occurrence.py``).
- A pixel is a refuge only if its occurrence meets the refuge threshold AND its
  support clears ``min_valid_obs``. Thin-support high-occurrence pixels are not
  counted (spec §8 guard 4).
- RA is a fixed-unit area (km^2 by registry), using the pixel cell area, never a
  proportion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import (
    OccurrenceResult,
    compute_occurrence,
    compute_refuge_area,
)


def _config(refuge_threshold: float = 0.90, min_valid_obs: int = 1) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "upstream",
            },
            "persistence": {"refuge_threshold": refuge_threshold},
            "validity": {"min_valid_obs": min_valid_obs},
        }
    )


def _occurrence_result(
    occurrence: np.ndarray, valid_count: np.ndarray, min_valid_obs: int
) -> OccurrenceResult:
    dims = ("y", "x")
    return OccurrenceResult(
        occurrence=xr.DataArray(occurrence.astype(float), dims=dims),
        valid_count=xr.DataArray(valid_count.astype(np.int64), dims=dims),
        min_valid_obs=min_valid_obs,
    )


def test_refuge_area_counts_pixels_at_or_above_threshold_times_cell_area():
    # 3 pixels: occurrences 95%, 90%, 50%. Threshold 90% -> 2 refuge pixels.
    occ = np.array([[95.0, 90.0, 50.0]])
    valid_count = np.array([[30, 30, 30]])
    result = _occurrence_result(occ, valid_count, min_valid_obs=20)
    cell_area_m2 = 900.0  # 30 m x 30 m

    ra = compute_refuge_area(result, cell_area_m2=cell_area_m2, config=_config())

    # 2 pixels * 900 m^2 = 1800 m^2 = 0.0018 km^2
    assert ra.value == pytest.approx(0.0018)
    assert ra.n_refuge_pixels == 2


def test_refuge_area_excludes_thin_support_pixels():
    # Two pixels above threshold, but one has valid_count below the floor.
    occ = np.array([[95.0, 99.0]])
    valid_count = np.array([[30, 5]])  # second pixel under floor of 20
    result = _occurrence_result(occ, valid_count, min_valid_obs=20)

    ra = compute_refuge_area(
        result, cell_area_m2=900.0, config=_config(min_valid_obs=20)
    )

    assert ra.n_refuge_pixels == 1
    assert ra.value == pytest.approx(0.0009)


def test_refuge_area_threshold_boundary_is_inclusive():
    # Occurrence exactly at the threshold counts (>=, per spec formula).
    occ = np.array([[90.0]])
    valid_count = np.array([[30]])
    result = _occurrence_result(occ, valid_count, min_valid_obs=20)

    ra = compute_refuge_area(result, cell_area_m2=900.0, config=_config(0.90))
    assert ra.n_refuge_pixels == 1


def test_refuge_area_threshold_sensitivity():
    occ = np.array([[70.0, 85.0, 95.0]])
    valid_count = np.array([[30, 30, 30]])
    result = _occurrence_result(occ, valid_count, min_valid_obs=1)

    ra_high = compute_refuge_area(result, cell_area_m2=900.0, config=_config(0.90))
    ra_low = compute_refuge_area(result, cell_area_m2=900.0, config=_config(0.80))

    assert ra_high.n_refuge_pixels == 1  # only 95%
    assert ra_low.n_refuge_pixels == 2  # 85% and 95%


def test_refuge_area_all_dry_is_zero_area():
    occ = np.array([[np.nan, np.nan]])  # never resolved
    valid_count = np.array([[0, 0]])
    result = _occurrence_result(occ, valid_count, min_valid_obs=1)

    ra = compute_refuge_area(result, cell_area_m2=900.0, config=_config())
    assert ra.n_refuge_pixels == 0
    assert ra.value == pytest.approx(0.0)


def test_refuge_area_end_to_end_from_occurrence():
    # A perennial pixel (always wet, well observed) and an ephemeral pixel.
    times = pd.to_datetime([f"200{y}-01-01" for y in range(1, 6)])  # 5 Januaries
    water = np.array([[[1, 0]]] * 5)  # pixel 0 always wet, pixel 1 always dry
    valid = np.ones_like(water)
    monthly = xr.Dataset(
        {
            "water": xr.DataArray(
                water.astype(bool), dims=("time", "y", "x"), coords={"time": times}
            ),
            "valid_obs": xr.DataArray(
                valid.astype(bool), dims=("time", "y", "x"), coords={"time": times}
            ),
        }
    )
    occ_result = compute_occurrence(monthly, config=_config(min_valid_obs=3))
    ra = compute_refuge_area(occ_result, cell_area_m2=900.0, config=_config(0.90, 3))

    assert ra.n_refuge_pixels == 1  # only the perennial pixel
    assert ra.value == pytest.approx(0.0009)
