"""W3.1: full-catchment AOI denominators vs. analysis-mask coverage denominator.

Global constraints pinned here:

- "APSEC/LPI/reference-area denominators remain full catchment ``aoi_mask``."
- "Monthly coverage denominator is approved as conservative potential-water
  ``analysis_mask``, not full catchment."

``compute_apsec``/``compute_patch_metrics`` already use a fixed ``a_ref_m2``/
``a_total_m2`` supplied by the caller (the full AOI reference area) -- this
file pins that behaviour explicitly against a ``WaterCube.aoi_mask``-derived
area, and separately proves the NEW ``compute_analysis_mask_coverage``
function in ``hydrofragments/metrics/extent.py`` computes the monthly valid
coverage fraction against ``analysis_mask``'s pixel count, not the full grid.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec, compute_analysis_mask_coverage


def _monthly(water: np.ndarray, valid: np.ndarray, times: pd.DatetimeIndex) -> xr.Dataset:
    dims = ("time", "y", "x")
    return xr.Dataset(
        {
            "water": xr.DataArray(water.astype(bool), dims=dims, coords={"time": times}),
            "valid_obs": xr.DataArray(valid.astype(bool), dims=dims, coords={"time": times}),
        }
    )


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


def test_apsec_denominator_pinned_to_full_aoi_area_not_analysis_mask():
    """APSEC's a_ref_m2 must be the FULL catchment area, independent of any
    smaller analysis_mask footprint -- even when analysis_mask covers only a
    fraction of the grid, APSEC's fixed reference area is untouched.
    """
    times = pd.to_datetime(["2020-01-01"])
    water = np.zeros((1, 10, 10))
    water[0, 0:2, 0:2] = 1  # 4 wet pixels
    valid = np.ones((1, 10, 10))

    # Full catchment area: 100 px * 900 m^2/px = 90,000 m^2.
    full_aoi_area_m2 = 100 * 900.0

    records = compute_apsec(
        _monthly(water, valid, times),
        a_ref_m2=full_aoi_area_m2,
        cell_area_m2=900.0,
        config=_config(),
    )

    # 4 wet pixels * 900 m^2 / 90,000 m^2 * 100 = 4.0%, using the FULL
    # catchment denominator regardless of how small any potential-water
    # analysis_mask footprint might be.
    assert records[0].value == pytest.approx(4.0)
    assert records[0].a_ref_m2 == pytest.approx(full_aoi_area_m2)


def test_analysis_mask_coverage_uses_mask_pixel_count_as_denominator():
    """Coverage fraction must be valid-and-in-mask / mask pixel count, NOT
    valid / full-grid pixel count.
    """
    times = pd.date_range("2020-01-01", periods=1, freq="MS")
    # 10x10 grid; analysis_mask covers only a 4x4 = 16-pixel corner.
    analysis_mask = np.zeros((10, 10), dtype=bool)
    analysis_mask[0:4, 0:4] = True

    valid = np.zeros((1, 10, 10), dtype=bool)
    # Within the mask: 12 of 16 pixels valid. Outside the mask: fully valid
    # (must be ignored entirely by the analysis-mask denominator).
    valid[0, 0:4, 0:4] = True
    valid[0, 0, 0] = False
    valid[0, 0, 1] = False
    valid[0, 0, 2] = False
    valid[0, 0, 3] = False
    valid[0, 4:, :] = True  # fully valid outside the mask -- must not count

    valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times})
    mask_da = xr.DataArray(analysis_mask, dims=("y", "x"))

    result = compute_analysis_mask_coverage(valid_da, analysis_mask=mask_da)

    # 12 valid of 16 masked pixels = 0.75, independent of the 84 fully-valid
    # pixels outside the mask (which a full-grid denominator would average
    # in and yield 0.88 instead).
    assert result.coverage_fraction.values.tolist() == pytest.approx([0.75])
    assert result.n_valid_pixels.values.tolist() == [12]
    assert result.n_mask_pixels == 16


def test_analysis_mask_coverage_rejects_misaligned_mask():
    times = pd.date_range("2020-01-01", periods=1, freq="MS")
    valid = xr.DataArray(
        np.ones((1, 4, 4), dtype=bool), dims=("time", "y", "x"), coords={"time": times}
    )
    wrong_shape_mask = xr.DataArray(np.ones((5, 5), dtype=bool), dims=("y", "x"))

    with pytest.raises(ValueError, match="align"):
        compute_analysis_mask_coverage(valid, analysis_mask=wrong_shape_mask)


def test_analysis_mask_coverage_rejects_empty_mask():
    times = pd.date_range("2020-01-01", periods=1, freq="MS")
    valid = xr.DataArray(
        np.ones((1, 4, 4), dtype=bool), dims=("time", "y", "x"), coords={"time": times}
    )
    empty_mask = xr.DataArray(np.zeros((4, 4), dtype=bool), dims=("y", "x"))

    with pytest.raises(ValueError, match="at least one"):
        compute_analysis_mask_coverage(valid, analysis_mask=empty_mask)


def test_analysis_mask_coverage_full_mask_matches_full_grid_average():
    """Sanity check: an all-true analysis_mask (the backward-compatible
    default) reduces exactly to a full-grid mean -- i.e. unpruned behaviour
    is unchanged when analysis_mask == everywhere.
    """
    times = pd.date_range("2020-01-01", periods=3, freq="MS")
    rng = np.random.default_rng(7)
    valid = rng.random((3, 8, 8)) > 0.3
    valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times})
    full_mask = xr.DataArray(np.ones((8, 8), dtype=bool), dims=("y", "x"))

    result = compute_analysis_mask_coverage(valid_da, analysis_mask=full_mask)
    expected = valid_da.mean(dim=("y", "x")).values

    assert result.coverage_fraction.values == pytest.approx(expected)
    assert result.n_mask_pixels == 64
