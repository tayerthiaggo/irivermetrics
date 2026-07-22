"""Milestone 9 — reconnection timing (spec §6.15) and refuge spatial
stability (spec §6.16), dynamics tranche.

Load-bearing contract:

- Reconnection timing prefers RC/DCI, but neither is implemented yet
  (Milestone 11). This module must use the documented fallback --
  `LPI_t >= t_LPI` -- and set `reconnection_metric_used = "LPI"` with
  `proxy_reconnection_flag = True` (spec §6.15 required output columns),
  never silently presenting the proxy as the preferred metric.
- `t_reconnect = first month after end-dry where LPI_t >= t_LPI, minus the
  end-dry month` (in whole months).
- Refuge spatial stability uses the end-dry-footprint Jaccard overlap
  between consecutive HYs (spec §6.16 option 1): `J_y = |R_y ∩ R_{y-1}| /
  |R_y ∪ R_{y-1}|`. `R_y` varies by HY (not a static long-term footprint).
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from hydrofragments.metrics.dynamics import (
    compute_reconnection_timing,
    compute_refuge_spatial_stability,
)
from tests.fixtures.analytic_masks import refuge_stability_fixture


def _lpi_series(monthly_values: list[tuple[str, float]]) -> list[tuple[datetime, float]]:
    return [(datetime.fromisoformat(date), value) for date, value in monthly_values]


def test_reconnection_timing_uses_lpi_proxy_and_flags_it():
    end_dry_month = datetime(2001, 10, 1)
    lpi_series = _lpi_series(
        [
            ("2001-10-01", 5.0),
            ("2001-11-01", 20.0),
            ("2001-12-01", 45.0),
            ("2002-01-01", 65.0),  # first month LPI >= 60 threshold
            ("2002-02-01", 80.0),
        ]
    )
    result = compute_reconnection_timing(
        lpi_series=lpi_series,
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )
    assert result.reconnection_metric_used == "LPI"
    assert result.proxy_reconnection_flag is True
    assert result.t_reconnect_months == 3  # Oct -> Jan is 3 months


def test_reconnection_timing_no_reconnection_within_series_is_none():
    end_dry_month = datetime(2001, 10, 1)
    lpi_series = _lpi_series(
        [("2001-10-01", 5.0), ("2001-11-01", 10.0), ("2001-12-01", 15.0)]
    )
    result = compute_reconnection_timing(
        lpi_series=lpi_series, end_dry_month=end_dry_month, lpi_threshold=60.0
    )
    assert result.t_reconnect_months is None


def test_reconnection_ignores_months_before_end_dry():
    end_dry_month = datetime(2001, 10, 1)
    result = compute_reconnection_timing(
        lpi_series=_lpi_series(
            [("2001-09-01", 80.0), ("2001-10-01", 10.0), ("2001-11-01", 65.0)]
        ),
        end_dry_month=end_dry_month,
        lpi_threshold=60.0,
    )

    assert result.t_reconnect_months == 1


def test_refuge_spatial_stability_jaccard_overlap_between_consecutive_end_dry_footprints():
    # R_{y-1}: 4 wet pixels; R_y: 3 of those 4 remain wet plus 1 new pixel.
    # Intersection = 3, union = 5 -> J = 0.6
    current_footprint, previous_footprint = refuge_stability_fixture()
    result = compute_refuge_spatial_stability(
        current_end_dry_footprint=current_footprint,
        previous_end_dry_footprint=previous_footprint,
    )
    assert result.jaccard == pytest.approx(0.6)


def test_refuge_spatial_stability_identical_footprints_is_one():
    footprint = np.array([[1, 0], [0, 1]], dtype=bool)
    result = compute_refuge_spatial_stability(
        current_end_dry_footprint=footprint, previous_end_dry_footprint=footprint
    )
    assert result.jaccard == pytest.approx(1.0)


def test_refuge_spatial_stability_no_previous_hy_is_none():
    footprint = np.array([[1, 0], [0, 1]], dtype=bool)
    result = compute_refuge_spatial_stability(
        current_end_dry_footprint=footprint, previous_end_dry_footprint=None
    )
    assert result.jaccard is None


def test_refuge_spatial_stability_both_empty_footprints_is_nan_not_zero_division():
    empty = np.zeros((2, 2), dtype=bool)
    result = compute_refuge_spatial_stability(
        current_end_dry_footprint=empty, previous_end_dry_footprint=empty
    )
    assert np.isnan(result.jaccard)
