"""Milestone 9 — surface-water extent contraction (spec §6.5, dynamics tranche).

Load-bearing contract:

- `contraction = (APSEC_enddry - APSEC_peak) / n_months` between the peak-wet
  and end-dry HY anchors (linear/OLS slope on evenly-spaced monthly points is
  numerically identical to the two-point endpoint slope; Q11 locks the method
  as `linear`, `minimum_points=3`).
- Must be computed on **both** `max_water`- and `median`-composited monthly
  APSEC series; disagreement in end-dry APSEC beyond
  `dynamics.composite_sensitivity_tolerance_pp` (default 10pp) sets
  `composite_sensitive = True`.
- A single-composite (already-monthly, non-dual) input must refuse the
  dual-composite check rather than fabricate a second series from the one
  it has.
- Fewer than `minimum_points` usable months between anchors suppresses the
  slope to NaN with a `low_df` flag, rather than reporting an unreliable
  regression.
- User-facing text/attrs must never use recession/flow language.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.dynamics import (
    DualCompositeUnavailableError,
    compute_extent_contraction,
)
from hydrofragments.metrics.extent import ApsecRecord


def _config(tolerance_pp: float = 10.0, minimum_points: int = 3) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "submonthly",
                "monthly_composite": "max_water",
                "composite_owner": "hydrofragments",
            },
            "dynamics": {
                "composite_sensitivity_tolerance_pp": tolerance_pp,
                "contraction_method": "linear",
                "minimum_points": minimum_points,
            },
        }
    )


def _apsec_series(monthly_values: list[tuple[str, float]]) -> list[ApsecRecord]:
    return [
        ApsecRecord(
            date=datetime.fromisoformat(date),
            value=value,
            n_water_pixels=0,
            a_ref_m2=1.0,
            cell_area_m2=1.0,
        )
        for date, value in monthly_values
    ]


# Peak Feb (90%), declining monthly to end-dry Oct (5%): 8 months, -85pp span.
_MAX_WATER_SERIES = _apsec_series(
    [
        ("2001-02-01", 90.0), ("2001-03-01", 78.0), ("2001-04-01", 66.0),
        ("2001-05-01", 54.0), ("2001-06-01", 42.0), ("2001-07-01", 30.0),
        ("2001-08-01", 18.0), ("2001-09-01", 10.0), ("2001-10-01", 5.0),
    ]
)
# Median composite: same shape, slightly lower (less biased) throughout.
_MEDIAN_SERIES = _apsec_series(
    [
        ("2001-02-01", 88.0), ("2001-03-01", 75.0), ("2001-04-01", 62.0),
        ("2001-05-01", 49.0), ("2001-06-01", 36.0), ("2001-07-01", 24.0),
        ("2001-08-01", 14.0), ("2001-09-01", 8.0), ("2001-10-01", 4.0),
    ]
)

_ANCHOR = {
    "hy": 2001,
    "peak_month": datetime(2001, 2, 1),
    "end_dry_month": datetime(2001, 10, 1),
    "confidence": "high",
}


def test_contraction_slope_sign_and_units_from_max_water_series():
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=_ANCHOR,
        config=_config(),
    )
    # Negative slope (drying); magnitude close to (5 - 90) / 8 = -10.625 pp/month.
    assert result.slope_pct_per_month < 0
    assert result.slope_pct_per_month == pytest.approx(-10.625, abs=0.5)
    assert result.n_points == 9
    assert result.low_df is False
    assert result.median_slope_pct_per_month == pytest.approx(-10.5, abs=0.5)
    assert result.median_n_points == 9
    assert result.median_low_df is False


def test_contraction_uses_elapsed_calendar_months_and_ignores_nan_records():
    max_series = _apsec_series(
        [("2001-02-01", 90.0), ("2001-03-01", 60.0), ("2001-04-01", np.nan), ("2001-05-01", 0.0)]
    )
    median_series = _apsec_series(
        [("2001-02-01", 88.0), ("2001-03-01", 58.0), ("2001-04-01", np.nan), ("2001-05-01", 0.0)]
    )
    anchor = {
        "hy": 2001,
        "peak_month": datetime(2001, 2, 1),
        "end_dry_month": datetime(2001, 5, 1),
        "confidence": "high",
    }

    result = compute_extent_contraction(
        max_water=max_series,
        median=median_series,
        anchor=anchor,
        config=_config(),
    )

    # Month offsets are [0, 1, 3], not compressed observation positions [0, 1, 2].
    assert result.n_points == 3
    assert result.slope_pct_per_month == pytest.approx(-30.0, abs=0.01)


def test_contraction_composite_sensitive_flag_set_beyond_tolerance():
    # Bump the median end-dry value so max_water vs median disagree by >10pp.
    median_diverged = _apsec_series(
        [
            ("2001-02-01", 88.0), ("2001-03-01", 75.0), ("2001-04-01", 62.0),
            ("2001-05-01", 49.0), ("2001-06-01", 36.0), ("2001-07-01", 24.0),
            ("2001-08-01", 14.0), ("2001-09-01", 8.0), ("2001-10-01", 20.0),
        ]
    )
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=median_diverged,
        anchor=_ANCHOR,
        config=_config(tolerance_pp=10.0),
    )
    # end-dry: max_water=5.0, median=20.0 -> disagreement 15pp > 10pp tolerance.
    assert result.composite_sensitive is True


def test_contraction_composite_not_sensitive_within_tolerance():
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=_ANCHOR,
        config=_config(tolerance_pp=10.0),
    )
    # end-dry: max_water=5.0, median=4.0 -> disagreement 1pp <= 10pp tolerance.
    assert result.composite_sensitive is False


def test_contraction_refuses_single_composite_input():
    with pytest.raises(DualCompositeUnavailableError):
        compute_extent_contraction(
            max_water=_MAX_WATER_SERIES,
            median=None,
            anchor=_ANCHOR,
            config=_config(),
        )


def test_contraction_low_df_flag_when_below_minimum_points():
    short_series = _apsec_series([("2001-02-01", 90.0), ("2001-03-01", 60.0)])
    short_median = _apsec_series([("2001-02-01", 88.0), ("2001-03-01", 58.0)])
    short_anchor = {
        "hy": 2001,
        "peak_month": datetime(2001, 2, 1),
        "end_dry_month": datetime(2001, 3, 1),
        "confidence": "high",
    }
    result = compute_extent_contraction(
        max_water=short_series,
        median=short_median,
        anchor=short_anchor,
        config=_config(minimum_points=3),
    )
    assert result.low_df is True
    assert np.isnan(result.slope_pct_per_month)


def test_contraction_missing_anchor_returns_none_result():
    incomplete_anchor = {
        "hy": 2001,
        "peak_month": None,
        "end_dry_month": datetime(2001, 10, 1),
        "confidence": "unassigned",
    }
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=incomplete_anchor,
        config=_config(),
    )
    assert result is None


def test_contraction_metric_name_carries_no_recession_language():
    # The *metric name* (the identifier that ends up in the tidy output
    # `metric` column and any manager-facing table) must never read as a
    # recession-constant/flow claim. The free-text `description` MAY name
    # "recession"/"hydrograph" once, specifically to disclaim them (spec
    # §6.5 `[AUDIT FIX]`) -- that is the opposite of the failure mode this
    # guard exists to catch.
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=_ANCHOR,
        config=_config(),
    )
    forbidden = ("recession", "flow", "hydrograph", "k-parameter", "discharge")
    name = result.metric_name.lower()
    for term in forbidden:
        assert term not in name


def test_contraction_description_disclaims_recession_framing():
    # The description must explicitly disclaim the recession-constant
    # reading, not merely avoid the word.
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=_ANCHOR,
        config=_config(),
    )
    description = result.description.lower()
    assert "not" in description
    assert "recession" in description or "hydrograph" in description


def test_contraction_propagates_hy_confidence():
    result = compute_extent_contraction(
        max_water=_MAX_WATER_SERIES,
        median=_MEDIAN_SERIES,
        anchor=_ANCHOR,
        config=_config(),
    )
    assert result.hy_confidence == "high"
