"""Milestone 9 — thin hydroyear adapter over external `hydroseason` package.

Load-bearing contract (Q7/V8, approved 2026-07-16):

- HY detection and season mapping are NOT reimplemented here. This module
  only calls `hydroseason.detect_hydrological_years` /
  `label_hydrological_months` and reshapes the result into HydroFragments'
  HY-anchor vocabulary (peak/end-dry months, HY id, season label,
  confidence).
- These are integration tests against the real installed `hydroseason`
  package (pinned `==0.1.0`), not unit tests of its detector algorithm.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import hydroseason

from hydrofragments.temporal.hydroyear import (
    HydroYearAdapterError,
    detect_hy_anchors,
    hydroseason_config_from_hydroconfig,
)
from hydrofragments.config import HydroConfig


def _monthly_extent_series(
    monthly_pct: dict[int, float], years: range
) -> pd.Series:
    """Build a repeating annual cycle from a 12-value calendar-month map."""
    start = f"{years.start}-01-01"
    end = f"{years.stop - 1}-12-01"
    dates = pd.date_range(start, end, freq="MS")
    values = [monthly_pct[int(date.month)] for date in dates]
    return pd.Series(values, index=dates, name="extent_pct")


# Clean seasonal cycle: wet Nov-Apr peaking Feb, dry Jul-Dec troughing Oct.
_REGULAR_CYCLE = {
    1: 70, 2: 90, 3: 80, 4: 60, 5: 40, 6: 25,
    7: 15, 8: 10, 9: 8, 10: 5, 11: 30, 12: 55,
}


def test_detect_hy_anchors_returns_peak_and_end_dry_month_for_regular_cycle():
    extent = _monthly_extent_series(_REGULAR_CYCLE, range(2001, 2011))
    result = detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())

    assert len(result.anchors) > 0
    row = result.anchors.iloc[0]
    assert row["peak_month"].month in {1, 2, 3}
    assert row["end_dry_month"].month in {9, 10, 11, 12}
    assert row["confidence"] in {"low", "medium", "high"}


def test_detect_hy_anchors_labels_months_with_hy_and_season():
    extent = _monthly_extent_series(_REGULAR_CYCLE, range(2001, 2011))
    result = detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())

    labels = result.month_labels
    assert set(labels["season"].unique()) <= {"Wet", "Dry", "unassigned"}
    assert labels["hy"].notna().any()


def test_detect_hy_anchors_under_drought_flags_low_confidence():
    # hydroseason's confidence is amplitude *relative to the series' own
    # typical (median) year* (see `hydro_year._assign_confidence`). Two
    # consecutive drought years (needed so the wet-search window, which
    # spans Nov(Y-1)..Apr(Y), does not straddle back into a normal prior
    # December) flattened to near-constant low extent must surface as "low"
    # confidence for the drought HY -- the adapter must not upgrade it.
    dates = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
    values = []
    for date in dates:
        pct = _REGULAR_CYCLE[int(date.month)]
        values.append(8.0 if date.year in (2004, 2005) else pct)
    extent = pd.Series(values, index=dates, name="extent_pct")
    result = detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())

    assert len(result.anchors) > 0
    assert "low" in set(result.anchors["confidence"])


def test_detect_hy_anchors_under_high_variability_returns_per_year_rows():
    # High year-to-year variability (alternating strong/weak wet seasons)
    # must not crash the adapter; every year with enough wet/dry coverage
    # still gets an anchor row (possibly low confidence), per hydroseason's
    # own per-year windowing contract.
    rng = np.random.default_rng(0)
    dates = pd.date_range("2001-01-01", "2010-12-01", freq="MS")
    base = np.array(
        [_REGULAR_CYCLE[int(date.month)] for date in dates], dtype=float
    )
    noisy = np.clip(base * rng.uniform(0.4, 1.6, size=len(base)), 0.0, 100.0)
    extent = pd.Series(noisy, index=dates, name="extent_pct")

    result = detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())
    assert len(result.anchors) >= 5


def test_detect_hy_anchors_missing_months_raises_adapter_error():
    dates = pd.date_range("2001-01-01", "2003-12-01", freq="MS")
    values = [_REGULAR_CYCLE[int(date.month)] for date in dates]
    extent = pd.Series(values, index=dates, name="extent_pct")
    extent = extent.drop(pd.Timestamp("2002-06-01"))

    with pytest.raises(HydroYearAdapterError):
        detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())


def test_detect_hy_anchors_records_hydroseason_version():
    extent = _monthly_extent_series(_REGULAR_CYCLE, range(2001, 2011))
    result = detect_hy_anchors(extent, config=hydroseason.HydroYearConfig())
    assert result.hydroseason_version == hydroseason.__version__


def test_local_hydrofragments_config_maps_to_hydroseason_config():
    local = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "hydroyear": {
                "algorithm": "hydroseason.detect_hydrological_years",
                "parameters": {"wet_start_month": 10, "min_wet_months": 3},
            },
        }
    )

    external = hydroseason_config_from_hydroconfig(local)

    assert external.wet_start_month == 10
    assert external.min_wet_months == 3
    assert external.wet_end_month == 4
