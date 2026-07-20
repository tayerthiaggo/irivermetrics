"""Baseline data-quality assessment and gapfill prescription (Section 2).

HydroFragments never gapfills (that behavior lives only in the companion tool
WaterMask-TSFill, out of scope here). This module only *assesses* how much
valid-observation coverage the input cube actually has and, when it falls
below the configured floors and the user has not declared ``gapfill: true``,
*recommends* pre-processing with WaterMask-TSFill. The recommendation is
purely advisory: it must never mutate ``cube.water``/``cube.valid_obs``.

Fixture layout for the hand-derived MNAR test (2 calendar years, 2 calendar
months -- Jan and Jul -- one pixel):

    Jan 2020: valid            -> water=True
    Jan 2021: valid            -> water=False
    Jul 2020: valid            -> water=True
    Jul 2021: NOT valid (missing observation)

Per-calendar-month ratio (sum(water & valid) / sum(valid)):
    Jan: water_valid=1, valid=2 -> ratio = 0.5
    Jul: water_valid=1, valid=1 -> ratio = 1.0

Season-stratified occurrence (equal-weight mean across contributing calendar
months) = mean(0.5, 1.0) = 0.75 -> 75.0 (percent), matching
``_season_stratified_occurrence`` in ``hydrofragments/metrics/persistence.py``.

Per-month valid fraction (valid observations / total timesteps in that
calendar-month bucket across the record):
    Jan: 2 valid / 2 timesteps = 1.0
    Jul: 1 valid / 2 timesteps = 0.5

Overall valid_obs coverage for the single pixel: 3 valid / 4 timesteps = 0.75.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig
from hydrofragments.guards.quality import (
    WATERMASK_TSFILL_HINT,
    BaselineQualityReport,
    assess_baseline_quality,
)
from hydrofragments.models import WaterCube


def _config(*, gapfill: bool = False, min_valid_obs: int = 20,
            min_valid_fraction_month: float = 0.70) -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "validity": {
                "min_valid_obs": min_valid_obs,
                "min_valid_fraction_month": min_valid_fraction_month,
            },
            "gapfill": gapfill,
        }
    )


def _low_coverage_cube() -> WaterCube:
    """A cube with deliberately poor valid-obs coverage (well under any floor)."""
    times = pd.date_range("2020-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(7)
    water = rng.random((24, 3, 3)) > 0.5
    valid = np.zeros((24, 3, 3), dtype=bool)
    # Only 2 of 24 months observed -> coverage is far below any reasonable floor.
    valid[0] = True
    valid[1] = True
    return WaterCube(
        water=xr.DataArray(water, dims=("time", "y", "x"), coords={"time": times}),
        valid_obs=xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times}),
        source="synthetic_low_coverage",
        cadence="monthly",
    )


def _high_coverage_cube() -> WaterCube:
    times = pd.date_range("2020-01-01", periods=24, freq="MS")
    rng = np.random.default_rng(3)
    water = rng.random((24, 3, 3)) > 0.5
    valid = np.ones((24, 3, 3), dtype=bool)
    return WaterCube(
        water=xr.DataArray(water, dims=("time", "y", "x"), coords={"time": times}),
        valid_obs=xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times}),
        source="synthetic_high_coverage",
        cadence="monthly",
    )


def _hand_derived_mnar_cube() -> WaterCube:
    """2-year, 2-calendar-month (Jan/Jul), single-pixel fixture -- see module docstring."""
    times = pd.to_datetime(
        ["2020-01-01", "2020-07-01", "2021-01-01", "2021-07-01"]
    )
    # water:  Jan20=True, Jul20=True, Jan21=False, Jul21=<don't care, invalid>
    water = np.array([True, True, False, False]).reshape(4, 1, 1)
    valid = np.array([True, True, True, False]).reshape(4, 1, 1)
    return WaterCube(
        water=xr.DataArray(water, dims=("time", "y", "x"), coords={"time": times}),
        valid_obs=xr.DataArray(valid, dims=("time", "y", "x"), coords={"time": times}),
        source="synthetic_hand_derived",
        cadence="monthly",
    )


class TestAssessBaselineQuality:
    def test_returns_baseline_quality_report(self):
        report = assess_baseline_quality(_high_coverage_cube(), config=_config())
        assert isinstance(report, BaselineQualityReport)

    def test_never_mutates_cube_data(self):
        cube = _low_coverage_cube()
        water_before = cube.water.copy(deep=True)
        valid_before = cube.valid_obs.copy(deep=True)
        assess_baseline_quality(cube, config=_config())
        xr.testing.assert_identical(cube.water, water_before)
        xr.testing.assert_identical(cube.valid_obs, valid_before)


class TestGapfillRecommendation:
    def test_low_coverage_and_gapfill_false_recommends(self):
        report = assess_baseline_quality(
            _low_coverage_cube(), config=_config(gapfill=False)
        )
        assert report.recommend_gapfill is True
        assert report.reason is not None
        assert "WaterMask-TSFill" in report.reason
        assert WATERMASK_TSFILL_HINT in report.reason

    def test_high_coverage_and_gapfill_false_does_not_recommend(self):
        report = assess_baseline_quality(
            _high_coverage_cube(), config=_config(gapfill=False)
        )
        assert report.recommend_gapfill is False
        assert report.reason is None

    def test_low_coverage_and_gapfill_true_suppresses_recommendation(self):
        report = assess_baseline_quality(
            _low_coverage_cube(), config=_config(gapfill=True)
        )
        assert report.recommend_gapfill is False
        assert report.reason is None


class TestMnarHandDerivedFixture:
    def test_per_month_valid_fraction_hand_checked(self):
        report = assess_baseline_quality(
            _hand_derived_mnar_cube(), config=_config()
        )
        # January bucket: 2 valid / 2 timesteps = 1.0
        # July bucket: 1 valid / 2 timesteps = 0.5
        by_month = dict(report.valid_fraction_by_month)
        assert by_month[1] == pytest.approx(1.0)
        assert by_month[7] == pytest.approx(0.5)

    def test_seasonal_occurrence_hand_checked(self):
        report = assess_baseline_quality(
            _hand_derived_mnar_cube(), config=_config()
        )
        # mean(0.5, 1.0) * 100 = 75.0, matching
        # metrics.persistence._season_stratified_occurrence exactly.
        assert report.seasonal_occurrence_pct == pytest.approx(75.0)

    def test_overall_valid_fraction_hand_checked(self):
        report = assess_baseline_quality(
            _hand_derived_mnar_cube(), config=_config()
        )
        # 3 valid observations out of 4 timesteps for the single pixel.
        assert report.overall_valid_fraction == pytest.approx(0.75)


class TestConfigDefault:
    def test_gapfill_defaults_false(self):
        assert HydroConfig.from_mapping(
            {
                "config_schema_version": "1.0.0",
                "input": {"kind": "generic_binary"},
                "temporal": {
                    "input_cadence": "monthly",
                    "monthly_composite": "supplied",
                    "composite_owner": "caller",
                },
            }
        ).gapfill is False


class TestValidateInputsWiring:
    def test_validation_report_warnings_contain_recommendation(self):
        from hydrofragments.api import validate_inputs

        report = validate_inputs(
            _low_coverage_cube(),
            "demo-aoi",
            config=_config(gapfill=False),
        )
        assert any("WaterMask-TSFill" in warning for warning in report.warnings)

    def test_validate_inputs_suppresses_when_gapfill_true(self):
        from hydrofragments.api import validate_inputs

        report = validate_inputs(
            _low_coverage_cube(),
            "demo-aoi",
            config=_config(gapfill=True),
        )
        assert not any("WaterMask-TSFill" in warning for warning in report.warnings)
