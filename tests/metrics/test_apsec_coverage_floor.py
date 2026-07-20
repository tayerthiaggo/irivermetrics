"""m7: APSEC per-month coverage floor flag.

Occurrence already floors on minimum valid-pixel support (see
``compute_occurrence`` / ``config.validity.min_valid_obs`` in
``hydrofragments/metrics/persistence.py``), but APSEC did not: a sparse
month's ``water`` count could be silently emitted with no warning that most
of the raster was unobserved that month.

This adds purely additive metadata: ``ApsecRecord.low_coverage_flag`` is
``True`` when a month's fraction of valid pixels (``valid_obs`` mean over the
spatial dims) falls below an optional ``min_valid_fraction`` threshold. The
APSEC ``value`` arithmetic itself is untouched -- see
``test_apsec_value_unchanged_by_coverage_floor`` below, which pins that a
flagged, low-coverage month still produces the same numeric value as before
this feature existed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import analyze, open_water_cube
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec
from hydrofragments.schema import EdgeFlag
from tests.fixtures.analytic_masks import apsec_low_coverage_month_fixture


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
    return xr.Dataset(
        {
            "water": xr.DataArray(water.astype(bool), dims=dims, coords={"time": times}),
            "valid_obs": xr.DataArray(valid.astype(bool), dims=dims, coords={"time": times}),
        }
    )


def test_low_coverage_month_flagged():
    # 6 months x 10x10 grid (100 px/month). Month 3 (index 3) has 10 invalid
    # pixels injected (90% coverage) -- below the 0.95 floor. All other
    # months are fully valid (100% coverage).
    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    water = np.zeros((6, 10, 10))
    valid = np.ones((6, 10, 10))
    valid[3, 0, 0:10] = 0  # 10 invalid pixels in month index 3 -> 90% valid

    records = compute_apsec(
        _monthly(water, valid, times),
        a_ref_m2=90000.0,
        cell_area_m2=900.0,
        config=_config(),
        valid_obs=_monthly(water, valid, times)["valid_obs"],
        min_valid_fraction=0.95,
    )

    assert records[3].low_coverage_flag is True
    assert all(records[i].low_coverage_flag is False for i in (0, 1, 2, 4, 5))


def test_no_floor_check_when_valid_obs_not_supplied():
    """Default behaviour (no valid_obs/min_valid_fraction) never flags."""
    times = pd.date_range("2020-01-01", periods=2, freq="MS")
    water = np.zeros((2, 3, 3))
    valid = np.ones((2, 3, 3))
    valid[1, :, :] = 0  # month 1 fully unobserved

    records = compute_apsec(
        _monthly(water, valid, times),
        a_ref_m2=8100.0,
        cell_area_m2=900.0,
        config=_config(),
    )

    assert all(r.low_coverage_flag is False for r in records)


def test_apsec_value_unchanged_by_coverage_floor():
    """The flag is purely additive metadata; the APSEC value must not change."""
    times = pd.to_datetime(["2020-01-01"])
    water = np.zeros((1, 3, 3))
    water[0, 0, :] = 1  # 3 wet pixels
    valid = np.ones((1, 3, 3))
    valid[0, 0, 0] = 0  # 1 invalid pixel -> 8/9 coverage, below a 0.95 floor

    monthly = _monthly(water, valid, times)
    a_ref_m2 = 8100.0
    cell_area_m2 = 900.0

    flagged = compute_apsec(
        monthly,
        a_ref_m2=a_ref_m2,
        cell_area_m2=cell_area_m2,
        config=_config(),
        valid_obs=monthly["valid_obs"],
        min_valid_fraction=0.95,
    )
    unflagged = compute_apsec(
        monthly, a_ref_m2=a_ref_m2, cell_area_m2=cell_area_m2, config=_config()
    )

    assert flagged[0].low_coverage_flag is True
    assert flagged[0].value == pytest.approx(unflagged[0].value)
    assert flagged[0].n_water_pixels == unflagged[0].n_water_pixels


def test_low_coverage_flag_defaults_to_false_dataclass_field():
    from hydrofragments.metrics.extent import ApsecRecord

    record = ApsecRecord(
        date=pd.Timestamp("2020-01-01").to_pydatetime(),
        value=0.0,
        n_water_pixels=0,
        a_ref_m2=100.0,
        cell_area_m2=1.0,
    )
    assert record.low_coverage_flag is False


def test_analyze_marks_low_coverage_apsec_month(tmp_path):
    """The live analyze() path must surface APSEC low-coverage metadata."""
    water_np, valid_np, times = apsec_low_coverage_month_fixture()
    water = xr.DataArray(
        water_np,
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    valid_obs = xr.DataArray(
        valid_np,
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, valid_obs=valid_obs, input_kind="generic_binary")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core"],
            "input": {"kind": "generic_binary"},
            "validity": {"min_valid_fraction_month": 0.75, "min_valid_obs": 1},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"output_dir": str(tmp_path)},
        }
    )

    result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=10.0)

    apsec = result.metrics_table[result.metrics_table["metric"] == "apsec"].sort_values(
        "date"
    )
    assert apsec["value"].tolist() == pytest.approx([100.0, 100.0])
    assert apsec["valid_fraction_month"].notna().all()
    assert apsec["valid_fraction_month"].astype(float).tolist() == pytest.approx(
        [1.0, 0.25]
    )
    assert apsec["min_valid_fraction_month"].notna().all()
    assert apsec["min_valid_fraction_month"].astype(float).tolist() == pytest.approx(
        [0.75, 0.75]
    )
    edge_flags = apsec["edge_flag"].tolist()
    assert pd.isna(edge_flags[0])
    assert edge_flags[1] == EdgeFlag.LOW_VALID_OBS.value
    assert apsec["is_reportable"].tolist() == [True, False]

    sparse_month = result.metrics_table[
        (result.metrics_table["date"] == pd.Timestamp("2020-02-01"))
        & (result.metrics_table["value_type"] == "monthly")
    ]
    assert set(sparse_month["metric"]) == {
        "apsec",
        "awmsi",
        "awre",
        "lpi",
        "number_of_pools",
    }
    assert sparse_month["valid_fraction_month"].astype(float).tolist() == pytest.approx(
        [0.25] * len(sparse_month)
    )
    assert set(sparse_month["edge_flag"]) == {EdgeFlag.LOW_VALID_OBS.value}
    assert sparse_month["is_reportable"].tolist() == [False] * len(sparse_month)
