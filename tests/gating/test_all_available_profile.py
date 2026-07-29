"""W4.1: ``all_available`` default profile and registry-wide metric coverage.

``all_available`` is the new default ``metric_profiles`` value (see
``hydrofragments/config.py``). It means "attempt every runtime-wired metric
(``hydrofragments.metrics.registry.RUNTIME_WIRED_METRIC_IDS``) whose
dependencies are actually present in this run's inputs" -- not "compute every
metric in the registry". Metrics outside ``RUNTIME_WIRED_METRIC_IDS`` (mesh,
reconnection_timing, refuge_spatial_stability, realised_connectivity, tcf)
are never selected by ``all_available``, and get their own explicit
non-dependency skip reasons in ``HydroResult.metric_coverage``.

Step 1: default-resolution tests (water + validity only).
Step 2: incremental dependency fixtures (width floor, channel, HY + dual
composite), each proving its metric computes exactly once.
Step 3 covered by ``tests/contracts/test_registry.py`` and reuse of
``_available_dependencies`` from both ``validate_inputs`` and ``analyze``.
Step 4: config default + explicit profile/override preservation.
Step 5: ``metric_coverage`` schema, computed-vs-skipped bucketing, and the
default_factory backward-compatibility contract.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import LineString, box

from hydrofragments import HydroConfig, HydroResult, analyze, open_water_cube
from hydrofragments.metrics import ApsecRecord
from hydrofragments.metrics.registry import (
    METRIC_REGISTRY,
    NOT_RUNTIME_WIRED_REASONS,
)
from hydrofragments.models import AnalysisInputs
from hydrofragments.spatial import create_channel_context


def _default_config(tmp_path, **overrides):
    base = {
        "config_schema_version": "1.0.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
        "output": {"output_dir": str(tmp_path)},
    }
    base.update(overrides)
    return HydroConfig.from_mapping(base)


# ---------------------------------------------------------------------------
# Step 1: default-resolution tests
# ---------------------------------------------------------------------------


def test_default_profile_is_all_available() -> None:
    """No metric_profiles key -> HydroConfig defaults to ("all_available",)."""
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
        }
    )
    assert config.metric_profiles == ("all_available",)


def test_default_run_selects_core_recurrence_and_hydroperiod_only(
    synthetic_cube, tmp_path
) -> None:
    """Water + validity only (no channel/width/HY/dual-composite inputs):
    default selects contracts_core + recurrence + hydroperiod, and records
    exact skip reasons for every channel/dynamics/width metric."""
    config = _default_config(tmp_path)

    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    computed_metrics = set(result.metrics_table["metric"])
    assert computed_metrics == {
        "occurrence",
        "refuge_area",
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
        "recurrence",
        "hydroperiod",
    }

    coverage = result.metric_coverage.set_index("metric")
    for metric_id in computed_metrics:
        assert coverage.loc[metric_id, "status"] == "computed"
        assert bool(coverage.loc[metric_id, "runtime_wired"]) is True

    # Channel/width/dynamics metrics are runtime-wired but their structural
    # dependencies are genuinely absent from this input -> missing dependency.
    for metric_id, expected_missing in (
        ("lpsec", "requires_channel"),
        ("inter_pool_gap", "requires_channel"),
        ("pool_width", "requires_width_floor"),
        ("extent_contraction", "requires_HY_anchor"),
    ):
        row = coverage.loc[metric_id]
        assert row["status"] == "skipped (missing dependency)"
        assert expected_missing in row["reason"]
        assert bool(row["runtime_wired"]) is True

    # Never-runtime-wired registry metrics get their own explicit reasons.
    for metric_id, expected_reason in NOT_RUNTIME_WIRED_REASONS.items():
        row = coverage.loc[metric_id]
        assert row["status"] == expected_reason
        assert row["reason"] == expected_reason
        assert bool(row["runtime_wired"]) is False


def test_metric_coverage_has_exactly_one_row_per_registry_metric(
    synthetic_cube, tmp_path
) -> None:
    config = _default_config(tmp_path)
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    coverage = result.metric_coverage
    assert sorted(coverage["metric"]) == sorted(METRIC_REGISTRY)
    assert coverage["metric"].is_unique
    assert list(coverage.columns) == [
        "metric",
        "runtime_wired",
        "status",
        "rows",
        "reportable_rows",
        "reason",
    ]


# ---------------------------------------------------------------------------
# Step 2: incremental dependency fixtures
# ---------------------------------------------------------------------------


def _patch_cube():
    times = pd.to_datetime(["2020-01-01"])
    mask = np.zeros((1, 5, 9), dtype=bool)
    mask[0, 0, 0:4] = True
    mask[0, 2:5, 6:9] = True
    return open_water_cube(
        xr.DataArray(mask, dims=("time", "y", "x"), coords={"time": times}),
        input_kind="generic_binary",
    )


def test_width_floor_unlocks_pool_width_computed_exactly_once(tmp_path) -> None:
    cube = _patch_cube()
    config = _default_config(
        tmp_path,
        patches={
            "connectivity_rule": 4,
            "min_patch_pixels": 3,
            "width_resolution_floor_pixels": 2.0,
        },
    )

    result = analyze(cube, aoi_id="demo", config=config, pixel_size_m=10.0)

    width_rows = result.metrics_table[result.metrics_table["metric"] == "pool_width"]
    # One row per statistic (mean/median/max/cv-if-finite) per month, not a
    # duplicate pass -- pin the exact count for this single-month fixture.
    assert len(width_rows) == 3  # mean, median, max (cv is non-finite: single patch of 1)
    coverage = result.metric_coverage.set_index("metric")
    assert coverage.loc["pool_width", "status"] == "computed"
    assert coverage.loc["pool_width", "rows"] == 3


def _channel_inputs():
    aoi = gpd.GeoDataFrame(geometry=[box(0, -1, 50, 1)], crs="EPSG:3577")
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [10],
            "To_Node": [11],
            "NextDownID": [-1],
        },
        geometry=[LineString([(0, 0), (50, 0)])],
        crs="EPSG:3577",
    )
    context = create_channel_context(
        "demo", aoi, drainage, drainage_id="synthetic-v1", target_crs="EPSG:3577"
    )
    return context


def test_real_channel_inputs_unlock_lpsec_and_gap_exactly_once(tmp_path) -> None:
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = xr.DataArray(
        np.ones((2, 1, 5), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = _default_config(tmp_path)
    context = _channel_inputs()

    result = analyze(
        cube,
        aoi_id="demo",
        config=config,
        inputs=AnalysisInputs(
            drainage=context,
            channel_wet_profiles=np.array(
                [[True, False, True, False, True], [True, True, True, True, True]]
            ),
            channel_segment_lengths_m=[10.0] * 5,
        ),
        pixel_size_m=30.0,
    )

    lpsec_rows = result.metrics_table[result.metrics_table["metric"] == "lpsec"]
    gap_rows = result.metrics_table[result.metrics_table["metric"] == "inter_pool_gap"]
    # 2 months -> exactly one lpsec row per month, computed exactly once.
    assert len(lpsec_rows) == 2
    assert len(gap_rows) > 0

    coverage = result.metric_coverage.set_index("metric")
    assert coverage.loc["lpsec", "status"] == "computed"
    assert coverage.loc["lpsec", "rows"] == 2
    assert coverage.loc["inter_pool_gap", "status"] == "computed"


def test_hy_and_dual_apsec_unlock_extent_contraction_exactly_once(tmp_path) -> None:
    times = pd.date_range("2001-01-01", periods=36, freq="MS")
    water = xr.DataArray(
        np.ones((36, 1, 1), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    cube = open_water_cube(water, input_kind="generic_binary")
    config = _default_config(
        tmp_path,
        temporal={
            "input_cadence": "monthly",
            "monthly_composite": "max_water",
            "composite_owner": "caller",
        },
    )
    extent_values = np.tile([70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55], 3)
    extent = pd.Series(extent_values, index=times)
    max_records = [
        ApsecRecord(
            date=ts.to_pydatetime(),
            value=float(value),
            n_water_pixels=0,
            a_ref_m2=1.0,
            cell_area_m2=1.0,
        )
        for ts, value in zip(times, extent_values)
    ]
    median_records = [
        ApsecRecord(
            date=ts.to_pydatetime(),
            value=float(value - 1),
            n_water_pixels=0,
            a_ref_m2=1.0,
            cell_area_m2=1.0,
        )
        for ts, value in zip(times, extent_values)
    ]

    result = analyze(
        cube,
        aoi_id="demo",
        config=config,
        inputs=AnalysisInputs(
            hydroyear_extent=extent,
            max_water_apsec=max_records,
            median_apsec=median_records,
        ),
    )

    contraction_rows = result.metrics_table[
        result.metrics_table["metric"] == "extent_contraction"
    ]
    assert len(contraction_rows) > 0
    assert set(contraction_rows["monthly_composite"]) == {"max_water", "median"}

    coverage = result.metric_coverage.set_index("metric")
    assert coverage.loc["extent_contraction", "status"] == "computed"
    assert coverage.loc["extent_contraction", "rows"] == len(contraction_rows)


# ---------------------------------------------------------------------------
# Step 4: explicit named profiles and metric overrides still work
# ---------------------------------------------------------------------------


def test_explicit_named_profile_still_narrows_selection(synthetic_cube, tmp_path) -> None:
    config = _default_config(tmp_path, metric_profiles=["pixel_temporal"])
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    metrics = set(result.metrics_table["metric"])
    assert metrics == {"recurrence", "hydroperiod"}
    assert metrics.isdisjoint({"number_of_pools", "lpi", "awre", "awmsi", "mesh"})


def test_explicit_contracts_core_profile_still_works(synthetic_cube, tmp_path) -> None:
    config = _default_config(tmp_path, metric_profiles=["contracts_core"])
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    metrics = set(result.metrics_table["metric"])
    assert metrics == {
        "occurrence",
        "refuge_area",
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
    }


def test_narrow_profile_does_not_mark_unselected_runtime_wired_metric_computed(
    synthetic_cube, tmp_path
) -> None:
    """A metric that is runtime-wired and whose dependency IS available (so
    it *would* run under all_available) must not be reported "computed" in
    metric_coverage when an explicit narrower profile never selected it --
    its kernel genuinely did not execute this run. Regression test: an
    earlier version of _build_metric_coverage derived "selected" purely from
    the registry-wide all_available plan, ignoring the caller's actual
    config.metric_profiles, and incorrectly reported recurrence/hydroperiod
    as "computed" (with a fabricated "no rows produced" reason) even though
    contracts_core alone never runs their kernel."""
    config = _default_config(tmp_path, metric_profiles=["contracts_core"])
    result = analyze(synthetic_cube, aoi_id="demo", config=config, pixel_size_m=30.0)

    assert {"recurrence", "hydroperiod"}.isdisjoint(
        set(result.metrics_table["metric"])
    )
    coverage = result.metric_coverage.set_index("metric")
    for metric_id in ("recurrence", "hydroperiod"):
        row = coverage.loc[metric_id]
        assert row["status"] == "skipped (not selected by profile)"
        assert bool(row["runtime_wired"]) is True
        assert row["rows"] == 0
        assert row["reportable_rows"] == 0


# ---------------------------------------------------------------------------
# Step 5: HydroResult.metric_coverage backward compatibility
# ---------------------------------------------------------------------------


def test_hydro_result_metric_coverage_defaults_when_omitted(tmp_path) -> None:
    """Existing construction sites that don't know about metric_coverage
    (built before this field existed) must keep working unchanged."""
    result = HydroResult(
        metrics_table=pd.DataFrame(),
        manifest={},
        output_dir=tmp_path,
        run_id="abc123",
    )
    assert isinstance(result.metric_coverage, pd.DataFrame)
    assert result.metric_coverage.empty
    assert list(result.metric_coverage.columns) == [
        "metric",
        "runtime_wired",
        "status",
        "rows",
        "reportable_rows",
        "reason",
    ]


def test_two_hydro_results_get_independent_default_coverage_frames(tmp_path) -> None:
    """default_factory (not a shared mutable default) must produce a fresh
    frame per instance."""
    first = HydroResult(
        metrics_table=pd.DataFrame(), manifest={}, output_dir=tmp_path, run_id="a"
    )
    second = HydroResult(
        metrics_table=pd.DataFrame(), manifest={}, output_dir=tmp_path, run_id="b"
    )
    assert first.metric_coverage is not second.metric_coverage
