"""End-to-end dynamics profile wiring for reconnection and refuge stability."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments import HydroConfig, analyze, open_water_cube
from hydrofragments.metrics.extent import ApsecRecord
from hydrofragments.models import AnalysisInputs
from hydrofragments.schema import EdgeFlag, SCHEMA_VERSION, WarningFlag
from hydrofragments.spatial import create_channel_context
from tests.fixtures.dynamics_pipeline import dynamics_pipeline_fixture
import geopandas as gpd
from shapely.geometry import LineString, box


def _dynamics_config(tmp_path, **changes: object) -> HydroConfig:
    mapping: dict[str, object] = {
        "config_schema_version": "1.0.0",
        "metric_profiles": ["dynamics"],
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "max_water",
            "composite_owner": "caller",
        },
        "output": {"output_dir": str(tmp_path)},
    }
    mapping.update(changes)
    return HydroConfig.from_mapping(mapping)


def _channel_context() -> object:
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
    return create_channel_context(
        "demo", aoi, drainage, drainage_id="synthetic-v1", target_crs="EPSG:3577"
    )


def _run_dynamics(fixture, tmp_path, **config_changes: object):
    config = _dynamics_config(tmp_path, **config_changes)
    return analyze(
        fixture.cube,
        aoi_id="demo",
        config=config,
        pixel_size_m=10.0,
        inputs=AnalysisInputs(
            hydroyear_extent=fixture.hydroyear_extent,
            max_water_apsec=fixture.max_water_apsec,
            median_apsec=fixture.median_apsec,
        ),
    )


def test_dynamics_profile_emits_all_three_metrics(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    metrics = set(result.metrics_table["metric"])
    assert metrics >= {
        "extent_contraction",
        "reconnection_timing",
        "refuge_spatial_stability",
    }
    assert "lpi" not in metrics


def test_lpi_support_computed_without_output_row(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    config = _dynamics_config(
        tmp_path,
        metric_overrides={"remove": ["lpi"]},
    )
    result = analyze(
        fixture.cube,
        aoi_id="demo",
        config=config,
        pixel_size_m=10.0,
        inputs=AnalysisInputs(
            hydroyear_extent=fixture.hydroyear_extent,
            max_water_apsec=fixture.max_water_apsec,
            median_apsec=fixture.median_apsec,
        ),
    )

    assert "lpi" not in set(result.metrics_table["metric"])
    reconnect = result.metrics_table[
        result.metrics_table["metric"] == "reconnection_timing"
    ]
    assert reconnect["reconnection_metric_used"].notna().any()


def test_reconnection_uses_lpi_proxy_with_warning(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    rows = result.metrics_table[
        result.metrics_table["metric"] == "reconnection_timing"
    ].sort_values("hy")
    reportable = rows[rows["is_reportable"]]
    assert not reportable.empty
    assert reportable.iloc[0]["reconnection_metric_used"] == "LPI"
    assert bool(reportable.iloc[0]["proxy_reconnection_flag"]) is True
    assert WarningFlag.PROXY_RECONNECTION.value in reportable.iloc[0]["warning_flags"]


def test_reconnection_threshold_equality_counts_as_crossing(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    rows = result.metrics_table[
        (result.metrics_table["metric"] == "reconnection_timing")
        & result.metrics_table["is_reportable"]
    ]
    assert rows.iloc[0]["value"] == 1.0


def test_refuge_stability_uses_common_valid_support(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    rows = result.metrics_table[
        result.metrics_table["metric"] == "refuge_spatial_stability"
    ].sort_values("hy")
    second_hy = rows.iloc[1]
    assert second_hy["is_reportable"]
    assert second_hy["value"] == pytest.approx(0.6)
    assert second_hy["valid_fraction_month"] is not pd.NA


def test_first_hy_refuge_stability_is_non_reportable(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    first = result.metrics_table[
        (result.metrics_table["metric"] == "refuge_spatial_stability")
        & (result.metrics_table["hy"] == fixture.expected_first_hy)
    ].iloc[0]
    assert first["is_reportable"] == False  # noqa: E712
    assert first["edge_flag"] == EdgeFlag.NO_PREVIOUS_HY.value


def test_emitted_rows_carry_hy_provenance(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    rows = result.metrics_table[
        result.metrics_table["metric"].isin(
            {"reconnection_timing", "refuge_spatial_stability"}
        )
    ]
    assert rows["date"].notna().all()
    assert rows["hy"].notna().all()
    assert rows["hy_anchor"].eq("end_dry").all()
    assert rows["hy_confidence"].notna().all()
    reconnect = rows[rows["metric"] == "reconnection_timing"].iloc[0]
    assert reconnect["connected_wet_metric"] == "LPI"
    assert reconnect["connected_wet_threshold"] == 50.0


def test_schema_version_is_current(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    result = _run_dynamics(fixture, tmp_path)

    assert set(result.metrics_table["schema_version"]) == {SCHEMA_VERSION}


def test_parallel_workers_match_serial_dynamics(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    serial = _run_dynamics(
        fixture,
        tmp_path / "serial",
        compute={"workers": 1},
    )
    parallel = _run_dynamics(
        fixture,
        tmp_path / "parallel",
        compute={"workers": 2},
    )

    dynamics_metrics = {"reconnection_timing", "refuge_spatial_stability"}
    serial_view = (
        serial.metrics_table[serial.metrics_table["metric"].isin(dynamics_metrics)]
        .sort_values(["metric", "hy"])
        .reset_index(drop=True)
    )
    parallel_view = (
        parallel.metrics_table[parallel.metrics_table["metric"].isin(dynamics_metrics)]
        .sort_values(["metric", "hy"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        serial_view[
            [
                "metric",
                "hy",
                "value",
                "is_reportable",
                "edge_flag",
                "reconnection_metric_used",
                "proxy_reconnection_flag",
                "warning_flags",
            ]
        ],
        parallel_view[
            [
                "metric",
                "hy",
                "value",
                "is_reportable",
                "edge_flag",
                "reconnection_metric_used",
                "proxy_reconnection_flag",
                "warning_flags",
            ]
        ],
    )


def test_lpsec_preferred_when_complete_channel_inputs_exist(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    times = pd.to_datetime(fixture.cube.water["time"].values)
    context = _channel_context()
    wet_profiles = []
    for timestamp in times:
        key = (timestamp.year, timestamp.month)
        end_dry_keys = {
            (ts.year, ts.month) for ts in fixture.expected_end_dry_months
        }
        reconnect_key = (
            fixture.expected_end_dry_months[0].year,
            fixture.expected_end_dry_months[0].month + 1,
        )
        if reconnect_key[1] > 12:
            reconnect_key = (reconnect_key[0] + 1, 1)
        if key in end_dry_keys:
            wet_profiles.append([False, False, False, False])
        elif key == reconnect_key:
            wet_profiles.append([True, True, True, True])
        else:
            wet_profiles.append([True, False, False, False])

    config = _dynamics_config(tmp_path)
    result = analyze(
        fixture.cube,
        aoi_id="demo",
        config=config,
        pixel_size_m=10.0,
        inputs=AnalysisInputs(
            hydroyear_extent=fixture.hydroyear_extent,
            max_water_apsec=fixture.max_water_apsec,
            median_apsec=fixture.median_apsec,
            drainage=context,
            channel_wet_profiles=wet_profiles,
            channel_segment_lengths_m=[10.0, 10.0, 10.0, 10.0],
        ),
    )

    rows = result.metrics_table[
        (result.metrics_table["metric"] == "reconnection_timing")
        & result.metrics_table["is_reportable"]
    ]
    assert rows.iloc[0]["reconnection_metric_used"] == "LPSEC"
    assert bool(rows.iloc[0]["proxy_reconnection_flag"]) is True


def test_lpsec_without_crossing_does_not_fall_back_to_lpi(tmp_path) -> None:
    fixture = dynamics_pipeline_fixture()
    times = pd.to_datetime(fixture.cube.water["time"].values)
    context = _channel_context()
    wet_profiles = [[True, False, False, False] for _ in times]
    segment_lengths = [10.0, 10.0, 10.0, 10.0]

    config = _dynamics_config(tmp_path)
    result = analyze(
        fixture.cube,
        aoi_id="demo",
        config=config,
        pixel_size_m=10.0,
        inputs=AnalysisInputs(
            hydroyear_extent=fixture.hydroyear_extent,
            max_water_apsec=fixture.max_water_apsec,
            median_apsec=fixture.median_apsec,
            drainage=context,
            channel_wet_profiles=wet_profiles,
            channel_segment_lengths_m=segment_lengths,
        ),
    )

    rows = result.metrics_table[
        result.metrics_table["metric"] == "reconnection_timing"
    ].sort_values("hy")
    first = rows.iloc[0]
    assert first["reconnection_metric_used"] == "LPSEC"
    assert first["is_reportable"] == False  # noqa: E712
    assert first["edge_flag"] == EdgeFlag.NO_THRESHOLD_CROSSING.value


def test_no_crossing_emits_non_reportable_reason(tmp_path) -> None:
    times = pd.date_range("2001-01-01", periods=36, freq="MS")
    water = np.zeros((36, 2, 2), dtype=bool)
    water[:, 0, 0] = True
    cube = open_water_cube(
        xr.DataArray(water.astype(np.uint8), dims=("time", "y", "x"), coords={"time": times}),
        input_kind="generic_binary",
    )
    extent = pd.Series(np.tile([70, 90, 80, 60, 40, 25, 15, 10, 8, 5, 30, 55], 3), index=times)
    max_records = [
        ApsecRecord(
            date=ts.to_pydatetime(),
            value=float(value),
            n_water_pixels=1,
            a_ref_m2=400.0,
            cell_area_m2=100.0,
        )
        for ts, value in zip(times, extent)
    ]
    median_records = [
        ApsecRecord(
            date=record.date,
            value=record.value - 1.0,
            n_water_pixels=record.n_water_pixels,
            a_ref_m2=record.a_ref_m2,
            cell_area_m2=record.cell_area_m2,
        )
        for record in max_records
    ]

    result = analyze(
        cube,
        aoi_id="demo",
        config=_dynamics_config(tmp_path),
        pixel_size_m=10.0,
        inputs=AnalysisInputs(
            hydroyear_extent=extent,
            max_water_apsec=max_records,
            median_apsec=median_records,
        ),
    )

    rows = result.metrics_table[
        result.metrics_table["metric"] == "reconnection_timing"
    ]
    assert rows["is_reportable"].eq(False).all()
    assert rows["edge_flag"].eq(EdgeFlag.NO_THRESHOLD_CROSSING.value).any()
