"""Milestone 11 pipeline wiring -- FIXED_NODES/GRAPH availability in validate_inputs."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString

from hydrofragments.api import validate_inputs
from hydrofragments.config import HydroConfig
from hydrofragments.models import WaterCube
from hydrofragments.schema import MetricDependency
from hydrofragments.spatial.context import SpatialContext


def _minimal_config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "metric_profiles": ["connectivity"],
        }
    )


def _cube() -> WaterCube:
    water = xr.DataArray(
        np.zeros((1, 4, 4), dtype=bool),
        dims=("time", "y", "x"),
    )
    valid = xr.ones_like(water, dtype=bool)
    return WaterCube(water=water, valid_obs=valid, source="test", cadence="monthly", crs=None, provenance=())


def _real_channel_context() -> SpatialContext:
    # has_real_channel requires drainage is not None, drainage_id is not
    # None, l_ref_m > 0, and proxy_channel is False (see
    # hydrofragments/spatial/context.py:47-55) -- a minimal one-reach
    # drainage GeoDataFrame satisfies the "real channel" contract without
    # invoking the full create_channel_context AOI-clip pipeline.
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": ["A"],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": ["-1"],
            "geometry": [LineString([(0, 0), (10, 0)])],
        },
        crs="EPSG:3577",
    )
    return SpatialContext(
        aoi_id="aoi-1",
        area_m2=100.0,
        drainage_id="drainage-1",
        l_ref_m=10.0,
        crs="EPSG:3577",
        drainage=drainage,
        proxy_channel=False,
    )


def test_fixed_nodes_and_graph_unavailable_without_drainage():
    report = validate_inputs(_cube(), "aoi-1", config=_minimal_config())

    skipped_ids = {metric_id for metric_id, _ in report.skipped_metrics}
    assert "realised_connectivity" in skipped_ids or "tcf" in skipped_ids


def test_fixed_nodes_and_graph_unavailable_with_real_channel_but_no_wet_reaches():
    # A real channel alone is not sufficient -- has_real_channel says the
    # drainage geometry is usable, not that any reach was ever observed
    # wet. Without wet_any_month, FIXED_NODES/GRAPH must stay unavailable.
    context = _real_channel_context()

    report = validate_inputs(_cube(), "aoi-1", config=_minimal_config(), drainage=context)

    skipped_ids = {metric_id for metric_id, _ in report.skipped_metrics}
    assert "realised_connectivity" in skipped_ids or "tcf" in skipped_ids


def test_fixed_nodes_and_graph_available_with_real_channel_and_wet_reaches():
    context = _real_channel_context()

    report = validate_inputs(
        _cube(),
        "aoi-1",
        config=_minimal_config(),
        drainage=context,
        wet_any_month={"A": True},
    )

    skipped_ids = {metric_id for metric_id, _ in report.skipped_metrics}
    assert "realised_connectivity" not in skipped_ids
    assert "tcf" not in skipped_ids


def test_fixed_nodes_and_graph_unavailable_when_all_reaches_dry():
    context = _real_channel_context()

    report = validate_inputs(
        _cube(),
        "aoi-1",
        config=_minimal_config(),
        drainage=context,
        wet_any_month={"A": False},
    )

    skipped_ids = {metric_id for metric_id, _ in report.skipped_metrics}
    assert "realised_connectivity" in skipped_ids or "tcf" in skipped_ids
