from __future__ import annotations

import hashlib
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from hydrofragments.spatial.context import (
    DrainageContractError,
    create_channel_context,
    ordered_reach_paths,
    validate_drainage_topology,
)


def _drainage(geometry: LineString, *, crs: str = "EPSG:3577") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "HydroID": [101],
            "From_Node": [1],
            "To_Node": [2],
            "NextDownID": [-1],
        },
        geometry=[geometry],
        crs=crs,
    )


def test_drainage_topology_requires_real_line_fields() -> None:
    drainage = _drainage(LineString([(0, 0), (1, 0)])).drop(columns="NextDownID")

    with pytest.raises(DrainageContractError, match="NextDownID"):
        validate_drainage_topology(drainage)


def test_drainage_topology_rejects_null_nodes() -> None:
    drainage = _drainage(LineString([(0, 0), (1, 0)]))
    drainage.loc[0, "To_Node"] = None

    with pytest.raises(DrainageContractError, match="null"):
        validate_drainage_topology(drainage)


def test_drainage_topology_rejects_internal_node_mismatch() -> None:
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [1, 2],
            "From_Node": [10, 99],
            "To_Node": [11, 12],
            "NextDownID": [2, -1],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:3577",
    )
    with pytest.raises(DrainageContractError, match="node|NextDownID"):
        validate_drainage_topology(drainage)


def test_drainage_topology_rejects_cycles() -> None:
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [1, 2],
            "From_Node": [10, 11],
            "To_Node": [11, 10],
            "NextDownID": [2, 1],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (0, 0)])],
        crs="EPSG:3577",
    )
    with pytest.raises(DrainageContractError, match="cycle"):
        validate_drainage_topology(drainage)


def test_ordered_reach_paths_are_stable_when_input_rows_are_reordered() -> None:
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [1, 2, 3],
            "From_Node": [10, 11, 12],
            "To_Node": [11, 12, 13],
            "NextDownID": [2, 3, -1],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(1, 0), (2, 0)]),
            LineString([(2, 0), (3, 0)]),
        ],
        crs="EPSG:3577",
    )
    assert ordered_reach_paths(drainage) == ((1, 2, 3),)
    assert ordered_reach_paths(drainage.iloc[::-1].copy()) == ((1, 2, 3),)


def test_channel_context_aligns_crs_clips_to_aoi_and_derives_real_l_ref() -> None:
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3577")
    drainage = _drainage(LineString([(-5, 5), (15, 5)]))

    context = create_channel_context(
        "reach-1",
        aoi,
        drainage,
        drainage_id="synthetic-v1",
        target_crs="EPSG:3577",
    )

    assert context.crs == "EPSG:3577"
    assert context.area_m2 == pytest.approx(100.0)
    assert context.l_ref_m == pytest.approx(10.0)
    assert context.has_real_channel
    assert context.proxy_channel is False
    assert context.drainage is not None
    assert context.drainage.total_bounds.tolist() == pytest.approx([0, 5, 10, 5])


def test_channel_context_explicitly_coreprojects_aoi_and_drainage() -> None:
    aoi = gpd.GeoDataFrame(
        geometry=[box(123.95, -18.15, 124.0, -18.10)], crs="EPSG:4326"
    )
    drainage = _drainage(
        LineString([(123.94, -18.125), (124.01, -18.125)]), crs="EPSG:4326"
    )

    context = create_channel_context(
        "reach-2",
        aoi,
        drainage,
        drainage_id="synthetic-geographic-v1",
        target_crs="EPSG:3577",
    )

    assert context.crs == "EPSG:3577"
    assert context.area_m2 > 0
    assert context.l_ref_m is not None and context.l_ref_m > 0
    assert str(context.drainage.crs) == "EPSG:3577"


def test_channel_context_rejects_projected_but_non_equal_area_crs() -> None:
    aoi = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)], crs="EPSG:3577")
    drainage = _drainage(LineString([(0, 5), (10, 5)]))
    with pytest.raises(ValueError, match="equal-area"):
        create_channel_context(
            "reach-web-mercator",
            aoi,
            drainage,
            drainage_id="synthetic-v1",
            target_crs="EPSG:3857",
        )


def test_real_fitzroy_drainage_matches_approved_contract() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "fitzroy_kimberley_drainage.gpkg"
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    drainage = gpd.read_file(path)
    topology = validate_drainage_topology(drainage)

    assert digest == "004442d0a65a7eeb51a335dbaa621e281f610080b31e7ae05ee9980a46dc3b3a"
    assert topology.feature_count == 291
    assert topology.crs == "EPSG:3577"
    assert topology.geometry_types == ("MultiLineString",)
    assert topology.terminal_reaches == 3
