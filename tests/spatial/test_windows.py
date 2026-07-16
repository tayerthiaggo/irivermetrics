from __future__ import annotations

import pytest
import geopandas as gpd
from shapely.geometry import LineString, box
from shapely.ops import unary_union

from hydrofragments.spatial.windows import (
    create_channel_windows,
    create_drainage_windows,
    create_regular_grid_windows,
)


def test_channel_windows_have_stable_ids_and_exact_run_lengths() -> None:
    channel = LineString([(0, 0), (12, 0)])

    first = create_channel_windows(channel, length_m=5.0)
    second = create_channel_windows(channel, length_m=5.0)

    assert [window.window_id for window in first] == [
        "channel-0001",
        "channel-0002",
        "channel-0003",
    ]
    assert [window.geometry.length for window in first] == pytest.approx([5, 5, 2])
    assert first == second


def test_regular_grid_windows_clip_to_aoi_without_extra_area() -> None:
    aoi = box(0, 0, 12, 7)
    windows = create_regular_grid_windows(aoi, cell_size_m=5.0)

    assert windows
    assert [window.window_id for window in windows] == sorted(
        window.window_id for window in windows
    )
    assert unary_union([window.geometry for window in windows]).area == pytest.approx(
        aoi.area
    )


def test_drainage_windows_follow_topology_and_ignore_input_row_order() -> None:
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [2, 1],
            "From_Node": [11, 10],
            "To_Node": [12, 11],
            "NextDownID": [-1, 2],
        },
        geometry=[
            LineString([(5, 0), (12, 0)]),
            LineString([(0, 0), (5, 0)]),
        ],
        crs="EPSG:3577",
    )

    first = create_drainage_windows(drainage, length_m=4.0)
    second = create_drainage_windows(drainage.iloc[::-1].copy(), length_m=4.0)

    assert [window.window_id for window in first] == [
        "reach-1-0001",
        "reach-1-0002",
        "reach-2-0001",
        "reach-2-0002",
    ]
    assert [window.geometry.length for window in first] == pytest.approx([4, 1, 4, 3])
    assert first == second
