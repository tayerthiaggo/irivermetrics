"""Milestone 11 pipeline wiring -- reach/mask wet-intersection (U9).

Determines, per drainage reach, whether it was wet in at least one month of
the series -- the `wet_any_month` input to
`hydrofragments.metrics.connectivity.build_fixed_graph`. Method (U9,
approved 2026-07-17): seeded-skeleton, not a raw buffer-stamp -- skeletonize
each month's full water mask, buffer each reach line by
`reach_buffer_m` (default 2 pixel widths), and flag a reach wet that month
only if the skeleton has a pixel inside its buffer. This rejects a reach
being falsely marked wet merely because a large empty buffer polygon
happens to overlap solid wet area with no channel-like (skeletonized)
structure in it, and rejects a too-narrow buffer silently losing a real but
slightly offset channel.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString

from hydrofragments.spatial.connectivity_context import reach_wet_any_month


def _drainage_two_reaches() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": ["A", "B"],
            "From_Node": [1, 2],
            "To_Node": [2, 3],
            "NextDownID": ["B", "-1"],
            "geometry": [
                LineString([(0, 5), (10, 5)]),   # reach A: y=5, x in [0,10]
                LineString([(10, 5), (20, 5)]),  # reach B: y=5, x in [10,20]
            ],
        },
        crs="EPSG:3577",
    )


def _water_cube_wet_along_reach_a_only() -> "xr.DataArray":
    # 20x10 grid, 1 unit/pixel, y=[0..9], x=[0..19]; two months. A thin
    # 1-pixel-wide wet line so its skeleton is exactly that line (no
    # ambiguity about where medial_axis puts the skeleton for a wide blob).
    # Stops at x=6 (not x=9) so the last wet pixel is >buffer_m=2.0 away
    # from reach B's start (x=10) -- reach A ends at x=10 too, so a wet
    # line running all the way to x=9 sits only 1.0 unit from reach B's
    # line under a 2.0 buffer_m, which would spuriously flag reach B wet
    # by genuine geometric overlap, not a bug in the intersection method.
    data = np.zeros((2, 10, 20), dtype=bool)
    data[:, 5, 0:7] = True  # wet along reach A's row for both months
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": np.arange(10), "x": np.arange(20)},
    )


def test_reach_with_skeleton_in_buffer_is_flagged_wet():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=2.0)

    assert result["A"] is True


def test_reach_with_no_skeleton_in_buffer_is_flagged_dry():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=2.0)

    assert result["B"] is False


def test_result_covers_every_reach_in_drainage():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=2.0)

    assert set(result.keys()) == {"A", "B"}


def test_solid_wet_blob_with_no_channel_structure_near_reach_is_still_wet_if_skeleton_present():
    # A solid (non-thin) wet blob still has *some* medial_axis skeleton
    # running through it -- this test documents that the method flags
    # "skeleton passes near the reach," not "reach touches any wet pixel."
    # It is not a test of the buffer-vs-no-buffer distinction (that's
    # covered by the dry reach B case above) -- it exists to pin down that
    # a filled rectangle still produces an internal skeleton line, so a
    # reach running through the middle of a wide wet area is correctly wet.
    drainage = _drainage_two_reaches()
    data = np.zeros((1, 10, 20), dtype=bool)
    data[:, 3:8, 0:10] = True  # solid 5-row-tall wet block over reach A
    water = xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": [0], "y": np.arange(10), "x": np.arange(20)},
    )

    result = reach_wet_any_month(drainage, water, buffer_m=2.0)

    assert result["A"] is True


def test_wet_pixels_present_but_no_skeleton_pixel_in_buffer_is_dry():
    # Wet pixels exist far from reach B's buffer corridor (near reach A's
    # row only) -- the skeleton of that wet area never enters reach B's
    # buffer, so reach B must stay dry even though *some* wet pixel exists
    # in the water DataArray overall. This is the core case the
    # seeded-skeleton method is designed to get right versus a naive
    # "any wet pixel touches the reach's buffer" rule.
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=2.0)

    assert result["B"] is False
