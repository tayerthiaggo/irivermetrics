"""Pinning test for M4: reach_wet_any_month must be output-stable across the
rewrite from per-reach full-frame masks to a single multilabel reach raster.

Fixture note: reach R1 and R2's 60m buffers around adjacent parallel lines
(120m apart) overlap -- this is deliberate. It pins the overlapping-buffer
attribution case that a naive last-writer-wins label raster would get wrong.
"""
from __future__ import annotations

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString

from hydrofragments.spatial.connectivity_context import reach_wet_any_month


def _fixture():
    y = np.arange(10, dtype=float) * -30.0 + 300.0
    x = np.arange(10, dtype=float) * 30.0
    water = np.zeros((2, 10, 10), dtype=bool)
    water[0, 4:6, 1:8] = True  # horizontal channel wet in month 0
    da = xr.DataArray(water, dims=("time", "y", "x"), coords={"y": y, "x": x})
    drainage = gpd.GeoDataFrame(
        {"HydroID": ["R1", "R2"]},
        geometry=[LineString([(30, 150), (210, 150)]),
                  LineString([(30, 30), (210, 30)])],
        crs="EPSG:32750",
    )
    return drainage, da


def test_reach_wet_output_stable():
    drainage, da = _fixture()
    got = reach_wet_any_month(drainage, da, buffer_m=60.0)
    assert got == {"R1": True, "R2": False}


def test_reach_wet_overlap_zone_credits_both_reaches():
    """Positive-case proof for the coord -> list[reach_index] overlap fallback.

    Reuses the ``_fixture()`` geometry (R1 and R2's 60m buffers around
    parallel lines 120m apart), but places the single wet pixel at
    row 7 (y=90), inside the buffers' real overlap band (row 7,
    columns 2-7 -- confirmed by directly inspecting
    ``_build_reach_label_raster``'s output for this fixture). Unlike
    ``test_reach_wet_output_stable``, whose wet channel (rows 4:6) never
    touches the overlap band, this test puts the only wet pixel exactly
    in the shared zone, so a naive last-writer-wins raster would mark
    only one reach wet while the ``overlap_pixels`` fallback must mark
    both.
    """
    drainage, da = _fixture()
    water = np.zeros_like(da.values)
    water[0, 7, 4] = True  # single wet pixel inside the R1/R2 overlap band
    da_overlap = xr.DataArray(water, dims=da.dims, coords=da.coords)

    got = reach_wet_any_month(drainage, da_overlap, buffer_m=60.0)

    assert got == {"R1": True, "R2": True}
