"""m10: ``open_water_cube(..., chunks=...)`` must actually be honored.

Previously ``open_water_cube`` accepted a ``chunks`` keyword argument but
immediately discarded it (``del variable_map, chunks``), so the resulting
cube's Dask array never reflected the requested chunk layout regardless of
what a caller passed in. This test opens a small on-disk zarr store with an
explicit ``chunks`` mapping and asserts the returned cube's water array is
actually chunked that way.
"""

from __future__ import annotations

import numpy as np

from hydrofragments.api import open_water_cube


def test_open_cube_honors_chunks(tmp_zarr_path):
    cube = open_water_cube(tmp_zarr_path, chunks={"time": 1, "y": 6, "x": 6})

    chunks = cube.water.chunks
    assert chunks is not None, "expected a dask-backed (chunked) array"
    # chunks is a tuple-of-tuples, one tuple of block sizes per dimension,
    # in the DataArray's own dim order (time, y, x for this fixture).
    dim_index = {dim: i for i, dim in enumerate(cube.water.dims)}
    assert chunks[dim_index["time"]][0] == 1
    assert chunks[dim_index["y"]][0] == 6
    assert chunks[dim_index["x"]][0] == 6


def test_open_cube_default_chunks_when_unspecified(tmp_zarr_path):
    """No ``chunks`` kwarg must still work (backward compatible, no crash)."""
    cube = open_water_cube(tmp_zarr_path)
    assert cube.water.shape == (6, 12, 12)


def test_open_cube_chunks_do_not_change_values(tmp_zarr_path):
    """Chunking is purely a scheduling concern -- computed values must match.

    Opens the same on-disk cube once with an explicit custom chunking and
    once with the library default, and checks the materialized boolean water
    array is bit-for-bit identical either way.
    """
    default_cube = open_water_cube(tmp_zarr_path)
    custom_cube = open_water_cube(tmp_zarr_path, chunks={"time": 2, "y": 4, "x": 4})

    np.testing.assert_array_equal(
        default_cube.water.values, custom_cube.water.values
    )
    np.testing.assert_array_equal(
        default_cube.valid_obs.values, custom_cube.valid_obs.values
    )
