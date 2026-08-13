"""Spatial grid contract tests (task 4)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from affine import Affine
from rasterio.transform import array_bounds, from_bounds

pytest.importorskip("rioxarray")

from hydrofragments.output.spatial import SpatialGrid


def _grid_array(
    *,
    y: np.ndarray,
    x: np.ndarray,
    crs: str | None = "EPSG:3577",
    values: np.ndarray | None = None,
) -> xr.DataArray:
    shape = (len(y), len(x))
    data = values if values is not None else np.zeros(shape, dtype=float)
    da = xr.DataArray(data, dims=("y", "x"), coords={"y": y, "x": x})
    if crs is not None:
        da = da.rio.write_crs(crs)
    return da


def test_ascending_coordinates_produce_expected_affine() -> None:
    y = np.array([40.0, 70.0, 100.0])
    x = np.array([10.0, 40.0, 70.0])
    grid = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))

    assert grid.transform == Affine(30.0, 0.0, -5.0, 0.0, 30.0, 25.0)
    assert grid.height == 3
    assert grid.width == 3


def test_descending_coordinates_produce_expected_affine() -> None:
    y = np.array([100.0, 70.0, 40.0])
    x = np.array([10.0, 40.0, 70.0])
    grid = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))

    assert grid.transform == Affine(30.0, 0.0, -5.0, 0.0, -30.0, 115.0)


def test_source_crs_is_preserved() -> None:
    grid = SpatialGrid.from_dataarray(
        _grid_array(
            y=np.array([100.0, 70.0]),
            x=np.array([10.0, 40.0]),
            crs="EPSG:4326",
        )
    )

    assert "4326" in grid.crs.to_wkt()


def test_wkt_attr_resolves_crs_without_rio_spatial_ref() -> None:
    y = np.array([100.0, 70.0])
    x = np.array([10.0, 40.0])
    source = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))
    da = _grid_array(y=y, x=x, crs=None)
    da.attrs["crs"] = source.crs.to_wkt()
    grid = SpatialGrid.from_dataarray(da, require_georeference=True)
    assert grid is not None
    assert grid.crs == source.crs


def test_missing_crs_fails_when_spatial_output_requested() -> None:
    da = _grid_array(
        y=np.array([100.0, 70.0]),
        x=np.array([10.0, 40.0]),
        crs=None,
    )

    with pytest.raises(ValueError, match="CRS"):
        SpatialGrid.from_dataarray(da, require_georeference=True)


def test_missing_crs_allowed_when_spatial_output_not_requested() -> None:
    da = _grid_array(
        y=np.array([100.0, 70.0]),
        x=np.array([10.0, 40.0]),
        crs=None,
    )

    assert SpatialGrid.from_dataarray(da, require_georeference=False) is None


def test_non_regular_coordinates_fail_when_spatial_output_requested() -> None:
    da = _grid_array(
        y=np.array([100.0, 70.0, 30.0]),
        x=np.array([10.0, 45.0, 70.0]),
    )

    with pytest.raises(ValueError, match="regular"):
        SpatialGrid.from_dataarray(da, require_georeference=True)


def test_non_regular_coordinates_allowed_when_spatial_output_not_requested() -> None:
    da = _grid_array(
        y=np.array([100.0, 70.0, 30.0]),
        x=np.array([10.0, 45.0, 70.0]),
    )

    assert SpatialGrid.from_dataarray(da, require_georeference=False) is None


def test_equal_shaped_shifted_transform_fails_alignment() -> None:
    y = np.array([100.0, 70.0])
    x = np.array([10.0, 40.0])
    grid = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))
    shifted = _grid_array(y=y, x=np.array([40.0, 70.0]))

    with pytest.raises(ValueError, match="transform|coordinate|align"):
        grid.validate_dataarray(shifted)


def test_swapped_dimensions_fail_alignment() -> None:
    y = np.array([100.0, 70.0])
    x = np.array([10.0, 40.0])
    grid = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))
    swapped = xr.DataArray(
        np.zeros((2, 2)),
        dims=("x", "y"),
        coords={"x": x, "y": y},
    ).rio.write_crs("EPSG:3577")

    with pytest.raises(ValueError, match="dimension|dims|ordered|align"):
        grid.validate_dataarray(swapped)


def test_reversed_coordinates_fail_alignment() -> None:
    y = np.array([100.0, 70.0])
    x = np.array([10.0, 40.0])
    grid = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))
    reversed_x = _grid_array(y=y, x=np.array([40.0, 10.0]))

    with pytest.raises(ValueError, match="coordinate|align"):
        grid.validate_dataarray(reversed_x)


def test_rasterio_metadata_round_trip_matches_source_contract() -> None:
    y = np.array([100.0, 70.0, 40.0])
    x = np.array([10.0, 40.0, 70.0])
    source = SpatialGrid.from_dataarray(_grid_array(y=y, x=x))

    west, south, east, north = array_bounds(
        source.height, source.width, source.transform
    )
    round_trip_transform = from_bounds(
        west, south, east, north, width=source.width, height=source.height
    )

    assert round_trip_transform == source.transform
    assert "3577" in source.crs.to_wkt()
