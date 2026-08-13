"""Immutable spatial grid contract for raster and vector export boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

_SPATIAL_DIM_CANDIDATES = (
    ("y", "x"),
    ("lat", "lon"),
    ("latitude", "longitude"),
)


def _as_float_coords(values: xr.DataArray | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _is_regular(coords: np.ndarray) -> bool:
    if coords.size < 2:
        return False
    diffs = np.diff(coords)
    return bool(np.allclose(diffs, diffs[0]))


def _transform_from_coords(x: np.ndarray, y: np.ndarray) -> Affine:
    if x.size < 2 or y.size < 2:
        raise ValueError("spatial coordinates must contain at least two values")
    dx = (x[-1] - x[0]) / (x.size - 1)
    dy = (y[-1] - y[0]) / (y.size - 1)
    return Affine(dx, 0.0, x[0] - dx / 2.0, 0.0, dy, y[0] - dy / 2.0)


def _resolve_spatial_dims(data: xr.DataArray) -> tuple[str, str, np.ndarray, np.ndarray]:
    if data.ndim != 2:
        raise ValueError("SpatialGrid requires a 2-D DataArray")

    dims = tuple(data.dims)
    y_dim: str | None = None
    x_dim: str | None = None

    if hasattr(data, "rio"):
        try:
            y_dim, x_dim = data.rio.y_dim, data.rio.x_dim
        except (AttributeError, ValueError):
            y_dim = x_dim = None

    if y_dim is None or x_dim is None or y_dim not in dims or x_dim not in dims:
        for candidate_y, candidate_x in _SPATIAL_DIM_CANDIDATES:
            if candidate_y in dims and candidate_x in dims:
                y_dim, x_dim = candidate_y, candidate_x
                break

    if y_dim is None or x_dim is None:
        if len(dims) != 2:
            raise ValueError(f"unable to resolve spatial dimensions from {dims}")
        y_dim, x_dim = dims

    if dims.index(y_dim) >= dims.index(x_dim):
        raise ValueError(
            f"spatial dimensions must be ordered ({y_dim}, {x_dim}), got {dims}"
        )

    return y_dim, x_dim, _as_float_coords(data[y_dim]), _as_float_coords(data[x_dim])


def _parse_crs(value: object) -> CRS | None:
    if value is None:
        return None
    try:
        return CRS.from_user_input(value)
    except Exception:
        return None


def _attr_crs_candidates(mapping: Mapping[str, object] | None) -> tuple[object, ...]:
    if not mapping:
        return ()
    return tuple(
        mapping[key]
        for key in ("crs", "crs_wkt", "spatial_ref")
        if key in mapping and mapping[key] not in (None, "")
    )


def _resolve_crs(data: xr.DataArray) -> CRS | None:
    candidates: list[object] = []
    if hasattr(data, "rio"):
        try:
            rio_crs = data.rio.crs
        except Exception:
            rio_crs = None
        if rio_crs is not None:
            candidates.append(rio_crs)
    candidates.extend(_attr_crs_candidates(dict(data.attrs)))
    encoding = getattr(data, "encoding", None)
    if isinstance(encoding, dict):
        candidates.extend(_attr_crs_candidates(encoding))
    parent = getattr(data, "_parent", None)
    parent_attrs = getattr(parent, "attrs", None)
    if isinstance(parent_attrs, dict):
        candidates.extend(_attr_crs_candidates(parent_attrs))
    for candidate in candidates:
        parsed = _parse_crs(candidate)
        if parsed is not None:
            return parsed
    return None


@dataclass(frozen=True)
class SpatialGrid:
    """Canonical CRS, affine transform, and coordinate axes for one raster grid."""

    crs: CRS
    transform: Affine
    height: int
    width: int
    y_dim: str
    x_dim: str
    y: np.ndarray
    x: np.ndarray

    @classmethod
    def from_dataarray(
        cls, data: xr.DataArray, *, require_georeference: bool = True
    ) -> SpatialGrid | None:
        """Build a grid contract from ``data``'s spatial metadata.

        When ``require_georeference`` is ``False``, incomplete georeferencing
        (missing CRS or non-regular coordinates) returns ``None`` instead of
        raising so tabular analysis can proceed without spatial export.
        """
        y_dim, x_dim, y, x = _resolve_spatial_dims(data)
        if not (_is_regular(y) and _is_regular(x)):
            if require_georeference:
                raise ValueError("spatial coordinates must be regular")
            return None

        crs = _resolve_crs(data)
        if crs is None:
            if require_georeference:
                raise ValueError("spatial output requires a resolvable CRS")
            return None

        if hasattr(data, "rio"):
            try:
                transform = data.rio.transform()
            except (AttributeError, ValueError):
                transform = _transform_from_coords(x, y)
        else:
            transform = _transform_from_coords(x, y)

        height = int(data.sizes[y_dim])
        width = int(data.sizes[x_dim])
        return cls(
            crs=crs,
            transform=transform,
            height=height,
            width=width,
            y_dim=y_dim,
            x_dim=x_dim,
            y=y,
            x=x,
        )

    def validate_dataarray(self, data: xr.DataArray) -> None:
        """Raise when ``data`` does not share this grid contract."""
        y_dim, x_dim, y, x = _resolve_spatial_dims(data)
        if (y_dim, x_dim) != (self.y_dim, self.x_dim):
            raise ValueError(
                f"data dims {(y_dim, x_dim)} do not align with grid dims "
                f"{(self.y_dim, self.x_dim)}"
            )
        if (int(data.sizes[y_dim]), int(data.sizes[x_dim])) != (self.height, self.width):
            raise ValueError(
                f"data shape {(data.sizes[y_dim], data.sizes[x_dim])} does not align "
                f"with grid shape {(self.height, self.width)}"
            )
        if not np.array_equal(y, self.y):
            raise ValueError("y coordinates do not align with grid contract")
        if not np.array_equal(x, self.x):
            raise ValueError("x coordinates do not align with grid contract")

        other = self.from_dataarray(data, require_georeference=True)
        if other is None:
            raise ValueError("data lacks a resolvable spatial grid contract")
        if other.crs != self.crs:
            raise ValueError("CRS does not align with grid contract")
        if other.transform != self.transform:
            raise ValueError("transform does not align with grid contract")


__all__ = ["SpatialGrid"]
