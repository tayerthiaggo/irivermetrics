"""Deterministic channel-length and regular-grid spatial windows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from shapely.geometry import LineString, MultiLineString, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import substring

from hydrofragments.spatial.context import ordered_reach_paths

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


@dataclass(frozen=True)
class SpatialWindow:
    window_id: str
    geometry: BaseGeometry


def _line_parts(channel: BaseGeometry) -> tuple[LineString, ...]:
    if isinstance(channel, LineString):
        return (channel,)
    if isinstance(channel, MultiLineString):
        return tuple(channel.geoms)
    raise ValueError("channel must be LineString or MultiLineString")


def create_channel_windows(
    channel: BaseGeometry, *, length_m: float
) -> tuple[SpatialWindow, ...]:
    """Split supplied ordered line parts into fixed-length windows."""
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    windows: list[SpatialWindow] = []
    for part in _line_parts(channel):
        if part.is_empty or part.length <= 0:
            continue
        start = 0.0
        while start < part.length:
            stop = min(start + length_m, part.length)
            geometry = substring(part, start, stop)
            windows.append(
                SpatialWindow(
                    window_id=f"channel-{len(windows) + 1:04d}",
                    geometry=geometry,
                )
            )
            start = stop
    return tuple(windows)


def create_drainage_windows(
    drainage: "GeoDataFrame", *, length_m: float
) -> tuple[SpatialWindow, ...]:
    """Split reaches in stable topology order, independent of input row order."""
    if length_m <= 0:
        raise ValueError("length_m must be positive")
    rows = drainage.set_index("HydroID", drop=False)
    ordered_ids: list[object] = []
    seen: set[object] = set()
    for path in ordered_reach_paths(drainage):
        for identifier in path:
            if identifier not in seen:
                ordered_ids.append(identifier)
                seen.add(identifier)

    windows: list[SpatialWindow] = []
    for identifier in ordered_ids:
        geometry = rows.at[identifier, "geometry"]
        parts = sorted(_line_parts(geometry), key=lambda part: part.wkb_hex)
        sequence = 0
        for part in parts:
            start = 0.0
            while start < part.length:
                stop = min(start + length_m, part.length)
                sequence += 1
                windows.append(
                    SpatialWindow(
                        window_id=f"reach-{identifier}-{sequence:04d}",
                        geometry=substring(part, start, stop),
                    )
                )
                start = stop
    return tuple(windows)


def create_regular_grid_windows(
    aoi: BaseGeometry, *, cell_size_m: float
) -> tuple[SpatialWindow, ...]:
    """Create stable row-major grid cells clipped to a no-drainage AOI."""
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")
    if aoi.is_empty or not aoi.is_valid:
        raise ValueError("aoi must be non-empty and valid")
    minx, miny, maxx, maxy = aoi.bounds
    rows = int(math.ceil((maxy - miny) / cell_size_m))
    cols = int(math.ceil((maxx - minx) / cell_size_m))
    windows: list[SpatialWindow] = []
    for row in range(rows):
        y0 = miny + row * cell_size_m
        y1 = min(y0 + cell_size_m, maxy)
        for col in range(cols):
            x0 = minx + col * cell_size_m
            x1 = min(x0 + cell_size_m, maxx)
            geometry = box(x0, y0, x1, y1).intersection(aoi)
            if geometry.is_empty or geometry.area <= 0:
                continue
            windows.append(
                SpatialWindow(
                    window_id=f"grid-r{row + 1:04d}-c{col + 1:04d}",
                    geometry=geometry,
                )
            )
    return tuple(windows)


__all__ = [
    "SpatialWindow",
    "create_channel_windows",
    "create_drainage_windows",
    "create_regular_grid_windows",
]
