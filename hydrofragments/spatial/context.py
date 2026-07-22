"""Fixed AOI and optional real-drainage spatial context.

Core channel lengths come only from caller-supplied line geometry clipped to
the AOI. Wet-mask skeletons are deliberately not accepted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import geopandas as gpd
import pyproj
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from geopandas import GeoDataFrame


_TOPOLOGY_COLUMNS = ("HydroID", "From_Node", "To_Node", "NextDownID")
_LINE_TYPES = frozenset({"LineString", "MultiLineString"})


class DrainageContractError(ValueError):
    """Raised when supplied drainage cannot provide a real channel contract."""


@dataclass(frozen=True)
class DrainageTopology:
    feature_count: int
    crs: str
    geometry_types: tuple[str, ...]
    terminal_reaches: int


@dataclass(frozen=True)
class SpatialContext:
    aoi_id: str
    area_m2: float
    drainage_id: str | None = None
    l_ref_m: float | None = None
    crs: str | None = None
    aoi_geometry: BaseGeometry | None = None
    drainage: "GeoDataFrame | None" = None
    proxy_channel: bool = False

    @property
    def has_real_channel(self) -> bool:
        return (
            self.drainage is not None
            and self.drainage_id is not None
            and self.l_ref_m is not None
            and self.l_ref_m > 0
            and not self.proxy_channel
        )


def _union(geometries: gpd.GeoSeries) -> BaseGeometry:
    union_all = getattr(geometries, "union_all", None)
    if union_all is not None:
        return union_all()
    return geometries.unary_union


def validate_drainage_topology(drainage: "GeoDataFrame") -> DrainageTopology:
    """Validate minimum U4/Q6 geometry, CRS, and topology fields."""
    if not isinstance(drainage, gpd.GeoDataFrame):
        raise DrainageContractError("drainage must be a GeoDataFrame")
    if drainage.empty:
        raise DrainageContractError("drainage must contain at least one feature")
    if drainage.crs is None:
        raise DrainageContractError("drainage CRS is required")

    missing = [name for name in _TOPOLOGY_COLUMNS if name not in drainage.columns]
    if missing:
        raise DrainageContractError(
            "drainage topology missing required columns: " + ", ".join(missing)
        )
    null_counts = drainage.loc[:, list(_TOPOLOGY_COLUMNS)].isna().sum()
    if int(null_counts.sum()) > 0:
        columns = ", ".join(null_counts[null_counts > 0].index.tolist())
        raise DrainageContractError(f"drainage topology contains null values: {columns}")

    geometry_types = tuple(sorted(drainage.geometry.geom_type.unique().tolist()))
    unsupported = sorted(set(geometry_types) - _LINE_TYPES)
    if unsupported:
        raise DrainageContractError(
            "drainage geometry must be LineString/MultiLineString, got "
            + ", ".join(unsupported)
        )
    if drainage.geometry.is_empty.any() or drainage.geometry.isna().any():
        raise DrainageContractError("drainage contains empty or null geometry")
    if not drainage.geometry.is_valid.all():
        raise DrainageContractError("drainage contains invalid geometry")
    if not drainage["HydroID"].is_unique:
        raise DrainageContractError("drainage HydroID values must be unique")

    rows = drainage.set_index("HydroID", drop=False)
    identifiers = set(rows.index.tolist())
    for _, row in drainage.iterrows():
        downstream = row["NextDownID"]
        if downstream not in identifiers:
            continue
        if row["To_Node"] != rows.at[downstream, "From_Node"]:
            raise DrainageContractError(
                "NextDownID topology has inconsistent To_Node/From_Node values"
            )

    visiting: set[object] = set()
    visited: set[object] = set()

    def visit(identifier: object) -> None:
        if identifier in visiting:
            raise DrainageContractError("drainage NextDownID topology contains a cycle")
        if identifier in visited:
            return
        visiting.add(identifier)
        downstream = rows.at[identifier, "NextDownID"]
        if downstream in identifiers:
            visit(downstream)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in sorted(identifiers, key=str):
        visit(identifier)

    terminal_reaches = int((drainage["NextDownID"].astype(str) == "-1").sum())
    return DrainageTopology(
        feature_count=len(drainage),
        crs=pyproj.CRS.from_user_input(drainage.crs).to_string(),
        geometry_types=geometry_types,
        terminal_reaches=terminal_reaches,
    )


def ordered_reach_paths(drainage: "GeoDataFrame") -> tuple[tuple[object, ...], ...]:
    """Return stable headwater-to-outlet HydroID paths from validated topology."""
    validate_drainage_topology(drainage)
    rows = drainage.set_index("HydroID", drop=False)
    identifiers = set(rows.index.tolist())
    internal_downstream = {
        value for value in rows["NextDownID"].tolist() if value in identifiers
    }
    heads = sorted(identifiers - internal_downstream, key=str)
    paths: list[tuple[object, ...]] = []
    for head in heads:
        path: list[object] = []
        current = head
        while current in identifiers:
            path.append(current)
            downstream = rows.at[current, "NextDownID"]
            if downstream not in identifiers:
                break
            current = downstream
        paths.append(tuple(path))
    return tuple(paths)


def create_spatial_context(aoi_id: str, area_m2: float) -> SpatialContext:
    """Create no-drainage AOI context; channel metrics remain unavailable."""
    if not aoi_id:
        raise ValueError("aoi_id must be non-empty")
    if area_m2 <= 0:
        raise ValueError("area_m2 must be positive")
    return SpatialContext(aoi_id=aoi_id, area_m2=float(area_m2))


def create_channel_context(
    aoi_id: str,
    aoi: "GeoDataFrame",
    drainage: "GeoDataFrame",
    *,
    drainage_id: str,
    target_crs: str,
) -> SpatialContext:
    """Co-project, clip, and derive fixed ``L_ref`` from real drainage lines."""
    if not aoi_id:
        raise ValueError("aoi_id must be non-empty")
    if not drainage_id:
        raise ValueError("drainage_id must be non-empty")
    if not isinstance(aoi, gpd.GeoDataFrame) or aoi.empty:
        raise ValueError("aoi must be a non-empty GeoDataFrame")
    if aoi.crs is None:
        raise ValueError("AOI CRS is required")
    validate_drainage_topology(drainage)

    resolved_crs = pyproj.CRS.from_user_input(target_crs)
    if resolved_crs.is_geographic:
        raise ValueError("target_crs must be projected for area and length metrics")
    operation = resolved_crs.coordinate_operation
    operation_text = " ".join(
        str(value)
        for value in (
            getattr(operation, "method_name", ""),
            getattr(operation, "name", ""),
        )
    ).lower()
    if "equal area" not in operation_text:
        raise ValueError("target_crs must use an equal-area projection")

    projected_aoi = aoi.to_crs(resolved_crs)
    projected_drainage = drainage.to_crs(resolved_crs)
    aoi_geometry = _union(projected_aoi.geometry)
    if aoi_geometry.is_empty or not aoi_geometry.is_valid:
        raise ValueError("AOI geometry must be non-empty and valid")

    clipped = projected_drainage.copy()
    clipped.geometry = clipped.geometry.intersection(aoi_geometry)
    clipped = clipped.loc[~clipped.geometry.is_empty & clipped.geometry.notna()].copy()
    if clipped.empty:
        raise DrainageContractError("drainage does not intersect AOI")

    l_ref_m = float(clipped.geometry.length.sum())
    if l_ref_m <= 0:
        raise DrainageContractError("clipped drainage has no positive channel length")

    return SpatialContext(
        aoi_id=aoi_id,
        area_m2=float(aoi_geometry.area),
        drainage_id=drainage_id,
        l_ref_m=l_ref_m,
        crs=resolved_crs.to_string(),
        aoi_geometry=aoi_geometry,
        drainage=clipped,
        proxy_channel=False,
    )


__all__ = [
    "DrainageContractError",
    "DrainageTopology",
    "SpatialContext",
    "create_channel_context",
    "create_spatial_context",
    "ordered_reach_paths",
    "validate_drainage_topology",
]
