"""Checkpoint-only vector export for monthly pool polygons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyogrio
from affine import Affine
from rasterio.features import shapes as rasterio_shapes
from shapely.geometry import shape as shapely_shape

from hydrofragments.output.spatial import SpatialGrid

if TYPE_CHECKING:
    from hydrofragments.patches.morphology import PatchProperties

SPATIAL_POOL_ALGORITHM_VERSION = "1.0.0"
MONTHLY_POOLS_LAYER = "monthly_pools"
GPKG_BATCH_TARGET_BYTES = 64 * 1024 * 1024
SPATIAL_GPKG_NAME = "spatial.gpkg"

POOL_VECTOR_COLUMNS: tuple[str, ...] = (
    "date",
    "pool_id",
    "label_id",
    "n_pixels",
    "area_m2",
    "perimeter_m",
    "major_axis_length_m",
    "width_m",
    "elongation_ratio",
    "shape_index",
    "geometry",
)

POOL_VECTOR_PYARROW_SCHEMA = pa.schema(
    [
        ("date", pa.timestamp("ns")),
        ("pool_id", pa.string()),
        ("label_id", pa.int32()),
        ("n_pixels", pa.int32()),
        ("area_m2", pa.float64()),
        ("perimeter_m", pa.float64()),
        ("major_axis_length_m", pa.float64()),
        ("width_m", pa.float64()),
        ("elongation_ratio", pa.float64()),
        ("shape_index", pa.float64()),
    ]
)


class VectorExportError(ValueError):
    """Raised when vector export inputs or artifacts are invalid."""


@dataclass(frozen=True)
class PoolVectorCheckpointMetadata:
    """Durable metadata for one monthly-pool vector checkpoint."""

    grid: SpatialGrid
    scientific_config_hash: str
    algorithm_version: str
    input_fingerprint: str
    month_partitions: tuple[str, ...]
    completed: bool = False

    def to_json(self) -> str:
        payload = {
            "grid": {
                "crs": self.grid.crs.to_wkt(),
                "transform": list(self.grid.transform),
                "height": self.grid.height,
                "width": self.grid.width,
                "y_dim": self.grid.y_dim,
                "x_dim": self.grid.x_dim,
                "y": self.grid.y.tolist(),
                "x": self.grid.x.tolist(),
            },
            "scientific_config_hash": self.scientific_config_hash,
            "algorithm_version": self.algorithm_version,
            "input_fingerprint": self.input_fingerprint,
            "month_partitions": list(self.month_partitions),
            "completed": self.completed,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> PoolVectorCheckpointMetadata:
        payload = json.loads(text)
        grid_payload = payload["grid"]
        from rasterio.crs import CRS

        grid = SpatialGrid(
            crs=CRS.from_wkt(grid_payload["crs"]),
            transform=Affine(*grid_payload["transform"]),
            height=int(grid_payload["height"]),
            width=int(grid_payload["width"]),
            y_dim=str(grid_payload["y_dim"]),
            x_dim=str(grid_payload["x_dim"]),
            y=np.asarray(grid_payload["y"], dtype=float),
            x=np.asarray(grid_payload["x"], dtype=float),
        )
        return cls(
            grid=grid,
            scientific_config_hash=str(payload["scientific_config_hash"]),
            algorithm_version=str(payload["algorithm_version"]),
            input_fingerprint=str(payload["input_fingerprint"]),
            month_partitions=tuple(str(item) for item in payload["month_partitions"]),
            completed=bool(payload.get("completed", False)),
        )


@dataclass(frozen=True)
class PoolVectorCheckpoint:
    """Completed monthly-pool vector checkpoint on disk."""

    root: Path
    metadata: PoolVectorCheckpointMetadata

    def validate_complete(self) -> None:
        if not self.metadata.completed:
            raise VectorExportError("checkpoint is incomplete and cannot be exported")
        if not (self.root / "COMPLETED").exists():
            raise VectorExportError("checkpoint completion marker is missing")
        if not (self.root / "metadata.json").exists():
            raise VectorExportError("checkpoint metadata is missing")


def _month_key(date: pd.Timestamp) -> str:
    return pd.Timestamp(date).strftime("%Y-%m-%d")


def build_pool_id(*, date: pd.Timestamp, window_id: str, label_id: int) -> str:
    return f"{_month_key(date)}:{window_id}:{label_id}"


def _nullable_float(value: float) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def _elongation_ratio(area_m2: float, major_axis_length_m: float) -> float | None:
    if major_axis_length_m <= 0 or area_m2 <= 0:
        return None
    return float(2.0 * np.sqrt(area_m2 / np.pi) / major_axis_length_m)


def _shape_index(area_m2: float, perimeter_m: float) -> float | None:
    if area_m2 <= 0:
        return None
    return float(0.25 * perimeter_m / np.sqrt(area_m2))


def _window_transform(
    grid: SpatialGrid,
    *,
    row_slice: slice,
    col_slice: slice,
) -> Affine:
    return grid.transform * Affine.translation(col_slice.start, row_slice.start)


def polygonize_pool_features(
    *,
    labels: np.ndarray,
    properties: Sequence[PatchProperties],
    date: pd.Timestamp,
    window_id: str,
    grid: SpatialGrid,
    row_slice: slice,
    col_slice: slice,
) -> list[dict[str, object]]:
    """Polygonize globally filtered labels for one admitted window."""

    retained = {int(prop.label): prop for prop in properties}
    if not retained:
        return []

    concrete = np.asarray(labels, dtype=np.int32)
    transform = _window_transform(grid, row_slice=row_slice, col_slice=col_slice)
    geometries: dict[int, object] = {}
    for geom_mapping, value in rasterio_shapes(
        concrete,
        mask=concrete > 0,
        transform=transform,
    ):
        label_id = int(value)
        if label_id not in retained:
            continue
        geometries[label_id] = shapely_shape(geom_mapping)

    rows: list[dict[str, object]] = []
    for label_id, prop in sorted(retained.items()):
        geometry = geometries.get(label_id)
        if geometry is None:
            continue
        rows.append(
            {
                "date": pd.Timestamp(date).to_datetime64(),
                "pool_id": build_pool_id(
                    date=date, window_id=window_id, label_id=label_id
                ),
                "label_id": np.int32(label_id),
                "n_pixels": np.int32(prop.area_pixels),
                "area_m2": float(prop.area_m2),
                "perimeter_m": float(prop.perimeter_m),
                "major_axis_length_m": float(prop.major_axis_length_m),
                "width_m": _nullable_float(prop.width_m),
                "elongation_ratio": _elongation_ratio(
                    prop.area_m2, prop.major_axis_length_m
                ),
                "shape_index": _shape_index(prop.area_m2, prop.perimeter_m),
                "geometry": geometry,
            }
        )
    return rows


def _empty_pool_geodataframe(crs) -> gpd.GeoDataFrame:
    frame = gpd.GeoDataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "pool_id": pd.Series(dtype="string"),
            "label_id": pd.Series(dtype="int32"),
            "n_pixels": pd.Series(dtype="int32"),
            "area_m2": pd.Series(dtype="float64"),
            "perimeter_m": pd.Series(dtype="float64"),
            "major_axis_length_m": pd.Series(dtype="float64"),
            "width_m": pd.Series(dtype="float64"),
            "elongation_ratio": pd.Series(dtype="float64"),
            "shape_index": pd.Series(dtype="float64"),
        },
        geometry=gpd.GeoSeries([], crs=crs),
        crs=crs,
    )
    return frame.loc[:, list(POOL_VECTOR_COLUMNS)]


def _rows_to_geodataframe(rows: Sequence[Mapping[str, object]], *, crs) -> gpd.GeoDataFrame:
    if not rows:
        return _empty_pool_geodataframe(crs)
    frame = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    return frame.loc[:, list(POOL_VECTOR_COLUMNS)]


def validate_pool_geodataframe(frame: gpd.GeoDataFrame, *, crs) -> None:
    """Validate schema, CRS, and geometry types for one pool batch."""

    if tuple(frame.columns) != POOL_VECTOR_COLUMNS:
        raise VectorExportError("monthly pool layer schema mismatch")
    if frame.crs is None or not frame.crs.equals(crs):
        raise VectorExportError("monthly pool layer CRS mismatch")
    if len(frame) == 0:
        return
    invalid = frame.geometry.isna() | (~frame.geometry.is_valid)
    if bool(invalid.any()):
        raise VectorExportError("monthly pool layer contains invalid geometries")


def _estimated_batch_bytes(frame: gpd.GeoDataFrame) -> int:
    geometry_bytes = int(frame.geometry.memory_usage(deep=True))
    tabular_bytes = int(frame.drop(columns="geometry").memory_usage(deep=True).sum())
    return geometry_bytes + tabular_bytes


def _write_partition_parquet(
    rows: Sequence[Mapping[str, object]],
    *,
    path: Path,
    crs,
) -> None:
    frame = _rows_to_geodataframe(rows, crs=crs)
    validate_pool_geodataframe(frame, crs=crs)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, schema_version="1.0.0")


def _read_partition_parquet(path: Path, *, crs) -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(path)
    if frame.crs is None:
        frame = frame.set_crs(crs, allow_override=True)
    validate_pool_geodataframe(frame, crs=crs)
    return frame


def _iter_partition_batches(
    partition_paths: Sequence[Path],
    *,
    crs,
    batch_target_bytes: int = GPKG_BATCH_TARGET_BYTES,
) -> list[gpd.GeoDataFrame]:
    batches: list[gpd.GeoDataFrame] = []
    current = _empty_pool_geodataframe(crs)
    current_bytes = 0
    for path in partition_paths:
        frame = _read_partition_parquet(path, crs=crs)
        if frame.empty:
            continue
        frame_bytes = _estimated_batch_bytes(frame)
        if (
            len(current) > 0
            and current_bytes + frame_bytes > batch_target_bytes
        ):
            batches.append(current)
            current = _empty_pool_geodataframe(crs)
            current_bytes = 0
        current = pd.concat([current, frame], ignore_index=True)
        current = gpd.GeoDataFrame(current, geometry="geometry", crs=crs)
        current_bytes = _estimated_batch_bytes(current)
    if len(current) > 0 or not batches:
        batches.append(current)
    return batches


def _validate_written_gpkg(path: Path, *, crs, expected_features: int) -> None:
    info = pyogrio.read_info(path, layer=MONTHLY_POOLS_LAYER)
    if int(info["features"]) != expected_features:
        raise VectorExportError(
            f"monthly_pools feature count mismatch: expected {expected_features}, "
            f"got {info['features']}"
        )
    bounds = info.get("bounds")
    if bounds is not None and expected_features > 0 and bounds == (0.0, 0.0, 0.0, 0.0):
        raise VectorExportError("monthly_pools bounds are degenerate")
    frame = pyogrio.read_dataframe(path, layer=MONTHLY_POOLS_LAYER, max_features=1)
    if frame.empty:
        validate_pool_geodataframe(_empty_pool_geodataframe(crs), crs=crs)
        return
    if frame.crs is None or not frame.crs.equals(crs):
        raise VectorExportError("reopened monthly_pools CRS mismatch")
    if not frame.geometry.iloc[0].is_valid:
        raise VectorExportError("reopened monthly_pools geometry is invalid")


def export_vectors_from_checkpoint(
    checkpoint: PoolVectorCheckpoint | Path | str,
    destination: Path | str,
    *,
    pixel_size_m: float | None = None,
) -> Path:
    """Stream checkpoint partitions into ``vectors/spatial.gpkg``."""

    del pixel_size_m  # reserved for future area tolerance checks

    if isinstance(checkpoint, gpd.GeoDataFrame):
        raise VectorExportError(
            "vector export must consume a checkpoint, not an in-memory GeoDataFrame"
        )

    if isinstance(checkpoint, PoolVectorCheckpoint):
        pool_checkpoint = checkpoint
    else:
        root = Path(checkpoint)
        metadata = PoolVectorCheckpointMetadata.from_json(
            (root / "metadata.json").read_text(encoding="utf-8")
        )
        pool_checkpoint = PoolVectorCheckpoint(root=root, metadata=metadata)

    pool_checkpoint.validate_complete()
    vectors_dir = Path(destination)
    vectors_dir.mkdir(parents=True, exist_ok=True)
    final_gpkg = vectors_dir / SPATIAL_GPKG_NAME
    if final_gpkg.exists():
        raise VectorExportError(f"refusing to overwrite existing artifact: {final_gpkg}")

    tmp_gpkg = vectors_dir / f".{SPATIAL_GPKG_NAME}.tmp.gpkg"
    if tmp_gpkg.exists():
        tmp_gpkg.unlink()

    crs = pool_checkpoint.metadata.grid.crs
    ordered_paths: list[Path] = []
    for month in pool_checkpoint.metadata.month_partitions:
        month_dir = pool_checkpoint.root / "partitions" / month
        if not month_dir.exists():
            continue
        ordered_paths.extend(sorted(month_dir.glob("*.parquet")))

    batches = _iter_partition_batches(ordered_paths, crs=crs)
    expected_features = sum(len(batch) for batch in batches)

    first = True
    for batch in batches:
        validate_pool_geodataframe(batch, crs=crs)
        pyogrio.write_dataframe(
            batch,
            tmp_gpkg,
            layer=MONTHLY_POOLS_LAYER,
            driver="GPKG",
            encoding="UTF-8",
            append=not first,
        )
        first = False
        del batch

    if first:
        pyogrio.write_dataframe(
            _empty_pool_geodataframe(crs),
            tmp_gpkg,
            layer=MONTHLY_POOLS_LAYER,
            driver="GPKG",
            encoding="UTF-8",
        )

    _validate_written_gpkg(tmp_gpkg, crs=crs, expected_features=expected_features)
    tmp_gpkg.replace(final_gpkg)
    return vectors_dir


__all__ = [
    "GPKG_BATCH_TARGET_BYTES",
    "MONTHLY_POOLS_LAYER",
    "POOL_VECTOR_COLUMNS",
    "SPATIAL_GPKG_NAME",
    "SPATIAL_POOL_ALGORITHM_VERSION",
    "PoolVectorCheckpoint",
    "PoolVectorCheckpointMetadata",
    "VectorExportError",
    "build_pool_id",
    "export_vectors_from_checkpoint",
    "polygonize_pool_features",
    "validate_pool_geodataframe",
    "_empty_pool_geodataframe",
    "_write_partition_parquet",
]
