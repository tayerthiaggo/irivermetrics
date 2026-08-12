"""Bounded spatial raster checkpoint consumers for the monthly streaming pass."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr
import zarr

from hydrofragments.config import HydroConfig
from hydrofragments.metrics.persistence import (
    HydroperiodResult,
    OccurrenceResult,
    RecurrenceResult,
)
from hydrofragments.output.spatial import SpatialGrid

if TYPE_CHECKING:
    from hydrofragments.analysis.window_stream import WindowMonthConsumer, WindowMonthResult

SPATIAL_RASTER_ALGORITHM_VERSION = "1.0.0"
UINT32_MAX = int(np.iinfo(np.uint32).max)
UINT16_MAX = int(np.iinfo(np.uint16).max)

_PERSISTENCE_METRIC_IDS = frozenset({"occurrence", "refuge_area"})
_TEMPORAL_METRIC_IDS = frozenset({"recurrence", "hydroperiod"})
_REFUGE_STABILITY_PRODUCT = "refuge_stability_rasters"
_PERSISTENCE_PRODUCT = "persistence_rasters"
_TEMPORAL_PRODUCT = "temporal_rasters"

_PRODUCT_SPECS: dict[str, dict[str, object]] = {
    "occurrence_water_valid": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "occurrence_valid": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "valid_count_total": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "recurrence_water_valid": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "recurrence_valid": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "year_valid_months": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "hydroperiod_wet": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "hydroperiod_valid": {
        "dtype": "uint32",
        "nodata": UINT32_MAX,
        "units": "months",
    },
    "refuge_stable_count": {
        "dtype": "uint16",
        "nodata": UINT16_MAX,
        "units": "eligible_hy_pairs",
    },
    "refuge_eligible_union": {
        "dtype": "uint16",
        "nodata": UINT16_MAX,
        "units": "eligible_hy_pairs",
    },
}


class CheckpointError(ValueError):
    """Raised when a spatial checkpoint is incomplete or incompatible."""


@dataclass(frozen=True)
class CheckpointMetadata:
    """Durable metadata for one spatial raster checkpoint."""

    grid: SpatialGrid
    scientific_config_hash: str
    algorithm_version: str
    products: tuple[str, ...]
    input_fingerprint: str
    dtype_nodata_units: Mapping[str, Mapping[str, object]]
    calendar_years: tuple[int, ...] = ()
    hydrological_years: tuple[int, ...] = ()
    chunk_inventory: tuple[str, ...] = ()
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
            "products": list(self.products),
            "input_fingerprint": self.input_fingerprint,
            "dtype_nodata_units": {
                key: dict(value) for key, value in self.dtype_nodata_units.items()
            },
            "calendar_years": list(self.calendar_years),
            "hydrological_years": list(self.hydrological_years),
            "chunk_inventory": list(self.chunk_inventory),
            "completed": self.completed,
        }
        return json.dumps(payload, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> CheckpointMetadata:
        payload = json.loads(text)
        grid_payload = payload["grid"]
        from affine import Affine
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
            products=tuple(str(item) for item in payload["products"]),
            input_fingerprint=str(payload["input_fingerprint"]),
            dtype_nodata_units={
                str(key): dict(value)
                for key, value in payload["dtype_nodata_units"].items()
            },
            calendar_years=tuple(int(item) for item in payload.get("calendar_years", [])),
            hydrological_years=tuple(
                int(item) for item in payload.get("hydrological_years", [])
            ),
            chunk_inventory=tuple(str(item) for item in payload.get("chunk_inventory", [])),
            completed=bool(payload.get("completed", False)),
        )


@dataclass(frozen=True)
class SpatialRasterCheckpoint:
    """Completed spatial raster checkpoint on disk."""

    root: Path
    metadata: CheckpointMetadata

    def validate_complete(self) -> None:
        if not self.metadata.completed:
            raise CheckpointError("checkpoint is incomplete and cannot be exported")
        marker = self.root / "COMPLETED"
        if not marker.exists():
            raise CheckpointError("checkpoint completion marker is missing")
        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise CheckpointError("checkpoint metadata is missing")


def resolve_checkpoint_root(
    config: HydroConfig,
    *,
    export_enabled: bool,
) -> tuple[Path, bool]:
    """Return ``(root, durable)`` for spatial raster sidecars."""

    if config.compute.checkpoint_path:
        return Path(config.compute.checkpoint_path), True
    if export_enabled and config.output.output_dir:
        return Path(config.output.output_dir) / ".spatial_checkpoints", True
    return Path(tempfile.mkdtemp(prefix="hf_spatial_ckpt_")), False


def checkpoint_products_for_run(
    *,
    selected_metric_ids: set[str] | frozenset[str],
    spatial_products: Sequence[str],
) -> tuple[str, ...]:
    products: list[str] = []
    if selected_metric_ids & _PERSISTENCE_METRIC_IDS or _PERSISTENCE_PRODUCT in spatial_products:
        products.append(_PERSISTENCE_PRODUCT)
    if selected_metric_ids & _TEMPORAL_METRIC_IDS or _TEMPORAL_PRODUCT in spatial_products:
        products.append(_TEMPORAL_PRODUCT)
    if _REFUGE_STABILITY_PRODUCT in spatial_products:
        products.append(_REFUGE_STABILITY_PRODUCT)
    return tuple(dict.fromkeys(products))


def needs_spatial_raster_checkpoint(
    *,
    selected_metric_ids: set[str] | frozenset[str] | None,
    spatial_products: Sequence[str],
) -> bool:
    if selected_metric_ids is None:
        return True
    return bool(checkpoint_products_for_run(
        selected_metric_ids=selected_metric_ids,
        spatial_products=spatial_products,
    ))


def _grids_equal(left: SpatialGrid, right: SpatialGrid) -> bool:
    return (
        left.height == right.height
        and left.width == right.width
        and left.y_dim == right.y_dim
        and left.x_dim == right.x_dim
        and left.crs == right.crs
        and left.transform == right.transform
        and np.array_equal(left.y, right.y)
        and np.array_equal(left.x, right.x)
    )


def try_open_completed_checkpoint(
    root: Path,
    *,
    grid: SpatialGrid,
    scientific_config_hash: str,
    products: Sequence[str],
    input_fingerprint: str,
) -> SpatialRasterCheckpoint | None:
    meta_path = root / "metadata.json"
    if not meta_path.exists():
        return None
    metadata = CheckpointMetadata.from_json(meta_path.read_text(encoding="utf-8"))
    if not metadata.completed:
        return None
    if metadata.algorithm_version != SPATIAL_RASTER_ALGORITHM_VERSION:
        return None
    if metadata.scientific_config_hash != scientific_config_hash:
        return None
    if metadata.input_fingerprint != input_fingerprint:
        return None
    if tuple(metadata.products) != tuple(products):
        return None
    if not _grids_equal(metadata.grid, grid):
        raise CheckpointError("checkpoint grid does not match the current run grid")
    checkpoint = SpatialRasterCheckpoint(root=root, metadata=metadata)
    checkpoint.validate_complete()
    return checkpoint


def _safe_uint32_add(
    target: zarr.Array,
    source: np.ndarray,
    *,
    row_slice: slice,
    col_slice: slice,
    month_index: int | None = None,
) -> None:
    values = source.astype(np.uint32, copy=False)
    if month_index is None:
        current = np.asarray(target[row_slice, col_slice], dtype=np.uint64)
    else:
        current = np.asarray(target[month_index, row_slice, col_slice], dtype=np.uint64)
    updated = current + values.astype(np.uint64)
    if np.any(updated > UINT32_MAX):
        raise OverflowError("uint32 raster counter overflow")
    if month_index is None:
        target[row_slice, col_slice] = updated.astype(np.uint32)
    else:
        target[month_index, row_slice, col_slice] = updated.astype(np.uint32)


def _safe_uint16_add(
    target: zarr.Array,
    source: np.ndarray,
    *,
    row_slice: slice,
    col_slice: slice,
) -> None:
    values = source.astype(np.uint16, copy=False)
    current = np.asarray(target[row_slice, col_slice], dtype=np.uint32)
    updated = current + values.astype(np.uint32)
    if np.any(updated > UINT16_MAX):
        raise OverflowError("uint16 raster counter overflow")
    target[row_slice, col_slice] = updated.astype(np.uint16)


def _chunk_shape(grid: SpatialGrid, config: HydroConfig) -> tuple[int, int]:
    target = config.compute.target_chunk_bytes
    if target is None:
        return (min(256, grid.height), min(256, grid.width))
    bytes_per_pixel = 4
    pixels = max(1, target // bytes_per_pixel)
    side = max(16, int(np.sqrt(pixels)))
    return (min(side, grid.height), min(side, grid.width))


def _template_dataarray(grid: SpatialGrid) -> xr.DataArray:
    return xr.DataArray(
        np.zeros((grid.height, grid.width), dtype=np.float32),
        dims=(grid.y_dim, grid.x_dim),
        coords={grid.y_dim: grid.y, grid.x_dim: grid.x},
    )


def grid_from_dataarray(da: xr.DataArray) -> SpatialGrid:
    """Resolve or synthesize a grid contract from a monthly water array."""

    grid = SpatialGrid.from_dataarray(da, require_georeference=False)
    if grid is not None:
        return grid
    spatial_dims = tuple(dim for dim in da.dims if dim != "time")
    if len(spatial_dims) != 2:
        raise ValueError("expected a 2-D spatial template")
    y_dim, x_dim = spatial_dims
    from affine import Affine
    from rasterio.crs import CRS

    y = np.arange(int(da.sizes[y_dim]), dtype=float)
    x = np.arange(int(da.sizes[x_dim]), dtype=float)
    return SpatialGrid(
        crs=CRS.from_string("+proj=longlat +datum=WGS84 +no_defs"),
        transform=Affine(1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        height=int(da.sizes[y_dim]),
        width=int(da.sizes[x_dim]),
        y_dim=y_dim,
        x_dim=x_dim,
        y=y,
        x=x,
    )


@dataclass
class SpatialRasterCheckpointAccumulator:
    """Checkpoint-backed counters updated during the monthly streaming pass."""

    root: Path
    grid: SpatialGrid
    config: HydroConfig
    products: tuple[str, ...]
    input_fingerprint: str
    durable: bool
    template: xr.DataArray
    analysis_mask: np.ndarray | None = None
    end_dry_lookup: dict[tuple[int, int], Mapping[str, object]] = field(
        default_factory=dict
    )
    _store_path: Path = field(init=False)
    _group: zarr.Group = field(init=False, repr=False)
    _occurrence_months: set[int] = field(default_factory=set, init=False)
    _current_calendar_year: int | None = field(default=None, init=False)
    _completed_years: list[int] = field(default_factory=list, init=False)
    _end_dry_states: list[Any] = field(default_factory=list, init=False)
    _refuge_pairs_processed: bool = field(default=False, init=False)
    _aborted: bool = field(default=False, init=False)

    @classmethod
    def create(
        cls,
        *,
        grid: SpatialGrid,
        config: HydroConfig,
        products: Sequence[str],
        input_fingerprint: str,
        template: xr.DataArray | None = None,
        analysis_mask: np.ndarray | None = None,
        end_dry_anchors: pd.DataFrame | None = None,
        export_enabled: bool = False,
        root: Path | None = None,
    ) -> SpatialRasterCheckpointAccumulator:
        durable_root, durable = resolve_checkpoint_root(
            config, export_enabled=export_enabled
        )
        checkpoint_root = root if root is not None else durable_root / "spatial_rasters"
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        product_tuple = tuple(products)
        existing = try_open_completed_checkpoint(
            checkpoint_root,
            grid=grid,
            scientific_config_hash=config.config_hash,
            products=product_tuple,
            input_fingerprint=input_fingerprint,
        )
        if existing is not None:
            return cls._from_existing(existing, config=config, template=template)

        end_dry_lookup: dict[tuple[int, int], Mapping[str, object]] = {}
        if end_dry_anchors is not None:
            from hydrofragments.metrics.dynamics import _month_key

            for anchor in end_dry_anchors.to_dict(orient="records"):
                end_dry = anchor.get("end_dry_month")
                if end_dry is None or pd.isna(end_dry):
                    continue
                end_dry_lookup[_month_key(end_dry)] = anchor

        accumulator = cls(
            root=checkpoint_root,
            grid=grid,
            config=config,
            products=product_tuple,
            input_fingerprint=input_fingerprint,
            durable=durable,
            template=template if template is not None else _template_dataarray(grid),
            analysis_mask=analysis_mask,
            end_dry_lookup=end_dry_lookup,
        )
        accumulator._initialize_store()
        return accumulator

    @classmethod
    def _from_existing(
        cls,
        checkpoint: SpatialRasterCheckpoint,
        *,
        config: HydroConfig,
        template: xr.DataArray | None,
    ) -> SpatialRasterCheckpointAccumulator:
        accumulator = cls(
            root=checkpoint.root,
            grid=checkpoint.metadata.grid,
            config=config,
            products=checkpoint.metadata.products,
            input_fingerprint=checkpoint.metadata.input_fingerprint,
            durable=True,
            template=template if template is not None else _template_dataarray(
                checkpoint.metadata.grid
            ),
        )
        accumulator._store_path = checkpoint.root / "counters.zarr"
        accumulator._group = zarr.open_group(accumulator._store_path, mode="r")
        accumulator._completed_years = list(checkpoint.metadata.calendar_years)
        accumulator._end_dry_states = []
        accumulator._refuge_pairs_processed = True
        if "occurrence_valid" in accumulator._group:
            for month in range(1, 13):
                if np.any(np.asarray(accumulator._group["occurrence_valid"][month - 1]) > 0):
                    accumulator._occurrence_months.add(month)
        return accumulator

    def _initialize_store(self) -> None:
        self._store_path = self.root / "counters.zarr"
        if self._store_path.exists():
            raise CheckpointError(
                "refusing to overwrite an existing spatial raster checkpoint"
            )
        self._group = zarr.open_group(self._store_path, mode="w")
        chunks = _chunk_shape(self.grid, self.config)
        shape_2d = (self.grid.height, self.grid.width)
        shape_month = (12, self.grid.height, self.grid.width)

        if _PERSISTENCE_PRODUCT in self.products:
            self._group.zeros(
                "occurrence_water_valid",
                shape=shape_month,
                chunks=(1, *chunks),
                dtype=np.uint32,
            )
            self._group.zeros(
                "occurrence_valid",
                shape=shape_month,
                chunks=(1, *chunks),
                dtype=np.uint32,
            )
            self._group.zeros(
                "valid_count_total",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint32,
            )
        if _TEMPORAL_PRODUCT in self.products:
            self._group.zeros(
                "recurrence_water_valid",
                shape=shape_month,
                chunks=(1, *chunks),
                dtype=np.uint32,
            )
            self._group.zeros(
                "recurrence_valid",
                shape=shape_month,
                chunks=(1, *chunks),
                dtype=np.uint32,
            )
            self._group.zeros(
                "hydroperiod_wet",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint32,
            )
            self._group.zeros(
                "hydroperiod_valid",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint32,
            )
            self._group.zeros(
                "year_valid_months",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint32,
            )
        if _REFUGE_STABILITY_PRODUCT in self.products:
            self._group.zeros(
                "refuge_stable_count",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint16,
            )
            self._group.zeros(
                "refuge_eligible_union",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint16,
            )
            self._group.zeros(
                "end_dry_water",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint8,
            )
            self._group.zeros(
                "end_dry_valid",
                shape=shape_2d,
                chunks=chunks,
                dtype=np.uint8,
            )

        inventory = sorted(self._group.array_keys())
        metadata = CheckpointMetadata(
            grid=self.grid,
            scientific_config_hash=self.config.config_hash,
            algorithm_version=SPATIAL_RASTER_ALGORITHM_VERSION,
            products=self.products,
            input_fingerprint=self.input_fingerprint,
            dtype_nodata_units={
                key: dict(_PRODUCT_SPECS[key])
                for key in inventory
                if key in _PRODUCT_SPECS
            },
            chunk_inventory=tuple(inventory),
            completed=False,
        )
        (self.root / "metadata.json").write_text(metadata.to_json(), encoding="utf-8")

    def as_consumer(self) -> WindowMonthConsumer:
        return _RasterBlockConsumer(self)

    def add_month(
        self,
        *,
        calendar_month: int,
        calendar_year: int,
        water: np.ndarray,
        valid_obs: np.ndarray,
        timestamp: pd.Timestamp | None = None,
    ) -> None:
        row_slice = slice(0, water.shape[0])
        col_slice = slice(0, water.shape[1])
        self._update_counters(
            calendar_month=calendar_month,
            calendar_year=calendar_year,
            row_slice=row_slice,
            col_slice=col_slice,
            water=water,
            valid_obs=valid_obs,
            timestamp=timestamp,
        )

    def _update_counters(
        self,
        *,
        calendar_month: int,
        calendar_year: int,
        row_slice: slice,
        col_slice: slice,
        water: np.ndarray,
        valid_obs: np.ndarray,
        timestamp: pd.Timestamp | None,
    ) -> None:
        month_index = int(calendar_month) - 1
        water_valid = (water & valid_obs).astype(np.uint32)

        if _PERSISTENCE_PRODUCT in self.products:
            _safe_uint32_add(
                self._group["occurrence_water_valid"],
                water_valid,
                row_slice=row_slice,
                col_slice=col_slice,
                month_index=month_index,
            )
            _safe_uint32_add(
                self._group["occurrence_valid"],
                valid_obs.astype(np.uint32),
                row_slice=row_slice,
                col_slice=col_slice,
                month_index=month_index,
            )
            _safe_uint32_add(
                self._group["valid_count_total"],
                valid_obs.astype(np.uint32),
                row_slice=row_slice,
                col_slice=col_slice,
            )
            self._occurrence_months.add(int(calendar_month))

        if _TEMPORAL_PRODUCT in self.products:
            self._advance_calendar_year(calendar_year)
            _safe_uint32_add(
                self._group["recurrence_water_valid"],
                water_valid,
                row_slice=row_slice,
                col_slice=col_slice,
                month_index=month_index,
            )
            _safe_uint32_add(
                self._group["recurrence_valid"],
                valid_obs.astype(np.uint32),
                row_slice=row_slice,
                col_slice=col_slice,
                month_index=month_index,
            )
            _safe_uint32_add(
                self._group["hydroperiod_wet"],
                water_valid,
                row_slice=row_slice,
                col_slice=col_slice,
            )
            _safe_uint32_add(
                self._group["hydroperiod_valid"],
                valid_obs.astype(np.uint32),
                row_slice=row_slice,
                col_slice=col_slice,
            )
            _safe_uint32_add(
                self._group["year_valid_months"],
                valid_obs.astype(np.uint32),
                row_slice=row_slice,
                col_slice=col_slice,
            )

        if timestamp is not None and self.end_dry_lookup:
            from hydrofragments.metrics.dynamics import _month_key

            anchor = self.end_dry_lookup.get(_month_key(timestamp))
            if anchor is not None:
                self._capture_end_dry_state(
                    anchor=anchor,
                    water=water,
                    valid_obs=valid_obs,
                )

    def _advance_calendar_year(self, calendar_year: int) -> None:
        if self._current_calendar_year is None:
            self._current_calendar_year = int(calendar_year)
            return
        if int(calendar_year) == self._current_calendar_year:
            return
        self._flush_completed_calendar_year(self._current_calendar_year)
        self._current_calendar_year = int(calendar_year)
        self._group["hydroperiod_wet"][...] = 0
        self._group["hydroperiod_valid"][...] = 0
        self._group["year_valid_months"][...] = 0

    def _flush_completed_calendar_year(self, year: int) -> None:
        year_group = self.root / "hydroperiod" / str(year)
        year_group.mkdir(parents=True, exist_ok=True)
        wet = np.asarray(self._group["hydroperiod_wet"], dtype=np.uint32)
        valid = np.asarray(self._group["hydroperiod_valid"], dtype=np.uint32)
        valid_months = np.asarray(self._group["year_valid_months"], dtype=np.uint32)
        zarr.save_array(year_group / "wet.zarr", wet)
        zarr.save_array(year_group / "valid.zarr", valid)
        zarr.save_array(year_group / "valid_months.zarr", valid_months)
        self._completed_years.append(int(year))

    def _capture_end_dry_state(
        self,
        *,
        anchor: Mapping[str, object],
        water: np.ndarray,
        valid_obs: np.ndarray,
    ) -> None:
        from hydrofragments.metrics.dynamics import EndDryState

        end_dry = anchor.get("end_dry_month")
        state = EndDryState(
            hy=int(anchor["hy"]),
            date=pd.Timestamp(end_dry).to_pydatetime(),
            water=np.asarray(water, dtype=bool),
            valid_obs=np.asarray(valid_obs, dtype=bool),
            hy_confidence=str(anchor.get("confidence", "unassigned")),
            anchor_missing=end_dry is None or pd.isna(end_dry),
        )
        self._end_dry_states.append(state)
        if _REFUGE_STABILITY_PRODUCT in self.products:
            hy_path = self.root / "end_dry" / str(state.hy)
            hy_path.mkdir(parents=True, exist_ok=True)
            zarr.save_array(hy_path / "water.zarr", water.astype(np.uint8))
            zarr.save_array(hy_path / "valid.zarr", valid_obs.astype(np.uint8))

    def _process_refuge_pairs(self) -> None:
        if self._refuge_pairs_processed or _REFUGE_STABILITY_PRODUCT not in self.products:
            return
        from hydrofragments.metrics.dynamics import evaluate_refuge_spatial_stability

        states = sorted(self._end_dry_states, key=lambda item: item.hy)
        for index in range(1, len(states)):
            previous = states[index - 1]
            current = states[index]
            evaluation = evaluate_refuge_spatial_stability(
                current=current,
                previous=previous,
                analysis_mask=self.analysis_mask,
                min_valid_fraction=self.config.validity.min_valid_fraction_month,
            )
            if evaluation.edge_flag is not None:
                continue
            mask = (
                self.analysis_mask
                if self.analysis_mask is not None
                else np.ones_like(current.water, dtype=bool)
            )
            common_valid = mask & previous.valid_obs & current.valid_obs
            previous_refuge = previous.water & common_valid
            current_refuge = current.water & common_valid
            stable = (previous_refuge & current_refuge).astype(np.uint16)
            union = (previous_refuge | current_refuge).astype(np.uint16)
            _safe_uint16_add(
                self._group["refuge_stable_count"],
                stable,
                row_slice=slice(None),
                col_slice=slice(None),
            )
            _safe_uint16_add(
                self._group["refuge_eligible_union"],
                union,
                row_slice=slice(None),
                col_slice=slice(None),
            )
        self._refuge_pairs_processed = True

    def finalize_occurrence(self) -> OccurrenceResult:
        if _PERSISTENCE_PRODUCT not in self.products:
            raise CheckpointError("persistence counters were not accumulated")

        ratios: list[np.ndarray] = []
        for month in sorted(self._occurrence_months):
            month_index = month - 1
            grouped_water = np.asarray(
                self._group["occurrence_water_valid"][month_index], dtype=np.float64
            )
            grouped_valid = np.asarray(
                self._group["occurrence_valid"][month_index], dtype=np.float64
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(grouped_valid > 0, grouped_water / grouped_valid, np.nan)
            ratios.append(ratio)

        stacked = np.stack(ratios, axis=0)
        occurrence_values = np.nanmean(stacked, axis=0) * 100.0
        valid_count = np.asarray(self._group["valid_count_total"], dtype=np.int64)
        min_valid_obs = self.config.validity.min_valid_obs
        supported = valid_count >= min_valid_obs
        occurrence_values = np.where(supported, occurrence_values, np.nan)

        occurrence = self.template.copy(data=occurrence_values.astype(np.float32))
        valid_count_da = self.template.copy(data=valid_count)
        return OccurrenceResult(
            occurrence=occurrence,
            valid_count=valid_count_da,
            min_valid_obs=min_valid_obs,
        )

    def finalize_recurrence(self) -> RecurrenceResult:
        if _TEMPORAL_PRODUCT not in self.products:
            raise CheckpointError("temporal counters were not accumulated")

        ratios: list[np.ndarray] = []
        months = sorted(self._occurrence_months) if self._occurrence_months else range(1, 13)
        for month in months:
            month_index = month - 1
            grouped_water = np.asarray(
                self._group["recurrence_water_valid"][month_index], dtype=np.float64
            )
            grouped_valid = np.asarray(
                self._group["recurrence_valid"][month_index], dtype=np.float64
            )
            if not np.any(grouped_valid > 0):
                continue
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(grouped_valid > 0, grouped_water / grouped_valid, np.nan)
            ratios.append(ratio)

        stacked = np.stack(ratios, axis=0) if ratios else np.full(
            (1, self.grid.height, self.grid.width), np.nan
        )
        recurrence_values = np.nanmean(stacked, axis=0) * 100.0

        if self._current_calendar_year is not None:
            self._flush_completed_calendar_year(self._current_calendar_year)
            self._current_calendar_year = None

        year_arrays: list[np.ndarray] = []
        for year in sorted(self._completed_years):
            valid_path = self.root / "hydroperiod" / str(year) / "valid_months.zarr"
            if valid_path.exists():
                year_arrays.append(np.asarray(zarr.open(valid_path, mode="r"), dtype=np.uint32))
        if year_arrays:
            stacked_years = np.stack(year_arrays, axis=0)
            valid_year_count = (stacked_years > 0).sum(axis=0).astype(np.uint16)
        else:
            year_valid = np.asarray(self._group["year_valid_months"], dtype=np.uint32)
            valid_year_count = (year_valid > 0).astype(np.uint16)

        recurrence = self.template.copy(data=recurrence_values.astype(np.float32))
        valid_year_count_da = self.template.copy(data=valid_year_count)
        return RecurrenceResult(
            recurrence=recurrence,
            valid_year_count=valid_year_count_da,
        )

    def finalize_hydroperiod(self) -> HydroperiodResult:
        if _TEMPORAL_PRODUCT not in self.products:
            raise CheckpointError("temporal counters were not accumulated")

        years: list[int] = []
        wet_arrays: list[np.ndarray] = []
        valid_arrays: list[np.ndarray] = []

        if self._current_calendar_year is not None:
            self._flush_completed_calendar_year(self._current_calendar_year)
            self._current_calendar_year = None

        for year in sorted(self._completed_years):
            year_dir = self.root / "hydroperiod" / str(year)
            wet_arrays.append(np.asarray(zarr.open(year_dir / "wet.zarr", mode="r")))
            valid_arrays.append(np.asarray(zarr.open(year_dir / "valid.zarr", mode="r")))
            years.append(int(year))

        if not years:
            years = [pd.Timestamp.now().year]
            wet_arrays = [np.asarray(self._group["hydroperiod_wet"], dtype=np.uint32)]
            valid_arrays = [np.asarray(self._group["hydroperiod_valid"], dtype=np.uint32)]

        wet_stack = np.stack(wet_arrays, axis=0)
        valid_stack = np.stack(valid_arrays, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            hydroperiod_values = np.where(
                valid_stack > 0,
                wet_stack / valid_stack,
                np.nan,
            ).astype(np.float32)

        hydroperiod = xr.DataArray(
            hydroperiod_values,
            dims=("year", self.grid.y_dim, self.grid.x_dim),
            coords={
                "year": years,
                self.grid.y_dim: self.grid.y,
                self.grid.x_dim: self.grid.x,
            },
        )
        valid_observed = xr.DataArray(
            valid_stack.astype(np.uint8),
            dims=("year", self.grid.y_dim, self.grid.x_dim),
            coords={
                "year": years,
                self.grid.y_dim: self.grid.y,
                self.grid.x_dim: self.grid.x,
            },
        )
        return HydroperiodResult(hydroperiod=hydroperiod, valid_observed_months=valid_observed)

    def finalize_refuge_stability_rasters(self) -> xr.Dataset:
        if _REFUGE_STABILITY_PRODUCT not in self.products:
            raise CheckpointError("refuge stability counters were not accumulated")
        self._process_refuge_pairs()
        stable = np.asarray(self._group["refuge_stable_count"], dtype=np.float32)
        union = np.asarray(self._group["refuge_eligible_union"], dtype=np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            frequency = np.where(union > 0, 100.0 * stable / union, np.nan).astype(np.float32)
        return xr.Dataset(
            {
                "refuge_stability_frequency": self.template.copy(data=frequency),
                "refuge_stability_union_pair_count": self.template.copy(
                    data=union.astype(np.uint16)
                ),
            }
        )

    def finalize_checkpoint(self) -> SpatialRasterCheckpoint:
        if self._current_calendar_year is not None:
            self._flush_completed_calendar_year(self._current_calendar_year)
            self._current_calendar_year = None
        self._process_refuge_pairs()

        metadata = CheckpointMetadata(
            grid=self.grid,
            scientific_config_hash=self.config.config_hash,
            algorithm_version=SPATIAL_RASTER_ALGORITHM_VERSION,
            products=self.products,
            input_fingerprint=self.input_fingerprint,
            dtype_nodata_units={
                key: dict(_PRODUCT_SPECS[key])
                for key in self._group.array_keys()
                if key in _PRODUCT_SPECS
            },
            calendar_years=tuple(sorted(self._completed_years)),
            hydrological_years=tuple(
                sorted({state.hy for state in self._end_dry_states})
            ),
            chunk_inventory=tuple(sorted(self._group.array_keys())),
            completed=True,
        )
        (self.root / "metadata.json").write_text(metadata.to_json(), encoding="utf-8")
        (self.root / "COMPLETED").write_text("ok", encoding="utf-8")
        checkpoint = SpatialRasterCheckpoint(root=self.root, metadata=metadata)
        checkpoint.validate_complete()
        return checkpoint

    def cleanup(self) -> None:
        if self.durable or not self.root.exists():
            return
        shutil.rmtree(self.root, ignore_errors=True)

    def abort(self) -> None:
        self._aborted = True
        if not self.durable and self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)


class _RasterBlockConsumer:
    def __init__(self, accumulator: SpatialRasterCheckpointAccumulator) -> None:
        self._accumulator = accumulator

    def consume(self, block: WindowMonthResult) -> None:
        from hydrofragments.analysis.window_stream import WindowMonthResult as _WindowMonthResult

        if not isinstance(block, _WindowMonthResult):
            raise TypeError("expected WindowMonthResult")
        valid_obs = block.valid_obs
        self._accumulator._update_counters(
            calendar_month=int(block.date.month),
            calendar_year=int(block.date.year),
            row_slice=block.row_slice,
            col_slice=block.col_slice,
            water=block.water,
            valid_obs=valid_obs,
            timestamp=block.date,
        )

    def finalize(self) -> object:
        return None

    def abort(self) -> None:
        self._accumulator.abort()


__all__ = [
    "CheckpointError",
    "CheckpointMetadata",
    "SPATIAL_RASTER_ALGORITHM_VERSION",
    "SpatialRasterCheckpoint",
    "SpatialRasterCheckpointAccumulator",
    "checkpoint_products_for_run",
    "grid_from_dataarray",
    "needs_spatial_raster_checkpoint",
    "resolve_checkpoint_root",
    "try_open_completed_checkpoint",
]
