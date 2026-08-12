"""Small immutable domain records used by the v1.2 public contracts."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.config import HydroConfig, LowSupportBehavior
from hydrofragments.schema import (
    EdgeFlag,
    HydrologicalState,
    MetricDependency,
    MetricFamily,
    OUTPUT_COLUMNS,
    SCHEMA_VERSION,
    SchemaError,
    Statistic,
    ValueType,
    WarningFlag,
    validate_metric_id,
)


@dataclass(frozen=True)
class MetricRecord:
    schema_version: str = SCHEMA_VERSION
    run_id: str = ""
    config_hash: str = ""
    package_version: str = ""
    git_sha: str = ""
    catchment_id: str = ""
    aoi_id: str = ""
    zone: str | None = None
    window_id: str | None = None
    date: datetime | date | None = None
    hy: int | None = None
    hy_anchor: str | None = None
    metric: str = ""
    metric_family: MetricFamily = MetricFamily.DIAGNOSTIC
    statistic: Statistic | None = None
    value: float | None = None
    unit: str = ""
    value_type: ValueType = ValueType.DIAGNOSTIC
    state: HydrologicalState | None = None
    n_pools: int | None = None
    n_valid_pixels: int | None = None
    n_water_pixels: int | None = None
    valid_fraction_month: float | None = None
    min_valid_fraction_month: float | None = None
    low_coverage_flag: bool | None = None
    edge_flag: EdgeFlag | None = None
    warning_flags: tuple[WarningFlag, ...] = ()
    is_reportable: bool = False
    hy_confidence: str | None = None
    composite_sensitive: bool | None = None
    source: str | None = None
    resolution_m: float | None = None
    crs: str | None = None
    area_unit: str | None = None
    length_unit: str | None = None
    monthly_composite: str | None = None
    water_threshold: float | None = None
    threshold_method: str | None = None
    min_patch_pixels: int | None = None
    min_patch_area_m2: float | None = None
    connectivity_rule: int | None = None
    metric_dependency: MetricDependency = MetricDependency.NONE
    proxy_channel: bool | None = None
    awre_length_method: str | None = None
    node_source: str | None = None
    connected_wet_metric: str | None = None
    connected_wet_threshold: float | None = None
    reconnection_metric_used: str | None = None
    proxy_reconnection_flag: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaError(
                f"schema_version must be {SCHEMA_VERSION}, got {self.schema_version}"
            )
        for name in (
            "run_id",
            "config_hash",
            "package_version",
            "git_sha",
            "catchment_id",
            "aoi_id",
            "metric",
            "unit",
        ):
            if not getattr(self, name):
                raise SchemaError(f"{name} must be a non-empty string")
        validate_metric_id(self.metric)
        if self.zone not in {None, "AOI", "channel", "1", "2", "3", "4"}:
            raise SchemaError(f"unsupported zone: {self.zone}")
        if self.connectivity_rule not in {None, 4, 8}:
            raise SchemaError("connectivity_rule must be 4, 8, or null")
        for name in ("valid_fraction_month", "min_valid_fraction_month"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise SchemaError(f"{name} must be between 0 and 1")
        if tuple(item.name for item in fields(self)) != OUTPUT_COLUMNS:
            raise RuntimeError("MetricRecord fields have drifted from OUTPUT_COLUMNS")

    def with_low_valid(
        self, behavior: LowSupportBehavior | str
    ) -> "MetricRecord":
        resolved = LowSupportBehavior(behavior)
        if resolved is LowSupportBehavior.METRIC_SPECIFIC:
            raise SchemaError(
                "metric_specific low-support behavior requires a registered metric rule"
            )
        value = (
            None
            if resolved is LowSupportBehavior.SUPPRESS_VALUE
            else self.value
        )
        return replace(
            self,
            value=value,
            edge_flag=EdgeFlag.LOW_VALID_OBS,
            is_reportable=False,
        )

    def with_patch_count(self, n_pools: int) -> "MetricRecord":
        if n_pools < 0:
            raise SchemaError("n_pools cannot be negative")
        if n_pools == 0:
            return replace(
                self,
                value=None,
                n_pools=0,
                edge_flag=EdgeFlag.N0,
                state=HydrologicalState.DRY,
            )
        return replace(self, n_pools=n_pools)

    def to_mapping(self) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if field_info.name == "warning_flags":
                row[field_info.name] = [item.value for item in value]
            elif isinstance(value, Enum):
                row[field_info.name] = value.value
            else:
                row[field_info.name] = value
        return row


def _spatial_dims(water: xr.DataArray) -> tuple[str, ...]:
    return tuple(dim for dim in water.dims if dim != "time")


def _default_mask(water: xr.DataArray) -> xr.DataArray:
    """All-true 2-D mask over ``water``'s spatial grid (unpruned default)."""
    spatial_dims = _spatial_dims(water)
    spatial_coords = {
        name: coord
        for name, coord in water.coords.items()
        if set(coord.dims) <= set(spatial_dims)
    }
    shape = tuple(water.sizes[dim] for dim in spatial_dims)
    return xr.DataArray(
        np.ones(shape, dtype=bool), dims=spatial_dims, coords=spatial_coords
    )


def _validate_mask_alignment(
    mask: xr.DataArray, *, water: xr.DataArray, name: str
) -> None:
    spatial_dims = _spatial_dims(water)
    if tuple(mask.dims) != spatial_dims:
        raise ValueError(
            f"{name} must align with water's spatial dims {spatial_dims}, "
            f"got {tuple(mask.dims)}"
        )
    expected_sizes = {dim: water.sizes[dim] for dim in spatial_dims}
    if dict(mask.sizes) != expected_sizes:
        raise ValueError(
            f"{name} must align with water's spatial grid {expected_sizes}, "
            f"got {dict(mask.sizes)}"
        )


@dataclass(frozen=True)
class WaterCube:
    """Canonical aligned water/valid time series for v1.2 analysis.

    ``aoi_mask`` and ``analysis_mask`` are optional aligned 2-D boolean masks
    over the spatial grid (same dims/sizes as ``water`` minus ``time``).
    ``aoi_mask`` supplies the fixed catchment reference area that
    APSEC/LPI/reference-area denominators must use (global constraint:
    "APSEC/LPI/reference-area denominators remain full catchment
    ``aoi_mask``"). ``analysis_mask`` supplies the conservative
    potential-water footprint used as the monthly coverage denominator and
    active-processing extent (global constraint: "Monthly coverage
    denominator is approved as conservative potential-water
    ``analysis_mask``, not full catchment").

    Every existing/legacy construction path omits both fields; they then
    default to all-true over the spatial grid, so current unpruned behaviour
    is completely unchanged -- a caller who never heard of these masks gets
    the same full-catchment answer as before this feature existed.
    """

    water: xr.DataArray
    valid_obs: xr.DataArray
    source: str
    cadence: str
    crs: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()
    aoi_mask: xr.DataArray | None = None
    analysis_mask: xr.DataArray | None = None

    def __post_init__(self) -> None:
        if self.aoi_mask is None:
            object.__setattr__(self, "aoi_mask", _default_mask(self.water))
        else:
            _validate_mask_alignment(self.aoi_mask, water=self.water, name="aoi_mask")
            object.__setattr__(self, "aoi_mask", self.aoi_mask.astype(bool))
        if self.analysis_mask is None:
            object.__setattr__(self, "analysis_mask", _default_mask(self.water))
        else:
            _validate_mask_alignment(
                self.analysis_mask, water=self.water, name="analysis_mask"
            )
            object.__setattr__(self, "analysis_mask", self.analysis_mask.astype(bool))


@dataclass(frozen=True)
class ValidationReport:
    """Input validation outcome without running metric kernels."""

    is_valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    resolved_profiles: tuple[str, ...]
    skipped_metrics: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class AnalysisInputs:
    """Optional advanced inputs for :func:`hydrofragments.api.analyze`.

    Every field is independently optional; the metric registry skips any
    metric whose required dependency is absent rather than raising. Grouped
    here (instead of nine separate ``analyze`` keyword arguments) so the
    dependency relationships between fields -- e.g. ``max_water_apsec`` and
    ``median_apsec`` must both be supplied together for dynamics metrics --
    live in one place rather than being implicit across a flat call site.
    """

    drainage: Any | None = None
    hydroyear_extent: pd.Series | None = None
    max_water_apsec: Sequence[Any] | None = None
    median_apsec: Sequence[Any] | None = None
    channel_wet_profiles: Sequence[Sequence[bool]] | None = None
    channel_segment_lengths_m: Sequence[float] | None = None


def _empty_metric_coverage() -> pd.DataFrame:
    """Empty, correctly-typed default for ``HydroResult.metric_coverage``.

    Used as the ``default_factory`` so every existing construction site that
    predates this field (built before ``analyze()`` populated coverage rows)
    keeps working unchanged -- it simply gets an empty, schema-shaped frame
    instead of a required constructor argument.
    """
    return pd.DataFrame(
        columns=[
            "metric",
            "runtime_wired",
            "status",
            "rows",
            "reportable_rows",
            "reason",
        ]
    )


@dataclass(frozen=True)
class HydroResult:
    """Materialised tidy metrics plus manifest paths for one analysis run."""

    metrics_table: pd.DataFrame
    manifest: Mapping[str, object]
    output_dir: Path | None
    run_id: str
    metric_coverage: pd.DataFrame = field(default_factory=_empty_metric_coverage)

    def write(self, path: str | Path, *, formats: Sequence[str] = ("parquet",)) -> Path:
        from hydrofragments.output.tables import (
            validate_table_formats,
            write_metric_coverage,
            write_output_tables,
        )

        validated_formats = validate_table_formats(formats)
        target = Path(path)
        write_output_tables(
            self.metrics_table,
            target,
            formats=validated_formats,
            include_vectors=False,
        )
        write_metric_coverage(self.metric_coverage, target)
        return target


__all__ = ["AnalysisInputs", "HydroResult", "MetricRecord", "ValidationReport", "WaterCube"]
