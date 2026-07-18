"""Small immutable domain records used by the v1.2 public contracts."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

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


@dataclass(frozen=True)
class WaterCube:
    """Canonical aligned water/valid time series for v1.2 analysis."""

    water: xr.DataArray
    valid_obs: xr.DataArray
    source: str
    cadence: str
    crs: str | None = None
    provenance: tuple[tuple[str, str], ...] = ()


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


@dataclass(frozen=True)
class HydroResult:
    """Materialised tidy metrics plus manifest paths for one analysis run."""

    metrics_table: pd.DataFrame
    manifest: Mapping[str, object]
    output_dir: Path
    run_id: str

    def write(self, path: str | Path, *, formats: Sequence[str] = ("parquet",)) -> Path:
        from hydrofragments.output.tables import write_output_tables

        target = Path(path)
        rows = self.metrics_table.to_dict(orient="records")
        write_output_tables(rows, target, formats=formats)
        return target


__all__ = ["AnalysisInputs", "HydroResult", "MetricRecord", "ValidationReport", "WaterCube"]
