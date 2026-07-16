"""Public HydroFragments v1.2 API."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments._version import __version__
from hydrofragments.compat import section_compat_rows
from hydrofragments.config import HydroConfig
from hydrofragments.guards.comparison import ComparisonGuardError, guard_comparison
from hydrofragments.io.adapters import parse_watermask_tsfill
from hydrofragments.io.alignment import validate_alignment
from hydrofragments.metrics.registry import MetricPlan, resolve_metrics
from hydrofragments.metrics.persistence import compute_hydroperiod, compute_recurrence
from hydrofragments.metrics.dynamics import compute_extent_contraction
from hydrofragments.metrics.extent import ApsecRecord
from hydrofragments.models import HydroResult, MetricRecord, ValidationReport, WaterCube
from hydrofragments.output.manifest import write_run_metadata
from hydrofragments.output.tables import records_to_frame
from hydrofragments.schema import (
    MetricDependency,
    MetricFamily,
    SCHEMA_VERSION,
    ValueType,
    WarningFlag,
)
from hydrofragments.temporal.cadence import detect_cadence
from hydrofragments.temporal.hydroyear import detect_hy_anchors


def _coerce_dataarray(source: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(source, xr.Dataset):
        if "water" in source:
            return source["water"]
        if len(source.data_vars) == 1:
            return next(iter(source.data_vars.values()))
        raise ValueError("Dataset must expose a single water variable or 'water'")
    return source


def open_water_cube(
    source: xr.DataArray | xr.Dataset | str | Path,
    *,
    valid_obs: xr.DataArray | None = None,
    variable_map: Mapping[str, str] | None = None,
    chunks: Mapping[str, int] | None = None,
    input_kind: str = "generic_binary",
) -> WaterCube:
    """Open a canonical aligned water/valid cube from supported sources."""
    del variable_map, chunks  # reserved for later adapter expansion

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".zarr" or path.name.endswith(".zarr"):
            dataset = xr.open_zarr(path)
            raw = dataset["water_mask"] if "water_mask" in dataset else dataset["water"]
            water, valid = parse_watermask_tsfill(raw)
            cadence = detect_cadence(water)
            crs = water.rio.crs.to_string() if hasattr(water, "rio") and water.rio.crs else None
            return WaterCube(
                water=water,
                valid_obs=valid,
                source=str(path),
                cadence=cadence,
                crs=crs,
                provenance=(("adapter", "watermask_tsfill"),),
            )
        raise ValueError(f"unsupported source path: {path}")

    array = _coerce_dataarray(source)
    if input_kind == "watermask_tsfill":
        water, valid = parse_watermask_tsfill(array)
    else:
        water = (array == 1).astype(bool)
        if valid_obs is None:
            valid = xr.ones_like(water, dtype=bool)
        else:
            valid = valid_obs.astype(bool)
            validate_alignment(water, valid)
    cadence = detect_cadence(water)
    crs = water.rio.crs.to_string() if hasattr(water, "rio") and water.rio.crs else None
    return WaterCube(
        water=water,
        valid_obs=valid,
        source=input_kind,
        cadence=cadence,
        crs=crs,
        provenance=(("input_kind", input_kind),),
    )


def validate_inputs(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    hydroyear_available: bool = False,
    dual_composites_available: bool = False,
) -> ValidationReport:
    """Validate contracts without computing metrics."""
    del drainage
    errors: list[str] = []
    warnings: list[str] = []
    if cube.water.sizes != cube.valid_obs.sizes:
        errors.append("water and valid_obs must share dimensions")
    if cube.water.dims != cube.valid_obs.dims:
        errors.append("water and valid_obs must share dimension order")
    try:
        validate_alignment(cube.water, cube.valid_obs)
    except ValueError as error:
        errors.append(str(error))

    available = {MetricDependency.VALIDITY}
    if config.patches.min_patch_pixels > 0:
        available.add(MetricDependency.PATCHES)
    if hydroyear_available:
        available.add(MetricDependency.HY_ANCHOR)
    if dual_composites_available:
        available.add(MetricDependency.DUAL_COMPOSITE)
    plan = resolve_metrics(config.metric_profiles, available_dependencies=available)
    skipped = tuple((item.metric_id, item.reason) for item in plan.skipped)
    if plan.skipped:
        warnings.append("some requested metrics are unavailable with current inputs")

    return ValidationReport(
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        resolved_profiles=config.metric_profiles,
        skipped_metrics=skipped,
    )


def _metric_record(
    *,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    metric: str,
    metric_family: MetricFamily,
    value: float | None,
    unit: str,
    value_type: ValueType,
    source: str,
    timestamp: datetime | None = None,
    resolution_m: float,
    crs: str,
    n_pools: int | None = None,
) -> MetricRecord:
    return MetricRecord(
        run_id=run_id,
        config_hash=config.config_hash,
        package_version=__version__,
        git_sha="unknown",
        catchment_id=catchment_id,
        aoi_id=aoi_id,
        zone="AOI",
        date=timestamp,
        metric=metric,
        metric_family=metric_family,
        value=value,
        unit=unit,
        value_type=value_type,
        n_pools=n_pools,
        warning_flags=(WarningFlag.LENGTH_CRS_CAVEAT,),
        is_reportable=value is not None and np.isfinite(value),
        source=source,
        resolution_m=resolution_m,
        crs=crs,
        area_unit="m2",
        length_unit="m",
        monthly_composite=config.temporal.monthly_composite,
        min_patch_pixels=config.patches.min_patch_pixels,
        min_patch_area_m2=float(resolution_m) ** 2 * config.patches.min_patch_pixels,
        connectivity_rule=config.patches.connectivity_rule,
    )


def _records_from_compat_rows(
    rows: list[dict[str, object]],
    *,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
) -> list[MetricRecord]:
    mapping = {
        "APSEC": (MetricFamily.EXTENT, "apsec", "percent", ValueType.MONTHLY),
        "n_patches": (
            MetricFamily.FRAGMENTATION,
            "number_of_pools",
            "count",
            ValueType.MONTHLY,
        ),
        "LPI": (MetricFamily.FRAGMENTATION, "lpi", "percent", ValueType.MONTHLY),
        "AWRe": (MetricFamily.MORPHOLOGY, "awre", "dimensionless", ValueType.MONTHLY),
        "AWMSI": (MetricFamily.MORPHOLOGY, "awmsi", "dimensionless", ValueType.MONTHLY),
        "pp_mean_%": (
            MetricFamily.PERSISTENCE,
            "occurrence",
            "percent",
            ValueType.RASTER_SUMMARY,
        ),
        "ra_area_km2": (
            MetricFamily.PERSISTENCE,
            "refuge_area",
            "km2",
            ValueType.RASTER_SUMMARY,
        ),
    }
    records: list[MetricRecord] = []
    for row in rows:
        timestamp = pd.Timestamp(row["date"]).to_pydatetime()
        for column, (family, metric_id, unit, value_type) in mapping.items():
            value = row.get(column)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                numeric_value = None
            else:
                numeric_value = float(value)
            records.append(
                _metric_record(
                    run_id=run_id,
                    config=config,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    metric=metric_id,
                    metric_family=family,
                    value=numeric_value,
                    unit=unit,
                    value_type=value_type,
                    timestamp=timestamp,
                    resolution_m=resolution_m,
                    crs=crs,
                    source=source,
                    n_pools=int(row["n_patches"]) if metric_id == "number_of_pools" else None,
                )
            )
    return records


def _temporal_profile_records(
    monthly: xr.Dataset,
    *,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    source: str,
    resolution_m: float,
    crs: str,
) -> list[MetricRecord]:
    """Emit AOI summaries for pixel-temporal kernels through canonical schema."""
    records: list[MetricRecord] = []
    recurrence = compute_recurrence(monthly, config=config)
    recurrence_value = recurrence.recurrence.mean(skipna=True).item()
    if recurrence_value is not None and np.isfinite(recurrence_value):
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="recurrence",
                metric_family=MetricFamily.PERSISTENCE,
                value=float(recurrence_value),
                unit="percent",
                value_type=ValueType.RASTER_SUMMARY,
                source=source,
                resolution_m=resolution_m,
                crs=crs,
            )
        )

    hydroperiod = compute_hydroperiod(monthly, config=config).hydroperiod
    for year in hydroperiod.coords["year"].values:
        value = hydroperiod.sel(year=year).mean(skipna=True).item()
        if value is None or not np.isfinite(value):
            continue
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="hydroperiod",
                metric_family=MetricFamily.PERSISTENCE,
                value=float(value),
                unit="fraction",
                value_type=ValueType.RASTER_SUMMARY,
                source=source,
                timestamp=datetime(int(year), 1, 1),
                resolution_m=resolution_m,
                crs=crs,
            )
        )
    return records


def _extent_contraction_records(
    *,
    anchors: pd.DataFrame,
    max_water: Sequence[ApsecRecord],
    median: Sequence[ApsecRecord],
    config: HydroConfig,
    run_id: str,
    catchment_id: str,
    aoi_id: str,
    source: str,
    resolution_m: float,
    crs: str,
) -> list[MetricRecord]:
    records: list[MetricRecord] = []
    for anchor in anchors.to_dict(orient="records"):
        result = compute_extent_contraction(
            max_water=max_water,
            median=median,
            anchor=anchor,
            config=config,
        )
        if result is None:
            continue
        for composite, value, low_df in (
            ("max_water", result.slope_pct_per_month, result.low_df),
            ("median", result.median_slope_pct_per_month, result.median_low_df),
        ):
            records.append(
                MetricRecord(
                    run_id=run_id,
                    config_hash=config.config_hash,
                    package_version=__version__,
                    git_sha="unknown",
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    zone="AOI",
                    hy=result.hy,
                    metric="extent_contraction",
                    metric_family=MetricFamily.DYNAMICS,
                    value=None if low_df or not np.isfinite(value) else float(value),
                    unit="percent_per_month",
                    value_type=ValueType.HY_SUMMARY,
                    hy_confidence=result.hy_confidence,
                    composite_sensitive=result.composite_sensitive,
                    monthly_composite=composite,
                    metric_dependency=MetricDependency.HY_ANCHOR,
                    is_reportable=not low_df and np.isfinite(value),
                    warning_flags=(
                        (WarningFlag.COMPOSITE_SENSITIVE,)
                        if result.composite_sensitive
                        else ()
                    ),
                    source=source,
                    resolution_m=resolution_m,
                    crs=crs,
                    area_unit="m2",
                    length_unit="m",
                    min_patch_pixels=config.patches.min_patch_pixels,
                    min_patch_area_m2=resolution_m**2 * config.patches.min_patch_pixels,
                    connectivity_rule=config.patches.connectivity_rule,
                )
            )
    return records


def analyze(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    pixel_size_m: float = 30.0,
    catchment_id: str | None = None,
    hydroyear_extent: pd.Series | None = None,
    max_water_apsec: Sequence[ApsecRecord] | None = None,
    median_apsec: Sequence[ApsecRecord] | None = None,
) -> HydroResult:
    """Execute configured metric profiles for one AOI.

    ``hydroyear_extent`` enables the external hydroseason adapter. Dynamics
    additionally requires caller-supplied APSEC records for both
    ``max_water_apsec`` and ``median_apsec``; absent either composite, the
    registry reports an explicit dependency skip.
    """
    del drainage

    report = validate_inputs(
        cube,
        aoi_id,
        config=config,
        hydroyear_available=hydroyear_extent is not None,
        dual_composites_available=(
            max_water_apsec is not None and median_apsec is not None
        ),
    )
    if not report.is_valid:
        raise ValueError("; ".join(report.errors))

    run_id = uuid4().hex
    catchment = catchment_id or aoi_id
    crs = cube.crs or config.spatial.target_crs
    section_area_km2 = float(cube.water.isel(time=0).size) * pixel_size_m**2 / 1_000_000.0
    monthly = xr.Dataset({"water": cube.water.astype(bool), "valid_obs": cube.valid_obs.astype(bool)})
    rows = section_compat_rows(
        monthly["water"],
        section=aoi_id,
        section_area_km2=section_area_km2,
        pixel_size_m=pixel_size_m,
        config=config,
    )
    records = _records_from_compat_rows(
        rows,
        run_id=run_id,
        config=config,
        catchment_id=catchment,
        aoi_id=aoi_id,
        resolution_m=pixel_size_m,
        crs=crs,
        source=cube.source,
    )
    hydroyear_result = None
    if hydroyear_extent is not None:
        hydroyear_result = detect_hy_anchors(
            hydroyear_extent, hydrofragments_config=config
        )

    selected_ids = {
        spec.metric_id for spec in resolve_metrics(
            config.metric_profiles,
            available_dependencies=(
                MetricDependency.VALIDITY,
                MetricDependency.PATCHES,
                *(
                    (MetricDependency.HY_ANCHOR,)
                    if hydroyear_result is not None
                    else ()
                ),
                *(
                    (MetricDependency.DUAL_COMPOSITE,)
                    if max_water_apsec is not None and median_apsec is not None
                    else ()
                ),
            ),
        ).selected
    }
    if {"recurrence", "hydroperiod"} & selected_ids:
        records.extend(
            _temporal_profile_records(
                monthly,
                run_id=run_id,
                config=config,
                catchment_id=catchment,
                aoi_id=aoi_id,
                source=cube.source,
                resolution_m=pixel_size_m,
                crs=crs,
            )
        )
    if (
        "extent_contraction" in selected_ids
        and hydroyear_result is not None
        and max_water_apsec is not None
        and median_apsec is not None
    ):
        records.extend(
            _extent_contraction_records(
                anchors=hydroyear_result.anchors,
                max_water=max_water_apsec,
                median=median_apsec,
                config=config,
                run_id=run_id,
                catchment_id=catchment,
                aoi_id=aoi_id,
                source=cube.source,
                resolution_m=pixel_size_m,
                crs=crs,
            )
        )
    frame = records_to_frame(records)

    output_dir = Path(config.output.output_dir or ".")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_run_metadata(
        output_dir,
        config,
        run_id=run_id,
        package_version=__version__,
        git_sha="unknown",
        input_fingerprint={
            "source": cube.source,
            "cadence": cube.cadence,
            "shape": dict(cube.water.sizes),
        },
        planned_backend="cpu",
        actual_backend_by_stage={"analyze": "cpu"},
        skipped_metrics=[
            {"metric_id": metric_id, "reason": reason}
            for metric_id, reason in report.skipped_metrics
        ],
        warnings=list(report.warnings),
        comparison_context={
            "aoi_id": aoi_id,
            "source": cube.source,
            "resolution_m": pixel_size_m,
            "crs": crs,
            "hydroseason_hy_count": (
                0 if hydroyear_result is None else len(hydroyear_result.anchors)
            ),
        },
        created_at=datetime.now(timezone.utc),
    )
    return HydroResult(
        metrics_table=frame,
        manifest={"run_manifest": str(manifest.manifest_path), "package_version": __version__},
        output_dir=output_dir,
        run_id=run_id,
    )


def compare_results(
    left: Mapping[str, object],
    right: Mapping[str, object],
    *,
    overrides: Mapping[str, str] | None = None,
):
    """Refuse incompatible scientific settings unless explicitly overridden."""
    return guard_comparison(left, right, overrides=overrides)


__all__ = [
    "ComparisonGuardError",
    "HydroConfig",
    "HydroResult",
    "MetricPlan",
    "SCHEMA_VERSION",
    "ValidationReport",
    "WaterCube",
    "__version__",
    "analyze",
    "compare_results",
    "open_water_cube",
    "validate_inputs",
]
