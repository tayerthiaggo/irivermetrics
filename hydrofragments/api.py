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
from hydrofragments.compute import resolve_execution_plan
from hydrofragments.guards import ComparisonGuardError, guard_comparison
from hydrofragments.io.adapters import parse_watermask_tsfill
from hydrofragments.io.alignment import validate_alignment
from hydrofragments.metrics import (
    ApsecRecord,
    MetricPlan,
    compute_extent_contraction,
    compute_hydroperiod,
    compute_inter_pool_gaps,
    compute_lpsec,
    compute_recurrence,
    resolve_metrics,
)
from hydrofragments.models import (
    AnalysisInputs,
    HydroResult,
    MetricRecord,
    ValidationReport,
    WaterCube,
)
from hydrofragments.output import records_to_frame, write_run_metadata
from hydrofragments.schema import (
    EdgeFlag,
    MetricDependency,
    MetricFamily,
    SCHEMA_VERSION,
    Statistic,
    ValueType,
    WarningFlag,
)
from hydrofragments.spatial import SpatialContext
from hydrofragments.temporal import detect_cadence, detect_hy_anchors


def _describe_chunks(array: xr.DataArray) -> str:
    """Serializable description of an array's actual Dask chunk sizes.

    Returns ``"none"`` for a non-Dask-backed (eager numpy) array, otherwise
    a compact ``"dim=size,dim=size,..."`` string using each dimension's
    first block size -- enough to verify what chunking was actually applied
    without embedding full nested chunk tuples in provenance/manifest text.
    """
    chunks = array.chunks
    if chunks is None:
        return "none"
    return ",".join(
        f"{dim}={sizes[0]}" for dim, sizes in zip(array.dims, chunks)
    )


def _coerce_dataarray(source: xr.DataArray | xr.Dataset) -> xr.DataArray:
    if isinstance(source, xr.Dataset):
        if "water" in source:
            return source["water"]
        if len(source.data_vars) == 1:
            return next(iter(source.data_vars.values()))
        raise ValueError("Dataset must expose a single water variable or 'water'")
    return source


_WATER_VALIDITY_ERROR = "water=True requires valid_obs=True for every pixel/month"


def _can_check_water_validity_eager(
    water: xr.DataArray, valid_obs: xr.DataArray
) -> bool:
    return water.chunks is None and valid_obs.chunks is None


def _has_water_without_valid_obs(water: xr.DataArray, valid_obs: xr.DataArray) -> bool:
    if not _can_check_water_validity_eager(water, valid_obs):
        return False
    invalid_water = water.astype(bool) & ~valid_obs.astype(bool)
    return bool(np.any(np.asarray(invalid_water.values, dtype=bool)))


def _ensure_water_implies_valid_obs(
    water: xr.DataArray, valid_obs: xr.DataArray
) -> None:
    if _has_water_without_valid_obs(water, valid_obs):
        raise ValueError(_WATER_VALIDITY_ERROR)


def open_water_cube(
    source: xr.DataArray | xr.Dataset | str | Path,
    *,
    valid_obs: xr.DataArray | None = None,
    variable_map: Mapping[str, str] | None = None,
    chunks: Mapping[str, int] | None = None,
    input_kind: str = "generic_binary",
    aoi_mask: xr.DataArray | None = None,
    analysis_mask: xr.DataArray | None = None,
) -> WaterCube:
    """Open a canonical aligned water/valid cube from supported sources.

    ``aoi_mask``/``analysis_mask`` are optional aligned 2-D boolean masks
    passed straight through to :class:`hydrofragments.models.WaterCube`
    (which validates alignment and defaults either to all-true when
    omitted -- see its docstring). Every caller that does not pass them
    keeps today's unpruned full-grid behaviour unchanged.
    """
    del variable_map  # reserved for later adapter expansion

    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".zarr" or path.name.endswith(".zarr"):
            dataset = xr.open_zarr(path, chunks=chunks if chunks is not None else "auto")
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
                provenance=(
                    ("adapter", "watermask_tsfill"),
                    ("chunks", _describe_chunks(water)),
                ),
                aoi_mask=aoi_mask,
                analysis_mask=analysis_mask,
            )
        raise ValueError(f"unsupported source path: {path}")

    array = _coerce_dataarray(source)
    if chunks is not None:
        array = array.chunk(chunks)
    if input_kind == "watermask_tsfill":
        water, valid = parse_watermask_tsfill(array)
    else:
        water = (array == 1).astype(bool)
        if valid_obs is None:
            valid = xr.ones_like(water, dtype=bool)
        else:
            valid = valid_obs.astype(bool)
            if chunks is not None:
                valid = valid.chunk(chunks)
            validate_alignment(water, valid)
            _ensure_water_implies_valid_obs(water, valid)
    cadence = detect_cadence(water)
    crs = water.rio.crs.to_string() if hasattr(water, "rio") and water.rio.crs else None
    return WaterCube(
        water=water,
        valid_obs=valid,
        source=input_kind,
        cadence=cadence,
        crs=crs,
        provenance=(
            ("input_kind", input_kind),
            ("chunks", _describe_chunks(water)),
        ),
        aoi_mask=aoi_mask,
        analysis_mask=analysis_mask,
    )


def validate_inputs(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    hydroyear_available: bool = False,
    dual_composites_available: bool = False,
    wet_any_month: Mapping[str, bool] | None = None,
) -> ValidationReport:
    """Validate contracts without running metric kernels."""
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
    else:
        if _has_water_without_valid_obs(cube.water, cube.valid_obs):
            errors.append(_WATER_VALIDITY_ERROR)

    available = {MetricDependency.VALIDITY}
    if config.patches.min_patch_pixels > 0:
        available.add(MetricDependency.PATCHES)
    if isinstance(drainage, SpatialContext) and drainage.has_real_channel:
        available.add(MetricDependency.CHANNEL)
    if config.patches.width_resolution_floor_pixels is not None:
        available.add(MetricDependency.WIDTH_FLOOR)
    if hydroyear_available:
        available.add(MetricDependency.HY_ANCHOR)
    if dual_composites_available:
        available.add(MetricDependency.DUAL_COMPOSITE)
    if (
        isinstance(drainage, SpatialContext)
        and drainage.has_real_channel
        and wet_any_month is not None
        and any(wet_any_month.values())
    ):
        available.add(MetricDependency.FIXED_NODES)
        available.add(MetricDependency.GRAPH)
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
    zone: str = "AOI",
    statistic: Statistic | None = None,
    metric_dependency: MetricDependency = MetricDependency.NONE,
    warning_flags: tuple[WarningFlag, ...] = (WarningFlag.LENGTH_CRS_CAVEAT,),
    n_valid_pixels: int | None = None,
    n_water_pixels: int | None = None,
    valid_fraction_month: float | None = None,
    min_valid_fraction_month: float | None = None,
    edge_flag: EdgeFlag | None = None,
    low_coverage_flag: bool | None = None,
    is_reportable: bool | None = None,
) -> MetricRecord:
    reportable = value is not None and np.isfinite(value)
    if is_reportable is not None:
        reportable = is_reportable
    return MetricRecord(
        run_id=run_id,
        config_hash=config.config_hash,
        package_version=__version__,
        git_sha="unknown",
        catchment_id=catchment_id,
        aoi_id=aoi_id,
        zone=zone,
        date=timestamp,
        metric=metric,
        metric_family=metric_family,
        statistic=statistic,
        value=value,
        unit=unit,
        value_type=value_type,
        n_pools=n_pools,
        n_valid_pixels=n_valid_pixels,
        n_water_pixels=n_water_pixels,
        valid_fraction_month=valid_fraction_month,
        min_valid_fraction_month=min_valid_fraction_month,
        edge_flag=edge_flag,
        low_coverage_flag=low_coverage_flag,
        warning_flags=warning_flags,
        is_reportable=reportable,
        source=source,
        resolution_m=resolution_m,
        crs=crs,
        area_unit="m2",
        length_unit="m",
        monthly_composite=config.temporal.monthly_composite,
        min_patch_pixels=config.patches.min_patch_pixels,
        min_patch_area_m2=float(resolution_m) ** 2 * config.patches.min_patch_pixels,
        connectivity_rule=config.patches.connectivity_rule,
        metric_dependency=metric_dependency,
    )


def _channel_profile_records(
    *,
    cube: WaterCube,
    context: SpatialContext,
    wet_profiles: Sequence[Sequence[bool]],
    segment_lengths_m: Sequence[float],
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
) -> list[MetricRecord]:
    wet = np.asarray(wet_profiles, dtype=bool)
    lengths = np.asarray(segment_lengths_m, dtype=float)
    if wet.ndim != 2:
        raise ValueError("channel_wet_profiles must have shape (time, segment)")
    if lengths.ndim != 1 or wet.shape[1] != lengths.size:
        raise ValueError("channel profile and segment lengths must align")
    if wet.shape[0] != cube.water.sizes.get("time", 0):
        raise ValueError("channel profile time axis must align with water cube")
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("channel_segment_lengths_m must be positive and finite")

    records: list[MetricRecord] = []
    times = pd.to_datetime(cube.water["time"].values)
    for timestamp, states in zip(times, wet):
        wetted_length_m = float(lengths[states].sum())
        lpsec = compute_lpsec(wetted_length_m, context=context)
        records.append(
            _metric_record(
                run_id=run_id,
                config=config,
                catchment_id=catchment_id,
                aoi_id=aoi_id,
                metric="lpsec",
                metric_family=MetricFamily.EXTENT,
                value=lpsec.value,
                unit="percent",
                value_type=ValueType.MONTHLY,
                source=source,
                timestamp=timestamp.to_pydatetime(),
                resolution_m=resolution_m,
                crs=crs,
                zone="1",
                metric_dependency=MetricDependency.CHANNEL,
            )
        )
        gaps = compute_inter_pool_gaps(states, segment_lengths_m=lengths)
        for statistic, value_m in (
            (Statistic.MEAN, gaps.mean_m),
            (Statistic.MEDIAN, gaps.median_m),
            (Statistic.MAX, gaps.max_m),
            (Statistic.CV, gaps.cv),
        ):
            if not np.isfinite(value_m):
                continue
            value = float(value_m if statistic is Statistic.CV else value_m / 1000.0)
            records.append(
                _metric_record(
                    run_id=run_id,
                    config=config,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    metric="inter_pool_gap",
                    metric_family=MetricFamily.CLUSTERING,
                    statistic=statistic,
                    value=value,
                    unit="dimensionless" if statistic is Statistic.CV else "km",
                    value_type=ValueType.MONTHLY,
                    source=source,
                    timestamp=timestamp.to_pydatetime(),
                    resolution_m=resolution_m,
                    crs=crs,
                    zone="1",
                    metric_dependency=MetricDependency.CHANNEL,
                )
            )
    return records


# Metric ids that can ever originate from section_compat_rows() /
# _records_from_compat_rows() below (the legacy wide-row bridge). Kept in
# sync with the "mapping" table inside _records_from_compat_rows -- used to
# skip the whole compat-row compute path when a narrow profile selects none
# of these (B1).
_COMPAT_ROW_METRIC_IDS = frozenset(
    {
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
        "pool_width",
        "occurrence",
        "refuge_area",
    }
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
        "APSEC": (MetricFamily.EXTENT, "apsec", "percent", ValueType.MONTHLY, None),
        "n_patches": (
            MetricFamily.FRAGMENTATION,
            "number_of_pools",
            "count",
            ValueType.MONTHLY,
            None,
        ),
        "LPI": (
            MetricFamily.FRAGMENTATION,
            "lpi",
            "percent",
            ValueType.MONTHLY,
            None,
        ),
        "AWRe": (
            MetricFamily.MORPHOLOGY,
            "awre",
            "dimensionless",
            ValueType.MONTHLY,
            None,
        ),
        "AWMSI": (
            MetricFamily.MORPHOLOGY,
            "awmsi",
            "dimensionless",
            ValueType.MONTHLY,
            None,
        ),
        "pool_width_mean": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MEAN,
        ),
        "pool_width_median": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MEDIAN,
        ),
        "pool_width_max": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "m",
            ValueType.MONTHLY,
            Statistic.MAX,
        ),
        "pool_width_cv": (
            MetricFamily.MORPHOLOGY,
            "pool_width",
            "dimensionless",
            ValueType.MONTHLY,
            Statistic.CV,
        ),
        "pp_mean_%": (
            MetricFamily.PERSISTENCE,
            "occurrence",
            "percent",
            ValueType.RASTER_SUMMARY,
            None,
        ),
        "ra_area_km2": (
            MetricFamily.PERSISTENCE,
            "refuge_area",
            "km2",
            ValueType.RASTER_SUMMARY,
            None,
        ),
    }
    records: list[MetricRecord] = []
    for row in rows:
        timestamp = pd.Timestamp(row["date"]).to_pydatetime()
        for column, (family, metric_id, unit, value_type, statistic) in mapping.items():
            value = row.get(column)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                numeric_value = None
            else:
                numeric_value = float(value)
            if metric_id == "pool_width" and (
                numeric_value is None or not np.isfinite(numeric_value)
            ):
                continue
            monthly_metric = value_type is ValueType.MONTHLY
            low_coverage = monthly_metric and bool(row.get("low_coverage_flag", False))
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
                    statistic=statistic,
                    timestamp=timestamp,
                    resolution_m=resolution_m,
                    crs=crs,
                    source=source,
                    n_pools=(
                        int(row["n_patches"])
                        if metric_id == "number_of_pools" and row.get("n_patches") is not None
                        else None
                    ),
                    n_valid_pixels=(
                        int(row["n_valid_pixels"])
                        if monthly_metric and row.get("n_valid_pixels") is not None
                        else None
                    ),
                    n_water_pixels=(
                        int(row["APSEC_n_water_pixels"])
                        if metric_id == "apsec"
                        and row.get("APSEC_n_water_pixels") is not None
                        else None
                    ),
                    valid_fraction_month=(
                        float(row["valid_fraction_month"])
                        if monthly_metric
                        and row.get("valid_fraction_month") is not None
                        else None
                    ),
                    min_valid_fraction_month=(
                        float(row["min_valid_fraction_month"])
                        if monthly_metric
                        and row.get("min_valid_fraction_month") is not None
                        else None
                    ),
                    edge_flag=(
                        EdgeFlag.LOW_VALID_OBS if low_coverage else None
                    ),
                    low_coverage_flag=low_coverage if monthly_metric else None,
                    warning_flags=(
                        row["pool_width_warning_flags"]
                        if metric_id == "pool_width"
                        and row.get("pool_width_warning_flags") is not None
                        else (WarningFlag.LENGTH_CRS_CAVEAT,)
                    ),
                    metric_dependency=(
                        MetricDependency.WIDTH_FLOOR
                        if metric_id == "pool_width"
                        else MetricDependency.NONE
                    ),
                    is_reportable=False if low_coverage else None,
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
    hydroperiod = compute_hydroperiod(monthly, config=config).hydroperiod
    years = [int(year) for year in hydroperiod.coords["year"].values]

    # Batch the AOI-mean recurrence scalar and every per-year AOI-mean
    # hydroperiod scalar into a single Dataset so they share one Dask graph
    # execution instead of one independent `.item()` materialization each
    # (m8: materialization count must not scale with the number of years).
    summary_vars: dict[str, xr.DataArray] = {
        "recurrence": recurrence.recurrence.mean(skipna=True),
    }
    for year in years:
        summary_vars[f"hydroperiod_{year}"] = (
            hydroperiod.sel(year=year).mean(skipna=True).drop_vars("year")
        )
    summary_ds = xr.Dataset(summary_vars).compute()

    recurrence_value = summary_ds["recurrence"].item()
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

    for year in years:
        value = summary_ds[f"hydroperiod_{year}"].item()
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
    inputs: AnalysisInputs | None = None,
    pixel_size_m: float = 30.0,
    catchment_id: str | None = None,
) -> HydroResult:
    """Execute configured metric profiles for one AOI.

    ``inputs`` bundles optional advanced inputs (drainage context, hydroyear
    extent, dual-composite APSEC, channel profiles) -- see
    :class:`hydrofragments.models.AnalysisInputs`. ``inputs.hydroyear_extent``
    enables the external hydroseason adapter. Dynamics additionally requires
    both ``inputs.max_water_apsec`` and ``inputs.median_apsec``; absent
    either composite, the registry reports an explicit dependency skip.
    """
    inputs = inputs or AnalysisInputs()
    drainage = inputs.drainage
    hydroyear_extent = inputs.hydroyear_extent
    max_water_apsec = inputs.max_water_apsec
    median_apsec = inputs.median_apsec
    channel_wet_profiles = inputs.channel_wet_profiles
    channel_segment_lengths_m = inputs.channel_segment_lengths_m

    execution_plan = resolve_execution_plan(
        accelerator=config.compute.accelerator,
        cuda_strict=config.compute.cuda_strict,
    )
    report = validate_inputs(
        cube,
        aoi_id,
        config=config,
        drainage=drainage,
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
    section_area_km2 = (
        float(cube.water.isel(time=0).size) * pixel_size_m**2 / 1_000_000.0
    )
    monthly = xr.Dataset(
        {
            "water": cube.water.astype(bool),
            "valid_obs": cube.valid_obs.astype(bool),
        }
    )
    hydroyear_result = None
    if hydroyear_extent is not None:
        hydroyear_result = detect_hy_anchors(
            hydroyear_extent, hydrofragments_config=config
        )

    # Resolve which metrics are selected BEFORE computing anything (B1): the
    # compat row bridge below only computes families whose metric ids are in
    # selected_ids, so an expensive family (e.g. patch morphology) that a
    # narrow profile never asked for is never run.
    selected_ids = {
        spec.metric_id for spec in resolve_metrics(
            config.metric_profiles,
            available_dependencies=(
                MetricDependency.VALIDITY,
                MetricDependency.PATCHES,
                *(
                    (MetricDependency.CHANNEL,)
                    if isinstance(drainage, SpatialContext)
                    and drainage.has_real_channel
                    else ()
                ),
                *(
                    (MetricDependency.WIDTH_FLOOR,)
                    if config.patches.width_resolution_floor_pixels is not None
                    else ()
                ),
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

    records: list[MetricRecord] = []
    if selected_ids & _COMPAT_ROW_METRIC_IDS:
        rows = section_compat_rows(
            monthly["water"],
            section=aoi_id,
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size_m,
            config=config,
            selected_ids=selected_ids,
            valid_obs=monthly["valid_obs"],
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
    records = [record for record in records if record.metric in selected_ids]
    if {"lpsec", "inter_pool_gap"} & selected_ids:
        if not isinstance(drainage, SpatialContext) or not drainage.has_real_channel:
            raise ValueError("channel profile requires a real SpatialContext")
        if channel_wet_profiles is None or channel_segment_lengths_m is None:
            raise ValueError(
                "channel profile requires channel_wet_profiles and "
                "channel_segment_lengths_m"
            )
        records.extend(
            _channel_profile_records(
                cube=cube,
                context=drainage,
                wet_profiles=channel_wet_profiles,
                segment_lengths_m=channel_segment_lengths_m,
                run_id=run_id,
                config=config,
                catchment_id=catchment,
                aoi_id=aoi_id,
                resolution_m=pixel_size_m,
                crs=crs,
                source=cube.source,
            )
        )
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
    backend_warnings = list(report.warnings)
    if (
        config.compute.accelerator == "auto"
        and execution_plan.fallback_reason is not None
    ):
        backend_warnings.append(
            f"accelerator_fallback: {execution_plan.fallback_reason}"
        )
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
            "chunks": _describe_chunks(cube.water),
        },
        planned_backend=execution_plan.planned_backend,
        actual_backend_by_stage={
            "analyze": "cpu",
            **execution_plan.actual_backend_by_stage,
        },
        backend_capabilities=execution_plan.capabilities.to_mapping(),
        skipped_metrics=[
            {"metric_id": metric_id, "reason": reason}
            for metric_id, reason in report.skipped_metrics
        ],
        warnings=backend_warnings,
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
    "AnalysisInputs",
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
