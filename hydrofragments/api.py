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
    analyze_pool_width_distribution,
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


def open_water_cube(
    source: xr.DataArray | xr.Dataset | str | Path,
    *,
    valid_obs: xr.DataArray | None = None,
    variable_map: Mapping[str, str] | None = None,
    chunks: Mapping[str, int] | None = None,
    input_kind: str = "generic_binary",
) -> WaterCube:
    """Open a canonical aligned water/valid cube from supported sources."""
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
    """Validate contracts without computing metrics."""
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
    low_coverage_flag: bool | None = None,
) -> MetricRecord:
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
        low_coverage_flag=low_coverage_flag,
        warning_flags=warning_flags,
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


def _pool_width_records_from_results(
    *,
    results: Sequence[Any | None],
    timestamps: Sequence[pd.Timestamp],
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
) -> list[MetricRecord]:
    """Build pool_width MetricRecords from already-computed PoolWidthDistributions.

    Shared by both the bundled path (``section_compat_rows()``'s
    ``width_sink``, one ``analyze_patch_bundle()`` call per month reused for
    both patch-family metrics and pool_width -- M2 follow-up) and the
    standalone path (``_pool_width_records()`` below, used when pool_width is
    selected without any patch-family metric, so there is no shared bundle
    call to piggyback on).
    """
    records: list[MetricRecord] = []
    for result, timestamp in zip(results, timestamps):
        if result is None:
            continue
        for statistic, value in (
            (Statistic.MEAN, result.mean_m),
            (Statistic.MEDIAN, result.median_m),
            (Statistic.MAX, result.max_m),
            (Statistic.CV, result.cv),
        ):
            if not np.isfinite(value):
                continue
            records.append(
                _metric_record(
                    run_id=run_id,
                    config=config,
                    catchment_id=catchment_id,
                    aoi_id=aoi_id,
                    metric="pool_width",
                    metric_family=MetricFamily.MORPHOLOGY,
                    statistic=statistic,
                    value=float(value),
                    unit="dimensionless" if statistic is Statistic.CV else "m",
                    value_type=ValueType.MONTHLY,
                    source=source,
                    timestamp=timestamp.to_pydatetime(),
                    resolution_m=resolution_m,
                    crs=crs,
                    metric_dependency=MetricDependency.WIDTH_FLOOR,
                    warning_flags=result.warning_flags,
                )
            )
    return records


def _pool_width_records(
    *,
    monthly: xr.Dataset,
    run_id: str,
    config: HydroConfig,
    catchment_id: str,
    aoi_id: str,
    resolution_m: float,
    crs: str,
    source: str,
) -> list[MetricRecord]:
    """Standalone pool_width computation: no shared patch-family bundle to reuse.

    Used only when ``pool_width`` is selected but no patch-family metric
    (``number_of_pools``/``lpi``/``awre``/``awmsi``) is, so
    ``section_compat_rows()`` never runs its per-month
    ``analyze_patch_bundle()`` loop for this call and there is nothing to
    piggyback pool_width onto.
    """
    floor = config.patches.width_resolution_floor_pixels
    if floor is None:
        raise ValueError("pool_width requires width_resolution_floor_pixels")
    timestamps = list(pd.to_datetime(monthly["time"].values))
    results = []
    for index in range(len(timestamps)):
        mask = np.asarray(
            (monthly["water"].isel(time=index) & monthly["valid_obs"].isel(time=index)).values,
            dtype=bool,
        )
        results.append(
            analyze_pool_width_distribution(
                mask,
                pixel_size_m=resolution_m,
                resolution_floor_pixels=floor,
                connectivity=config.patches.connectivity_rule,
                min_patch_pixels=config.patches.min_patch_pixels,
            )
        )
    return _pool_width_records_from_results(
        results=results,
        timestamps=timestamps,
        run_id=run_id,
        config=config,
        catchment_id=catchment_id,
        aoi_id=aoi_id,
        resolution_m=resolution_m,
        crs=crs,
        source=source,
    )


# Metric ids that can ever originate from section_compat_rows() /
# _records_from_compat_rows() below (the legacy wide-row bridge). Kept in
# sync with the "mapping" table inside _records_from_compat_rows -- used to
# skip the whole compat-row compute path when a narrow profile selects none
# of these (B1).
_COMPAT_ROW_METRIC_IDS = frozenset(
    {"apsec", "number_of_pools", "lpi", "awre", "awmsi", "occurrence", "refuge_area"}
)

# Patch-family metric ids: when any of these is selected, section_compat_rows()
# runs analyze_patch_bundle() once per month. If pool_width is *also*
# selected, that same bundle call can be asked to compute width too (via
# width_sink) instead of a second independent label/crop/measure pass -- see
# the bundle_width_floor / width_sink wiring in analyze() below (Task 1
# follow-up to M2's analyze_patch_bundle).
_PATCH_FAMILY_METRIC_IDS = frozenset({"number_of_pools", "lpi", "awre", "awmsi"})


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
                    n_pools=(
                        int(row["n_patches"])
                        if metric_id == "number_of_pools" and row.get("n_patches") is not None
                        else None
                    ),
                    low_coverage_flag=(
                        bool(row["low_coverage_flag"])
                        if metric_id == "apsec" and row.get("low_coverage_flag") is not None
                        else None
                    ),
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

    # When both a patch-family metric (number_of_pools/lpi/awre/awmsi) and
    # pool_width are selected, section_compat_rows() already runs
    # analyze_patch_bundle() once per month for the patch-family row values;
    # ask it to also compute pool_width in that same bundle call via
    # width_sink, instead of _pool_width_records() relabeling every month a
    # second time (Task 1 follow-up to M2's analyze_patch_bundle).
    want_patch_family = bool(selected_ids & _PATCH_FAMILY_METRIC_IDS)
    want_pool_width = "pool_width" in selected_ids
    bundle_width_floor = (
        config.patches.width_resolution_floor_pixels
        if want_patch_family and want_pool_width
        else None
    )
    width_sink: list[Any] | None = [] if bundle_width_floor is not None else None

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
            width_resolution_floor_pixels=bundle_width_floor,
            width_sink=width_sink,
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
    if want_pool_width:
        if width_sink is not None:
            # section_compat_rows() already computed one PoolWidthDistribution
            # per month in the same analyze_patch_bundle() call it used for
            # the patch-family rows above -- reuse those instead of relabeling.
            records.extend(
                _pool_width_records_from_results(
                    results=width_sink,
                    timestamps=list(pd.to_datetime(monthly["time"].values)),
                    run_id=run_id,
                    config=config,
                    catchment_id=catchment,
                    aoi_id=aoi_id,
                    resolution_m=pixel_size_m,
                    crs=crs,
                    source=cube.source,
                )
            )
        else:
            records.extend(
                _pool_width_records(
                    monthly=monthly,
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
