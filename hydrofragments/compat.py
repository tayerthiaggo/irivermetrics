"""Legacy compatibility helpers and dropped-metric migration errors."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
import warnings
from typing import Iterable, Iterator, Literal, Sequence

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec
import hydrofragments.metrics.patches as patch_metrics
import hydrofragments.metrics.persistence as persistence
from hydrofragments.metrics.persistence import compute_refuge_area
from hydrofragments.metrics.registry import resolve_metrics
from hydrofragments.schema import MetricDependency
from hydrofragments.spatial.active_windows import AnalysisWindow, independent_active_windows

DROPPED_LEGACY_METRICS: dict[str, str] = {
    "PF": (
        "Removed in HydroFragments v1.2: naive patches-per-area fragmentation "
        "index. Use LPI (fixed AOI denominator) and number_of_pools instead."
    ),
    "PLF": (
        "Removed in HydroFragments v1.2: naive patches-per-length index. "
        "Use LPI with an explicit channel reference when drainage is available."
    ),
    "AWMPA": (
        "Removed in HydroFragments v1.2: area-weighted mean patch area is not "
        "part of the canonical register."
    ),
    "AWMPL": (
        "Removed in HydroFragments v1.2: area-weighted mean patch length is not "
        "part of the canonical register."
    ),
    "AWMPW": (
        "Removed in HydroFragments v1.2: area-weighted mean patch width is "
        "deferred pending resolution-floor validation."
    ),
}

FORBIDDEN_LEGACY_COLUMNS = frozenset(DROPPED_LEGACY_METRICS) | {"LPSEC"}

RETAINED_COMPAT_COLUMNS = (
    "date",
    "section",
    "section_area_km2",
    "n_patches",
    "APSEC",
    "AWMSI",
    "AWRe",
    "LPI",
    "pp_mean_%",
    "ra_area_km2",
)


class LegacyMetricMigrationError(ValueError):
    """Raised when a caller requests a metric removed from v1.2."""


def request_legacy_metrics(metric_ids: Iterable[str]) -> None:
    """Fail fast with migration guidance for dropped legacy metrics."""
    requested = [str(item).strip() for item in metric_ids if str(item).strip()]
    if not requested:
        return
    messages: list[str] = []
    for metric_id in requested:
        key = metric_id.upper()
        if key in DROPPED_LEGACY_METRICS:
            messages.append(f"{key}: {DROPPED_LEGACY_METRICS[key]}")
        elif key == "LPSEC":
            messages.append(
                "LPSEC: channel-dependent extent metric excluded from v1.2.0 "
                "core until a real drainage L_ref contract is active."
            )
    if messages:
        raise LegacyMetricMigrationError(
            "Requested legacy metrics are not available in HydroFragments v1.2. "
            "See docs/migration_v1_2.md. " + " ".join(messages)
        )
    unknown = [item for item in requested if item.upper() not in DROPPED_LEGACY_METRICS | {"LPSEC"}]
    if unknown:
        raise LegacyMetricMigrationError(
            "Unknown legacy metric request(s): "
            + ", ".join(unknown)
            + ". Canonical v1.2 metrics are documented in docs/migration_v1_2.md."
        )


def legacy_hydro_config(
    *,
    min_patch_size: int = 2,
    metric_profiles: tuple[str, ...] = ("contracts_core",),
) -> HydroConfig:
    """Build a v1.2 config for ecofragments-shaped calls."""
    min_patch_pixels = max(3, int(min_patch_size) + 1)
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": list(metric_profiles),
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {
                "min_patch_pixels": min_patch_pixels,
                "connectivity_rule": 8,
            },
        }
    )


def _monthly_dataset(
    da_feature: xr.DataArray,
    *,
    valid_obs: xr.DataArray | None = None,
    time_index: int | None = None,
) -> xr.Dataset:
    """Build the bounded ``{water, valid_obs}`` payload for ONE month.

    ``da_feature``/``valid_obs`` may be Dask-backed (lazy). This function
    keeps them lazy right up until the last possible moment, then
    materialises BOTH together in a single fused
    ``xr.Dataset({...}).compute()`` call -- never a separate ``.load()`` per
    array. That fusion eliminates a redundant read (the same bounded input
    would otherwise be pulled twice: once for ``water``, once for
    ``valid_obs``).

    Fusion alone is not enough at catchment scale: a section can span many
    months, and fusing two whole-cube reads into one whole-cube read still
    materialises the entire ``(time, y, x)`` array at once. ``time_index``,
    when given, selects exactly one month's 2-D ``(y, x)`` slice via
    ``.isel(time=time_index)`` *before* the compute, so peak memory here is
    bounded by one month's data, never the whole section's time series --
    this is the literal fix this function exists to provide (see
    ``docs/superpowers/plans/2026-07-27-dea-zones-and-catchment-speed.md``
    section 3.3, Step 2). ``time_index`` is optional only so this function
    remains directly testable/usable against an already-single-month or
    otherwise pre-bounded input (e.g. unit tests exercising the fused-compute
    property in isolation); ``section_compat_rows`` always passes it.

    There must be no cube-wide pre-read (e.g. a separate ``.any().compute()``
    reachability check) anywhere in this path -- any such pass would re-read
    the same bounded source before patch work starts, exactly the
    anti-pattern this function exists to eliminate.

    When ``valid_obs`` is omitted, an all-true mask is synthesised (matching
    legacy callers that have no separate validity concept -- mask == water).
    """
    water = (da_feature == 1).astype(bool)
    if valid_obs is None:
        valid_obs = xr.ones_like(water, dtype=bool)
    else:
        valid_obs = valid_obs.astype(bool)
    combined = xr.Dataset({"water": water, "valid_obs": valid_obs})
    if time_index is not None:
        combined = combined.isel(time=time_index)
    return combined.compute()


def _resolve_local_label_threshold_bytes(config: HydroConfig) -> int | None:
    """Thread a configured ``target_chunk_bytes`` into patch labeling.

    ``None`` (the config default) preserves ``label_components``'s own
    ``ComputePolicy``-class-default fallback exactly as before this
    resolution existed.
    """
    target_chunk_bytes = config.compute.target_chunk_bytes
    if target_chunk_bytes is None:
        return None
    return ComputePolicy(target_chunk_bytes=target_chunk_bytes).target_chunk_bytes


def _measure_month_patch_properties(
    water_month: np.ndarray,
    *,
    analysis_mask: np.ndarray | None,
    windows: Sequence[AnalysisWindow] | None,
    pixel_size_m: float,
    connectivity: int,
    min_patch_pixels: int,
    include_width: bool,
    local_label_threshold_bytes: int | None,
):
    """Measure one month's patch properties, windowed when profitable.

    ``windows`` is the (already-computed) :func:`independent_active_windows`
    partition of ``analysis_mask``, hoisted by the caller so it is computed
    ONCE per section rather than once per month -- the window partition is a
    property of ``analysis_mask`` alone (invariant across every month in a
    section), never of any per-month data, so recomputing it per month would
    be pure waste. When ``analysis_mask`` is ``None`` or covers the whole
    grid (the common full-AOI/legacy case -- ``independent_active_windows``
    then returns exactly one window spanning the grid), this measures the
    whole mask in one call, byte-identical to calling
    ``measure_patch_properties`` directly. When ``analysis_mask`` is a real,
    narrower footprint that splits into multiple independent windows, each
    window's crop is measured separately and every window's properties are
    concatenated -- never reduced per window -- so the caller can reduce
    once across the full set.
    """
    if analysis_mask is None:
        return patch_metrics.measure_patch_properties(
            water_month,
            pixel_size_m=pixel_size_m,
            connectivity=connectivity,
            min_patch_pixels=min_patch_pixels,
            include_width=include_width,
            local_label_threshold_bytes=local_label_threshold_bytes,
        )

    assert windows is not None
    if len(windows) <= 1:
        return patch_metrics.measure_patch_properties(
            water_month,
            pixel_size_m=pixel_size_m,
            connectivity=connectivity,
            min_patch_pixels=min_patch_pixels,
            include_width=include_width,
            local_label_threshold_bytes=local_label_threshold_bytes,
        )

    properties = []
    for window in windows:
        row0, col0, row1, col1 = window.bbox
        crop = water_month[row0:row1, col0:col1]
        properties.extend(
            patch_metrics.measure_patch_properties(
                crop,
                pixel_size_m=pixel_size_m,
                connectivity=connectivity,
                min_patch_pixels=min_patch_pixels,
                include_width=include_width,
                local_label_threshold_bytes=local_label_threshold_bytes,
            )
        )
    return properties


@dataclass(frozen=True)
class _MonthPayload:
    """Bounded, plain-NumPy input for exactly one month's patch/APSEC/coverage
    work -- the unit of work dispatched to a serial call, thread, or process
    worker by :func:`section_compat_rows`.

    Every field here is either a plain NumPy array (already realised by
    :func:`_monthly_dataset` before this payload is built -- never a Dask
    graph or xarray object backed by a remote source), a Python primitive, or
    a frozen dataclass of primitives/tuples (:class:`HydroConfig`,
    :class:`~hydrofragments.spatial.active_windows.AnalysisWindow`). This is
    what makes ``_MonthPayload`` safe to pickle across a Windows spawned
    process boundary -- see the module-level ``ProcessPoolExecutor`` note in
    :func:`section_compat_rows`.

    ``water_month``/``coverage_valid_obs_month`` are carried on the payload
    (rather than only on some separate "result") for two reasons: (1)
    ``_month_row`` needs them to run the patch/APSEC/coverage computation,
    and (2) the caller needs them back, in ``time_index`` order, to feed the
    section-level ``_OccurrenceAccumulator`` (a running, order-independent
    sum -- see its docstring) without re-materialising the month a second
    time.
    """

    time_index: int
    timestamp: pd.Timestamp
    water_month: np.ndarray
    coverage_valid_obs_month: np.ndarray | None
    config: HydroConfig
    pixel_size_m: float
    a_ref_m2: float
    cell_area_m2: float
    min_valid_fraction: float | None
    analysis_mask_np: np.ndarray | None
    windows: Sequence[AnalysisWindow] | None
    want_patches: bool
    want_width: bool
    want_apsec: bool
    local_label_threshold_bytes: int | None


def _build_month_payload(
    da_feature: xr.DataArray,
    *,
    valid_obs: xr.DataArray | None,
    time_index: int,
    timestamp: pd.Timestamp,
    config: HydroConfig,
    pixel_size_m: float,
    a_ref_m2: float,
    cell_area_m2: float,
    min_valid_fraction: float | None,
    analysis_mask_np: np.ndarray | None,
    windows: Sequence[AnalysisWindow] | None,
    want_patches: bool,
    want_width: bool,
    want_apsec: bool,
    local_label_threshold_bytes: int | None,
) -> _MonthPayload:
    """Materialise ONE month's bounded ``water``/``valid_obs`` payload.

    This is the only point where ``time_index`` touches the (possibly
    Dask-backed) ``da_feature``/``valid_obs`` source -- see
    ``_monthly_dataset``'s fused, per-month-bounded ``.compute()``. The
    returned :class:`_MonthPayload` never carries the source array itself,
    only its already-realised one-month NumPy slice, so it is safe to hand to
    any executor (serial, thread, or process).
    """
    monthly = _monthly_dataset(da_feature, valid_obs=valid_obs, time_index=time_index)
    water_month = np.asarray(monthly["water"].values, dtype=bool)
    coverage_valid_obs_month = (
        np.asarray(monthly["valid_obs"].values, dtype=bool)
        if valid_obs is not None
        else None
    )
    return _MonthPayload(
        time_index=time_index,
        timestamp=timestamp,
        water_month=water_month,
        coverage_valid_obs_month=coverage_valid_obs_month,
        config=config,
        pixel_size_m=pixel_size_m,
        a_ref_m2=a_ref_m2,
        cell_area_m2=cell_area_m2,
        min_valid_fraction=min_valid_fraction,
        analysis_mask_np=analysis_mask_np,
        windows=windows,
        want_patches=want_patches,
        want_width=want_width,
        want_apsec=want_apsec,
        local_label_threshold_bytes=local_label_threshold_bytes,
    )


def _month_row(payload: _MonthPayload) -> dict[str, object]:
    """Compute one month's patch/width/APSEC/coverage row from a bounded
    :class:`_MonthPayload`.

    Module-level (not a closure or method) and picklable so it can be
    dispatched to a :class:`~concurrent.futures.ProcessPoolExecutor` worker
    on Windows, where only picklable top-level callables and plain-data
    arguments can safely cross the spawned process boundary. Reproduces
    exactly what the inline per-month loop body in ``section_compat_rows``
    used to compute -- this function's extraction changes only *where* the
    per-month work runs, never *what* it computes.

    Raises the same ``_WATER_VALIDITY_ERROR`` as before when
    ``water=True, valid_obs=False`` anywhere in this month's payload.

    Does NOT touch the section-level ``_OccurrenceAccumulator`` -- occurrence
    is a whole-series aggregate fed by the caller, in deterministic
    ``time_index`` order, from the ``water_month``/``coverage_valid_obs_month``
    this function's return value carries back (see ``per_month_row``'s
    ``"water_month"``/``"coverage_valid_obs_month"`` keys below).
    """
    water_month = payload.water_month
    coverage_valid_obs_month = payload.coverage_valid_obs_month
    config = payload.config
    pixel_size_m = payload.pixel_size_m

    if coverage_valid_obs_month is not None:
        invalid_water = water_month & ~coverage_valid_obs_month
        if np.any(invalid_water):
            raise ValueError(_WATER_VALIDITY_ERROR)

    n_patches: object = None
    awmsi = float("nan")
    awre = float("nan")
    lpi = float("nan")
    width_values: dict[str, float] = {
        column: float("nan") for column in _POOL_WIDTH_STAT_COLUMNS.values()
    }
    width_warning_flags: tuple = ()
    if payload.want_patches or payload.want_width:
        month_properties = _measure_month_patch_properties(
            water_month,
            analysis_mask=payload.analysis_mask_np,
            windows=payload.windows,
            pixel_size_m=pixel_size_m,
            connectivity=config.patches.connectivity_rule,
            min_patch_pixels=config.patches.min_patch_pixels,
            include_width=payload.want_width,
            local_label_threshold_bytes=payload.local_label_threshold_bytes,
        )
        patch_result, width_result = patch_metrics.reduce_patch_properties(
            month_properties,
            pixel_size_m=pixel_size_m,
            a_total_m2=payload.a_ref_m2,
            include_width=payload.want_width,
            resolution_floor_pixels=config.patches.width_resolution_floor_pixels,
        )
        if payload.want_patches:
            n_patches = patch_result.number_of_pools
            awmsi = patch_result.awmsi
            awre = patch_result.awre
            lpi = patch_result.lpi
        if width_result is not None:
            width_values = {
                "pool_width_mean": width_result.mean_m,
                "pool_width_median": width_result.median_m,
                "pool_width_max": width_result.max_m,
                "pool_width_cv": width_result.cv,
            }
            width_warning_flags = width_result.warning_flags

    apsec_value = float("nan")
    apsec_n_water_pixels = None
    if payload.want_apsec:
        month_ds = xr.Dataset(
            {
                "water": (("y", "x"), water_month),
                "valid_obs": (
                    ("y", "x"),
                    coverage_valid_obs_month
                    if coverage_valid_obs_month is not None
                    else np.ones_like(water_month, dtype=bool),
                ),
            }
        ).expand_dims(time=[payload.timestamp])
        month_valid_obs_da = (
            month_ds["valid_obs"] if coverage_valid_obs_month is not None else None
        )
        apsec_record = compute_apsec(
            month_ds,
            a_ref_m2=payload.a_ref_m2,
            cell_area_m2=payload.cell_area_m2,
            config=config,
            valid_obs=month_valid_obs_da,
            min_valid_fraction=payload.min_valid_fraction,
        )[0]
        apsec_value = apsec_record.value
        apsec_n_water_pixels = apsec_record.n_water_pixels

    low_coverage = False
    valid_fraction = None
    n_valid_pixels = None
    if coverage_valid_obs_month is not None:
        valid_fraction = float(coverage_valid_obs_month.mean())
        n_valid_pixels = int(coverage_valid_obs_month.sum())
        low_coverage = valid_fraction < payload.min_valid_fraction

    return {
        "time_index": payload.time_index,
        "timestamp": payload.timestamp,
        "n_patches": n_patches,
        "awmsi": awmsi,
        "awre": awre,
        "lpi": lpi,
        "width_values": width_values,
        "width_warning_flags": width_warning_flags,
        "apsec_value": apsec_value,
        "apsec_n_water_pixels": apsec_n_water_pixels,
        "low_coverage": low_coverage,
        "valid_fraction": valid_fraction,
        "n_valid_pixels": n_valid_pixels,
        "has_coverage": coverage_valid_obs_month is not None,
        "water_month": water_month,
        "coverage_valid_obs_month": coverage_valid_obs_month,
    }


def _run_month_rows(
    payloads: Iterator[_MonthPayload],
    *,
    workers: int,
    executor_kind: Literal["thread", "process"],
) -> list[dict[str, object]]:
    """Run ``_month_row`` over ``payloads`` with bounded producer/consumer
    concurrency and return results in deterministic ``time_index`` order.

    ``workers <= 1`` runs serially in-process with no executor at all --
    this is the exact pre-existing code path (same call, same order, same
    process), preserving today's behavior and avoiding any executor
    overhead for the default configuration.

    For ``workers > 1``, at most ``2 * workers`` payloads are ever
    constructed-but-not-yet-consumed at once: payloads are pulled from the
    ``payloads`` iterator and submitted lazily, one at a time, only as
    earlier futures complete, rather than eagerly building/submitting every
    month up front (a "map everything" approach would defeat the point of
    bounding in-flight memory at catchment scale). Results are collected as
    they complete (so a slow month never blocks faster months from
    finishing) but are sorted back into ``time_index`` order before being
    returned -- output order must never depend on completion order.
    """
    if workers <= 1:
        return [_month_row(payload) for payload in payloads]

    max_in_flight = 2 * workers
    executor_cls = ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor

    results: list[dict[str, object]] = []
    with executor_cls(max_workers=workers) as executor:
        in_flight: dict = {}
        payload_iter = iter(payloads)

        def _fill() -> None:
            while len(in_flight) < max_in_flight:
                try:
                    payload = next(payload_iter)
                except StopIteration:
                    return
                future = executor.submit(_month_row, payload)
                in_flight[future] = None

        _fill()
        while in_flight:
            done, _pending = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                del in_flight[future]
                results.append(future.result())
            _fill()

    results.sort(key=lambda row: row["time_index"])
    return results


_PATCH_METRIC_IDS = frozenset({"number_of_pools", "lpi", "awre", "awmsi"})
_PERSISTENCE_METRIC_IDS = frozenset({"occurrence", "refuge_area"})
_WATER_VALIDITY_ERROR = "water=True requires valid_obs=True for every pixel/month"
_POOL_WIDTH_STAT_COLUMNS = {
    "mean": "pool_width_mean",
    "median": "pool_width_median",
    "max": "pool_width_max",
    "cv": "pool_width_cv",
}


class _OccurrenceAccumulator:
    """Streaming, one-month-at-a-time equivalent of :func:`compute_occurrence`
    plus :func:`compute_refuge_area`.

    ``compute_occurrence`` groups the WHOLE ``(time, y, x)`` cube by calendar
    month (``groupby("time.month").sum(dim="time")``) to build the
    season-stratified P-native ratio (Decision Gate 0 / U2 / Q1). That
    reduction needs every month's data at once *in its original form*, so it
    cannot be called directly against a payload that only ever holds one
    month at a time.

    It CAN be computed incrementally, because every quantity it sums is a
    0/1 (boolean-as-float64) count: ``sum`` over any grouping or processing
    order of 0/1 values is an exact integer in float64 regardless of
    accumulation order (no associativity/rounding risk), so accumulating
    per-calendar-month running totals one month at a time and finalising
    after the loop reproduces ``compute_occurrence``'s whole-array groupby
    result bit-for-bit. This is what lets ``section_compat_rows`` materialise
    only one month's ``water``/``valid_obs`` at a time while still producing
    the exact same occurrence/refuge-area values as before.
    """

    def __init__(self) -> None:
        self._water_valid_by_month: dict[int, np.ndarray] = {}
        self._valid_by_month: dict[int, np.ndarray] = {}
        self._valid_count_total: np.ndarray | None = None

    def add_month(
        self, *, calendar_month: int, water: np.ndarray, valid_obs: np.ndarray
    ) -> None:
        water_valid = (water & valid_obs).astype(np.float64)
        valid = valid_obs.astype(np.float64)

        if calendar_month in self._water_valid_by_month:
            self._water_valid_by_month[calendar_month] += water_valid
            self._valid_by_month[calendar_month] += valid
        else:
            self._water_valid_by_month[calendar_month] = water_valid
            self._valid_by_month[calendar_month] = valid

        valid_count_term = valid_obs.astype(np.int64)
        if self._valid_count_total is None:
            self._valid_count_total = valid_count_term.copy()
        else:
            self._valid_count_total += valid_count_term

    def finalize(self, *, config: HydroConfig) -> persistence.OccurrenceResult:
        if self._valid_count_total is None:
            raise ValueError("_OccurrenceAccumulator.finalize called with no months added")

        ratios = []
        for calendar_month, grouped_water in sorted(self._water_valid_by_month.items()):
            grouped_valid = self._valid_by_month[calendar_month]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.where(grouped_valid > 0, grouped_water / grouped_valid, np.nan)
            ratios.append(ratio)

        stacked = np.stack(ratios, axis=0)
        # `np.nanmean` warns "Mean of empty slice" for pixels where every
        # calendar month is unsupported (all-NaN down that axis); xarray's
        # `.mean(dim=..., skipna=True)` (the pre-fix code path) does not
        # surface that warning, so it is suppressed here to keep behavior
        # (not just values) identical.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            occurrence_values = np.nanmean(stacked, axis=0) * 100.0

        min_valid_obs = config.validity.min_valid_obs
        supported = self._valid_count_total >= min_valid_obs
        occurrence_values = np.where(supported, occurrence_values, np.nan)

        occurrence = xr.DataArray(occurrence_values)
        valid_count = xr.DataArray(self._valid_count_total)

        return persistence.OccurrenceResult(
            occurrence=occurrence,
            valid_count=valid_count,
            min_valid_obs=min_valid_obs,
        )


def section_compat_rows(
    da_feature: xr.DataArray,
    *,
    section: str,
    section_area_km2: float,
    pixel_size_m: float,
    config: HydroConfig,
    selected_ids: set[str] | None = None,
    valid_obs: xr.DataArray | None = None,
    analysis_mask: xr.DataArray | None = None,
    executor_kind: Literal["thread", "process"] = "thread",
) -> list[dict[str, object]]:
    """Compute retained v1.2 metrics in a legacy-compatible wide row shape.

    ``selected_ids`` is an optional B1 optimisation: when provided (only the
    canonical ``analyze()`` path does this), families whose metric ids are
    absent from ``selected_ids`` are skipped entirely rather than computed
    and discarded. When ``None`` (the default -- used by the legacy
    ``calculate_metrics_compat`` shim, which has no concept of "selected
    metrics" and always wants the full fixed wide-row export), every family
    is computed exactly as before. Skipped families still populate their row
    keys with ``None``/``nan`` placeholders so ``compat_dataframe()`` and
    ``_records_from_compat_rows`` (which filters by metric id after
    construction) never see a missing key.

    ``analysis_mask``, when given, is the conservative 2-D potential-water
    footprint (:class:`hydrofragments.models.WaterCube.analysis_mask`). When
    it splits into more than one :func:`independent_active_windows` window,
    each window's patch properties are measured separately and concatenated,
    then reduced into LPI/AWRe/AWMSI/width/counts exactly once across the
    full concatenated set -- never per-window aggregates combined
    afterward. Omitting it (the default) or supplying an all-true mask
    (a single whole-grid window) reproduces today's single whole-mask
    measurement exactly.
    """
    want_patches = selected_ids is None or bool(selected_ids & _PATCH_METRIC_IDS)
    want_width = selected_ids is not None and "pool_width" in selected_ids
    want_persistence = selected_ids is None or bool(selected_ids & _PERSISTENCE_METRIC_IDS)
    want_apsec = selected_ids is None or "apsec" in selected_ids

    if valid_obs is not None:
        # Alignment is checked against metadata alone (dims/sizes), which is
        # always cheap on a lazy (Dask-backed) array -- no compute triggered
        # here.
        if (
            tuple(valid_obs.dims) != tuple(da_feature.dims)
            or dict(valid_obs.sizes) != dict(da_feature.sizes)
        ):
            raise ValueError("valid_obs must align with water")

    spatial_dims = tuple(dim for dim in da_feature.dims if dim != "time")
    analysis_mask_np: np.ndarray | None = None
    if analysis_mask is not None:
        if tuple(analysis_mask.dims) != spatial_dims:
            raise ValueError("analysis_mask must align with water's spatial dims")
        expected_sizes = {dim: da_feature.sizes[dim] for dim in spatial_dims}
        if dict(analysis_mask.sizes) != expected_sizes:
            raise ValueError("analysis_mask must align with water's spatial grid")
        # analysis_mask is always 2-D (never (time, y, x)), so materialising
        # it once here -- regardless of whether it is Dask-backed -- never
        # reintroduces the whole-cube-at-once anti-pattern _monthly_dataset
        # exists to avoid.
        analysis_mask_np = np.asarray(analysis_mask.values, dtype=bool)

    # The independent-active-windows partition is a property of
    # `analysis_mask` alone -- it never depends on any per-month data -- so
    # it is computed exactly ONCE here per section and reused for every
    # month in the loop below, rather than recomputed on every iteration.
    windows: Sequence[AnalysisWindow] | None = None
    if analysis_mask_np is not None:
        windows = independent_active_windows(
            xr.DataArray(analysis_mask_np, dims=("y", "x")),
            connectivity=config.patches.connectivity_rule,
        )

    local_label_threshold_bytes = _resolve_local_label_threshold_bytes(config)

    # `da_feature["time"]` is always a plain (non-Dask) numpy coordinate --
    # reading it does not materialise any pixel data. This gives the section's
    # month count/timestamps up front so the loop below can bound each
    # iteration to exactly one month, rather than requiring the whole
    # (time, y, x) cube to be resident at once just to learn its length.
    timestamps = pd.to_datetime(da_feature["time"].values)
    n_time = len(timestamps)

    cell_area_m2 = float(pixel_size_m) ** 2
    a_ref_m2 = float(section_area_km2) * 1_000_000.0
    min_valid_fraction = (
        config.validity.min_valid_fraction_month if valid_obs is not None else None
    )

    occurrence_acc = _OccurrenceAccumulator() if want_persistence else None

    # Bounded producer/consumer dispatch (W3.2): each month's payload is
    # built lazily -- one at a time, from the fused per-month
    # `_monthly_dataset` compute -- and handed to `_month_row`, either
    # in-process (workers <= 1, today's exact serial path and default) or
    # via a bounded thread/process pool (workers > 1, gated by
    # `config.compute.workers`; see `_run_month_rows`'s
    # `2 * workers`-in-flight bound). `_month_row` never touches the shared
    # `_OccurrenceAccumulator` -- that is a whole-series running sum fed
    # below, in deterministic `time_index` order, from each result's
    # returned `water_month`/`coverage_valid_obs_month` -- so results may
    # come back in any completion order and still reproduce byte-identical
    # occurrence/refuge-area output (the accumulator's sum is
    # order-independent; see its docstring).
    def _payloads() -> Iterator[_MonthPayload]:
        for time_index in range(n_time):
            yield _build_month_payload(
                da_feature,
                valid_obs=valid_obs,
                time_index=time_index,
                timestamp=timestamps[time_index],
                config=config,
                pixel_size_m=pixel_size_m,
                a_ref_m2=a_ref_m2,
                cell_area_m2=cell_area_m2,
                min_valid_fraction=min_valid_fraction,
                analysis_mask_np=analysis_mask_np,
                windows=windows,
                want_patches=want_patches,
                want_width=want_width,
                want_apsec=want_apsec,
                local_label_threshold_bytes=local_label_threshold_bytes,
            )

    per_month = _run_month_rows(
        _payloads(), workers=config.compute.workers, executor_kind=executor_kind
    )
    # `_run_month_rows` already sorts by `time_index`; feeding the
    # accumulator in that fixed order (rather than whatever order results
    # happened to be produced/collected in) keeps behavior identical to the
    # pre-refactor chronological loop even though the sum itself is
    # order-independent.
    if occurrence_acc is not None:
        for month in per_month:
            calendar_month = int(month["timestamp"].month)
            water_month = month["water_month"]
            occurrence_acc.add_month(
                calendar_month=calendar_month,
                water=water_month,
                valid_obs=(
                    month["coverage_valid_obs_month"]
                    if month["coverage_valid_obs_month"] is not None
                    else np.ones_like(water_month, dtype=bool)
                ),
            )

    pp_mean = float("nan")
    refuge = None
    if occurrence_acc is not None:
        occurrence = occurrence_acc.finalize(config=config)
        refuge = compute_refuge_area(
            occurrence, cell_area_m2=cell_area_m2, config=config
        )
        # xarray's `.mean(skipna=True)` (the pre-fix code path) does not
        # surface a "Mean of empty slice" warning for an all-NaN occurrence
        # surface; `np.nanmean` does, so it is suppressed here for identical
        # observable behavior, matching `_OccurrenceAccumulator.finalize`.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            pp_mean = float(np.nanmean(occurrence.occurrence.values))
        if np.isnan(pp_mean):
            pp_mean = float("nan")

    rows: list[dict[str, object]] = []
    for month in per_month:
        row = {
            "date": pd.Timestamp(month["timestamp"]),
            "section": section,
            "section_area_km2": section_area_km2,
            "n_patches": month["n_patches"],
            "APSEC": month["apsec_value"],
            "AWMSI": month["awmsi"],
            "AWRe": month["awre"],
            "LPI": month["lpi"],
            "pp_mean_%": pp_mean,
            "ra_area_km2": refuge.value if refuge is not None else float("nan"),
        }
        if want_width:
            row.update(month["width_values"])
            row["pool_width_warning_flags"] = month["width_warning_flags"]
        if month["has_coverage"]:
            row.update(
                {
                    "low_coverage_flag": month["low_coverage"],
                    "valid_fraction_month": month["valid_fraction"],
                    "min_valid_fraction_month": min_valid_fraction,
                    "n_valid_pixels": month["n_valid_pixels"],
                    "APSEC_n_water_pixels": month["apsec_n_water_pixels"],
                }
            )
        rows.append(row)
    return rows


def compat_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    forbidden = FORBIDDEN_LEGACY_COLUMNS.intersection(frame.columns)
    if forbidden:
        raise LegacyMetricMigrationError(
            "Compatibility output must not include dropped metrics: "
            + ", ".join(sorted(forbidden))
        )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["n_patches"] = frame["n_patches"].astype("int32")
    return frame.sort_values(["section", "date"]).reset_index(drop=True)


def calculate_metrics_compat(
    da_wmask: xr.DataArray | xr.Dataset | str,
    *,
    rcor_extent: gpd.GeoDataFrame | str | None = None,
    outdir: str | None = None,
    section_length: float | None = None,
    section_name_col: str | None = None,
    min_patch_size: int = 2,
    img_ext: str = ".tif",
    export_shp: bool = False,
    export_PP: bool = False,
    fill_nodata: bool = True,
    legacy_metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Run retained v1.2 metrics and return a non-canonical wide pivot."""
    import os
    import tempfile

    from ecofragments.utils import calc_metrics

    if legacy_metrics is not None:
        request_legacy_metrics(legacy_metrics)

    if export_shp:
        raise LegacyMetricMigrationError(
            "export_shp is not supported on the v1.2 compatibility facade. "
            "Use hydrofragments analyze() with output.include_vectors instead."
        )
    if export_PP:
        raise LegacyMetricMigrationError(
            "export_PP via ecofragments.calculate_metrics is replaced by tidy "
            "occurrence/refuge rasters from hydrofragments.analyze()."
        )

    config = legacy_hydro_config(min_patch_size=min_patch_size)
    resolve_metrics(
        config.metric_profiles,
        available_dependencies={
            MetricDependency.VALIDITY,
            MetricDependency.PATCHES,
        },
    )

    if rcor_extent is None:
        outdir = outdir or tempfile.mkdtemp(prefix="hydrofragments_")
        array = calc_metrics.coerce_water_mask_dataarray(da_wmask)
        if array.sizes.get("time", 0) < 2:
            raise ValueError("at least two timesteps are required to calculate metrics")
        pixel_size = 30.0
        if hasattr(array, "rio") and "x" in array.coords and "y" in array.coords:
            try:
                pixel_size = float(abs(array.rio.resolution()[0]))
            except Exception:
                pixel_size = 30.0
        section_area_km2 = float(array.isel(time=0).size) * pixel_size**2 / 1_000_000.0
        rows = section_compat_rows(
            array,
            section="AOI",
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size,
            config=config,
        )
        metrics_df = compat_dataframe(rows)
        metrics_df.to_csv(os.path.join(outdir, "ecof_metrics.csv"), index=False)
        return metrics_df

    da_wmask, rcor_extent, section_length, crs, pixel_size, outdir = calc_metrics.validate(
        da_wmask,
        rcor_extent,
        outdir,
        section_length,
        img_ext,
        section_name_col,
    )
    da_wmask, rcor_extent = calc_metrics.preprocess(
        da_wmask, rcor_extent, fill_nodata
    )

    rows: list[dict[str, object]] = []
    for _, feature in rcor_extent.iterrows():
        prepared = calc_metrics.preprocess_feature_operations(
            da_wmask, feature, section_name_col
        )
        rows.extend(
            section_compat_rows(
                prepared["da_wmask_feature"],
                section=str(prepared["section"]),
                section_area_km2=float(prepared["section_area"]),
                pixel_size_m=float(pixel_size),
                config=config,
            )
        )

    metrics_df = compat_dataframe(rows)
    if outdir is not None:
        metrics_df.to_csv(f"{outdir}/ecof_metrics.csv", index=False)

    return metrics_df


__all__ = [
    "DROPPED_LEGACY_METRICS",
    "FORBIDDEN_LEGACY_COLUMNS",
    "LegacyMetricMigrationError",
    "RETAINED_COMPAT_COLUMNS",
    "calculate_metrics_compat",
    "compat_dataframe",
    "legacy_hydro_config",
    "request_legacy_metrics",
    "section_compat_rows",
]
