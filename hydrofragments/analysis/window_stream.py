"""Byte-admitted spatial-window / time-block streaming for section analysis."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
import gc
import tempfile
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.extent import compute_apsec
import hydrofragments.metrics.patches as patch_metrics
from hydrofragments.metrics.patches import PatchProperties
from hydrofragments.spatial.active_windows import AnalysisWindow

DEFAULT_WORKER_MEMORY_FRACTION = 0.5

# Names of export-only checkpoint consumers that must not be built when exports
# are disabled (Task 7/8 will register real classes against these literals).
_EXPORT_ONLY_CONSUMER_NAMES = frozenset(
    {
        "PoolCheckpointConsumer",
        "VectorCheckpointConsumer",
        "ExportCheckpointConsumer",
    }
)


class MemoryBudgetExceeded(MemoryError):
    """Raised before materializing a morphology crop that exceeds the worker budget."""

    def __init__(
        self,
        message: str,
        *,
        window_id: str | None = None,
        estimated_live_bytes: int | None = None,
        budget_bytes: int | None = None,
        mitigation: str | None = None,
    ) -> None:
        super().__init__(message)
        self.window_id = window_id
        self.estimated_live_bytes = estimated_live_bytes
        self.budget_bytes = budget_bytes
        self.mitigation = mitigation


@dataclass(frozen=True)
class MetricPartial:
    """Small additive metric fragment from one admitted spatial block."""

    name: str
    payload: Mapping[str, Any]


from hydrofragments.patches.labels import LabelCheckpointRef


@dataclass(frozen=True)
class MeasuredPatchBundle:
    """Patch properties plus either in-memory labels or a label checkpoint."""

    properties: tuple[PatchProperties, ...] = ()
    labels: np.ndarray | None = None
    label_checkpoint: LabelCheckpointRef | None = None
    window_id: str | None = None

    def __post_init__(self) -> None:
        if self.labels is not None and self.label_checkpoint is not None:
            raise ValueError("MeasuredPatchBundle cannot hold both labels and a checkpoint")
        if self.labels is None and self.label_checkpoint is None and self.properties:
            raise ValueError("MeasuredPatchBundle with properties requires labels or checkpoint")


@dataclass(frozen=True)
class WindowMonthResult:
    """One admitted spatial window for exactly one month."""

    time_index: int
    date: pd.Timestamp
    window_id: str
    row_slice: slice
    col_slice: slice
    estimated_live_bytes: int
    metric_partials: Mapping[str, MetricPartial]
    water: np.ndarray
    valid_obs: np.ndarray
    patch_bundle: MeasuredPatchBundle | None = None


class WindowMonthConsumer(Protocol):
  def consume(self, block: WindowMonthResult) -> None: ...
  def finalize(self) -> object: ...
  def abort(self) -> None: ...


def resolve_worker_byte_budget(
    config: HydroConfig,
    *,
    in_flight_slots: int = 1,
) -> int:
    """Derive per-slot admitted live bytes from compute policy fields."""

    target = (
        config.compute.target_chunk_bytes
        if config.compute.target_chunk_bytes is not None
        else ComputePolicy().target_chunk_bytes
    )
    fraction = (
        config.compute.worker_memory_fraction
        if config.compute.worker_memory_fraction is not None
        else DEFAULT_WORKER_MEMORY_FRACTION
    )
    workers = max(1, config.compute.workers)
    slots = max(1, in_flight_slots)
    total = int(target * fraction)
    if workers > 1:
        return max(1, total // slots)
    return max(1, total // slots)


def _estimate_live_bytes(*arrays: np.ndarray, multiplier: float = 4.0) -> int:
    nbytes = sum(int(arr.nbytes) for arr in arrays)
    return int(np.ceil(nbytes * multiplier))


def _materialize_window_month(
    da_feature: xr.DataArray,
    valid_obs: xr.DataArray | None,
    *,
    time_index: int,
    window: AnalysisWindow,
) -> tuple[np.ndarray, np.ndarray]:
    row0, col0, row1, col1 = window.bbox
    water_sel = da_feature.isel(time=time_index, y=slice(row0, row1), x=slice(col0, col1))
    water = (water_sel == 1).astype(bool)
    if valid_obs is None:
        valid = xr.ones_like(water, dtype=bool)
    else:
        valid = valid_obs.isel(time=time_index, y=slice(row0, row1), x=slice(col0, col1)).astype(
            bool
        )
    combined = xr.Dataset({"water": water, "valid_obs": valid}).compute()
    return (
        np.asarray(combined["water"].values, dtype=bool),
        np.asarray(combined["valid_obs"].values, dtype=bool),
    )


def _build_patch_bundle(
    water_crop: np.ndarray,
    *,
    window: AnalysisWindow,
    pixel_size_m: float,
    connectivity: int,
    min_patch_pixels: int,
    include_width: bool,
    local_label_threshold_bytes: int | None,
    max_component_bytes: int,
    spill_dir: Path | None,
) -> MeasuredPatchBundle | None:
    label_measure, checkpoint = patch_metrics.label_and_measure_window(
        water_crop,
        pixel_size_m=pixel_size_m,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        include_width=include_width,
        local_label_threshold_bytes=local_label_threshold_bytes,
        max_component_bytes=max_component_bytes,
        window_id=window.window_id,
        spill_dir=spill_dir,
    )
    if label_measure is None and checkpoint is None:
        return None
    properties = label_measure.properties if label_measure is not None else ()
    labels = label_measure.labels if label_measure is not None else None
    return MeasuredPatchBundle(
        properties=properties,
        labels=labels,
        label_checkpoint=checkpoint,
        window_id=window.window_id,
    )


def _coverage_partials(
    valid_obs: np.ndarray,
    *,
    analysis_mask_crop: np.ndarray | None,
) -> dict[str, MetricPartial]:
    selection = (
        valid_obs if analysis_mask_crop is None else valid_obs[analysis_mask_crop]
    )
    return {
        "coverage": MetricPartial(
            name="coverage",
            payload={
                "valid_pixels": int(selection.sum()),
                "total_pixels": int(selection.size),
            },
        )
    }


def stream_month_windows(
    da_feature: xr.DataArray,
    valid_obs: xr.DataArray | None,
    *,
    time_index: int,
    timestamp: pd.Timestamp,
    windows: Sequence[AnalysisWindow],
    consumers: Sequence[WindowMonthConsumer],
    budget_bytes: int,
    pixel_size_m: float,
    connectivity: int,
    min_patch_pixels: int,
    want_patches: bool,
    want_width: bool,
    analysis_mask_np: np.ndarray | None,
    local_label_threshold_bytes: int | None,
    spill_dir: Path | None = None,
) -> None:
    """Materialize and drain one month's windows under byte admission."""

    admitted_live = 0
    active_blocks: list[WindowMonthResult] = []

    def _release_block(block: WindowMonthResult) -> None:
        nonlocal admitted_live
        admitted_live -= block.estimated_live_bytes

    try:
        for window in windows:
            row0, col0, row1, col1 = window.bbox
            water_crop, valid_crop = _materialize_window_month(
                da_feature,
                valid_obs,
                time_index=time_index,
                window=window,
            )
            analysis_crop = (
                None
                if analysis_mask_np is None
                else analysis_mask_np[row0:row1, col0:col1]
            )
            patch_bundle = None
            if want_patches or want_width:
                patch_bundle = _build_patch_bundle(
                    water_crop,
                    window=window,
                    pixel_size_m=pixel_size_m,
                    connectivity=connectivity,
                    min_patch_pixels=min_patch_pixels,
                    include_width=want_width,
                    local_label_threshold_bytes=local_label_threshold_bytes,
                    max_component_bytes=budget_bytes,
                    spill_dir=spill_dir,
                )
            metric_partials = _coverage_partials(
                valid_crop, analysis_mask_crop=analysis_crop
            )
            estimated = _estimate_live_bytes(water_crop, valid_crop)
            if patch_bundle is not None and patch_bundle.labels is not None:
                estimated = _estimate_live_bytes(water_crop, valid_crop, patch_bundle.labels)
            if admitted_live + estimated > budget_bytes and active_blocks:
                raise MemoryBudgetExceeded(
                    f"admitted live bytes would exceed worker budget "
                    f"({admitted_live + estimated} > {budget_bytes}) for "
                    f"window={window.window_id}",
                    window_id=window.window_id,
                    estimated_live_bytes=admitted_live + estimated,
                    budget_bytes=budget_bytes,
                    mitigation="reduce target_chunk_bytes, worker count, or spatial window size",
                )
            block = WindowMonthResult(
                time_index=time_index,
                date=timestamp,
                window_id=window.window_id,
                row_slice=slice(row0, row1),
                col_slice=slice(col0, col1),
                estimated_live_bytes=estimated,
                metric_partials=metric_partials,
                water=water_crop,
                valid_obs=valid_crop,
                patch_bundle=patch_bundle,
            )
            admitted_live += estimated
            active_blocks.append(block)
            for consumer in consumers:
                consumer.consume(block)
            _release_block(block)
            active_blocks.clear()
            del block, water_crop, valid_crop, patch_bundle
            gc.collect()
    except Exception:
        for consumer in consumers:
            consumer.abort()
        raise


@dataclass
class _MonthStreamState:
    """Accumulators for one month's streamed windows."""

    time_index: int
    timestamp: pd.Timestamp
    config: HydroConfig
    pixel_size_m: float
    a_ref_m2: float
    cell_area_m2: float
    min_valid_fraction: float | None
    want_patches: bool
    want_width: bool
    want_apsec: bool
    has_coverage: bool
    grid_shape: tuple[int, int]
    patch_properties: list[PatchProperties] = field(default_factory=list)
    water_month: np.ndarray | None = None
    coverage_valid_obs_month: np.ndarray | None = None
    coverage_valid_pixels: int = 0
    coverage_total_pixels: int = 0

    def as_consumer(self) -> WindowMonthConsumer:
        return _MonthStreamConsumer(self)


class _MonthStreamConsumer:
    def __init__(self, state: _MonthStreamState) -> None:
        self._state = state

    def consume(self, block: WindowMonthResult) -> None:
        state = self._state
        if state.water_month is None:
            state.water_month = np.zeros(state.grid_shape, dtype=bool)
            if state.has_coverage:
                state.coverage_valid_obs_month = np.zeros(state.grid_shape, dtype=bool)
        state.water_month[block.row_slice, block.col_slice] = block.water
        if state.coverage_valid_obs_month is not None:
            state.coverage_valid_obs_month[block.row_slice, block.col_slice] = block.valid_obs
        if block.patch_bundle is not None:
            state.patch_properties.extend(block.patch_bundle.properties)
        partial = block.metric_partials.get("coverage")
        if partial is not None:
            state.coverage_valid_pixels += int(partial.payload["valid_pixels"])
            state.coverage_total_pixels += int(partial.payload["total_pixels"])

    def finalize(self) -> object:
        return None

    def abort(self) -> None:
        self._state.patch_properties.clear()
        self._state.water_month = None
        self._state.coverage_valid_obs_month = None


_WATER_VALIDITY_ERROR = "water=True requires valid_obs=True for every pixel/month"
_POOL_WIDTH_STAT_COLUMNS = {
    "mean": "pool_width_mean",
    "median": "pool_width_median",
    "max": "pool_width_max",
    "cv": "pool_width_cv",
}


def _finalize_month_row(state: _MonthStreamState) -> dict[str, object]:
    water_month = state.water_month
    if water_month is None:
        water_month = np.zeros(state.grid_shape, dtype=bool)
    coverage_valid_obs_month = state.coverage_valid_obs_month

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
    if state.want_patches or state.want_width:
        patch_result, width_result = patch_metrics.reduce_patch_properties(
            state.patch_properties,
            pixel_size_m=state.pixel_size_m,
            a_total_m2=state.a_ref_m2,
            include_width=state.want_width,
            resolution_floor_pixels=state.config.patches.width_resolution_floor_pixels,
        )
        if state.want_patches:
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
    if state.want_apsec:
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
        ).expand_dims(time=[state.timestamp])
        month_valid_obs_da = (
            month_ds["valid_obs"] if coverage_valid_obs_month is not None else None
        )
        apsec_record = compute_apsec(
            month_ds,
            a_ref_m2=state.a_ref_m2,
            cell_area_m2=state.cell_area_m2,
            config=state.config,
            valid_obs=month_valid_obs_da,
            min_valid_fraction=state.min_valid_fraction,
        )[0]
        apsec_value = apsec_record.value
        apsec_n_water_pixels = apsec_record.n_water_pixels

    low_coverage = False
    valid_fraction = None
    n_valid_pixels = None
    if state.has_coverage:
        if state.coverage_total_pixels > 0:
            valid_fraction = float(
                state.coverage_valid_pixels / state.coverage_total_pixels
            )
            n_valid_pixels = state.coverage_valid_pixels
            low_coverage = valid_fraction < state.min_valid_fraction
        else:
            valid_fraction = float("nan")
            n_valid_pixels = 0
            low_coverage = True

    occurrence_payload = {
        "calendar_month": int(state.timestamp.month),
        "water_month": water_month,
        "coverage_valid_obs_month": coverage_valid_obs_month,
    }

    return {
        "time_index": state.time_index,
        "timestamp": state.timestamp,
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
        "has_coverage": state.has_coverage,
        "occurrence_payload": occurrence_payload,
    }


@dataclass(frozen=True)
class _MonthStreamJob:
    time_index: int
    timestamp: pd.Timestamp
    config: HydroConfig
    pixel_size_m: float
    a_ref_m2: float
    cell_area_m2: float
    min_valid_fraction: float | None
    analysis_mask_np: np.ndarray | None
    windows: Sequence[AnalysisWindow]
    want_patches: bool
    want_width: bool
    want_apsec: bool
    local_label_threshold_bytes: int | None
    budget_bytes: int
    grid_shape: tuple[int, int]
    has_coverage: bool


def _process_month_stream_job(
    da_feature: xr.DataArray,
    valid_obs: xr.DataArray | None,
    job: _MonthStreamJob,
    *,
    spill_dir: Path | None,
) -> dict[str, object]:
    state = _MonthStreamState(
        time_index=job.time_index,
        timestamp=job.timestamp,
        config=job.config,
        pixel_size_m=job.pixel_size_m,
        a_ref_m2=job.a_ref_m2,
        cell_area_m2=job.cell_area_m2,
        min_valid_fraction=job.min_valid_fraction,
        want_patches=job.want_patches,
        want_width=job.want_width,
        want_apsec=job.want_apsec,
        has_coverage=job.has_coverage,
        grid_shape=job.grid_shape,
    )
    consumers: list[WindowMonthConsumer] = [state.as_consumer()]
    stream_month_windows(
        da_feature,
        valid_obs,
        time_index=job.time_index,
        timestamp=job.timestamp,
        windows=job.windows,
        consumers=consumers,
        budget_bytes=job.budget_bytes,
        pixel_size_m=job.pixel_size_m,
        connectivity=job.config.patches.connectivity_rule,
        min_patch_pixels=job.config.patches.min_patch_pixels,
        want_patches=job.want_patches,
        want_width=job.want_width,
        analysis_mask_np=job.analysis_mask_np,
        local_label_threshold_bytes=job.local_label_threshold_bytes,
        spill_dir=spill_dir,
    )
    row = _finalize_month_row(state)
    state.patch_properties.clear()
    state.water_month = None
    state.coverage_valid_obs_month = None
    return row


def build_section_consumers(
    *,
    export_enabled: bool,
    extra_consumers: Sequence[WindowMonthConsumer] = (),
) -> list[WindowMonthConsumer]:
    """Return metric consumers only when spatial export is disabled."""

    if export_enabled:
        for consumer in extra_consumers:
            name = type(consumer).__name__
            if name in _EXPORT_ONLY_CONSUMER_NAMES:
                return list(extra_consumers)
    else:
        for consumer in extra_consumers:
            name = type(consumer).__name__
            if name in _EXPORT_ONLY_CONSUMER_NAMES:
                raise ValueError(
                    f"export-disabled analysis cannot construct export consumer {name}"
                )
    return list(extra_consumers)


def stream_section_month_rows(
    da_feature: xr.DataArray,
    *,
    valid_obs: xr.DataArray | None,
    timestamps: Sequence[pd.Timestamp],
    config: HydroConfig,
    pixel_size_m: float,
    a_ref_m2: float,
    cell_area_m2: float,
    min_valid_fraction: float | None,
    analysis_mask_np: np.ndarray | None,
    windows: Sequence[AnalysisWindow],
    want_patches: bool,
    want_width: bool,
    want_apsec: bool,
    local_label_threshold_bytes: int | None,
    workers: int,
    executor_kind: Literal["thread", "process"],
    export_enabled: bool = False,
    extra_consumers: Sequence[WindowMonthConsumer] = (),
    occurrence_feeder: Any | None = None,
) -> list[dict[str, object]]:
    """Stream all months under a spatial byte budget; omit retained full-grid payloads."""

    spatial_dims = tuple(dim for dim in da_feature.dims if dim != "time")
    grid_shape = tuple(int(da_feature.sizes[dim]) for dim in spatial_dims)
    has_coverage = valid_obs is not None
    in_flight_slots = 1 if workers <= 1 else 2 * workers
    budget_bytes = resolve_worker_byte_budget(config, in_flight_slots=in_flight_slots)

    build_section_consumers(export_enabled=export_enabled, extra_consumers=extra_consumers)

    spill_dir: Path | None = None
    if not export_enabled:
        spill_dir = Path(tempfile.mkdtemp(prefix="hf_scientific_spill_"))

    jobs = [
        _MonthStreamJob(
            time_index=time_index,
            timestamp=pd.Timestamp(timestamps[time_index]),
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
            budget_bytes=budget_bytes,
            grid_shape=grid_shape,
            has_coverage=has_coverage,
        )
        for time_index in range(len(timestamps))
    ]

    def _feed_occurrence(row: dict[str, object]) -> None:
        if occurrence_feeder is None:
            return
        payload = row.get("occurrence_payload")
        if not isinstance(payload, dict):
            return
        occurrence_feeder.add_month(
            calendar_month=int(payload["calendar_month"]),
            water=payload["water_month"],
            valid_obs=(
                payload["coverage_valid_obs_month"]
                if payload["coverage_valid_obs_month"] is not None
                else np.ones_like(payload["water_month"], dtype=bool)
            ),
        )
        payload.clear()

    def _public_row(row: dict[str, object]) -> dict[str, object]:
        _feed_occurrence(row)
        return {k: v for k, v in row.items() if k != "occurrence_payload"}

    if workers <= 1:
        results = [
            _public_row(
                _process_month_stream_job(
                    da_feature, valid_obs, job, spill_dir=spill_dir
                )
            )
            for job in jobs
        ]
    else:
        max_in_flight = 2 * workers
        executor_cls = (
            ThreadPoolExecutor if executor_kind == "thread" else ProcessPoolExecutor
        )
        results: list[dict[str, object]] = []
        with executor_cls(max_workers=workers) as executor:
            in_flight: dict = {}
            job_iter = iter(jobs)

            def _fill() -> None:
                while len(in_flight) < max_in_flight:
                    try:
                        job = next(job_iter)
                    except StopIteration:
                        return
                    future = executor.submit(
                        _process_month_stream_job,
                        da_feature,
                        valid_obs,
                        job,
                        spill_dir=spill_dir,
                    )
                    in_flight[future] = None

            _fill()
            while in_flight:
                done, _pending = wait(list(in_flight), return_when=FIRST_COMPLETED)
                for future in done:
                    del in_flight[future]
                    results.append(_public_row(future.result()))
                _fill()
        results.sort(key=lambda row: row["time_index"])

    if spill_dir is not None and spill_dir.exists():
        for child in spill_dir.iterdir():
            if child.is_dir():
                for nested in child.rglob("*"):
                    if nested.is_file():
                        nested.unlink()
                child.rmdir()
            elif child.is_file():
                child.unlink()
        spill_dir.rmdir()

    return results


__all__ = [
    "DEFAULT_WORKER_MEMORY_FRACTION",
    "MemoryBudgetExceeded",
    "MetricPartial",
    "MeasuredPatchBundle",
    "WindowMonthConsumer",
    "WindowMonthResult",
    "build_section_consumers",
    "resolve_worker_byte_budget",
    "stream_month_windows",
    "stream_section_month_rows",
]
