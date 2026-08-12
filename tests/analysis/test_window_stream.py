"""Byte-admitted window streaming memory regressions and correctness."""

from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from hydrofragments.analysis.window_stream import (
    MemoryBudgetExceeded,
    MeasuredPatchBundle,
    MetricPartial,
    WindowMonthResult,
    build_section_consumers,
    resolve_worker_byte_budget,
    stream_month_windows,
    stream_section_month_rows,
)
from hydrofragments.config import HydroConfig
from hydrofragments.patches.components import ComponentCrop
from hydrofragments.patches.morphology import measure_components
from hydrofragments.section_analysis import analyze_section_rows
from hydrofragments.patches.labels import LabelCheckpointRef
from hydrofragments.spatial.active_windows import AnalysisWindow


def _config(
    *,
    workers: int = 1,
    target_chunk_bytes: int | None = None,
    worker_memory_fraction: float | None = None,
) -> HydroConfig:
    compute: dict[str, object] = {"workers": workers}
    if target_chunk_bytes is not None:
        compute["target_chunk_bytes"] = target_chunk_bytes
    if worker_memory_fraction is not None:
        compute["worker_memory_fraction"] = worker_memory_fraction
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "patches": {"min_patch_pixels": 1, "connectivity_rule": 8},
            "compute": compute,
        }
    )


def _cube(
    *,
    n_time: int,
    n_y: int,
    n_x: int,
    seed: int,
    chunks: tuple[int, int, int] | None = None,
):
    rng = np.random.default_rng(seed)
    water = (rng.random((n_time, n_y, n_x)) < 0.25).astype("int8")
    valid = rng.random((n_time, n_y, n_x)) >= 0.05
    water = water & valid.astype("int8")
    times = (np.datetime64("2000-01", "M") + np.arange(n_time)).astype("datetime64[ns]")
    coords = {
        "time": times,
        "y": np.arange(n_y, dtype=float) * -30.0,
        "x": np.arange(n_x, dtype=float) * 30.0,
    }
    if chunks is None:
        da_feature = xr.DataArray(water, dims=("time", "y", "x"), coords=coords)
        valid_da = xr.DataArray(valid, dims=("time", "y", "x"), coords=coords)
    else:
        da_feature = xr.DataArray(
            da.from_array(water, chunks=chunks),
            dims=("time", "y", "x"),
            coords=coords,
        )
        valid_da = xr.DataArray(
            da.from_array(valid, chunks=chunks),
            dims=("time", "y", "x"),
            coords=coords,
        )
    return da_feature, valid_da


class _SpyConsumer:
    def __init__(self) -> None:
        self.blocks: list[WindowMonthResult] = []
        self.aborted = False

    def consume(self, block: WindowMonthResult) -> None:
        self.blocks.append(block)

    def finalize(self) -> object:
        return len(self.blocks)

    def abort(self) -> None:
        self.aborted = True
        self.blocks.clear()


def test_resolve_worker_byte_budget_uses_policy_fields():
    config = _config(
        workers=2,
        target_chunk_bytes=1_000_000,
        worker_memory_fraction=0.5,
    )
    per_slot = resolve_worker_byte_budget(config, in_flight_slots=4)
    assert per_slot == 125_000


def test_completed_block_arrays_become_collectible():
    da_feature, valid_da = _cube(n_time=2, n_y=6, n_x=6, seed=1)
    windows = (AnalysisWindow(window_id="window-0001", bbox=(0, 0, 6, 6)),)
    refs: list[weakref.ReferenceType] = []

    class _TrackingConsumer:
        def consume(self, block: WindowMonthResult) -> None:
            refs.append(weakref.ref(block.water))

        def finalize(self) -> object:
            return None

        def abort(self) -> None:
            pass

    stream_month_windows(
        da_feature,
        valid_da,
        time_index=0,
        timestamp=pd.Timestamp(da_feature["time"].values[0]),
        windows=windows,
        consumers=[_TrackingConsumer()],
        budget_bytes=resolve_worker_byte_budget(_config()),
        pixel_size_m=30.0,
        connectivity=8,
        min_patch_pixels=1,
        want_patches=True,
        want_width=False,
        analysis_mask_np=None,
        local_label_threshold_bytes=None,
    )
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_stream_section_month_rows_480_month_small_grid_bounded():
    """480-month small grid: rows complete without retaining every month's full grid."""

    n_time, n_y, n_x = 480, 4, 4
    da_feature, valid_da = _cube(n_time=n_time, n_y=n_y, n_x=n_x, seed=2)
    config = _config(target_chunk_bytes=256_000, worker_memory_fraction=0.25)
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0

    rows = stream_section_month_rows(
        da_feature,
        valid_obs=valid_da,
        timestamps=pd.to_datetime(da_feature["time"].values),
        config=config,
        pixel_size_m=30.0,
        a_ref_m2=section_area_km2 * 1_000_000.0,
        cell_area_m2=900.0,
        min_valid_fraction=config.validity.min_valid_fraction_month,
        analysis_mask_np=None,
        windows=(AnalysisWindow(window_id="window-0001", bbox=(0, 0, n_y, n_x)),),
        want_patches=True,
        want_width=False,
        want_apsec=False,
        local_label_threshold_bytes=None,
        workers=1,
        executor_kind="thread",
    )
    assert len(rows) == n_time
    assert "water_month" not in rows[0]
    assert "occurrence_payload" not in rows[0]


def test_large_dask_grid_sparse_windows_byte_budget():
    n_time, n_y, n_x = 3, 128, 128
    da_feature, valid_da = _cube(
        n_time=n_time,
        n_y=n_y,
        n_x=n_x,
        seed=3,
        chunks=(1, 32, 32),
    )
    analysis_mask = np.zeros((n_y, n_x), dtype=bool)
    analysis_mask[10:20, 10:20] = True
    windows = (AnalysisWindow(window_id="window-0001", bbox=(8, 8, 24, 24)),)
    config = _config(target_chunk_bytes=512_000, worker_memory_fraction=0.25)

    rows = stream_section_month_rows(
        da_feature,
        valid_obs=valid_da,
        timestamps=pd.to_datetime(da_feature["time"].values),
        config=config,
        pixel_size_m=30.0,
        a_ref_m2=float(n_y * n_x) * 900.0,
        cell_area_m2=900.0,
        min_valid_fraction=config.validity.min_valid_fraction_month,
        analysis_mask_np=analysis_mask,
        windows=windows,
        want_patches=True,
        want_width=False,
        want_apsec=False,
        local_label_threshold_bytes=256_000,
        workers=1,
        executor_kind="thread",
    )
    assert len(rows) == n_time


def test_out_of_order_completion_sorted_by_time_index():
    da_feature, valid_da = _cube(n_time=6, n_y=8, n_x=8, seed=4)
    config = _config(workers=2)
    section_area_km2 = float(8 * 8) * 900.0 / 1_000_000.0

    with mock.patch(
        "hydrofragments.analysis.window_stream._process_month_stream_job",
        side_effect=lambda da_f, valid, job, spill_dir=None: {
            "time_index": job.time_index,
            "timestamp": job.timestamp,
            "n_patches": job.time_index,
        },
    ):
        rows = stream_section_month_rows(
            da_feature,
            valid_obs=valid_da,
            timestamps=pd.to_datetime(da_feature["time"].values),
            config=config,
            pixel_size_m=30.0,
            a_ref_m2=section_area_km2 * 1_000_000.0,
            cell_area_m2=900.0,
            min_valid_fraction=config.validity.min_valid_fraction_month,
            analysis_mask_np=None,
            windows=(AnalysisWindow(window_id="window-0001", bbox=(0, 0, 8, 8)),),
            want_patches=True,
            want_width=False,
            want_apsec=False,
            local_label_threshold_bytes=None,
            workers=2,
            executor_kind="thread",
        )
    assert [row["time_index"] for row in rows] == list(range(6))


def test_exception_cancels_consumers():
    da_feature, valid_da = _cube(n_time=1, n_y=4, n_x=4, seed=5)
    consumer = _SpyConsumer()

    class _FailConsumer:
        def consume(self, block: WindowMonthResult) -> None:
            raise RuntimeError("boom")

        def finalize(self) -> object:
            return None

        def abort(self) -> None:
            consumer.abort()

    with pytest.raises(RuntimeError, match="boom"):
        stream_month_windows(
            da_feature,
            valid_da,
            time_index=0,
            timestamp=pd.Timestamp(da_feature["time"].values[0]),
            windows=(AnalysisWindow(window_id="window-0001", bbox=(0, 0, 4, 4)),),
            consumers=[_FailConsumer()],
            budget_bytes=10_000_000,
            pixel_size_m=30.0,
            connectivity=8,
            min_patch_pixels=1,
            want_patches=False,
            want_width=False,
            analysis_mask_np=None,
            local_label_threshold_bytes=None,
        )
    assert consumer.aborted


def test_morphology_crop_over_budget_raises_before_materialization():
    huge = np.ones((500, 500), dtype=bool)
    crop = ComponentCrop(label=1, bbox=(0, 0, 500, 500), mask=huge)
    with pytest.raises(MemoryBudgetExceeded) as excinfo:
        measure_components(
            (crop,),
            pixel_size_m=30.0,
            include_width=True,
            max_component_bytes=1024,
            window_id="window-0001",
        )
    err = excinfo.value
    assert err.window_id == "window-0001"
    assert err.budget_bytes == 1024
    assert err.mitigation is not None


def test_export_disabled_rejects_export_checkpoint_consumer():
    class PoolCheckpointConsumer:
        def consume(self, block: WindowMonthResult) -> None:
            pass

        def finalize(self) -> object:
            return None

        def abort(self) -> None:
            pass

    with pytest.raises(ValueError, match="export-disabled"):
        build_section_consumers(
            export_enabled=False,
            extra_consumers=[PoolCheckpointConsumer()],
        )


def test_small_grid_byte_identical_to_eager_reference():
    n_time, n_y, n_x = 6, 12, 12
    da_feature, valid_da = _cube(n_time=n_time, n_y=n_y, n_x=n_x, seed=6)
    config = _config(workers=1)
    section_area_km2 = float(n_y * n_x) * 900.0 / 1_000_000.0
    selected = {
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
        "occurrence",
        "refuge_area",
        "apsec",
    }

    eager_rows = analyze_section_rows(
        da_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=config,
        valid_obs=valid_da,
        selected_ids=selected,
    )

    lazy_feature, lazy_valid = _cube(
        n_time=n_time,
        n_y=n_y,
        n_x=n_x,
        seed=6,
        chunks=(1, 6, 6),
    )
    lazy_rows = analyze_section_rows(
        lazy_feature,
        section="AOI",
        section_area_km2=section_area_km2,
        pixel_size_m=30.0,
        config=config,
        valid_obs=lazy_valid,
        selected_ids=selected,
    )

    assert len(eager_rows) == len(lazy_rows) == n_time
    for eager_row, lazy_row in zip(eager_rows, lazy_rows):
        for key in eager_row:
            a, b = eager_row[key], lazy_row[key]
            if isinstance(a, float) and np.isnan(a):
                assert isinstance(b, float) and np.isnan(b)
            else:
                assert a == b, f"mismatch at {key}"


def test_one_fused_compute_per_window_materialization():
    da_feature, valid_da = _cube(
        n_time=1,
        n_y=8,
        n_x=8,
        seed=7,
        chunks=(1, 4, 4),
    )
    compute_calls: list[int] = []
    real_compute = da.compute

    def counting_compute(*args, **kwargs):
        compute_calls.append(len(args))
        return real_compute(*args, **kwargs)

    with mock.patch.object(da, "compute", side_effect=counting_compute):
        stream_month_windows(
            da_feature,
            valid_da,
            time_index=0,
            timestamp=pd.Timestamp(da_feature["time"].values[0]),
            windows=(AnalysisWindow(window_id="window-0001", bbox=(0, 0, 8, 8)),),
            consumers=[_SpyConsumer()],
            budget_bytes=10_000_000,
            pixel_size_m=30.0,
            connectivity=8,
            min_patch_pixels=1,
            want_patches=False,
            want_width=False,
            analysis_mask_np=None,
            local_label_threshold_bytes=None,
        )
    assert compute_calls == [2]


def test_measured_patch_bundle_validation():
    with pytest.raises(ValueError, match="cannot hold both"):
        MeasuredPatchBundle(
            properties=(),
            labels=np.zeros((2, 2), dtype=np.int32),
            label_checkpoint=LabelCheckpointRef(path="x", count=0, shape=(2, 2)),
        )


def test_metric_partial_carries_additive_payload():
    partial = MetricPartial(name="coverage", payload={"valid_pixels": 3, "total_pixels": 9})
    assert partial.payload["valid_pixels"] == 3
