"""Deterministic CPU reference benchmark for the certified reduction stages.

This module is intentionally orchestration-only. It owns materialisation and
timing, keeps the scientific kernels unchanged, and never imports optional GPU
packages. Its output is the prerequisite evidence for the optional CUDA
milestone, not a promise of GPU speedup.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any, Iterable

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.compute import ComputePolicy
from hydrofragments.config import HydroConfig
from hydrofragments.metrics import compute_apsec, compute_occurrence
from hydrofragments.pipeline import assemble_monthly_pipeline


@dataclass(frozen=True)
class BenchmarkSpec:
    dataset_id: str
    shape: tuple[int, int, int]
    chunks: tuple[int, int, int]
    seed: int
    wet_fraction: float
    missing_fraction: float


DEFAULT_CASES: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        dataset_id="B0_analytic",
        shape=(12, 32, 32),
        chunks=(4, 16, 16),
        seed=1201,
        wet_fraction=0.35,
        missing_fraction=0.10,
    ),
    BenchmarkSpec(
        dataset_id="B0_fragmentation",
        shape=(24, 64, 64),
        chunks=(6, 16, 16),
        seed=1202,
        wet_fraction=0.10,
        missing_fraction=0.05,
    ),
    BenchmarkSpec(
        dataset_id="B0_scale",
        shape=(12, 128, 128),
        chunks=(4, 32, 32),
        seed=1203,
        wet_fraction=0.50,
        missing_fraction=0.02,
    ),
)


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _synthetic_case(spec: BenchmarkSpec) -> tuple[xr.DataArray, xr.DataArray]:
    rng = np.random.default_rng(spec.seed)
    water_values = rng.random(spec.shape) < spec.wet_fraction
    valid_values = rng.random(spec.shape) >= spec.missing_fraction
    times = pd.date_range("2020-01-01", periods=spec.shape[0], freq="MS")
    coords = {"time": times, "y": np.arange(spec.shape[1]), "x": np.arange(spec.shape[2])}
    dims = ("time", "y", "x")
    water = xr.DataArray(
        da.from_array(water_values, chunks=spec.chunks), dims=dims, coords=coords
    )
    valid_obs = xr.DataArray(
        da.from_array(valid_values, chunks=spec.chunks), dims=dims, coords=coords
    )
    return water, valid_obs


def _config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "validity": {"min_valid_obs": 1},
        }
    )


def _checksum(*arrays: Any) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _summary(values: Iterable[float]) -> dict[str, Any]:
    samples = [float(value) for value in values]
    return {
        "runs": len(samples),
        "median_seconds": float(statistics.median(samples)),
        "p95_seconds": float(np.percentile(samples, 95)),
    }


def _run_case(spec: BenchmarkSpec, *, repeats: int, warmup: bool) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    water, valid_obs = _synthetic_case(spec)
    config = _config()
    policy = ComputePolicy(checkpoint="none", accelerator="none")
    timings: dict[str, list[float]] = {
        "assemble_monthly": [],
        "monthly_reduction": [],
        "occurrence": [],
        "apsec": [],
    }
    graph_metrics = {"graph_task_count": 0, "graph_bytes": 0}

    def execute() -> tuple[str, int, int]:
        start = time.perf_counter()
        pipeline = assemble_monthly_pipeline(
            water,
            valid_obs,
            input_cadence="monthly",
            monthly_composite="supplied",
            composite_owner="caller",
            policy=policy,
        )
        timings["assemble_monthly"].append(time.perf_counter() - start)
        graph = pipeline.dataset.__dask_graph__()
        graph_metrics["graph_task_count"] = 0 if graph is None else len(graph)
        graph_metrics["graph_bytes"] = 0 if graph is None else len(str(graph).encode("utf-8"))

        start = time.perf_counter()
        with dask.config.set(scheduler="single-threaded"):
            monthly = pipeline.dataset.compute()
        timings["monthly_reduction"].append(time.perf_counter() - start)

        start = time.perf_counter()
        occurrence = compute_occurrence(monthly, config=config)
        occurrence_values = occurrence.occurrence.compute().values
        valid_count_values = occurrence.valid_count.compute().values
        timings["occurrence"].append(time.perf_counter() - start)

        start = time.perf_counter()
        apsec = compute_apsec(
            monthly,
            a_ref_m2=float(spec.shape[1] * spec.shape[2]),
            cell_area_m2=1.0,
            config=config,
        )
        timings["apsec"].append(time.perf_counter() - start)

        apsec_values = np.asarray([record.value for record in apsec], dtype=np.float64)
        checksum = _checksum(
            monthly["water"].values,
            monthly["valid_obs"].values,
            occurrence_values,
            valid_count_values,
            apsec_values,
        )
        return checksum, int(valid_count_values.sum()), len(apsec)

    if warmup:
        execute()
        for values in timings.values():
            values.clear()
    checksum, valid_count_sum, apsec_months = execute()
    for _ in range(repeats - 1):
        next_checksum, next_valid_count_sum, next_apsec_months = execute()
        if (next_checksum, next_valid_count_sum, next_apsec_months) != (
            checksum,
            valid_count_sum,
            apsec_months,
        ):
            raise RuntimeError(f"non-deterministic CPU output for {spec.dataset_id}")

    stage_names = tuple(timings)
    return {
        "dataset_id": spec.dataset_id,
        "shape": list(spec.shape),
        "chunks": [list(axis) for axis in da.core.normalize_chunks(spec.chunks, spec.shape)],
        "repeats": repeats,
        "output_checksum": checksum,
        "valid_count_sum": valid_count_sum,
        "apsec_months": apsec_months,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "gpu": None,
            "driver": None,
        },
        "dask": {
            "scheduler": "single-threaded",
            "workers": 1,
            "threads_per_worker": 1,
            **graph_metrics,
            "scheduler_overhead_fraction": None,
            "spill_bytes": 0,
        },
        "memory": {
            "peak_rss_bytes": None,
            "peak_managed_bytes": None,
            "peak_vram_bytes": None,
            "vram_limit_bytes": None,
        },
        "io": {
            "bytes_read": None,
            "bytes_written": None,
            "host_device_transfer_bytes": 0,
            "host_device_transfer_seconds": 0.0,
        },
        "backend_actual_by_stage": {name: "cpu" for name in stage_names},
        "stages": [
            {
                "stage": name,
                "backend_planned": "cpu",
                "backend_actual": "cpu",
                "host_device_transfer_bytes": 0,
                "host_device_transfer_seconds": 0.0,
                "graph_task_count": graph_metrics["graph_task_count"],
                "graph_bytes": graph_metrics["graph_bytes"],
                "peak_rss_bytes": None,
                "peak_vram_bytes": None,
                "vram_limit_bytes": None,
                "scheduler_overhead_fraction": None,
                "timing_seconds": _summary(timings[name]),
            }
            for name in stage_names
        ],
    }


def run_cpu_baseline(
    *,
    cases: Iterable[BenchmarkSpec] = DEFAULT_CASES,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run deterministic CPU reference cases and return a JSON-safe mapping."""

    resolved_cases = tuple(cases)
    return {
        "schema_version": "1.0.0",
        "baseline": "cpu_reference",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "backend_planned": "cpu",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _version("numpy"),
            "dask": _version("dask"),
            "xarray": _version("xarray"),
            "cupy_imported": "cupy" in sys.modules,
            "scheduler": "single-threaded",
            "workers": 1,
            "threads_per_worker": 1,
        },
        "cases": [
            _run_case(spec, repeats=repeats, warmup=warmup)
            for spec in resolved_cases
        ],
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# CPU reference benchmark baseline",
        "",
        "Deterministic CPU baseline for certified array reductions. CUDA is not enabled.",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Created: `{payload['created_at']}`",
        f"- Backend planned: `{payload['backend_planned']}`",
        "",
        "| Dataset | Stage | backend_actual | Median seconds | p95 seconds | Graph tasks | Peak RSS | Peak VRAM | Transfer bytes |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in payload["cases"]:
        for stage in case["stages"]:
            timing = stage["timing_seconds"]
            lines.append(
                f"| {case['dataset_id']} | {stage['stage']} | "
                f"{stage['backend_actual']} | {timing['median_seconds']:.6f} | "
                f"{timing['p95_seconds']:.6f} | {stage['graph_task_count']} | "
                f"{stage['peak_rss_bytes'] or 'n/a'} | {stage['peak_vram_bytes'] or 'n/a'} | "
                f"{stage['host_device_transfer_bytes']} |"
            )
    lines.extend(
        [
            "",
            "Checksums are scientific-output evidence; timing values are host-specific.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_cpu_baseline(
    output_dir: str | Path,
    *,
    cases: Iterable[BenchmarkSpec] = DEFAULT_CASES,
    repeats: int = 3,
    warmup: bool = True,
) -> dict[str, Any]:
    """Run baseline and write machine-readable JSON plus human Markdown."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_cpu_baseline(cases=cases, repeats=repeats, warmup=warmup)
    json_path = target / "cpu_baseline.json"
    markdown_path = target / "cpu_baseline.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_files"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return payload


__all__ = ["BenchmarkSpec", "DEFAULT_CASES", "run_cpu_baseline", "write_cpu_baseline"]
