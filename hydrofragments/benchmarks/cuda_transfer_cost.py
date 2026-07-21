"""Wall-time (including host<->device transfer) benchmark: CPU vs CUDA.

Companion to :mod:`hydrofragments.benchmarks.cuda_parity`. Where the parity
harness answers "does CUDA compute the same answer as CPU", this harness
answers "is CUDA actually faster once transfer cost is included, and at what
input size does that become true (the crossover size)". Both answers are
required evidence before :func:`hydrofragments.compute.capabilities.detect_capabilities`
will graduate a stage from ``CUDA_CANDIDATE_STAGES`` into ``enabled_cuda_stages``.

Import-safe without CuPy (mirrors ``CUDABackend.__init__``'s lazy
``importlib.import_module("cupy")`` pattern) so this module can be imported
and invoked in CPU-only CI without a GPU runner; it will typically not
execute its real CUDA path there.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
import time
from typing import Any, Iterable

import numpy as np

from hydrofragments.compute.backends.cpu import CPUBackend
from hydrofragments.compute.capabilities import CUDA_CANDIDATE_STAGES

DEFAULT_SIZES: tuple[int, ...] = (32, 64, 128, 256)
DEFAULT_REPEATS = 3

_STAGE_METHOD: dict[str, str] = {
    "valid_counts": "valid_counts",
    "monthly_reduction": "wet_counts",
    "occurrence": "mean",
}


def _synthetic_arrays(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shape = (12, size, size)
    water = rng.random(shape) < 0.3
    valid_obs = rng.random(shape) >= 0.05
    return water, valid_obs


def _time_call(fn, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(min(samples))


def _measure_stage_at_size(
    stage: str,
    size: int,
    cuda_module: Any,
    *,
    seed: int,
    repeats: int,
) -> dict[str, Any]:
    from hydrofragments.compute.backends.cuda import CUDABackend

    method_name = _STAGE_METHOD.get(stage)
    water, valid_obs = _synthetic_arrays(size, seed)
    cpu_backend = CPUBackend()
    cuda_backend = CUDABackend(cupy_module=cuda_module)

    if method_name is None:
        return {
            "size": size,
            "cpu_wall_seconds": 0.0,
            "cuda_wall_seconds_incl_transfer": 0.0,
            "cuda_transfer_seconds": None,
            "note": "no CUDABackend method implemented for this candidate stage yet",
        }

    def _cpu_call() -> None:
        if method_name == "mean":
            cpu_backend.mean(water.astype("float32"), axis=0)
        elif method_name == "valid_counts":
            cpu_backend.valid_counts(valid_obs, axis=0)
        else:
            cpu_backend.wet_counts(water, valid_obs, axis=0)

    def _cuda_call_incl_transfer() -> None:
        # Transfer cost is incurred implicitly: CUDABackend.asarray()s its
        # inputs from host arrays on every call, and reading a scalar/array
        # result back below materializes it on host again (get() when the
        # module provides it, else a plain array already on host for the
        # numpy CI shim).
        if method_name == "mean":
            result = cuda_backend.mean(water.astype("float32"), axis=0)
        elif method_name == "valid_counts":
            result = cuda_backend.valid_counts(valid_obs, axis=0)
        else:
            result = cuda_backend.wet_counts(water, valid_obs, axis=0)
        if hasattr(result, "get"):
            result.get()

    cpu_seconds = _time_call(_cpu_call, repeats)
    cuda_seconds = _time_call(_cuda_call_incl_transfer, repeats)

    return {
        "size": size,
        "cpu_wall_seconds": cpu_seconds,
        "cuda_wall_seconds_incl_transfer": cuda_seconds,
        "cuda_transfer_seconds": None,
        "note": None,
    }


def _crossover_size(measurements: list[dict[str, Any]]) -> int | None:
    """Smallest swept size at which CUDA wall time (incl. transfer) beats CPU."""

    for measurement in sorted(measurements, key=lambda item: item["size"]):
        if measurement["note"] is not None:
            continue
        if measurement["cuda_wall_seconds_incl_transfer"] < measurement["cpu_wall_seconds"]:
            return int(measurement["size"])
    return None


def _net_speedup_at_max_size(measurements: list[dict[str, Any]]) -> float | None:
    implemented = [item for item in measurements if item["note"] is None]
    if not implemented:
        return None
    largest = max(implemented, key=lambda item: item["size"])
    if largest["cuda_wall_seconds_incl_transfer"] <= 0.0:
        return None
    return float(largest["cpu_wall_seconds"] / largest["cuda_wall_seconds_incl_transfer"])


def run_cuda_transfer_cost(
    *,
    stages: Iterable[str] = CUDA_CANDIDATE_STAGES,
    sizes: Iterable[int] = DEFAULT_SIZES,
    repeats: int = DEFAULT_REPEATS,
    seed: int = 3101,
    cupy_module: Any | None = None,
) -> dict[str, Any]:
    """Measure CPU vs CUDA (incl. transfer) wall time across input sizes.

    Returns a clean "skipped" result (no exception) when CuPy is not
    importable, so this harness is always safe to invoke in CPU-only CI.
    ``cupy_module`` allows tests (and, on a GPU host, real CuPy) to supply
    the array module explicitly.
    """

    resolved_module = cupy_module
    if resolved_module is None:
        try:
            resolved_module = importlib.import_module("cupy")
        except Exception as error:  # optional dependency and runtime failures
            return {
                "schema_version": "1.0.0",
                "baseline": "cuda_transfer_cost",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cupy_available": False,
                "skipped": True,
                "skip_reason": f"CuPy unavailable: {error}",
                "cases": [],
                "stages": [],
            }

    resolved_stages = tuple(stages)
    resolved_sizes = tuple(sizes)

    stages_out = []
    for stage in resolved_stages:
        measurements = [
            _measure_stage_at_size(
                stage, size, resolved_module, seed=seed, repeats=repeats
            )
            for size in resolved_sizes
        ]
        stages_out.append(
            {
                "stage": stage,
                "measurements": measurements,
                "crossover_size": _crossover_size(measurements),
                "net_speedup_at_max_size": _net_speedup_at_max_size(measurements),
            }
        )

    return {
        "schema_version": "1.0.0",
        "baseline": "cuda_transfer_cost",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cupy_available": True,
        "skipped": False,
        "skip_reason": None,
        "sizes": list(resolved_sizes),
        "repeats": repeats,
        "stages": stages_out,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# CUDA transfer-cost benchmark",
        "",
        "Wall time (including host<->device transfer) for CUDA candidate stages, CPU vs CUDA.",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Created: `{payload['created_at']}`",
        f"- CuPy available: `{payload['cupy_available']}`",
    ]
    if payload["skipped"]:
        lines.append(f"- Skipped: `{payload['skip_reason']}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "",
            "| Stage | Crossover size | Net speedup @ max size |",
            "| --- | ---: | ---: |",
        ]
    )
    for stage in payload["stages"]:
        lines.append(
            f"| {stage['stage']} | {stage['crossover_size']} | "
            f"{stage['net_speedup_at_max_size']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_cuda_transfer_cost(
    output_dir: str | Path,
    *,
    stages: Iterable[str] = CUDA_CANDIDATE_STAGES,
    sizes: Iterable[int] = DEFAULT_SIZES,
    repeats: int = DEFAULT_REPEATS,
    cupy_module: Any | None = None,
) -> dict[str, Any]:
    """Run the transfer-cost harness and write JSON plus Markdown reports."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_cuda_transfer_cost(
        stages=stages, sizes=sizes, repeats=repeats, cupy_module=cupy_module
    )
    json_path = target / "cuda_transfer_cost.json"
    markdown_path = target / "cuda_transfer_cost.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_files"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return payload


if __name__ == "__main__":
    result = write_cuda_transfer_cost(Path(__file__).parent / "results")
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, indent=2))


__all__ = [
    "DEFAULT_SIZES",
    "DEFAULT_REPEATS",
    "run_cuda_transfer_cost",
    "write_cuda_transfer_cost",
]
