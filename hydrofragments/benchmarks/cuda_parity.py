"""Numeric-parity benchmark harness: CPU reference vs CUDA candidate stages.

This module is intentionally import-safe without CuPy installed. It mirrors
:class:`hydrofragments.compute.backends.cuda.CUDABackend`'s lazy-import
pattern so it can run (and self-skip cleanly) in CPU-only CI. Its output is
evidence input for :func:`hydrofragments.compute.capabilities.detect_capabilities`
-- a stage only graduates from ``CUDA_CANDIDATE_STAGES`` to
``enabled_cuda_stages`` once a recorded baseline shows parity here *and*
net transfer-cost speedup in :mod:`hydrofragments.benchmarks.cuda_transfer_cost`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable

import numpy as np

from hydrofragments.compute.backends.cpu import CPUBackend
from hydrofragments.compute.capabilities import CUDA_CANDIDATE_STAGES, FLOATING_TOLERANCES


@dataclass(frozen=True)
class ParitySpec:
    dataset_id: str
    shape: tuple[int, int, int]
    seed: int
    wet_fraction: float
    missing_fraction: float
    dtype: str


DEFAULT_CASES: tuple[ParitySpec, ...] = (
    ParitySpec(
        dataset_id="P0_small",
        shape=(6, 16, 16),
        seed=2101,
        wet_fraction=0.35,
        missing_fraction=0.10,
        dtype="float32",
    ),
    ParitySpec(
        dataset_id="P0_medium",
        shape=(12, 64, 64),
        seed=2102,
        wet_fraction=0.20,
        missing_fraction=0.05,
        dtype="float32",
    ),
    ParitySpec(
        dataset_id="P0_large",
        shape=(12, 128, 128),
        seed=2103,
        wet_fraction=0.50,
        missing_fraction=0.02,
        dtype="float64",
    ),
)

# Only reduction stages with a live CUDABackend method are directly
# comparable today. sentinel_normalization/masks are candidate kernels
# without a CUDABackend implementation yet; they are recorded as
# "not_implemented" rather than silently skipped so the report stays
# truthful about coverage gaps.
_STAGE_METHOD: dict[str, str] = {
    "valid_counts": "valid_counts",
    "monthly_reduction": "wet_counts",
    "occurrence": "mean",
}


def _synthetic_arrays(spec: ParitySpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(spec.seed)
    water = rng.random(spec.shape) < spec.wet_fraction
    valid_obs = rng.random(spec.shape) >= spec.missing_fraction
    return water, valid_obs


def _run_stage(
    stage: str,
    cpu_backend: CPUBackend,
    cuda_backend: Any,
    water: np.ndarray,
    valid_obs: np.ndarray,
    dtype: str,
) -> dict[str, Any]:
    method_name = _STAGE_METHOD.get(stage)
    if method_name is None:
        return {
            "stage": stage,
            "parity_pass": None,
            "max_abs_diff": None,
            "tolerance": None,
            "note": "no CUDABackend method implemented for this candidate stage yet",
        }

    if method_name == "mean":
        cpu_value = getattr(cpu_backend, method_name)(water.astype(dtype), axis=0)
        cuda_value = getattr(cuda_backend, method_name)(water.astype(dtype), axis=0)
        tolerance = FLOATING_TOLERANCES[dtype]
    elif method_name == "valid_counts":
        cpu_value = getattr(cpu_backend, method_name)(valid_obs, axis=0)
        cuda_value = getattr(cuda_backend, method_name)(valid_obs, axis=0)
        tolerance = 0.0
    else:
        cpu_value = getattr(cpu_backend, method_name)(water, valid_obs, axis=0)
        cuda_value = getattr(cuda_backend, method_name)(water, valid_obs, axis=0)
        tolerance = 0.0

    cpu_array = np.asarray(cpu_value)
    cuda_array = np.asarray(cuda_value)
    if cpu_array.shape != cuda_array.shape:
        max_abs_diff = float("inf")
    else:
        max_abs_diff = float(np.max(np.abs(cpu_array.astype(np.float64) - cuda_array.astype(np.float64))))

    return {
        "stage": stage,
        "parity_pass": bool(max_abs_diff <= tolerance),
        "max_abs_diff": max_abs_diff,
        "tolerance": float(tolerance),
        "note": None,
    }


def _run_case(spec: ParitySpec, cuda_module: Any) -> dict[str, Any]:
    from hydrofragments.compute.backends.cuda import CUDABackend

    water, valid_obs = _synthetic_arrays(spec)
    cpu_backend = CPUBackend()
    cuda_backend = CUDABackend(cupy_module=cuda_module)

    stages = [
        _run_stage(stage, cpu_backend, cuda_backend, water, valid_obs, spec.dtype)
        for stage in CUDA_CANDIDATE_STAGES
    ]
    return {
        "dataset_id": spec.dataset_id,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "stages": stages,
    }


def run_cuda_parity(
    *,
    cases: Iterable[ParitySpec] = DEFAULT_CASES,
    cupy_module: Any | None = None,
) -> dict[str, Any]:
    """Run CPU-vs-CUDA numeric parity checks for every candidate stage.

    Returns a clean "skipped" result (no exception) when CuPy is not
    importable, so this harness is always safe to invoke in CPU-only CI.
    ``cupy_module`` allows tests to inject a stand-in (e.g. NumPy itself,
    mirroring :class:`CUDABackend`'s own test-time shim) without requiring a
    real GPU.
    """

    resolved_module = cupy_module
    if resolved_module is None:
        try:
            resolved_module = importlib.import_module("cupy")
        except Exception as error:  # optional dependency and runtime failures
            return {
                "schema_version": "1.0.0",
                "baseline": "cuda_parity",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cupy_available": False,
                "skipped": True,
                "skip_reason": f"CuPy unavailable: {error}",
                "all_pass": None,
                "cases": [],
            }

    resolved_cases = tuple(cases)
    cases_out = [_run_case(spec, resolved_module) for spec in resolved_cases]
    all_pass = all(
        stage["parity_pass"] is not False
        for case in cases_out
        for stage in case["stages"]
    )

    return {
        "schema_version": "1.0.0",
        "baseline": "cuda_parity",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cupy_available": True,
        "skipped": False,
        "skip_reason": None,
        "floating_tolerances": dict(FLOATING_TOLERANCES),
        "candidate_stages": list(CUDA_CANDIDATE_STAGES),
        "all_pass": all_pass,
        "cases": cases_out,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# CUDA parity benchmark",
        "",
        "Numeric-equality evidence for CUDA candidate stages against the CPU reference.",
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
            f"- All stages pass: `{payload['all_pass']}`",
            "",
            "| Dataset | Stage | parity_pass | max_abs_diff | tolerance |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    for case in payload["cases"]:
        for stage in case["stages"]:
            lines.append(
                f"| {case['dataset_id']} | {stage['stage']} | {stage['parity_pass']} | "
                f"{stage['max_abs_diff']} | {stage['tolerance']} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def write_cuda_parity(
    output_dir: str | Path,
    *,
    cases: Iterable[ParitySpec] = DEFAULT_CASES,
    cupy_module: Any | None = None,
) -> dict[str, Any]:
    """Run the parity harness and write machine-readable JSON plus Markdown."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_cuda_parity(cases=cases, cupy_module=cupy_module)
    json_path = target / "cuda_parity.json"
    markdown_path = target / "cuda_parity.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_files"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return payload


if __name__ == "__main__":
    result = write_cuda_parity(Path(__file__).parent / "results")
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


__all__ = ["ParitySpec", "DEFAULT_CASES", "run_cuda_parity", "write_cuda_parity"]
