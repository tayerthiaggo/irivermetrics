"""Numba-vs-baseline parity and speedup benchmark: candidate JIT kernels.

Twin of :mod:`hydrofragments.benchmarks.cuda_parity` and
:mod:`hydrofragments.benchmarks.cuda_transfer_cost` combined into a single
harness. The CUDA gate needs two separate reports because CUDA has a real
host<->device transfer cost that only shows up once data crosses the PCIe
bus; a same-process ``@njit`` call has no such concept -- the only costs are
(a) numeric correctness and (b) wall time, including the one-time JIT
warm-up/compilation cost on first call. This module measures both in one
pass per kernel.

Import-safe without Numba installed (mirrors
:class:`hydrofragments.compute.backends.cuda.CUDABackend`'s lazy
``importlib.import_module`` pattern and
:mod:`hydrofragments.metrics.clustering_numba`'s own lazy-import contract):
it self-skips cleanly with ``skipped: true`` rather than raising, so it is
always safe to invoke in CPU-only, Numba-absent CI.

Output is evidence input for
:func:`hydrofragments.compute.capabilities.gated_kernels_from_baseline` --
a kernel only graduates from ``NUMBA_CANDIDATE_KERNELS`` to
``numba_enabled_kernels`` once a hand-curated
``benchmarks/results/numba_baseline.json`` (mirroring
``cuda_baseline.json``'s schema) records both ``parity_pass: true`` and
``speedup_pass: true`` for that kernel, distilled from this harness's
``numba_kernels.json``/``.md`` report pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time
from typing import Any, Callable, Iterable

import numpy as np

from hydrofragments.compute.capabilities import NUMBA_CANDIDATE_KERNELS
from hydrofragments.metrics.clustering_numba import (
    _build_numba_kernel,
    _pure_python_inter_pool_gaps,
    numba_available,
)


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    size: int
    seed: int
    wet_fraction: float


DEFAULT_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(case_id="N0_small", size=64, seed=4101, wet_fraction=0.5),
    BenchmarkCase(case_id="N0_medium", size=2_000, seed=4102, wet_fraction=0.3),
    BenchmarkCase(case_id="N0_large", size=50_000, seed=4103, wet_fraction=0.2),
)

DEFAULT_REPEATS = 3

# Only "inter_pool_gap_runs" has a live Numba prototype today -- see
# hydrofragments/compute/capabilities.py's NUMBA_CANDIDATE_KERNELS docstring
# for why the EDT/width candidate is absent.
_KERNEL_BASELINE: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "inter_pool_gap_runs": _pure_python_inter_pool_gaps,
}


def _synthetic_case(case: BenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    wet = rng.random(case.size) < case.wet_fraction
    lengths = rng.uniform(1.0, 50.0, size=case.size)
    return wet, lengths


def _time_call(fn: Callable[[], Any], repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(min(samples))


def _run_kernel(
    kernel_name: str,
    numba_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    cases: Iterable[BenchmarkCase],
    repeats: int,
    cold_probe: Callable[[], Callable[[np.ndarray, np.ndarray], np.ndarray]] | None = None,
) -> dict[str, Any]:
    baseline_fn = _KERNEL_BASELINE[kernel_name]
    resolved_cases = tuple(cases)

    cases_out = []
    for case in resolved_cases:
        wet, lengths = _synthetic_case(case)
        baseline_result = np.asarray(baseline_fn(wet, lengths), dtype=np.float64)
        numba_result = np.asarray(numba_kernel(wet, lengths), dtype=np.float64)

        if baseline_result.shape != numba_result.shape:
            max_abs_diff = float("inf")
        else:
            max_abs_diff = float(np.max(np.abs(baseline_result - numba_result))) if baseline_result.size else 0.0

        cases_out.append(
            {
                "case_id": case.case_id,
                "size": case.size,
                "parity_pass": bool(max_abs_diff == 0.0),
                "max_abs_diff": max_abs_diff,
            }
        )

    all_parity_pass = all(case["parity_pass"] for case in cases_out)

    # Speedup: measured on the largest case, reporting cold (first-ever
    # call, JIT compile included) and warm (subsequent, post-compile) wall
    # time separately -- matching cpu_baseline.py's warmup: bool convention,
    # but here both numbers are always reported rather than gated behind a
    # flag, since the cold-start cost is itself evidence a reader needs (a
    # kernel that's only faster after amortizing a 200ms compile cost over
    # millions of calls is a different claim than one that's faster
    # immediately).
    #
    # IMPORTANT: cold timing must run before *any* other call to
    # numba_kernel -- Numba caches the compiled dispatcher on the function
    # object itself (not per input array), so if the parity loop above
    # already called this same kernel object, "cold" would silently measure
    # an already-warm call. cold_probe isolates this by resolving a fresh,
    # never-yet-called dispatcher via kernel_factory when one is supplied
    # (run_numba_benchmark's real path); tests using kernel_override without
    # a factory get a best-effort cold measurement against the (possibly
    # already-warm) override instead, since a bare callable has no
    # "recompile fresh" hook.
    largest = max(resolved_cases, key=lambda item: item.size)
    wet, lengths = _synthetic_case(largest)

    cold_kernel = cold_probe() if cold_probe is not None else numba_kernel
    cold_start = time.perf_counter()
    cold_kernel(wet, lengths)
    cold_wall_seconds = float(time.perf_counter() - cold_start)

    warm_baseline_seconds = _time_call(lambda: baseline_fn(wet, lengths), repeats)
    warm_numba_seconds = _time_call(lambda: numba_kernel(wet, lengths), repeats)
    speedup_pass = bool(
        warm_numba_seconds > 0.0 and warm_numba_seconds < warm_baseline_seconds
    )

    return {
        "kernel": kernel_name,
        "cases": cases_out,
        "parity_pass": all_parity_pass,
        "warm_speedup": {
            "measured_at_size": largest.size,
            "cold_wall_seconds": cold_wall_seconds,
            "warm_baseline_seconds": warm_baseline_seconds,
            "warm_numba_seconds": warm_numba_seconds,
            "speedup_ratio": (
                warm_baseline_seconds / warm_numba_seconds
                if warm_numba_seconds > 0.0
                else None
            ),
        },
        "speedup_pass": speedup_pass,
    }


def run_numba_benchmark(
    *,
    cases: Iterable[BenchmarkCase] = DEFAULT_CASES,
    repeats: int = DEFAULT_REPEATS,
    kernel_override: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run parity + warm/cold speedup checks for every Numba candidate kernel.

    Returns a clean "skipped" result (no exception) when Numba is not
    importable/usable, so this harness is always safe to invoke in
    Numba-absent CI. ``kernel_override`` lets tests inject a stand-in
    callable (e.g. a deliberately-wrong function) to exercise the
    parity-failure reporting path without needing Numba to misbehave.
    """
    if kernel_override is None:
        if not numba_available():
            return {
                "schema_version": "1.0.0",
                "baseline": "numba_kernels",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "numba_available": False,
                "skipped": True,
                "skip_reason": "Numba unavailable: import or JIT compilation failed",
                "all_parity_pass": None,
                "kernels": [],
            }

        from hydrofragments.metrics.clustering_numba import _compute_inter_pool_gaps_numba

        kernel_fn = _compute_inter_pool_gaps_numba
        # _compute_inter_pool_gaps_numba caches its compiled dispatcher at
        # module scope (hydrofragments.metrics.clustering_numba._NUMBA_KERNEL),
        # so once the parity-checking pass below has called it even once,
        # every later call -- including one meant to measure "cold" JIT
        # compile time -- would actually be warm. _build_numba_kernel()
        # constructs a brand-new, never-yet-called @njit dispatcher on
        # every call, giving _run_kernel a true cold instance to time that
        # is fully independent of the cached one used for parity/warm
        # timings.
        cold_probe = _build_numba_kernel
    else:
        kernel_fn = kernel_override
        cold_probe = None

    resolved_cases = tuple(cases)
    kernels_out = [
        _run_kernel(
            kernel_name,
            kernel_fn,
            cases=resolved_cases,
            repeats=repeats,
            cold_probe=cold_probe,
        )
        for kernel_name in NUMBA_CANDIDATE_KERNELS
    ]
    all_parity_pass = all(kernel["parity_pass"] for kernel in kernels_out)

    return {
        "schema_version": "1.0.0",
        "baseline": "numba_kernels",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "numba_available": True,
        "skipped": False,
        "skip_reason": None,
        "candidate_kernels": list(NUMBA_CANDIDATE_KERNELS),
        "all_parity_pass": all_parity_pass,
        "kernels": kernels_out,
    }


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Numba kernel benchmark",
        "",
        "Numeric-parity and warm/cold speedup evidence for Numba candidate kernels "
        "against the pure-Python/NumPy reference.",
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Created: `{payload['created_at']}`",
        f"- Numba available: `{payload['numba_available']}`",
    ]
    if payload["skipped"]:
        lines.append(f"- Skipped: `{payload['skip_reason']}`")
        lines.append("")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            f"- All kernels pass parity: `{payload['all_parity_pass']}`",
            "",
            "| Kernel | parity_pass | speedup_pass | warm baseline (s) | warm numba (s) | speedup ratio | cold (s) |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for kernel in payload["kernels"]:
        speedup = kernel["warm_speedup"]
        ratio = speedup["speedup_ratio"]
        lines.append(
            f"| {kernel['kernel']} | {kernel['parity_pass']} | {kernel['speedup_pass']} | "
            f"{speedup['warm_baseline_seconds']:.6f} | {speedup['warm_numba_seconds']:.6f} | "
            f"{ratio if ratio is None else f'{ratio:.2f}'} | {speedup['cold_wall_seconds']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def gate_evidence_from_report(payload: dict[str, Any]) -> dict[str, Any]:
    """Distill this harness's raw report into the compact gate-evidence shape.

    :func:`hydrofragments.compute.capabilities.gated_kernels_from_baseline`
    reads ``{"kernels": {kernel_name: {"parity_pass", "speedup_pass"}}}`` --
    a dict keyed by kernel name, one boolean pair per kernel. This harness's
    own raw report (this function's input) keeps ``kernels`` as a *list* of
    per-kernel detail records (cases, timings), matching the CUDA harnesses'
    ``cuda_parity.json``/``cuda_transfer_cost.json`` list convention. This
    function performs the same curatorial distillation
    ``docs/acceleration.md`` describes a human doing by hand for
    ``cuda_baseline.json`` -- here it is offered as a helper (still requires
    a human to review and commit the result) rather than purely manual,
    since with no separate transfer-cost report to cross-reference there is
    nothing extra for a human to reconcile beyond what this function does.
    """
    if payload.get("skipped"):
        return {
            "schema_version": "1.0.0",
            "baseline": "numba_gate_evidence",
            "created_at": payload.get("created_at"),
            "kernels": {},
        }
    return {
        "schema_version": "1.0.0",
        "baseline": "numba_gate_evidence",
        "created_at": payload.get("created_at"),
        "kernels": {
            kernel["kernel"]: {
                "parity_pass": kernel["parity_pass"],
                "speedup_pass": kernel["speedup_pass"],
                "speedup_ratio": kernel["warm_speedup"]["speedup_ratio"],
            }
            for kernel in payload["kernels"]
        },
    }


def write_numba_benchmark(
    output_dir: str | Path,
    *,
    cases: Iterable[BenchmarkCase] = DEFAULT_CASES,
    repeats: int = DEFAULT_REPEATS,
) -> dict[str, Any]:
    """Run the benchmark harness and write machine-readable JSON plus Markdown.

    Writes the raw per-case/per-timing report to ``numba_kernels.json`` /
    ``.md`` in ``output_dir`` -- filename matches this module's name, same
    convention as ``cuda_parity.py`` writing ``cuda_parity.json``. This is
    evidence *input* (analogous to ``cuda_parity.json`` +
    ``cuda_transfer_cost.json`` combined, since a same-process JIT call has
    no separate transfer-cost report to produce), not the compact gate file
    ``detect_capabilities`` reads directly -- that file is
    ``benchmarks/results/numba_baseline.json`` (repo root), produced by
    calling :func:`gate_evidence_from_report` on this function's result (or
    hand-writing the equivalent) after a human reviews the report, per
    ``docs/acceleration.md``. Keeping these two files under different
    basenames (``numba_kernels.json`` vs ``numba_baseline.json``) avoids the
    name collision CUDA's two-report/one-gate-file split doesn't have to
    worry about, since CUDA's raw reports and gate file already have
    distinct names (``cuda_parity.json`` / ``cuda_transfer_cost.json`` vs
    ``cuda_baseline.json``).
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_numba_benchmark(cases=cases, repeats=repeats)
    json_path = target / "numba_kernels.json"
    markdown_path = target / "numba_kernels.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_files"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return payload


__all__ = [
    "BenchmarkCase",
    "DEFAULT_CASES",
    "DEFAULT_REPEATS",
    "run_numba_benchmark",
    "write_numba_benchmark",
]
