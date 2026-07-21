"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.numba_kernels import (
    DEFAULT_CASES,
    BenchmarkCase,
    gate_evidence_from_report,
    run_numba_benchmark,
    write_numba_benchmark,
)

__all__ = [
    "DEFAULT_CASES",
    "BenchmarkCase",
    "gate_evidence_from_report",
    "run_numba_benchmark",
    "write_numba_benchmark",
]
