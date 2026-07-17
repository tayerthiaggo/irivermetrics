"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.cpu_baseline import (
    BenchmarkSpec,
    DEFAULT_CASES,
    run_cpu_baseline,
    write_cpu_baseline,
)

__all__ = ["BenchmarkSpec", "DEFAULT_CASES", "run_cpu_baseline", "write_cpu_baseline"]
