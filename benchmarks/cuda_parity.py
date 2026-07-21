"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.cuda_parity import (
    DEFAULT_CASES,
    ParitySpec,
    run_cuda_parity,
    write_cuda_parity,
)

__all__ = ["DEFAULT_CASES", "ParitySpec", "run_cuda_parity", "write_cuda_parity"]
