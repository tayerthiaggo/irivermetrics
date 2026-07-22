"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.cuda_transfer_cost import (
    DEFAULT_REPEATS,
    DEFAULT_SIZES,
    run_cuda_transfer_cost,
    write_cuda_transfer_cost,
)

__all__ = [
    "DEFAULT_REPEATS",
    "DEFAULT_SIZES",
    "run_cuda_transfer_cost",
    "write_cuda_transfer_cost",
]
