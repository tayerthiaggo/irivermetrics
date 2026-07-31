"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.end_to_end_workflow import (
    CandidateSpec,
    DeferredCaseSpec,
    FITZROY_CASE,
    GILBERT_CASE,
    LARGE_CATCHMENT_CASE,
    RealCaseSpec,
    run_end_to_end_matrix,
    write_end_to_end_baseline,
)

__all__ = [
    "CandidateSpec",
    "DeferredCaseSpec",
    "FITZROY_CASE",
    "GILBERT_CASE",
    "LARGE_CATCHMENT_CASE",
    "RealCaseSpec",
    "run_end_to_end_matrix",
    "write_end_to_end_baseline",
]
