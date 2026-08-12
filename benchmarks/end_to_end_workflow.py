"""Command-facing compatibility import for the packaged benchmark harness."""

from hydrofragments.benchmarks.end_to_end_workflow import (
    CandidateSpec,
    DeferredCaseSpec,
    FITZROY_CASE,
    GILBERT_CASE,
    LARGE_CATCHMENT_CASE,
    RealCaseSpec,
    SPATIAL_EXPORT_SCENARIOS,
    run_end_to_end_matrix,
    run_spatial_export_matrix,
    write_end_to_end_baseline,
    write_spatial_export_baseline,
)

__all__ = [
    "CandidateSpec",
    "DeferredCaseSpec",
    "FITZROY_CASE",
    "GILBERT_CASE",
    "LARGE_CATCHMENT_CASE",
    "RealCaseSpec",
    "SPATIAL_EXPORT_SCENARIOS",
    "run_end_to_end_matrix",
    "run_spatial_export_matrix",
    "write_end_to_end_baseline",
    "write_spatial_export_baseline",
]
