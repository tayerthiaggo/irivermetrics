"""Execution policy and Dask chunk contracts."""

from hydrofragments.compute.chunks import (
    ChunkBudgetError,
    ChunkDiagnostics,
    validate_chunk_budget,
)
from hydrofragments.compute.policy import (
    ComputePolicy,
    ComputePolicyError,
)
from hydrofragments.compute.capabilities import (
    BackendCapabilities,
    CUDA_CANDIDATE_STAGES,
    CapabilityError,
    DEFAULT_STAGES,
    ExecutionPlan,
    FLOATING_TOLERANCES,
    detect_capabilities,
    gated_stages_from_baseline,
    resolve_execution_plan,
)

__all__ = [
    "ChunkBudgetError",
    "ChunkDiagnostics",
    "ComputePolicy",
    "ComputePolicyError",
    "BackendCapabilities",
    "CUDA_CANDIDATE_STAGES",
    "CapabilityError",
    "DEFAULT_STAGES",
    "ExecutionPlan",
    "FLOATING_TOLERANCES",
    "detect_capabilities",
    "gated_stages_from_baseline",
    "resolve_execution_plan",
    "validate_chunk_budget",
]
