"""Immutable execution policy resolved before pipeline assembly."""

from __future__ import annotations

from dataclasses import dataclass, field

from hydrofragments.compute.capabilities import BackendCapabilities


class ComputePolicyError(ValueError):
    """Raised when execution policy cannot describe a safe M4 run."""


# Safe default when no capabilities are supplied: no CUDA stages enabled,
# cuda_available False. This preserves the pre-existing blanket refusal for
# every caller that doesn't opt into the `capabilities` parameter.
_NO_CUDA_EVIDENCE = BackendCapabilities.cpu_only(
    "no capabilities supplied to ComputePolicy"
)


@dataclass(frozen=True)
class ComputePolicy:
    """Execution-only settings for lazy temporal stages and their checkpoint.

    ``capabilities`` is optional, evidence-gated CUDA capability information
    (see :mod:`hydrofragments.compute.capabilities`). When omitted (the
    default), ``accelerator="cuda"`` is always refused -- the same blanket
    "CUDA is not certified" behavior this policy has always had. Passing a
    ``capabilities`` whose ``enabled_cuda_stages`` is non-empty is required
    to unlock ``accelerator="cuda"``; this mirrors, at the whole-policy
    granularity, the same evidence gate that
    :func:`hydrofragments.compute.capabilities.resolve_execution_plan`
    already applies per-stage.
    """

    target_chunk_bytes: int = 128 * 1024 * 1024
    live_array_multiplier: float = 4.0
    checkpoint: str = "zarr"
    accelerator: str = "none"
    scheduler: str | None = None
    capabilities: BackendCapabilities | None = None
    actual_backend: str = field(init=False, default="cpu")

    def __post_init__(self) -> None:
        if self.checkpoint not in {"none", "persist", "zarr"}:
            raise ComputePolicyError(
                "checkpoint must be one of: none, persist, zarr"
            )
        resolved_capabilities = self.capabilities
        if resolved_capabilities is None:
            resolved_capabilities = _NO_CUDA_EVIDENCE
            object.__setattr__(self, "capabilities", resolved_capabilities)
        if self.accelerator == "cuda" and not resolved_capabilities.enabled_cuda_stages:
            raise ComputePolicyError(
                "CUDA execution is not certified for the Milestone 4 pipeline"
            )


__all__ = ["ComputePolicy", "ComputePolicyError"]
