"""Immutable execution policy resolved before pipeline assembly."""

from __future__ import annotations

from dataclasses import dataclass, field


class ComputePolicyError(ValueError):
    """Raised when execution policy cannot describe a safe M4 run."""


@dataclass(frozen=True)
class ComputePolicy:
    """Execution-only settings for lazy temporal stages and their checkpoint."""

    target_chunk_bytes: int = 128 * 1024 * 1024
    live_array_multiplier: float = 4.0
    checkpoint: str = "zarr"
    accelerator: str = "none"
    scheduler: str | None = None
    actual_backend: str = field(init=False, default="cpu")

    def __post_init__(self) -> None:
        if self.checkpoint not in {"none", "persist", "zarr"}:
            raise ComputePolicyError(
                "checkpoint must be one of: none, persist, zarr"
            )
        if self.accelerator == "cuda":
            raise ComputePolicyError(
                "CUDA execution is not certified for the Milestone 4 pipeline"
            )


DEFAULT_WORKER_MEMORY_FRACTION = 0.5
_MIN_WORKER_BUDGET_TARGET_BYTES = 1024


def resolve_worker_byte_budget(config, *, in_flight_slots: int = 1) -> int:
    """Derive per-slot admitted live bytes from compute policy fields."""

    target = (
        config.compute.target_chunk_bytes
        if config.compute.target_chunk_bytes is not None
        else ComputePolicy().target_chunk_bytes
    )
    if target < _MIN_WORKER_BUDGET_TARGET_BYTES:
        target = ComputePolicy().target_chunk_bytes
    fraction = (
        config.compute.worker_memory_fraction
        if config.compute.worker_memory_fraction is not None
        else DEFAULT_WORKER_MEMORY_FRACTION
    )
    slots = max(1, in_flight_slots)
    total = int(target * fraction)
    return max(1, total // slots)


__all__ = [
    "ComputePolicy",
    "ComputePolicyError",
    "DEFAULT_WORKER_MEMORY_FRACTION",
    "resolve_worker_byte_budget",
]
