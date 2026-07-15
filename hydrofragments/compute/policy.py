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


__all__ = ["ComputePolicy", "ComputePolicyError"]
