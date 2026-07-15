"""Dask chunk inspection and byte-budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import dask.array as da


class ChunkBudgetError(ValueError):
    """Raised when a stage's largest live chunk exceeds its byte budget."""


@dataclass(frozen=True)
class ChunkDiagnostics:
    """Serializable evidence for one named stage's largest Dask chunk."""

    stage: str
    max_chunk_shape: tuple[int, ...]
    max_chunk_bytes: int
    live_array_multiplier: float
    estimated_live_bytes: int
    target_chunk_bytes: int

    def to_mapping(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "max_chunk_shape": list(self.max_chunk_shape),
            "max_chunk_bytes": self.max_chunk_bytes,
            "live_array_multiplier": self.live_array_multiplier,
            "estimated_live_bytes": self.estimated_live_bytes,
            "target_chunk_bytes": self.target_chunk_bytes,
        }


def _array_data(array: Any) -> da.Array:
    data = getattr(array, "data", array)
    if not isinstance(data, da.Array):
        raise TypeError("chunk contracts require a Dask-backed array")
    return data


def validate_chunk_budget(
    array: Any,
    *,
    target_chunk_bytes: int,
    live_array_multiplier: float,
    stage: str,
) -> ChunkDiagnostics:
    """Inspect without rechunking and reject unsafe amplified live chunks."""

    data = _array_data(array)
    if any(
        not math.isfinite(float(size))
        for axis_chunks in data.chunks
        for size in axis_chunks
    ):
        raise ChunkBudgetError(
            f"unknown chunk sizes are unsafe for stage={stage}; "
            "resolve metadata without materializing raster values"
        )
    max_chunk_shape = tuple(max(axis_chunks) for axis_chunks in data.chunks)
    max_chunk_bytes = math.prod(max_chunk_shape) * data.dtype.itemsize
    estimated_live_bytes = math.ceil(max_chunk_bytes * live_array_multiplier)
    diagnostics = ChunkDiagnostics(
        stage=stage,
        max_chunk_shape=max_chunk_shape,
        max_chunk_bytes=max_chunk_bytes,
        live_array_multiplier=live_array_multiplier,
        estimated_live_bytes=estimated_live_bytes,
        target_chunk_bytes=target_chunk_bytes,
    )
    if estimated_live_bytes > target_chunk_bytes:
        raise ChunkBudgetError(
            f"unsafe chunk layout for stage={stage}: "
            f"estimated_live_bytes={estimated_live_bytes} exceeds "
            f"target_chunk_bytes={target_chunk_bytes}; "
            f"multiplier={live_array_multiplier:g}"
        )
    return diagnostics


__all__ = [
    "ChunkBudgetError",
    "ChunkDiagnostics",
    "validate_chunk_budget",
]
