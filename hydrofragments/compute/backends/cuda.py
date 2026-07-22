"""Lazy CuPy backend for certified reductions only.

Instantiation is explicit and never occurs during ``import hydrofragments``.
Dispatch remains evidence-gated by :mod:`hydrofragments.compute.capabilities`.
"""

from __future__ import annotations

import importlib
from typing import Any


class CUDABackend:
    """CuPy implementation matching :class:`CPUBackend` reduction contracts."""

    name = "cuda"

    def __init__(self, cupy_module: Any | None = None) -> None:
        if cupy_module is None:
            try:
                cupy_module = importlib.import_module("cupy")
            except Exception as error:
                raise RuntimeError(f"CuPy/CUDA unavailable: {error}") from error
        self.xp = cupy_module

    def valid_counts(self, valid_obs: Any, *, axis: int = 0) -> Any:
        return self.xp.sum(self.xp.asarray(valid_obs, dtype=bool), axis=axis, dtype=self.xp.int64)

    def wet_counts(self, water: Any, valid_obs: Any, *, axis: int = 0) -> Any:
        return self.xp.sum(
            self.xp.asarray(water, dtype=bool) & self.xp.asarray(valid_obs, dtype=bool),
            axis=axis,
            dtype=self.xp.int64,
        )

    def mean(self, values: Any, *, axis: int | None = None) -> Any:
        return self.xp.mean(self.xp.asarray(values), axis=axis)


__all__ = ["CUDABackend"]

