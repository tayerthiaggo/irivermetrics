"""CPU reference implementation for certified integer/count reductions."""

from __future__ import annotations

from typing import Any

import numpy as np


class CPUBackend:
    """Reference backend. Inputs are concrete NumPy-compatible blocks."""

    name = "cpu"

    def valid_counts(self, valid_obs: Any, *, axis: int = 0) -> np.ndarray:
        return np.sum(np.asarray(valid_obs, dtype=bool), axis=axis, dtype=np.int64)

    def wet_counts(
        self, water: Any, valid_obs: Any, *, axis: int = 0
    ) -> np.ndarray:
        return np.sum(
            np.asarray(water, dtype=bool) & np.asarray(valid_obs, dtype=bool),
            axis=axis,
            dtype=np.int64,
        )

    def mean(self, values: Any, *, axis: int | None = None) -> np.ndarray:
        return np.mean(np.asarray(values), axis=axis)


__all__ = ["CPUBackend"]

