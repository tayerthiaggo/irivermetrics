"""Numba JIT prototype for the inter-pool-gap run-length loop.

Twin of :mod:`hydrofragments.compute.backends.cuda` for a Numba kernel
instead of a CuPy backend: lazily imports ``numba`` (never at package import
time), and if it is unavailable for any reason, transparently falls back to
a pure-Python/NumPy implementation instead of raising. Nothing in this
module is wired into the live ``analyze()`` path by default -- see
``hydrofragments/metrics/clustering.py::compute_inter_pool_gaps``, which
remains the certified reference implementation and is unmodified by this
module. This kernel only becomes eligible to run via
``hydrofragments.compute.capabilities.NUMBA_CANDIDATE_KERNELS`` /
``numba_enabled_kernels`` once ``hydrofragments/benchmarks/numba_kernels.py``
records parity + speedup evidence; see ``docs/acceleration.md``.

The kernel computes exactly what
:func:`hydrofragments.metrics.clustering.compute_inter_pool_gaps` computes
internally before summary statistics are derived: the array of bounded dry
"gap" run-length sums (a dry run flanked by a wet cell on both sides), in
channel order. It does not replace or alter that function's numeric
behavior -- it is a proven-equivalent alternative array-producing step,
usable by callers who want the same numbers computed via `@njit` once
benchmark evidence says it is worth doing.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_NUMBA_KERNEL: Any | None = None
_NUMBA_IMPORT_ATTEMPTED: bool = False


def _pure_python_inter_pool_gaps(wet: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Reference NumPy/Python implementation -- always available fallback.

    Structurally identical to the while-loop inside
    :func:`hydrofragments.metrics.clustering.compute_inter_pool_gaps`; kept
    as a free function here so both the Numba path and the "Numba absent"
    fallback path can share one obviously-correct implementation to diff
    the JIT kernel against.
    """
    gaps: list[float] = []
    index = 0
    size = wet.shape[0]
    while index < size:
        if wet[index]:
            index += 1
            continue
        start = index
        while index < size and not wet[index]:
            index += 1
        bounded = start > 0 and index < size
        if bounded:
            total = 0.0
            for i in range(start, index):
                total += lengths[i]
            gaps.append(total)
    return np.asarray(gaps, dtype=np.float64)


def _build_numba_kernel() -> Any | None:
    """Attempt to JIT-compile the kernel. Returns ``None`` on any failure.

    Lazy-import mirrors :class:`hydrofragments.compute.backends.cuda.CUDABackend`'s
    ``importlib.import_module("cupy")`` pattern: never imported at module
    import time, and any failure (missing package, incompatible platform,
    LLVM/toolchain issue) degrades to "kernel unavailable" rather than
    propagating -- calling code must never see an ImportError from this
    module.
    """
    try:
        numba = importlib.import_module("numba")
    except Exception:  # optional dependency and any runtime failure
        return None

    try:
        @numba.njit
        def _kernel(wet: np.ndarray, lengths: np.ndarray) -> np.ndarray:
            size = wet.shape[0]
            # Numba requires a concrete, growable buffer strategy; a plain
            # Python list of float64 scalars compiles fine under nopython
            # mode and mirrors the pure-Python accumulation exactly.
            gaps = []
            index = 0
            while index < size:
                if wet[index]:
                    index += 1
                    continue
                start = index
                while index < size and not wet[index]:
                    index += 1
                bounded = start > 0 and index < size
                if bounded:
                    total = 0.0
                    for i in range(start, index):
                        total += lengths[i]
                    gaps.append(total)
            out = np.empty(len(gaps), dtype=np.float64)
            for i in range(len(gaps)):
                out[i] = gaps[i]
            return out

        return _kernel
    except Exception:  # pragma: no cover - defensive, mirrors CUDA's breadth
        return None


def numba_available() -> bool:
    """Return whether a working Numba kernel could be (or was) built.

    Triggers the lazy build/import on first call and caches the result, same
    caching shape as ``_NUMBA_KERNEL`` below.
    """
    _ensure_kernel()
    return _NUMBA_KERNEL is not None


def _ensure_kernel() -> None:
    global _NUMBA_KERNEL, _NUMBA_IMPORT_ATTEMPTED
    if _NUMBA_IMPORT_ATTEMPTED:
        return
    _NUMBA_KERNEL = _build_numba_kernel()
    _NUMBA_IMPORT_ATTEMPTED = True


def _compute_inter_pool_gaps_numba(wet: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Return bounded dry-run-length sums, via Numba if available.

    Falls back transparently to :func:`_pure_python_inter_pool_gaps` when
    Numba is not installed or fails to compile -- never raises
    ``ImportError`` to the caller. This function is a prototype callable
    directly by benchmarks/tests; it is not (yet) wired into
    :func:`hydrofragments.metrics.clustering.compute_inter_pool_gaps`'s call
    path -- see docs/acceleration.md for the scoping decision.
    """
    wet_array = np.asarray(wet, dtype=bool)
    lengths_array = np.asarray(lengths, dtype=np.float64)

    _ensure_kernel()
    if _NUMBA_KERNEL is None:
        return _pure_python_inter_pool_gaps(wet_array, lengths_array)
    return np.asarray(_NUMBA_KERNEL(wet_array, lengths_array), dtype=np.float64)


__all__ = [
    "numba_available",
    "_compute_inter_pool_gaps_numba",
    "_pure_python_inter_pool_gaps",
]
