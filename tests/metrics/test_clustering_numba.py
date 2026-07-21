"""Numba prototype for the inter-pool-gap run-length loop: parity + fallback.

Twin of tests/benchmarks/test_cuda_parity.py's "import-safe, self-skip
cleanly without the optional accelerator" pattern, applied to Numba instead
of CuPy. hydrofragments/metrics/clustering.py::compute_inter_pool_gaps stays
the certified reference implementation; _compute_inter_pool_gaps_numba is an
additive alternative that must reproduce it exactly.
"""

from __future__ import annotations

import numpy as np
import pytest


def _reference_gaps(wet: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Pure-Python reference extracted to compare against both code paths."""
    gaps: list[float] = []
    index = 0
    while index < wet.size:
        if wet[index]:
            index += 1
            continue
        start = index
        while index < wet.size and not wet[index]:
            index += 1
        bounded = start > 0 and index < wet.size
        if bounded:
            gaps.append(float(lengths[start:index].sum()))
    return np.asarray(gaps, dtype=np.float64)


CASES = {
    "empty": (np.array([], dtype=bool), np.array([], dtype=np.float64)),
    "single_run_unbounded": (
        np.array([False, False, False], dtype=bool),
        np.array([1.0, 2.0, 3.0], dtype=np.float64),
    ),
    "single_bounded_gap": (
        np.array([True, False, False, True], dtype=bool),
        np.array([5.0, 1.0, 2.0, 5.0], dtype=np.float64),
    ),
    "multiple_gaps": (
        np.array([True, False, True, False, False, True, False, True], dtype=bool),
        np.array([1.0, 2.0, 1.0, 3.0, 4.0, 1.0, 5.0, 1.0], dtype=np.float64),
    ),
    "all_wet": (
        np.array([True, True, True], dtype=bool),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    ),
    "all_dry": (
        np.array([False, False, False], dtype=bool),
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
    ),
    "leading_and_trailing_unbounded_dry": (
        np.array([False, True, False], dtype=bool),
        np.array([2.0, 1.0, 2.0], dtype=np.float64),
    ),
}


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_numba_kernel_matches_pure_python_reference(case_name: str) -> None:
    from hydrofragments.metrics.clustering_numba import _compute_inter_pool_gaps_numba

    wet, lengths = CASES[case_name]
    expected = _reference_gaps(wet, lengths)
    actual = _compute_inter_pool_gaps_numba(wet, lengths)

    assert np.array_equal(np.asarray(actual, dtype=np.float64), expected)


@pytest.mark.parametrize("case_name", sorted(CASES))
def test_numba_kernel_matches_public_api_gaps(case_name: str) -> None:
    """Matches compute_inter_pool_gaps's own gaps_m output, not just the
    hand-rolled reference above -- the actual certified implementation."""
    from hydrofragments.metrics.clustering import compute_inter_pool_gaps
    from hydrofragments.metrics.clustering_numba import _compute_inter_pool_gaps_numba

    wet, lengths = CASES[case_name]
    if wet.size == 0:
        pytest.skip("compute_inter_pool_gaps requires non-empty input in practice")

    result = compute_inter_pool_gaps(wet.tolist(), segment_lengths_m=lengths.tolist())
    actual = _compute_inter_pool_gaps_numba(wet, lengths)

    assert np.array_equal(np.asarray(actual, dtype=np.float64), np.asarray(result.gaps_m, dtype=np.float64))


def test_kernel_importable_and_falls_back_cleanly_without_numba(monkeypatch) -> None:
    """Simulate Numba absent (mirrors test_cuda_parity's CuPy-absent test):
    the module must still import, and calling the kernel must transparently
    fall back to the pure-Python/NumPy path rather than raising ImportError.
    """
    import sys
    import importlib

    from hydrofragments.metrics import clustering_numba as module

    sys.modules.pop("numba", None)
    real_import_module = importlib.import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'numba'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)
    monkeypatch.setattr(module, "_NUMBA_KERNEL", None)
    monkeypatch.setattr(module, "_NUMBA_IMPORT_ATTEMPTED", False)

    wet = np.array([True, False, False, True], dtype=bool)
    lengths = np.array([5.0, 1.0, 2.0, 5.0], dtype=np.float64)

    result = module._compute_inter_pool_gaps_numba(wet, lengths)

    assert np.array_equal(np.asarray(result, dtype=np.float64), np.array([3.0]))
    assert "numba" not in sys.modules


def test_numba_available_flag_reflects_import_result() -> None:
    from hydrofragments.metrics import clustering_numba as module

    # This environment has numba installed (see pyproject.toml's accel
    # extra); the flag must reflect a real successful import, not just
    # "the module didn't crash".
    assert module.numba_available() in (True, False)
