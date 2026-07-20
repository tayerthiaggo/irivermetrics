"""Tier A synthetic analytic mask fixtures (Milestone 1).

Tiny, hand-calculable boolean masks with documented ground truth (component count,
pixel area, connectivity behaviour). They exist so kernels — legacy today, v1.2 later —
can be checked against known-correct answers, independent of any historical CSV.

See tests/fixtures/README.md for tier definitions and how each mask is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def empty_mask(shape: tuple[int, int] = (6, 6)) -> np.ndarray:
    """All-dry mask. Ground truth: 0 connected components, 0 wet pixels."""
    return np.zeros(shape, dtype=bool)


def full_mask(shape: tuple[int, int] = (6, 6)) -> np.ndarray:
    """All-wet mask. Ground truth: 1 connected component covering every pixel."""
    return np.ones(shape, dtype=bool)


def diagonal_pair_mask() -> np.ndarray:
    """Two pixels touching only at a corner, embedded in a 4x4 dry canvas.

    Ground truth: 1 connected component under 8-connectivity (the legacy kernel's
    ``structure=np.ones((3, 3))``); would be 2 separate components under 4-connectivity.
    """
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    return mask


def one_pixel_noise_mask() -> np.ndarray:
    """One isolated 1-pixel speck plus a separate 2x2 block, well apart.

    Ground truth: 2 connected components (1 pixel + 4 pixels = 5 wet pixels total)
    before any small-object filtering is applied.
    """
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = True
    mask[3:5, 3:5] = True
    return mask


def mask_with_hole() -> np.ndarray:
    """A 5x5 wet square with a single dry pixel hole in the centre.

    Ground truth: 1 connected component (the pixels remain 8-connected around the
    hole), 24 wet pixels (25 - 1 hole).
    """
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    return mask


def long_bar_mask(length: int = 20) -> np.ndarray:
    """A single-pixel-wide straight horizontal bar of the given length.

    Ground truth: 1 connected component, ``length`` wet pixels, already its own
    morphological skeleton, longest path == ``length - 1`` unit steps.
    """
    mask = np.zeros((3, length), dtype=bool)
    mask[1, :] = True
    return mask


def padded_square_mask(square_size: int = 5, pad: int = 2) -> np.ndarray:
    """A solid wet square centred in a dry canvas, with real background on all sides.

    Ground truth: 1 connected component; Euclidean-distance-transform maximum sits at
    the square's centre pixel (needed to make EDT/width characterisation meaningful —
    an all-wet mask with no true background gives no defined distance-to-shore).
    """
    canvas_size = square_size + 2 * pad
    mask = np.zeros((canvas_size, canvas_size), dtype=bool)
    mask[pad : pad + square_size, pad : pad + square_size] = True
    return mask


def chunk_crossing_mask(n_chunks: int, chunk_size: int = 4) -> np.ndarray:
    """A single component straddling ``n_chunks`` equal-width chunks along axis 1.

    Ground truth: 1 connected component under whole-array (unchunked) labelling,
    ``chunk_size * n_chunks`` wet pixels, spanning chunk boundaries at
    ``x = chunk_size, 2*chunk_size, ..., (n_chunks - 1) * chunk_size``. This locks in
    the reference truth that Milestone 6's global-label reconciliation must reproduce
    once real Dask chunking is introduced; it is exercised here only against the
    current whole-array legacy kernel.
    """
    if n_chunks < 2:
        raise ValueError("chunk_crossing_mask requires at least 2 chunks to be meaningful")
    width = n_chunks * chunk_size
    mask = np.zeros((chunk_size, width), dtype=bool)
    mask[chunk_size // 2, :] = True
    return mask


def recurrence_temporal_fixture() -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Three-year 1x1 temporal fixture for season-stratified recurrence.

    Ground truth: January wet fraction is 2/3; February has one observed dry
    year, so equal supported-month weighting gives recurrence 1/3 (33.33%).
    """
    times = pd.to_datetime(
        [
            "2001-01-01", "2001-02-01",
            "2002-01-01", "2002-02-01",
            "2003-01-01", "2003-02-01",
        ]
    )
    water = np.array([[[1]], [[0]], [[1]], [[0]], [[0]], [[0]]])
    valid = np.array([[[1]], [[0]], [[1]], [[0]], [[1]], [[1]]])
    return water, valid, times


def apsec_low_coverage_month_fixture() -> tuple[
    np.ndarray, np.ndarray, pd.DatetimeIndex
]:
    """Two-month 2x2 APSEC fixture with one sparse month.

    Ground truth at 10 m resolution over four pixels: both months remain
    APSEC 100% under the current compat value mask; month 1 has valid fraction
    1.0 and month 2 has valid fraction 0.25, below a 0.75 coverage floor.
    """
    times = pd.to_datetime(["2020-01-01", "2020-02-01"])
    water = np.ones((2, 2, 2), dtype=np.uint8)
    valid = np.array(
        [
            [[1, 1], [1, 1]],
            [[1, 0], [0, 0]],
        ],
        dtype=bool,
    )
    return water, valid, times


def temporal_summary_hand_fixture() -> tuple[
    np.ndarray, np.ndarray, pd.DatetimeIndex, dict[str, float]
]:
    """Two-year 2x2 fixture for hand-derived temporal AOI summaries.

    Ground truth:
    - AOI-mean recurrence = mean(50%, 50%, 50%, 8.333333%) = 39.583333%.
    - AOI-mean hydroperiod 2020 = mean(1/2, 1, 1/2, 1/12) = 0.520833.
    - AOI-mean hydroperiod 2021 = mean(1/2, 0, 1/2, 1/12) = 0.270833.
    """
    times = pd.date_range("2020-01-01", periods=24, freq="MS")
    water = np.zeros((24, 2, 2), dtype=bool)
    valid = np.ones((24, 2, 2), dtype=bool)

    water[0:6, 0, 0] = True
    water[12:18, 0, 0] = True
    water[0:12, 0, 1] = True
    valid[6:12, 1, 0] = False
    valid[18:24, 1, 0] = False
    water[0:3, 1, 0] = True
    water[12:15, 1, 0] = True
    water[0, 1, 1] = True
    water[13, 1, 1] = True

    expected = {
        "recurrence": 39.58333333333333,
        "hydroperiod_2020": 0.5208333333333334,
        "hydroperiod_2021": 0.2708333333333333,
    }
    return water, valid, times, expected


def refuge_stability_fixture() -> tuple[np.ndarray, np.ndarray]:
    """Two consecutive end-dry footprints with Jaccard overlap 3/5 = 0.6."""
    previous = np.array([[1, 1, 0], [1, 1, 0]], dtype=bool)
    current = np.array([[1, 1, 0], [1, 0, 1]], dtype=bool)
    return current, previous
