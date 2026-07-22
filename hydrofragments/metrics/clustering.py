"""Longitudinal inter-pool gaps on an ordered real channel profile."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InterPoolGapResult:
    gaps_m: tuple[float, ...]
    mean_m: float
    median_m: float
    max_m: float
    cv: float
    percent_above_threshold: float
    threshold_m: float | None


def compute_inter_pool_gaps(
    wet: Sequence[bool],
    *,
    segment_lengths_m: Sequence[float],
    threshold_m: float | None = None,
) -> InterPoolGapResult:
    """Summarise dry runs bounded by wet runs in supplied channel order."""
    states = np.asarray(wet, dtype=bool)
    lengths = np.asarray(segment_lengths_m, dtype=float)
    if states.ndim != 1 or lengths.ndim != 1:
        raise ValueError("wet and segment_lengths_m must be one-dimensional")
    if states.size != lengths.size:
        raise ValueError("wet and segment_lengths_m must have equal length")
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_lengths_m must contain positive finite values")
    if threshold_m is not None and threshold_m < 0:
        raise ValueError("threshold_m cannot be negative")

    gaps: list[float] = []
    index = 0
    while index < states.size:
        if states[index]:
            index += 1
            continue
        start = index
        while index < states.size and not states[index]:
            index += 1
        bounded = start > 0 and index < states.size
        if bounded:
            gaps.append(float(lengths[start:index].sum()))

    if not gaps:
        nan = float("nan")
        return InterPoolGapResult((), nan, nan, nan, nan, nan, threshold_m)

    values = np.asarray(gaps, dtype=float)
    mean = float(values.mean())
    cv = float(values.std(ddof=0) / mean) if values.size > 1 and mean > 0 else math.nan
    percent = (
        math.nan
        if threshold_m is None
        else float(np.count_nonzero(values > threshold_m) / values.size * 100.0)
    )
    return InterPoolGapResult(
        gaps_m=tuple(gaps),
        mean_m=mean,
        median_m=float(np.median(values)),
        max_m=float(values.max()),
        cv=cv,
        percent_above_threshold=percent,
        threshold_m=threshold_m,
    )


__all__ = ["InterPoolGapResult", "compute_inter_pool_gaps"]
