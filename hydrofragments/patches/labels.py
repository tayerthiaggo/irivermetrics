"""Globally reconciled connected-component labels for one monthly mask.

This module is the explicit boundary between chunked monthly masks and the CPU
morphology reference. Dask-backed inputs use ``dask-image`` to reconcile
component membership across spatial chunk boundaries before the result is
materialised. Minimum mapping unit filtering and deterministic label
normalisation happen only after that global reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass

import dask.array as da
from dask_image import ndmeasure
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class LabelResult:
    """Canonical labels and retained component count for one 2-D month."""

    labels: np.ndarray
    count: int


def _structure(connectivity: int) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def _materialize_global_labels(
    mask: np.ndarray | da.Array, *, structure: np.ndarray
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("patch labeling requires a 2-D mask")

    if isinstance(mask, da.Array):
        labels, _ = ndmeasure.label(mask.astype(bool), structure=structure)
        return np.asarray(labels.compute())

    concrete = np.asarray(mask, dtype=bool)
    labels, _ = ndimage.label(concrete, structure=structure)
    return labels


def _filter_and_normalize(
    raw_labels: np.ndarray, *, min_patch_pixels: int
) -> LabelResult:
    flat = raw_labels.reshape(-1)

    # Raw label IDs from ndimage/dask-image labeling are a dense range of
    # non-negative ints (0 = background), so bincount gives per-ID pixel
    # counts in O(P+K) without sorting -- unlike np.unique(..., counts).
    max_raw_id = int(flat.max()) if flat.size else 0
    counts = np.bincount(flat, minlength=max_raw_id + 1)

    # First-occurrence index per raw ID, O(P): np.minimum.at accumulates the
    # smallest (i.e. first, since flat is row-major) position per raw ID with
    # well-defined, order-independent ufunc.at semantics -- unlike basic
    # fancy-index assignment, whose behavior on repeated indices is
    # explicitly documented as unspecified. Raw IDs need not be monotonic
    # with row-major occurrence order after dask-image's cross-chunk
    # reconciliation, so this cannot be assumed for free -- it must be
    # computed explicitly, same as the np.unique(return_index=True) it
    # replaces.
    first = np.full(max_raw_id + 1, flat.size, dtype=np.intp)
    positions = np.arange(flat.size, dtype=np.intp)
    np.minimum.at(first, flat, positions)

    raw_ids = np.arange(max_raw_id + 1)
    foreground = raw_ids != 0
    retained_ids = np.flatnonzero(foreground & (counts >= min_patch_pixels))
    retained_ids = retained_ids[
        np.argsort(first[retained_ids], kind="stable")
    ]

    count = int(retained_ids.size)
    if count > np.iinfo(np.int32).max:
        raise OverflowError("component count exceeds int32 label capacity")

    lookup = np.zeros(max_raw_id + 1, dtype=np.int32)
    lookup[retained_ids] = np.arange(1, count + 1, dtype=np.int32)
    labels = lookup[flat].reshape(raw_labels.shape)
    return LabelResult(labels=labels, count=count)


def label_components(
    mask: np.ndarray | da.Array,
    *,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
) -> LabelResult:
    """Label and globally filter one 2-D binary monthly mask.

    Returned IDs are contiguous ``int32`` values ordered by each retained
    component's first pixel in row-major order. Consequently, equivalent masks
    have identical IDs even when their Dask chunk layouts differ.
    """
    if min_patch_pixels < 1:
        raise ValueError("min_patch_pixels must be at least 1")

    raw_labels = _materialize_global_labels(
        mask, structure=_structure(connectivity)
    )
    return _filter_and_normalize(
        raw_labels, min_patch_pixels=min_patch_pixels
    )


__all__ = ["LabelResult", "label_components"]

