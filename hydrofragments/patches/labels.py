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

from hydrofragments.compute.policy import ComputePolicy


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


def _default_local_label_threshold_bytes() -> int:
    """Byte threshold derived from the existing compute memory policy.

    Reuses ``ComputePolicy.target_chunk_bytes`` -- the same byte budget
    already used to gate safe Dask chunk sizes elsewhere in the pipeline --
    instead of introducing a second, independent memory-budget setting.
    """
    return ComputePolicy().target_chunk_bytes


def _materialize_global_labels(
    mask: np.ndarray | da.Array,
    *,
    structure: np.ndarray,
    local_label_threshold_bytes: int | None = None,
) -> tuple[np.ndarray, bool]:
    """Return ``(raw_labels, already_row_major)``.

    ``already_row_major`` is ``True`` exactly when ``raw_labels`` came from
    ``scipy.ndimage.label``, which scans its input in row-major order and
    assigns the next integer ID the first time it encounters a new
    component -- so its raw IDs are *always* already in ascending
    first-occurrence order by construction (proven empirically, not just
    assumed, in
    ``tests/parity/test_label_normalization_scipy_bypass_parity.py``).
    ``dask_image.ndmeasure.label``'s cross-chunk reconciliation can permute
    raw IDs relative to row-major occurrence order, so that branch reports
    ``False``.
    """
    if mask.ndim != 2:
        raise ValueError("patch labeling requires a 2-D mask")

    if isinstance(mask, da.Array):
        threshold = (
            _default_local_label_threshold_bytes()
            if local_label_threshold_bytes is None
            else local_label_threshold_bytes
        )
        if mask.nbytes <= threshold:
            # Small enough to materialize as one NumPy array: cross-chunk
            # reconciliation via dask-image is pure overhead here, and
            # _filter_and_normalize's row-major first-occurrence
            # normalization guarantees this branch and the dask-image
            # branch below produce byte-identical labels for the same mask.
            concrete = np.asarray(mask, dtype=bool)
            labels, _ = ndimage.label(concrete, structure=structure)
            return labels, True

        labels, _ = ndmeasure.label(mask.astype(bool), structure=structure)
        return np.asarray(labels.compute()), False

    concrete = np.asarray(mask, dtype=bool)
    labels, _ = ndimage.label(concrete, structure=structure)
    return labels, True


def _filter_and_normalize(
    raw_labels: np.ndarray,
    *,
    min_patch_pixels: int,
    already_row_major: bool = False,
) -> LabelResult:
    flat = raw_labels.reshape(-1)

    # Raw label IDs from ndimage/dask-image labeling are a dense range of
    # non-negative ints (0 = background), so bincount gives per-ID pixel
    # counts in O(P+K) without sorting -- unlike np.unique(..., counts).
    max_raw_id = int(flat.max()) if flat.size else 0
    counts = np.bincount(flat, minlength=max_raw_id + 1)

    if already_row_major:
        # scipy.ndimage.label's raw IDs are already ascending in row-major
        # first-occurrence order by construction (see
        # _materialize_global_labels's docstring and the parity proof in
        # tests/parity/test_label_normalization_scipy_bypass_parity.py), so
        # sorting retained IDs by first-occurrence position is equivalent to
        # sorting them by raw ID -- np.argsort(retained_ids) is already the
        # identity permutation. Skip the np.minimum.at reorder entirely.
        raw_ids = np.arange(max_raw_id + 1)
        foreground = raw_ids != 0
        retained_ids = np.flatnonzero(
            foreground & (counts >= min_patch_pixels)
        )
    else:
        # First-occurrence index per raw ID, O(P): np.minimum.at accumulates
        # the smallest (i.e. first, since flat is row-major) position per
        # raw ID with well-defined, order-independent ufunc.at semantics --
        # unlike basic fancy-index assignment, whose behavior on repeated
        # indices is explicitly documented as unspecified. Raw IDs need not
        # be monotonic with row-major occurrence order after dask-image's
        # cross-chunk reconciliation, so this cannot be assumed for free --
        # it must be computed explicitly, same as the
        # np.unique(return_index=True) it replaces.
        first = np.full(max_raw_id + 1, flat.size, dtype=np.intp)
        positions = np.arange(flat.size, dtype=np.intp)
        np.minimum.at(first, flat, positions)

        raw_ids = np.arange(max_raw_id + 1)
        foreground = raw_ids != 0
        retained_ids = np.flatnonzero(
            foreground & (counts >= min_patch_pixels)
        )
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
    local_label_threshold_bytes: int | None = None,
) -> LabelResult:
    """Label and globally filter one 2-D binary monthly mask.

    Returned IDs are contiguous ``int32`` values ordered by each retained
    component's first pixel in row-major order. Consequently, equivalent masks
    have identical IDs even when their Dask chunk layouts differ.

    A Dask-backed ``mask`` whose total byte size is at or below
    ``local_label_threshold_bytes`` (default: derived from
    ``ComputePolicy.target_chunk_bytes``) is materialized eagerly and labeled
    with ``scipy.ndimage.label`` instead of paying for ``dask-image``'s
    cross-chunk reconciliation, which is pure overhead once a mask already
    fits comfortably in memory. Above the threshold, the dask-image path is
    unchanged. Both paths give identical normalized labels for the same mask.
    """
    if min_patch_pixels < 1:
        raise ValueError("min_patch_pixels must be at least 1")

    raw_labels, already_row_major = _materialize_global_labels(
        mask,
        structure=_structure(connectivity),
        local_label_threshold_bytes=local_label_threshold_bytes,
    )
    return _filter_and_normalize(
        raw_labels,
        min_patch_pixels=min_patch_pixels,
        already_row_major=already_row_major,
    )


__all__ = ["LabelResult", "label_components"]

