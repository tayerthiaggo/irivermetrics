"""Globally reconciled connected-component labels for one monthly mask.

This module is the explicit boundary between chunked monthly masks and the CPU
morphology reference. Dask-backed inputs use ``dask-image`` to reconcile
component membership across spatial chunk boundaries before the result is
materialised. Minimum mapping unit filtering and deterministic label
normalisation happen only after that global reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile

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


@dataclass(frozen=True)
class LabelCheckpointRef:
    """On-disk normalized label checkpoint for a window exceeding RAM budget."""

    path: str
    count: int
    shape: tuple[int, int]


def _iter_zarr_chunks(array):
    height, width = array.shape
    chunk_y, chunk_x = array.chunks
    for row0 in range(0, height, chunk_y):
        row1 = min(row0 + chunk_y, height)
        for col0 in range(0, width, chunk_x):
            col1 = min(col0 + chunk_x, width)
            yield row0, row1, col0, col1, np.asarray(array[row0:row1, col0:col1])


def _as_chunked_dask(mask: np.ndarray | da.Array, *, threshold_bytes: int) -> da.Array:
    if isinstance(mask, da.Array):
        return mask.astype(bool, copy=False)
    concrete = np.asarray(mask, dtype=bool)
    itemsize = max(1, int(concrete.dtype.itemsize))
    side = max(1, int(np.sqrt(max(threshold_bytes, itemsize) / itemsize)))
    side = max(1, min(side, concrete.shape[0], concrete.shape[1]))
    return da.from_array(concrete, chunks=(side, side))


def _write_normalized_label_checkpoint(
    mask: np.ndarray | da.Array,
    *,
    connectivity: int,
    min_patch_pixels: int,
    threshold_bytes: int,
    spill_dir: Path,
) -> LabelCheckpointRef:
    import zarr

    labeled, _nlabels = ndmeasure.label(
        _as_chunked_dask(mask, threshold_bytes=threshold_bytes),
        structure=_structure(connectivity),
    )
    spill_dir.mkdir(parents=True, exist_ok=True)
    raw_path = spill_dir / "raw_labels.zarr"
    dest_path = spill_dir / "labels.zarr"
    labeled.astype(np.int32).to_zarr(str(raw_path), overwrite=True)
    raw = zarr.open(str(raw_path), mode="r")
    height, width = raw.shape
    max_raw_id = 0
    for _row0, _row1, _col0, _col1, block in _iter_zarr_chunks(raw):
        if block.size:
            max_raw_id = max(max_raw_id, int(block.max()))
    counts = np.zeros(max_raw_id + 1, dtype=np.int64)
    first = np.full(max_raw_id + 1, height * width, dtype=np.int64)
    for row0, row1, col0, col1, block in _iter_zarr_chunks(raw):
        if not block.size:
            continue
        rows = np.arange(row0, row1, dtype=np.int64)[:, None]
        cols = np.arange(col0, col1, dtype=np.int64)[None, :]
        positions = rows * width + cols
        flat_labels = block.reshape(-1)
        flat_pos = np.broadcast_to(positions, block.shape).reshape(-1)
        counts += np.bincount(flat_labels, minlength=max_raw_id + 1)
        nonzero = flat_labels != 0
        if np.any(nonzero):
            np.minimum.at(first, flat_labels[nonzero], flat_pos[nonzero])
    raw_ids = np.arange(max_raw_id + 1)
    retained_ids = np.flatnonzero((raw_ids != 0) & (counts >= min_patch_pixels))
    retained_ids = retained_ids[np.argsort(first[retained_ids], kind="stable")]
    count = int(retained_ids.size)
    if count > np.iinfo(np.int32).max:
        raise OverflowError("component count exceeds int32 label capacity")
    lookup = np.zeros(max_raw_id + 1, dtype=np.int32)
    lookup[retained_ids] = np.arange(1, count + 1, dtype=np.int32)
    dest = zarr.open(
        str(dest_path),
        mode="w",
        shape=raw.shape,
        chunks=raw.chunks,
        dtype=np.int32,
    )
    for row0, row1, col0, col1, block in _iter_zarr_chunks(raw):
        dest[row0:row1, col0:col1] = lookup[block]
    shutil.rmtree(raw_path, ignore_errors=True)
    return LabelCheckpointRef(path=str(dest_path), count=count, shape=tuple(raw.shape))


def label_bboxes_from_checkpoint(checkpoint: LabelCheckpointRef) -> list[tuple[int, int, int, int]]:
    """Return unpadded bboxes ``(row0, col0, row1, col1)`` from a label Zarr store."""

    import zarr

    stored = zarr.open(checkpoint.path, mode="r")
    count = checkpoint.count
    if count <= 0:
        return []
    height, width = stored.shape
    row0 = np.full(count + 1, height, dtype=np.int64)
    row1 = np.zeros(count + 1, dtype=np.int64)
    col0 = np.full(count + 1, width, dtype=np.int64)
    col1 = np.zeros(count + 1, dtype=np.int64)
    present = np.zeros(count + 1, dtype=bool)
    for y0, y1, x0, x1, block in _iter_zarr_chunks(stored):
        labels = np.unique(block)
        labels = labels[labels > 0]
        for label in labels:
            rows, cols = np.nonzero(block == label)
            present[int(label)] = True
            row0[label] = min(row0[label], y0 + int(rows.min()))
            row1[label] = max(row1[label], y0 + int(rows.max()) + 1)
            col0[label] = min(col0[label], x0 + int(cols.min()))
            col1[label] = max(col1[label], x0 + int(cols.max()) + 1)
    return [
        (int(row0[label]), int(col0[label]), int(row1[label]), int(col1[label]))
        for label in range(1, count + 1)
        if present[label]
    ]


def iter_checkpoint_component_crops(
    checkpoint: LabelCheckpointRef, *, padding: int = 1
):
    """Yield padded component crops by reading only each component's bbox."""

    from hydrofragments.patches.components import ComponentCrop
    import zarr

    if padding < 0:
        raise ValueError("padding cannot be negative")
    stored = zarr.open(checkpoint.path, mode="r")
    height, width = stored.shape
    bboxes = label_bboxes_from_checkpoint(checkpoint)
    for label, (row0, col0, row1, col1) in enumerate(bboxes, start=1):
        expanded_row0 = max(0, row0 - padding)
        expanded_row1 = min(height, row1 + padding)
        expanded_col0 = max(0, col0 - padding)
        expanded_col1 = min(width, col1 + padding)
        window = np.asarray(
            stored[expanded_row0:expanded_row1, expanded_col0:expanded_col1]
        )
        component = window == label
        component = np.pad(
            component,
            (
                (
                    padding - (row0 - expanded_row0),
                    padding - (expanded_row1 - row1),
                ),
                (
                    padding - (col0 - expanded_col0),
                    padding - (expanded_col1 - col1),
                ),
            ),
            mode="constant",
            constant_values=False,
        )
        yield ComponentCrop(
            label=label,
            bbox=(row0, col0, row1, col1),
            mask=component,
        )


def label_components_to_checkpoint(
    mask: np.ndarray | da.Array,
    *,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    local_label_threshold_bytes: int | None = None,
    spill_dir: Path | None = None,
) -> tuple[LabelResult | None, LabelCheckpointRef | None]:
    """Return in-memory labels or a spill checkpoint when the window exceeds budget."""

    if mask.ndim != 2:
        raise ValueError("patch labeling requires a 2-D mask")
    if min_patch_pixels < 1:
        raise ValueError("min_patch_pixels must be at least 1")

    nbytes = mask.nbytes if hasattr(mask, "nbytes") else np.asarray(mask).nbytes
    threshold = (
        _default_local_label_threshold_bytes()
        if local_label_threshold_bytes is None
        else local_label_threshold_bytes
    )

    if nbytes <= threshold:
        return label_components(
            mask,
            connectivity=connectivity,
            min_patch_pixels=min_patch_pixels,
            local_label_threshold_bytes=local_label_threshold_bytes,
        ), None

    parent = spill_dir if spill_dir is not None else Path(tempfile.mkdtemp())
    checkpoint = _write_normalized_label_checkpoint(
        mask,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        threshold_bytes=threshold,
        spill_dir=parent,
    )
    return None, checkpoint


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


__all__ = [
    "LabelCheckpointRef",
    "LabelResult",
    "iter_checkpoint_component_crops",
    "label_bboxes_from_checkpoint",
    "label_components",
    "label_components_to_checkpoint",
]

