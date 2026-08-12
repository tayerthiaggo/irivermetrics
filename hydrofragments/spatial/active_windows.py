"""Independent active-processing windows derived from an ``analysis_mask``.

``analysis_mask`` is the conservative potential-water footprint (see
:mod:`hydrofragments.models`'s ``WaterCube.analysis_mask`` and
``hydrofragments/io/cache_footprints.py``'s ``VerifiedCacheFootprints``):
a superset of every pixel that could ever be wet across the whole monthly
series. Because it is conservative, any *actual* monthly wet mask is a
pixelwise subset of it.

:func:`independent_active_windows` partitions that footprint into
"AnalysisWindow" bounding boxes that can each be processed completely
independently -- i.e. patch/connected-component metrics measured separately
on each window's crop and then concatenated must equal the same metrics
measured on the full mask in one pass. This is what makes it safe for a
later task (W4.3, not implemented here) to skip empty/dry windows and only
materialise/measure the active ones, without silently changing LPI, AWRe,
AWMSI, or any other spatial metric that depends on connected-component
identity.

The correctness argument for why concatenation is safe:

1. ``scipy.ndimage.label`` (with the SAME ``connectivity`` used later to
   label actual wet pixels) is run directly on ``analysis_mask`` itself.
   Two pixels that end up in different raw ``analysis_mask`` components are,
   by definition of connected-component labeling, not reachable from one
   another under that connectivity while staying inside the mask. Since any
   real wet pixel is a subset of ``analysis_mask``, a wet connected
   component can *never* span two different ``analysis_mask`` components --
   it would have needed a path through masked-out (guaranteed-dry) pixels to
   do so, and connectivity is defined purely in terms of the pixel grid, not
   pixel values along a path outside the crop.
2. Each raw component's bounding box is expanded by ``halo_pixels`` (a
   defensive margin so that any local morphology op near a window boundary
   -- e.g. a future distance transform in W4.3/W3.5 -- is not edge-clipped
   relative to a full-mask computation) and then snapped outward to
   ``align_pixels`` boundaries (an I/O/chunk-alignment convenience with no
   correctness role of its own).
3. Expansion and alignment can cause two originally separate boxes to
   overlap or touch. Any such windows are merged (repeatedly, to a fixed
   point) into one box before returning -- an unmerged pair of overlapping
   windows would either double-count the overlap region if measured
   independently, or (worse) risk splitting a component that a subsequent
   halo/alignment pass caused to newly span the two boxes.

The result is windows that are pairwise disjoint (no overlap) after the
merge step, collectively covering every pixel that could ever be wet, with
each retained wet component guaranteed to lie entirely inside exactly one
window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import dask.array as da
import numpy as np
from scipy import ndimage
import xarray as xr

from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.patches.labels import label_components

BBox = tuple[int, int, int, int]  # (row0, col0, row1, col1); row1/col1 exclusive


@dataclass(frozen=True)
class AnalysisWindow:
    """One independent active-processing window over a 2-D raster grid.

    ``bbox`` is ``(row0, col0, row1, col1)`` in raster (row-major, 0-based,
    half-open) index space -- ``mask[row0:row1, col0:col1]`` is the window's
    crop. ``window_id`` is deterministic given identical inputs (stable row-
    major ordering by each window's top-left corner), so re-running
    :func:`independent_active_windows` on the same ``analysis_mask`` always
    produces identically-ordered, identically-identified windows regardless
    of any incidental dict/set iteration order elsewhere.
    """

    window_id: str
    bbox: BBox


def _structure(connectivity: Literal[4, 8]) -> np.ndarray:
    if connectivity == 4:
        return ndimage.generate_binary_structure(2, 1)
    if connectivity == 8:
        return ndimage.generate_binary_structure(2, 2)
    raise ValueError("connectivity must be 4 or 8")


def _component_bboxes(mask: np.ndarray, *, connectivity: Literal[4, 8]) -> list[BBox]:
    labels, count = ndimage.label(mask, structure=_structure(connectivity))
    if count == 0:
        return []
    bboxes: list[BBox] = []
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        row_slice, col_slice = slices
        bboxes.append(
            (int(row_slice.start), int(col_slice.start), int(row_slice.stop), int(col_slice.stop))
        )
    return bboxes


def _expand_and_align(
    bbox: BBox, *, halo_pixels: int, align_pixels: int, height: int, width: int
) -> BBox:
    row0, col0, row1, col1 = bbox
    row0 = max(0, row0 - halo_pixels)
    col0 = max(0, col0 - halo_pixels)
    row1 = min(height, row1 + halo_pixels)
    col1 = min(width, col1 + halo_pixels)

    if align_pixels > 1:
        row0 = (row0 // align_pixels) * align_pixels
        col0 = (col0 // align_pixels) * align_pixels
        row1 = min(height, -(-row1 // align_pixels) * align_pixels)
        col1 = min(width, -(-col1 // align_pixels) * align_pixels)

    return (row0, col0, row1, col1)


def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    a_row0, a_col0, a_row1, a_col1 = a
    b_row0, b_col0, b_row1, b_col1 = b
    row_overlap = a_row0 < b_row1 and b_row0 < a_row1
    col_overlap = a_col0 < b_col1 and b_col0 < a_col1
    return row_overlap and col_overlap


def _merge_bbox(a: BBox, b: BBox) -> BBox:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def _merge_overlapping(bboxes: list[BBox]) -> list[BBox]:
    """Repeatedly merge any pairwise-overlapping boxes to a fixed point.

    Merging two boxes can create a larger box that now overlaps a third
    previously-separate box, so this must iterate until no merge occurs in
    a full pass -- a single pass is not sufficient in general.
    """
    current = list(bboxes)
    changed = True
    while changed:
        changed = False
        merged: list[BBox] = []
        for box in current:
            absorbed = False
            for index, existing in enumerate(merged):
                if _bboxes_overlap(existing, box):
                    merged[index] = _merge_bbox(existing, box)
                    absorbed = True
                    changed = True
                    break
            if not absorbed:
                merged.append(box)
        current = merged
    return current


def _mask_nbytes(mask: xr.DataArray) -> int:
    data = mask.data
    if isinstance(data, da.Array):
        return int(data.nbytes)
    return int(np.asarray(data).nbytes)


def _component_bboxes_from_labels(labels: np.ndarray, *, connectivity: Literal[4, 8]) -> list[BBox]:
    count = int(labels.max())
    if count == 0:
        return []
    bboxes: list[BBox] = []
    for slices in ndimage.find_objects(labels):
        if slices is None:
            continue
        row_slice, col_slice = slices
        bboxes.append(
            (int(row_slice.start), int(col_slice.start), int(row_slice.stop), int(col_slice.stop))
        )
    return bboxes


def independent_active_windows(
    analysis_mask: xr.DataArray,
    *,
    connectivity: Literal[4, 8],
    halo_pixels: int = 1,
    align_pixels: int = 512,
) -> Sequence[AnalysisWindow]:
    """Partition ``analysis_mask`` into independent active-processing windows.

    Distinct windows are valid only when no possible retained wet component
    can cross between them under ``connectivity`` -- see the module
    docstring for the full correctness argument. Overlapping halos (which
    can arise after halo expansion and ``align_pixels`` snapping) are merged
    before windows are returned.

    Returns an empty sequence for an all-false ``analysis_mask`` (nothing to
    process) and, at the other extreme, a single window spanning the whole
    grid for an all-true mask.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if analysis_mask.ndim != 2:
        raise ValueError("analysis_mask must be a 2-D array")
    if halo_pixels < 0:
        raise ValueError("halo_pixels must be non-negative")
    if align_pixels < 1:
        raise ValueError("align_pixels must be at least 1")

    nbytes = _mask_nbytes(analysis_mask)
    threshold = ComputePolicy().target_chunk_bytes
    data = analysis_mask.data
    if isinstance(data, da.Array) and nbytes > threshold:
        label_result = label_components(
            data.astype(bool),
            connectivity=connectivity,
            min_patch_pixels=1,
            local_label_threshold_bytes=threshold,
        )
        raw_bboxes = _component_bboxes_from_labels(label_result.labels, connectivity=connectivity)
    else:
        mask = np.asarray(analysis_mask.values, dtype=bool)
        height, width = mask.shape
        raw_bboxes = _component_bboxes(mask, connectivity=connectivity)
        if not raw_bboxes:
            return ()
        expanded = [
            _expand_and_align(
                bbox,
                halo_pixels=halo_pixels,
                align_pixels=align_pixels,
                height=height,
                width=width,
            )
            for bbox in raw_bboxes
        ]
        merged = _merge_overlapping(expanded)
        merged.sort(key=lambda bbox: (bbox[0], bbox[1]))
        return tuple(
            AnalysisWindow(window_id=f"window-{index + 1:04d}", bbox=bbox)
            for index, bbox in enumerate(merged)
        )

    if not raw_bboxes:
        return ()

    height = int(analysis_mask.sizes[analysis_mask.dims[0]])
    width = int(analysis_mask.sizes[analysis_mask.dims[1]])

    expanded = [
        _expand_and_align(
            bbox,
            halo_pixels=halo_pixels,
            align_pixels=align_pixels,
            height=height,
            width=width,
        )
        for bbox in raw_bboxes
    ]
    merged = _merge_overlapping(expanded)

    # Deterministic row-major ordering by each window's top-left corner so
    # window identity/order never depends on scipy.ndimage.label's internal
    # raw-ID assignment order or dict/set iteration order.
    merged.sort(key=lambda bbox: (bbox[0], bbox[1]))

    return tuple(
        AnalysisWindow(window_id=f"window-{index + 1:04d}", bbox=bbox)
        for index, bbox in enumerate(merged)
    )


__all__ = ["AnalysisWindow", "BBox", "independent_active_windows"]
