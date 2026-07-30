"""Compact exact CPU morphology for complete component crops.

Calculates area, raster-edge perimeter, regionprops major-axis length, and
Milestone 10 EDT maximum planform width. Width is morphology only, not a
depth/storage proxy. Optional real-channel skeleton length may be attached by
the channel path; this module never derives a wet-mask core ``L_ref``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops_table

from hydrofragments.patches.components import BBox, ComponentCrop


@dataclass(frozen=True)
class PatchProperties:
    """Numeric morphology for one retained patch."""

    label: int
    bbox: BBox
    area_pixels: int
    area_m2: float
    perimeter_m: float
    major_axis_length_m: float
    width_m: float = float("nan")
    width_pixels: float = float("nan")
    skeleton_length_m: float | None = None
    length_method: str = "major_axis"


def _pixel_edge_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return int(horizontal + vertical)


def _bulk_pixel_edge_perimeters(
    crops: tuple[ComponentCrop, ...],
) -> dict[int, int]:
    """Compute the pixel-edge perimeter for every crop with one bulk call.

    Mirrors ``_bulk_major_axis_lengths``'s batching strategy: every crop's
    mask is placed into disjoint blocks of one shared boolean composite,
    stacked down the rows, each block given the widest crop's column
    extent, so no two crops' bounding boxes can ever touch or overlap. One
    ``np.pad`` + two diff-and-count calls are then done ONCE on the whole
    composite instead of once per crop, and the horizontal/vertical edge
    counts are split back out per crop by its own row range within the
    composite.

    This reproduces the per-crop ``_pixel_edge_perimeter`` result bit-for-bit
    because every ``ComponentCrop.mask`` already carries its own 1px false
    border (``iter_component_crops``'s default padding): stacking blocks
    with zero gap means the row directly above and below each block is
    already all-False (that block's own padding), exactly matching what a
    standalone per-crop ``np.pad`` would produce at those same rows. Extra
    zero-columns to the right of a narrower block (padding it out to the
    composite's shared width) are also all-False, contributing no spurious
    horizontal edges. This identity is proven for randomized shapes and
    every real component in ``tests/wmask_ts.nc`` by
    ``tests/parity/test_perimeter_bulk_parity.py`` before this path is
    trusted.
    """
    max_width = max(crop.mask.shape[1] for crop in crops)
    total_height = sum(crop.mask.shape[0] for crop in crops)
    composite = np.zeros((total_height, max_width), dtype=bool)

    row_offset = 0
    offsets: list[tuple[int, int, int]] = []
    for crop in crops:
        mask = np.asarray(crop.mask, dtype=bool)
        height, width = mask.shape
        composite[row_offset : row_offset + height, 0:width] = mask
        offsets.append((crop.label, row_offset, row_offset + height))
        row_offset += height

    padded = np.pad(composite, 1, constant_values=False)
    horizontal_edges = padded[:, 1:] != padded[:, :-1]
    vertical_edges = padded[1:, :] != padded[:-1, :]

    perimeters: dict[int, int] = {}
    for label, row0, row1 in offsets:
        # composite row r is padded row r + 1. horizontal_edges' row axis
        # aligns with padded rows directly; vertical_edges' row r represents
        # the boundary between padded rows r and r + 1, so the block's rows
        # [row0+1, row1+1) need vertical_edges rows [row0, row1).
        horizontal = np.count_nonzero(horizontal_edges[row0 + 1 : row1 + 1, :])
        vertical = np.count_nonzero(vertical_edges[row0:row1, :])
        perimeters[label] = int(horizontal + vertical)
    return perimeters


def _bulk_major_axis_lengths(
    crops: tuple[ComponentCrop, ...],
) -> dict[int, float]:
    """Compute ``axis_major_length`` for every crop with one bulk call.

    ``axis_major_length`` is derived from central image moments, which are
    translation-invariant by construction (the centroid is subtracted before
    the moments are formed), and ``regionprops_table`` delegates to the same
    ``RegionProperties`` machinery as a per-component ``regionprops`` call.
    So placing every crop's mask into disjoint blocks of one shared labeled
    raster and reading back by label is expected to reproduce the
    per-component result bit-for-bit; that identity is what
    ``tests/parity/test_regionprops_parity.py`` proves before this path is
    trusted.

    Crops are laid out block-diagonally (stacked down the rows, each block
    given the widest crop's column extent) so no two crops' bounding boxes
    can ever touch or overlap, regardless of their individual shapes.
    """
    max_width = max(crop.mask.shape[1] for crop in crops)
    total_height = sum(crop.mask.shape[0] for crop in crops)
    composite = np.zeros((total_height, max_width), dtype=np.int64)

    row_offset = 0
    for crop in crops:
        mask = np.asarray(crop.mask, dtype=bool)
        height, width = mask.shape
        block = composite[row_offset : row_offset + height, 0:width]
        block[mask] = crop.label
        row_offset += height

    table = regionprops_table(composite, properties=["label", "axis_major_length"])
    return dict(zip(table["label"].tolist(), table["axis_major_length"].tolist()))


def _measure_component(
    crop: ComponentCrop,
    *,
    pixel_size_m: float,
    include_width: bool,
    major_axis_length_pixels: float,
    perimeter_pixels: int,
) -> PatchProperties:
    mask = np.asarray(crop.mask, dtype=bool)
    area_pixels = int(np.count_nonzero(mask))
    if area_pixels == 0:
        raise ValueError(f"component {crop.label} has no foreground pixels")

    if include_width:
        # Maximum inscribed pool diameter is 2 * the global maximum of the
        # Euclidean distance transform. The medial axis (skeleton ridge of
        # the distance transform) is not needed: it is a strict superset of
        # what this kernel consumes, and computing it is the dominant cost
        # of this function by a wide margin (W3.5). Restricting the max to
        # the medial-axis skeleton (the prior approach) can, in general,
        # UNDER-measure the true maximum -- skimage's medial_axis is a
        # topology-preserving thinning, not a guarantee that every
        # global-distance-maximum pixel survives thinning (proven false on
        # real Fitzroy catchment data and the textbook solid-disk case by
        # tests/parity/test_medial_axis_vs_edt_max_width.py) -- so this
        # also fixes a latent underestimate for "blob"-shaped pools, not
        # only a performance win.
        dist = distance_transform_edt(mask)
        width_pixels = float(2.0 * dist.max())
    else:
        width_pixels = float("nan")
    return PatchProperties(
        label=crop.label,
        bbox=crop.bbox,
        area_pixels=area_pixels,
        area_m2=float(area_pixels * pixel_size_m**2),
        perimeter_m=float(perimeter_pixels * pixel_size_m),
        major_axis_length_m=float(major_axis_length_pixels * pixel_size_m),
        width_m=width_pixels * pixel_size_m,
        width_pixels=width_pixels,
    )


def measure_components(
    crops: Iterable[ComponentCrop], *, pixel_size_m: float, include_width: bool = False
) -> tuple[PatchProperties, ...]:
    """Measure complete bounded crops with the CPU reference implementation."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    materialized = tuple(crops)
    if not materialized:
        return ()
    major_axis_lengths = _bulk_major_axis_lengths(materialized)
    perimeters = _bulk_pixel_edge_perimeters(materialized)
    return tuple(
        _measure_component(
            crop,
            pixel_size_m=pixel_size_m,
            include_width=include_width,
            major_axis_length_pixels=major_axis_lengths[crop.label],
            perimeter_pixels=perimeters[crop.label],
        )
        for crop in materialized
    )


__all__ = ["PatchProperties", "measure_components"]
