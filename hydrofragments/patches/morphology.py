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
from skimage.measure import regionprops_table
from skimage.morphology import medial_axis

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
) -> PatchProperties:
    mask = np.asarray(crop.mask, dtype=bool)
    area_pixels = int(np.count_nonzero(mask))
    if area_pixels == 0:
        raise ValueError(f"component {crop.label} has no foreground pixels")

    if include_width:
        axis, dist = medial_axis(mask, return_distance=True)
        width_pixels = float((2.0 * dist[axis]).max())
    else:
        width_pixels = float("nan")
    return PatchProperties(
        label=crop.label,
        bbox=crop.bbox,
        area_pixels=area_pixels,
        area_m2=float(area_pixels * pixel_size_m**2),
        perimeter_m=float(_pixel_edge_perimeter(mask) * pixel_size_m),
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
    return tuple(
        _measure_component(
            crop,
            pixel_size_m=pixel_size_m,
            include_width=include_width,
            major_axis_length_pixels=major_axis_lengths[crop.label],
        )
        for crop in materialized
    )


__all__ = ["PatchProperties", "measure_components"]
