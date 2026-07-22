"""Bounded component crops derived in one scan of a canonical label raster."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from scipy import ndimage


BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ComponentCrop:
    """One component's padded boolean crop and unpadded raster bounding box."""

    label: int
    bbox: BBox
    mask: np.ndarray


def iter_component_crops(
    labels: np.ndarray, *, padding: int = 1
) -> Iterator[ComponentCrop]:
    """Extract complete components without a full-raster scan per label.

    ``scipy.ndimage.find_objects`` obtains every bounding box in one pass.
    Padding is component-specific background: neighbouring labels are excluded,
    and synthetic background is added where a component touches a raster edge.
    """
    concrete = np.asarray(labels)
    if concrete.ndim != 2:
        raise ValueError("component crops require a 2-D label raster")
    if padding < 0:
        raise ValueError("padding cannot be negative")

    for label, slices in enumerate(ndimage.find_objects(concrete), start=1):
        if slices is None:
            continue
        row_slice, col_slice = slices
        row0, row1 = int(row_slice.start), int(row_slice.stop)
        col0, col1 = int(col_slice.start), int(col_slice.stop)

        expanded_row0 = max(0, row0 - padding)
        expanded_row1 = min(concrete.shape[0], row1 + padding)
        expanded_col0 = max(0, col0 - padding)
        expanded_col1 = min(concrete.shape[1], col1 + padding)
        component = (
            concrete[
                expanded_row0:expanded_row1,
                expanded_col0:expanded_col1,
            ]
            == label
        )

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


def extract_component_crops(
    labels: np.ndarray, *, padding: int = 1
) -> tuple[ComponentCrop, ...]:
    """Materialize component crops for small direct/test use."""
    return tuple(iter_component_crops(labels, padding=padding))


def bucket_component_crops(
    crops: Iterable[ComponentCrop], *, target_pixels: int
) -> Iterator[tuple[ComponentCrop, ...]]:
    """Group ordered crops into bounded units using padded crop pixels as work."""
    if target_pixels < 1:
        raise ValueError("target_pixels must be at least 1")

    current: list[ComponentCrop] = []
    current_pixels = 0
    for crop in crops:
        crop_pixels = int(crop.mask.size)
        if current and current_pixels + crop_pixels > target_pixels:
            yield tuple(current)
            current = []
            current_pixels = 0
        current.append(crop)
        current_pixels += crop_pixels
    if current:
        yield tuple(current)


__all__ = [
    "BBox",
    "ComponentCrop",
    "bucket_component_crops",
    "extract_component_crops",
    "iter_component_crops",
]
