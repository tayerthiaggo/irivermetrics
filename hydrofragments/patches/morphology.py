"""Compact exact CPU morphology for complete component crops.

Only properties needed by Milestone 6 are calculated: area, raster-edge
perimeter, and regionprops major-axis length. Skeleton length, width, paths,
and vector geometry belong to later gated milestones.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from skimage.measure import regionprops

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
    length_method: str = "major_axis"


def _pixel_edge_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(np.asarray(mask, dtype=bool), 1, constant_values=False)
    horizontal = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    vertical = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    return int(horizontal + vertical)


def _measure_component(
    crop: ComponentCrop, *, pixel_size_m: float
) -> PatchProperties:
    mask = np.asarray(crop.mask, dtype=bool)
    area_pixels = int(np.count_nonzero(mask))
    if area_pixels == 0:
        raise ValueError(f"component {crop.label} has no foreground pixels")

    (region,) = regionprops(mask.astype(np.uint8))
    return PatchProperties(
        label=crop.label,
        bbox=crop.bbox,
        area_pixels=area_pixels,
        area_m2=float(area_pixels * pixel_size_m**2),
        perimeter_m=float(_pixel_edge_perimeter(mask) * pixel_size_m),
        major_axis_length_m=float(region.axis_major_length * pixel_size_m),
    )


def measure_components(
    crops: Iterable[ComponentCrop], *, pixel_size_m: float
) -> tuple[PatchProperties, ...]:
    """Measure complete bounded crops with the CPU reference implementation."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    return tuple(
        _measure_component(crop, pixel_size_m=pixel_size_m) for crop in crops
    )


__all__ = ["PatchProperties", "measure_components"]

