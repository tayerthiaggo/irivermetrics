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
from skimage.measure import regionprops
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


def _measure_component(
    crop: ComponentCrop, *, pixel_size_m: float, include_width: bool
) -> PatchProperties:
    mask = np.asarray(crop.mask, dtype=bool)
    area_pixels = int(np.count_nonzero(mask))
    if area_pixels == 0:
        raise ValueError(f"component {crop.label} has no foreground pixels")

    (region,) = regionprops(mask.astype(np.uint8))
    if include_width:
        axis = medial_axis(mask)
        width_pixels = float((2.0 * distance_transform_edt(mask)[axis]).max())
    else:
        width_pixels = float("nan")
    return PatchProperties(
        label=crop.label,
        bbox=crop.bbox,
        area_pixels=area_pixels,
        area_m2=float(area_pixels * pixel_size_m**2),
        perimeter_m=float(_pixel_edge_perimeter(mask) * pixel_size_m),
        major_axis_length_m=float(region.axis_major_length * pixel_size_m),
        width_m=width_pixels * pixel_size_m,
        width_pixels=width_pixels,
    )


def measure_components(
    crops: Iterable[ComponentCrop], *, pixel_size_m: float, include_width: bool = False
) -> tuple[PatchProperties, ...]:
    """Measure complete bounded crops with the CPU reference implementation."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    return tuple(
        _measure_component(
            crop, pixel_size_m=pixel_size_m, include_width=include_width
        )
        for crop in crops
    )


__all__ = ["PatchProperties", "measure_components"]
