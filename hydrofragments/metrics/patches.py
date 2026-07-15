"""Minimal fixed-contract patch metrics: N, LPI, AWRe, and AWMSI.

LPI uses fixed AOI/landscape area. AWRe and AWMSI are area-weighted shape
summaries whose weights use retained patch area, as locked by spec sections
6.1-6.3. The no-channel Milestone 6 path always reports
``awre_length_method='major_axis'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from hydrofragments.patches.components import (
    bucket_component_crops,
    iter_component_crops,
)
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import PatchProperties, measure_components
from hydrofragments.schema import EdgeFlag


@dataclass(frozen=True)
class PatchMetricResult:
    number_of_pools: int
    n_water_pixels: int
    lpi: float
    awre: float
    awmsi: float
    edge_flag: EdgeFlag | None
    awre_length_method: str


def compute_patch_metrics(
    properties: Iterable[PatchProperties], *, a_total_m2: float
) -> PatchMetricResult:
    """Aggregate compact patch properties using exact v1.2 formulas."""
    if a_total_m2 <= 0:
        raise ValueError("a_total_m2 must be positive")

    patches = tuple(properties)
    number_of_pools = len(patches)
    if number_of_pools == 0:
        return PatchMetricResult(
            number_of_pools=0,
            n_water_pixels=0,
            lpi=float("nan"),
            awre=float("nan"),
            awmsi=float("nan"),
            edge_flag=EdgeFlag.N0,
            awre_length_method="major_axis",
        )

    methods = {patch.length_method for patch in patches}
    if methods != {"major_axis"}:
        raise ValueError("Milestone 6 AWRe requires major_axis for every patch")

    areas = np.asarray([patch.area_m2 for patch in patches], dtype=float)
    perimeters = np.asarray(
        [patch.perimeter_m for patch in patches], dtype=float
    )
    lengths = np.asarray(
        [patch.major_axis_length_m for patch in patches], dtype=float
    )
    total_wet_area = float(areas.sum())
    weights = areas / total_wet_area

    lpi = float(areas.max() / a_total_m2 * 100.0)
    awmsi = float(np.sum((0.25 * perimeters / np.sqrt(areas)) * weights))
    if np.all(np.isfinite(lengths) & (lengths > 0)):
        elongation = 2.0 * np.sqrt(areas / np.pi) / lengths
        awre = float(np.sum(elongation * weights))
    else:
        awre = float("nan")

    return PatchMetricResult(
        number_of_pools=number_of_pools,
        n_water_pixels=int(sum(patch.area_pixels for patch in patches)),
        lpi=lpi,
        awre=awre,
        awmsi=awmsi,
        edge_flag=EdgeFlag.N1 if number_of_pools == 1 else None,
        awre_length_method="major_axis",
    )


def analyze_patch_metrics(
    mask: Any,
    *,
    pixel_size_m: float,
    a_total_m2: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
) -> PatchMetricResult:
    """Run the exact CPU reference pipeline for one monthly 2-D mask."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if a_total_m2 <= 0:
        raise ValueError("a_total_m2 must be positive")
    labels = label_components(
        mask,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
    )
    properties: list[PatchProperties] = []
    crops = iter_component_crops(labels.labels)
    for bucket in bucket_component_crops(
        crops, target_pixels=target_component_pixels
    ):
        properties.extend(
            measure_components(bucket, pixel_size_m=pixel_size_m)
        )
    return compute_patch_metrics(properties, a_total_m2=a_total_m2)


__all__ = [
    "PatchMetricResult",
    "analyze_patch_metrics",
    "compute_patch_metrics",
]
