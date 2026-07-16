"""Fixed-contract patch metrics and guarded secondary summaries.

LPI uses fixed AOI/landscape area. AWRe and AWMSI are area-weighted shape
summaries whose weights use retained patch area, as locked by spec sections
6.1-6.4. MESH remains registry-gated by real-data correlation. Pool widths are
unweighted planform morphology, never depth/storage claims, and require an
explicit resolution floor. AWRe forbids mixing skeleton and major-axis lengths
within one run.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from hydrofragments.patches.components import (
    bucket_component_crops,
    iter_component_crops,
)
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import PatchProperties, measure_components
from hydrofragments.schema import EdgeFlag, WarningFlag


@dataclass(frozen=True)
class PatchMetricResult:
    number_of_pools: int
    n_water_pixels: int
    lpi: float
    awre: float
    awmsi: float
    mesh_m2: float | None
    edge_flag: EdgeFlag | None
    awre_length_method: str


@dataclass(frozen=True)
class PoolWidthDistribution:
    widths_m: tuple[float, ...]
    mean_m: float
    median_m: float
    max_m: float
    cv: float
    suppressed_pools: int
    warning_flags: tuple[WarningFlag, ...]


@dataclass(frozen=True)
class MeshCorrelationGate:
    enabled: bool
    correlation: float
    threshold: float
    sample_size: int
    reason: str


def compute_pool_width_distribution(
    properties: Iterable[PatchProperties],
    *,
    pixel_size_m: float,
    resolution_floor_pixels: float | None,
) -> PoolWidthDistribution:
    """Summarise one EDT maximum-width observation per pool, unweighted."""
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if resolution_floor_pixels is None or resolution_floor_pixels <= 0:
        raise ValueError("resolution_floor_pixels must be explicitly positive")

    patches = tuple(properties)
    reliable = tuple(
        float(patch.width_m)
        for patch in patches
        if np.isfinite(patch.width_pixels)
        and patch.width_pixels > resolution_floor_pixels
    )
    suppressed = len(patches) - len(reliable)
    warnings = (
        (WarningFlag.WIDTH_RESOLUTION_FLOOR,) if suppressed else ()
    )
    if not reliable:
        nan = float("nan")
        return PoolWidthDistribution((), nan, nan, nan, nan, suppressed, warnings)

    widths = np.asarray(reliable, dtype=float)
    mean = float(widths.mean())
    cv = float(widths.std(ddof=0) / mean) if widths.size > 1 and mean > 0 else math.nan
    return PoolWidthDistribution(
        widths_m=reliable,
        mean_m=mean,
        median_m=float(np.median(widths)),
        max_m=float(widths.max()),
        cv=cv,
        suppressed_pools=suppressed,
        warning_flags=warnings,
    )


def evaluate_mesh_correlation_gate(
    *,
    lpi: Sequence[float],
    mesh: Sequence[float],
    threshold: float = 0.9,
) -> MeshCorrelationGate:
    """Enable MESH only when finite real-data Pearson r does not exceed gate."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    lpi_values = np.asarray(lpi, dtype=float)
    mesh_values = np.asarray(mesh, dtype=float)
    if lpi_values.ndim != 1 or mesh_values.ndim != 1:
        raise ValueError("lpi and mesh must be one-dimensional")
    if lpi_values.size != mesh_values.size:
        raise ValueError("lpi and mesh must have equal length")
    finite = np.isfinite(lpi_values) & np.isfinite(mesh_values)
    left = lpi_values[finite]
    right = mesh_values[finite]
    if left.size < 3 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return MeshCorrelationGate(
            enabled=False,
            correlation=float("nan"),
            threshold=threshold,
            sample_size=int(left.size),
            reason="MESH disabled: correlation gate requires at least 3 varying pairs",
        )
    correlation = float(np.corrcoef(left, right)[0, 1])
    enabled = correlation <= threshold
    reason = (
        f"MESH enabled: r={correlation:.6f} <= {threshold}"
        if enabled
        else f"MESH disabled: r={correlation:.6f} > {threshold}"
    )
    return MeshCorrelationGate(
        enabled=enabled,
        correlation=correlation,
        threshold=threshold,
        sample_size=int(left.size),
        reason=reason,
    )


def compute_patch_metrics(
    properties: Iterable[PatchProperties], *, a_total_m2: float, include_mesh: bool = False
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
            mesh_m2=float("nan") if include_mesh else None,
            edge_flag=EdgeFlag.N0,
            awre_length_method="major_axis",
        )

    methods = {patch.length_method for patch in patches}
    if len(methods) != 1 or not methods <= {"major_axis", "skeleton"}:
        raise ValueError(
            "AWRe length_method cannot mix skeleton and major_axis within one run"
        )
    (length_method,) = tuple(methods)

    areas = np.asarray([patch.area_m2 for patch in patches], dtype=float)
    perimeters = np.asarray(
        [patch.perimeter_m for patch in patches], dtype=float
    )
    if length_method == "skeleton":
        if any(patch.skeleton_length_m is None for patch in patches):
            raise ValueError("skeleton AWRe requires skeleton_length_m for every patch")
        lengths = np.asarray(
            [patch.skeleton_length_m for patch in patches], dtype=float
        )
    else:
        lengths = np.asarray(
            [patch.major_axis_length_m for patch in patches], dtype=float
        )
    total_wet_area = float(areas.sum())
    weights = areas / total_wet_area

    lpi = float(areas.max() / a_total_m2 * 100.0)
    awmsi = float(np.sum((0.25 * perimeters / np.sqrt(areas)) * weights))
    mesh_m2 = float(np.sum(areas**2) / a_total_m2) if include_mesh else None
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
        mesh_m2=mesh_m2,
        edge_flag=EdgeFlag.N1 if number_of_pools == 1 else None,
        awre_length_method=length_method,
    )


def analyze_patch_metrics(
    mask: Any,
    *,
    pixel_size_m: float,
    a_total_m2: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
    include_mesh: bool = False,
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
    return compute_patch_metrics(
        properties, a_total_m2=a_total_m2, include_mesh=include_mesh
    )


def analyze_pool_width_distribution(
    mask: Any,
    *,
    pixel_size_m: float,
    resolution_floor_pixels: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
) -> PoolWidthDistribution:
    """Run optional EDT width work without changing the core patch path."""
    labels = label_components(
        mask,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
    )
    properties: list[PatchProperties] = []
    for bucket in bucket_component_crops(
        iter_component_crops(labels.labels), target_pixels=target_component_pixels
    ):
        properties.extend(
            measure_components(
                bucket, pixel_size_m=pixel_size_m, include_width=True
            )
        )
    return compute_pool_width_distribution(
        properties,
        pixel_size_m=pixel_size_m,
        resolution_floor_pixels=resolution_floor_pixels,
    )


__all__ = [
    "PatchMetricResult",
    "MeshCorrelationGate",
    "PoolWidthDistribution",
    "analyze_patch_metrics",
    "analyze_pool_width_distribution",
    "compute_pool_width_distribution",
    "compute_patch_metrics",
    "evaluate_mesh_correlation_gate",
]
