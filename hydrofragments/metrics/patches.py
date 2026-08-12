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

from pathlib import Path

from hydrofragments.patches import (
    PatchProperties,
    bucket_component_crops,
    iter_component_crops,
    label_components,
    measure_components,
)
from hydrofragments.patches.labels import LabelCheckpointRef, label_components_to_checkpoint
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
class WindowLabelMeasureResult:
    """Labels/properties for one admitted window."""

    properties: tuple[PatchProperties, ...]
    labels: np.ndarray | None = None


def label_and_measure_window(
    mask: Any,
    *,
    pixel_size_m: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
    include_width: bool = False,
    local_label_threshold_bytes: int | None = None,
    max_component_bytes: int | None = None,
    window_id: str | None = None,
    spill_dir: Path | None = None,
) -> tuple[WindowLabelMeasureResult | None, LabelCheckpointRef | None]:
    """Label one window mask once and measure properties without retaining labels."""

    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if not np.any(mask):
        return None, None

    label_result, checkpoint = label_components_to_checkpoint(
        mask,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        local_label_threshold_bytes=local_label_threshold_bytes,
        spill_dir=spill_dir,
    )
    if checkpoint is not None:
        import zarr

        labels = zarr.open(checkpoint.path, mode="r")[:]
        properties: list[PatchProperties] = []
        crops = iter_component_crops(np.asarray(labels, dtype=np.int32))
        for bucket in bucket_component_crops(
            crops, target_pixels=target_component_pixels
        ):
            properties.extend(
                measure_components(
                    bucket,
                    pixel_size_m=pixel_size_m,
                    include_width=include_width,
                    max_component_bytes=max_component_bytes,
                    window_id=window_id,
                )
            )
        return WindowLabelMeasureResult(properties=tuple(properties), labels=None), checkpoint

    assert label_result is not None
    labels = label_result.labels
    properties: list[PatchProperties] = []
    crops = iter_component_crops(labels)
    for bucket in bucket_component_crops(crops, target_pixels=target_component_pixels):
        properties.extend(
            measure_components(
                bucket,
                pixel_size_m=pixel_size_m,
                include_width=include_width,
                window_id=window_id,
            )
        )
    return WindowLabelMeasureResult(
        properties=tuple(properties),
        labels=labels,
    ), checkpoint


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
    # MESH value is in m² by contract; registry unit label must match.
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


def measure_patch_properties(
    mask: Any,
    *,
    pixel_size_m: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
    include_width: bool = False,
    local_label_threshold_bytes: int | None = None,
) -> Sequence[PatchProperties]:
    """Label, crop, and measure one 2-D mask exactly once.

    ``local_label_threshold_bytes`` is forwarded verbatim to
    :func:`label_components` (``None`` preserves its own
    ``ComputePolicy``-derived default). This is the single point where a
    mask is labeled; callers that need properties from several independent
    windows call this once per window and concatenate the results before
    reducing, rather than reducing per window (see
    :mod:`hydrofragments.spatial.active_windows`).
    """
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    labels = label_components(
        mask,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        local_label_threshold_bytes=local_label_threshold_bytes,
    )
    properties: list[PatchProperties] = []
    crops = iter_component_crops(labels.labels)
    for bucket in bucket_component_crops(
        crops, target_pixels=target_component_pixels
    ):
        properties.extend(
            measure_components(
                bucket, pixel_size_m=pixel_size_m, include_width=include_width
            )
        )
    return properties


def reduce_patch_properties(
    properties: Sequence[PatchProperties],
    *,
    pixel_size_m: float,
    a_total_m2: float,
    include_mesh: bool = False,
    include_width: bool = False,
    resolution_floor_pixels: float | None = None,
) -> tuple[PatchMetricResult, PoolWidthDistribution | None]:
    """Aggregate already-measured properties into one core/width result.

    Callers concatenating properties from multiple independent windows must
    call this exactly once across the full concatenated sequence -- LPI,
    AWRe, AWMSI, width distribution, and counts are all computed once over
    every property, never per-window then combined.
    """
    if a_total_m2 <= 0:
        raise ValueError("a_total_m2 must be positive")
    core = compute_patch_metrics(
        properties, a_total_m2=a_total_m2, include_mesh=include_mesh
    )
    width = None
    if include_width:
        width = compute_pool_width_distribution(
            properties,
            pixel_size_m=pixel_size_m,
            resolution_floor_pixels=resolution_floor_pixels,
        )
    return core, width


def analyze_patch_bundle(
    mask: Any,
    *,
    pixel_size_m: float,
    a_total_m2: float,
    connectivity: int = 8,
    min_patch_pixels: int = 3,
    target_component_pixels: int = 1_000_000,
    include_mesh: bool = False,
    include_width: bool = False,
    resolution_floor_pixels: float | None = None,
    local_label_threshold_bytes: int | None = None,
) -> tuple[PatchMetricResult, PoolWidthDistribution | None]:
    """Label, crop, measure, and reduce one monthly 2-D mask in one call.

    Thin wrapper: :func:`measure_patch_properties` then
    :func:`reduce_patch_properties`, kept as one call for callers that only
    ever have a single whole-mask window and want the historical one-call
    shape. Core patch metrics are always produced. When ``include_width`` is
    set, the same measured ``properties`` (with EDT width already attached)
    are also reduced into a ``PoolWidthDistribution`` -- avoiding a second
    label/crop/measure pass over the same mask (M2).
    """
    if pixel_size_m <= 0:
        raise ValueError("pixel_size_m must be positive")
    if a_total_m2 <= 0:
        raise ValueError("a_total_m2 must be positive")
    properties = measure_patch_properties(
        mask,
        pixel_size_m=pixel_size_m,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        target_component_pixels=target_component_pixels,
        include_width=include_width,
        local_label_threshold_bytes=local_label_threshold_bytes,
    )
    return reduce_patch_properties(
        properties,
        pixel_size_m=pixel_size_m,
        a_total_m2=a_total_m2,
        include_mesh=include_mesh,
        include_width=include_width,
        resolution_floor_pixels=resolution_floor_pixels,
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
    core, _ = analyze_patch_bundle(
        mask,
        pixel_size_m=pixel_size_m,
        a_total_m2=a_total_m2,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        target_component_pixels=target_component_pixels,
        include_mesh=include_mesh,
        include_width=False,
    )
    return core


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
    # a_total_m2 only feeds core patch metrics (lpi/mesh), which are
    # discarded here; any positive placeholder keeps analyze_patch_bundle's
    # validation happy without affecting the returned width distribution.
    _, width = analyze_patch_bundle(
        mask,
        pixel_size_m=pixel_size_m,
        a_total_m2=1.0,
        connectivity=connectivity,
        min_patch_pixels=min_patch_pixels,
        target_component_pixels=target_component_pixels,
        include_mesh=False,
        include_width=True,
        resolution_floor_pixels=resolution_floor_pixels,
    )
    return width


__all__ = [
    "PatchMetricResult",
    "MeshCorrelationGate",
    "PoolWidthDistribution",
    "analyze_patch_bundle",
    "analyze_patch_metrics",
    "analyze_pool_width_distribution",
    "compute_pool_width_distribution",
    "compute_patch_metrics",
    "evaluate_mesh_correlation_gate",
    "label_and_measure_window",
    "measure_patch_properties",
    "reduce_patch_properties",
]
