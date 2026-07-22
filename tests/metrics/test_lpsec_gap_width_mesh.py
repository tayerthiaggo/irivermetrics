from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

from hydrofragments.metrics.clustering import compute_inter_pool_gaps
from hydrofragments.metrics.extent import compute_lpsec
from hydrofragments.patches.components import extract_component_crops
from hydrofragments.patches.labels import label_components
from hydrofragments.patches.morphology import measure_components
from hydrofragments.schema import WarningFlag
from hydrofragments.spatial.context import (
    create_channel_context,
    create_spatial_context,
)


def _channel_context(length_m: float = 100.0):
    aoi = gpd.GeoDataFrame(geometry=[box(0, -1, length_m, 1)], crs="EPSG:3577")
    drainage = gpd.GeoDataFrame(
        {
            "HydroID": [1],
            "From_Node": [10],
            "To_Node": [11],
            "NextDownID": [-1],
        },
        geometry=[LineString([(0, 0), (length_m, 0)])],
        crs="EPSG:3577",
    )
    return create_channel_context(
        "channel-aoi",
        aoi,
        drainage,
        drainage_id="synthetic-v1",
        target_crs="EPSG:3577",
    )


def test_lpsec_uses_real_fixed_l_ref_and_allows_braided_values_over_100() -> None:
    result = compute_lpsec(125.0, context=_channel_context(100.0))

    assert result.value == pytest.approx(125.0)
    assert result.wetted_length_m == 125.0
    assert result.l_ref_m == pytest.approx(100.0)
    assert result.exceeds_reference is True


def test_lpsec_refuses_context_without_real_drainage() -> None:
    context = create_spatial_context("no-channel", area_m2=100.0)

    with pytest.raises(ValueError, match="real channel|L_ref"):
        compute_lpsec(10.0, context=context)


def test_inter_pool_gap_preserves_order_and_sums_bounded_dry_runs() -> None:
    result = compute_inter_pool_gaps(
        wet=[True, True, False, False, True, False, True, True],
        segment_lengths_m=[10.0] * 8,
        threshold_m=15.0,
    )

    assert result.gaps_m == pytest.approx((20.0, 10.0))
    assert result.mean_m == pytest.approx(15.0)
    assert result.median_m == pytest.approx(15.0)
    assert result.max_m == pytest.approx(20.0)
    assert result.cv == pytest.approx(1.0 / 3.0)
    assert result.percent_above_threshold == pytest.approx(50.0)


def test_inter_pool_gap_ignores_unbounded_leading_and_trailing_dry_runs() -> None:
    result = compute_inter_pool_gaps(
        wet=[False, True, False, False, True, False],
        segment_lengths_m=[2, 2, 3, 4, 2, 100],
    )

    assert result.gaps_m == pytest.approx((7.0,))
    assert math.isnan(result.cv)


def test_mesh_uses_fixed_aoi_area_and_full_patch_distribution() -> None:
    from hydrofragments.metrics.patches import analyze_patch_metrics

    mask = np.zeros((4, 6), dtype=bool)
    mask[0:2, 0:2] = True  # 4 m2
    mask[3, 3:6] = True  # 3 m2

    result = analyze_patch_metrics(
        mask,
        pixel_size_m=1.0,
        a_total_m2=20.0,
        connectivity=4,
        min_patch_pixels=3,
        include_mesh=True,
    )

    assert result.mesh_m2 == pytest.approx((4.0**2 + 3.0**2) / 20.0)


def test_width_distribution_is_unweighted_and_suppresses_floor_bound_pools() -> None:
    from hydrofragments.metrics.patches import compute_pool_width_distribution

    mask = np.zeros((5, 9), dtype=bool)
    mask[0, 0:4] = True  # one-pixel bar: EDT width 2 pixels, suppressed
    mask[2:5, 6:9] = True  # 3x3 block: EDT medial width 4 pixels
    labels = label_components(mask, connectivity=4, min_patch_pixels=3)
    properties = measure_components(
        extract_component_crops(labels.labels),
        pixel_size_m=10.0,
        include_width=True,
    )

    result = compute_pool_width_distribution(
        properties,
        pixel_size_m=10.0,
        resolution_floor_pixels=2.0,
    )

    assert result.widths_m == pytest.approx((40.0,))
    assert result.mean_m == pytest.approx(40.0)
    assert result.median_m == pytest.approx(40.0)
    assert result.max_m == pytest.approx(40.0)
    assert math.isnan(result.cv)
    assert result.suppressed_pools == 1
    assert WarningFlag.WIDTH_RESOLUTION_FLOOR in result.warning_flags


def test_width_distribution_requires_explicit_approved_floor() -> None:
    from hydrofragments.metrics.patches import compute_pool_width_distribution

    with pytest.raises(ValueError, match="resolution_floor_pixels"):
        compute_pool_width_distribution((), pixel_size_m=30.0, resolution_floor_pixels=None)


def test_mesh_gate_disables_mesh_above_hard_correlation_threshold() -> None:
    from hydrofragments.metrics.patches import evaluate_mesh_correlation_gate

    gate = evaluate_mesh_correlation_gate(
        lpi=[1.0, 2.0, 3.0, 4.0],
        mesh=[2.0, 4.0, 6.0, 8.0],
    )

    assert gate.enabled is False
    assert gate.correlation == pytest.approx(1.0)
    assert "0.9" in gate.reason


def test_mesh_gate_keeps_mesh_when_nonredundancy_is_demonstrated() -> None:
    from hydrofragments.metrics.patches import evaluate_mesh_correlation_gate

    gate = evaluate_mesh_correlation_gate(
        lpi=[1.0, 2.0, 3.0, 4.0],
        mesh=[1.0, 4.0, 2.0, 3.0],
    )

    assert gate.enabled is True
    assert abs(gate.correlation) <= 0.9
