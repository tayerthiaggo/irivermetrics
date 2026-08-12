from __future__ import annotations

import pytest


def test_contracts_core_resolves_only_approved_core_metrics() -> None:
    from hydrofragments.metrics.registry import resolve_metrics

    plan = resolve_metrics(
        ("contracts_core",),
        available_dependencies={"requires_validity", "requires_patches"},
    )

    assert tuple(spec.metric_id for spec in plan.selected) == (
        "occurrence",
        "refuge_area",
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
    )
    assert plan.skipped == ()


def test_missing_dependencies_produce_explicit_skips() -> None:
    from hydrofragments.metrics.registry import resolve_metrics

    plan = resolve_metrics(("channel",), available_dependencies=set())

    assert plan.selected == ()
    assert {(skip.metric_id, skip.reason) for skip in plan.skipped} == {
        ("lpsec", "missing dependencies: requires_channel"),
        ("inter_pool_gap", "missing dependencies: requires_channel"),
    }


def test_only_metrics_with_missing_dependencies_are_skipped() -> None:
    from hydrofragments.metrics.registry import resolve_metrics

    plan = resolve_metrics(
        ("contracts_core",), available_dependencies={"requires_patches"}
    )

    assert tuple(spec.metric_id for spec in plan.selected) == (
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
    )
    assert tuple(skip.metric_id for skip in plan.skipped) == (
        "occurrence",
        "refuge_area",
    )
    assert all(
        skip.reason == "missing dependencies: requires_validity"
        for skip in plan.skipped
    )


def test_mesh_stays_disabled_until_correlation_gate_is_approved() -> None:
    from hydrofragments.metrics.registry import resolve_metrics

    blocked = resolve_metrics(
        ("secondary",), available_dependencies={"requires_patches"}
    )
    approved = resolve_metrics(
        ("secondary",),
        available_dependencies={
            "requires_patches",
            "requires_mesh_validation",
            "requires_width_floor",
        },
    )

    assert blocked.selected == ()
    assert {(skip.metric_id, skip.reason) for skip in blocked.skipped} == {
        ("mesh", "missing dependencies: requires_mesh_validation"),
        ("pool_width", "missing dependencies: requires_width_floor"),
    }
    assert tuple(spec.metric_id for spec in approved.selected) == (
        "mesh",
        "pool_width",
    )


@pytest.mark.parametrize(
    "metric_id",
    [
        "PF",
        "PLF",
        "AWMPA",
        "AWMPL",
        "AWMPW",
        "PCF",
        "NNI",
        "degree_centrality",
        "betweenness_centrality",
    ],
)
def test_registry_cannot_initialize_forbidden_metric(metric_id: str) -> None:
    from hydrofragments.metrics.registry import MetricSpec
    from hydrofragments.schema import MetricDependency, MetricFamily, SchemaError

    with pytest.raises(SchemaError, match="forbidden"):
        MetricSpec(
            metric_id=metric_id,
            family=MetricFamily.DIAGNOSTIC,
            unit="dimensionless",
            dependencies=(MetricDependency.NONE,),
        )


def test_unknown_profile_is_rejected() -> None:
    from hydrofragments.metrics.registry import RegistryError, resolve_metrics

    with pytest.raises(RegistryError, match="unknown metric profile"):
        resolve_metrics(("everything",), available_dependencies=set())


def test_dynamics_profile_includes_all_dynamics_metrics() -> None:
    from hydrofragments.metrics.registry import PROFILES

    assert PROFILES["dynamics"] == (
        "extent_contraction",
        "reconnection_timing",
        "refuge_spatial_stability",
    )


def test_all_available_includes_new_dynamics_metrics() -> None:
    from hydrofragments.metrics.registry import (
        ALL_AVAILABLE_PROFILE,
        PROFILES,
        RUNTIME_WIRED_METRIC_IDS,
    )

    assert "reconnection_timing" in PROFILES[ALL_AVAILABLE_PROFILE]
    assert "refuge_spatial_stability" in PROFILES[ALL_AVAILABLE_PROFILE]
    assert "reconnection_timing" in RUNTIME_WIRED_METRIC_IDS
    assert "refuge_spatial_stability" in RUNTIME_WIRED_METRIC_IDS


def test_obsolete_skip_reasons_removed_for_new_dynamics_metrics() -> None:
    from hydrofragments.metrics.registry import NOT_RUNTIME_WIRED_REASONS

    assert "reconnection_timing" not in NOT_RUNTIME_WIRED_REASONS
    assert "refuge_spatial_stability" not in NOT_RUNTIME_WIRED_REASONS


def test_metric_overrides_are_applied_before_dependency_resolution() -> None:
    from hydrofragments.config import MetricOverrides
    from hydrofragments.metrics.registry import resolve_metrics

    plan = resolve_metrics(
        ("contracts_core",),
        available_dependencies={
            "requires_validity",
            "requires_patches",
            "requires_HY_anchor",
            "requires_dual_composite",
        },
        metric_overrides=MetricOverrides(add=("extent_contraction",), remove=()),
    )

    assert tuple(spec.metric_id for spec in plan.selected) == (
        "occurrence",
        "refuge_area",
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
        "extent_contraction",
    )


def test_removing_lpi_suppresses_output_but_keeps_reconnection_support() -> None:
    from hydrofragments.config import MetricOverrides
    from hydrofragments.metrics.registry import resolve_metrics

    plan = resolve_metrics(
        ("dynamics",),
        available_dependencies={
            "requires_HY_anchor",
            "requires_dual_composite",
            "requires_patches",
        },
        metric_overrides=MetricOverrides(remove=("lpi",)),
    )

    selected_ids = {spec.metric_id for spec in plan.selected}
    internal_ids = {spec.metric_id for spec in plan.internal_support}

    assert "lpi" not in selected_ids
    assert "reconnection_timing" in selected_ids
    assert "lpi" in internal_ids
