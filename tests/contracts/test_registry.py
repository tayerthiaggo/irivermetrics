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

