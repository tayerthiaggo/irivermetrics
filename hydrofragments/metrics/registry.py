"""Metadata-only metric registry and dependency planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from hydrofragments.schema import (
    MetricDependency,
    MetricFamily,
    Statistic,
    ValueType,
    validate_metric_id,
)


class RegistryError(ValueError):
    """Raised when metric selection cannot be resolved."""


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    family: MetricFamily
    unit: str
    statistic: Statistic | None = None
    value_type: ValueType = ValueType.MONTHLY
    tier: str = "core"
    dependencies: tuple[MetricDependency, ...] = (MetricDependency.NONE,)

    def __post_init__(self) -> None:
        validate_metric_id(self.metric_id)
        if not self.unit:
            raise RegistryError(f"metric {self.metric_id} must declare a unit")


@dataclass(frozen=True)
class MetricSkip:
    metric_id: str
    reason: str


@dataclass(frozen=True)
class MetricPlan:
    selected: tuple[MetricSpec, ...]
    skipped: tuple[MetricSkip, ...]


_METRICS = (
    MetricSpec(
        "occurrence",
        MetricFamily.PERSISTENCE,
        "percent",
        value_type=ValueType.RASTER_SUMMARY,
        dependencies=(MetricDependency.VALIDITY,),
    ),
    MetricSpec(
        "refuge_area",
        MetricFamily.PERSISTENCE,
        "km2",
        dependencies=(MetricDependency.VALIDITY,),
    ),
    MetricSpec("apsec", MetricFamily.EXTENT, "percent"),
    MetricSpec(
        "number_of_pools",
        MetricFamily.FRAGMENTATION,
        "count",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "lpi",
        MetricFamily.FRAGMENTATION,
        "percent",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "awre",
        MetricFamily.MORPHOLOGY,
        "dimensionless",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "awmsi",
        MetricFamily.MORPHOLOGY,
        "dimensionless",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "recurrence",
        MetricFamily.PERSISTENCE,
        "percent",
        tier="pixel_temporal",
        dependencies=(MetricDependency.VALIDITY,),
    ),
    MetricSpec(
        "hydroperiod",
        MetricFamily.PERSISTENCE,
        "fraction",
        tier="pixel_temporal",
        dependencies=(MetricDependency.VALIDITY,),
    ),
    MetricSpec(
        "extent_contraction",
        MetricFamily.DYNAMICS,
        "percent_per_month",
        value_type=ValueType.HY_SUMMARY,
        tier="dynamics",
        dependencies=(MetricDependency.HY_ANCHOR, MetricDependency.DUAL_COMPOSITE),
    ),
    MetricSpec(
        "reconnection_timing",
        MetricFamily.DYNAMICS,
        "month",
        value_type=ValueType.HY_SUMMARY,
        tier="dynamics",
        dependencies=(MetricDependency.HY_ANCHOR,),
    ),
    MetricSpec(
        "refuge_spatial_stability",
        MetricFamily.DYNAMICS,
        "dimensionless",
        value_type=ValueType.HY_SUMMARY,
        tier="dynamics",
        dependencies=(MetricDependency.HY_ANCHOR,),
    ),
    MetricSpec(
        "lpsec",
        MetricFamily.EXTENT,
        "percent",
        tier="channel",
        dependencies=(MetricDependency.CHANNEL,),
    ),
    MetricSpec(
        "inter_pool_gap",
        MetricFamily.CLUSTERING,
        "km",
        statistic=Statistic.MEAN,
        tier="channel",
        dependencies=(MetricDependency.CHANNEL,),
    ),
    MetricSpec(
        "mesh",
        MetricFamily.FRAGMENTATION,
        "km2",
        tier="secondary",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "pool_width",
        MetricFamily.MORPHOLOGY,
        "m",
        statistic=Statistic.MEAN,
        tier="secondary",
        dependencies=(MetricDependency.PATCHES,),
    ),
    MetricSpec(
        "realised_connectivity",
        MetricFamily.CONNECTIVITY,
        "dimensionless",
        tier="connectivity",
        dependencies=(MetricDependency.FIXED_NODES, MetricDependency.GRAPH),
    ),
    MetricSpec(
        "tcf",
        MetricFamily.CONNECTIVITY,
        "percent",
        tier="connectivity",
        dependencies=(
            MetricDependency.FIXED_NODES,
            MetricDependency.GRAPH,
            MetricDependency.VALIDITY,
        ),
    ),
)

METRIC_REGISTRY = {spec.metric_id: spec for spec in _METRICS}

PROFILES = {
    "contracts_core": (
        "occurrence",
        "refuge_area",
        "apsec",
        "number_of_pools",
        "lpi",
        "awre",
        "awmsi",
    ),
    "pixel_temporal": ("recurrence", "hydroperiod"),
    "dynamics": (
        "extent_contraction",
    ),
    "channel": ("lpsec", "inter_pool_gap"),
    "secondary": ("mesh", "pool_width"),
    "connectivity": ("realised_connectivity", "tcf"),
}


def _dependencies(values: Iterable[MetricDependency | str]) -> set[MetricDependency]:
    try:
        return {MetricDependency(value) for value in values}
    except ValueError as error:
        raise RegistryError(f"unknown metric dependency: {error}") from error


def resolve_metrics(
    profiles: Iterable[str],
    *,
    available_dependencies: Iterable[MetricDependency | str],
) -> MetricPlan:
    available = _dependencies(available_dependencies)
    metric_ids: list[str] = []
    for profile in profiles:
        if profile not in PROFILES:
            raise RegistryError(f"unknown metric profile: {profile}")
        for metric_id in PROFILES[profile]:
            if metric_id not in metric_ids:
                metric_ids.append(metric_id)

    selected: list[MetricSpec] = []
    skipped: list[MetricSkip] = []
    for metric_id in metric_ids:
        spec = METRIC_REGISTRY[metric_id]
        required = set(spec.dependencies) - {MetricDependency.NONE}
        missing = sorted(required - available, key=lambda item: item.value)
        if missing:
            reason = "missing dependencies: " + ", ".join(
                item.value for item in missing
            )
            skipped.append(MetricSkip(metric_id=metric_id, reason=reason))
        else:
            selected.append(spec)
    return MetricPlan(selected=tuple(selected), skipped=tuple(skipped))


__all__ = [
    "METRIC_REGISTRY",
    "PROFILES",
    "MetricPlan",
    "MetricSkip",
    "MetricSpec",
    "RegistryError",
    "resolve_metrics",
]
