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
        "m2",
        tier="secondary",
        dependencies=(
            MetricDependency.PATCHES,
            MetricDependency.MESH_VALIDATION,
        ),
    ),
    MetricSpec(
        "pool_width",
        MetricFamily.MORPHOLOGY,
        "m",
        statistic=Statistic.MEAN,
        tier="secondary",
        dependencies=(MetricDependency.PATCHES, MetricDependency.WIDTH_FLOOR),
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

# Metrics that are actually wired into hydrofragments.api.analyze()'s
# execution path and can be attempted by a default run. This deliberately
# excludes:
#   - "mesh": validation-gated (requires an approved mesh-correlation gate)
#     and never emitted by analyze() regardless of dependency availability.
#   - "reconnection_timing" / "refuge_spatial_stability": kernel-only, not
#     wired into analyze()'s execution path.
#   - "realised_connectivity" / "tcf": runtime-deferred, not wired into
#     analyze()'s execution path.
RUNTIME_WIRED_METRIC_IDS = (
    "occurrence",
    "refuge_area",
    "apsec",
    "number_of_pools",
    "lpi",
    "awre",
    "awmsi",
    "recurrence",
    "hydroperiod",
    "extent_contraction",
    "lpsec",
    "inter_pool_gap",
    "pool_width",
)

# Explicit, non-runtime-wired skip reasons for every registry metric that is
# not in RUNTIME_WIRED_METRIC_IDS. Used to populate HydroResult.metric_coverage
# rows for registry entries that a default "all_available" run never attempts.
NOT_RUNTIME_WIRED_REASONS = {
    "mesh": "skipped (validation disabled)",
    "reconnection_timing": "skipped (not runtime wired)",
    "refuge_spatial_stability": "skipped (not runtime wired)",
    "realised_connectivity": "skipped (runtime deferred)",
    "tcf": "skipped (runtime deferred)",
}

# Special profile value meaning "every runtime-wired metric whose dependencies
# are present in this run's inputs" -- see resolve_metrics().
ALL_AVAILABLE_PROFILE = "all_available"

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
    ALL_AVAILABLE_PROFILE: RUNTIME_WIRED_METRIC_IDS,
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


def registry_wide_plan(
    *, available_dependencies: Iterable[MetricDependency | str]
) -> MetricPlan:
    """Resolve every registry metric (not just a caller-chosen profile).

    Used to build ``HydroResult.metric_coverage``, which must include one row
    per registry entry -- runtime-wired or not -- rather than only the
    metrics a particular profile selects. Runtime-wired metrics are
    ``selected`` or ``skipped`` (missing dependency) exactly as
    :func:`resolve_metrics` would report for the ``all_available`` profile.
    Registry metrics outside :data:`RUNTIME_WIRED_METRIC_IDS` are always
    reported skipped, with the explicit non-dependency reason from
    :data:`NOT_RUNTIME_WIRED_REASONS` -- never "missing dependency", since
    their absence from a default run is a wiring/validation decision, not a
    property of this run's inputs.
    """
    plan = resolve_metrics(
        (ALL_AVAILABLE_PROFILE,), available_dependencies=available_dependencies
    )
    skipped = list(plan.skipped)
    for metric_id, reason in NOT_RUNTIME_WIRED_REASONS.items():
        skipped.append(MetricSkip(metric_id=metric_id, reason=reason))
    return MetricPlan(selected=plan.selected, skipped=tuple(skipped))


__all__ = [
    "ALL_AVAILABLE_PROFILE",
    "METRIC_REGISTRY",
    "NOT_RUNTIME_WIRED_REASONS",
    "PROFILES",
    "RUNTIME_WIRED_METRIC_IDS",
    "MetricPlan",
    "MetricSkip",
    "MetricSpec",
    "RegistryError",
    "registry_wide_plan",
    "resolve_metrics",
]
