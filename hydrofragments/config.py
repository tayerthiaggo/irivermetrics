"""Immutable, validated HydroFragments configuration contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised when a configuration mapping violates the public contract."""


class LowSupportBehavior(str, Enum):
    """Approved handling for values with insufficient valid observations."""

    SUPPRESS_VALUE = "suppress_value"
    EMIT_FLAGGED_VALUE = "emit_flagged_value"
    METRIC_SPECIFIC = "metric_specific"


@dataclass(frozen=True)
class InputConfig:
    kind: str
    variable_map: tuple[tuple[str, str], ...] = ()
    water_threshold: float | None = None
    threshold_method: str | None = None
    probability_source: str | None = None


@dataclass(frozen=True)
class ValidityConfig:
    policy: str = "p_native_season_stratified_v1"
    min_valid_obs: int = 20
    min_valid_fraction_month: float = 0.70
    low_support_behavior: str = LowSupportBehavior.SUPPRESS_VALUE.value


@dataclass(frozen=True)
class WindowingConfig:
    mode: str = "none"
    length_m: float | None = None


@dataclass(frozen=True)
class SpatialConfig:
    target_crs: str = "EPSG:3577"
    area_method: str = "projected"
    windowing: WindowingConfig = field(default_factory=WindowingConfig)


@dataclass(frozen=True)
class PatchesConfig:
    min_patch_pixels: int = 3
    connectivity_rule: int = 8
    width_resolution_floor_pixels: float | None = None


@dataclass(frozen=True)
class PersistenceConfig:
    refuge_threshold: float = 0.90


@dataclass(frozen=True)
class ZonesConfig:
    t_persist: float = 0.50
    t_season: float = 0.10


@dataclass(frozen=True)
class TemporalConfig:
    input_cadence: str
    monthly_composite: str
    composite_owner: str


@dataclass(frozen=True)
class DynamicsConfig:
    composite_sensitivity_tolerance_pp: float = 10.0
    contraction_method: str = "linear"
    minimum_points: int = 3


@dataclass(frozen=True)
class HydroYearConfig:
    algorithm: str | None = None
    parameters: tuple[tuple[str, Any], ...] = ()


_HYDROSEASON_DEFAULT_PARAMETERS: dict[str, Any] = {
    "wet_start_month": 11,
    "wet_end_month": 4,
    "dry_start_month": 7,
    "dry_end_month": 12,
    "min_wet_months": 2,
    "min_dry_months": 2,
    "low_confidence_ratio": 0.25,
    "medium_confidence_ratio": 0.5,
}


@dataclass(frozen=True)
class ChannelConfig:
    source: str | None = None
    node_source: str | None = None


@dataclass(frozen=True)
class StateConfig:
    enabled: bool = False
    connectivity_metric: str | None = None
    connectivity_threshold: float | None = None


@dataclass(frozen=True)
class ConnectivityConfig:
    edge_rule: str | None = None


@dataclass(frozen=True)
class ComputeConfig:
    accelerator: str = "auto"
    cuda_strict: bool = False
    target_chunk_bytes: int | None = None
    worker_memory_fraction: float | None = None
    checkpoint: str = "zarr"
    scheduler: str = "local"
    workers: int = 1
    scheduler_address: str | None = None
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class OutputConfig:
    formats: tuple[str, ...] = ("parquet",)
    include_patch_table: bool = False
    include_vectors: bool = False
    output_dir: str | None = None


@dataclass(frozen=True)
class MetricOverrides:
    add: tuple[str, ...] = ()
    remove: tuple[str, ...] = ()
    reasons: tuple[tuple[str, str], ...] = ()


_TOP_LEVEL_KEYS = {
    "config_schema_version",
    "run_label",
    "metric_profiles",
    "metric_overrides",
    "input",
    "validity",
    "spatial",
    "patches",
    "persistence",
    "zones",
    "temporal",
    "dynamics",
    "hydroyear",
    "channel",
    "state",
    "connectivity",
    "compute",
    "output",
}


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{path} keys must be strings")
    return value


def _strict_section(
    source: Mapping[str, Any], path: str, allowed: set[str]
) -> Mapping[str, Any]:
    unknown = sorted(set(source) - allowed)
    if unknown:
        raise ConfigError(f"unknown config key: {path}.{unknown[0]}")
    return source


def _section(
    source: Mapping[str, Any], name: str, allowed: set[str]
) -> Mapping[str, Any]:
    return _strict_section(_mapping(source.get(name, {}), name), name, allowed)


def _required(source: Mapping[str, Any], key: str, path: str) -> Any:
    value = source.get(key)
    if value is None or value == "":
        raise ConfigError(f"{path}.{key} is required")
    return value


def _fraction(value: object, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{path} must be a number between 0 and 1") from error
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{path} must be between 0 and 1")
    return result


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ConfigError("configuration must contain JSON-compatible values") from error


def _sha256_json(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HydroConfig:
    """Resolved scientific and execution configuration."""

    config_schema_version: str
    input: InputConfig
    temporal: TemporalConfig
    run_label: str | None = None
    metric_profiles: tuple[str, ...] = ("contracts_core",)
    metric_overrides: MetricOverrides = field(default_factory=MetricOverrides)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    spatial: SpatialConfig = field(default_factory=SpatialConfig)
    patches: PatchesConfig = field(default_factory=PatchesConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    zones: ZonesConfig = field(default_factory=ZonesConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    hydroyear: HydroYearConfig = field(default_factory=HydroYearConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    state: StateConfig = field(default_factory=StateConfig)
    connectivity: ConnectivityConfig = field(default_factory=ConnectivityConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "HydroConfig":
        source = _strict_section(_mapping(raw, "config"), "config", _TOP_LEVEL_KEYS)

        config_schema_version = str(
            _required(source, "config_schema_version", "config")
        )

        input_raw = _section(
            source,
            "input",
            {
                "kind",
                "variable_map",
                "water_threshold",
                "threshold_method",
                "probability_source",
            },
        )
        kind = str(_required(input_raw, "kind", "input"))
        if kind not in {
            "watermask_tsfill",
            "generic_binary",
            "generic_probability",
        }:
            raise ConfigError(f"input.kind has unsupported value: {kind}")
        if kind == "generic_probability":
            for key in (
                "water_threshold",
                "threshold_method",
                "probability_source",
            ):
                _required(input_raw, key, "input")
        variable_map_raw = _mapping(input_raw.get("variable_map", {}), "input.variable_map")
        input_config = InputConfig(
            kind=kind,
            variable_map=tuple(
                sorted((str(key), str(value)) for key, value in variable_map_raw.items())
            ),
            water_threshold=(
                None
                if input_raw.get("water_threshold") is None
                else _fraction(input_raw["water_threshold"], "input.water_threshold")
            ),
            threshold_method=input_raw.get("threshold_method"),
            probability_source=input_raw.get("probability_source"),
        )

        validity_raw = _section(
            source,
            "validity",
            {
                "policy",
                "min_valid_obs",
                "min_valid_fraction_month",
                "low_support_behavior",
            },
        )
        min_valid_obs = int(validity_raw.get("min_valid_obs", 20))
        if min_valid_obs < 1:
            raise ConfigError("validity.min_valid_obs must be at least 1")
        low_support_behavior = str(
            validity_raw.get(
                "low_support_behavior", LowSupportBehavior.SUPPRESS_VALUE.value
            )
        )
        if low_support_behavior not in {item.value for item in LowSupportBehavior}:
            raise ConfigError(
                "validity.low_support_behavior has unsupported value: "
                f"{low_support_behavior}"
            )
        validity = ValidityConfig(
            policy=str(
                validity_raw.get("policy", "p_native_season_stratified_v1")
            ),
            min_valid_obs=min_valid_obs,
            min_valid_fraction_month=_fraction(
                validity_raw.get("min_valid_fraction_month", 0.70),
                "validity.min_valid_fraction_month",
            ),
            low_support_behavior=low_support_behavior,
        )

        windowing_raw = _strict_section(
            _mapping(
                _section(
                    source,
                    "spatial",
                    {"target_crs", "area_method", "windowing"},
                ).get("windowing", {}),
                "spatial.windowing",
            ),
            "spatial.windowing",
            {"mode", "length_m"},
        )
        spatial_raw = _section(
            source, "spatial", {"target_crs", "area_method", "windowing"}
        )
        window_mode = str(windowing_raw.get("mode", "none"))
        if window_mode not in {"none", "channel_length", "regular_grid"}:
            raise ConfigError(f"spatial.windowing.mode has unsupported value: {window_mode}")
        window_length = windowing_raw.get("length_m")
        if window_mode == "channel_length" and window_length is None:
            window_length = 5000.0
        spatial = SpatialConfig(
            target_crs=str(spatial_raw.get("target_crs", "EPSG:3577")),
            area_method=str(spatial_raw.get("area_method", "projected")),
            windowing=WindowingConfig(
                mode=window_mode,
                length_m=None if window_length is None else float(window_length),
            ),
        )
        if spatial.area_method not in {"projected", "per_pixel"}:
            raise ConfigError(
                f"spatial.area_method has unsupported value: {spatial.area_method}"
            )

        patches_raw = _section(
            source,
            "patches",
            {
                "min_patch_pixels",
                "connectivity_rule",
                "width_resolution_floor_pixels",
            },
        )
        patches = PatchesConfig(
            min_patch_pixels=int(patches_raw.get("min_patch_pixels", 3)),
            connectivity_rule=int(patches_raw.get("connectivity_rule", 8)),
            width_resolution_floor_pixels=(
                None
                if patches_raw.get("width_resolution_floor_pixels") is None
                else float(patches_raw["width_resolution_floor_pixels"])
            ),
        )
        if patches.min_patch_pixels < 1:
            raise ConfigError("patches.min_patch_pixels must be at least 1")
        if patches.connectivity_rule not in {4, 8}:
            raise ConfigError("patches.connectivity_rule must be 4 or 8")
        if patches.width_resolution_floor_pixels is not None and (
            not math.isfinite(patches.width_resolution_floor_pixels)
            or patches.width_resolution_floor_pixels <= 0
        ):
            raise ConfigError(
                "patches.width_resolution_floor_pixels must be positive and finite"
            )

        persistence_raw = _section(
            source, "persistence", {"refuge_threshold"}
        )
        persistence = PersistenceConfig(
            refuge_threshold=_fraction(
                persistence_raw.get("refuge_threshold", 0.90),
                "persistence.refuge_threshold",
            )
        )

        zones_raw = _section(source, "zones", {"t_persist", "t_season"})
        zones = ZonesConfig(
            t_persist=_fraction(zones_raw.get("t_persist", 0.50), "zones.t_persist"),
            t_season=_fraction(zones_raw.get("t_season", 0.10), "zones.t_season"),
        )
        if zones.t_season >= zones.t_persist:
            raise ConfigError("zones.t_season must be less than zones.t_persist")

        temporal_raw = _section(
            source,
            "temporal",
            {"input_cadence", "monthly_composite", "composite_owner"},
        )
        temporal = TemporalConfig(
            input_cadence=str(_required(temporal_raw, "input_cadence", "temporal")),
            monthly_composite=str(
                _required(temporal_raw, "monthly_composite", "temporal")
            ),
            composite_owner=str(
                _required(temporal_raw, "composite_owner", "temporal")
            ),
        )
        if temporal.monthly_composite not in {
            "max_water",
            "median",
            "mode",
            "end_of_month_nearest",
            "supplied",
        }:
            raise ConfigError(
                "temporal.monthly_composite has unsupported value: "
                f"{temporal.monthly_composite}"
            )
        if temporal.composite_owner not in {"hydrofragments", "upstream", "caller"}:
            raise ConfigError(
                "temporal.composite_owner has unsupported value: "
                f"{temporal.composite_owner}"
            )

        dynamics_raw = _section(
            source,
            "dynamics",
            {
                "composite_sensitivity_tolerance_pp",
                "contraction_method",
                "minimum_points",
            },
        )
        dynamics = DynamicsConfig(
            composite_sensitivity_tolerance_pp=float(
                dynamics_raw.get("composite_sensitivity_tolerance_pp", 10.0)
            ),
            contraction_method=str(
                dynamics_raw.get("contraction_method") or "linear"
            ),
            minimum_points=int(
                dynamics_raw.get("minimum_points")
                if dynamics_raw.get("minimum_points") is not None
                else 3
            ),
        )
        if dynamics.contraction_method not in {"linear", "theil_sen"}:
            raise ConfigError(
                "dynamics.contraction_method must be 'linear' or 'theil_sen'"
            )
        if dynamics.minimum_points < 3:
            raise ConfigError("dynamics.minimum_points must be at least 3")

        hydroyear_raw = _section(source, "hydroyear", {"algorithm", "parameters"})
        hydroyear_parameters = _mapping(
            hydroyear_raw.get("parameters", {}), "hydroyear.parameters"
        )
        unknown_hydroseason = sorted(
            set(hydroyear_parameters) - set(_HYDROSEASON_DEFAULT_PARAMETERS)
        )
        if unknown_hydroseason:
            raise ConfigError(
                "unknown hydroseason config key: hydroyear.parameters."
                + unknown_hydroseason[0]
            )
        resolved_hydroyear_parameters = {
            **_HYDROSEASON_DEFAULT_PARAMETERS,
            **hydroyear_parameters,
        }
        hydroyear = HydroYearConfig(
            algorithm=(
                str(hydroyear_raw.get("algorithm"))
                if hydroyear_raw.get("algorithm")
                else "hydroseason.detect_hydrological_years"
            ),
            parameters=tuple(sorted(resolved_hydroyear_parameters.items())),
        )

        channel_raw = _section(source, "channel", {"source", "node_source"})
        channel = ChannelConfig(
            source=channel_raw.get("source"),
            node_source=channel_raw.get("node_source"),
        )

        state_raw = _section(
            source,
            "state",
            {"enabled", "connectivity_metric", "connectivity_threshold"},
        )
        state = StateConfig(
            enabled=bool(state_raw.get("enabled", False)),
            connectivity_metric=state_raw.get("connectivity_metric"),
            connectivity_threshold=(
                None
                if state_raw.get("connectivity_threshold") is None
                else float(state_raw["connectivity_threshold"])
            ),
        )
        if state.enabled and (
            state.connectivity_metric is None or state.connectivity_threshold is None
        ):
            raise ConfigError(
                "state.enabled requires state.connectivity_metric and "
                "state.connectivity_threshold"
            )
        if state.connectivity_metric is not None and state.connectivity_metric not in {
            "RC",
            "LPSEC",
            "LPI",
            "DCI",
        }:
            raise ConfigError(
                "state.connectivity_metric must be one of RC, LPSEC, LPI, DCI"
            )

        connectivity_raw = _section(source, "connectivity", {"edge_rule"})
        connectivity = ConnectivityConfig(edge_rule=connectivity_raw.get("edge_rule"))

        compute_raw = _section(
            source,
            "compute",
            {
                "accelerator",
                "cuda_strict",
                "target_chunk_bytes",
                "worker_memory_fraction",
                "checkpoint",
                "scheduler",
                "workers",
                "scheduler_address",
                "checkpoint_path",
            },
        )
        compute = ComputeConfig(
            accelerator=str(compute_raw.get("accelerator", "auto")),
            cuda_strict=bool(compute_raw.get("cuda_strict", False)),
            target_chunk_bytes=(
                None
                if compute_raw.get("target_chunk_bytes") is None
                else int(compute_raw["target_chunk_bytes"])
            ),
            worker_memory_fraction=(
                None
                if compute_raw.get("worker_memory_fraction") is None
                else _fraction(
                    compute_raw["worker_memory_fraction"],
                    "compute.worker_memory_fraction",
                )
            ),
            checkpoint=str(compute_raw.get("checkpoint", "zarr")),
            scheduler=str(compute_raw.get("scheduler", "local")),
            workers=int(compute_raw.get("workers", 1)),
            scheduler_address=compute_raw.get("scheduler_address"),
            checkpoint_path=compute_raw.get("checkpoint_path"),
        )
        if compute.accelerator not in {"none", "auto", "cuda"}:
            raise ConfigError(
                f"compute.accelerator has unsupported value: {compute.accelerator}"
            )
        if compute.cuda_strict and compute.accelerator != "cuda":
            raise ConfigError("compute.cuda_strict requires accelerator='cuda'")

        output_raw = _section(
            source,
            "output",
            {"formats", "include_patch_table", "include_vectors", "output_dir"},
        )
        output = OutputConfig(
            formats=tuple(sorted(set(output_raw.get("formats", ("parquet",))))),
            include_patch_table=bool(output_raw.get("include_patch_table", False)),
            include_vectors=bool(output_raw.get("include_vectors", False)),
            output_dir=output_raw.get("output_dir"),
        )

        profiles_raw = source.get("metric_profiles", ("contracts_core",))
        if isinstance(profiles_raw, str):
            raise ConfigError("metric_profiles must be a sequence, not a string")
        metric_profiles = tuple(sorted(set(str(item) for item in profiles_raw)))
        if not metric_profiles:
            raise ConfigError("metric_profiles must not be empty")

        overrides_raw = _section(
            source, "metric_overrides", {"add", "remove", "reasons"}
        )
        reasons_raw = _mapping(
            overrides_raw.get("reasons", {}), "metric_overrides.reasons"
        )
        metric_overrides = MetricOverrides(
            add=tuple(sorted(set(overrides_raw.get("add", ())))),
            remove=tuple(sorted(set(overrides_raw.get("remove", ())))),
            reasons=tuple(
                sorted((str(key), str(value)) for key, value in reasons_raw.items())
            ),
        )

        return cls(
            config_schema_version=config_schema_version,
            input=input_config,
            temporal=temporal,
            run_label=source.get("run_label"),
            metric_profiles=metric_profiles,
            metric_overrides=metric_overrides,
            validity=validity,
            spatial=spatial,
            patches=patches,
            persistence=persistence,
            zones=zones,
            dynamics=dynamics,
            hydroyear=hydroyear,
            channel=channel,
            state=state,
            connectivity=connectivity,
            compute=compute,
            output=output,
        )

    def scientific_config(self) -> dict[str, Any]:
        """Return the canonical scientifically meaningful configuration."""

        return {
            "channel": {
                "node_source": self.channel.node_source,
                "source": self.channel.source,
            },
            "config_schema_version": self.config_schema_version,
            "connectivity": {"edge_rule": self.connectivity.edge_rule},
            "dynamics": {
                "composite_sensitivity_tolerance_pp": (
                    self.dynamics.composite_sensitivity_tolerance_pp
                ),
                "contraction_method": self.dynamics.contraction_method,
                "minimum_points": self.dynamics.minimum_points,
            },
            "hydroyear": {
                "algorithm": self.hydroyear.algorithm,
                "parameters": dict(self.hydroyear.parameters),
            },
            "input": {
                "kind": self.input.kind,
                "probability_source": self.input.probability_source,
                "threshold_method": self.input.threshold_method,
                "variable_map": dict(self.input.variable_map),
                "water_threshold": self.input.water_threshold,
            },
            "metric_overrides": {
                "add": list(self.metric_overrides.add),
                "reasons": dict(self.metric_overrides.reasons),
                "remove": list(self.metric_overrides.remove),
            },
            "metric_profiles": list(self.metric_profiles),
            "patches": {
                "connectivity_rule": self.patches.connectivity_rule,
                "min_patch_pixels": self.patches.min_patch_pixels,
                "width_resolution_floor_pixels": (
                    self.patches.width_resolution_floor_pixels
                ),
            },
            "persistence": {"refuge_threshold": self.persistence.refuge_threshold},
            "spatial": {
                "area_method": self.spatial.area_method,
                "target_crs": self.spatial.target_crs,
                "windowing": {
                    "length_m": self.spatial.windowing.length_m,
                    "mode": self.spatial.windowing.mode,
                },
            },
            "state": {
                "connectivity_metric": self.state.connectivity_metric,
                "connectivity_threshold": self.state.connectivity_threshold,
                "enabled": self.state.enabled,
            },
            "temporal": {
                "composite_owner": self.temporal.composite_owner,
                "input_cadence": self.temporal.input_cadence,
                "monthly_composite": self.temporal.monthly_composite,
            },
            "validity": {
                "low_support_behavior": self.validity.low_support_behavior,
                "min_valid_fraction_month": self.validity.min_valid_fraction_month,
                "min_valid_obs": self.validity.min_valid_obs,
                "policy": self.validity.policy,
            },
            "zones": {
                "t_persist": self.zones.t_persist,
                "t_season": self.zones.t_season,
            },
        }

    def execution_config(self) -> dict[str, Any]:
        """Return execution and output settings excluded from ``config_hash``."""

        return {
            "compute": {
                "accelerator": self.compute.accelerator,
                "checkpoint": self.compute.checkpoint,
                "checkpoint_path": self.compute.checkpoint_path,
                "cuda_strict": self.compute.cuda_strict,
                "scheduler": self.compute.scheduler,
                "scheduler_address": self.compute.scheduler_address,
                "target_chunk_bytes": self.compute.target_chunk_bytes,
                "worker_memory_fraction": self.compute.worker_memory_fraction,
                "workers": self.compute.workers,
            },
            "output": {
                "formats": list(self.output.formats),
                "include_patch_table": self.output.include_patch_table,
                "include_vectors": self.output.include_vectors,
                "output_dir": self.output.output_dir,
            },
        }

    @property
    def config_hash(self) -> str:
        return _sha256_json(self.scientific_config())

    @property
    def execution_hash(self) -> str:
        return _sha256_json(self.execution_config())


__all__ = [
    "ConfigError",
    "HydroConfig",
    "LowSupportBehavior",
]
