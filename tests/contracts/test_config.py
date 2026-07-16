from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest


def minimal_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "config_schema_version": "1.2.0",
        "input": {"kind": "generic_binary"},
        "temporal": {
            "input_cadence": "monthly",
            "monthly_composite": "supplied",
            "composite_owner": "caller",
        },
    }
    config.update(overrides)
    return config


def test_unknown_top_level_config_key_is_rejected() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*mystery"):
        HydroConfig.from_mapping(minimal_config(mystery=True))


def test_unknown_nested_config_key_is_rejected_with_its_path() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match=r"unknown config key.*validity\.mystery"):
        HydroConfig.from_mapping(
            minimal_config(validity={"mystery": "not-scientific"})
        )


@pytest.mark.parametrize(
    "missing_key",
    ["water_threshold", "threshold_method", "probability_source"],
)
def test_probability_input_requires_threshold_provenance(missing_key: str) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    probability = {
        "kind": "generic_probability",
        "water_threshold": 0.5,
        "threshold_method": "fixed",
        "probability_source": "model-v3",
    }
    probability.pop(missing_key)

    with pytest.raises(ConfigError, match=missing_key):
        HydroConfig.from_mapping(minimal_config(input=probability))


@pytest.mark.parametrize(
    "state",
    [
        {"enabled": True, "connectivity_metric": "LPI"},
        {"enabled": True, "connectivity_threshold": 50.0},
    ],
)
def test_enabled_state_requires_metric_and_threshold(
    state: dict[str, object],
) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="state"):
        HydroConfig.from_mapping(minimal_config(state=state))


def test_cuda_strict_requires_cuda_accelerator() -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="cuda_strict"):
        HydroConfig.from_mapping(
            minimal_config(compute={"accelerator": "auto", "cuda_strict": True})
        )


@pytest.mark.parametrize("floor", [0, -1, float("nan")])
def test_width_resolution_floor_must_be_positive_and_finite(floor: float) -> None:
    from hydrofragments.config import ConfigError, HydroConfig

    with pytest.raises(ConfigError, match="width_resolution_floor_pixels"):
        HydroConfig.from_mapping(
            minimal_config(patches={"width_resolution_floor_pixels": floor})
        )


def test_approved_validity_contract_is_the_resolved_default() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.validity.policy == "p_native_season_stratified_v1"
    assert config.validity.min_valid_obs == 20
    assert config.validity.min_valid_fraction_month == 0.70
    assert config.validity.low_support_behavior == "suppress_value"


def test_extent_contraction_defaults_are_locked() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    assert config.dynamics.contraction_method == "linear"
    assert config.dynamics.minimum_points == 3


def test_config_is_immutable() -> None:
    from hydrofragments.config import HydroConfig

    config = HydroConfig.from_mapping(minimal_config())

    with pytest.raises(FrozenInstanceError):
        config.run_label = "changed"  # type: ignore[misc]
