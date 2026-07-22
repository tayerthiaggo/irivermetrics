from __future__ import annotations

from copy import deepcopy

import pytest


def manifest() -> dict[str, object]:
    return {
        "run_id": "run-left",
        "comparison": {
            "aoi_id": "reach-01",
            "source": "wofs",
            "resolution_m": 30.0,
            "crs": "EPSG:3577",
            "validity_policy": "p_native_season_stratified_v1",
            "monthly_composite": "supplied",
        },
        "resolved_config": {
            "input": {
                "water_threshold": None,
                "threshold_method": None,
            },
            "validity": {
                "policy": "p_native_season_stratified_v1",
                "min_valid_obs": 20,
                "min_valid_fraction_month": 0.70,
            },
            "spatial": {
                "target_crs": "EPSG:3577",
                "area_method": "projected",
            },
            "patches": {
                "min_patch_pixels": 3,
                "connectivity_rule": 8,
            },
            "temporal": {"monthly_composite": "supplied"},
        },
    }


@pytest.mark.parametrize(
    ("path", "different", "field"),
    [
        (("comparison", "source"), "sentinel-2", "source"),
        (("comparison", "resolution_m"), 10.0, "resolution_m"),
        (("comparison", "aoi_id"), "reach-02", "aoi_id"),
        (
            ("comparison", "validity_policy"),
            "p_provenance_v1",
            "validity_policy",
        ),
        (("comparison", "monthly_composite"), "max_water", "monthly_composite"),
        (("comparison", "crs"), "EPSG:6933", "crs"),
        (
            ("resolved_config", "patches", "min_patch_pixels"),
            5,
            "min_patch_pixels",
        ),
        (
            ("resolved_config", "patches", "connectivity_rule"),
            4,
            "connectivity_rule",
        ),
    ],
)
def test_comparison_rejects_scientific_mismatch_by_default(
    path: tuple[str, ...], different: object, field: str
) -> None:
    from hydrofragments.guards.comparison import (
        ComparisonGuardError,
        guard_comparison,
    )

    left = manifest()
    right = deepcopy(left)
    right["run_id"] = "run-right"
    target = right
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = different  # type: ignore[index]

    with pytest.raises(ComparisonGuardError, match=field):
        guard_comparison(left, right)


def test_comparison_rejects_missing_required_context() -> None:
    from hydrofragments.guards.comparison import (
        ComparisonGuardError,
        guard_comparison,
    )

    right = manifest()
    del right["comparison"]["resolution_m"]  # type: ignore[index]

    with pytest.raises(ComparisonGuardError, match="resolution_m.*missing"):
        guard_comparison(manifest(), right)


def test_explicit_override_is_reasoned_and_recorded() -> None:
    from hydrofragments.guards.comparison import guard_comparison

    right = manifest()
    right["run_id"] = "run-right"
    right["comparison"]["source"] = "sentinel-2"  # type: ignore[index]

    approval = guard_comparison(
        manifest(),
        right,
        overrides={
            "source": "cross-sensor sensitivity analysis; not direct trend"
        },
    )

    assert approval.approved is True
    assert approval.left_run_id == "run-left"
    assert approval.right_run_id == "run-right"
    assert approval.overrides == {
        "source": "cross-sensor sensitivity analysis; not direct trend"
    }
    assert approval.to_mapping()["mismatches"] == [
        {
            "field": "source",
            "left": "wofs",
            "right": "sentinel-2",
            "override_reason": (
                "cross-sensor sensitivity analysis; not direct trend"
            ),
        }
    ]


def test_override_without_reason_is_rejected() -> None:
    from hydrofragments.guards.comparison import (
        ComparisonGuardError,
        guard_comparison,
    )

    with pytest.raises(ComparisonGuardError, match="non-empty reason"):
        guard_comparison(manifest(), manifest(), overrides={"source": ""})


def test_unknown_override_field_is_rejected() -> None:
    from hydrofragments.guards.comparison import (
        ComparisonGuardError,
        guard_comparison,
    )

    with pytest.raises(ComparisonGuardError, match="unknown comparison override"):
        guard_comparison(
            manifest(),
            manifest(),
            overrides={"weather": "analyst accepts mismatch"},
        )
