"""Milestone 9 — hydroseason version/config provenance in the run manifest.

Plan requirement (implementation_plan.md §M9 "External dependency"):
record `hydroseason` version and passed `HydroYearConfig` in run
config/manifest, so any HY-dependent run is fully reproducible without
re-deriving which detector version/config produced its anchors.
"""
from __future__ import annotations

from datetime import datetime, timezone

import hydroseason
from hydrofragments.config import HydroConfig
from hydrofragments.output.manifest import build_run_manifest


def _config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "hydroyear": {
                "algorithm": "hydroseason.detect_hydrological_years",
                "parameters": {
                    "wet_start_month": 11,
                    "wet_end_month": 4,
                    "dry_start_month": 7,
                    "dry_end_month": 12,
                },
            },
        }
    )


def _base_arguments() -> dict[str, object]:
    return {
        "run_id": "run-hy-001",
        "package_version": "1.2.0",
        "git_sha": "abc123",
        "created_at": datetime(2026, 7, 16, 0, 0, 0, tzinfo=timezone.utc),
        "input_fingerprint": {"digest": "sha256:input"},
        "planned_backend": "cpu",
        "actual_backend_by_stage": {"monthly": "cpu"},
    }


def test_manifest_records_hydroseason_version_when_supplied():
    manifest = build_run_manifest(
        _config(),
        **_base_arguments(),
        dependency_versions={"hydroseason": "0.1.1"},
    )
    assert manifest["versions"] == {
        "package_version": "1.2.0",
        "git_sha": "abc123",
        "hydroseason": "0.1.1",
    }


def test_manifest_records_hydroseason_version_automatically():
    manifest = build_run_manifest(_config(), **_base_arguments())

    assert manifest["versions"]["hydroseason"] == hydroseason.__version__


def test_manifest_hydroyear_config_appears_in_resolved_config():
    manifest = build_run_manifest(_config(), **_base_arguments())
    assert manifest["resolved_config"]["hydroyear"]["algorithm"] == (
        "hydroseason.detect_hydrological_years"
    )
    assert manifest["resolved_config"]["hydroyear"]["parameters"][
        "wet_start_month"
    ] == 11


def test_manifest_versions_unchanged_when_dependency_versions_omitted():
    manifest = build_run_manifest(_config(), **_base_arguments())
    assert manifest["versions"] == {
        "package_version": "1.2.0",
        "git_sha": "abc123",
        "hydroseason": hydroseason.__version__,
    }


def test_manifest_rejects_hydroseason_version_mismatch():
    import pytest
    from hydrofragments.output.manifest import ManifestError

    with pytest.raises(ManifestError, match="does not match installed"):
        build_run_manifest(
            _config(),
            **_base_arguments(),
            dependency_versions={"hydroseason": "9.9.9"},
        )
