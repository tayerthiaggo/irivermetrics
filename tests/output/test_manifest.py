from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig


def config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "compute": {
                "checkpoint": "zarr",
                "scheduler": "threads",
                "workers": 2,
            },
            "output": {
                "formats": ["parquet", "csv"],
                "include_vectors": False,
            },
        }
    )


def manifest_arguments() -> dict[str, object]:
    return {
        "run_id": "run-001",
        "package_version": "1.2.0",
        "git_sha": "abc123",
        "created_at": datetime(2026, 7, 15, 1, 2, 3, tzinfo=timezone.utc),
        "input_fingerprint": {
            "adapter": "generic_binary",
            "digest": "sha256:input",
        },
        "planned_backend": "cpu",
        "actual_backend_by_stage": {
            "monthly": "cpu",
            "patch_morphology": "cpu",
        },
        "skipped_metrics": [
            {"metric": "lpsec", "reason": "requires_channel"}
        ],
        "warnings": ["length_crs_caveat"],
        "timings_seconds": {"monthly": 1.25},
        "comparison_context": {
            "aoi_id": "reach-01",
            "source": "wofs",
            "resolution_m": 30.0,
            "crs": "EPSG:3577",
        },
        "artifacts": {
            "metrics": "metrics",
            "rasters": "rasters",
        },
    }


def test_build_manifest_contains_complete_reproducibility_context() -> None:
    from hydrofragments.output.manifest import build_run_manifest
    from hydrofragments.schema import SCHEMA_VERSION

    resolved = config()
    manifest = build_run_manifest(resolved, **manifest_arguments())

    assert manifest["manifest_schema_version"] == "1.0.0"
    assert manifest["output_schema_version"] == SCHEMA_VERSION
    assert manifest["run_id"] == "run-001"
    assert manifest["created_at"] == "2026-07-15T01:02:03Z"
    assert manifest["config_hash"] == resolved.config_hash
    assert manifest["execution_hash"] == resolved.execution_hash
    assert manifest["resolved_config"] == resolved.scientific_config()
    assert manifest["execution_config"] == resolved.execution_config()
    assert manifest["input_fingerprint"]["digest"] == "sha256:input"
    assert manifest["versions"] == {
        "package_version": "1.2.0",
        "git_sha": "abc123",
    }
    assert manifest["backend"] == {
        "planned": "cpu",
        "actual_by_stage": {
            "monthly": "cpu",
            "patch_morphology": "cpu",
        },
    }
    assert manifest["skipped_metrics"] == [
        {"metric": "lpsec", "reason": "requires_channel"}
    ]
    assert manifest["warnings"] == ["length_crs_caveat"]
    assert manifest["timings_seconds"] == {"monthly": 1.25}
    assert manifest["comparison"]["validity_policy"] == (
        "p_native_season_stratified_v1"
    )
    assert manifest["comparison"]["monthly_composite"] == "supplied"


def test_write_metadata_emits_canonical_config_and_manifest(tmp_path: Path) -> None:
    from hydrofragments.output.manifest import write_run_metadata

    resolved = config()
    artifacts = write_run_metadata(
        tmp_path,
        resolved,
        **manifest_arguments(),
    )

    assert artifacts.config_path == tmp_path / "config.json"
    assert artifacts.manifest_path == tmp_path / "run_manifest.json"
    assert json.loads(artifacts.config_path.read_text(encoding="utf-8")) == (
        resolved.scientific_config()
    )
    written = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert written["artifacts"] == {
        "config": "config.json",
        "manifest": "run_manifest.json",
        "metrics": "metrics",
        "rasters": "rasters",
    }
    assert artifacts.config_path.read_bytes().endswith(b"\n")
    assert artifacts.manifest_path.read_bytes().endswith(b"\n")


def test_validate_bundle_reopens_without_source_data(tmp_path: Path) -> None:
    from hydrofragments.output.manifest import (
        validate_result_bundle,
        write_run_metadata,
    )

    (tmp_path / "metrics").mkdir()
    (tmp_path / "rasters").mkdir()
    write_run_metadata(tmp_path, config(), **manifest_arguments())

    manifest = validate_result_bundle(tmp_path)

    assert manifest["run_id"] == "run-001"
    assert manifest["input_fingerprint"]["digest"] == "sha256:input"


def test_manifest_rejects_absolute_artifact_paths() -> None:
    from hydrofragments.output.manifest import ManifestError, build_run_manifest

    arguments = manifest_arguments()
    arguments["artifacts"] = {"metrics": str(Path.cwd() / "metrics")}

    with pytest.raises(ManifestError, match="relative"):
        build_run_manifest(config(), **arguments)


def test_bundle_validation_detects_config_tampering(tmp_path: Path) -> None:
    from hydrofragments.output.manifest import (
        ManifestError,
        validate_result_bundle,
        write_run_metadata,
    )

    (tmp_path / "metrics").mkdir()
    (tmp_path / "rasters").mkdir()
    artifacts = write_run_metadata(tmp_path, config(), **manifest_arguments())
    artifacts.config_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="config_hash"):
        validate_result_bundle(tmp_path)


def test_full_core_bundle_reopens_without_source_data(tmp_path: Path) -> None:
    from hydrofragments.models import MetricRecord
    from hydrofragments.output.manifest import (
        validate_result_bundle,
        write_run_metadata,
    )
    from hydrofragments.output.rasters import write_persistence_rasters
    from hydrofragments.output.tables import (
        read_tidy_parquet,
        write_output_tables,
    )
    from hydrofragments.schema import MetricFamily, ValueType

    resolved = config()
    record = MetricRecord(
        run_id="run-001",
        config_hash=resolved.config_hash,
        package_version="1.2.0",
        git_sha="abc123",
        catchment_id="fitzroy",
        aoi_id="reach-01",
        metric="apsec",
        metric_family=MetricFamily.EXTENT,
        value=12.5,
        unit="percent",
        value_type=ValueType.MONTHLY,
        is_reportable=True,
    )
    write_output_tables([record], tmp_path)
    rasters = xr.Dataset(
        {
            "occurrence": (("y", "x"), np.array([[100.0]])),
            "valid_count": (("y", "x"), np.array([[12]])),
            "refuge_mask": (("y", "x"), np.array([[True]])),
        },
        attrs={"validity_policy": resolved.validity.policy},
    )
    write_persistence_rasters(rasters, tmp_path / "rasters")
    write_run_metadata(tmp_path, resolved, **manifest_arguments())

    manifest = validate_result_bundle(tmp_path)
    table = read_tidy_parquet(tmp_path / "metrics")
    occurrence = xr.open_zarr(
        tmp_path / "rasters" / "occurrence", chunks=None
    )

    assert manifest["run_id"] == "run-001"
    assert table.loc[0, "metric"] == "apsec"
    assert occurrence["occurrence"].item() == 100.0
    occurrence.close()
