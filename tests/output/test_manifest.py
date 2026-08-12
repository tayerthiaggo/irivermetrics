from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import hydroseason
import numpy as np
import pytest
import xarray as xr

from hydrofragments.config import HydroConfig


def config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
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
        "package_version": "0.1.0",
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
        "package_version": "0.1.0",
        "git_sha": "abc123",
        "hydroseason": hydroseason.__version__,
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
        package_version="0.1.0",
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


def _zone_mask(value: int = 2) -> np.ndarray:
    return np.array([[value, 0], [0, value]], dtype=np.uint8)


def _dea_provenance_arguments(
    *, mask: np.ndarray | None = None, with_planning_footprint: bool = True
) -> dict[str, object]:
    from hydrofragments.output.manifest import build_dea_provenance

    return {
        "dea_provenance": build_dea_provenance(
            config(),
            product="ga_ls_wo_fq_myear_3",
            version="0.1.0",
            item_ids=("ga_ls_wo_fq_myear_3_x11y40_2023--P1Y",),
            crs="EPSG:3577",
            resolution=30.0,
            time_span="1987-01-01/2025-12-31",
            zone_mask=mask if mask is not None else _zone_mask(),
            planning_footprint={
                "digest": "sha256:planning-footprint",
                "factor": 4,
                "safety_cells": 1,
                "covered_years": (2020, 2021, 2022),
                "source_collection": "ga_ls_wo_fq_myear_3",
                "source_version": "0.1.0",
                "source_lineage": "dea-wo-stats",
            }
            if with_planning_footprint
            else None,
        )
    }


def test_manifest_without_dea_provenance_has_no_dea_section() -> None:
    from hydrofragments.output.manifest import build_run_manifest

    manifest = build_run_manifest(config(), **manifest_arguments())

    assert "dea_provenance" not in manifest


def test_manifest_records_dea_provenance_when_supplied() -> None:
    from hydrofragments.output.manifest import build_run_manifest

    arguments = manifest_arguments()
    arguments.update(_dea_provenance_arguments())
    manifest = build_run_manifest(config(), **arguments)

    dea = manifest["dea_provenance"]
    assert dea["product"] == "ga_ls_wo_fq_myear_3"
    assert dea["version"] == "0.1.0"
    assert dea["item_ids"] == ["ga_ls_wo_fq_myear_3_x11y40_2023--P1Y"]
    assert dea["crs"] == "EPSG:3577"
    assert dea["resolution"] == 30.0
    assert dea["time_span"] == "1987-01-01/2025-12-31"
    assert dea["zone_thresholds"] == {"t_persist": 0.50, "t_season": 0.10}
    assert isinstance(dea["zone_mask_digest"], str) and len(dea["zone_mask_digest"]) == 64

    planning = dea["planning_footprint"]
    assert planning == {
        "digest": "sha256:planning-footprint",
        "factor": 4,
        "safety_cells": 1,
        "covered_years": [2020, 2021, 2022],
        "source_collection": "ga_ls_wo_fq_myear_3",
        "source_version": "0.1.0",
        "source_lineage": "dea-wo-stats",
    }


def test_manifest_dea_provenance_without_planning_footprint_omits_section() -> None:
    from hydrofragments.output.manifest import build_run_manifest

    arguments = manifest_arguments()
    arguments.update(
        _dea_provenance_arguments(with_planning_footprint=False)
    )
    manifest = build_run_manifest(config(), **arguments)

    dea = manifest["dea_provenance"]
    assert "planning_footprint" not in dea
    assert dea["product"] == "ga_ls_wo_fq_myear_3"


def test_manifest_zone_mask_digest_changes_with_mask_content() -> None:
    from hydrofragments.output.manifest import build_run_manifest

    arguments_a = manifest_arguments()
    arguments_a.update(_dea_provenance_arguments(mask=_zone_mask(2)))
    manifest_a = build_run_manifest(config(), **arguments_a)

    arguments_b = manifest_arguments()
    arguments_b.update(_dea_provenance_arguments(mask=_zone_mask(3)))
    manifest_b = build_run_manifest(config(), **arguments_b)

    digest_a = manifest_a["dea_provenance"]["zone_mask_digest"]
    digest_b = manifest_b["dea_provenance"]["zone_mask_digest"]
    assert digest_a != digest_b


def test_manifest_zone_mask_digest_stable_for_identical_content() -> None:
    from hydrofragments.output.manifest import build_run_manifest

    arguments_a = manifest_arguments()
    arguments_a.update(_dea_provenance_arguments(mask=_zone_mask(2)))
    manifest_a = build_run_manifest(config(), **arguments_a)

    arguments_b = manifest_arguments()
    arguments_b.update(_dea_provenance_arguments(mask=_zone_mask(2)))
    manifest_b = build_run_manifest(config(), **arguments_b)

    digest_a = manifest_a["dea_provenance"]["zone_mask_digest"]
    digest_b = manifest_b["dea_provenance"]["zone_mask_digest"]
    assert digest_a == digest_b


def test_manifest_dea_provenance_round_trips_through_bundle(
    tmp_path: Path,
) -> None:
    from hydrofragments.output.manifest import (
        validate_result_bundle,
        write_run_metadata,
    )

    (tmp_path / "metrics").mkdir()
    (tmp_path / "rasters").mkdir()
    arguments = manifest_arguments()
    arguments.update(_dea_provenance_arguments())
    write_run_metadata(tmp_path, config(), **arguments)

    manifest = validate_result_bundle(tmp_path)

    dea = manifest["dea_provenance"]
    assert dea["product"] == "ga_ls_wo_fq_myear_3"
    assert dea["planning_footprint"]["digest"] == "sha256:planning-footprint"
    assert dea["time_span"] == "1987-01-01/2025-12-31"


def test_manifest_without_dea_provenance_round_trips_with_no_section(
    tmp_path: Path,
) -> None:
    from hydrofragments.output.manifest import (
        validate_result_bundle,
        write_run_metadata,
    )

    (tmp_path / "metrics").mkdir()
    (tmp_path / "rasters").mkdir()
    write_run_metadata(tmp_path, config(), **manifest_arguments())

    manifest = validate_result_bundle(tmp_path)

    assert "dea_provenance" not in manifest
