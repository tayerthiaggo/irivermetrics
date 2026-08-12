"""Atomic bundle finalizer and failure-injection tests."""

from __future__ import annotations

import json
import multiprocessing
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("rioxarray")

from hydrofragments.config import HydroConfig
from hydrofragments.models import MetricRecord
from hydrofragments.output.bundle import (
    ArtifactRegistration,
    BundleError,
    assert_output_dir_available,
    commit_staged_bundle,
    open_bundle_transaction,
    recover_or_remove_staging,
    staging_directory_for,
)
from hydrofragments.output.manifest import (
    LEGACY_MANIFEST_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    ManifestError,
    build_artifact_inventory,
    hash_directory_tree,
    hash_file,
    validate_result_bundle,
)
from hydrofragments.output.rasters import (
    RASTER_PRODUCT_CONTRACTS,
    RasterExportError,
    write_geotiff_from_dataarray,
)
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.output.tables import write_output_tables
from hydrofragments.schema import MetricFamily, ValueType


def _config() -> HydroConfig:
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
                "formats": ["parquet"],
                "include_vectors": False,
            },
        }
    )


def _manifest_kwargs() -> dict[str, object]:
    return {
        "package_version": "0.1.0",
        "git_sha": "abc123",
        "input_fingerprint": {"adapter": "generic_binary", "digest": "sha256:input"},
        "planned_backend": "cpu",
        "actual_backend_by_stage": {"monthly": "cpu"},
        "timings_seconds": {"analysis": 1.0, "finalization": 0.5},
        "peak_rss_bytes": 1024 * 1024,
    }


def _metric_record(config: HydroConfig) -> MetricRecord:
    return MetricRecord(
        run_id="run-001",
        config_hash=config.config_hash,
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


def _grid(shape: tuple[int, int] = (2, 3)) -> SpatialGrid:
    import xarray as xr

    y = np.linspace(100.0, 40.0, shape[0])
    x = np.linspace(10.0, 70.0, shape[1])
    da = xr.DataArray(np.zeros(shape, dtype=float), dims=("y", "x"), coords={"y": y, "x": x})
    da = da.rio.write_crs("EPSG:3577")
    return SpatialGrid.from_dataarray(da, require_georeference=True)


def _publish_minimal_bundle(output_dir: Path) -> None:
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    try:
        write_output_tables([_metric_record(config)], transaction.root)
        transaction.write_config()
        transaction.register_artifact(
            ArtifactRegistration(name="metrics", relative_path="metrics")
        )
        transaction.finalize(**_manifest_kwargs())
    except Exception:
        transaction.abort()
        raise


def test_staging_directory_name_is_deterministic(tmp_path: Path) -> None:
    target = tmp_path / "my_run"
    staging = staging_directory_for(target, "run-001")
    assert staging == tmp_path / "my_run.run-001.staging"


def test_rejects_non_empty_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "my_run"
    target.mkdir()
    (target / "leftover.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleError, match="empty"):
        assert_output_dir_available(target)


def test_atomic_publish_creates_valid_manifest_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    _publish_minimal_bundle(output_dir)

    assert output_dir.exists()
    assert not (tmp_path / "bundle.run-001.staging").exists()
    manifest = validate_result_bundle(output_dir)
    assert manifest["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["peak_rss_bytes"] == 1024 * 1024
    inventory = manifest["artifact_inventory"]
    metrics_entry = next(item for item in inventory if item["name"] == "metrics")
    assert metrics_entry["byte_size"] > 0
    assert len(metrics_entry["sha256"]) == 64
    assert metrics_entry["scientific_config_hash"] == _config().config_hash


def test_directory_artifact_uses_tree_digest(tmp_path: Path) -> None:
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    (metrics_dir / "part-000.parquet").write_bytes(b"abc")
    (metrics_dir / "nested" / "part-001.parquet").parent.mkdir()
    (metrics_dir / "nested" / "part-001.parquet").write_bytes(b"def")

    total, digest = hash_directory_tree(metrics_dir)
    assert total == 6
    assert len(digest) == 64


def test_failure_before_manifest_leaves_no_final_target(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    write_output_tables([_metric_record(config)], transaction.root)
    transaction.write_config()
    transaction.register_artifact(
        ArtifactRegistration(name="metrics", relative_path="metrics")
    )

    def _fail_before_manifest(_path: Path, _manifest: dict, _inventory: list) -> None:
        raise RuntimeError("validation failed")

    with pytest.raises(RuntimeError, match="validation failed"):
        transaction.finalize(**_manifest_kwargs(), validate_hook=_fail_before_manifest)

    transaction.abort()
    assert not output_dir.exists()
    assert not transaction.staging_dir.exists()


def test_failure_during_manifest_write_cleans_staging(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    write_output_tables([_metric_record(config)], transaction.root)
    transaction.write_config()
    transaction.register_artifact(
        ArtifactRegistration(name="metrics", relative_path="metrics")
    )

    def _fail_on_manifest(path: Path, _manifest: dict) -> None:
        path.write_text("{}", encoding="utf-8")
        raise RuntimeError("manifest publication failed")

    with pytest.raises(RuntimeError, match="manifest publication failed"):
        transaction.finalize(**_manifest_kwargs(), manifest_write_hook=_fail_on_manifest)

    transaction.abort()
    assert not output_dir.exists()
    assert not transaction.staging_dir.exists()


def test_geotiff_window_failure_does_not_publish_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    grid = _grid()
    data = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    import xarray as xr

    occurrence = xr.DataArray(
        data,
        dims=("y", "x"),
        coords={"y": grid.y, "x": grid.x},
    ).rio.write_crs(grid.crs)
    raster_dir = transaction.root / "rasters"
    raster_dir.mkdir()

    original = __import__(
        "hydrofragments.output.rasters", fromlist=["_write_geotiff_windowed"]
    )._write_geotiff_windowed
    calls = {"count": 0}

    def _fail_halfway(**kwargs: object) -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RasterExportError("window loop failed")
        original(**kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "hydrofragments.output.rasters._write_geotiff_windowed",
        _fail_halfway,
    )
    try:
        with pytest.raises(RasterExportError, match="window loop failed"):
            write_geotiff_from_dataarray(
                occurrence,
                raster_dir / RASTER_PRODUCT_CONTRACTS["occurrence"].filename,
                grid=grid,
                contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
                source_name="occurrence",
                metadata={
                    "algorithm_version": "1.0.0",
                    "scientific_config_hash": config.config_hash,
                },
            )
    finally:
        monkeypatch.undo()

    assert not output_dir.exists()
    assert transaction.staging_dir.exists()


def test_inventory_includes_spatial_metadata_for_geotiff(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    grid = _grid()
    import xarray as xr

    occurrence = xr.DataArray(
        np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32),
        dims=("y", "x"),
        coords={"y": grid.y, "x": grid.x},
    ).rio.write_crs(grid.crs)
    raster_dir = transaction.root / "rasters"
    raster_dir.mkdir()
    write_geotiff_from_dataarray(
        occurrence,
        raster_dir / RASTER_PRODUCT_CONTRACTS["occurrence"].filename,
        grid=grid,
        contract=RASTER_PRODUCT_CONTRACTS["occurrence"],
        source_name="occurrence",
        metadata={
            "algorithm_version": "1.0.0",
            "scientific_config_hash": config.config_hash,
        },
    )
    import rasterio

    tif_path = raster_dir / RASTER_PRODUCT_CONTRACTS["occurrence"].filename
    with rasterio.open(tif_path) as dataset:
        spatial = {
            "crs": dataset.crs.to_wkt(),
            "shape": [grid.height, grid.width],
            "dtype": "float32",
            "band_count": dataset.count,
        }
    transaction.write_config()
    transaction.register_artifact(
        ArtifactRegistration(
            name="occurrence",
            relative_path="rasters/occurrence.tif",
            media_type="image/tiff",
            algorithm_version="1.0.0",
            spatial=spatial,
        )
    )
    transaction.finalize(**_manifest_kwargs())

    manifest = validate_result_bundle(output_dir)
    entry = next(
        item
        for item in manifest["artifact_inventory"]
        if item["relative_path"] == "rasters/occurrence.tif"
    )
    assert entry["spatial"]["band_count"] == 1


def test_recover_complete_staged_bundle(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    write_output_tables([_metric_record(config)], transaction.root)
    transaction.write_config()
    transaction.register_artifact(
        ArtifactRegistration(name="metrics", relative_path="metrics")
    )
    inventory = build_artifact_inventory(
        transaction.root,
        transaction._artifacts,
        config=config,
    )
    manifest_path = transaction.root / "run_manifest.json"
    from hydrofragments.output.manifest import build_run_manifest

    manifest = build_run_manifest(
        config,
        run_id="run-001",
        artifact_inventory=inventory,
        artifacts={"metrics": "metrics"},
        manifest_schema_version=MANIFEST_SCHEMA_VERSION,
        created_at=datetime(2026, 8, 12, tzinfo=timezone.utc),
        **_manifest_kwargs(),
    )
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    transaction._write_transaction_record("staged")
    staging = transaction.staging_dir

    recovered = recover_or_remove_staging(staging, target_dir=output_dir, run_id="run-001")
    assert recovered == output_dir
    assert output_dir.exists()
    validate_result_bundle(output_dir)


def test_incomplete_staging_is_removed_on_recovery(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    config = _config()
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    transaction.write_config()
    staging = transaction.staging_dir

    recover_or_remove_staging(staging, target_dir=output_dir, run_id="run-001")
    assert not staging.exists()
    assert not output_dir.exists()


def _subprocess_worker(
    tmp: str,
    *,
    phase: str,
) -> None:
    output_dir = Path(tmp) / "bundle"
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "output": {"formats": ["parquet"], "include_vectors": False},
        }
    )
    transaction = open_bundle_transaction(output_dir, run_id="run-001", config=config)
    write_output_tables(
        [
            MetricRecord(
                run_id="run-001",
                config_hash=config.config_hash,
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
        ],
        transaction.root,
    )
    transaction.write_config()
    transaction.register_artifact(
        ArtifactRegistration(name="metrics", relative_path="metrics")
    )
    if phase == "before_commit":
        transaction._write_transaction_record("staged")
        return
    transaction.finalize(
        package_version="0.1.0",
        git_sha="abc123",
        input_fingerprint={"digest": "sha256:input"},
        planned_backend="cpu",
        actual_backend_by_stage={"monthly": "cpu"},
    )


@pytest.mark.parametrize("phase", ["before_commit", "after_commit"])
def test_subprocess_boundary_leaves_expected_bundle_state(
    tmp_path: Path,
    phase: str,
) -> None:
    if phase == "before_commit":
        proc = multiprocessing.Process(
            target=_subprocess_worker,
            args=(str(tmp_path),),
            kwargs={"phase": phase},
        )
        proc.start()
        proc.join(timeout=30)
        assert not (tmp_path / "bundle").exists()
        assert (tmp_path / "bundle.run-001.staging").exists()
        return

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from tests.output.test_bundle import _subprocess_worker; "
                f"_subprocess_worker({str(tmp_path)!r}, phase='after_commit')"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    manifest = validate_result_bundle(tmp_path / "bundle")
    assert manifest["run_id"] == "run-001"


def test_commit_refuses_to_overwrite_nonempty_target(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    target = tmp_path / "bundle"
    staging.mkdir()
    (staging / "config.json").write_text("{}", encoding="utf-8")
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(BundleError, match="non-empty"):
        commit_staged_bundle(staging, target)


def test_tampered_inventory_digest_fails_validation(tmp_path: Path) -> None:
    output_dir = tmp_path / "bundle"
    _publish_minimal_bundle(output_dir)
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_inventory"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="sha256 mismatch"):
        validate_result_bundle(output_dir)
