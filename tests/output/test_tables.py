from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine

from hydrofragments.models import MetricRecord
from hydrofragments.output.spatial import SpatialGrid
from hydrofragments.schema import (
    MetricDependency,
    MetricFamily,
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    ValueType,
    WarningFlag,
)


def metric_record(**changes: object) -> MetricRecord:
    values: dict[str, object] = {
        "run_id": "run-001",
        "config_hash": "a" * 64,
        "package_version": "0.1.0",
        "git_sha": "abc123",
        "catchment_id": "fitzroy",
        "aoi_id": "reach-01",
        "zone": "AOI",
        "date": datetime(2026, 1, 1),
        "metric": "apsec",
        "metric_family": MetricFamily.EXTENT,
        "value": 12.5,
        "unit": "percent",
        "value_type": ValueType.MONTHLY,
        "warning_flags": (WarningFlag.LENGTH_CRS_CAVEAT,),
        "is_reportable": True,
        "source": "wofs",
        "resolution_m": 30.0,
        "crs": "EPSG:3577",
        "area_unit": "m2",
        "length_unit": "m",
        "monthly_composite": "supplied",
        "min_patch_pixels": 3,
        "min_patch_area_m2": 2700.0,
        "connectivity_rule": 8,
        "metric_dependency": MetricDependency.NONE,
    }
    values.update(changes)
    return MetricRecord(**values)


def test_records_to_frame_has_exact_column_order_types_and_nullable_values() -> None:
    from hydrofragments.output.tables import records_to_frame

    frame = records_to_frame(
        [metric_record(value=None, hy=None, proxy_channel=None)]
    )

    assert tuple(frame.columns) == OUTPUT_COLUMNS
    assert frame.loc[0, "value"] is pd.NA
    assert frame.loc[0, "hy"] is pd.NA
    assert frame.loc[0, "proxy_channel"] is pd.NA
    assert {
        column: str(frame.dtypes[column])
        for column in OUTPUT_COLUMNS
        if OUTPUT_DTYPES[column] != "list[string]"
    } == {
        column: dtype
        for column, dtype in OUTPUT_DTYPES.items()
        if dtype != "list[string]"
    }
    assert frame.loc[0, "warning_flags"] == ["length_crs_caveat"]


def test_records_to_frame_rejects_missing_non_nullable_identity() -> None:
    from hydrofragments.output.tables import TableSchemaError, records_to_frame

    row = metric_record().to_mapping()
    row["run_id"] = None

    with pytest.raises(TableSchemaError, match="run_id.*null"):
        records_to_frame([row])


def test_records_to_frame_rejects_dropped_legacy_metrics() -> None:
    from hydrofragments.output.tables import TableSchemaError, records_to_frame

    row = metric_record().to_mapping()
    row["metric"] = "PF"

    with pytest.raises(TableSchemaError, match="forbidden"):
        records_to_frame([row])


def test_parquet_writer_uses_stable_hive_partition_paths(tmp_path: Path) -> None:
    from hydrofragments.output.tables import write_tidy_parquet

    metrics_dir = write_tidy_parquet(
        [
            metric_record(),
            metric_record(
                metric="occurrence",
                metric_family=MetricFamily.PERSISTENCE,
                metric_dependency=MetricDependency.VALIDITY,
                value_type=ValueType.RASTER_SUMMARY,
            ),
        ],
        tmp_path / "metrics",
    )

    relative_files = {
        path.relative_to(metrics_dir).as_posix()
        for path in metrics_dir.rglob("*.parquet")
    }
    assert len(relative_files) == 2
    assert any(
        path.startswith("metric_family=extent/value_type=monthly/")
        for path in relative_files
    )
    assert any(
        path.startswith(
            "metric_family=persistence/value_type=raster_summary/"
        )
        for path in relative_files
    )


def test_partitioned_parquet_reopens_with_exact_schema(tmp_path: Path) -> None:
    from hydrofragments.output.tables import (
        read_tidy_parquet,
        write_tidy_parquet,
    )

    path = write_tidy_parquet([metric_record()], tmp_path / "metrics")

    restored = read_tidy_parquet(path)

    assert tuple(restored.columns) == OUTPUT_COLUMNS
    assert restored.loc[0, "metric"] == "apsec"
    assert restored.loc[0, "warning_flags"] == ["length_crs_caveat"]
    assert restored.loc[0, "metric_family"] == "extent"
    assert restored.loc[0, "value_type"] == "monthly"


def test_csv_export_flattens_and_restores_warning_flags(tmp_path: Path) -> None:
    from hydrofragments.output.tables import read_tidy_csv, write_tidy_csv

    path = write_tidy_csv(
        [
            metric_record(
                warning_flags=(
                    WarningFlag.LENGTH_CRS_CAVEAT,
                    WarningFlag.COMPOSITE_SENSITIVE,
                )
            )
        ],
        tmp_path / "metrics.csv",
    )

    raw = pd.read_csv(path, keep_default_na=False)
    assert raw.loc[0, "warning_flags"] == (
        "length_crs_caveat;composite_sensitive"
    )
    restored = read_tidy_csv(path)
    assert restored.loc[0, "warning_flags"] == [
        "length_crs_caveat",
        "composite_sensitive",
    ]
    assert tuple(restored.columns) == OUTPUT_COLUMNS


def test_disabled_vector_export_never_consumes_patch_geometries(
    tmp_path: Path,
) -> None:
    from hydrofragments.output.tables import write_output_tables

    def patch_geometries():
        raise AssertionError("patch geometry source was consumed")
        yield  # pragma: no cover

    artifacts = write_output_tables(
        [metric_record()],
        tmp_path,
        include_vectors=False,
        patch_geometries=patch_geometries(),
    )

    assert artifacts.metrics_dir.is_dir()
    assert artifacts.vectors_path is None


def test_vector_export_is_checkpoint_only_and_does_not_receive_metrics(
    tmp_path: Path,
) -> None:
    from hydrofragments.output.tables import write_output_tables

    checkpoint = tmp_path / "monthly.zarr"
    checkpoint.mkdir()
    calls: list[tuple[Path, Path]] = []

    def exporter(source: Path, destination: Path) -> Path:
        calls.append((source, destination))
        destination.mkdir(parents=True)
        return destination

    artifacts = write_output_tables(
        [metric_record()],
        tmp_path / "bundle",
        include_vectors=True,
        vector_checkpoint=checkpoint,
        vector_exporter=exporter,
    )

    assert calls == [(checkpoint, tmp_path / "bundle" / "vectors")]
    assert artifacts.vectors_path == tmp_path / "bundle" / "vectors"


def test_validate_table_formats_rejects_unknown_literals() -> None:
    from hydrofragments.output.tables import validate_table_formats

    with pytest.raises(ValueError, match="unknown table format"):
        validate_table_formats(("parquet", "gpkg"))


def test_write_output_tables_accepts_formats_tuple(tmp_path: Path) -> None:
    from hydrofragments.output.tables import write_output_tables

    artifacts = write_output_tables(
        [metric_record()],
        tmp_path,
        formats=("parquet", "csv"),
    )

    assert artifacts.metrics_dir.is_dir()
    assert artifacts.csv_path is not None
    assert artifacts.csv_path.exists()


def test_records_to_frame_accepts_homogeneous_legacy_schema() -> None:
    from hydrofragments.output.tables import records_to_frame

    row = metric_record().to_mapping()
    row["schema_version"] = "1.0.0"
    frame = records_to_frame([row])
    assert frame.loc[0, "schema_version"] == "1.0.0"


def test_records_to_frame_rejects_mixed_schema_versions() -> None:
    from hydrofragments.output.tables import TableSchemaError, records_to_frame

    current = metric_record().to_mapping()
    legacy = metric_record(metric="occurrence").to_mapping()
    legacy["schema_version"] = "1.0.0"

    with pytest.raises(TableSchemaError, match="mixed metric schema versions"):
        records_to_frame([current, legacy])


def test_pyarrow_is_declared_for_canonical_parquet_output() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert '"pyarrow>=' in pyproject


def test_write_output_tables_uses_vector_exporter_for_monthly_pools(
    tmp_path: Path,
) -> None:
    from hydrofragments.config import HydroConfig
    from hydrofragments.output.checkpoints import PoolCheckpointConsumer
    from hydrofragments.output.tables import write_output_tables
    from hydrofragments.output.vectors import export_vectors_from_checkpoint
    from rasterio.crs import CRS

    test_crs = CRS.from_string(
        "+proj=tmerc +lat_0=0 +lon_0=0 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    grid = SpatialGrid(
        crs=test_crs,
        transform=Affine(30.0, 0.0, 0.0, 0.0, -30.0, 180.0),
        height=6,
        width=6,
        y_dim="y",
        x_dim="x",
        y=np.arange(6, dtype=float) * 30.0,
        x=np.arange(6, dtype=float) * 30.0,
    )
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.1.0",
            "input": {"kind": "watermask_tsfill"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "upstream",
            },
            "output": {
                "output_dir": str(tmp_path),
                "spatial_products": ["monthly_pools"],
            },
        }
    )
    consumer = PoolCheckpointConsumer.create(
        grid=grid,
        config=config,
        input_fingerprint="tables",
        root=tmp_path / "pool_checkpoint",
        export_enabled=True,
    )
    checkpoint = consumer.as_consumer().finalize()

    artifacts = write_output_tables(
        [metric_record()],
        tmp_path / "bundle",
        include_vectors=True,
        vector_checkpoint=checkpoint.root,
        vector_exporter=lambda source, destination: export_vectors_from_checkpoint(
            source, destination
        ),
    )

    gpkg = artifacts.vectors_path / "spatial.gpkg"
    assert gpkg.exists()
