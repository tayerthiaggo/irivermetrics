"""Canonical tidy table writers and isolated optional exports."""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import pandas as pd

from hydrofragments.models import MetricRecord
from hydrofragments.schema import (
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    SchemaError,
    validate_metric_id,
)

_GIT_SHA_ENV = "HYDROFRAGMENTS_GIT_SHA"
_PACKAGE_METADATA_REVISION_KEYS = (
    "Source-Revision-Id",
    "Revision-Id",
    "Git-Commit",
)
_SUPPORTED_TABLE_FORMATS = frozenset({"parquet", "csv"})


PARQUET_PARTITION_COLUMNS = ("metric_family", "value_type")
NON_NULLABLE_COLUMNS = (
    "schema_version",
    "run_id",
    "config_hash",
    "package_version",
    "git_sha",
    "catchment_id",
    "aoi_id",
    "metric",
    "metric_family",
    "unit",
    "value_type",
    "warning_flags",
    "is_reportable",
    "metric_dependency",
)


class TableSchemaError(ValueError):
    """Raised when rows cannot conform to the canonical output schema."""


@dataclass(frozen=True)
class TableArtifacts:
    """Paths written by :func:`write_output_tables`."""

    metrics_dir: Path
    csv_path: Path | None = None
    vectors_path: Path | None = None


_METRIC_COVERAGE_COLUMNS = (
    "metric",
    "runtime_wired",
    "status",
    "rows",
    "reportable_rows",
    "reason",
)


Row = MetricRecord | Mapping[str, object]
VectorExporter = Callable[[Path, Path], Path]


def resolve_git_sha() -> str:
    """Resolve one git revision for an entire analysis run.

    Precedence: CI environment variable, installed package metadata, local
    Git ``HEAD``, then the literal ``unknown``.
    """

    env_value = os.environ.get(_GIT_SHA_ENV, "").strip()
    if env_value:
        return env_value

    try:
        metadata = importlib.metadata.metadata("hydrofragments")
        for key in _PACKAGE_METADATA_REVISION_KEYS:
            value = metadata.get(key, "").strip()
            if value:
                return value
    except importlib.metadata.PackageNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if completed.returncode == 0:
            head = completed.stdout.strip()
            if head:
                return head
    except (OSError, subprocess.SubprocessError):
        pass

    return "unknown"


def validate_table_formats(formats: Sequence[str]) -> tuple[str, ...]:
    """Validate public table export format literals."""

    if not formats:
        raise ValueError("formats cannot be empty")
    normalized = tuple(formats)
    for format_name in normalized:
        if format_name not in _SUPPORTED_TABLE_FORMATS:
            raise ValueError(f"unknown table format: {format_name}")
    return normalized


def _row_mapping(row: Row) -> dict[str, object]:
    if isinstance(row, MetricRecord):
        return row.to_mapping()
    if not isinstance(row, Mapping):
        raise TableSchemaError("metric rows must be MetricRecord objects or mappings")
    unknown = sorted(set(row) - set(OUTPUT_COLUMNS))
    if unknown:
        raise TableSchemaError(f"unknown output column: {unknown[0]}")
    return {column: row.get(column) for column in OUTPUT_COLUMNS}


def _validate_rows(rows: list[dict[str, object]]) -> None:
    for index, row in enumerate(rows):
        flags = row["warning_flags"]
        if hasattr(flags, "tolist"):
            flags = flags.tolist()  # type: ignore[union-attr]
            row["warning_flags"] = flags
        for column in NON_NULLABLE_COLUMNS:
            value = row[column]
            if value is None or value is pd.NA:
                raise TableSchemaError(f"row {index} column {column} cannot be null")
            if isinstance(value, str) and not value:
                raise TableSchemaError(f"row {index} column {column} cannot be empty")
        try:
            validate_metric_id(str(row["metric"]))
        except SchemaError as error:
            raise TableSchemaError(str(error)) from error
        if not isinstance(flags, (list, tuple)) or not all(
            isinstance(flag, str) for flag in flags
        ):
            raise TableSchemaError(
                f"row {index} column warning_flags must be a list of strings"
            )


def _cast_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column, dtype in OUTPUT_DTYPES.items():
        if dtype == "list[string]":
            result[column] = result[column].map(
                lambda value: list(value) if isinstance(value, (list, tuple)) else []
            )
        elif dtype == "datetime64[ns]":
            result[column] = pd.to_datetime(result[column], errors="raise").astype(
                "datetime64[ns]"
            )
        else:
            result[column] = result[column].astype(dtype)
    return result.loc[:, OUTPUT_COLUMNS]


def records_to_frame(records: Iterable[Row] | pd.DataFrame) -> pd.DataFrame:
    """Validate records and return exact-order nullable pandas columns."""

    if isinstance(records, pd.DataFrame):
        unknown = sorted(set(records.columns) - set(OUTPUT_COLUMNS))
        if unknown:
            raise TableSchemaError(f"unknown output column: {unknown[0]}")
        rows = [
            {column: row.get(column) for column in OUTPUT_COLUMNS}
            for row in records.to_dict(orient="records")
        ]
    else:
        rows = [_row_mapping(row) for row in records]
    _validate_rows(rows)
    return _cast_frame(pd.DataFrame(rows, columns=OUTPUT_COLUMNS))


def write_tidy_parquet(
    records: Iterable[Row] | pd.DataFrame,
    destination: str | Path,
) -> Path:
    """Write canonical rows as a stable hive-partitioned Parquet dataset."""

    frame = records_to_frame(records)
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(
        path,
        engine="pyarrow",
        index=False,
        partition_cols=list(PARQUET_PARTITION_COLUMNS),
    )
    return path


def read_tidy_parquet(source: str | Path) -> pd.DataFrame:
    """Reopen and validate a canonical partitioned Parquet dataset."""

    frame = pd.read_parquet(Path(source), engine="pyarrow")
    return records_to_frame(frame)


def write_tidy_csv(
    records: Iterable[Row] | pd.DataFrame,
    destination: str | Path,
) -> Path:
    """Write optional flattened CSV; Parquet remains schema authority."""

    frame = records_to_frame(records)
    flattened = frame.copy()
    flattened["warning_flags"] = flattened["warning_flags"].map(";".join)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    return path


def read_tidy_csv(source: str | Path) -> pd.DataFrame:
    """Read optional CSV and restore the canonical warning-list representation."""

    frame = pd.read_csv(Path(source))
    frame["warning_flags"] = frame["warning_flags"].map(
        lambda value: []
        if pd.isna(value) or value == ""
        else str(value).split(";")
    )
    return records_to_frame(frame)


def write_metric_coverage(
    coverage: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Write the one-row-per-registry-metric coverage table as CSV.

    ``coverage`` is :func:`hydrofragments.api._build_metric_coverage`'s
    output (also ``HydroResult.metric_coverage``) -- plain diagnostic rows
    (metric id, whether it is runtime-wired, computed/skipped status, row
    counts, skip/data-quality reason), never part of the canonical
    ``OUTPUT_COLUMNS`` metric schema, so it deliberately does not go through
    :func:`records_to_frame`/:func:`write_tidy_parquet`. CSV only: this is a
    small, human-readable run-diagnostic table, not a scientific data
    product requiring Parquet's typed columnar guarantees.
    """
    unknown = sorted(set(coverage.columns) - set(_METRIC_COVERAGE_COLUMNS))
    if unknown:
        raise TableSchemaError(f"unknown metric_coverage column: {unknown[0]}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "metric_coverage.csv"
    coverage.loc[:, list(_METRIC_COVERAGE_COLUMNS)].to_csv(path, index=False)
    return path


def write_output_tables(
    records: Iterable[Row] | pd.DataFrame,
    output_dir: str | Path,
    *,
    formats: Sequence[str] = ("parquet",),
    export_csv: bool = False,
    include_vectors: bool = False,
    patch_geometries: Iterable[object] | None = None,
    vector_checkpoint: str | Path | None = None,
    vector_exporter: VectorExporter | None = None,
) -> TableArtifacts:
    """Write tables once and optionally run a checkpoint-only vector export.

    ``formats`` selects canonical table products. Parquet is the schema
    authority; CSV is an optional flattened export. Spatial vectors are never
    accepted as in-memory patch geometries.

    When vectors are disabled, ``patch_geometries`` is deliberately untouched.
    When enabled, vector work receives only a durable checkpoint path and its
    destination, keeping metric records and metric computation outside its DAG.
    """

    validated_formats = validate_table_formats(formats)
    export_csv = export_csv or "csv" in validated_formats
    write_parquet = "parquet" in validated_formats

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame = records_to_frame(records)

    metrics_dir = root / "metrics"
    if write_parquet:
        metrics_dir = write_tidy_parquet(frame, metrics_dir)
    else:
        metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        write_tidy_csv(frame, root / "metrics.csv") if export_csv else None
    )

    vectors_path = None
    if include_vectors:
        if patch_geometries is not None:
            raise ValueError(
                "vector export must consume a checkpoint, not accumulated patch geometries"
            )
        if vector_checkpoint is None or vector_exporter is None:
            raise ValueError(
                "vector export requires vector_checkpoint and vector_exporter"
            )
        checkpoint = Path(vector_checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        vectors_path = Path(vector_exporter(checkpoint, root / "vectors"))

    return TableArtifacts(metrics_dir, csv_path, vectors_path)


__all__ = [
    "NON_NULLABLE_COLUMNS",
    "PARQUET_PARTITION_COLUMNS",
    "TableArtifacts",
    "TableSchemaError",
    "read_tidy_csv",
    "read_tidy_parquet",
    "records_to_frame",
    "resolve_git_sha",
    "validate_table_formats",
    "write_metric_coverage",
    "write_output_tables",
    "write_tidy_csv",
    "write_tidy_parquet",
]
