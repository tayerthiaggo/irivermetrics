"""Canonical tidy table writers and isolated optional exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd

from hydrofragments.models import MetricRecord
from hydrofragments.schema import (
    OUTPUT_COLUMNS,
    OUTPUT_DTYPES,
    SchemaError,
    validate_metric_id,
)


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


Row = MetricRecord | Mapping[str, object]
VectorExporter = Callable[[Path, Path], Path]


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


def write_output_tables(
    records: Iterable[Row] | pd.DataFrame,
    output_dir: str | Path,
    *,
    export_csv: bool = False,
    include_vectors: bool = False,
    patch_geometries: Iterable[object] | None = None,
    vector_checkpoint: str | Path | None = None,
    vector_exporter: VectorExporter | None = None,
) -> TableArtifacts:
    """Write tables once and optionally run a checkpoint-only vector export.

    When vectors are disabled, ``patch_geometries`` is deliberately untouched.
    When enabled, vector work receives only a durable checkpoint path and its
    destination, keeping metric records and metric computation outside its DAG.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame = records_to_frame(records)
    metrics_dir = write_tidy_parquet(frame, root / "metrics")
    csv_path = write_tidy_csv(frame, root / "metrics.csv") if export_csv else None

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
    "write_output_tables",
    "write_tidy_csv",
    "write_tidy_parquet",
]
