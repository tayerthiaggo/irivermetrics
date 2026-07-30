"""Self-contained run configuration and manifest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np

from hydrofragments.config import HydroConfig
from hydrofragments.schema import SCHEMA_VERSION


MANIFEST_SCHEMA_VERSION = "1.0.0"


class ManifestError(ValueError):
    """Raised when run metadata is incomplete or not self-contained."""


@dataclass(frozen=True)
class RunMetadataArtifacts:
    config_path: Path
    manifest_path: Path


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _hash_json(value: object) -> str:
    return hashlib.sha256(_json_bytes(value).rstrip(b"\n")).hexdigest()


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ManifestError("created_at must include a timezone")
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _relative_artifacts(
    artifacts: Mapping[str, str | Path],
) -> dict[str, str]:
    resolved: dict[str, str] = {
        "config": "config.json",
        "manifest": "run_manifest.json",
    }
    for name, raw_path in artifacts.items():
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(
                f"artifact path for {name} must be relative to the run directory"
            )
        resolved[str(name)] = PurePosixPath(*path.parts).as_posix()
    return dict(sorted(resolved.items()))


def _hash_array(mask: Any) -> str:
    """A stable SHA-256 content digest over a raster mask's raw values.

    Consistent with this module's ``_hash_json``/``_json_bytes`` convention
    (canonical bytes -> sha256 hex digest) rather than importing a different
    hashing utility from elsewhere in the codebase. Uses the array's raw
    bytes plus its shape and dtype (not JSON) since the payload is a numpy
    array, not JSON-serializable structured data -- shape/dtype are included
    so two differently-shaped or differently-typed arrays with coincidentally
    identical byte patterns never collide.
    """
    array = np.asarray(mask)
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(array).tobytes())
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    return hasher.hexdigest()


def build_dea_provenance(
    config: HydroConfig,
    *,
    product: str,
    version: str | None,
    item_ids: Sequence[str],
    crs: str,
    resolution: float | None,
    time_span: str | None,
    zone_mask: Any,
    planning_footprint: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the ``dea_provenance`` manifest section for a DEA-zoned run.

    Records the DEA product id/version/STAC item IDs/CRS/resolution/
    time_span (from ``WoStatistics``/its ``provenance`` mapping), the zone
    thresholds actually used (``config.zones.t_persist``/``t_season`` --
    the SAME ``config`` object ``build_run_manifest`` already receives), and
    a content digest of the DEA zone mask itself (``ZoneResult.mask``,
    hashed via :func:`_hash_array`, this module's own hashing convention --
    the digest is computed here rather than added as a field on
    ``ZoneResult``, which is out of scope for this call site).

    ``time_span`` is recorded distinctly under ``dea_provenance`` and is
    never merged into or reconciled against any other run-level date range
    already present in the manifest (DEA's multi-year coverage and the
    local cube's own date range can legitimately differ -- see the plan's
    temporal-mismatch risk note).

    ``planning_footprint``, when supplied, carries a ``WetPlanningFootprint``
    (W1.5)'s own ``digest``/``factor``/``safety_cells``/``covered_years``/
    ``source_collection``/``source_version``/``source_lineage`` through
    byte-for-byte unchanged -- this function never re-derives or reformats
    that digest, only passes it through.
    """
    section: dict[str, object] = {
        "product": product,
        "version": version,
        "item_ids": list(item_ids),
        "crs": crs,
        "resolution": resolution,
        "time_span": time_span,
        "zone_thresholds": {
            "t_persist": config.zones.t_persist,
            "t_season": config.zones.t_season,
        },
        "zone_mask_digest": _hash_array(zone_mask),
    }
    if planning_footprint is not None:
        footprint = dict(planning_footprint)
        section["planning_footprint"] = {
            "digest": footprint["digest"],
            "factor": footprint["factor"],
            "safety_cells": footprint["safety_cells"],
            "covered_years": list(footprint["covered_years"]),
            "source_collection": footprint["source_collection"],
            "source_version": footprint["source_version"],
            "source_lineage": footprint["source_lineage"],
        }
    return section


def build_run_manifest(
    config: HydroConfig,
    *,
    run_id: str,
    package_version: str,
    git_sha: str,
    input_fingerprint: Mapping[str, object],
    planned_backend: str,
    actual_backend_by_stage: Mapping[str, str],
    skipped_metrics: Sequence[Mapping[str, object]] = (),
    warnings: Sequence[str] = (),
    timings_seconds: Mapping[str, float] | None = None,
    comparison_context: Mapping[str, object] | None = None,
    artifacts: Mapping[str, str | Path] | None = None,
    created_at: datetime | None = None,
    dependency_versions: Mapping[str, str] | None = None,
    backend_capabilities: Mapping[str, object] | None = None,
    dea_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build complete machine-readable provenance for one immutable run.

    ``dependency_versions`` records pinned external packages whose version
    affects scientific output but that are not HydroFragments itself -- e.g.
    ``{"hydroseason": "0.1.0"}`` for any run using HY-dependent metrics
    (spec M9: "record hydroseason version ... in run config/manifest").

    ``dea_provenance``, when supplied (build it with
    :func:`build_dea_provenance`), is recorded verbatim as the manifest's
    ``dea_provenance`` section for a run that used DEA Water Observation
    Statistics for zoning. A run that never touched DEA statistics (the
    common local-cube-only case) omits the argument entirely and the
    manifest has no ``dea_provenance`` key at all -- not a present section
    full of ``null``s -- following the same optional/additive pattern
    ``backend_capabilities``/``comparison_context`` already establish here.
    """

    for name, value in {
        "run_id": run_id,
        "package_version": package_version,
        "git_sha": git_sha,
        "planned_backend": planned_backend,
    }.items():
        if not value:
            raise ManifestError(f"{name} must be a non-empty string")
    if not input_fingerprint:
        raise ManifestError("input_fingerprint cannot be empty")

    versions: dict[str, object] = {
        "package_version": package_version,
        "git_sha": git_sha,
    }
    # HY settings are resolved into HydroConfig, so every manifest can record
    # the pinned provider version without relying on callers to remember it.
    try:
        import hydroseason

        hydroseason_version = hydroseason.__version__
        versions["hydroseason"] = hydroseason_version
    except ImportError as error:  # pragma: no cover - dependency is mandatory
        raise ManifestError("hydroseason is required for manifest provenance") from error
    supplied_dependencies = dict(dependency_versions or {})
    supplied_hydroseason = supplied_dependencies.pop("hydroseason", None)
    if supplied_hydroseason is not None and supplied_hydroseason != hydroseason_version:
        raise ManifestError(
            "dependency_versions.hydroseason does not match installed hydroseason"
        )
    versions.update(supplied_dependencies)

    comparison = dict(comparison_context or {})
    comparison.update(
        {
            "validity_policy": config.validity.policy,
            "monthly_composite": config.temporal.monthly_composite,
        }
    )

    backend: dict[str, object] = {
        "planned": planned_backend,
        "actual_by_stage": dict(actual_backend_by_stage),
    }
    if backend_capabilities is not None:
        backend["capabilities"] = dict(backend_capabilities)

    manifest: dict[str, object] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "output_schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": _utc_text(created_at or datetime.now(timezone.utc)),
        "config_hash": config.config_hash,
        "execution_hash": config.execution_hash,
        "package_version": package_version,
        "git_sha": git_sha,
        "resolved_config": config.scientific_config(),
        "execution_config": config.execution_config(),
        "input_fingerprint": dict(input_fingerprint),
        "versions": versions,
        "backend": backend,
        "skipped_metrics": [dict(item) for item in skipped_metrics],
        "warnings": list(warnings),
        "timings_seconds": dict(timings_seconds or {}),
        "comparison": comparison,
        "artifacts": _relative_artifacts(artifacts or {}),
    }
    if dea_provenance is not None:
        manifest["dea_provenance"] = dict(dea_provenance)
    return manifest


def write_run_metadata(
    output_dir: str | Path,
    config: HydroConfig,
    **manifest_arguments: object,
) -> RunMetadataArtifacts:
    """Write canonical scientific config plus its linked run manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    manifest_path = root / "run_manifest.json"
    manifest = build_run_manifest(config, **manifest_arguments)  # type: ignore[arg-type]
    config_path.write_bytes(_json_bytes(config.scientific_config()))
    manifest_path.write_bytes(_json_bytes(manifest))
    return RunMetadataArtifacts(config_path, manifest_path)


def read_run_manifest(source: str | Path) -> dict[str, object]:
    """Read a run manifest from a file or bundle directory."""

    path = Path(source)
    if path.is_dir():
        path = path / "run_manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read run manifest: {path}") from error
    if not isinstance(value, dict):
        raise ManifestError("run manifest must be a JSON object")
    return value


def validate_result_bundle(output_dir: str | Path) -> dict[str, object]:
    """Validate bundle integrity without reopening any source dataset."""

    root = Path(output_dir)
    manifest = read_run_manifest(root)
    if manifest.get("manifest_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestError("unsupported manifest_schema_version")
    if manifest.get("output_schema_version") != SCHEMA_VERSION:
        raise ManifestError("unsupported output_schema_version")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ManifestError("manifest artifacts must be an object")
    relative = _relative_artifacts(
        {str(name): str(path) for name, path in artifacts.items()}
    )
    for name, relative_path in relative.items():
        if not (root / relative_path).exists():
            raise ManifestError(f"missing bundle artifact {name}: {relative_path}")

    config_path = root / str(relative["config"])
    try:
        config_value = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError("config.json is not valid JSON") from error
    if _hash_json(config_value) != manifest.get("config_hash"):
        raise ManifestError("config_hash does not match config.json")
    if config_value != manifest.get("resolved_config"):
        raise ManifestError("resolved_config does not match config.json")
    if not manifest.get("input_fingerprint"):
        raise ManifestError("input_fingerprint is missing")
    return manifest


__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "ManifestError",
    "RunMetadataArtifacts",
    "build_dea_provenance",
    "build_run_manifest",
    "read_run_manifest",
    "validate_result_bundle",
    "write_run_metadata",
]
