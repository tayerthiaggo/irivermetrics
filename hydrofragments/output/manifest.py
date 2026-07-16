"""Self-contained run configuration and manifest artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence

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
) -> dict[str, object]:
    """Build complete machine-readable provenance for one immutable run.

    ``dependency_versions`` records pinned external packages whose version
    affects scientific output but that are not HydroFragments itself -- e.g.
    ``{"hydroseason": "0.1.0"}`` for any run using HY-dependent metrics
    (spec M9: "record hydroseason version ... in run config/manifest").
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

    return {
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
        "backend": {
            "planned": planned_backend,
            "actual_by_stage": dict(actual_backend_by_stage),
        },
        "skipped_metrics": [dict(item) for item in skipped_metrics],
        "warnings": list(warnings),
        "timings_seconds": dict(timings_seconds or {}),
        "comparison": comparison,
        "artifacts": _relative_artifacts(artifacts or {}),
    }


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
    "build_run_manifest",
    "read_run_manifest",
    "validate_result_bundle",
    "write_run_metadata",
]
