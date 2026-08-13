"""Light in-memory analysis result types used without GIS writer imports."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from hydrofragments._version import __version__
from hydrofragments.config import HydroConfig
from hydrofragments.output.manifest import build_run_manifest


@dataclass(frozen=True)
class CoreAnalysisResult:
    """In-memory analysis outcome before optional bundle publication."""

    metrics_table: pd.DataFrame
    metric_coverage: pd.DataFrame
    run_id: str
    git_sha: str
    report_warnings: tuple[str, ...]
    skipped_metrics: tuple[tuple[str, str], ...]
    execution_plan_mapping: Mapping[str, object]
    input_fingerprint: Mapping[str, object]
    comparison_context: Mapping[str, object]
    hydroyear_result: Any = None
    raster_checkpoint: Any = None
    pool_checkpoint_root: Path | None = None
    spatial_grid: Any = None


def build_in_memory_manifest(
    config: HydroConfig,
    core: CoreAnalysisResult,
    *,
    created_at: datetime | None = None,
) -> dict[str, object]:
    """Build a complete manifest dictionary without filesystem writes."""

    manifest = build_run_manifest(
        config,
        run_id=core.run_id,
        package_version=__version__,
        git_sha=core.git_sha,
        input_fingerprint=core.input_fingerprint,
        planned_backend=str(core.execution_plan_mapping.get("planned_backend", "cpu")),
        actual_backend_by_stage=dict(
            core.execution_plan_mapping.get("actual_backend_by_stage", {})
        ),
        backend_capabilities=dict(
            core.execution_plan_mapping.get("backend_capabilities", {})
        ),
        skipped_metrics=[
            {"metric_id": metric_id, "reason": reason}
            for metric_id, reason in core.skipped_metrics
        ],
        warnings=list(core.report_warnings),
        comparison_context=core.comparison_context,
        artifacts={},
        artifact_inventory=[],
        created_at=created_at or datetime.now(timezone.utc),
    )
    manifest_dict = dict(manifest)
    manifest_dict["manifest_path"] = None
    return manifest_dict
