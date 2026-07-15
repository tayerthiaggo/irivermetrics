"""M4 temporal graph assembly and explicit monthly checkpoint orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from os import PathLike
from pathlib import Path
from typing import Any

import xarray as xr

from hydrofragments.compute.chunks import ChunkDiagnostics, validate_chunk_budget
from hydrofragments.compute.policy import ComputePolicy
from hydrofragments.temporal.composites import build_monthly_products


class CheckpointError(RuntimeError):
    """Raised when a monthly checkpoint is missing required orchestration data."""


@dataclass(frozen=True)
class MaterializationEvent:
    """Visible record of a pipeline-owned materialization decision."""

    action: str
    checkpoint: str
    location: str | None
    materialization_occurred: bool

    def to_mapping(self) -> dict[str, object]:
        return {
            "action": self.action,
            "checkpoint": self.checkpoint,
            "location": self.location,
            "materialization_occurred": self.materialization_occurred,
        }


@dataclass(frozen=True)
class PipelineDiagnostics:
    """Dask graph, chunk, and checkpoint evidence for the monthly stage."""

    graph_task_count: int
    chunks: tuple[ChunkDiagnostics, ...]
    materialization_events: tuple[MaterializationEvent, ...] = ()

    def with_event(self, event: MaterializationEvent) -> "PipelineDiagnostics":
        return replace(
            self,
            materialization_events=self.materialization_events + (event,),
        )

    def to_mapping(self) -> dict[str, object]:
        return {
            "graph_task_count": self.graph_task_count,
            "chunks": [item.to_mapping() for item in self.chunks],
            "materialization": [
                item.to_mapping() for item in self.materialization_events
            ],
        }


@dataclass(frozen=True)
class MonthlyPipelineResult:
    """Monthly dataset plus diagnostics suitable for a run manifest."""

    dataset: xr.Dataset
    diagnostics: PipelineDiagnostics
    policy: ComputePolicy

    @property
    def manifest(self) -> dict[str, Any]:
        compute = self.diagnostics.to_mapping()
        compute.update(
            {
                "actual_backend": self.policy.actual_backend,
                "checkpoint": self.policy.checkpoint,
            }
        )
        return {"compute": compute}


def _graph_task_count(dataset: xr.Dataset) -> int:
    graph = dataset.__dask_graph__()
    return 0 if graph is None else len(graph)


def _dataset_chunk_diagnostics(
    dataset: xr.Dataset,
    policy: ComputePolicy,
    *,
    stage_prefix: str,
) -> tuple[ChunkDiagnostics, ...]:
    return tuple(
        validate_chunk_budget(
            variable,
            target_chunk_bytes=policy.target_chunk_bytes,
            live_array_multiplier=policy.live_array_multiplier,
            stage=f"{stage_prefix}.{name}",
        )
        for name, variable in dataset.data_vars.items()
    )


def assemble_monthly_pipeline(
    water: xr.DataArray,
    valid_obs: xr.DataArray,
    *,
    input_cadence: str,
    monthly_composite: str,
    composite_owner: str,
    policy: ComputePolicy,
) -> MonthlyPipelineResult:
    """Assemble and inspect the M4 graph without materializing raster values."""

    input_diagnostics = (
        validate_chunk_budget(
            water,
            target_chunk_bytes=policy.target_chunk_bytes,
            live_array_multiplier=policy.live_array_multiplier,
            stage="input.water",
        ),
        validate_chunk_budget(
            valid_obs,
            target_chunk_bytes=policy.target_chunk_bytes,
            live_array_multiplier=policy.live_array_multiplier,
            stage="input.valid_obs",
        ),
    )
    dataset = build_monthly_products(
        water,
        valid_obs,
        input_cadence=input_cadence,
        monthly_composite=monthly_composite,
        composite_owner=composite_owner,
    )
    diagnostics = PipelineDiagnostics(
        graph_task_count=_graph_task_count(dataset),
        chunks=input_diagnostics
        + _dataset_chunk_diagnostics(dataset, policy, stage_prefix="monthly"),
    )
    return MonthlyPipelineResult(dataset, diagnostics, policy)


def _open_reusable_checkpoint(
    path: Path,
    *,
    monthly_composite: str,
    composite_owner: str,
) -> xr.Dataset:
    dataset = xr.open_zarr(path, chunks={})
    expected = {
        "hydrofragments_checkpoint_stage": "monthly",
        "monthly_composite": monthly_composite,
        "composite_owner": composite_owner,
    }
    mismatches = {
        key: (dataset.attrs.get(key), value)
        for key, value in expected.items()
        if dataset.attrs.get(key) != value
    }
    if mismatches:
        dataset.close()
        raise CheckpointError(f"checkpoint provenance mismatch: {mismatches}")
    return dataset


def _checkpoint_result(
    result: MonthlyPipelineResult,
    *,
    checkpoint_path: Path | None,
) -> MonthlyPipelineResult:
    policy = result.policy
    if policy.checkpoint == "none":
        event = MaterializationEvent("skipped", "none", None, False)
        return replace(result, diagnostics=result.diagnostics.with_event(event))

    if policy.checkpoint == "persist":
        persist_kwargs = {} if policy.scheduler is None else {"scheduler": policy.scheduler}
        dataset = result.dataset.persist(**persist_kwargs)
        event = MaterializationEvent("persisted", "persist", None, True)
    else:
        if checkpoint_path is None:
            raise CheckpointError("zarr checkpoint requires checkpoint_path")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        staged = result.dataset.assign_attrs(
            hydrofragments_checkpoint_stage="monthly"
        )
        staged.to_zarr(checkpoint_path, mode="w")
        dataset = xr.open_zarr(checkpoint_path, chunks={})
        event = MaterializationEvent(
            "written", "zarr", str(checkpoint_path), True
        )

    diagnostics = PipelineDiagnostics(
        graph_task_count=_graph_task_count(dataset),
        chunks=result.diagnostics.chunks
        + _dataset_chunk_diagnostics(dataset, policy, stage_prefix="checkpoint"),
        materialization_events=result.diagnostics.materialization_events + (event,),
    )
    return MonthlyPipelineResult(dataset, diagnostics, policy)


def run_monthly_pipeline(
    water: xr.DataArray,
    valid_obs: xr.DataArray,
    *,
    input_cadence: str,
    monthly_composite: str,
    composite_owner: str,
    policy: ComputePolicy,
    checkpoint_path: str | PathLike[str] | None = None,
    reuse_existing: bool = True,
) -> MonthlyPipelineResult:
    """Run the only M4 materialization boundary or reuse its durable output."""

    resolved_path = None if checkpoint_path is None else Path(checkpoint_path)
    if (
        policy.checkpoint == "zarr"
        and reuse_existing
        and resolved_path is not None
        and resolved_path.exists()
    ):
        dataset = _open_reusable_checkpoint(
            resolved_path,
            monthly_composite=monthly_composite,
            composite_owner=composite_owner,
        )
        event = MaterializationEvent(
            "reused", "zarr", str(resolved_path), False
        )
        diagnostics = PipelineDiagnostics(
            graph_task_count=_graph_task_count(dataset),
            chunks=_dataset_chunk_diagnostics(
                dataset, policy, stage_prefix="checkpoint"
            ),
            materialization_events=(event,),
        )
        return MonthlyPipelineResult(dataset, diagnostics, policy)

    assembled = assemble_monthly_pipeline(
        water,
        valid_obs,
        input_cadence=input_cadence,
        monthly_composite=monthly_composite,
        composite_owner=composite_owner,
        policy=policy,
    )
    return _checkpoint_result(assembled, checkpoint_path=resolved_path)


__all__ = [
    "CheckpointError",
    "MaterializationEvent",
    "MonthlyPipelineResult",
    "PipelineDiagnostics",
    "assemble_monthly_pipeline",
    "run_monthly_pipeline",
]
