from __future__ import annotations

import sys

import numpy as np
import pytest
import xarray as xr


def test_cpu_only_capability_detection_never_imports_cupy() -> None:
    from hydrofragments.compute.capabilities import detect_capabilities

    sys.modules.pop("cupy", None)
    capabilities = detect_capabilities()

    assert capabilities.cuda_available is False
    assert capabilities.enabled_cuda_stages == ()
    assert "cupy" not in sys.modules


def test_auto_truthfully_falls_back_to_cpu_when_cuda_unavailable() -> None:
    from hydrofragments.compute.capabilities import (
        BackendCapabilities,
        resolve_execution_plan,
    )

    capabilities = BackendCapabilities.cpu_only("CuPy not installed")
    plan = resolve_execution_plan(accelerator="auto", capabilities=capabilities)

    assert plan.planned_backend == "cpu"
    assert set(plan.actual_backend_by_stage.values()) == {"cpu"}
    assert "CuPy not installed" in plan.fallback_reason
    assert plan.to_mapping()["actual_by_stage"]["valid_counts"] == "cpu"


def test_cuda_request_fails_in_strict_mode_instead_of_claiming_cpu() -> None:
    from hydrofragments.compute.capabilities import (
        BackendCapabilities,
        CapabilityError,
        resolve_execution_plan,
    )

    with pytest.raises(CapabilityError, match="CUDA.*unavailable"):
        resolve_execution_plan(
            accelerator="cuda",
            cuda_strict=True,
            capabilities=BackendCapabilities.cpu_only("CUDA unavailable"),
        )


def test_unsupported_stage_skeleton_graph_vector_stays_cpu() -> None:
    from hydrofragments.compute.capabilities import (
        BackendCapabilities,
        resolve_execution_plan,
    )

    capabilities = BackendCapabilities(
        cupy_available=True,
        cuda_available=True,
        enabled_cuda_stages=("valid_counts",),
        reason=None,
    )
    plan = resolve_execution_plan(accelerator="auto", capabilities=capabilities)

    assert plan.planned_backend == "cuda"
    assert plan.actual_backend_by_stage["valid_counts"] == "cuda"
    for stage in ("skeleton", "graph", "vector"):
        assert plan.actual_backend_by_stage[stage] == "cpu"


def test_floating_tolerance_is_declared_by_dtype() -> None:
    from hydrofragments.compute.capabilities import FLOATING_TOLERANCES

    assert FLOATING_TOLERANCES["float32"] > 0
    assert FLOATING_TOLERANCES["float64"] > 0
    assert FLOATING_TOLERANCES["float32"] >= FLOATING_TOLERANCES["float64"]


def test_analyze_manifest_records_actual_backend_for_each_stage(tmp_path) -> None:
    from hydrofragments.api import analyze
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import WaterCube

    times = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[ns]")
    water = xr.DataArray(
        np.ones((2, 2, 2), dtype=bool),
        dims=("time", "y", "x"),
        coords={"time": times},
    )
    valid_obs = xr.ones_like(water, dtype=bool)
    cube = WaterCube(water, valid_obs, "generic_binary", "monthly")
    config = HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "compute": {"accelerator": "auto"},
            "output": {"output_dir": str(tmp_path)},
        }
    )

    analyze(cube, "demo", config=config)
    manifest = __import__("json").loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    actual = manifest["backend"]["actual_by_stage"]
    assert actual["valid_counts"] == "cpu"
    assert actual["skeleton"] == "cpu"
    assert manifest["backend"]["planned"] == "cpu"
    assert manifest["backend"]["capabilities"]["cuda_available"] is False
    assert manifest["backend"]["capabilities"]["enabled_cuda_stages"] == []
