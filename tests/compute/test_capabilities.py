from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr


def _assert_capabilities_consistent(capabilities) -> None:
    if not capabilities.cupy_available:
        assert capabilities.cuda_available is False
        assert capabilities.device_count == 0
        return

    if capabilities.cuda_available:
        assert capabilities.device_count >= 1
        assert capabilities.cuda_runtime_version is not None
    else:
        assert capabilities.device_count == 0


def _assert_plan_consistent_with_capabilities(plan) -> None:
    _assert_capabilities_consistent(plan.capabilities)
    enabled = set(plan.capabilities.enabled_cuda_stages)
    if not plan.capabilities.cuda_available or not enabled:
        assert plan.planned_backend == "cpu"
        assert set(plan.actual_backend_by_stage.values()) == {"cpu"}
        return

    assert plan.planned_backend == "cuda"
    for stage, backend in plan.actual_backend_by_stage.items():
        assert backend == ("cuda" if stage in enabled else "cpu")


def test_cpu_only_capability_detection_never_imports_cupy() -> None:
    from hydrofragments.compute.capabilities import detect_capabilities

    sys.modules.pop("cupy", None)
    capabilities = detect_capabilities()

    _assert_capabilities_consistent(capabilities)
    assert capabilities.cuda_available is False
    assert capabilities.enabled_cuda_stages == ()
    assert "cupy" not in sys.modules


def test_detect_capabilities_reports_missing_cupy_without_importing_it() -> None:
    from hydrofragments.compute.capabilities import detect_capabilities

    sys.modules.pop("cupy", None)

    def _import_module(name, package=None):
        if name == "cupy":
            raise ImportError("CuPy unavailable in test")
        return importlib.import_module(name, package)

    with patch(
        "hydrofragments.compute.capabilities.importlib.import_module",
        side_effect=_import_module,
    ):
        capabilities = detect_capabilities(probe_cuda=True)

    _assert_capabilities_consistent(capabilities)
    assert capabilities.cuda_available is False
    assert capabilities.cupy_available is False
    assert "CuPy unavailable" in (capabilities.reason or "")
    assert "cupy" not in sys.modules


def test_detect_capabilities_reports_cuda_devices_consistently() -> None:
    from hydrofragments.compute.capabilities import detect_capabilities

    class _FakeArray:
        def __iadd__(self, other):
            return self

    class _FakeDevice:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        @property
        def mem_info(self):
            return (1_000, 2_000)

    fake_cupy = ModuleType("cupy")
    fake_cupy.__version__ = "13.0.0"
    fake_cupy.is_available = lambda: True
    fake_cupy.zeros = lambda *args, **kwargs: _FakeArray()
    fake_cupy.float32 = float
    fake_cupy.cuda = SimpleNamespace(
        runtime=SimpleNamespace(
            getDeviceCount=lambda: 2,
            runtimeGetVersion=lambda: 12_000,
        ),
        Device=lambda index: _FakeDevice(),
        Stream=SimpleNamespace(null=SimpleNamespace(synchronize=lambda: None)),
    )

    sys.modules.pop("cupy", None)
    with patch(
        "hydrofragments.compute.capabilities.importlib.import_module",
        return_value=fake_cupy,
    ):
        capabilities = detect_capabilities(probe_cuda=True)

    _assert_capabilities_consistent(capabilities)
    assert capabilities.cupy_available is True
    assert capabilities.cuda_available is True
    assert capabilities.device_count == 2
    assert capabilities.enabled_cuda_stages == ()


def test_auto_truthfully_falls_back_to_cpu_when_cuda_unavailable() -> None:
    from hydrofragments.compute.capabilities import (
        BackendCapabilities,
        resolve_execution_plan,
    )

    capabilities = BackendCapabilities.cpu_only("CuPy not installed")
    plan = resolve_execution_plan(accelerator="auto", capabilities=capabilities)

    _assert_plan_consistent_with_capabilities(plan)
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
        device_count=1,
        cuda_runtime_version=12_000,
        enabled_cuda_stages=("valid_counts",),
        reason=None,
    )
    plan = resolve_execution_plan(accelerator="auto", capabilities=capabilities)

    _assert_plan_consistent_with_capabilities(plan)
    assert plan.planned_backend == "cuda"
    assert plan.actual_backend_by_stage["valid_counts"] == "cuda"
    for stage in ("skeleton", "graph", "vector"):
        assert plan.actual_backend_by_stage[stage] == "cpu"


def test_floating_tolerance_is_declared_by_dtype() -> None:
    from hydrofragments.compute.capabilities import FLOATING_TOLERANCES

    assert FLOATING_TOLERANCES["float32"] > 0
    assert FLOATING_TOLERANCES["float64"] > 0
    assert FLOATING_TOLERANCES["float32"] >= FLOATING_TOLERANCES["float64"]


def test_analyze_manifest_records_actual_backend_for_each_stage(
    tmp_path, monkeypatch
) -> None:
    from hydrofragments.api import analyze
    from hydrofragments.compute import capabilities as capabilities_module
    from hydrofragments.compute.capabilities import BackendCapabilities
    from hydrofragments.config import HydroConfig
    from hydrofragments.models import WaterCube

    monkeypatch.setattr(
        capabilities_module,
        "detect_capabilities",
        lambda **kwargs: BackendCapabilities.cpu_only("controlled test probe"),
    )

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
            "config_schema_version": "1.0.0",
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


@pytest.mark.hardware
def test_detect_capabilities_hardware_smoke() -> None:
    from hydrofragments.compute.capabilities import detect_capabilities

    sys.modules.pop("cupy", None)
    capabilities = detect_capabilities(probe_cuda=True)
    _assert_capabilities_consistent(capabilities)
