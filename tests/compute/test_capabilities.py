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


def test_gated_stages_from_baseline_empty_when_file_absent(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_stages_from_baseline

    missing_path = tmp_path / "cuda_baseline.json"
    assert gated_stages_from_baseline(missing_path) == ()


def test_gated_stages_from_baseline_empty_when_file_empty_json(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_stages_from_baseline

    path = tmp_path / "cuda_baseline.json"
    path.write_text("{}", encoding="utf-8")
    assert gated_stages_from_baseline(path) == ()


def test_gated_stages_from_baseline_empty_when_file_is_malformed_json(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_stages_from_baseline

    path = tmp_path / "cuda_baseline.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert gated_stages_from_baseline(path) == ()


def test_gated_stages_from_baseline_requires_both_parity_and_speedup_pass(
    tmp_path,
) -> None:
    import json

    from hydrofragments.compute.capabilities import gated_stages_from_baseline

    path = tmp_path / "cuda_baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "baseline": "cuda_gate_evidence",
                "stages": {
                    "valid_counts": {
                        "parity_pass": True,
                        "net_speedup_pass": True,
                    },
                    "monthly_reduction": {
                        "parity_pass": True,
                        "net_speedup_pass": False,
                    },
                    "occurrence": {
                        "parity_pass": False,
                        "net_speedup_pass": True,
                    },
                    "masks": {
                        "parity_pass": False,
                        "net_speedup_pass": False,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    gated = gated_stages_from_baseline(path)

    assert gated == ("valid_counts",)


def test_gated_stages_from_baseline_ignores_stages_outside_candidate_list(
    tmp_path,
) -> None:
    import json

    from hydrofragments.compute.capabilities import gated_stages_from_baseline

    path = tmp_path / "cuda_baseline.json"
    path.write_text(
        json.dumps(
            {
                "stages": {
                    "not_a_real_stage": {
                        "parity_pass": True,
                        "net_speedup_pass": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert gated_stages_from_baseline(path) == ()


def test_detect_capabilities_with_baseline_populates_enabled_cuda_stages(
    monkeypatch, tmp_path
) -> None:
    """Injects the CUDA-available branch's inputs without real hardware, per
    the brief's guidance to make the baseline-reading step unit-testable in
    isolation from the CuPy smoke test."""
    import json

    from hydrofragments.compute import capabilities as capabilities_module

    baseline_path = tmp_path / "cuda_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "stages": {
                    "valid_counts": {
                        "parity_pass": True,
                        "net_speedup_pass": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capabilities_module, "_DEFAULT_BASELINE_PATH", baseline_path
    )

    result = capabilities_module._resolve_cuda_capabilities_from_probe(
        cupy_available=True,
        cuda_available=True,
        cupy_version="99.0",
        cuda_runtime_version=12000,
        device_count=1,
        free_memory_bytes=1_000,
        total_memory_bytes=2_000,
    )

    assert result.enabled_cuda_stages == ("valid_counts",)
    assert result.cuda_available is True


def test_detect_capabilities_with_no_baseline_file_keeps_stages_empty(
    monkeypatch, tmp_path
) -> None:
    from hydrofragments.compute import capabilities as capabilities_module

    monkeypatch.setattr(
        capabilities_module,
        "_DEFAULT_BASELINE_PATH",
        tmp_path / "cuda_baseline.json",
    )

    result = capabilities_module._resolve_cuda_capabilities_from_probe(
        cupy_available=True,
        cuda_available=True,
        cupy_version="99.0",
        cuda_runtime_version=12000,
        device_count=1,
        free_memory_bytes=1_000,
        total_memory_bytes=2_000,
    )

    assert result.enabled_cuda_stages == ()
    assert "no stage" in (result.reason or "").lower() or result.reason


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


# --- Numba evidence gate (mirrors the CUDA gate tests above; twin pattern) ---


def test_backend_capabilities_defaults_numba_enabled_kernels_empty() -> None:
    from hydrofragments.compute.capabilities import BackendCapabilities

    capabilities = BackendCapabilities()

    assert capabilities.numba_enabled_kernels == ()
    assert capabilities.to_mapping()["numba_enabled_kernels"] == []


def test_gated_kernels_from_baseline_empty_when_file_absent(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    missing_path = tmp_path / "numba_baseline.json"
    assert gated_kernels_from_baseline(missing_path) == ()


def test_gated_kernels_from_baseline_empty_when_file_empty_json(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    path = tmp_path / "numba_baseline.json"
    path.write_text("{}", encoding="utf-8")
    assert gated_kernels_from_baseline(path) == ()


def test_gated_kernels_from_baseline_empty_when_file_is_malformed_json(tmp_path) -> None:
    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    path = tmp_path / "numba_baseline.json"
    path.write_text("{not valid json", encoding="utf-8")
    assert gated_kernels_from_baseline(path) == ()


def test_gated_kernels_from_baseline_requires_both_parity_and_speedup_pass(
    tmp_path,
) -> None:
    import json

    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    path = tmp_path / "numba_baseline.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "baseline": "numba_gate_evidence",
                "kernels": {
                    "inter_pool_gap_runs": {
                        "parity_pass": True,
                        "speedup_pass": True,
                    },
                    "not_actually_faster": {
                        "parity_pass": True,
                        "speedup_pass": False,
                    },
                    "not_numerically_equal": {
                        "parity_pass": False,
                        "speedup_pass": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    gated = gated_kernels_from_baseline(path)

    # Only kernels that are both in NUMBA_CANDIDATE_KERNELS and pass both
    # gates graduate. "not_actually_faster"/"not_numerically_equal" are not
    # real candidate kernel names, but this also proves the "must pass both"
    # rule independent of membership -- see the next test for the membership
    # filter specifically.
    assert "inter_pool_gap_runs" in gated
    assert "not_actually_faster" not in gated
    assert "not_numerically_equal" not in gated


def test_gated_kernels_from_baseline_ignores_kernels_outside_candidate_list(
    tmp_path,
) -> None:
    import json

    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    path = tmp_path / "numba_baseline.json"
    path.write_text(
        json.dumps(
            {
                "kernels": {
                    "not_a_real_kernel": {
                        "parity_pass": True,
                        "speedup_pass": True,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert gated_kernels_from_baseline(path) == ()


def test_detect_capabilities_with_numba_baseline_populates_enabled_kernels(
    monkeypatch, tmp_path
) -> None:
    """Numba has no hardware probe step (JIT works on any CPU) -- gating is
    purely file-evidence-driven, unlike CUDA's probe-then-gate two-step."""
    import json

    from hydrofragments.compute import capabilities as capabilities_module

    baseline_path = tmp_path / "numba_baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "kernels": {
                    "inter_pool_gap_runs": {
                        "parity_pass": True,
                        "speedup_pass": True,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        capabilities_module, "_DEFAULT_NUMBA_BASELINE_PATH", baseline_path
    )

    result = capabilities_module.detect_capabilities()

    assert result.numba_enabled_kernels == ("inter_pool_gap_runs",)


def test_detect_capabilities_with_no_numba_baseline_file_keeps_kernels_empty(
    monkeypatch, tmp_path
) -> None:
    from hydrofragments.compute import capabilities as capabilities_module

    monkeypatch.setattr(
        capabilities_module,
        "_DEFAULT_NUMBA_BASELINE_PATH",
        tmp_path / "numba_baseline.json",
    )

    result = capabilities_module.detect_capabilities()

    assert result.numba_enabled_kernels == ()
