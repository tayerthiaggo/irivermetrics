from __future__ import annotations

import json
import sys


def test_transfer_cost_harness_skips_cleanly_without_cupy(monkeypatch) -> None:
    """CuPy-absent CI must get a clean 'skipped' result, never an ImportError."""
    from hydrofragments.benchmarks.cuda_transfer_cost import run_cuda_transfer_cost

    sys.modules.pop("cupy", None)
    real_import_module = __import__("importlib").import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    result = run_cuda_transfer_cost()

    assert result["schema_version"] == "1.0.0"
    assert result["baseline"] == "cuda_transfer_cost"
    assert result["cupy_available"] is False
    assert result["skipped"] is True
    assert "CuPy" in result["skip_reason"]
    assert result["cases"] == []


def test_transfer_cost_harness_records_crossover_with_cupy_available() -> None:
    """With CuPy simulated, each stage/size gets CPU vs CUDA wall time and a
    crossover size is recorded (or None if GPU never wins net of transfer)."""
    from hydrofragments.benchmarks.cuda_transfer_cost import run_cuda_transfer_cost
    from hydrofragments.compute.capabilities import CUDA_CANDIDATE_STAGES

    import numpy as np

    result = run_cuda_transfer_cost(
        cupy_module=np, sizes=(8, 16), repeats=1
    )

    assert result["skipped"] is False
    assert result["cupy_available"] is True
    stages_covered = {stage["stage"] for stage in result["stages"]}
    assert stages_covered == set(CUDA_CANDIDATE_STAGES)
    for stage in result["stages"]:
        assert stage["measurements"]
        for measurement in stage["measurements"]:
            assert measurement["cpu_wall_seconds"] >= 0.0
            assert measurement["cuda_wall_seconds_incl_transfer"] >= 0.0
            assert "size" in measurement
        # crossover_size is either None (GPU never nets faster in this sweep)
        # or one of the swept sizes.
        assert stage["crossover_size"] is None or stage["crossover_size"] in (8, 16)
        assert "net_speedup_at_max_size" in stage


def test_write_cuda_transfer_cost_writes_json_report(tmp_path) -> None:
    from hydrofragments.benchmarks.cuda_transfer_cost import write_cuda_transfer_cost

    sys.modules.pop("cupy", None)
    result = write_cuda_transfer_cost(tmp_path)

    json_path = tmp_path / "cuda_transfer_cost.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "cuda_transfer_cost"
    assert result["report_files"]["json"] == str(json_path)
