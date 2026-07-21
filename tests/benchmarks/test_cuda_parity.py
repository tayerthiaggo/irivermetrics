from __future__ import annotations

import json
import sys


def test_parity_harness_skips_cleanly_without_cupy(monkeypatch) -> None:
    """CuPy-absent CI must get a clean 'skipped' result, never an ImportError."""
    from hydrofragments.benchmarks.cuda_parity import run_cuda_parity

    sys.modules.pop("cupy", None)
    real_import_module = __import__("importlib").import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "cupy":
            raise ModuleNotFoundError("No module named 'cupy'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)

    result = run_cuda_parity()

    assert result["schema_version"] == "1.0.0"
    assert result["baseline"] == "cuda_parity"
    assert result["cupy_available"] is False
    assert result["skipped"] is True
    assert "CuPy" in result["skip_reason"]
    assert result["cases"] == []
    assert "cupy" not in sys.modules


def test_parity_harness_covers_all_candidate_stages_when_cupy_available() -> None:
    """With CuPy simulated by the numpy shim, every candidate stage is checked."""
    from hydrofragments.benchmarks.cuda_parity import run_cuda_parity
    from hydrofragments.compute.capabilities import CUDA_CANDIDATE_STAGES

    import numpy as np

    result = run_cuda_parity(cupy_module=np)

    assert result["skipped"] is False
    assert result["cupy_available"] is True
    stages_covered = {stage["stage"] for case in result["cases"] for stage in case["stages"]}
    assert stages_covered == set(CUDA_CANDIDATE_STAGES)

    # Stages with a live CUDABackend method must pass parity exactly (the
    # numpy shim is bit-identical to CPUBackend). Candidate stages without a
    # CUDABackend implementation yet (sentinel_normalization, masks) report
    # parity_pass=None -- not silently skipped, not falsely claimed passing.
    implemented_stages = {"valid_counts", "monthly_reduction", "occurrence"}
    for case in result["cases"]:
        for stage in case["stages"]:
            if stage["stage"] in implemented_stages:
                assert stage["parity_pass"] is True
                assert stage["max_abs_diff"] >= 0.0
                assert stage["tolerance"] >= 0.0
            else:
                assert stage["parity_pass"] is None
                assert stage["note"]


def test_parity_harness_reports_failure_when_tolerance_exceeded() -> None:
    """A backend that disagrees beyond tolerance must be reported, not hidden."""
    from hydrofragments.benchmarks.cuda_parity import run_cuda_parity

    import numpy as np

    class _NoisyCupyShim:
        """Wraps numpy but perturbs sums to simulate a non-matching backend."""

        float32 = np.float32
        float64 = np.float64
        int64 = np.int64
        bool_ = np.bool_

        def __getattr__(self, item):
            return getattr(np, item)

        def sum(self, *args, **kwargs):
            result = np.sum(*args, **kwargs)
            return result + 1000

        def asarray(self, *args, **kwargs):
            return np.asarray(*args, **kwargs)

        def mean(self, *args, **kwargs):
            return np.mean(*args, **kwargs)

    result = run_cuda_parity(cupy_module=_NoisyCupyShim())

    assert result["skipped"] is False
    failing_stages = [
        stage
        for case in result["cases"]
        for stage in case["stages"]
        if not stage["parity_pass"]
    ]
    assert failing_stages
    assert result["all_pass"] is False


def test_write_cuda_parity_writes_json_report(tmp_path) -> None:
    from hydrofragments.benchmarks.cuda_parity import write_cuda_parity

    sys.modules.pop("cupy", None)
    result = write_cuda_parity(tmp_path)

    json_path = tmp_path / "cuda_parity.json"
    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "cuda_parity"
    assert result["report_files"]["json"] == str(json_path)
