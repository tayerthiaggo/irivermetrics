"""Numba parity + speedup benchmark harness tests.

Twin of tests/benchmarks/test_cuda_parity.py + test_cuda_transfer_cost.py
combined into one harness, since a same-process JIT call has no "transfer
cost" concept to benchmark separately -- see
hydrofragments/benchmarks/numba_kernels.py's module docstring.
"""

from __future__ import annotations

import json
import sys


def test_numba_benchmark_skips_cleanly_without_numba(monkeypatch) -> None:
    """Numba-absent CI must get a clean 'skipped' result, never an ImportError."""
    from hydrofragments.benchmarks.numba_kernels import run_numba_benchmark

    sys.modules.pop("numba", None)
    real_import_module = __import__("importlib").import_module

    def _fake_import_module(name, *args, **kwargs):
        if name == "numba":
            raise ModuleNotFoundError("No module named 'numba'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr("importlib.import_module", _fake_import_module)
    from hydrofragments.metrics import clustering_numba as clustering_numba_module

    monkeypatch.setattr(clustering_numba_module, "_NUMBA_KERNEL", None)
    monkeypatch.setattr(clustering_numba_module, "_NUMBA_IMPORT_ATTEMPTED", False)

    result = run_numba_benchmark()

    assert result["schema_version"] == "1.0.0"
    assert result["baseline"] == "numba_kernels"
    assert result["numba_available"] is False
    assert result["skipped"] is True
    assert "numba" in result["skip_reason"].lower()
    assert result["kernels"] == []
    assert "numba" not in sys.modules


def test_numba_benchmark_covers_candidate_kernels_when_numba_available() -> None:
    from hydrofragments.benchmarks.numba_kernels import run_numba_benchmark
    from hydrofragments.compute.capabilities import NUMBA_CANDIDATE_KERNELS

    result = run_numba_benchmark()

    if result["skipped"]:
        # Environment genuinely lacks a working Numba; the skip-path test
        # above already covers this branch explicitly via monkeypatch, so
        # only assert the shape here.
        assert result["kernels"] == []
        return

    kernels_covered = {kernel["kernel"] for kernel in result["kernels"]}
    assert kernels_covered == set(NUMBA_CANDIDATE_KERNELS)
    for kernel in result["kernels"]:
        assert kernel["parity_pass"] in (True, False)
        assert "cases" in kernel
        assert kernel["cases"]
        for case in kernel["cases"]:
            assert case["parity_pass"] is True
            assert case["max_abs_diff"] == 0.0
        assert "warm_speedup" in kernel
        assert "cold_wall_seconds" in kernel["warm_speedup"]
        assert "warm_baseline_seconds" in kernel["warm_speedup"]
        assert "warm_numba_seconds" in kernel["warm_speedup"]
        assert "speedup_pass" in kernel


def test_write_numba_benchmark_writes_json_report(tmp_path) -> None:
    from hydrofragments.benchmarks.numba_kernels import write_numba_benchmark

    result = write_numba_benchmark(tmp_path)

    # Raw report filename matches the module name (numba_kernels.py ->
    # numba_kernels.json), same convention as cuda_parity.py -> cuda_parity.json.
    # This is distinct from the compact gate file the capability gate reads
    # (benchmarks/results/numba_baseline.json), which is produced separately
    # by gate_evidence_from_report -- see that function's docstring.
    json_path = tmp_path / "numba_kernels.json"
    markdown_path = tmp_path / "numba_kernels.md"
    assert json_path.exists()
    assert markdown_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "numba_kernels"
    assert result["report_files"]["json"] == str(json_path)


def test_numba_benchmark_reports_parity_failure_when_kernels_disagree() -> None:
    """A kernel that disagrees with the baseline must be reported, not hidden."""
    from hydrofragments.benchmarks import numba_kernels

    def _broken_kernel(wet, lengths):
        import numpy as np

        return np.array([999.0])

    result = numba_kernels.run_numba_benchmark(kernel_override=_broken_kernel)

    assert result["skipped"] is False
    failing = [kernel for kernel in result["kernels"] if not kernel["parity_pass"]]
    assert failing
    assert result["all_parity_pass"] is False


def test_gate_evidence_from_report_produces_dict_keyed_by_kernel_name() -> None:
    """gated_kernels_from_baseline expects {"kernels": {name: {...}}}, a
    dict -- but run_numba_benchmark's raw report keeps "kernels" as a list
    of per-kernel detail records (cases, timings), matching the CUDA
    harnesses' list convention. gate_evidence_from_report bridges the two,
    and its output must be directly consumable by
    hydrofragments.compute.capabilities.gated_kernels_from_baseline.
    """
    from hydrofragments.benchmarks.numba_kernels import (
        gate_evidence_from_report,
        run_numba_benchmark,
    )

    report = run_numba_benchmark()
    evidence = gate_evidence_from_report(report)

    assert isinstance(evidence["kernels"], dict)
    if not report["skipped"]:
        assert "inter_pool_gap_runs" in evidence["kernels"]
        entry = evidence["kernels"]["inter_pool_gap_runs"]
        assert entry["parity_pass"] is True
        assert entry["speedup_pass"] in (True, False)


def test_gate_evidence_from_report_is_readable_by_the_capability_gate(tmp_path) -> None:
    """End-to-end: a benchmark report distilled by gate_evidence_from_report
    and written to disk must be exactly what gated_kernels_from_baseline
    graduates a kernel from, when both gates pass."""
    import json

    from hydrofragments.benchmarks.numba_kernels import gate_evidence_from_report
    from hydrofragments.compute.capabilities import gated_kernels_from_baseline

    fabricated_report = {
        "skipped": False,
        "created_at": "2026-07-21T00:00:00+00:00",
        "kernels": [
            {
                "kernel": "inter_pool_gap_runs",
                "parity_pass": True,
                "speedup_pass": True,
                "warm_speedup": {"speedup_ratio": 2.0},
            }
        ],
    }
    evidence = gate_evidence_from_report(fabricated_report)
    path = tmp_path / "numba_baseline.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    assert gated_kernels_from_baseline(path) == ("inter_pool_gap_runs",)
