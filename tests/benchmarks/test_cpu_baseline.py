from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import tomllib
from packaging.requirements import Requirement


@pytest.fixture(autouse=True)
def _isolate_cupy_import_state():
    saved = sys.modules.pop("cupy", None)
    yield
    if saved is not None:
        sys.modules["cupy"] = saved
    else:
        sys.modules.pop("cupy", None)


def test_cpu_baseline_is_deterministic_and_records_stage_backends() -> None:
    from hydrofragments.benchmarks.cpu_baseline import run_cpu_baseline

    first = run_cpu_baseline(repeats=1, warmup=False)
    second = run_cpu_baseline(repeats=1, warmup=False)

    assert first["schema_version"] == "1.0.0"
    assert first["baseline"] == "cpu_reference"
    assert first["backend_planned"] == "cpu"
    assert first["environment"]["cupy_imported"] is False
    assert first["cases"]
    assert [case["output_checksum"] for case in first["cases"]] == [
        case["output_checksum"] for case in second["cases"]
    ]
    for case in first["cases"]:
        assert case["backend_actual_by_stage"]
        assert set(case["backend_actual_by_stage"].values()) == {"cpu"}
        assert case["output_checksum"]
        assert case["stages"]
        assert case["hardware"]["gpu"] is None
        assert case["dask"]["scheduler"] == "single-threaded"
        assert case["dask"]["graph_task_count"] > 0
        assert all(stage["backend_actual"] == "cpu" for stage in case["stages"])
        assert all(stage["peak_vram_bytes"] is None for stage in case["stages"])

    assert "cupy" not in sys.modules


def test_cpu_baseline_writes_machine_and_human_reports(tmp_path) -> None:
    from hydrofragments.benchmarks.cpu_baseline import write_cpu_baseline

    result = write_cpu_baseline(tmp_path, repeats=1, warmup=False)

    json_path = tmp_path / "cpu_baseline.json"
    report_path = tmp_path / "cpu_baseline.md"
    assert result["report_files"] == {
        "json": str(json_path),
        "markdown": str(report_path),
    }
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "cpu_reference"
    report = report_path.read_text(encoding="utf-8")
    assert "CPU reference benchmark baseline" in report
    assert "backend_actual" in report


def test_cuda_dependencies_are_optional_in_project_metadata() -> None:
    project = tomllib.loads(
        Path("pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    mandatory = [Requirement(req) for req in project["dependencies"]]
    cuda = [Requirement(req) for req in project["optional-dependencies"]["cuda"]]
    assert not any(req.name.lower().startswith("cupy") for req in mandatory)
    assert any(req.name.lower().startswith("cupy") for req in cuda)
