"""Offline tests for the dynamics/spatial-export benchmark gate (Task 12)."""

from __future__ import annotations

import json

import pytest

from hydrofragments.benchmarks import end_to_end_workflow as e2e


def _fake_spatial_run(
    *,
    scenario_id: str,
    total_seconds: float = 5.0,
    peak_rss_bytes: int = 80_000_000,
    metric_parity_holds: bool | None = True,
    export_retry: dict | None = None,
    status: str = "ok",
):
    payload = {
        "status": status,
        "scenario_id": scenario_id,
        "timings_seconds": {
            "core_analysis": total_seconds * 0.85,
            "bundle_validation": total_seconds * 0.15,
            "total": total_seconds,
        },
        "peak_rss_bytes": peak_rss_bytes,
        "metric_parity_holds": metric_parity_holds,
        "source_materializations": 12,
        "label_passes": 12,
        "metrics_digest": f"digest-{scenario_id}",
        "coverage_digest": f"coverage-{scenario_id}",
    }
    if export_retry is not None:
        payload["export_retry"] = export_retry
        payload["checkpoint_state"] = {"raster_checkpoint_root": "/tmp/ckpt"}
    return payload


def _install_fake_spatial_subprocess(monkeypatch, *, seconds_by_scenario, rss_by_scenario=None):
    rss_by_scenario = rss_by_scenario or {}
    calls: list[dict] = []

    def fake_run(payload, *, poll_interval_s=0.05):
        calls.append(dict(payload))
        scenario_id = payload["scenario_id"]
        if payload.get("phase") == "export_retry":
            return {
                "status": "ok",
                "scenario_id": scenario_id,
                "phase": "export_retry",
                "timings_seconds": {"output_finalize": 0.5, "total": 0.5},
                "peak_rss_bytes": rss_by_scenario.get(scenario_id, 80_000_000),
                "source_materializations": 0,
                "label_passes": 0,
            }
        seconds = seconds_by_scenario.get(scenario_id, 5.0)
        rss = rss_by_scenario.get(scenario_id, 80_000_000)
        if payload.get("expect_failure"):
            return {
                "status": "expected_failure",
                "scenario_id": scenario_id,
                "error_type": "MemoryBudgetExceeded",
                "error_message": "synthetic benchmark failure",
                "peak_rss_bytes": rss,
            }
        run = _fake_spatial_run(
            scenario_id=scenario_id,
            total_seconds=seconds,
            peak_rss_bytes=rss,
            metric_parity_holds=scenario_id != "candidate_export_off_bad_parity",
            export_retry={"placeholder": True} if scenario_id == "checkpoint_export_retry" else None,
        )
        return run

    def fake_true_baseline(*, output_dir, workers=1):
        calls.append({"scenario_id": "baseline_export_off", "output_dir": str(output_dir), "workers": workers})
        seconds = seconds_by_scenario.get("baseline_export_off", 5.0)
        rss = rss_by_scenario.get("baseline_export_off", 80_000_000)
        return _fake_spatial_run(
            scenario_id="baseline_export_off",
            total_seconds=seconds,
            peak_rss_bytes=rss,
        )

    monkeypatch.setattr(e2e, "_run_spatial_export_subprocess", fake_run)
    monkeypatch.setattr(e2e, "_run_true_baseline_export_off_trial", fake_true_baseline)
    return calls


def _sample_spatial_scenarios_for_gates():
    def ok_run(scenario_id, *, total=10.0, rss=100_000_000, status="ok", error_type=None):
        payload = {
            "status": status,
            "timings_seconds": {"total": total},
            "peak_rss_bytes": rss,
        }
        if error_type is not None:
            payload["error_type"] = error_type
        return payload

    return [
        {
            "scenario_id": "baseline_export_off",
            "status": "ok",
            "runs": [ok_run("baseline_export_off", total=10.0, rss=100_000_000) for _ in range(3)],
        },
        {
            "scenario_id": "candidate_export_off",
            "status": "ok",
            "runs": [
                {**ok_run("candidate_export_off", total=10.5, rss=102_000_000), "metric_parity_holds": True}
                for _ in range(3)
            ],
        },
        {
            "scenario_id": "candidate_all_products",
            "status": "ok",
            "runs": [ok_run("candidate_all_products", total=11.0, rss=120_000_000) for _ in range(3)],
        },
        {
            "scenario_id": "long_480_memory",
            "status": "ok",
            "runs": [ok_run("long_480_memory", total=60.0, rss=108_000_000)],
        },
        {
            "scenario_id": "large_spatial_sparse",
            "status": "ok",
            "runs": [ok_run("large_spatial_sparse", total=1.2, rss=100_040_000)],
        },
        {
            "scenario_id": "large_spatial_single_component",
            "status": "ok",
            "runs": [
                {
                    "status": "expected_failure",
                    "error_type": "MemoryBudgetExceeded",
                }
            ],
        },
        {
            "scenario_id": "checkpoint_export_retry",
            "status": "ok",
            "runs": [
                {
                    "status": "ok",
                    "export_retry": {"source_materializations": 0},
                }
            ],
        },
    ]


def test_spatial_export_matrix_schema_and_gates(monkeypatch, tmp_path):
    seconds = {scenario.scenario_id: 10.0 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    seconds["candidate_export_off"] = 10.5
    seconds["candidate_all_products"] = 11.0
    rss = {scenario.scenario_id: 100_000_000 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    rss["candidate_all_products"] = 120_000_000
    _install_fake_spatial_subprocess(monkeypatch, seconds_by_scenario=seconds, rss_by_scenario=rss)

    payload = e2e.run_spatial_export_matrix(workdir=tmp_path, repeats=2, warmup=0)

    assert payload["schema_version"] == e2e.SPATIAL_EXPORT_SCHEMA_VERSION
    assert payload["baseline"] == e2e.SPATIAL_EXPORT_BASELINE
    assert len(payload["scenarios"]) == len(e2e.SPATIAL_EXPORT_SCENARIOS)

    skipped = [s for s in payload["scenarios"] if s["status"] == "skipped"]
    assert any(s["scenario_id"] == "zarr_local_subset" for s in skipped)

    gates = payload["gates"]
    assert gates["export_off_within_10pct_gate"] is True
    assert gates["export_off_peak_rss_within_gate"] is True
    assert gates["all_products_peak_rss_within_125pct_gate"] is True
    assert gates["long_480_memory_within_gate"] is True
    assert gates["large_spatial_rss_within_125pct_admission_gate"] is True
    assert gates["large_spatial_single_component_fail_fast"] is True
    assert gates["metric_parity_on_off_holds"] is True
    assert gates["checkpoint_retry_skips_source_reads"] is True
    assert gates["true_baseline_commit"] == e2e.SPATIAL_EXPORT_TRUE_BASELINE_COMMIT


def test_summarize_spatial_export_gates_real_logic():
    gates = e2e._summarize_spatial_export_gates(_sample_spatial_scenarios_for_gates())

    assert gates["export_off_within_10pct_gate"] is True
    assert gates["export_off_peak_rss_within_gate"] is True
    assert gates["all_products_peak_rss_within_125pct_gate"] is True
    assert gates["long_480_memory_within_gate"] is True
    assert gates["large_spatial_rss_within_125pct_admission_gate"] is True
    assert gates["large_spatial_single_component_fail_fast"] is True
    assert gates["metric_parity_on_off_holds"] is True
    assert gates["checkpoint_retry_skips_source_reads"] is True


def test_export_off_gate_flags_regression_over_10pct(monkeypatch, tmp_path):
    seconds = {scenario.scenario_id: 10.0 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    seconds["candidate_export_off"] = 12.0
    _install_fake_spatial_subprocess(monkeypatch, seconds_by_scenario=seconds)

    payload = e2e.run_spatial_export_matrix(workdir=tmp_path, repeats=1, warmup=0)
    gates = payload["gates"]

    assert gates["export_off_regression_fraction"] == pytest.approx(0.2)
    assert gates["export_off_within_10pct_gate"] is False


def test_all_products_rss_gate_flags_over_125pct(monkeypatch, tmp_path):
    seconds = {scenario.scenario_id: 10.0 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    rss = {scenario.scenario_id: 100_000_000 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    rss["candidate_all_products"] = 140_000_000
    _install_fake_spatial_subprocess(monkeypatch, seconds_by_scenario=seconds, rss_by_scenario=rss)

    payload = e2e.run_spatial_export_matrix(workdir=tmp_path, repeats=1, warmup=0)
    gates = payload["gates"]

    assert gates["all_products_peak_rss_fraction_of_core"] == pytest.approx(1.4)
    assert gates["all_products_peak_rss_within_125pct_gate"] is False


def test_write_spatial_export_baseline_round_trips(monkeypatch, tmp_path):
    seconds = {scenario.scenario_id: 5.0 for scenario in e2e.SPATIAL_EXPORT_SCENARIOS}
    _install_fake_spatial_subprocess(monkeypatch, seconds_by_scenario=seconds)

    result = e2e.write_spatial_export_baseline(
        tmp_path / "results",
        workdir=tmp_path / "workdir",
        repeats=1,
        warmup=0,
        baseline_commit="4fab7df",
    )

    json_path = tmp_path / "results" / "dynamics_spatial_exports.json"
    md_path = tmp_path / "results" / "dynamics_spatial_exports.md"
    assert json_path.exists()
    assert md_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline_commit"] == "4fab7df"
    assert "Promotion gates" in md_path.read_text(encoding="utf-8")
    assert result["report_files"]["json"] == str(json_path)


@pytest.mark.slow
def test_large_spatial_single_component_raises_memory_budget_exceeded(tmp_path):
    scenario = next(
        s
        for s in e2e.SPATIAL_EXPORT_SCENARIOS
        if s.scenario_id == "large_spatial_single_component"
    )
    payload = e2e._spatial_export_subprocess_payload(
        scenario=scenario,
        output_dir=tmp_path / "out",
        workdir=tmp_path / "work",
    )
    result = e2e._run_spatial_export_subprocess(payload)
    assert result["status"] == "expected_failure"
    assert result["error_type"] == "MemoryBudgetExceeded"


@pytest.mark.slow
def test_one_compact_spatial_export_subprocess_smoke(tmp_path):
    """Single real subprocess smoke for the compact georef fixture."""

    scenario = next(
        s for s in e2e.SPATIAL_EXPORT_SCENARIOS if s.scenario_id == "candidate_export_off"
    )
    payload = e2e._spatial_export_subprocess_payload(
        scenario=scenario,
        output_dir=tmp_path / "out",
        workdir=tmp_path / "work",
    )
    result = e2e._run_spatial_export_subprocess(payload)
    assert result["status"] == "ok"
    assert result["timings_seconds"]["total"] > 0
    assert result.get("metric_parity_holds") is True
