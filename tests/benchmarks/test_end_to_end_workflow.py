"""Offline tests for the W3.7 end-to-end workflow benchmark harness.

These tests never touch the network or spawn real subprocesses: they
monkeypatch :func:`hydrofragments.benchmarks.end_to_end_workflow._run_case_subprocess`
(the module's actual external boundary -- one isolated subprocess per
candidate/mode) with a deterministic fake, following the same "offline
fake hydroseason" convention already used in
``tests/integration/test_dea_workflow.py`` (patch the external boundary,
verify the orchestrator's own behaviour: call shape, gate arithmetic,
schema completeness, deterministic digest/report output).

The REAL live-network Fitzroy run (this module's actual reason for
existing) is executed separately, once, as a real script invocation --
see the task-3.7 report for that run's numbers. It is deliberately not a
pytest test: it makes live DEA/STAC calls and takes minutes, which does
not belong in the fast offline suite.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from hydrofragments.benchmarks import end_to_end_workflow as e2e
from hydrofragments.benchmarks._e2e_worker import (
    planning_footprint_native_wet_pixel_superset_metrics,
)


def _fake_case_result(*, candidate_id, mode, total_seconds, peak_rss_bytes, n_water=None):
    n_water = n_water or {"2020-01-01": 100, "2020-02-01": 110}
    return {
        "status": "ok",
        "mode": mode,
        "candidate_id": candidate_id,
        "timings_seconds": {
            "dea_planning": total_seconds * 0.1,
            "wofs_query_and_acquisition": total_seconds * 0.6,
            "metric_processing": total_seconds * 0.25,
            "output_write": total_seconds * 0.05,
            "total": total_seconds,
        },
        "metrics_digest": "digest-" + candidate_id,
        "metrics_row_count": 42,
        "n_water_by_month": n_water,
        "analysis_mask_valid_observation_fraction": 0.903,
        "planning_footprint_native_wet_pixel_superset_holds": True,
        "planning_footprint_native_wet_pixel_coverage_fraction": 1.0,
        "planning_footprint_superset_reason": None,
        "footprint_factor": 4,
        "footprint_used": True,
        "cache_path": "/fake/cache",
        "subprocess_wall_seconds": total_seconds,
        "peak_rss_bytes": peak_rss_bytes,
    }


def _install_fake_subprocess(monkeypatch, *, per_candidate_cold_seconds, rss_by_candidate=None):
    """Patch the subprocess boundary with a deterministic fake.

    ``per_candidate_cold_seconds`` maps candidate_id -> cold total seconds;
    warm is always cold * 0.1 (a clean 90% speedup) and cold==warm digests
    match by default, matching a real cache-hit rerun.
    """
    rss_by_candidate = rss_by_candidate or {}
    calls: list[dict] = []

    def fake_run_case_subprocess(*, candidate, real_case, cache_dir, output_dir, mode, poll_interval_s=0.05):
        calls.append(
            {
                "candidate_id": candidate.candidate_id,
                "mode": mode,
                "cache_dir": str(cache_dir),
                "output_dir": str(output_dir),
            }
        )
        cold_seconds = per_candidate_cold_seconds[candidate.candidate_id]
        seconds = cold_seconds if mode == "cold" else cold_seconds * 0.1
        rss = rss_by_candidate.get(candidate.candidate_id, 100_000_000)
        return _fake_case_result(
            candidate_id=candidate.candidate_id,
            mode=mode,
            total_seconds=seconds,
            peak_rss_bytes=rss,
        )

    monkeypatch.setattr(e2e, "_run_case_subprocess", fake_run_case_subprocess)
    return calls


# ---------------------------------------------------------------------------
# Step 1: deterministic output digest + phase schema tests
# ---------------------------------------------------------------------------


def test_matrix_is_deterministic_given_deterministic_subprocess_results(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    first = e2e.run_end_to_end_matrix(workdir=tmp_path / "run1")
    second = e2e.run_end_to_end_matrix(workdir=tmp_path / "run2")

    first_digests = [c["cold"]["metrics_digest"] for c in first["cases"]["fitzroy_compact"]["candidates"]]
    second_digests = [c["cold"]["metrics_digest"] for c in second["cases"]["fitzroy_compact"]["candidates"]]
    assert first_digests == second_digests

    first_gates = [c["all_measurable_gates_pass"] for c in first["cases"]["fitzroy_compact"]["candidates"]]
    second_gates = [c["all_measurable_gates_pass"] for c in second["cases"]["fitzroy_compact"]["candidates"]]
    assert first_gates == second_gates


def test_schema_has_explicit_null_fields_for_deferred_cases(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)

    assert set(payload["cases"]) == {"fitzroy_compact", "gilbert_thin_braided", "large_catchment"}

    for deferred_id in ("gilbert_thin_braided", "large_catchment"):
        case = payload["cases"][deferred_id]
        assert case["status"] == "skipped"
        assert isinstance(case["skipped_reason"], str) and case["skipped_reason"]
        assert case["candidates"] is None
        # Required gate fields are present and explicitly null, never omitted.
        gates = case["gates"]
        assert "cold_median_at_least_30pct_faster_than_full_aoi" in gates
        assert gates["cold_median_at_least_30pct_faster_than_full_aoi"] is None
        assert "warm_rerun_at_least_80pct_faster_than_cold_full_aoi" in gates
        assert gates["warm_rerun_at_least_80pct_faster_than_cold_full_aoi"] is None
        assert "zero_stac_calls_on_warm_rerun" in gates
        assert gates["zero_stac_calls_on_warm_rerun"] is None


def test_schema_never_fabricates_numbers_for_deferred_cases(monkeypatch, tmp_path):
    """No numeric (non-null, non-string) value may appear under a skipped case's gates."""
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)

    for deferred_id in ("gilbert_thin_braided", "large_catchment"):
        for value in payload["cases"][deferred_id]["gates"].values():
            assert not isinstance(value, (int, float)), (
                f"deferred case {deferred_id!r} gate value {value!r} must not be numeric"
            )


def test_json_round_trips_through_write_end_to_end_baseline(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    result = e2e.write_end_to_end_baseline(tmp_path / "results", workdir=tmp_path / "workdir")

    json_path = tmp_path / "results" / "end_to_end_workflow.json"
    md_path = tmp_path / "results" / "end_to_end_workflow.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == e2e.SCHEMA_VERSION
    assert payload["baseline"] == "end_to_end_workflow"

    report = md_path.read_text(encoding="utf-8")
    assert "End-to-end workflow benchmark" in report
    assert "Gilbert" in report
    assert "large catchment" in report.lower() or "large-catchment" in report.lower()
    assert result["report_files"]["json"] == str(json_path)
    assert result["report_files"]["markdown"] == str(md_path)


# ---------------------------------------------------------------------------
# Gate arithmetic: regression / RSS / warm-speedup / equality, all
# independently hand-derived against the fake subprocess results above.
# ---------------------------------------------------------------------------


def test_regression_gate_flags_over_10pct_slowdown_vs_serial(monkeypatch, tmp_path):
    # factor4_workers1 is the serial baseline (factor=4, workers=1): 10s cold.
    # factor4_workers2 candidate is deliberately 20% slower (12s) -> fails gate.
    # factor4_workers4 candidate is within 5% (10.5s) -> passes gate.
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    per_candidate_cold_seconds["factor4_workers2"] = 12.0
    per_candidate_cold_seconds["factor4_workers4"] = 10.5
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidates = {
        c["candidate_id"]: c for c in payload["cases"]["fitzroy_compact"]["candidates"]
    }

    assert candidates["factor4_workers1"]["regression_fraction_vs_serial"] == pytest.approx(0.0)
    assert candidates["factor4_workers1"]["regression_within_10pct_gate"] is True

    assert candidates["factor4_workers2"]["regression_fraction_vs_serial"] == pytest.approx(0.2)
    assert candidates["factor4_workers2"]["regression_within_10pct_gate"] is False

    assert candidates["factor4_workers4"]["regression_fraction_vs_serial"] == pytest.approx(0.05)
    assert candidates["factor4_workers4"]["regression_within_10pct_gate"] is True


def test_peak_rss_gate_flags_over_125pct_of_serial(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    rss_by_candidate = {c.candidate_id: 100_000_000 for c in e2e.FITZROY_CASE.candidates}
    # serial baseline (factor4_workers1) = 100MB. factor4_workers2 = 140MB (fails, >125%).
    # factor4_workers4 = 120MB (passes, <=125%).
    rss_by_candidate["factor4_workers2"] = 140_000_000
    rss_by_candidate["factor4_workers4"] = 120_000_000
    _install_fake_subprocess(
        monkeypatch,
        per_candidate_cold_seconds=per_candidate_cold_seconds,
        rss_by_candidate=rss_by_candidate,
    )

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidates = {
        c["candidate_id"]: c for c in payload["cases"]["fitzroy_compact"]["candidates"]
    }

    assert candidates["factor4_workers1"]["peak_rss_fraction_of_serial"] == pytest.approx(1.0)
    assert candidates["factor4_workers1"]["peak_rss_within_125pct_gate"] is True

    assert candidates["factor4_workers2"]["peak_rss_fraction_of_serial"] == pytest.approx(1.4)
    assert candidates["factor4_workers2"]["peak_rss_within_125pct_gate"] is False

    assert candidates["factor4_workers4"]["peak_rss_fraction_of_serial"] == pytest.approx(1.2)
    assert candidates["factor4_workers4"]["peak_rss_within_125pct_gate"] is True


def test_warm_speedup_fraction_computed_from_cold_and_warm_totals(monkeypatch, tmp_path):
    # Fake harness always makes warm = cold * 0.1 -> 90% speedup.
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidate = payload["cases"]["fitzroy_compact"]["candidates"][0]

    assert candidate["warm_speedup_fraction_vs_own_cold"] == pytest.approx(0.9)


def test_cold_warm_equality_gates_use_digest_and_n_water_dict_equality(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidate = payload["cases"]["fitzroy_compact"]["candidates"][0]

    # Fake subprocess returns identical digest/n_water for cold and warm
    # (both built from the same candidate_id-derived digest string).
    assert candidate["cold_warm_metrics_equal"] is True
    assert candidate["cold_warm_n_water_equal"] is True
    assert candidate["planning_footprint_native_wet_pixel_superset_holds"] is True


def test_a_failing_candidate_never_passes_all_measurable_gates(monkeypatch, tmp_path):
    def fake_run_case_subprocess(*, candidate, real_case, cache_dir, output_dir, mode, poll_interval_s=0.05):
        if candidate.candidate_id == "factor3_workers1" and mode == "cold":
            return {
                "status": "error",
                "mode": mode,
                "candidate_id": candidate.candidate_id,
                "error_type": "RuntimeError",
                "error_message": "simulated STAC failure",
                "peak_rss_bytes": None,
            }
        return _fake_case_result(
            candidate_id=candidate.candidate_id, mode=mode, total_seconds=10.0, peak_rss_bytes=100_000_000
        )

    monkeypatch.setattr(e2e, "_run_case_subprocess", fake_run_case_subprocess)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidates = {
        c["candidate_id"]: c for c in payload["cases"]["fitzroy_compact"]["candidates"]
    }
    failed = candidates["factor3_workers1"]
    assert failed["cold"]["status"] == "error"
    assert failed["all_measurable_gates_pass"] is False
    assert failed["cold_warm_metrics_equal"] is None
    # A failing candidate must never show up as the recommendation's pick.
    rec = payload["recommendation"]
    if rec["verdict"] == "fastest_passing_candidate_identified":
        assert rec["fastest_passing_candidate_id"] != "factor3_workers1"


@pytest.mark.parametrize("candidate_id", ["factor4_workers1", "factor3_workers4"])
def test_all_candidates_include_full_candidate_metadata(monkeypatch, tmp_path, candidate_id):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    candidates = {
        c["candidate_id"]: c for c in payload["cases"]["fitzroy_compact"]["candidates"]
    }
    record = candidates[candidate_id]
    assert record["factor"] in (3, 4)
    assert record["workers"] in (1, 2, 4)
    assert "cold" in record and "warm" in record


def test_recommendation_picks_fastest_passing_candidate(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    per_candidate_cold_seconds["factor3_workers2"] = 8.0  # fastest, still within 10% regression? No -- faster than serial.
    _install_fake_subprocess(monkeypatch, per_candidate_cold_seconds=per_candidate_cold_seconds)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    rec = payload["recommendation"]

    assert rec["verdict"] == "fastest_passing_candidate_identified"
    assert rec["fastest_passing_candidate_id"] == "factor3_workers2"
    assert "NOT promoted" in rec["promotion_status"]


def test_recommendation_reports_no_passing_candidate_when_all_fail_a_gate(monkeypatch, tmp_path):
    per_candidate_cold_seconds = {c.candidate_id: 10.0 for c in e2e.FITZROY_CASE.candidates}
    rss_by_candidate = {c.candidate_id: 1_000_000_000 for c in e2e.FITZROY_CASE.candidates}
    _install_fake_subprocess(
        monkeypatch,
        per_candidate_cold_seconds=per_candidate_cold_seconds,
        rss_by_candidate=rss_by_candidate,
    )

    def fake_run_case_subprocess(*, candidate, real_case, cache_dir, output_dir, mode, poll_interval_s=0.05):
        seconds = 10.0 if mode == "cold" else 1.0
        return _fake_case_result(
            candidate_id=candidate.candidate_id,
            mode=mode,
            total_seconds=seconds,
            peak_rss_bytes=1_000_000_000,
            n_water={"2020-01-01": 100} if mode == "cold" else {"2020-01-01": 999},
        )

    monkeypatch.setattr(e2e, "_run_case_subprocess", fake_run_case_subprocess)

    payload = e2e.run_end_to_end_matrix(workdir=tmp_path)
    rec = payload["recommendation"]
    assert rec["verdict"] == "no_passing_candidate"

    # Every candidate here has identical cold/warm timing (10.0s) and RSS
    # (1e9 bytes, equal to the serial baseline), so every candidate passes
    # the timing/RSS-only gate even though none passes all_measurable_gates_pass
    # (n_water_by_month differs cold vs warm). The secondary note should
    # surface a real candidate, not stay silently None.
    assert rec["timing_rss_only_note"] is not None
    assert "factor4_workers1" in rec["timing_rss_only_note"] or any(
        cid in rec["timing_rss_only_note"]
        for cid in ("factor4_workers1", "factor4_workers2", "factor4_workers4",
                    "factor3_workers1", "factor3_workers2", "factor3_workers4")
    )

    candidates = {
        c["candidate_id"]: c for c in payload["cases"]["fitzroy_compact"]["candidates"]
    }
    assert all(c["timing_rss_gates_pass"] is True for c in candidates.values())
    assert all(c["all_measurable_gates_pass"] is False for c in candidates.values())


# ---------------------------------------------------------------------------
# planning_footprint_native_wet_pixel_superset_metrics: proves the real
# W1.5 superset gate (native_mask <= expand(coarse_mask)), not the old
# mislabeled data-completeness fraction. Uses a synthetic
# hydroseason._io_dea_stats.WetPlanningFootprint-shaped object (native_mask/
# coarse_mask xarray DataArrays + factor), matching the "offline fake
# hydroseason" convention used in tests/io/test_dea.py -- no live network
# call needed for this unit-level proof.
# ---------------------------------------------------------------------------


@dataclass
class _FakeFootprint:
    """Minimal stand-in for hydroseason._io_dea_stats.WetPlanningFootprint.

    Only exposes the three attributes
    planning_footprint_native_wet_pixel_superset_metrics actually reads:
    native_mask, coarse_mask (xarray DataArrays with .values), and factor.
    """

    native_mask: "xr.DataArray"
    coarse_mask: "xr.DataArray"
    factor: int


def _make_footprint(native: np.ndarray, coarse: np.ndarray, factor: int) -> _FakeFootprint:
    return _FakeFootprint(
        native_mask=xr.DataArray(native.astype(bool)),
        coarse_mask=xr.DataArray(coarse.astype(bool)),
        factor=factor,
    )


def test_superset_metrics_returns_none_with_reason_when_footprint_is_none():
    result = planning_footprint_native_wet_pixel_superset_metrics(None)

    assert result["planning_footprint_native_wet_pixel_superset_holds"] is None
    assert result["planning_footprint_native_wet_pixel_coverage_fraction"] is None
    assert isinstance(result["planning_footprint_superset_reason"], str)
    assert result["planning_footprint_superset_reason"]


def test_superset_metrics_passes_when_native_wet_pixel_is_fully_covered():
    """The defining correctness property (mirrors hydroseason's own
    test_footprint_isolated_one_pixel_water_survives_round_trip): a single
    isolated native wet pixel whose coarse cell is marked wet must expand
    back to cover it exactly -- native <= expand(coarse) holds, factor=4."""
    native = np.zeros((16, 16), dtype=bool)
    native[3, 11] = True
    coarse = np.zeros((4, 4), dtype=bool)
    coarse[3 // 4, 11 // 4] = True  # the coarse cell containing (3, 11)
    footprint = _make_footprint(native, coarse, factor=4)

    result = planning_footprint_native_wet_pixel_superset_metrics(footprint)

    assert result["planning_footprint_native_wet_pixel_superset_holds"] is True
    assert result["planning_footprint_native_wet_pixel_coverage_fraction"] == pytest.approx(1.0)
    assert result["planning_footprint_superset_reason"] is None


def test_superset_metrics_fails_when_a_native_wet_pixel_falls_outside_expanded_coarse_mask():
    """A native wet pixel whose coarse cell was NOT marked wet (simulating a
    broken/incomplete planning footprint) must be caught: superset_holds is
    False and coverage_fraction is strictly less than 1.0."""
    native = np.zeros((16, 16), dtype=bool)
    native[3, 11] = True  # coarse cell (0, 2) for factor=4
    native[9, 1] = True  # coarse cell (2, 0) for factor=4 -- left uncovered below
    coarse = np.zeros((4, 4), dtype=bool)
    coarse[0, 2] = True  # covers (3, 11) only; (2, 0) deliberately left False
    footprint = _make_footprint(native, coarse, factor=4)

    result = planning_footprint_native_wet_pixel_superset_metrics(footprint)

    assert result["planning_footprint_native_wet_pixel_superset_holds"] is False
    assert result["planning_footprint_native_wet_pixel_coverage_fraction"] == pytest.approx(0.5)
    assert result["planning_footprint_superset_reason"] is None


def test_superset_metrics_vacuously_passes_with_no_native_wet_pixels():
    native = np.zeros((8, 8), dtype=bool)
    coarse = np.zeros((2, 2), dtype=bool)
    footprint = _make_footprint(native, coarse, factor=4)

    result = planning_footprint_native_wet_pixel_superset_metrics(footprint)

    assert result["planning_footprint_native_wet_pixel_superset_holds"] is True
    assert result["planning_footprint_native_wet_pixel_coverage_fraction"] == pytest.approx(1.0)


def test_superset_metrics_matches_hydroseason_expansion_technique_on_partial_edge_block():
    """Mirrors hydroseason's own
    test_footprint_partial_edge_block_preserved_not_dropped: a native grid
    whose size is not a multiple of factor must still round-trip correctly
    via the identical coarse.repeat(...)[:H, :W] truncation."""
    native = np.zeros((10, 10), dtype=bool)
    native[9, 9] = True  # in the partial trailing 2x2 block for factor=4
    coarse = np.zeros((3, 3), dtype=bool)
    coarse[2, 2] = True  # trailing partial block marked wet
    footprint = _make_footprint(native, coarse, factor=4)

    result = planning_footprint_native_wet_pixel_superset_metrics(footprint)

    assert result["planning_footprint_native_wet_pixel_superset_holds"] is True
    assert result["planning_footprint_native_wet_pixel_coverage_fraction"] == pytest.approx(1.0)
