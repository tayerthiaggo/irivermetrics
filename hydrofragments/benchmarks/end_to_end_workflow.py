"""Real user-input-to-table benchmark: the W3.7 promotion gate.

This module is the sibling of :mod:`hydrofragments.benchmarks.cpu_baseline`
(same dataclass-spec + JSON/Markdown dual-output convention, same
orchestration-only scope -- it owns timing, subprocess isolation, and report
assembly, never a scientific kernel). Where ``cpu_baseline`` times synthetic
in-memory reduction stages, this module times the REAL pipeline a user
actually runs: :func:`hydrofragments.io.dea.open_wo_statistics_for_zoning`
-> ``hydroseason.acquire_wofs_cache`` -> :func:`hydrofragments.api.analyze`,
against a real catchment fixture, over live DEA/STAC network calls for the
cold-acquisition phase.

Scope (controller-approved, see task-3.7 report for the full rationale):
Fitzroy is the only catchment with local fixtures in this repository.
Gilbert (thin/braided) and a large catchment have no local AOI/drainage
fixture in this worktree and are NOT run here -- their gate fields are
explicit ``null`` with a documented ``skipped_reason``, never fabricated,
never silently dropped from the schema (see ``GILBERT_CASE``/
``LARGE_CATCHMENT_CASE`` below and ``_skipped_case_result``).

Each case (a factor/workers candidate) is run in an isolated subprocess
(``python -m hydrofragments.benchmarks._e2e_worker``, invoked via
``subprocess.run`` from :func:`_run_case_subprocess`) so that:

- peak RSS is measured for exactly one case's real memory footprint (via
  ``psutil`` process-tree polling from the parent), never contaminated by
  a previous case's Dask/xarray/GDAL state left resident in this process;
- a cold run's on-disk cache directory is fully separate per case, so a
  "cold" measurement never accidentally reuses another case's warm cache.

``psutil`` is not a declared HydroFragments dependency (it rides in
transitively today); peak-RSS collection degrades to an explicit ``null``
rather than a hard import failure if it is ever unavailable, matching
``cpu_baseline.py``'s "unavailable measurements stay null, never invented"
convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

SCHEMA_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Case specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpec:
    """One factor/workers candidate to benchmark against a real catchment."""

    candidate_id: str
    factor: int
    workers: int
    executor_kind: str = "thread"  # analyze()'s only reachable metric-processing mode; see module docstring in report


@dataclass(frozen=True)
class RealCaseSpec:
    """A real catchment fixture this worktree can actually run."""

    case_id: str
    description: str
    aoi_path: str
    drainage_path: str | None
    start_date: str
    end_date: str
    aoi_id: str
    candidates: tuple[CandidateSpec, ...]


@dataclass(frozen=True)
class DeferredCaseSpec:
    """A required plan case with no local fixture in this worktree."""

    case_id: str
    description: str
    skipped_reason: str


FITZROY_CASE = RealCaseSpec(
    case_id="fitzroy_compact",
    description="Fitzroy (Kimberley) -- compact catchment, the plan's 'compact Fitzroy' required case",
    aoi_path="data/fitzroy_kimberley_aoi.geojson",
    drainage_path="data/fitzroy_kimberley_drainage.geojson",
    start_date="2020-01-01",
    end_date="2020-12-31",
    aoi_id="fitzroy_kimberley",
    candidates=(
        CandidateSpec(candidate_id="factor4_workers1", factor=4, workers=1),
        CandidateSpec(candidate_id="factor4_workers2", factor=4, workers=2),
        CandidateSpec(candidate_id="factor4_workers4", factor=4, workers=4),
        CandidateSpec(candidate_id="factor3_workers1", factor=3, workers=1),
        CandidateSpec(candidate_id="factor3_workers2", factor=3, workers=2),
        CandidateSpec(candidate_id="factor3_workers4", factor=3, workers=4),
    ),
)

GILBERT_CASE = DeferredCaseSpec(
    case_id="gilbert_thin_braided",
    description="Gilbert -- thin/braided catchment, plan's 'cold Gilbert >= 30% faster than full AOI' gate",
    skipped_reason=(
        "No local AOI/drainage fixture for Gilbert exists in this worktree "
        "(only data/fitzroy_kimberley_*). The controller-approved scope for "
        "this task run is Fitzroy-only; sourcing a Gilbert geometry was "
        "explicitly out of scope, not attempted. This gate is NOT satisfied "
        "by any number in this report."
    ),
)

LARGE_CATCHMENT_CASE = DeferredCaseSpec(
    case_id="large_catchment",
    description="One large catchment -- plan's third required case",
    skipped_reason=(
        "No local AOI/drainage fixture for a large catchment exists in this "
        "worktree. The controller-approved scope for this task run is "
        "Fitzroy-only; sourcing a large-catchment geometry was explicitly "
        "out of scope, not attempted. This gate is NOT satisfied by any "
        "number in this report."
    ),
)


# ---------------------------------------------------------------------------
# Gate thresholds (plan text, Fitzroy-applicable subset only)
# ---------------------------------------------------------------------------

COMPACT_REGRESSION_MAX_FRACTION = 0.10  # "compact Fitzroy regression no worse than 10%"
PEAK_RSS_MAX_FRACTION_OF_SERIAL = 1.25  # "peak RSS no more than 125% of serial baseline"


# ---------------------------------------------------------------------------
# Subprocess case runner
# ---------------------------------------------------------------------------


def _psutil_module():
    try:
        import psutil  # type: ignore

        return psutil
    except ImportError:
        return None


def _run_case_subprocess(
    *,
    candidate: CandidateSpec,
    real_case: RealCaseSpec,
    cache_dir: Path,
    output_dir: Path,
    mode: str,
    poll_interval_s: float = 0.05,
) -> dict[str, Any]:
    """Run one cold/warm candidate in an isolated subprocess; return its JSON report.

    ``mode`` is ``"cold"`` (cache_dir must not already contain this
    candidate's data -- caller's responsibility to pass a fresh/empty
    ``cache_dir`` for a genuinely cold run) or ``"warm"`` (reruns against
    the SAME ``cache_dir`` a prior cold run just populated, so
    ``hydroseason.acquire_wofs_cache``'s own resumability contract makes
    zero further STAC calls).

    Peak RSS is polled from the parent process via ``psutil`` at
    ``poll_interval_s`` while the child subprocess runs, tracking the
    child's own RSS (not the parent's) -- ``None`` if ``psutil`` is
    unavailable.
    """

    payload = {
        "candidate_id": candidate.candidate_id,
        "factor": candidate.factor,
        "workers": candidate.workers,
        "executor_kind": candidate.executor_kind,
        "aoi_path": real_case.aoi_path,
        "drainage_path": real_case.drainage_path,
        "start_date": real_case.start_date,
        "end_date": real_case.end_date,
        "aoi_id": real_case.aoi_id,
        "cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "mode": mode,
    }

    proc = subprocess.Popen(
        [sys.executable, "-m", "hydrofragments.benchmarks._e2e_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(REPO_ROOT),
        text=True,
    )
    assert proc.stdin is not None
    proc.stdin.write(json.dumps(payload))
    proc.stdin.close()

    psutil = _psutil_module()
    peak_rss_bytes: int | None = None
    ps_proc = None
    if psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

    wall_start = time.perf_counter()
    while True:
        if ps_proc is not None:
            try:
                rss = ps_proc.memory_info().rss
                peak_rss_bytes = rss if peak_rss_bytes is None else max(peak_rss_bytes, rss)
                for child in ps_proc.children(recursive=True):
                    try:
                        crss = child.memory_info().rss
                        peak_rss_bytes = max(peak_rss_bytes, crss)
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            proc.wait(timeout=poll_interval_s)
            break
        except subprocess.TimeoutExpired:
            continue
    wall_seconds = time.perf_counter() - wall_start

    stdout, stderr = proc.communicate()

    if proc.returncode != 0:
        raise RuntimeError(
            f"end-to-end benchmark subprocess for candidate="
            f"{candidate.candidate_id!r} mode={mode!r} exited "
            f"{proc.returncode}. stderr:\n{stderr}\nstdout:\n{stdout}"
        )

    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"end-to-end benchmark subprocess for candidate="
            f"{candidate.candidate_id!r} mode={mode!r} produced non-JSON "
            f"stdout:\n{stdout}\nstderr:\n{stderr}"
        ) from exc

    result["subprocess_wall_seconds"] = wall_seconds
    result["peak_rss_bytes"] = peak_rss_bytes
    return result


def _run_real_candidate(
    *, candidate: CandidateSpec, real_case: RealCaseSpec, workdir: Path
) -> dict[str, Any]:
    """Run one candidate's cold-then-warm pair against a real catchment; return its record."""

    cache_dir = workdir / candidate.candidate_id / "wofs_cache"
    cold_output_dir = workdir / candidate.candidate_id / "output_cold"
    warm_output_dir = workdir / candidate.candidate_id / "output_warm"
    if cache_dir.exists():
        shutil.rmtree(cache_dir)

    cold = _run_case_subprocess(
        candidate=candidate,
        real_case=real_case,
        cache_dir=cache_dir,
        output_dir=cold_output_dir,
        mode="cold",
    )
    warm = _run_case_subprocess(
        candidate=candidate,
        real_case=real_case,
        cache_dir=cache_dir,
        output_dir=warm_output_dir,
        mode="warm",
    )

    cold_ok = cold.get("status") == "ok"
    warm_ok = warm.get("status") == "ok"

    metrics_equal = None
    n_water_equal = None
    superset_holds = None
    if cold_ok and warm_ok:
        metrics_equal = cold.get("metrics_digest") == warm.get("metrics_digest")
        n_water_equal = cold.get("n_water_by_month") == warm.get("n_water_by_month")
        superset_holds = cold.get("planning_footprint_native_wet_pixel_superset_holds")

    return {
        "candidate_id": candidate.candidate_id,
        "factor": candidate.factor,
        "workers": candidate.workers,
        "executor_kind": candidate.executor_kind,
        "cold": cold,
        "warm": warm,
        "cold_warm_metrics_equal": metrics_equal,
        "cold_warm_n_water_equal": n_water_equal,
        "planning_footprint_native_wet_pixel_superset_holds": superset_holds,
    }


def _skipped_case_result(spec: DeferredCaseSpec) -> dict[str, Any]:
    return {
        "case_id": spec.case_id,
        "description": spec.description,
        "status": "skipped",
        "skipped_reason": spec.skipped_reason,
        "candidates": None,
        "gates": {
            "cold_median_at_least_30pct_faster_than_full_aoi": None,
            "warm_rerun_at_least_80pct_faster_than_cold_full_aoi": None,
            "zero_stac_calls_on_warm_rerun": None,
        },
    }


# ---------------------------------------------------------------------------
# Top-level matrix runner
# ---------------------------------------------------------------------------


def run_end_to_end_matrix(*, workdir: str | Path) -> dict[str, Any]:
    """Run the real Fitzroy candidate matrix (live network) plus deferred stubs.

    Returns a JSON-safe mapping: schema version, environment, the real
    Fitzroy case's per-candidate cold/warm results and promotion-gate
    verdicts, explicit ``skipped`` entries for Gilbert and the large
    catchment (never fabricated, never omitted), and a
    ``recommendation`` derived only from Fitzroy evidence.
    """

    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)

    candidate_records = [
        _run_real_candidate(candidate=candidate, real_case=FITZROY_CASE, workdir=workdir_path)
        for candidate in FITZROY_CASE.candidates
    ]

    fitzroy_result = _summarize_fitzroy(candidate_records)

    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": "end_to_end_workflow",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope_note": (
            "Controller-approved reduced scope: only Fitzroy has a local AOI/"
            "drainage fixture in this worktree. Gilbert and the large-"
            "catchment required cases are explicitly not run -- see their "
            "'skipped' entries below, not omitted from this schema."
        ),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases": {
            "fitzroy_compact": fitzroy_result,
            "gilbert_thin_braided": _skipped_case_result(GILBERT_CASE),
            "large_catchment": _skipped_case_result(LARGE_CATCHMENT_CASE),
        },
        "recommendation": _recommendation(fitzroy_result),
    }


def _summarize_fitzroy(candidate_records: list[dict[str, Any]]) -> dict[str, Any]:
    serial = next(
        (r for r in candidate_records if r["workers"] == 1 and r["factor"] == 4), None
    )
    serial_cold_seconds = None
    serial_peak_rss = None
    if serial is not None and serial["cold"].get("status") == "ok":
        serial_cold_seconds = serial["cold"]["timings_seconds"]["total"]
        serial_peak_rss = serial["cold"].get("peak_rss_bytes")

    gated_candidates = []
    for record in candidate_records:
        cold = record["cold"]
        warm = record["warm"]
        cold_ok = cold.get("status") == "ok"
        warm_ok = warm.get("status") == "ok"

        regression_fraction = None
        regression_within_10pct = None
        if cold_ok and serial_cold_seconds:
            cold_seconds = cold["timings_seconds"]["total"]
            regression_fraction = (cold_seconds - serial_cold_seconds) / serial_cold_seconds
            regression_within_10pct = regression_fraction <= COMPACT_REGRESSION_MAX_FRACTION

        rss_fraction_of_serial = None
        rss_within_125pct = None
        if cold_ok and serial_peak_rss:
            candidate_rss = cold.get("peak_rss_bytes")
            if candidate_rss is not None:
                rss_fraction_of_serial = candidate_rss / serial_peak_rss
                rss_within_125pct = rss_fraction_of_serial <= PEAK_RSS_MAX_FRACTION_OF_SERIAL

        warm_speedup_fraction = None
        if cold_ok and warm_ok:
            cold_seconds = cold["timings_seconds"]["total"]
            warm_seconds = warm["timings_seconds"]["total"]
            if cold_seconds:
                warm_speedup_fraction = (cold_seconds - warm_seconds) / cold_seconds

        # Timing/RSS-only pass: decoupled from the exact-equality gates below
        # so a real, honest miss on planning_footprint_native_wet_pixel_superset_holds
        # (which would mean the W1.5 superset proof itself failed for this
        # candidate's footprint -- a genuinely surprising, worth-separating
        # signal) doesn't also hide otherwise-useful timing/RSS signal from
        # this report.
        timing_rss_gates_pass = bool(
            cold_ok
            and warm_ok
            and (regression_within_10pct in (True, None))
            and (rss_within_125pct in (True, None))
        )
        all_gates_pass = bool(
            timing_rss_gates_pass
            and record["cold_warm_metrics_equal"]
            and record["cold_warm_n_water_equal"]
            and record["planning_footprint_native_wet_pixel_superset_holds"]
        )

        gated_candidates.append(
            {
                **record,
                "regression_fraction_vs_serial": regression_fraction,
                "regression_within_10pct_gate": regression_within_10pct,
                "peak_rss_fraction_of_serial": rss_fraction_of_serial,
                "peak_rss_within_125pct_gate": rss_within_125pct,
                "warm_speedup_fraction_vs_own_cold": warm_speedup_fraction,
                "timing_rss_gates_pass": timing_rss_gates_pass,
                "all_measurable_gates_pass": all_gates_pass,
            }
        )

    return {
        "case_id": FITZROY_CASE.case_id,
        "description": FITZROY_CASE.description,
        "status": "ok",
        "serial_baseline_candidate_id": serial["candidate_id"] if serial else None,
        "candidates": gated_candidates,
        "gates": {
            "exact_metrics_table_and_per_metric_value_equality": (
                "measured per-candidate as cold_warm_metrics_equal (cold vs "
                "warm rerun of the SAME candidate); cross-candidate full-AOI-"
                "vs-pruned equality was not separately run this session -- "
                "see report notes"
            ),
            "n_water_equality_every_month": "measured per-candidate as cold_warm_n_water_equal",
            "count_wet_planning_footprint_covers_100pct_of_native_wet_pixels": (
                "measured per-candidate as "
                "planning_footprint_native_wet_pixel_superset_holds (native_mask <= "
                "expand(coarse_mask), the same superset property W1.5 proves in "
                "hydroseason)"
            ),
            "cold_gilbert_at_least_30pct_faster_than_full_aoi": None,
            "warm_rerun_at_least_80pct_faster_than_cold_full_aoi": (
                "Fitzroy's own warm-vs-cold speedup is reported per candidate "
                "as warm_speedup_fraction_vs_own_cold; the plan's 80% figure "
                "is stated against Gilbert specifically and is NOT claimed "
                "satisfied here even where Fitzroy's own number exceeds it"
            ),
            "compact_fitzroy_regression_no_worse_than_10pct": (
                "measured per-candidate as regression_within_10pct_gate, "
                "relative to the factor=4/workers=1 serial baseline"
            ),
            "peak_rss_no_more_than_125pct_of_serial": (
                "measured per-candidate as peak_rss_within_125pct_gate"
            ),
        },
    }


def _recommendation(fitzroy_result: dict[str, Any]) -> dict[str, Any]:
    if fitzroy_result.get("status") != "ok":
        return {
            "verdict": "no_recommendation",
            "reason": "Fitzroy case did not complete successfully.",
        }

    passing = [
        c for c in fitzroy_result["candidates"] if c.get("all_measurable_gates_pass")
    ]
    if passing:
        fastest = min(passing, key=lambda c: c["cold"]["timings_seconds"]["total"])
        return {
            "verdict": "fastest_passing_candidate_identified",
            "fastest_passing_candidate_id": fastest["candidate_id"],
            "fastest_passing_factor": fastest["factor"],
            "fastest_passing_workers": fastest["workers"],
            "reason": (
                f"candidate={fastest['candidate_id']} passed all measurable "
                "Fitzroy-only gates (metrics/n_water/coverage equality, "
                "regression <=10%, peak RSS <=125% of serial) and had the "
                "lowest cold total seconds among passing candidates."
            ),
            "promotion_status": (
                "NOT promoted to any production default in this task. The "
                "plan's promotion gate requires thin/braided Gilbert, compact "
                "Fitzroy, AND one large catchment all passing before a default "
                "is changed; this run covers Fitzroy only. Recorded here as "
                "evidence for a future task once Gilbert/large-catchment "
                "fixtures exist."
            ),
        }

    # No candidate passed every gate (e.g.
    # planning_footprint_native_wet_pixel_superset_holds could be False/None
    # for a genuinely surprising reason, or footprint was None this run --
    # see report notes). Still surface which candidate would have won on
    # timing/RSS alone, so this report stays informative rather than a flat
    # "nothing passed" with no further signal.
    timing_rss_passing = [
        c for c in fitzroy_result["candidates"] if c.get("timing_rss_gates_pass")
    ]
    timing_rss_note = None
    if timing_rss_passing:
        fastest_timing = min(
            timing_rss_passing, key=lambda c: c["cold"]["timings_seconds"]["total"]
        )
        timing_rss_note = (
            f"candidate={fastest_timing['candidate_id']} was fastest among "
            "candidates passing ONLY the timing/RSS gates (regression <=10%, "
            "peak RSS <=125%) -- informational, since it did NOT also pass "
            "the exact-equality/coverage gates required for "
            "all_measurable_gates_pass."
        )

    return {
        "verdict": "no_passing_candidate",
        "reason": (
            "No candidate passed every measurable Fitzroy gate. Default "
            "settings are left unchanged."
        ),
        "timing_rss_only_note": timing_rss_note,
    }


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def _fmt_seconds(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_bytes(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{int(value):,}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# End-to-end workflow benchmark (W3.7 benchmark gate)",
        "",
        payload["scope_note"],
        "",
        f"- Schema: `{payload['schema_version']}`",
        f"- Created: `{payload['created_at']}`",
        "",
        "## Fitzroy (compact) -- real, live-network run",
        "",
    ]

    fitzroy = payload["cases"]["fitzroy_compact"]
    if fitzroy["status"] != "ok":
        lines.append(f"Status: `{fitzroy['status']}`")
    else:
        lines.extend(
            [
                "| Candidate | factor | workers | cold total s | warm total s | "
                "warm speedup | regression vs serial | peak RSS (cold) | RSS vs serial | "
                "native-wet superset holds (cold) | superset coverage (cold) | "
                "valid obs. fraction (cold) | timing/RSS gates | all gates |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | :---: | :---: |",
            ]
        )
        for c in fitzroy["candidates"]:
            cold_total = (
                c["cold"]["timings_seconds"]["total"]
                if c["cold"].get("status") == "ok"
                else None
            )
            warm_total = (
                c["warm"]["timings_seconds"]["total"]
                if c["warm"].get("status") == "ok"
                else None
            )
            superset_holds = c["cold"].get("planning_footprint_native_wet_pixel_superset_holds")
            superset_coverage = c["cold"].get("planning_footprint_native_wet_pixel_coverage_fraction")
            valid_obs_fraction = c["cold"].get("analysis_mask_valid_observation_fraction")
            superset_holds_str = (
                "n/a" if superset_holds is None else ("yes" if superset_holds else "no")
            )
            lines.append(
                f"| {c['candidate_id']} | {c['factor']} | {c['workers']} | "
                f"{_fmt_seconds(cold_total)} | {_fmt_seconds(warm_total)} | "
                f"{_fmt_pct(c['warm_speedup_fraction_vs_own_cold'])} | "
                f"{_fmt_pct(c['regression_fraction_vs_serial'])} | "
                f"{_fmt_bytes(c['cold'].get('peak_rss_bytes'))} | "
                f"{_fmt_pct(c['peak_rss_fraction_of_serial'])} | "
                f"{superset_holds_str} | "
                f"{_fmt_pct(superset_coverage)} | "
                f"{_fmt_pct(valid_obs_fraction)} | "
                f"{'yes' if c.get('timing_rss_gates_pass') else 'no'} | "
                f"{'yes' if c['all_measurable_gates_pass'] else 'no'} |"
            )
        lines.append("")
        lines.append("Promotion gates applied (Fitzroy-only subset -- see module docstring):")
        for gate, detail in fitzroy["gates"].items():
            lines.append(f"- `{gate}`: {detail}")

    lines.extend(["", "## Recommendation", ""])
    rec = payload["recommendation"]
    for key, value in rec.items():
        lines.append(f"- **{key}**: {value}")

    lines.extend(["", "## Deferred (not run -- no local fixture)", ""])
    for case_id in ("gilbert_thin_braided", "large_catchment"):
        case = payload["cases"][case_id]
        lines.append(f"### {case['description']}")
        lines.append("")
        lines.append(f"Status: `{case['status']}`")
        lines.append("")
        lines.append(f"Reason: {case['skipped_reason']}")
        lines.append("")
        lines.append("Gate fields (explicit null, not fabricated, not omitted):")
        for gate_name, gate_value in case["gates"].items():
            lines.append(f"- `{gate_name}`: `{gate_value}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_end_to_end_baseline(
    output_dir: str | Path, *, workdir: str | Path
) -> dict[str, Any]:
    """Run the real matrix and write machine-readable JSON plus human Markdown."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = run_end_to_end_matrix(workdir=workdir)
    json_path = target / "end_to_end_workflow.json"
    markdown_path = target / "end_to_end_workflow.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    payload["report_files"] = {"json": str(json_path), "markdown": str(markdown_path)}
    return payload


__all__ = [
    "CandidateSpec",
    "RealCaseSpec",
    "DeferredCaseSpec",
    "FITZROY_CASE",
    "GILBERT_CASE",
    "LARGE_CATCHMENT_CASE",
    "run_end_to_end_matrix",
    "write_end_to_end_baseline",
]
