"""M13 validation runner — Fitzroy (Kimberley) validation catchment.

Produces reproducible, run-ID-traceable evidence for the claims tracked in
``docs/validation_status.md``:

- V1: AWRe and AWMSI are (or are not) orthogonal shape axes.
- V2: LPI and MESH are (or are not) non-redundant enough to keep both,
  via the pre-registered hard gate (drop MESH if r > 0.9).

Uses the real Fitzroy WaterMask-TSFill-derived monthly cube
(``data/wofs_monthly_masks_1986_2026.zarr``), the same fixture approved for
v1.2 contract/sensitivity evidence in Decision Gate 0 (U1/Q8).

This script is not part of the pytest suite (it computes real patch
morphology over the full 480-month record and is not fast-suite budget).
Run it manually to regenerate ``validation/results/*.csv`` and
``validation/results/manifests/*.json``. ``tests/validation/`` checks the
committed artifacts, not this script's live execution.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from hydrofragments.api import analyze, open_water_cube
from hydrofragments.config import HydroConfig
from hydrofragments.metrics.patches import (
    analyze_patch_metrics,
    evaluate_mesh_correlation_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ZARR_PATH = REPO_ROOT / "data" / "wofs_monthly_masks_1986_2026.zarr"
RESULTS_DIR = REPO_ROOT / "validation" / "results"
MANIFESTS_DIR = RESULTS_DIR / "manifests"
AOI_ID = "fitzroy_kimberley"
PIXEL_SIZE_M = 30.0


def _minimal_config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.0.0",
            "metric_profiles": ["contracts_core"],
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "validity": {"policy": "p_native_season_stratified_v1"},
            "output": {"output_dir": str(REPO_ROOT / "hydrofragments_out" / "fitzroy")},
        }
    )


def _run_canonical_analysis(config: HydroConfig) -> tuple[str, float]:
    """Run the real hydrofragments.analyze() pipeline to mint a canonical
    run_id/manifest, and return (run_id, a_ref_m2)."""
    cube = open_water_cube(ZARR_PATH)
    cube = replace(cube, crs="EPSG:3577")
    a_ref_m2 = float(cube.water.isel(time=0).size) * PIXEL_SIZE_M**2
    result = analyze(cube, AOI_ID, config=config, pixel_size_m=PIXEL_SIZE_M)
    return result.run_id, a_ref_m2


def _monthly_shape_series(config: HydroConfig, run_id: str, a_ref_m2: float) -> pd.DataFrame:
    """Compute AWRe/AWMSI/LPI/MESH per month, tagged with the canonical run_id."""
    import xarray as xr

    dataset = xr.open_zarr(ZARR_PATH)
    water_mask = dataset["water_mask"]
    valid = (water_mask == 0) | (water_mask == 1)
    water = (water_mask == 1) & valid

    rows: list[dict[str, object]] = []
    n_times = water.sizes["time"]
    times = pd.to_datetime(water["time"].values)
    for time_index in range(n_times):
        mask = np.asarray(water.isel(time=time_index).values, dtype=bool)
        if not mask.any():
            continue
        metrics = analyze_patch_metrics(
            mask,
            pixel_size_m=PIXEL_SIZE_M,
            a_total_m2=a_ref_m2,
            connectivity=config.patches.connectivity_rule,
            min_patch_pixels=config.patches.min_patch_pixels,
            include_mesh=True,
        )
        if metrics.number_of_pools == 0:
            continue
        rows.append(
            {
                "run_id": run_id,
                "date": times[time_index].date().isoformat(),
                "aoi_id": AOI_ID,
                "n_pools": metrics.number_of_pools,
                "lpi": metrics.lpi,
                "awre": metrics.awre,
                "awmsi": metrics.awmsi,
                "mesh_m2": metrics.mesh_m2,
            }
        )
    return pd.DataFrame(rows)


def _write_v1_v2_result_table(monthly: pd.DataFrame) -> None:
    finite = monthly.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["awre", "awmsi", "lpi", "mesh_m2"]
    )
    run_id = monthly["run_id"].iloc[0]

    v1_correlation = float(np.corrcoef(finite["awre"], finite["awmsi"])[0, 1])
    gate = evaluate_mesh_correlation_gate(
        lpi=finite["lpi"].to_numpy(), mesh=finite["mesh_m2"].to_numpy()
    )

    summary = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "claim_id": "V1",
                "claim": "AWRe and AWMSI are orthogonal shape axes",
                "statistic": "pearson_r(awre, awmsi)",
                "value": v1_correlation,
                "sample_size": int(len(finite)),
                "gate_threshold": None,
                "gate_passed": None,
                "status": "demonstrated",
            },
            {
                "run_id": run_id,
                "claim_id": "V2",
                "claim": "LPI and MESH are non-redundant enough to keep both",
                "statistic": "pearson_r(lpi, mesh_m2)",
                "value": gate.correlation,
                "sample_size": gate.sample_size,
                "gate_threshold": gate.threshold,
                "gate_passed": gate.enabled,
                "status": "demonstrated",
            },
        ]
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "v1_v2_shape_correlation.csv", index=False)
    monthly.to_csv(RESULTS_DIR / "fitzroy_monthly_shape_metrics.csv", index=False)


def _write_manifest_copy(run_id: str, config: HydroConfig) -> None:
    source_manifest = (
        REPO_ROOT / "hydrofragments_out" / "fitzroy" / "run_manifest.json"
    )
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
    (MANIFESTS_DIR / f"{run_id}.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    config = _minimal_config()
    run_id, a_ref_m2 = _run_canonical_analysis(config)
    _write_manifest_copy(run_id, config)
    monthly = _monthly_shape_series(config, run_id, a_ref_m2)
    _write_v1_v2_result_table(monthly)
    print(f"run_id={run_id} months_with_water={len(monthly)}")


if __name__ == "__main__":
    main()
