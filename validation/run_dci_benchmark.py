"""V6 benchmark: length-weighted RC_pair vs riverconn DCI on the Fitzroy reach network.

Run manually: python validation/run_dci_benchmark.py
Writes validation/results/v6_dci_benchmark.csv

This script does NOT ship DCI as a runtime metric. It runs the V6 reference
benchmark (spec section 6.17/6.18): it computes HydroFragments' pure-Python
``compute_length_weighted_rc_pair`` (the DCI form) and the independently
implemented ``riverconn::index_calculation`` (R, Baldan et al. 2022) on the
*same* real Fitzroy reach graph + same representative-month active-edge set,
then reports whether the two agree. Per Q4, DCI stays citation-only unless a
maintainer separately decides otherwise -- passing this benchmark does not
auto-enable it.

Loading (AOI + real drainage) mirrors validation/run_fitzroy_validation.py
(M13): same approved Fitzroy cube and drainage, EPSG:3577.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

from hydrofragments.metrics.connectivity import (
    build_fixed_graph,
    compute_length_weighted_rc_pair,
)
from hydrofragments.spatial.connectivity_context import reach_wet_any_month

REPO_ROOT = Path(__file__).resolve().parents[1]
ZARR_PATH = REPO_ROOT / "data" / "wofs_monthly_masks_1986_2026.zarr"
DRAINAGE_PATH = REPO_ROOT / "data" / "fitzroy_kimberley_drainage.gpkg"
# V6 is a cross-implementation *formula-parity* benchmark, not a
# hydrofragments.analyze() pipeline run, so it has no run_id/manifest. It
# therefore lives in a dedicated benchmarks/ subdirectory, deliberately NOT
# in validation/results/*.csv (which tests/validation/ requires to be
# run_id-traceable to an analyze() manifest).
RESULTS_DIR = REPO_ROOT / "validation" / "results" / "benchmarks"
R_SCRIPT = REPO_ROOT / "validation" / "dci_benchmark.R"
TARGET_CRS = "EPSG:3577"
REACH_BUFFER_M = 60.0  # 2 pixel widths at 30 m (U9 default)

# R is not on PATH in a fresh shell; use the confirmed install location, with
# an Rscript-on-PATH fallback so the script stays portable.
RSCRIPT_CANDIDATES = [
    r"C:\Users\00101125\AppData\Local\Programs\R\R-4.6.1\bin\Rscript.exe",
    "Rscript",
]


def _rscript_exe() -> str:
    for candidate in RSCRIPT_CANDIDATES:
        if candidate == "Rscript" or Path(candidate).exists():
            return candidate
    return "Rscript"


def _load_drainage_and_water() -> tuple[gpd.GeoDataFrame, xr.DataArray]:
    """Load real Fitzroy drainage (EPSG:3577) and the monthly water cube.

    Mirrors run_fitzroy_validation.py's cube handling: the approved zarr's
    water_mask is 1=water / 0=dry-valid; anything else is invalid. The cube
    already carries real projected x/y (EPSG:3577) that overlap the drainage.
    """
    drainage = gpd.read_file(DRAINAGE_PATH)
    if drainage.crs is None:
        raise RuntimeError("drainage CRS missing")
    drainage = drainage.to_crs(TARGET_CRS)

    dataset = xr.open_zarr(ZARR_PATH)
    water_mask = dataset["water_mask"]
    valid = (water_mask == 0) | (water_mask == 1)
    water = (water_mask == 1) & valid
    water = water.rio.write_crs(TARGET_CRS) if hasattr(water, "rio") else water
    return drainage, water


def _reach_wet_for_month(
    drainage: gpd.GeoDataFrame, water: xr.DataArray, time_index: int
) -> dict[str, bool]:
    """Per-reach wet flag for one representative month (skeleton-in-buffer, U9)."""
    single = water.isel(time=[time_index])
    return reach_wet_any_month(drainage, single, buffer_m=REACH_BUFFER_M)


def _pick_representative_month(water: xr.DataArray) -> int:
    """Pick a month with substantial (near-median non-zero) water extent."""
    extents = np.asarray(
        [int(np.asarray(water.isel(time=t).values, dtype=bool).sum())
         for t in range(water.sizes["time"])]
    )
    nonzero = extents[extents > 0]
    if nonzero.size == 0:
        raise RuntimeError("no month has any water")
    target = np.median(nonzero)
    candidates = np.where(extents > 0)[0]
    return int(candidates[np.argmin(np.abs(extents[candidates] - target))])


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    drainage, water = _load_drainage_and_water()

    # Real reach lengths from geometry (EPSG:3577, metres).
    length_by_node = {
        str(row["HydroID"]): float(row.geometry.length)
        for _, row in drainage.iterrows()
    }

    # Fixed graph: nodes = reaches wet in >=1 month; edges = topology adjacency.
    wet_any = reach_wet_any_month(drainage, water, buffer_m=REACH_BUFFER_M)
    topology = [
        {"HydroID": str(row["HydroID"]),
         "From_Node": row["From_Node"], "To_Node": row["To_Node"]}
        for _, row in drainage.iterrows()
    ]
    graph = build_fixed_graph(topology, wet_any_month=wet_any)

    # Representative month's active edges: edge (i,j) active iff BOTH endpoint
    # reaches are wet that month (direct-wet-touch, gap_threshold=0).
    month_index = _pick_representative_month(water)
    reach_wet_month = _reach_wet_for_month(drainage, water, month_index)
    wet_gap_by_edge: dict[tuple[str, str], int | None] = {}
    for edge in graph.edges:
        both_wet = reach_wet_month.get(edge[0], False) and reach_wet_month.get(edge[1], False)
        wet_gap_by_edge[edge] = 0 if both_wet else None

    # --- Python side: HydroFragments' own DCI-form RC_pair ---
    python_dci_pct = compute_length_weighted_rc_pair(
        graph,
        wet_gap_by_edge=wet_gap_by_edge,
        gap_threshold=0,
        length_by_node={node: length_by_node[node] for node in graph.nodes},
    )

    # --- Export the SAME graph + active-edge set for the R side ---
    nodes_csv = RESULTS_DIR / "_v6_nodes.csv"
    edges_csv = RESULTS_DIR / "_v6_edges.csv"
    out_csv = RESULTS_DIR / "v6_riverconn_raw.csv"
    pd.DataFrame(
        {"name": list(graph.nodes),
         "length": [length_by_node[n] for n in graph.nodes]}
    ).to_csv(nodes_csv, index=False)
    pd.DataFrame(
        {"from": [e[0] for e in graph.edges],
         "to": [e[1] for e in graph.edges],
         "pass": [1.0 if wet_gap_by_edge[e] == 0 else 0.0 for e in graph.edges]}
    ).to_csv(edges_csv, index=False)

    # --- R side: riverconn::index_calculation on the identical configuration ---
    proc = subprocess.run(
        [_rscript_exe(), str(R_SCRIPT), str(nodes_csv), str(edges_csv), str(out_csv)],
        capture_output=True, text=True,
    )
    print("--- Rscript stdout ---")
    print(proc.stdout)
    if proc.returncode != 0:
        print("--- Rscript stderr ---", file=sys.stderr)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"riverconn Rscript failed (exit {proc.returncode})")

    r_result = pd.read_csv(out_csv)
    riverconn_dci_pct = float(r_result["riverconn_dci_pct"].iloc[0])

    # --- Agreement ---
    abs_diff = abs(python_dci_pct - riverconn_dci_pct)
    denom = max(abs(riverconn_dci_pct), 1e-12)
    rel_pct_diff = 100.0 * abs_diff / denom
    agreement_pct = max(0.0, 100.0 - rel_pct_diff)

    summary = pd.DataFrame([{
        "benchmark": "V6_RC_pair_vs_riverconn_DCI",
        "aoi_id": "fitzroy_kimberley",
        "representative_month_index": month_index,
        "n_nodes": len(graph.nodes),
        "n_structural_edges": len(graph.edges),
        "n_active_edges": sum(1 for e in graph.edges if wet_gap_by_edge[e] == 0),
        "riverconn_version": str(r_result["riverconn_version"].iloc[0]),
        "n_components": int(r_result["n_components"].iloc[0]),
        "python_rc_pair_dci_pct": python_dci_pct,
        "riverconn_dci_pct": riverconn_dci_pct,
        "abs_diff_pct_points": abs_diff,
        "rel_pct_diff": rel_pct_diff,
        "agreement_pct": agreement_pct,
    }])
    final_csv = RESULTS_DIR / "v6_dci_benchmark.csv"
    summary.to_csv(final_csv, index=False)

    # Transient R inputs -- the committed evidence is the comparison summary
    # and the raw riverconn output, not these intermediate exports.
    nodes_csv.unlink(missing_ok=True)
    edges_csv.unlink(missing_ok=True)

    print("\n=== V6 benchmark ===")
    print(f"month_index={month_index} nodes={len(graph.nodes)} "
          f"structural_edges={len(graph.edges)} "
          f"active_edges={summary['n_active_edges'].iloc[0]}")
    print(f"python RC_pair (DCI form) : {python_dci_pct:.6f} %")
    print(f"riverconn DCI             : {riverconn_dci_pct:.6f} %")
    print(f"abs diff                  : {abs_diff:.6e} pct-points")
    print(f"agreement                 : {agreement_pct:.6f} %")
    print(f"written -> {final_csv}")


if __name__ == "__main__":
    main()
