# HydroFragments User-Ready Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take HydroFragments from "numerically correct but scale-blocked" to "user-ready at real satellite-cube scale," fixing every item in `pipelines/hydrofragments-audit/audit-report.md` (1 blocker, 4 major, 13 minor) in dependency order.

**Architecture:** Three tracks executed in order. **Track A (code-readiness):** gate metric compute by selected profile before any kernel runs (B1), then remove redundant/quadratic work (M2, M3, M4) and harden edge cases (m1, m5, m6, m7). **Track B (parity-gated perf):** micro-optimizations that touch verified numeric outputs (m3, m4, m9) — each preceded by a golden-fixture parity test that pins current output *before* the rewrite. **Track C (docs/paper):** reconcile spec §6.17 with the shipped estimator and add provenance/unit notes (M1, m11, m12). Correctness reference (CPU labels/morphology) is never replaced by distributed/GPU paths in this milestone.

**Tech Stack:** Python, xarray + Dask, NumPy, SciPy (`ndimage`), scikit-image (`regionprops`, `medial_axis`, `distance_transform_edt`), dask-image, rasterio, pytest.

## Global Constraints

- **No numeric-output change on any verified metric.** Any task that touches a formula path (labels, morphology, connectivity counting, patch metrics) must be preceded by a golden-fixture parity test asserting bit-for-bit / `np.allclose` identity of output before and after. If a rewrite changes a number, it is wrong — revert.
- **CPU label + morphology stays the numerical reference.** No distributed-label or GPU-morphology path enters this milestone (audit conflict C4).
- **Occurrence estimator ships season-stratified (per U2/Q1).** Do not "fix" it back to pooled. Spec catches up to code, not the reverse (audit M1, user domain call).
- **`connectivity` is an optional profile.** Nothing in Track A/B may alter core (non-connectivity) results (`metrics/connectivity.py` module contract).
- **Preserve all warning-flag + resolution-floor semantics** exactly through any patch-path refactor (edge flags, width-resolution-floor suppression).
- Dep floors already committed: geopandas `1.1.4`, setuptools `78.1.1` — do not downgrade.
- Every task ends with `pytest` green on the touched module + a commit.

---

## File Structure

Files created or modified, grouped by responsibility:

**Track A — profile gating + redundant/quadratic work + edge hardening**
- Modify: `hydrofragments/api.py` — `analyze()` resolves `selected_ids` *before* compute; dispatch only requested metric families (B1). Batch temporal-summary `.compute()` (m8).
- Modify: `hydrofragments/metrics/compat.py` — keep `section_compat_rows()` as legacy-only path; APSEC computed once over time axis (m2).
- Modify: `hydrofragments/metrics/extent.py` — vectorized APSEC over time axis (m2); per-month coverage floor flag (m7).
- Modify: `hydrofragments/metrics/patches.py` — single per-month `PatchAnalysis` bundle: label once, crop once, measure once with `include_width` toggle (M2).
- Modify: `hydrofragments/metrics/connectivity.py` — O(V) adjacency dict + `Counter`-based reachable-pair count (M3).
- Modify: `hydrofragments/spatial/connectivity_context.py` — single multilabel reach raster, skeletonize once per month, index labels under skeleton (M4).
- Modify: `hydrofragments/metrics/dynamics.py` — dry-down degrades to NaN on missing end-dry (m1); `_first_crossing` asserts/sorts ordering (m5).

**Track B — parity-gated micro-optimizations**
- Create: `tests/parity/test_label_normalization_parity.py`, `tests/parity/test_morphology_width_parity.py`, `tests/parity/test_regionprops_parity.py` — golden fixtures pinning current output.
- Modify: `hydrofragments/patches/labels.py` — `np.bincount` + lookup table instead of `np.unique` sort/`searchsorted` (m3).
- Modify: `hydrofragments/patches/morphology.py` — distance-returning `medial_axis` (single EDT) (m4); `regionprops_table` bulk path (m9).

**Track C — docs/paper**
- Modify: `docs/HydroFragments_v1.2_spec.md` — §6.17 stratified occurrence; θ/MMU provenance; unit-convention notes (M1, m11, m12).
- Modify: `hydrofragments/metrics/registry.py` — MESH unit label `km2` → `m2` (m6).
- Modify: `hydrofragments/metrics/patches.py` — MESH value comment / relabel alignment (m6).

**Cross-cutting scaffolding**
- Create: `tests/parity/conftest.py` — shared synthetic monthly-mask fixtures used across parity + gating tests.

---

## Track A — Code-Readiness

### Task 1: Golden snapshot of `analyze()` row output (B1 safety net)

Lock canonical row presence/order/values before touching the compute path (audit conflict C3 — `_records_from_compat_rows` parity is the main risk).

**Files:**
- Create: `tests/parity/conftest.py`
- Create: `tests/gating/test_analyze_row_snapshot.py`
- Reference: `hydrofragments/api.py:597-700`, `hydrofragments/metrics/compat.py`

**Interfaces:**
- Consumes: `hydrofragments.api.analyze(...)` (existing public signature — do not change it).
- Produces: `synthetic_cube` fixture (a small `WaterCube`-shaped object with `water`, `valid_obs`, `crs`, `source`, `time`/`y`/`x`), reused by Tasks 2, 6, 7.

- [ ] **Step 1: Write the fixture**

In `tests/parity/conftest.py`, build a deterministic 6-month × 12×12 cube with two stable water patches and one intermittent one:

```python
import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def synthetic_cube():
    """6 months, 12x12, deterministic water + validity."""
    rng = np.random.default_rng(1729)
    t, y, x = 6, 12, 12
    water = np.zeros((t, y, x), dtype=bool)
    water[:, 2:5, 2:5] = True                 # stable patch A
    water[:, 7:10, 7:11] = True               # stable patch B
    water[::2, 5:7, 5:7] = True               # intermittent patch C (even months)
    valid = np.ones((t, y, x), dtype=bool)
    valid[3, :, :] = rng.random((y, x)) > 0.1  # one partially-invalid month
    times = np.array(
        ["2015-01", "2015-02", "2015-03", "2015-04", "2015-05", "2015-06"],
        dtype="datetime64[M]",
    ).astype("datetime64[ns]")
    ys = np.arange(y, dtype=float) * -30.0 + 8_000_000.0
    xs = np.arange(x, dtype=float) * 30.0 + 500_000.0
    ds = xr.Dataset(
        {
            "water": (("time", "y", "x"), water),
            "valid_obs": (("time", "y", "x"), valid),
        },
        coords={"time": times, "y": ys, "x": xs},
    )
    ds.attrs["crs"] = "EPSG:32750"
    ds.attrs["source"] = "synthetic"
    return ds
```

- [ ] **Step 2: Write the snapshot test**

In `tests/gating/test_analyze_row_snapshot.py`, call `analyze()` on the fixture through the same wrapper production uses, and assert on a stable serialization (metric, tier, value rounded). Discover the real construction call first:

Run: `pytest -q tests/compat/test_hydrofragments_public_api.py -x` to see how the suite already builds an `analyze()` input, and mirror that exact wrapper here rather than inventing one.

```python
import json
import numpy as np
from hydrofragments.api import analyze  # adjust import to match compat suite usage


def _serialize(records):
    return sorted(
        (r.metric, r.tier if hasattr(r, "tier") else "", round(float(r.value), 6))
        for r in records
        if np.isfinite(getattr(r, "value", float("nan")))
    )


def test_analyze_row_snapshot(synthetic_cube, snapshot_path=None):
    records = analyze(synthetic_cube)  # match real call from compat suite
    got = _serialize(records)
    # Write once, then freeze:
    # json.dump(got, open("tests/gating/analyze_snapshot.json", "w"), indent=2)
    expected = json.load(open("tests/gating/analyze_snapshot.json"))
    assert got == [tuple(row) for row in expected]
```

- [ ] **Step 3: Generate the frozen snapshot**

Uncomment the `json.dump` line, run once, re-comment. Commit the JSON as the golden baseline.

Run: `pytest tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS (self-consistent on first freeze).

- [ ] **Step 4: Commit**

```bash
git add tests/parity/conftest.py tests/gating/test_analyze_row_snapshot.py tests/gating/analyze_snapshot.json
git commit -m "test: snapshot analyze() row output before B1 gating"
```

---

### Task 2: B1 — resolve `selected_ids` before compute, dispatch only requested families

**Files:**
- Modify: `hydrofragments/api.py:597-656`
- Modify: `hydrofragments/metrics/compat.py` (mark `section_compat_rows()` legacy-only path)
- Test: `tests/gating/test_profile_gating.py`

**Interfaces:**
- Consumes: `resolve_metrics(...)` → `.selected` (already used at `api.py:627`); `synthetic_cube` fixture (Task 1).
- Produces: no signature change to `analyze()`. Internal: `selected_ids` set is computed at the *top* of the compute region and gates each family dispatch.

- [ ] **Step 1: Write the failing gating test**

Assert that when only a narrow profile is requested, the expensive families are not computed. Use a spy on the patch-metric entry point:

```python
from unittest import mock
from hydrofragments.api import analyze


def test_narrow_profile_skips_patch_morphology(synthetic_cube):
    with mock.patch("hydrofragments.api.analyze_patch_metrics") as patch_spy:
        analyze(synthetic_cube, metric_profiles=["persistence"])  # match real kwarg name
    patch_spy.assert_not_called()


def test_persistence_profile_still_emits_occurrence(synthetic_cube):
    records = analyze(synthetic_cube, metric_profiles=["persistence"])
    metrics = {r.metric for r in records}
    assert "occurrence" in metrics
```

Adjust `metric_profiles=` to the real `analyze()` kwarg (confirm against `api.py:626-655`).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/gating/test_profile_gating.py -v`
Expected: FAIL — `patch_spy` *is* called today because `section_compat_rows()` computes everything before selection.

- [ ] **Step 3: Reorder `analyze()` — selection first, then dispatch**

Move the `selected_ids` block (currently `api.py:626-655`) to run **before** `section_compat_rows()`. Then guard each family. Replace the eager `rows = section_compat_rows(...)` region with conditional dispatch:

```python
    # --- selection resolved FIRST (moved up from :626) ---
    selected_ids = {
        spec.metric_id for spec in resolve_metrics(
            config.metric_profiles,
            available_dependencies=(...),   # keep the exact tuple from old :629-653
        ).selected
    }

    records: list = []

    _PERSISTENCE_IDS = {"occurrence", "recurrence", "refuge", "hydroperiod"}
    _PATCH_IDS = {"number_of_pools", "lpi", "awre", "awmsi", "mesh"}

    if selected_ids & (_PERSISTENCE_IDS | _PATCH_IDS | {"apsec"}):
        rows = section_compat_rows(       # legacy row bridge, now gated
            monthly["water"],
            section=aoi_id,
            section_area_km2=section_area_km2,
            pixel_size_m=pixel_size_m,
            config=config,
            selected_ids=selected_ids,    # NEW: pass through so compat skips unselected families
        )
        records = [
            r for r in _records_from_compat_rows(
                rows, run_id=run_id, config=config, catchment_id=catchment,
                aoi_id=aoi_id, resolution_m=pixel_size_m, crs=crs, source=cube.source,
            )
            if r.metric in selected_ids
        ]
```

Then thread `selected_ids` into `section_compat_rows()` in `compat.py` so it computes occurrence/refuge/APSEC/patch-morphology only for families whose ids are present. Keep the channel-profile and pool-width extension blocks (`api.py:657-700`) exactly as-is — they already gate on `selected_ids`.

- [ ] **Step 4: Run gating tests + snapshot**

Run: `pytest tests/gating/ -v`
Expected: both new tests PASS; **`test_analyze_row_snapshot` still PASS** (identical rows for a full profile — proves reordering didn't change output).

- [ ] **Step 5: Run full compat suite**

Run: `pytest tests/compat/test_hydrofragments_public_api.py -v`
Expected: PASS (no public-API regression).

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/api.py hydrofragments/metrics/compat.py tests/gating/test_profile_gating.py
git commit -m "fix: gate analyze() metric compute by selected profile before kernels run (B1)"
```

---

### Task 3: m2 — compute APSEC once over the time axis

Subsumed into B1's rewrite region; do it now while the compat path is open (audit m2: becomes a bigger Dask-graph bug if `.load()` removed without it).

**Files:**
- Modify: `hydrofragments/metrics/extent.py:87-106`
- Modify: `hydrofragments/metrics/compat.py:146-155`
- Test: `tests/metrics/test_apsec_vectorized.py`

**Interfaces:**
- Consumes: `compute_apsec(monthly, ...)` — extend to accept a stacked time dimension and return one record per timestamp.
- Produces: `compute_apsec` returns `tuple[ApsecRecord, ...]` keyed/ordered by timestamp; callers map records by `date`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from hydrofragments.metrics.extent import compute_apsec


def test_apsec_batched_matches_per_month(synthetic_cube):
    water = synthetic_cube["water"]
    batched = compute_apsec(water, pixel_size_m=30.0)   # one call, all months
    per_month = [
        compute_apsec(water.isel(time=[i]), pixel_size_m=30.0)[0]
        for i in range(water.sizes["time"])
    ]
    assert [r.value for r in batched] == [r.value for r in per_month]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/metrics/test_apsec_vectorized.py -v`
Expected: FAIL — current `compute_apsec` is single-month (`extent.py:87/91`).

- [ ] **Step 3: Vectorize `compute_apsec`**

Reduce over `("y", "x")` across the whole time axis in one xarray reduction, emit one `ApsecRecord` per `time` coordinate. Preserve the exact per-month arithmetic (percent of section area). Update `compat.py:146/155` to call once and map records by timestamp instead of looping months.

- [ ] **Step 4: Run + snapshot**

Run: `pytest tests/metrics/test_apsec_vectorized.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS both (values identical — this is a batching change, not a formula change).

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/metrics/extent.py hydrofragments/metrics/compat.py tests/metrics/test_apsec_vectorized.py
git commit -m "perf: compute APSEC once over time axis instead of per month (m2)"
```

---

### Task 4: M2 — single per-month patch bundle (label once, crop once, measure once)

**Files:**
- Modify: `hydrofragments/metrics/patches.py:210-271`
- Test: `tests/metrics/test_patch_bundle.py`

**Interfaces:**
- Consumes: `label_components`, `iter_component_crops`, `bucket_component_crops`, `measure_components` (existing).
- Produces: `analyze_patch_bundle(mask, *, pixel_size_m, a_total_m2, connectivity, min_patch_pixels, target_component_pixels, include_mesh, include_width, resolution_floor_pixels=None) -> tuple[PatchMetricResult, PoolWidthDistribution | None]` — one label+crop pass, core metrics always, width distribution only when `include_width`.

- [ ] **Step 1: Write the failing test**

Assert the bundle produces identical results to the two existing separate calls, with one label pass:

```python
from unittest import mock
import numpy as np
from hydrofragments.metrics import patches


def _mask():
    m = np.zeros((12, 12), dtype=bool)
    m[2:5, 2:5] = True
    m[7:10, 7:11] = True
    return m


def test_bundle_matches_separate_calls():
    mask = _mask()
    core_ref = patches.analyze_patch_metrics(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0, include_mesh=True
    )
    width_ref = patches.analyze_pool_width_distribution(
        mask, pixel_size_m=30.0, resolution_floor_pixels=2.0
    )
    core, width = patches.analyze_patch_bundle(
        mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
        include_mesh=True, include_width=True, resolution_floor_pixels=2.0,
    )
    assert core == core_ref
    assert width == width_ref


def test_bundle_labels_once():
    mask = _mask()
    with mock.patch.object(
        patches, "label_components", wraps=patches.label_components
    ) as spy:
        patches.analyze_patch_bundle(
            mask, pixel_size_m=30.0, a_total_m2=12 * 12 * 900.0,
            include_mesh=False, include_width=True, resolution_floor_pixels=2.0,
        )
    assert spy.call_count == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/metrics/test_patch_bundle.py -v`
Expected: FAIL — `analyze_patch_bundle` does not exist.

- [ ] **Step 3: Implement the bundle**

Add `analyze_patch_bundle()` that labels+crops+buckets once, then calls `measure_components(..., include_width=include_width)` a single time, and derives both `compute_patch_metrics(...)` and (when `include_width`) `compute_pool_width_distribution(...)` from the shared `properties`. Keep `analyze_patch_metrics` and `analyze_pool_width_distribution` as thin wrappers over the bundle so external callers and the snapshot don't move. Preserve `EdgeFlag.N1` and width-resolution-floor suppression exactly.

- [ ] **Step 4: Rewire the caller**

In `api.py`, where both core patch metrics and `pool_width` are selected, call `analyze_patch_bundle(..., include_width=("pool_width" in selected_ids))` once per month instead of `analyze_patch_metrics` + `_pool_width_records` labeling separately.

- [ ] **Step 5: Run + snapshot**

Run: `pytest tests/metrics/test_patch_bundle.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS both.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/metrics/patches.py hydrofragments/api.py tests/metrics/test_patch_bundle.py
git commit -m "perf: single per-month patch label/crop/measure bundle (M2)"
```

---

### Task 5: M3 — O(V) connectivity graph build + reachable-pair count

**Files:**
- Modify: `hydrofragments/metrics/connectivity.py:74-91` (adjacency), `:159-170` (pair count)
- Test: `tests/metrics/test_connectivity_scaling.py`

**Interfaces:**
- Consumes: `build_fixed_graph`, `compute_realised_connectivity` (existing signatures unchanged).
- Produces: identical `FixedGraph.edges` (same order) and identical `rc_pair_pct` — pure algebra change.

- [ ] **Step 1: Write the failing/guard test**

Pin current output on a known topology, then assert equality after rewrite (this test protects the number; it passes today and must keep passing):

```python
from math import comb
from collections import Counter
from hydrofragments.metrics.connectivity import (
    build_fixed_graph, compute_realised_connectivity,
)


def _topology():
    # chain: A->n1->n2->n3 ; B branches at n2
    return [
        {"HydroID": "A", "From_Node": 0, "To_Node": 1},
        {"HydroID": "B", "From_Node": 1, "To_Node": 2},
        {"HydroID": "C", "From_Node": 2, "To_Node": 3},
        {"HydroID": "D", "From_Node": 1, "To_Node": 9},  # sibling of B
    ]


def test_graph_edges_stable():
    g = build_fixed_graph(_topology(), wet_any_month={k: True for k in "ABCD"})
    # A.To=1 == B.From=1 and D.From=1 -> edges (A,B),(A,D); B.To=2 == C.From=2 -> (B,C)
    assert g.edges == (("A", "B"), ("A", "D"), ("B", "C"))


def test_rc_pair_value_stable():
    g = build_fixed_graph(_topology(), wet_any_month={k: True for k in "ABCD"})
    edges = {e: 0 for e in g.edges}  # all active, gap 0
    res = compute_realised_connectivity(g, wet_gap_by_edge=edges, gap_threshold=0)
    # all 4 connected -> comb(4,2)/comb(4,2) = 100
    assert res.rc_pair_pct == 100.0
```

- [ ] **Step 2: Run to verify it passes on current code**

Run: `pytest tests/metrics/test_connectivity_scaling.py -v`
Expected: PASS (this is the pin — capture current behavior before rewrite).

- [ ] **Step 3: Rewrite adjacency to O(V)**

Replace the nested loop (`connectivity.py:80-85`) with a `From_Node → [HydroID]` index, then emit an edge for each kept node by looking up children of its `To_Node`. Preserve edge emission order (iterate `nodes` in existing order, children sorted by their position in `nodes`).

```python
    from_index: dict[object, list[str]] = {}
    for node in nodes:
        fn, _ = from_to_node[node]
        from_index.setdefault(fn, []).append(node)

    node_order = {node: i for i, node in enumerate(nodes)}
    edges = []
    for node_a in nodes:
        _, to_node_a = from_to_node[node_a]
        children = sorted(
            (c for c in from_index.get(to_node_a, []) if node_order[c] > node_order[node_a]),
            key=node_order.__getitem__,
        )
        edges.extend((node_a, node_b) for node_b in children)
```

- [ ] **Step 4: Rewrite reachable-pair count to O(V)**

Replace the O(V²) double loop (`connectivity.py:163-169`) with a `Counter` over roots:

```python
        roots = [_find(parent, node) for node in graph.nodes]
        sizes = Counter(roots)
        reachable_pairs = sum(comb(size, 2) for size in sizes.values())
        rc_pair_pct = 100.0 * reachable_pairs / total_pairs
```

Add `from collections import Counter` at the top.

- [ ] **Step 5: Run to verify unchanged**

Run: `pytest tests/metrics/test_connectivity_scaling.py -v`
Expected: PASS — identical edges and RC value.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/metrics/connectivity.py tests/metrics/test_connectivity_scaling.py
git commit -m "perf: O(V) connectivity adjacency + reachable-pair count (M3)"
```

---

### Task 6: M4 — single multilabel reach raster in `reach_wet_any_month`

**Files:**
- Modify: `hydrofragments/spatial/connectivity_context.py:49-98`
- Test: `tests/spatial/test_reach_wet_parity.py`

**Interfaces:**
- Consumes: `reach_wet_any_month(drainage, water, *, buffer_m)` (signature unchanged).
- Produces: identical `dict[str, bool]` output — pins reach-attribution semantics (skeleton pixel inside reach buffer).

- [ ] **Step 1: Write the parity test**

Build a tiny drainage GeoDataFrame + water DataArray, capture current output, assert equality after rewrite:

```python
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import LineString
from hydrofragments.spatial.connectivity_context import reach_wet_any_month


def _fixture():
    y = np.arange(10, dtype=float) * -30.0 + 300.0
    x = np.arange(10, dtype=float) * 30.0
    water = np.zeros((2, 10, 10), dtype=bool)
    water[0, 4:6, 1:8] = True  # horizontal channel wet in month 0
    da = xr.DataArray(water, dims=("time", "y", "x"), coords={"y": y, "x": x})
    drainage = gpd.GeoDataFrame(
        {"HydroID": ["R1", "R2"]},
        geometry=[LineString([(30, 150), (210, 150)]),
                  LineString([(30, 30), (210, 30)])],
        crs="EPSG:32750",
    )
    return drainage, da


def test_reach_wet_output_stable():
    drainage, da = _fixture()
    got = reach_wet_any_month(drainage, da, buffer_m=60.0)
    assert got == {"R1": True, "R2": False}
```

- [ ] **Step 2: Run to pin current behavior**

Run: `pytest tests/spatial/test_reach_wet_parity.py -v`
Expected: PASS (adjust the asserted dict to whatever current code returns, then freeze).

- [ ] **Step 3: Rewrite to a single multilabel raster**

Rasterize each reach buffer once into an integer label raster (`reach_index → label`; for overlapping buffers, keep a coord→set-of-reaches map). Skeletonize each month **once** (already the case), then flag wet reaches by taking `np.unique(label_raster[skeleton])` and marking those reach ids — no per-reach full-frame `&`:

```python
    reach_ids = [str(r["HydroID"]) for _, r in drainage.iterrows()]
    label_raster = np.zeros((len(y_coords), len(x_coords)), dtype=np.int32)
    for i, (_, reach) in enumerate(drainage.iterrows(), start=1):
        m = _reach_buffer_mask(reach.geometry.buffer(buffer_m),
                               transform=transform, y_coords=y_coords, x_coords=x_coords)
        label_raster[m] = i  # NOTE: overlapping buffers — last-writer-wins;
                             # if fixture shows overlap, switch to coord->set map instead

    result = {rid: False for rid in reach_ids}
    for month_index in range(water.sizes["time"]):
        if all(result.values()):
            break
        month_mask = np.asarray(water.isel(time=month_index).values, dtype=bool)
        if not month_mask.any():
            continue
        skeleton = medial_axis(month_mask)
        for lab in np.unique(label_raster[skeleton]):
            if lab != 0:
                result[reach_ids[lab - 1]] = True
    return result
```

If the parity test exposes an overlapping-buffer case where two reaches share pixels, keep a `coord → list[reach_index]` fallback for those pixels instead of last-writer-wins — the audit flags overlapping-buffer attribution as the risk (M4).

- [ ] **Step 4: Run to verify unchanged**

Run: `pytest tests/spatial/test_reach_wet_parity.py -v`
Expected: PASS — identical dict.

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/spatial/connectivity_context.py tests/spatial/test_reach_wet_parity.py
git commit -m "perf: single multilabel reach raster in reach_wet_any_month (M4)"
```

---

### Task 7: m1 + m5 — dynamics edge hardening

**Files:**
- Modify: `hydrofragments/metrics/dynamics.py:96-105` (m1), `:199-212` (m5)
- Test: `tests/metrics/test_dynamics_edges.py`

**Interfaces:**
- Consumes: `compute_extent_contraction`, `_end_dry_value`, `_first_crossing` (module-internal).
- Produces: `_end_dry_value` returns `float("nan")` instead of raising when no matching end-dry record; result carries `end_dry_disagreement_pp = NaN` in that case. `_first_crossing` asserts monotonic month ordering.

- [ ] **Step 1: Write the failing tests**

```python
import numpy as np
import pytest
from hydrofragments.metrics import dynamics


def test_end_dry_value_degrades_to_nan():
    # no record matching the requested end-dry month
    from datetime import date
    val = dynamics._end_dry_value([], date(2015, 6, 1))
    assert np.isnan(val)


def test_first_crossing_rejects_unsorted():
    from datetime import date
    series = [(date(2015, 3, 1), 1.0), (date(2015, 1, 1), 9.0)]  # out of order
    with pytest.raises(AssertionError):
        dynamics._first_crossing(series, end_dry_month=date(2015, 1, 1), threshold=5.0)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/metrics/test_dynamics_edges.py -v`
Expected: FAIL — `_end_dry_value` raises `ValueError`; `_first_crossing` silently accepts unsorted input.

- [ ] **Step 3: Implement m1 (degrade to NaN)**

Replace the `raise ValueError(...)` at `dynamics.py:105` with `return float("nan")`. In `compute_extent_contraction`, `abs(nan - x)` already yields NaN, and `composite_sensitive = nan > tolerance` → `False`; that is acceptable — the HY slope stays computable instead of aborting.

- [ ] **Step 4: Implement m5 (assert ordering)**

At the top of `_first_crossing`, assert the series is non-decreasing in month before iterating:

```python
    months = [m for m, _ in series]
    assert months == sorted(months), "reconnection series must be month-sorted"
```

- [ ] **Step 5: Run to verify pass + snapshot**

Run: `pytest tests/metrics/test_dynamics_edges.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS all.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/metrics/dynamics.py tests/metrics/test_dynamics_edges.py
git commit -m "fix: degrade dry-down to NaN on missing end-dry; assert reconnection series ordering (m1, m5)"
```

---

### Task 8: m7 — APSEC per-month coverage floor flag

**Files:**
- Modify: `hydrofragments/metrics/extent.py:94-106`
- Test: `tests/metrics/test_apsec_coverage_floor.py`

**Interfaces:**
- Consumes: `ApsecRecord` (add field), `compute_apsec` (Task 3 batched form).
- Produces: `ApsecRecord.low_coverage_flag: bool` (default `False`), set `True` when valid-pixel count for that month is below `config`-supplied floor. Value unchanged; flag is additive metadata.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from hydrofragments.metrics.extent import compute_apsec


def test_low_coverage_month_flagged(synthetic_cube):
    recs = compute_apsec(
        synthetic_cube["water"], pixel_size_m=30.0,
        valid_obs=synthetic_cube["valid_obs"], min_valid_fraction=0.95,
    )
    # month 3 has ~10% invalid pixels injected in the fixture
    assert recs[3].low_coverage_flag is True
    assert all(recs[i].low_coverage_flag is False for i in (0, 1, 2, 4, 5))
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/metrics/test_apsec_coverage_floor.py -v`
Expected: FAIL — `ApsecRecord` has no `low_coverage_flag`; `compute_apsec` takes no `valid_obs`/`min_valid_fraction`.

- [ ] **Step 3: Add the field + optional floor**

Add `low_coverage_flag: bool = False` to `ApsecRecord`. In `compute_apsec`, if `valid_obs` supplied, compute per-month valid fraction and set the flag below `min_valid_fraction`. Leave the APSEC value arithmetic untouched (occurrence already floors; this only annotates).

- [ ] **Step 4: Run + snapshot**

Run: `pytest tests/metrics/test_apsec_coverage_floor.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS both (value unchanged; snapshot serializes value, not flag).

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/metrics/extent.py tests/metrics/test_apsec_coverage_floor.py
git commit -m "feat: add APSEC per-month coverage floor flag (m7)"
```

---

### Task 9: m8 — batch temporal AOI-summary materialization

**Files:**
- Modify: `hydrofragments/api.py:438-460`, `hydrofragments/metrics/persistence.py:178-200`
- Test: `tests/api/test_temporal_summary_batch.py`

**Interfaces:**
- Consumes: existing temporal-summary builders in `persistence.py`.
- Produces: one `xr.Dataset` of requested scalar summaries computed with a single `.compute()`; scalar rows extracted from the materialized result. Values identical.

- [ ] **Step 1: Write the guard test**

Count `.compute()` / `.item()` invocations via a Dask callback or a spy, and assert one materialization:

```python
from unittest import mock
from hydrofragments.api import analyze


def test_temporal_summaries_materialize_once(synthetic_cube):
    import dask.array as darr
    calls = {"n": 0}
    real = darr.Array.compute

    def counting(self, *a, **k):
        calls["n"] += 1
        return real(self, *a, **k)

    with mock.patch.object(darr.Array, "compute", counting):
        analyze(synthetic_cube, metric_profiles=["persistence"])
    assert calls["n"] <= 2  # was many; now batched
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/api/test_temporal_summary_batch.py -v`
Expected: FAIL — current path triggers a separate `.item()` per scalar (`persistence.py:178-200`).

- [ ] **Step 3: Batch the summaries**

Build all requested temporal summaries into one dataset, call `.compute()` once, then read scalars from the materialized result instead of per-summary `.item()`.

- [ ] **Step 4: Run + snapshot**

Run: `pytest tests/api/test_temporal_summary_batch.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/api.py hydrofragments/metrics/persistence.py tests/api/test_temporal_summary_batch.py
git commit -m "perf: batch temporal AOI summaries into one materialization (m8)"
```

---

### Task 10: m10 — honor `chunks` in `open_water_cube()`

**Files:**
- Modify: `hydrofragments/api.py:67-76`, `hydrofragments/chunks.py:45`
- Test: `tests/api/test_open_cube_chunks.py`

**Interfaces:**
- Consumes: `open_water_cube(..., chunks=...)`.
- Produces: opened cube whose Dask chunk sizes match the requested `chunks`; chosen chunks recorded in the run manifest.

- [ ] **Step 1: Write the failing test**

```python
from hydrofragments.api import open_water_cube


def test_open_cube_honors_chunks(tmp_zarr_path):   # provide a small zarr fixture
    cube = open_water_cube(tmp_zarr_path, chunks={"time": 1, "y": 6, "x": 6})
    assert cube.water.data.chunksize[1] == 6
```

Add a `tmp_zarr_path` fixture writing the `synthetic_cube` to zarr in `conftest.py`.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/api/test_open_cube_chunks.py -v`
Expected: FAIL — `chunks` currently discarded (`api.py:67-76`).

- [ ] **Step 3: Thread `chunks` through**

Pass `chunks` into the zarr open / array wrap in `open_water_cube`, use `chunks.py:45` planning, and record chosen chunks in the manifest dict.

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/api/test_open_cube_chunks.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/api.py hydrofragments/chunks.py tests/api/test_open_cube_chunks.py
git commit -m "feat: honor chunks in open_water_cube and record in manifest (m10)"
```

---

## Track B — Parity-Gated Micro-Optimizations

Each task in this track: **write the golden-fixture parity test first, freeze current output, then rewrite, then prove identity.** Do not proceed to the rewrite step until the freeze passes (audit conflict C2 — science wins, perf gated).

### Task 11: m3 — bincount label normalization (parity-gated)

**Files:**
- Create: `tests/parity/test_label_normalization_parity.py`
- Modify: `hydrofragments/patches/labels.py:51-77`

**Interfaces:**
- Consumes: `label_components(mask, *, connectivity, min_patch_pixels)` (signature unchanged).
- Produces: identical `LabelResult.labels` (row-major first-pixel ID ordering) and `count`.

- [ ] **Step 1: Write + freeze the parity fixture**

Pin labels on a fragmented multi-component mask (dense and sparse-Dask), asserting exact array + count:

```python
import numpy as np
import dask.array as da
from hydrofragments.patches.labels import label_components


def _fragmented():
    m = np.zeros((16, 16), dtype=bool)
    m[1:3, 1:3] = True; m[1:3, 6:9] = True
    m[10:14, 2:5] = True; m[12, 12] = True  # singleton -> dropped by min_patch
    return m


def test_label_ids_row_major_stable():
    m = _fragmented()
    res = label_components(m, connectivity=8, min_patch_pixels=3)
    # freeze: np.save("tests/parity/labels_golden.npy", res.labels)
    golden = np.load("tests/parity/labels_golden.npy")
    np.testing.assert_array_equal(res.labels, golden)
    assert res.count == int(golden.max())


def test_label_dask_matches_dense():
    m = _fragmented()
    dense = label_components(m, min_patch_pixels=3).labels
    chunked = label_components(da.from_array(m, chunks=(8, 8)), min_patch_pixels=3).labels
    np.testing.assert_array_equal(dense, chunked)
```

- [ ] **Step 2: Freeze golden + run**

Uncomment the `np.save`, run once, re-comment, commit `labels_golden.npy`.

Run: `pytest tests/parity/test_label_normalization_parity.py -v`
Expected: PASS.

- [ ] **Step 3: Rewrite `_filter_and_normalize` with bincount**

Replace `np.unique(sort)` + `searchsorted` (`labels.py:54-55/75`) with `np.bincount` for counts and a lookup table indexed by raw label ID → `lookup[raw_labels]`. Preserve the row-major first-pixel ordering of retained IDs exactly (`first` occurrence order). O(P+K) instead of O(P log P).

- [ ] **Step 4: Run parity to prove identity**

Run: `pytest tests/parity/test_label_normalization_parity.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS all — arrays bit-identical.

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/patches/labels.py tests/parity/test_label_normalization_parity.py tests/parity/labels_golden.npy
git commit -m "perf: bincount label normalization, parity-gated (m3)"
```

---

### Task 12: m4 + m9 — single-EDT width + bulk regionprops (parity-gated)

**Files:**
- Create: `tests/parity/test_morphology_width_parity.py`, `tests/parity/test_regionprops_parity.py`
- Modify: `hydrofragments/patches/morphology.py:45-68`

**Interfaces:**
- Consumes: `measure_components(crops, *, pixel_size_m, include_width)` (signature unchanged).
- Produces: identical `PatchProperties` (`width_pixels`, `width_m`, `major_axis_length_m`, `perimeter_m`) for every component.

- [ ] **Step 1: Write + freeze both parity fixtures**

```python
import numpy as np
from hydrofragments.patches.components import ComponentCrop, BBox
from hydrofragments.patches.morphology import measure_components


def _crop():
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:6, 1:8] = True
    mask[3, 8:9] = False
    return ComponentCrop(label=1, bbox=BBox(0, 0, 7, 9), mask=mask)


def test_width_and_major_axis_stable():
    (p,) = measure_components([_crop()], pixel_size_m=30.0, include_width=True)
    # freeze these floats on first run, then assert
    assert round(p.width_pixels, 9) == 6.0            # replace with frozen value
    assert round(p.major_axis_length_m, 6) == 180.0   # replace with frozen value
    assert round(p.perimeter_m, 6) == p.perimeter_m
```

Run once to capture the true frozen floats, paste them in, re-run.

- [ ] **Step 2: Run to pin current behavior**

Run: `pytest tests/parity/test_morphology_width_parity.py tests/parity/test_regionprops_parity.py -v`
Expected: PASS (frozen to current output).

- [ ] **Step 3: m4 — single EDT via distance-returning medial_axis**

Replace the two-call pattern (`morphology.py:55-56`, `medial_axis(mask)` then separate `distance_transform_edt(mask)`) with `medial_axis(mask, return_distance=True)` → `(axis, dist)`, then `width_pixels = float((2.0 * dist[axis]).max())`. One EDT instead of two.

- [ ] **Step 4: m9 — bulk regionprops**

Replace per-component `regionprops(mask.astype(np.uint8))` with a `regionprops_table(labels, properties=["axis_major_length"])` bulk call over the labeled crop bucket, mapping results back by label. **Only if** the parity test proves `axis_major_length` identical to the per-component path; if any float differs, keep per-component `regionprops` (audit: major-axis parity must be certified — science-adjacent).

- [ ] **Step 5: Run parity to prove identity**

Run: `pytest tests/parity/ tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS all — every float identical.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/patches/morphology.py tests/parity/test_morphology_width_parity.py tests/parity/test_regionprops_parity.py
git commit -m "perf: single-EDT width + bulk regionprops, parity-gated (m4, m9)"
```

---

## Track C — Docs / Paper

### Task 13: m6 — MESH unit label/value alignment

**Files:**
- Modify: `hydrofragments/metrics/registry.py:142-144`
- Modify: `hydrofragments/metrics/patches.py:191` (comment/value)
- Test: `tests/metrics/test_mesh_units.py`

**Interfaces:**
- Consumes: `registry` MESH `MetricSpec`; `compute_patch_metrics(..., include_mesh=True)`.
- Produces: registry unit label matches the value MESH actually carries (`m2`). No consumer currently reads it wrong (`validation/run_fitzroy_validation.py:107` reads raw `mesh_m2`).

- [ ] **Step 1: Write the failing test**

```python
from hydrofragments.metrics import registry


def test_mesh_unit_label_matches_value():
    spec = next(s for s in registry.all_specs() if s.metric_id == "mesh")  # adjust accessor
    assert spec.unit == "m2"   # value is m^2 (patches.py:191), so label must be m2
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/metrics/test_mesh_units.py -v`
Expected: FAIL — registry labels `km2` (`registry.py:144`) but value is m² (`patches.py:191`).

- [ ] **Step 3: Fix the label**

Change `registry.py:144` `"km2"` → `"m2"` (value already correct; MESH never emitted at runtime, so this is a dormant-trap fix, not a behavior change). Add a one-line comment at `patches.py:191` noting the value is m² by contract.

- [ ] **Step 4: Run + snapshot**

Run: `pytest tests/metrics/test_mesh_units.py tests/gating/test_analyze_row_snapshot.py -v`
Expected: PASS both.

- [ ] **Step 5: Commit**

```bash
git add hydrofragments/metrics/registry.py hydrofragments/metrics/patches.py tests/metrics/test_mesh_units.py
git commit -m "fix: align MESH registry unit label with m2 value (m6)"
```

---

### Task 14: M1 + m11 + m12 — spec reconcile and provenance notes

Documentation only. No code path changes. This is the publication-integrity track; blocks the *paper*, not *use*.

**Files:**
- Modify: `docs/HydroFragments_v1.2_spec.md` (§6.17 and provenance/unit sections)

**Interfaces:** none (docs).

- [ ] **Step 1: Confirm the shipped estimator wording**

Read `hydrofragments/metrics/persistence.py:86-131` to quote the exact season-stratified estimator: per-calendar-month `Σ(water&valid)/Σ(valid)` over years, then equal-weight mean over months × 100.

- [ ] **Step 2: Rewrite spec §6.17 (M1)**

Replace the pooled `Σ_t W / Σ_t V · 100` formula with the stratified form as shipped, and add an `[AUDIT FIX 2026-07-18]` note: "U2/Q1 supersedes the pooled formula; stratified corrects seasonal MNAR missingness. Cite stratified as the shipped occurrence estimator." (Per the audit's single user domain call, stratified is the shipped choice; note pooled is available as an option if the paper prefers it.)

- [ ] **Step 3: Add provenance + non-comparability notes (m11)**

Document θ=90% RA (`persistence.py:151`) and `min_patch_pixels=3` MMU (`patches.py:217`, vs Paper 2's 2) alongside every derived claim, and state that N/LPI/MESH are **not comparable across 30 m vs 10 m** resolutions.

- [ ] **Step 4: Add unit-convention notes (m12)**

Note hydroperiod is a fraction while occurrence/recurrence are percent (`persistence.py:203`); clarify recurrence ≈ occurrence-core (`:165-188`); document AWRe all-or-nothing NaN (`patches.py:192-196`).

- [ ] **Step 5: Commit**

```bash
git add docs/HydroFragments_v1.2_spec.md
git commit -m "docs: reconcile spec §6.17 with stratified occurrence; add provenance + unit notes (M1, m11, m12)"
```

---

## Not in this plan (deferred by audit)

Per `audit-report.md` "Not blocking":
- **CUDA / GPU morphology** (efficiency #14) — scaffold is truthful; keep advertised as candidate/incomplete.
- **Distributed labels** (m13, efficiency #4) — bounded by one 2-D month; CPU stays the reference (conflict C4). Defer until profiling proves need.
- **Benchmark harness expansion** (efficiency #15) — add stages *after* B1/M2/M3 land so numbers reflect the fixed hot path.
- **Empirical validation claims** (science V1–V8) — validate-before-paper with Gilbert data; formulas already sound.

---

## Self-Review

**Spec coverage** (audit item → task):
- B1 → Task 2 (+ snapshot Task 1). M2 → Task 4. M3 → Task 5. M4 → Task 6. M1 → Task 14.
- m1 → Task 7. m2 → Task 3. m3 → Task 11. m4 → Task 12. m5 → Task 7. m6 → Task 13. m7 → Task 8. m8 → Task 9. m9 → Task 12. m10 → Task 10. m11 → Task 14. m12 → Task 14. m13 → deferred (explicit).
- All 18 audit items placed; m13 explicitly deferred with rationale.

**Conflict handling:** C1 (RC algebra) → Task 5 pins value first. C2 (label/width/regionprops parity) → Tasks 11–12 freeze golden fixtures before rewrite. C3 (compat row bridge) → Task 1 snapshot gates Task 2. C4 (distributed/GPU) → excluded from milestone.

**Ordering:** Track A first (B1 unblocks scale; Task 1 snapshot precedes every mutation and re-runs after each). Track B only after Track A stabilizes the hot path. Track C last (docs). Parity tests precede every numeric-path rewrite per Global Constraints.
