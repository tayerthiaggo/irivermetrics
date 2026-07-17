# Post-M11 Cleanup and Follow-Ups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the five gaps left after Milestone 11 (connectivity module): stale checklist state, the still-open M11 checkboxes, the pending Q9 config-hash golden-hash decision, wiring `connectivity.py` into `analyze()`, and the V6 RC/DCI validation benchmark.

**Architecture:** Five independent workstreams, ordered cheapest/lowest-risk first. Tasks 1-2 are mechanical doc sync (no code, no tests beyond existing suite staying green). Task 3 closes a decision-gate row against an existing test. Task 4 is new pipeline-integration code requiring its own TDD cycle and a new decisions.md row (the wet-intersection method is itself a modelling choice, same pattern as U8). Task 5 is external validation requiring a new dependency (`riverconn`) and a hand/reference DCI computation on the real Fitzroy drainage graph.

**Tech Stack:** Python 3.14, pytest, geopandas/shapely (existing), R + `riverconn` R package (Baldan et al. 2022) via subprocess or rpy2 (new, Task 5 only -- `riverconn` is an R package, not Python; confirmed 2026-07-17, no PyPI distribution under that or similar names).

## Global Constraints

- No production code changes without a failing test first (TDD, per `superpowers:test-driven-development` — already governs this repo's CLAUDE.md-equivalent workflow).
- Every claim of "done"/"passing" requires a fresh full-suite run (`python -m pytest`) pasted into the task, per `superpowers:verification-before-completion`.
- Every new modelling choice (e.g. wet-intersection method for Task 4) gets its own `docs/audit/decisions.md` row with owner/evidence/approval date before implementation, same pattern as U8 (see `docs/audit/decisions.md` U8 row added 2026-07-17).
- Connectivity remains an optional profile: Task 4 must not change results for any run that does not request `metric_profiles: ["connectivity"]`.
- DCI stays citation-only until V6 (Task 5) passes; Task 5 failing or being inconclusive must NOT flip DCI to a runtime metric.
- Never edit `hydrofragments_out/run_manifest.json` by hand — it is a generated artifact that changes when tests run; leave it alone in every task's diff review.

---

### Task 1: Sync `execution_checklist.md` checkbox state to real repo state

**Files:**
- Modify: `docs/audit/execution_checklist.md`

**Interfaces:**
- Consumes: nothing (pure doc edit).
- Produces: nothing consumed by later tasks — purely a trust/accuracy fix.

**Context:** The prior-session review (see SessionStart hook context from 2026-07-17) found that Section 3 top-level task boxes for M0–M8 are almost all `[ ]` despite the corresponding files existing on disk. Example: `docs/audit/execution_checklist.md:134` reads `* [ ] Implement hydrofragments/config.py, schema.py, models.py, metrics/registry.py.` — all four files exist and are covered by passing tests.

This task is mechanical: for every `[ ]` task/test-first checkbox under Milestones 0–8, verify the referenced file(s) exist and the referenced behavior has a passing test, then flip to `[x]`. Leave `[ ]` only for items with no corresponding file/test.

- [ ] **Step 1: Enumerate every unchecked box under Milestones 0-8**

Run:
```bash
grep -n "\[ \]" docs/audit/execution_checklist.md
```

Read the output and, for each line between the `### Milestone 0` header (line 67) and the `### Milestone 9` header (line 288), note the referenced file path(s).

- [ ] **Step 2: Verify each referenced file exists**

For each unique file path mentioned in an unchecked box (e.g. `hydrofragments/config.py`, `hydrofragments/schema.py`, `hydrofragments/models.py`, `hydrofragments/metrics/registry.py`, `hydrofragments/io/adapters.py`, `hydrofragments/io/validity.py`, `hydrofragments/io/alignment.py`, `hydrofragments/spatial/crs.py`, `hydrofragments/spatial/context.py`, `hydrofragments/temporal/cadence.py`, `hydrofragments/compute/policy.py`, `hydrofragments/compute/chunks.py`, `hydrofragments/temporal/composites.py`, `hydrofragments/pipeline.py`, `hydrofragments/metrics/persistence.py`, `hydrofragments/metrics/extent.py`, `hydrofragments/guards/scientific.py`, `hydrofragments/output/rasters.py`, `hydrofragments/patches/labels.py`, `hydrofragments/patches/components.py`, `hydrofragments/patches/morphology.py`, `hydrofragments/metrics/patches.py`, `hydrofragments/output/tables.py`, `hydrofragments/output/manifest.py`, `hydrofragments/guards/comparison.py`), run:

```bash
python - <<'PY'
from pathlib import Path
paths = [
    "hydrofragments/config.py", "hydrofragments/schema.py", "hydrofragments/models.py",
    "hydrofragments/metrics/registry.py", "hydrofragments/io/adapters.py",
    "hydrofragments/io/validity.py", "hydrofragments/io/alignment.py",
    "hydrofragments/spatial/crs.py", "hydrofragments/spatial/context.py",
    "hydrofragments/temporal/cadence.py", "hydrofragments/compute/policy.py",
    "hydrofragments/compute/chunks.py", "hydrofragments/temporal/composites.py",
    "hydrofragments/pipeline.py", "hydrofragments/metrics/persistence.py",
    "hydrofragments/metrics/extent.py", "hydrofragments/guards/scientific.py",
    "hydrofragments/output/rasters.py", "hydrofragments/patches/labels.py",
    "hydrofragments/patches/components.py", "hydrofragments/patches/morphology.py",
    "hydrofragments/metrics/patches.py", "hydrofragments/output/tables.py",
    "hydrofragments/output/manifest.py", "hydrofragments/guards/comparison.py",
]
for p in paths:
    print(("OK  " if Path(p).exists() else "MISS"), p)
PY
```

Expected: every path prints `OK`. If any prints `MISS`, leave that task's checkbox as `[ ]` and note it — it is a genuine gap, not a doc-sync fix.

- [ ] **Step 3: Verify test coverage claims for test-first checkboxes**

For each unchecked test-first item (e.g. "Golden tests for cross-platform stability of `config_hash`" at line 138), grep for a matching test name:

```bash
grep -rn "def test_" tests/contracts/ tests/metrics/ tests/io/ tests/spatial/ tests/temporal/ tests/patches/ tests/output/ tests/guards/ tests/compute/ | grep -i "golden\|hash\|denominator\|occurrence\|sentinel\|crs\|chunk\|checkpoint\|patch\|label\|connectivity\|comparison"
```

Cross-reference each result against the checklist line it corresponds to. If a passing test exists, flip that box to `[x]`.

- [ ] **Step 4: Edit the checklist file**

For each verified item, change `* [ ]` to `* [x]` in `docs/audit/execution_checklist.md` using targeted Edit calls (one per checkbox line, to avoid accidental over-matching — the file has ~150 `[ ]` occurrences and a blind replace-all would be wrong). Do not touch any box you could not verify in Steps 2-3.

- [ ] **Step 5: Run the full suite to confirm this was a doc-only change**

Run: `python -m pytest`
Expected: same pass/skip/fail counts as before this task started (this task must not change test behavior — it only edits a markdown file).

- [ ] **Step 6: Commit**

```bash
git add docs/audit/execution_checklist.md
git commit -m "docs: sync execution_checklist.md M0-M8 checkboxes to actual repo state"
```

---

### Task 2: Tick the M11 checkboxes for work already done

**Files:**
- Modify: `docs/audit/execution_checklist.md:354-362`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing.

**Context:** `docs/audit/execution_checklist.md:354-362` (Milestone 11 section) still shows:
```
* **Tasks**:
  * [ ] Implement `hydrofragments/metrics/connectivity.py`.
  * [ ] Implement `tests/connectivity/test_rc.py`, `test_tcf.py`, `test_dci_reference.py`.
* **Test-First Requirements**:
  * [ ] Verify stable node sources and edge rules across temporal sequences.
  * [ ] Validate RC edge fractions and reachability on simple linear graphs.
  * [ ] Verify DCI parity against `riverconn`/Conefor references if DCI is approved.
```

As of 2026-07-17, `hydrofragments/metrics/connectivity.py` exists with `build_fixed_graph`, `compute_realised_connectivity`, `compute_tcf`, backed by `tests/connectivity/test_fixed_graph.py`, `test_rc.py`, `test_tcf.py`, `test_reconnection_preference.py` (31 tests, all passing). `test_dci_reference.py` does NOT exist and must stay that way until Task 5 (V6) passes — DCI is citation-only (Q4).

- [ ] **Step 1: Verify the connectivity test files and their pass state**

Run: `python -m pytest tests/connectivity/ -v`
Expected: all tests in `test_fixed_graph.py`, `test_rc.py`, `test_tcf.py`, `test_reconnection_preference.py` PASS (31 total as of 2026-07-17; re-count if this task runs later and more were added).

- [ ] **Step 2: Edit the checklist**

In `docs/audit/execution_checklist.md`:
- Line 355: `* [ ] Implement hydrofragments/metrics/connectivity.py.` → `* [x] Implement hydrofragments/metrics/connectivity.py.`
- Line 356: `* [ ] Implement tests/connectivity/test_rc.py, test_tcf.py, test_dci_reference.py.` → `* [x] Implement tests/connectivity/test_rc.py, test_tcf.py, test_fixed_graph.py, test_reconnection_preference.py. test_dci_reference.py intentionally not created -- DCI stays citation-only per Q4 until V6 passes (see Task 5 of docs/superpowers/plans/2026-07-17-post-m11-cleanup.md).`
- Line 358: `* [ ] Verify stable node sources and edge rules across temporal sequences.` → `* [x] Verify stable node sources and edge rules across temporal sequences.`
- Line 359: `* [ ] Validate RC edge fractions and reachability on simple linear graphs.` → `* [x] Validate RC edge fractions and reachability on simple linear graphs.`
- Line 360: leave `* [ ] Verify DCI parity against riverconn/Conefor references if DCI is approved.` UNCHECKED — this is Task 5, not yet done.

Also update the note under the Milestone 11 header itself, since it is currently marked `(Gated)`: add a line directly below the `### Milestone 11: Connectivity Tranche (Gated)` header:
```
**Status (2026-07-17):** Gate closed. Node source and edge rule fixed via decisions.md U8 (external_network drainage reaches, configurable dry-gap threshold default 0). RC/TCF implemented and tested. Pipeline wiring into `analyze()` and V6 DCI benchmark remain open -- see `docs/superpowers/plans/2026-07-17-post-m11-cleanup.md` Tasks 4-5.
```

- [ ] **Step 3: Run the full suite to confirm this was doc-only**

Run: `python -m pytest`
Expected: identical counts to before this task.

- [ ] **Step 4: Commit**

```bash
git add docs/audit/execution_checklist.md
git commit -m "docs: tick M11 checklist boxes for completed connectivity.py work"
```

---

### Task 3: Close Q9 (config_hash golden hash) or explicitly downgrade it

**Files:**
- Modify: `docs/audit/decisions.md` (Q9 row, currently lines 173-184)
- Read only: `tests/contracts/test_hashing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: nothing consumed by later tasks.

**Context:** `docs/audit/decisions.md` Q9 row says `Status: pending golden cross-platform tests`. But `tests/contracts/test_hashing.py` already contains `test_minimal_scientific_config_has_stable_golden_hash`, which asserts a literal SHA-256 string (`GOLDEN_MINIMAL_CONFIG_HASH`) computed on this machine/OS. That test passing locally is NOT the same claim as "verified stable across platforms" — the row's blocking condition is specifically cross-platform stability, which requires running the same test on a second OS (or in CI on a different runner OS) and confirming the hash matches.

This task does not require actually provisioning a second OS. It closes the decision honestly based on what evidence actually exists: either (a) if CI already runs this test cross-platform, point to that and approve; or (b) if not, downgrade the row to make the real status explicit rather than leaving a stale "pending" that nobody re-checks.

- [ ] **Step 1: Check whether CI runs the test suite on more than one OS**

```bash
find .github/workflows -type f 2>/dev/null
```

If workflow files exist, read them and check the `runs-on` / `matrix.os` keys.

- [ ] **Step 2a: If CI already covers 2+ OSes and the hashing test is in the default test run**

Update `docs/audit/decisions.md` Q9 row:
- `Status` → `approved`
- `Evidence artifact` → append: `; cross-platform coverage confirmed via .github/workflows/<file>.yml (runs on <os list>); tests/contracts/test_hashing.py::test_minimal_scientific_config_has_stable_golden_hash and test_execution_and_human_fields_do_not_change_config_hash run in that matrix`
- `Approval date` → today's date, with a note: "closed based on existing CI matrix coverage, not a new manual run"

- [ ] **Step 2b: If CI does not cover 2+ OSes (the likely case)**

Update `docs/audit/decisions.md` Q9 row `Status` field to:
```
`pending` -- golden hash test exists and passes locally (tests/contracts/test_hashing.py::test_minimal_scientific_config_has_stable_golden_hash), but has only been run on one OS/machine. Not CI-enforced across platforms. Downgraded from silent-pending to explicitly non-blocking for M2/M7 core work per this row's own "Affected milestones" -- reproducibility claims involving cross-OS config_hash stability must not be made until this is closed for real.
```
Do not flip to `approved`. Leave `Approval date` as `—`.

- [ ] **Step 3: Run the hashing tests fresh to confirm they still pass on this machine**

Run: `python -m pytest tests/contracts/test_hashing.py -v`
Expected: 4 passed (as of 2026-07-17; re-count if the file has grown).

- [ ] **Step 4: Commit**

```bash
git add docs/audit/decisions.md
git commit -m "docs: close or explicitly downgrade Q9 config_hash cross-platform decision"
```

---

### Task 4: Wire `connectivity.py` into `analyze()`

**Files:**
- Modify: `hydrofragments/api.py` (`validate_inputs` at line 104, `analyze` at line 530)
- Modify: `docs/audit/decisions.md` (new row, e.g. U9, for the wet-intersection method)
- Create: `hydrofragments/spatial/connectivity_context.py`
- Test: `tests/connectivity/test_wet_intersection.py`
- Test: `tests/connectivity/test_pipeline_wiring.py`

**Interfaces:**
- Consumes: `hydrofragments.metrics.connectivity.build_fixed_graph(topology, *, wet_any_month)`, `compute_realised_connectivity(graph, *, wet_gap_by_edge, gap_threshold=0)`, `compute_tcf(graph, *, monthly_active, monthly_valid)` (all from Task 4's predecessor, M11, already implemented in `hydrofragments/metrics/connectivity.py`). Consumes `hydrofragments.spatial.context.SpatialContext` (`.drainage`, `.has_real_channel`) and `hydrofragments.schema.MetricDependency.FIXED_NODES` / `.GRAPH` (already defined in `hydrofragments/schema.py:72,74`).
- Produces: `hydrofragments.spatial.connectivity_context.reach_wet_any_month(drainage, water, *, buffer_m) -> dict[str, bool]` and `reach_gap_by_edge_for_month(drainage, water_month, edges, *, buffer_m) -> dict[tuple[str,str], int|float|None]` for later tasks/callers. `validate_inputs` gains `available.add(MetricDependency.FIXED_NODES)` / `.add(MetricDependency.GRAPH)` when connectivity is available.

**Context:** This is the largest task in this plan and the one place where the plan cannot fully pre-decide the science. `wet_any_month` and the per-month `wet_gap_by_edge` require deciding HOW a raster water mask intersects a vector reach line — e.g. "buffer the reach line by N pixels/meters and check any wet pixel falls inside the buffer" — which is a new modelling choice, not something `decisions.md` U8 covers (U8 only fixed node source and edge activation threshold *given* per-edge gap values; it did not fix how those gap values get computed from raster+vector). This must get its own decisions.md row before being implemented, matching the U8 pattern from M11.

- [ ] **Step 1: Draft and get approval for the wet-intersection method decision**

Before writing any code, propose a concrete method and ask for approval (this step is a conversation with the user, not an automated step). Proposed default to offer: buffer each clipped drainage reach line by `spatial.pixel_size_m` (or a configurable `connectivity.reach_buffer_m`, default = 1 pixel width) using `shapely`/`geopandas.buffer`, then for a given month's water mask, a reach is "wet" if `xr.DataArray.rio.clip([buffer_polygon])` has `.sum() > 0`. Get an explicit approval and record it as a new `docs/audit/decisions.md` row (e.g. `U9 -- reach/mask wet-intersection method`) with the same fields as the U8 row (Decision, Status, Evidence artifact, Owner, Approval date, Consequence if wrong, Affected milestones: M11).

Do not proceed to Step 2 until this row exists with `Status: approved`.

- [ ] **Step 2: Write the failing test for `reach_wet_any_month`**

Create `tests/connectivity/test_wet_intersection.py`:

```python
"""Milestone 11 pipeline wiring -- reach/mask wet-intersection (U9).

Determines, per drainage reach, whether it was wet in at least one month of
the series -- the `wet_any_month` input to
`hydrofragments.metrics.connectivity.build_fixed_graph`.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import LineString

from hydrofragments.spatial.connectivity_context import reach_wet_any_month


def _drainage_two_reaches() -> "gpd.GeoDataFrame":
    return gpd.GeoDataFrame(
        {
            "HydroID": ["A", "B"],
            "From_Node": [1, 2],
            "To_Node": [2, 3],
            "NextDownID": ["B", "-1"],
            "geometry": [
                LineString([(0, 5), (10, 5)]),   # reach A: y=5, x in [0,10]
                LineString([(10, 5), (20, 5)]),  # reach B: y=5, x in [10,20]
            ],
        },
        crs="EPSG:3577",
    )


def _water_cube_wet_along_reach_a_only() -> "xr.DataArray":
    # 20x10 grid, 1 unit/pixel, y=[0..9], x=[0..19]; two months.
    data = np.zeros((2, 10, 20), dtype=bool)
    data[:, 5, 0:10] = True  # wet along reach A's row for both months
    return xr.DataArray(
        data,
        dims=("time", "y", "x"),
        coords={"time": [0, 1], "y": np.arange(10), "x": np.arange(20)},
    )


def test_reach_intersecting_wet_pixels_is_flagged_wet():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=1.0)

    assert result["A"] is True


def test_reach_never_intersecting_wet_pixels_is_flagged_dry():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=1.0)

    assert result["B"] is False


def test_result_covers_every_reach_in_drainage():
    drainage = _drainage_two_reaches()
    water = _water_cube_wet_along_reach_a_only()

    result = reach_wet_any_month(drainage, water, buffer_m=1.0)

    assert set(result.keys()) == {"A", "B"}
```

- [ ] **Step 3: Run to verify RED**

Run: `python -m pytest tests/connectivity/test_wet_intersection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'hydrofragments.spatial.connectivity_context'`.

- [ ] **Step 4: Implement minimal `reach_wet_any_month`**

Create `hydrofragments/spatial/connectivity_context.py`:

```python
"""Reach/water-mask intersection for the RC/TCF fixed graph (U9).

Decides, per drainage reach, whether the reach was ever wet across a
monthly series -- the ``wet_any_month`` input consumed by
:func:`hydrofragments.metrics.connectivity.build_fixed_graph`. Method
(U9, approved <date>): buffer each reach line by ``buffer_m`` and flag it
wet if any wet pixel's cell falls inside that buffer in any month.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import geopandas as gpd
    import xarray as xr


def reach_wet_any_month(
    drainage: "gpd.GeoDataFrame",
    water: "xr.DataArray",
    *,
    buffer_m: float,
) -> dict[str, bool]:
    """Return, per reach ``HydroID``, whether it intersects wet pixels in >=1 month."""
    result: dict[str, bool] = {}
    for _, reach in drainage.iterrows():
        buffer_polygon = reach.geometry.buffer(buffer_m)
        clipped = water.rio.write_crs(drainage.crs, inplace=False).rio.clip(
            [buffer_polygon], drop=False, invert=False
        )
        result[str(reach["HydroID"])] = bool(clipped.sum() > 0)
    return result


__all__ = ["reach_wet_any_month"]
```

Note: this is a starting implementation, not a final one -- if `rio.clip` requires a spatial-ref-aware DataArray that the test fixture does not have, adapt using `rasterio.features.geometry_mask` directly against `water.y`/`water.x` coordinates instead. Whichever approach is used, keep the public signature identical to what Step 2's test expects.

- [ ] **Step 5: Run to verify GREEN**

Run: `python -m pytest tests/connectivity/test_wet_intersection.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add hydrofragments/spatial/connectivity_context.py tests/connectivity/test_wet_intersection.py docs/audit/decisions.md
git commit -m "feat: add reach/water-mask wet-intersection for connectivity graph (U9)"
```

- [ ] **Step 7: Write the failing test for `validate_inputs` wiring**

Add to `tests/connectivity/test_pipeline_wiring.py` (new file):

```python
"""Milestone 11 pipeline wiring -- FIXED_NODES/GRAPH availability in validate_inputs."""
from __future__ import annotations

import numpy as np
import xarray as xr

from hydrofragments.api import validate_inputs
from hydrofragments.config import HydroConfig
from hydrofragments.models import WaterCube
from hydrofragments.schema import MetricDependency
from hydrofragments.spatial.context import SpatialContext


def _minimal_config() -> HydroConfig:
    return HydroConfig.from_mapping(
        {
            "config_schema_version": "1.2.0",
            "input": {"kind": "generic_binary"},
            "temporal": {
                "input_cadence": "monthly",
                "monthly_composite": "supplied",
                "composite_owner": "caller",
            },
            "metric_profiles": ["connectivity"],
        }
    )


def _cube() -> WaterCube:
    water = xr.DataArray(
        np.zeros((1, 4, 4), dtype=bool),
        dims=("time", "y", "x"),
    )
    valid = xr.ones_like(water, dtype=bool)
    return WaterCube(water=water, valid_obs=valid, source="test", cadence="monthly", crs=None, provenance=())


def test_fixed_nodes_and_graph_unavailable_without_drainage():
    report = validate_inputs(_cube(), "aoi-1", config=_minimal_config())

    assert any("realised_connectivity" in reason or "tcf" in reason for _, reason in
               [(s.metric_id, s.reason) for s in
                __import__("hydrofragments.metrics.registry", fromlist=["resolve_metrics"])
                .resolve_metrics(["connectivity"], available_dependencies={MetricDependency.VALIDITY}).skipped])


def test_fixed_nodes_and_graph_available_with_real_channel_and_wet_reaches():
    context = SpatialContext(
        aoi_id="aoi-1",
        area_m2=100.0,
        drainage_id="drainage-1",
        l_ref_m=10.0,
        crs="EPSG:3577",
        proxy_channel=False,
    )
    # has_real_channel requires drainage is not None too -- construct with a
    # minimal non-None placeholder consistent with SpatialContext's contract
    # (see hydrofragments/spatial/context.py:44-55); adjust this fixture to
    # match whatever real construction validate_inputs expects once Step 8's
    # implementation reveals the exact call shape.
    report = validate_inputs(
        _cube(), "aoi-1", config=_minimal_config(), drainage=context
    )

    assert report is not None  # placeholder assertion, tighten once available-dependency wiring lands
```

(This test will need small adjustments once Step 8 is implemented and the exact `validate_inputs` connectivity-availability signature is known -- e.g. it may need a `connectivity_available: bool = False` explicit parameter rather than inferring purely from `drainage.has_real_channel`, since `has_real_channel` alone does not guarantee any reach is wet. Resolve this ambiguity by making `validate_inputs` accept an explicit `wet_any_month: Mapping[str, bool] | None = None` parameter -- callers who already ran `reach_wet_any_month` pass it in; `FIXED_NODES`/`GRAPH` become available only if `wet_any_month` is supplied AND at least one value is `True`.)

- [ ] **Step 8: Run to verify RED, then implement minimal wiring in `validate_inputs`**

Run: `python -m pytest tests/connectivity/test_pipeline_wiring.py -v` — expect failure/error reflecting the missing parameter.

Modify `hydrofragments/api.py` `validate_inputs` (starting at line 104):

```python
def validate_inputs(
    cube: WaterCube,
    aoi_id: str,
    *,
    config: HydroConfig,
    drainage: Any | None = None,
    hydroyear_available: bool = False,
    dual_composites_available: bool = False,
    wet_any_month: Mapping[str, bool] | None = None,
) -> ValidationReport:
    """Validate contracts without computing metrics."""
    errors: list[str] = []
    warnings: list[str] = []
    if cube.water.sizes != cube.valid_obs.sizes:
        errors.append("water and valid_obs must share dimensions")
    if cube.water.dims != cube.valid_obs.dims:
        errors.append("water and valid_obs must share dimension order")
    try:
        validate_alignment(cube.water, cube.valid_obs)
    except ValueError as error:
        errors.append(str(error))

    available = {MetricDependency.VALIDITY}
    if config.patches.min_patch_pixels > 0:
        available.add(MetricDependency.PATCHES)
    if isinstance(drainage, SpatialContext) and drainage.has_real_channel:
        available.add(MetricDependency.CHANNEL)
    if config.patches.width_resolution_floor_pixels is not None:
        available.add(MetricDependency.WIDTH_FLOOR)
    if hydroyear_available:
        available.add(MetricDependency.HY_ANCHOR)
    if dual_composites_available:
        available.add(MetricDependency.DUAL_COMPOSITE)
    if (
        isinstance(drainage, SpatialContext)
        and drainage.has_real_channel
        and wet_any_month is not None
        and any(wet_any_month.values())
    ):
        available.add(MetricDependency.FIXED_NODES)
        available.add(MetricDependency.GRAPH)
    plan = resolve_metrics(config.metric_profiles, available_dependencies=available)
    skipped = tuple((item.metric_id, item.reason) for item in plan.skipped)
    if plan.skipped:
        warnings.append("some requested metrics are unavailable with current inputs")
```

(Only the new `wet_any_month` parameter and the `FIXED_NODES`/`GRAPH` conditional block are additions; everything else in the function body after line 140 is unchanged and must be preserved exactly as currently written -- re-read `hydrofragments/api.py:104-149` in full before editing, since this plan only shows the modified prefix.)

Then simplify `tests/connectivity/test_pipeline_wiring.py`'s second test to assert directly on `available` behavior via `resolve_metrics`, once the real `SpatialContext` construction requirements are confirmed by running it.

- [ ] **Step 9: Run to verify GREEN**

Run: `python -m pytest tests/connectivity/test_pipeline_wiring.py -v`
Expected: both tests pass. Iterate on the `SpatialContext` fixture construction in Step 7/8 until green -- do not weaken the assertions to make it pass artificially.

- [ ] **Step 10: Run the full suite**

Run: `python -m pytest`
Expected: all previously-passing tests still pass; new tests pass; no regressions in `tests/test_integration.py` or other `api.py` consumers.

- [ ] **Step 11: Commit**

```bash
git add hydrofragments/api.py tests/connectivity/test_pipeline_wiring.py
git commit -m "feat: wire FIXED_NODES/GRAPH connectivity dependencies into validate_inputs"
```

**Note on scope:** This task wires *availability* (`validate_inputs`) only. Actually invoking `build_fixed_graph`/`compute_realised_connectivity`/`compute_tcf` inside `analyze()` to produce `MetricRecord` rows (mirroring how `_channel_profile_records` at `hydrofragments/api.py:202` produces LPSEC records) is enough additional surface area — new record-building function, manifest fields for `node_source`/`edge_rule`, month-by-month iteration — that it should be its own follow-up task once this wiring lands and is reviewed. Flag that explicitly when this task is reported done; do not silently expand scope to cover it.

---

### Task 5: V6 -- RC/TCF vs DCI reference validation

**Files:**
- Create: `validation/run_dci_benchmark.py`
- Create: `tests/connectivity/test_dci_reference.py`
- Modify: `docs/validation_status.md` (V6 row)
- Modify: `pyproject.toml` (add `riverconn` as an optional/dev dependency)

**Interfaces:**
- Consumes: `hydrofragments.metrics.connectivity.build_fixed_graph`, `compute_realised_connectivity` (from M11); real drainage at `data/fitzroy_kimberley_drainage.gpkg` (U4/Q6); real water cube at `data/wofs_monthly_masks_1986_2026.zarr`.
- Produces: a validation report row (V6) with either "demonstrated" (parity within an agreed tolerance) or "not yet passed" status, per the `docs/validation_status.md` convention established in M13.

**Context:** `docs/audit/scientific_metrics_audit.md:254` and `implementation_plan.md` both specify this benchmark: `RC_pair` (reach-length-weighted) against a directly computed DCI (`riverconn`/Conefor) on the real reach network. This is what would eventually let DCI move from citation-only to a runtime metric (Q4) -- but this task's job is only to run the benchmark and report agreement/disagreement, not to implement DCI as a shipped metric regardless of outcome (per Global Constraints).

**Update (2026-07-17, controller pre-dispatch check):** `riverconn` is an **R package** (Baldan et al. 2022), not a Python package -- confirmed via `pip install riverconn` (no matching distribution) and `pip install river-connectivity` / `pyriverconn` (neither exists either). The maintainer chose to proceed via **R `riverconn` invoked from this task**, either as an `Rscript` subprocess call or via `rpy2`, rather than switching to Conefor or reporting V6 blocked. This requires R itself to be installed in this environment -- that has NOT yet been confirmed (the controller's shell access was intermittently unavailable when this check was attempted). This is now Step 1.

- [ ] **Step 1: Confirm R and the `riverconn` R package are available, or install them**

```bash
Rscript --version
```

If R is not installed, this environment does not have an R runtime and you must install one (or, if you cannot install system-level software from this environment, report BLOCKED with exactly that finding -- do not silently fall back to Conefor or a fabricated DCI calculation, since the maintainer explicitly chose the R-riverconn path over those alternatives).

If R is available, install the `riverconn` R package and confirm it loads:

```bash
Rscript -e 'if (!requireNamespace("riverconn", quietly = TRUE)) install.packages("riverconn", repos = "https://cloud.r-project.org"); library(riverconn); packageVersion("riverconn")'
```

Choose ONE integration approach and use it consistently for the rest of this task:
- **Subprocess (recommended, simpler):** write a small `.R` script that reads a CSV/GeoJSON export of the fixed graph (nodes, edges, reach lengths, active-edge flags for the month being benchmarked), calls `riverconn`'s DCI function, and writes the result to a CSV. Invoke it from Python via `subprocess.run(["Rscript", "path/to/script.R", ...])` and read the CSV result back.
- **rpy2 (tighter integration, more setup):** use the `rpy2` Python package to call `riverconn` functions in-process. Only choose this if `rpy2` installs cleanly in this environment; if it fights with R/Python ABI compatibility, fall back to the subprocess approach instead of spending excessive effort on rpy2 setup.

If, after actually attempting installation, R or `riverconn` genuinely cannot be made to work in this environment (not just "looks like effort," but a real blocker -- e.g. no internet access for CRAN, no permission to install R at all), report BLOCKED with the specific error, rather than silently substituting Conefor or fabricating a DCI calculation. The maintainer will decide how to proceed from there.

- [ ] **Step 2: Write the failing test for reach-length-weighted RC_pair**

The existing `compute_realised_connectivity` in `hydrofragments/metrics/connectivity.py` computes unweighted `rc_pair_pct` (equal-weight reachable pairs). DCI needs reach-length weighting per `docs/HydroFragments_v1.2_spec.md:452`: `DCI_t = 100 * sum(len_i * len_j * c_ij,t) / (sum(len_i))^2`. Add this as a new function rather than modifying the existing one (the unweighted form is still needed and already tested).

Create `tests/connectivity/test_dci_reference.py`:

```python
"""Milestone 11 -- V6 benchmark: reach-length-weighted RC_pair vs riverconn DCI.

Citation-only per Q4 until this benchmark passes; passing does NOT
auto-enable DCI as a shipped runtime metric -- that is a separate decision.
"""
from __future__ import annotations

import pytest

from hydrofragments.metrics.connectivity import FixedGraph, compute_length_weighted_rc_pair


def test_length_weighted_rc_pair_matches_hand_computed_dci_on_linear_graph():
    # Two reaches, lengths 10 and 30, connected -- fully connected DCI on a
    # linear graph with all fragments merged reduces to 100% by definition
    # (Cote et al. 2009): every unit of length can reach every other unit.
    graph = FixedGraph(node_source="external_network", nodes=("A", "B"), edges=(("A", "B"),))
    result = compute_length_weighted_rc_pair(
        graph, wet_gap_by_edge={("A", "B"): 0}, gap_threshold=0,
        length_by_node={"A": 10.0, "B": 30.0},
    )
    assert result == pytest.approx(100.0)


def test_length_weighted_rc_pair_disconnected_reflects_fragment_size_squared():
    # Cote et al. 2009 DCI formula on disconnected fragments of length 10
    # and 30 out of 40 total: DCI = 100 * (10^2 + 30^2) / 40^2 = 62.5
    graph = FixedGraph(node_source="external_network", nodes=("A", "B"), edges=(("A", "B"),))
    result = compute_length_weighted_rc_pair(
        graph, wet_gap_by_edge={("A", "B"): None}, gap_threshold=0,
        length_by_node={"A": 10.0, "B": 30.0},
    )
    assert result == pytest.approx(62.5)
```

- [ ] **Step 3: Run to verify RED**

Run: `python -m pytest tests/connectivity/test_dci_reference.py -v`
Expected: FAIL, `ImportError: cannot import name 'compute_length_weighted_rc_pair'`.

- [ ] **Step 4: Implement `compute_length_weighted_rc_pair`**

Add to `hydrofragments/metrics/connectivity.py` (after `compute_realised_connectivity`):

```python
def compute_length_weighted_rc_pair(
    graph: FixedGraph,
    *,
    wet_gap_by_edge: Mapping[tuple[str, str], "int | float | None"],
    gap_threshold: "int | float" = 0,
    length_by_node: Mapping[str, float],
) -> float:
    """Reach-length-weighted RC_pair -- the DCI form (Cote et al. 2009, spec 6.17).

    ``DCI_t = 100 * sum_{i<j}(len_i * len_j * c_ij,t) / (sum(len_i))^2`` where
    ``c_ij,t = 1`` if fragments i,j are connected under the active-edge
    subgraph, else 0. Positioned as citation-only validation support (Q4) --
    this function existing does not make DCI a shipped runtime metric.
    """
    active_edges = [
        edge
        for edge in graph.edges
        if (gap := wet_gap_by_edge.get(edge)) is not None
        and gap <= gap_threshold
    ]
    parent = {node: node for node in graph.nodes}
    for node_a, node_b in active_edges:
        _union(parent, node_a, node_b)

    total_length = sum(length_by_node[node] for node in graph.nodes)
    if total_length == 0:
        return float("nan")

    numerator = 0.0
    nodes = graph.nodes
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if _find(parent, nodes[i]) == _find(parent, nodes[j]):
                numerator += length_by_node[nodes[i]] * length_by_node[nodes[j]]
    # Diagonal terms (i==j) contribute len_i^2 to the standard DCI form.
    for node in nodes:
        numerator += length_by_node[node] ** 2

    return 100.0 * numerator / (total_length ** 2)
```

Add `compute_length_weighted_rc_pair` to the module's `__all__` list.

- [ ] **Step 5: Run to verify GREEN**

Run: `python -m pytest tests/connectivity/test_dci_reference.py -v`
Expected: 2 passed. If the diagonal-term handling doesn't match Cote et al. 2009's exact convention, adjust based on a hand-recomputation against the two known test cases before moving on -- do not adjust the test expectations to match a possibly-wrong implementation.

- [ ] **Step 6: Run the real benchmark against `riverconn`**

Create `validation/run_dci_benchmark.py` (a script, not a test -- run once and its output feeds Step 7):

```python
"""V6 benchmark: length-weighted RC_pair vs riverconn DCI on the Fitzroy reach network.

Run manually: python validation/run_dci_benchmark.py
Writes validation/results/v6_dci_benchmark.csv
"""
from __future__ import annotations

import geopandas as gpd
import pandas as pd

from hydrofragments.metrics.connectivity import build_fixed_graph, compute_length_weighted_rc_pair
from hydrofragments.spatial.context import create_channel_context

# Fill in with the actual approved AOI/drainage load calls used elsewhere in
# validation/run_fitzroy_validation.py (M13) -- reuse that script's loading
# pattern rather than re-deriving it here.
```

This script's exact content depends on how `validation/run_fitzroy_validation.py` (built in M13) loads the AOI and drainage -- read that file first and mirror its loading calls before filling in the benchmark logic. Compute `compute_length_weighted_rc_pair` on the real Fitzroy graph for a representative month, then compute DCI on the same graph/edges using `riverconn`'s API (read `riverconn`'s docs via the `context7` MCP tool for current API syntax before writing this part, per this project's global instruction to always check current docs for library usage). Write both numbers plus their percent agreement to `validation/results/v6_dci_benchmark.csv`.

- [ ] **Step 7: Update `docs/validation_status.md` V6 row**

Based on the actual agreement number from Step 6, update the V6 row to either "demonstrated" (with the CSV path and agreement percentage as evidence) or "not yet passed / blocked" (with the specific numeric disagreement and, if `riverconn` was unusable in Step 1, a note that Conefor was not attempted either and why).

- [ ] **Step 8: Run the full suite**

Run: `python -m pytest`
Expected: no regressions; new `test_dci_reference.py` tests pass.

- [ ] **Step 9: Commit**

```bash
git add hydrofragments/metrics/connectivity.py tests/connectivity/test_dci_reference.py validation/run_dci_benchmark.py validation/results/v6_dci_benchmark.csv docs/validation_status.md pyproject.toml
git commit -m "feat: add length-weighted RC_pair (DCI form) and V6 riverconn benchmark"
```

---

## Self-Review Notes

- **Spec coverage:** Task 1-2 address checklist accuracy (no spec section, pure process hygiene). Task 3 addresses Q9 (decisions.md, spec §"Hashing rules"). Task 4 addresses spec §6.13/6.11 pipeline reachability (RC/TCF must actually run, not just exist as pure functions) plus a new U9 decision gate for the wet-intersection method, mirroring U8's gate pattern. Task 5 addresses V6 (spec §6.18 validation matrix) and the DCI form from spec §6.17.
- **Placeholder scan:** Task 4 Step 7-8's test fixture for `SpatialContext` construction is intentionally left partially open ("adjust this fixture... once Step 8's implementation reveals the exact call shape") because `validate_inputs`'s exact connectivity-availability signature is a real design decision being made inside this task, not something resolvable before writing code — flagged explicitly rather than hidden. Task 5 Step 6's benchmark script is intentionally a skeleton pointing at `validation/run_fitzroy_validation.py` as the pattern to mirror, since duplicating that script's ~100 lines of AOI/drainage loading here would drift out of sync with it.
- **Type consistency:** `build_fixed_graph`, `compute_realised_connectivity`, `compute_tcf`, `FixedGraph` signatures used in Tasks 4-5 match their actual definitions in `hydrofragments/metrics/connectivity.py` as implemented in M11 (verified against the file directly, not from memory). `MetricDependency.FIXED_NODES` / `.GRAPH` match `hydrofragments/schema.py:72,74`.
