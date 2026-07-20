# HydroFragments User-Ready Follow-Ups — Handoff

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Context:** `docs/superpowers/plans/2026-07-18-hydrofragments-user-ready.md` (14 tasks, all complete, merged to `development` at `f536417`) fixed the audit's one blocker (B1) plus 4 major and 13 minor items. Three items were deliberately **descoped** during execution because closing them fully would have required a real design decision or risked exactly the kind of numeric drift the plan's Global Constraints forbid. Each was reviewed, accepted as out-of-scope-for-that-task, and explicitly called out in the final whole-branch review as a live, tracked follow-up — not a defect in the merged work.

This file is the handoff for those three items. Read `pipelines/hydrofragments-audit/audit-report.md` and the parent plan above for full background before starting; this file assumes that context.

## Global Constraints (carried over from the parent plan — still binding)

- **No numeric-output change on any verified metric** without a golden-fixture parity test proving identity before/after. `tests/gating/test_analyze_row_snapshot.py` + `tests/gating/analyze_snapshot.json` is the standing regression gate — it must still pass, unchanged, after every task below.
- **CPU label + morphology stays the numerical reference.** No distributed-label or GPU-morphology path.
- Every task ends with the relevant test suite green + a commit.

---

## Task 1: Reconcile the `water` vs `water & valid_obs` mask divergence, then wire `analyze_patch_bundle()` into `api.py`

**Origin:** Task 4 of the parent plan (M2 — single per-month patch bundle). `analyze_patch_bundle()` was added to `hydrofragments/metrics/patches.py` and is tested/proven correct in isolation, but was **never wired into `api.py`**, so the cross-call redundancy the audit originally flagged (label/crop/measure running twice per month when both core patch metrics and `pool_width` are selected) still exists today.

**Why it was blocked:** the two current call sites use genuinely different masks:

- Core patch metrics: `section_compat_rows()` (`hydrofragments/compat.py:128-215`) calls `_monthly_dataset()` (`hydrofragments/compat.py:116-121`), which **hardcodes** `valid_obs = xr.ones_like(water, dtype=bool)` — i.e. every pixel is treated as valid regardless of the cube's real `valid_obs`. The mask fed to `analyze_patch_metrics` (`compat.py:187-198`) is therefore `water` alone.
- `pool_width`: `_pool_width_records()` (`hydrofragments/api.py:328-345`) builds its mask as `monthly["water"].isel(...) & monthly["valid_obs"].isel(...)` — the cube's **real** `valid_obs`.

For `generic_binary` input (`open_water_cube`'s default path, `api.py:91-100`), `water = (array == 1)` and `valid = valid_obs.astype(bool)` are set completely independently — `water=True, valid_obs=False` is a reachable, unguarded state (no invariant `water ⊆ valid_obs` anywhere in `hydrofragments/guards/` or `hydrofragments/io/alignment.py`, confirmed during Task 4's review). Bundling the two call sites onto one shared mask would silently change `pool_width`'s numeric output whenever the masks diverge — or, if you instead narrow `section_compat_rows` to `water & valid_obs`, it changes `lpi`/`awre`/`awmsi`/`number_of_pools` instead. Either direction is a real behavior change requiring its own parity fixture, not a mechanical refactor.

**Files:**
- `hydrofragments/compat.py:116-121` (`_monthly_dataset`), `:128-215` (`section_compat_rows`)
- `hydrofragments/api.py:328-386` (`_pool_width_records`), `:690-720` (canonical `analyze()` dispatch region)
- `hydrofragments/metrics/patches.py` (`analyze_patch_bundle`, already correct — reference only, should not need changes)
- New: parity/golden-fixture test proving whichever direction you pick

**Interfaces:**
- Consumes: `analyze_patch_bundle(mask, *, pixel_size_m, a_total_m2, connectivity, min_patch_pixels, target_component_pixels, include_mesh, include_width, resolution_floor_pixels=None)` — already implemented, do not change its signature.
- Produces: `section_compat_rows()` and `_pool_width_records()` (or their replacement) call `analyze_patch_bundle()` once per month instead of `analyze_patch_metrics()` + a second independent labeling pass.

- [ ] **Step 1: Make the mask-semantics decision (human input required — do not guess)**

  This is a domain/product decision, not an engineering one. Options, as identified during Task 4's review:
  - **(a)** Scope the bundle to only fire when `config` guarantees `water ⊆ valid_obs` (e.g. only for `watermask_tsfill` input — confirmed in `hydrofragments/io/adapters.py:5-8` that `parse_watermask_tsfill` computes `water = (da_in == 1) & valid`, so the invariant holds there), falling back to today's two separate calls otherwise. Zero numeric-output risk, partial perf win, more branching in `api.py`.
  - **(b)** Unify onto `water & valid_obs` everywhere (change `_monthly_dataset` to stop hardcoding all-True `valid_obs`). This is a deliberate, real behavior change to `lpi`/`awre`/`awmsi`/`number_of_pools` for any run where the cube's `valid_obs` isn't all-True — needs its own golden-fixture parity test explicitly proving the new values are correct, and needs sign-off that this is the intended semantics (arguably it *should* have been this way all along, since "unobserved pixels count as land/dry" is a stronger scientific claim than "unobserved pixels are excluded" — but that's exactly the kind of call this file can't make for you).
  - **(c)** Leave it split permanently, close the follow-up as "won't fix," and only chase the perf win if profiling on a real production AOI shows patch labeling is still a bottleneck after all of B1/M2/M3/M4's other fixes.

  Get explicit user sign-off on (a), (b), or (c) before writing code. If (c), close this task and skip to Task 2.

- [ ] **Step 2: Write the parity/decision test first**

  Whichever of (a)/(b) is chosen, write a test on a `generic_binary` fixture with at least one `water=True, valid_obs=False` pixel that pins the chosen behavior explicitly (either "bundle is skipped, output identical to today" for (a), or "output changes to X, here's why X is correct" for (b)).

- [ ] **Step 3: Wire the bundle into `api.py`**

  Route `section_compat_rows()`'s patch-metric loop and `_pool_width_records()` through one `analyze_patch_bundle(..., include_width=("pool_width" in selected_ids))` call per month.

- [ ] **Step 4: Run regression suite**

  Run: `pytest tests/gating/ tests/metrics/test_patch_bundle.py tests/compat/ -v`
  Expected: PASS. For option (b), `tests/gating/analyze_snapshot.json` will need a deliberate, reviewed update — do not silently regenerate it; diff the old vs. new values and confirm each change is expected.

- [ ] **Step 5: Commit**

  ```bash
  git add hydrofragments/api.py hydrofragments/compat.py <new test files>
  git commit -m "perf: wire analyze_patch_bundle into api.py call sites (Task 4 follow-up)"
  ```

---

## Task 2: Wire `compat.py`'s `compute_apsec` call to pass `valid_obs`/`min_valid_fraction`

**Origin:** Task 8 of the parent plan (m7 — APSEC per-month coverage floor flag). `ApsecRecord.low_coverage_flag` and `compute_apsec`'s optional `valid_obs`/`min_valid_fraction` parameters (`hydrofragments/metrics/extent.py:68-107`) were added and are fully tested in isolation (`tests/metrics/test_apsec_coverage_floor.py`), but the only call site on the live `analyze()`/compat path — `section_compat_rows()` at `hydrofragments/compat.py:170-176` — never passes them. **`low_coverage_flag` is always `False` in production output today.**

**Why it matters:** a user could reasonably assume, from the schema/docs, that this flag is live once they see it in `ApsecRecord`. It is not wired. This is a trap of false confidence, flagged explicitly in the final whole-branch review as worth closing (or at minimum documenting loudly) before anyone relies on it.

**Files:**
- `hydrofragments/compat.py:154-176` (`section_compat_rows`, the `apsec_records = compute_apsec(...)` call)
- `hydrofragments/metrics/extent.py:68-107` (`compute_apsec` — already correct, reference only)
- `hydrofragments/config.py` — check whether a `min_valid_fraction` config knob already exists (search for `min_valid_fraction_month` per the spec's item 5, `docs/HydroFragments_v1.2_spec.md`) or needs to be added
- New/modified: `tests/metrics/test_apsec_coverage_floor.py` or a new integration test exercising the flag through `analyze()`

**Interfaces:**
- Consumes: `compute_apsec(monthly, *, a_ref_m2, cell_area_m2, config, valid_obs=None, min_valid_fraction=None)` — signature already correct.
- Produces: `section_compat_rows()` passes `valid_obs=monthly["valid_obs"]` and a real `min_valid_fraction` threshold (sourced from config — check `config.temporal.min_valid_fraction_month` per spec item 5, or a new/existing `PersistenceConfig`/`PatchesConfig` field) so `low_coverage_flag` reflects real per-month coverage on the production path.

- [ ] **Step 1: Find or add the config knob**

  Read `hydrofragments/config.py` in full. Per `docs/HydroFragments_v1.2_spec.md` item 5 ("Monthly validity and per-pixel validity are separate... use `min_valid_fraction_month` for whether a monthly AOI/zone metric is reportable"), there may already be a field named close to this. If it exists, use it. If not, add it to the appropriate config dataclass with a sensible default (check what occurrence's existing coverage floor uses, e.g. `min_valid_obs`, for a consistent default philosophy) — this is additive config, should not require a migration.

- [ ] **Step 2: Write the failing integration test**

  Exercise this through the real `analyze()` entry point (not just `compute_apsec` directly, which Task 8 already covered) — build a cube with a genuinely sparse month, run `analyze()` with an `apsec`-selecting profile, and assert the emitted APSEC record's metadata carries `low_coverage_flag=True` for that month. Check how `MetricRecord`/`_records_from_compat_rows` currently exposes (or doesn't expose) per-record flags — you may need to extend the record/row shape to carry this flag through to output, since Task 8 only added it to the internal `ApsecRecord`, not necessarily to the public `MetricRecord` schema. Investigate this gap before writing the test; it may be a bigger piece of work than the config-wiring alone.

- [ ] **Step 3: Wire `section_compat_rows()`**

  Pass `valid_obs=monthly["valid_obs"]` and the resolved `min_valid_fraction` into the existing `compute_apsec(...)` call (`compat.py:170-176`). Confirm the APSEC **value** is unaffected — only the flag.

- [ ] **Step 4: Run regression suite**

  Run: `pytest tests/gating/ tests/metrics/test_apsec_coverage_floor.py tests/metrics/test_apsec_vectorized.py tests/compat/ -v`
  Expected: PASS, `tests/gating/analyze_snapshot.json` unchanged (value-only snapshot, flag is metadata).

- [ ] **Step 5: Commit**

  ```bash
  git add hydrofragments/compat.py hydrofragments/config.py <test files>
  git commit -m "feat: wire APSEC coverage-floor flag into live analyze() path (Task 8 follow-up)"
  ```

---

## Task 3: Strengthen Task 9's numeric-equivalence proof for batched temporal summaries

**Origin:** Task 9 of the parent plan (m8 — batch temporal AOI summaries). `_temporal_profile_records()` (`hydrofragments/api.py:465-535`) was rewritten to materialize all temporal AOI summaries (AOI-mean recurrence + per-hydroperiod-year values) in one `xr.Dataset().compute()` call instead of N separate `.item()` calls — which, in the installed xarray version, actually **raised `NotImplementedError`** on dask-backed arrays rather than merely being slow. The fix is real, verified, and low-risk.

**The residual gap:** the only numeric-equivalence test (`tests/api/test_temporal_summary_batch.py:164-`, `test_batched_temporal_summaries_match_eager_nondask_values`) compares a dask-backed run against an eager (numpy-backed) run through the **same post-refactor code**. Both paths share the identical batching/dict-assembly logic in `_temporal_profile_records`, so this test can catch dask-specific execution bugs but **cannot** catch a backend-agnostic bug in the batching logic itself — e.g. a value silently assigned to the wrong `hydroperiod_{year}` dict key, or an off-by-one in per-year windowing. That exact bug class would produce matching-but-wrong values on both sides of the existing test and pass it anyway. This was accepted as non-blocking in the final review (the diff is small and mechanical — same `.mean(skipna=True)` calls as before, just batched), but it is a real, named gap in the test suite, not a hypothetical one.

**Files:**
- `hydrofragments/api.py:465-535` (`_temporal_profile_records` — should not need code changes, this is a test-only task)
- `tests/api/test_temporal_summary_batch.py` (extend)
- `hydrofragments/metrics/persistence.py:165-208` (`compute_recurrence`, `compute_hydroperiod` — read for hand-derivation)

**Interfaces:**
- Consumes: `analyze(cube, ..., metric_profiles=["pixel_temporal"])` — existing public entry point, no signature change.
- Produces: a new test that pins actual recurrence/hydroperiod-year values against a hand-derived or independently-computed expected value, not just cross-backend equality.

- [ ] **Step 1: Build a small, fully hand-traceable fixture**

  Reuse or shrink the existing `_raw_arrays`/`_dask_cube` helpers in `tests/api/test_temporal_summary_batch.py` down to something small enough to hand-compute by inspection — e.g. 2 years (24 months), a tiny grid (2x2 or 3x3), with a deliberately simple, documented water/valid pattern (not `rng.random(...)`) so the "correct" recurrence and each year's hydroperiod value can be written down in the test as a literal expected number, derived independently of `_temporal_profile_records`'s own code.

- [ ] **Step 2: Hand-derive the expected values**

  Using `compute_recurrence`/`compute_hydroperiod`'s documented formulas (read `hydrofragments/metrics/persistence.py:165-208` and the reconciled spec section from the parent plan's Task 14, `docs/HydroFragments_v1.2_spec.md` §6.12/§6.17-adjacent hydroperiod section), compute by hand (or with a tiny standalone numpy script, not the library code) what the AOI-mean recurrence and each year's hydroperiod value should be for your fixture. Write these as literal `pytest.approx(...)` expected values in the test — this is the part that actually proves batching correctness, since it's independent of both the dask and eager code paths.

- [ ] **Step 3: Add the test**

  New test function (e.g. `test_batched_temporal_summaries_match_hand_derived_values`) asserting `analyze()`'s real output against your hand-derived numbers.

- [ ] **Step 4: Run**

  Run: `pytest tests/api/test_temporal_summary_batch.py -v`
  Expected: PASS. If it does NOT pass, this may reveal Task 9's batching actually has a value bug the eager-vs-dask test missed — investigate for real before "fixing" the test to match, per this project's TDD discipline.

- [ ] **Step 5: Commit**

  ```bash
  git add tests/api/test_temporal_summary_batch.py
  git commit -m "test: add hand-derived value proof for batched temporal summaries (Task 9 follow-up)"
  ```

---

## Self-Review

- Task 1 → closes Task 4's `api.py` wiring gap (requires human decision on mask semantics before code).
- Task 2 → closes Task 8's `compat.py` wiring gap (may require extending the public record schema, investigate scope during Step 2).
- Task 3 → closes Task 9's residual test-coverage gap (test-only, no production code expected to change).
- All three were independently confirmed still-present and accurately characterized by the parent plan's final whole-branch review (commit range `77d9e45..36c6a7e`, merged at `f536417`).
- None of these are blockers for the merged `development` branch — they are backlog, ordered here by how much decision-making they require before implementation (Task 1 needs a real product call; Task 2 needs scope investigation; Task 3 is pure test-writing).
