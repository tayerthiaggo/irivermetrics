# HydroFragments Audit — User-Ready Roadmap

## Verdict

**Ships today for correctness. Not yet ready at real data scale.**

Science axis: **zero blockers, zero WRONG formulas.** All three previously-flagged science blockers (occurrence denominator, refuge static footprint, dry-down recession naming) are already fixed in code. Every shipped metric formula is verified correct.

The one thing standing between "computes right answers" and "user-ready" is **performance/architecture**: the canonical `analyze()` path (`api.py:597-609`) forces the whole time-y-x cube into host memory via `section_compat_rows()` → `_monthly_dataset()` → `.load()`, *before* metric selection, and computes all core metrics regardless of the requested profile. On a real satellite cube this defeats Dask chunking and can OOM. That makes it a **scale blocker** — the tool is unusable on production AOIs even though it is numerically correct on small ones.

**1 blocker (perf/scale). 4 major. Rest minor/nice-to-have.**

Everything else — CUDA, distributed labels, benchmark expansion — is deferrable.

---

## Blockers (must fix)

### B1 — [perf] Whole-cube `.load()` + no profile gating in canonical `analyze()`
- **file:line:** `api.py:603` (call), `compat.py:116-118` (`_monthly_dataset().load()`), `api.py:626-656` (selection happens *after* compute)
- **defect:** `analyze()` builds an eager `xr.Dataset` then calls `section_compat_rows()`, which materializes the full T·Y·X cube at the compat boundary and computes occurrence, refuge, APSEC, and patch morphology for **every** month. `selected_ids` is resolved only afterward (`api.py:626`) and used just to *filter* already-computed records. A narrow profile still pays full patch labeling/morphology + full-cube memory.
- **fix:** Resolve `selected_ids` **before** any kernel runs. Dispatch only requested metric families (APSEC only if `apsec` selected; `analyze_patch_metrics()` only if one of `number_of_pools/lpi/awre/awmsi/mesh`; occurrence/refuge only for persistence outputs). Route monthly compositing through `run_monthly_pipeline()` instead of the compat facade; keep `section_compat_rows()` as a legacy-only path. Guard with snapshot tests for row presence/order parity (`_records_from_compat_rows` is the current canonical bridge — parity is the main risk).
- **why blocking:** This is the only issue that makes the correct tool *unusable at real scale* — O(T·Y·X) eager RAM before selection + wasted full patch work. Perf issue promoted to blocker per contract (perf that makes tool unusable at scale = blocker). Combines efficiency findings #1 + #2 (same root cause, fix together).

---

## Major (should fix before wide release)

### M1 — [doc/science] Spec §6.17 occurrence formula disagrees with shipped estimator
- **file:line:** `persistence.py:86-131` (code) vs `docs/HydroFragments_v1.2_spec.md` §6.17
- **defect:** Code ships the **season-stratified equal-weight mean** (per-calendar-month `Σ(water&valid)/Σ(valid)` over years, then mean over months × 100), per Decision Gate U2/Q1. Spec §6.17 still prints the **pooled** `Σ_t W/Σ_t V·100`. Different estimators (equal only under equal calendar-month support). Formula is *correct and better* — corrects seasonal MNAR missingness — but a reviewer diffing spec-vs-code flags it as a discrepancy.
- **fix:** Update §6.17 to the stratified form, or add an `[AUDIT FIX]` note that U2/Q1 supersedes the pooled formula. **User domain call attached (see below):** decide whether stratified or pooled PP is the one cited in the paper.
- **why major:** Not a code bug; it's a publication-integrity item. Blocks *wide release / paper*, not *use*.

### M2 — [perf] Per-month patch label/crop done twice when pool-width selected
- **file:line:** `patches.py:225` + `patches.py:253`, `api.py:317/322/680`
- **defect:** `analyze_patch_metrics()` labels+crops each month for core patch metrics; if `pool_width` is selected, `_pool_width_records()` loops the same months and labels+crops **again**. Up to ~2× label/crop work.
- **fix:** Per-month `PatchAnalysis` bundle: label once, crop once, measure once with `include_width` toggled when any width stat is selected; emit core + pool-width records from shared properties. Must preserve warning flags + width-resolution-floor suppression semantics.
- **why major:** Highest-leverage patch-path win after B1; doubles the dominant cost on width-heavy runs. Not a scale *blocker* only because it's bounded by B1's gating (skipped entirely if width not selected). Efficiency #6.

### M3 — [perf] O(V²) connectivity: graph build + reachable-pair count
- **file:line:** `connectivity.py:74-82` (nested pair scan in `build_fixed_graph`), `connectivity.py:163-166` (pairwise reachable count in `compute_realised_connectivity`)
- **defect:** `build_fixed_graph()` finds adjacencies by nested loop `to_node_a == from_node_b` (O(V²)); RC counts reachable pairs by checking every root pair (O(V²) per month → O(M·V²)).
- **fix:** Adjacency: dict `From_Node → reach IDs`, look up children by `To_Node`. RC: `Counter` over roots, `Σ comb(size,2) / comb(V,2)`. **Formula identical** — pure algebra, no numeric change.
- **why major:** Small patch, big win for large drainage networks; low risk. Science audit confirms RC/pair formula correct, so this is safe to rewrite.

### M4 — [perf] `reach_wet_any_month()` keeps R dense masks + per-reach full-frame intersect
- **file:line:** `connectivity_context.py:77-95`
- **defect:** Rasterizes a full boolean buffer mask **per reach** (memory O(R·Y·X)); each month skeletonizes the full frame and intersects against every not-yet-wet reach (O(T·R·Y·X)). Can dominate connectivity runs.
- **fix:** Single integer/multilabel reach raster (or sparse coord→reach map for overlapping buffers); skeletonize each month once; identify wet reaches by indexing/unique-counting labels under skeleton pixels.
- **why major:** Can dominate connectivity wall-clock/memory. Med/high effort + overlapping-buffer attribution risk keep it below B1. Science audit confirms seeded-skeleton method sound — preserve exact reach-attribution semantics. Efficiency #11.

---

## Minor / nice-to-have

### m1 — [code] Dry-down crashes on missing end-dry APSEC record
- **file:line:** `dynamics.py:96-105` (raises `ValueError`), used at `:161-165`
- **fix:** Degrade to `disagreement=NaN` / flag instead of raising; aborting an otherwise-computable HY slope is too harsh. Confirmed: `_end_dry_value` raises if no matching month record.
- **rank why:** Robustness edge case, not wrong math. Rare trigger. Science audit item #2.

### m2 — [perf] APSEC recomputed one month at a time
- **file:line:** `compat.py:146/155`, `extent.py:87/91`
- **fix:** Call `compute_apsec(monthly, ...)` once over the time axis, map records by timestamp. (Largely subsumed by B1's rewrite — do it there.)
- **rank why:** Low effort/low risk; becomes a *bigger* Dask-graph bug if `.load()` removed without it, so pair with B1. Efficiency #3.

### m3 — [perf] Label normalization uses sort/searchsorted over all pixels
- **file:line:** `labels.py:54-55/75`
- **fix:** `np.bincount` for counts + lookup table indexed by raw label ID → `lookup[raw_labels]` (O(P+K) vs O(P log P)). Needs tests for deterministic row-major ID ordering + sparse dask-image labels.
- **rank why:** Matters on highly fragmented scenes; med risk (ID ordering). Efficiency #5.

### m4 — [perf] Width path likely computes EDT twice
- **file:line:** `morphology.py:55-56`
- **fix:** Use distance-returning `medial_axis` API (skeleton + distance together), reuse on skeleton pixels. **Requires exact width-pixel parity fixture before changing** — science-adjacent. Efficiency #7.
- **rank why:** Low effort but touches width numerics; parity gate mandatory.

### m5 — [code] `_first_crossing` assumes sorted series
- **file:line:** `dynamics.py:199-212`
- **fix:** Sort on entry or assert the ordering contract. Science audit item #4.
- **rank why:** Latent correctness-if-contract-violated; cheap defensive fix.

### m6 — [code] MESH unit label vs value (latent)
- **file:line:** `patches.py:191` returns m², `registry.py:142` labels `km2`
- **fix:** Convert `mesh_m2/1e6` or relabel registry `m2`. **Currently harmless** — MESH never emitted at runtime; only consumer (`validation/run_fitzroy_validation.py:107`) reads raw `mesh_m2` correctly.
- **rank why:** Dormant metadata trap; fix before MESH ever wired to output. Both audits agree it's inert now.

### m7 — [code] APSEC has no per-month coverage floor
- **file:line:** `extent.py:94-106`
- **fix:** Add a min-valid-pixel coverage flag on `ApsecRecord` (occurrence already floors; APSEC doesn't). Science audit item #5.
- **rank why:** Sparse-month values emit silently; documentation/flag hardening.

### m8 — [perf] Temporal AOI summaries trigger separate scalar `.item()` materializations
- **file:line:** `api.py:438-460`, `persistence.py:178-200`
- **fix:** Build requested temporal summaries in one dataset, `.compute()` once, extract all scalar rows. Efficiency #12.
- **rank why:** Scheduler-overhead only; arithmetic already valid.

### m9 — [code] Region props one skimage object per component
- **file:line:** `morphology.py:53/71`
- **fix:** Evaluate `regionprops_table()` bulk path or vectorized moment reducer; **major-axis parity must be certified** (science-adjacent). Efficiency #8.
- **rank why:** Overhead on many-tiny-patch masks; parity gate keeps it minor.

### m10 — [perf] `chunks` discarded in `open_water_cube()`
- **file:line:** `api.py:67-76`, `chunks.py:45`
- **fix:** Honor `chunks` in zarr open/array wrap, record chosen chunks in manifest, add explicit rechunk-planning step. Efficiency #13.
- **rank why:** Users can't tune large runs without bypassing API — real, but a knob, not a correctness/scale wall once B1 lands.

### m11 — [doc] Threshold/constant provenance + cross-resolution non-comparability
- **file:line:** `persistence.py:151` (θ=90% RA), `patches.py:217` (`min_patch_pixels=3` vs Paper 2's 2)
- **fix:** Report θ and MMU alongside every derived claim; document that N/LPI/MESH aren't comparable across 30 m vs 10 m. Science audit item #6.
- **rank why:** Paper hygiene; already carried in output metadata.

### m12 — [doc] Unit-convention + redundancy notes
- **file:line:** `persistence.py:203` (hydroperiod is fraction while occurrence/recurrence are percent — verify registry), `:165-188` (recurrence ≈ occurrence-core, clarify), `patches.py:192-196` (AWRe all-or-nothing NaN, document)
- **rank why:** Clarity, not correctness. Science audit item #7.

### m13 — [perf] Dask label global materialization per month
- **file:line:** `labels.py:42-44`, `patches.py:225`
- **fix:** Make the compute() boundary an explicit per-month label checkpoint with chunk/shape diagnostics; staged distributed labels only if scaling past one 2-D month. **Exact cross-chunk component identity must not change.** Efficiency #4.
- **rank why:** Peak-memory limiter on huge AOIs, but bounded by one 2-D month; defer until profiling proves need.

---

## Conflicts resolved

The two audits **do not truly conflict** — they touch overlapping modules but the science audit already certified the formulas the perf audit wants to speed up. The genuine tension is *"can this perf rewrite change a verified-correct number?"* Resolutions:

| # | Conflict | Winner | Why |
|---|----------|--------|-----|
| C1 | M3 rewrites RC / reachable-pair counting for O(V²)→O(V). Could a component-size formula change the verified RC value? | **Perf wins** | Science audit verified `100·Σcomb(size,2)/comb(V,2)` *is* the formula; Counter-based counting is identical algebra. Zero numeric change. Safe. |
| C2 | m4 (EDT-once) + m9 (`regionprops_table`) + m3 (bincount labels) speed up width/major-axis/label paths the science audit certified for **exact values** (2·EDT width, major-axis length, row-major IDs). | **Science wins — perf gated** | Do NOT merge any of these without an exact-parity fixture pinning current output. Science correctness outranks perf. Ship the parity test first, optimization second. |
| C3 | B1 routes canonical compute through `run_monthly_pipeline()` instead of `section_compat_rows()`, which is the current canonical **row bridge** for retained metrics. | **Perf wins, with snapshot guard** | The scale wall must fall, but `_records_from_compat_rows()` parity (row presence/order/values) must be locked by snapshot tests before switching. Correct-but-unusable is not user-ready. |
| C4 | m13 / distributed labels + CUDA morphology could change component identity at chunk boundaries. | **Science wins — keep CPU reference** | Keep CPU label/morphology as numerical reference until cross-chunk parity is *proven*. No distributed/GPU morphology in the user-ready milestone. |

**Rule applied throughout:** science correctness blockers outrank perf; the *only* perf item promoted to blocker (B1) is promoted because it makes the correct tool unusable at real scale, not because it's fast-to-fix.

---

## Not blocking — safe to ship without

- **CUDA / GPU acceleration** (efficiency #14). Scaffold is truthful, `enabled_cuda_stages` empty, `analyze()` honestly records CPU backend. No silent GPU claim. Keep advertised as candidate/incomplete. Not needed for user-ready.
- **Benchmark harness expansion** (efficiency #15). `cpu_baseline.py` misses the real hot path (`analyze()`, patch morphology, width, connectivity), but that's *measurement* debt, not a runtime defect. Add stages after B1/M2/M3 land so numbers reflect the fixed path.
- **m6 MESH unit label** — inert (never emitted). Fix opportunistically.
- **m8, m10, m13** — scheduler/chunk/memory tuning. Real at extreme scale but B1 removes the acute wall; these are follow-on tuning.
- **All doc items (M1 caveat aside, m11, m12)** — required before the *paper*, not before a user runs the tool. M1 specifically blocks *publication*, not *execution*.
- **Empirical validation claims** (science open questions V1–V8: AWRe⊥AWMSI, LPI/MESH r>0.9 gate, dry-down max_water bias, RC/DCI-vs-riverconn) — validate-before-paper with Gilbert data. Formulas sound; out of scope for code readiness.

---

## Single user domain call (carry to paper, not code)

The shipped occurrence estimator is a **deliberate improvement** over Paper 2's pooled PP — season-stratified equal-weight mean corrects seasonal missingness (MNAR). It is *not* literally `WP/valid_obs`. **Decide which form to cite/publish** (stratified as shipped, or offer pooled as an option). This drives M1's doc fix, not the reverse. Not a bug — a scientific choice already made in code that the spec/paper must catch up to.
