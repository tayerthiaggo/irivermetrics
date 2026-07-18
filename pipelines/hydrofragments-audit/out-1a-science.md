# Science Audit output

## Task
Verify every fragmentation metric in HydroFragments — formula-as-coded, source attribution, math correctness, unit consistency, edge cases — on the **science axis only** (efficiency handled in a separate stage).

## What this stage did

**Code read (verbatim):** `hydrofragments/metrics/{extent,patches,clustering,dynamics,persistence,connectivity,registry}.py`, `hydrofragments/spatial/connectivity_context.py`. Cross-checked feeder `hydrofragments/patches/morphology.py` (perimeter/width/length) and output emission in `hydrofragments/api.py`.

**Docs read:** `docs/paper-summary.md` (Paper 2 = canonical formulas), `docs/audit/scientific_metrics_audit.md` (prior adversarial audit — verified, not trusted blindly), `docs/HydroFragments_v1.2_spec.md`, `docs/metrics/dynamics.md`.

**Metrics verified:** APSEC, LPSEC, occurrence (PP), refuge area (RA), recurrence, hydroperiod, LPI, MESH, AWMSI, AWRe, number-of-pools, pool-width distribution, inter-pool gap, dry-down (extent contraction), reconnection timing, refuge spatial stability (Jaccard), RC (edge + pair), length-weighted RC_pair (DCI form), TCF.

**Prior-audit claims re-checked against live code:** occurrence-denominator bug (A5/F-1), refuge static-footprint bug (§6.16), dry-down recession misnaming (R2), NNI presence (R5). All resolved in code — see Findings.

**Verification of agent flags:** two "major unit drift" flags raised by sub-agents (MESH km2, inter_pool_gap km) were checked against `api.py` emission and **downgraded** — the output layer converts correctly (inter_pool_gap) or the metric is never emitted at runtime (MESH). Details below.

---

## Findings

### Connectivity / DCI / RC / TCF — highest formula risk, audited directly

**RC edge + pair — `connectivity.py:127-179`**
- Formula: `rc_edge_pct = 100·n_active/n_total`; `rc_pair_pct = 100·reachable_pairs/comb(|V|,2)` via union-find on active edges. Active edge = gap not None AND `gap <= gap_threshold` (default 0 = direct wet touch).
- Source: spec §6.13 / §6.11a; DCI positioning Cote et al. 2009.
- **Verdict: CORRECT.** Fixed external-network nodes (never monthly patch labels); structurally-dry reaches pre-filtered so denominators aren't diluted; `0/0 → NaN` (not 0) for empty graph. Snapshot semantics correct. Severity: n/a.

**Length-weighted RC_pair (DCI form) — `connectivity.py:182-224`**
- Formula: `100·Σ_k(L_k²)/(ΣL_k)²`, `L_k` = total length of connected component k. Docstring derivation from `DCI = 100·Σ_ij(len_i·len_j·c_ij)/(Σlen)²` with `c_ii=1` reduces exactly to the component-length-squared form.
- Source: Cote et al. 2009 DCI; spec §6.17.
- **Verdict: CORRECT.** Algebra verified: single connected network → 100; split {10,30}/40 → `100·(100+900)/1600 = 62.5` (matches docstring). `ΣL=0 → NaN`. Correctly walled as citation-only (Q4), not a shipped runtime metric. Severity: n/a.

**TCF — `connectivity.py:245-283`**
- Formula: per node `tcf_pct = 100·active_months/valid_months`, active counted only where valid. `valid_months=0 → NaN`; chronically isolated (valid, never active) → legit `0.0`.
- Source: spec §6.11 (renamed PCF→TCF to avoid PC collision); Rubio & Saura 2012.
- **Verdict: CORRECT.** Node identity fixed by graph, not monthly detections. Div-by-zero → NaN. Severity: n/a.

**Reach-wet detection — `connectivity_context.py:60-98`**
- Method: per-month `medial_axis` skeleton of water mask; reach wet iff ≥1 skeleton pixel inside reach-line buffer (default 2 px). `wet_any_month` = OR across series.
- **Verdict: CORRECT (method sound).** Seeded-skeleton avoids buffer-stamp false-credit; half-pixel transform offset correct. Severity: n/a. Note: skeleton quality drives it (documented upstream risk, not a code defect).

### Extent / persistence

**APSEC — `extent.py:67`** — `(n_water·cell_area)/A_ref·100`, fixed AOI denominator, water counted only where `water & valid_obs`. Source spec §6.17 / Paper 2 (Tayer 2023b). **CORRECT.** Non-circular. Can exceed 100 (no cap, correct). Minor: no per-month coverage floor — a sparsely-observed month still emits a value.

**LPSEC — `extent.py:50`** — `wetted_length/L_ref·100`, requires real channel (`has_real_channel` guard), >100% permitted (braided). Source spec §6.17 / Paper 2. **CORRECT.** Length in equal-area CRS flagged (`length_crs_caveat`). Severity: n/a.

**Occurrence / Pixel Persistence — `persistence.py:86-131`** — season-stratified: per calendar month `Σ(water&valid)/Σ(valid)` over years, then `mean_m(·)·100`, suppressed to NaN where `valid_count < min_valid_obs` (default 20). Source Paper 2 PP / spec §6.17 / Decision Gate U2·Q1; Pekel 2016, Mueller 2016. **CORRECT — prior A5/F-1 blocker FIXED** (denominator is valid_obs, not total timesteps, + floor). ⚠ **DOC DEFECT (major-for-paper):** code ships the *stratified equal-weight mean*; spec §6.17 still prints the *pooled* `Σ_t W/Σ_t V·100`. Different estimators (equal only under equal calendar-month support). §6.17 never updated to match U2/Q1.

**Refuge Area — `persistence.py:134-162`** — `count(OCC ≥ θ·100 AND valid_count ≥ min_valid_obs)·cell_area/1e6` km². θ default 0.90. Source spec §6.17 / Paper 2. **CORRECT.** Units clean (count·m²→km²). Minor: θ=90% is a Paper-2 magic constant (carried in output `refuge_threshold` — good; report sensitivity in any claim). Minor drift: Paper 2 uses `>90%`, spec+code use `≥` — code correctly follows newer spec.

**Recurrence — `persistence.py:165-188`** — `mean_m(Σ_t(water&valid)/Σ_t(valid))·100`. **CORRECT** but algebraically identical to occurrence's core estimator minus the min_valid_obs suppression. Minor: worth a clarifying note / DRY refactor.

**Hydroperiod — `persistence.py:191-207`** — per pixel-year `Σ(water&valid)/Σ(valid)`, returns **fraction [0,1]** (not ×100). Source spec §6.12. **CORRECT** (spec-correct as fraction). Minor: unit convention differs from sibling occurrence/recurrence (percent) in same file — verify registry/output expects fraction.

### Patches / morphology / clustering

**LPI — `patches.py:189`** — `max(aᵢ)/A_total·100`. Source spec §6.3, McGarigal & Marks 1995. **CORRECT — non-circular confirmed:** denominator is fixed `a_total_m2`, NOT Σaᵢ (weights computed separately at `:186-187`). Empty → NaN + `EdgeFlag.N0`.

**MESH — `patches.py:191`** — `Σ(aᵢ²)/A_total`, returned as `mesh_m2` (m²). Source spec §6.4, Jaeger 2000. **CORRECT formula, non-circular** (fixed `A_total`). Gated behind r(LPI,MESH)>0.9 correlation gate (implements spec hard gate). ⚠ **UNIT LABEL (minor, latent):** registry labels `mesh` `km2` but code returns m². **Verified harmless:** MESH is never emitted as a runtime metric record (no emission in `api.py`); only consumer is `validation/run_fitzroy_validation.py:107` which reads raw `mesh_m2` correctly. The km2 label is dormant metadata — no active 1e6 output error. Fix before MESH is ever wired to output: convert `mesh_m2/1e6` or relabel registry `m2`.

**AWMSI — `patches.py:190`** — `Σ[(0.25·pᵢ/√aᵢ)·(aᵢ/Σaᵢ)]`. Source spec §6.2 / Paper 2, McGarigal & Marks 1995. **CORRECT.** `0.25` constant exact + correctly placed; weights sum to 1; dimensionless. Perimeter = 4-connected raster-edge × pixel_size (FRAGSTATS convention). Severity: n/a.

**AWRe — `patches.py:192-196`** — `Σ[(2√(aᵢ/π)/lᵢ)·(aᵢ/Σaᵢ)]`. Source spec §6.1 / Paper 2, adapted from Schumm 1956. **CORRECT.** `2√(a/π)/l` = equal-area-circle diameter ÷ length; dimensionless. Skeleton-vs-major-axis length method locked per run (spec §6.1). Minor: one zero/NaN length nulls AWRe for whole month (all-or-nothing) rather than dropping that patch — conservative, document.

**Number of pools — `patches.py:152`** — `len(patches)`, 8-connectivity, `min_patch_pixels=3` filter. **CORRECT.** Minor: default 3 (30 m WOfS MMU, spec §6) vs Paper 2's `min_pool_size=2` (10 m Sentinel) — deliberate spec-over-paper, document so N/LPI/MESH aren't compared cross-resolution.

**Pool-width distribution — `patches.py:61` / `morphology.py:56`** — `width_m = max(2·EDT(mask)[medial_axis])·pixel_size`; retains only patches with `width_pixels > resolution_floor_pixels` (mandatory positive floor or raises). Source spec §6.9, Pavelsky & Smith 2008, Yang 2020. **CORRECT + EDGE-CASE HANDLED:** `2×EDT` half→full width standard; resolution floor implements prior-audit R3/F-9 narrow-channel guard; empty-after-filter → NaN + `WIDTH_RESOLUTION_FLOOR` warning. Residual 30 m morphology risk is validate-before-paper (R3), already guarded — not a code bug.

**Inter-pool gap — `clustering.py:23`** — maximal interior dry runs (bounded wet-both-sides) summed via `segment_lengths_m`; leading/trailing dry excluded. Stats mean/median/max/CV + `percent_above_threshold`. Source spec §6.8/§6.10 (metric of record over NNI), Sheldon 2010, Fullerton 2010. **CORRECT — right 1D geometry.** No NNI/Clark-Evans 2D-CSR machinery present. ✅ **NNI absent from codebase entirely** — implements prior-audit R5/F-7 (cut NNI). Registry labels `km`; **verified `api.py:278` converts `value_m/1000.0` → km, so label is CORRECT** (contra sub-agent's "major drift" flag).

### Dynamics

**Dry-down / extent contraction — `dynamics.py:108-178`** — OLS (`np.polyfit`) or Theil–Sen slope of APSEC (%) vs elapsed-month index, peak→end-dry inclusive; non-finite dropped; `low_df → NaN` when `<minimum_points` (default 3, config rejects <3). Dual-composite: `median=None` raises rather than fabricates; `composite_sensitive = |APSEC_max_enddry − APSEC_median_enddry| > 10pp`. Source spec §6.5 (Q11), Costigan 2016 / Gallart 2012. **CORRECT.** ✅ **Recession-misnaming (R2/F-4) RESOLVED:** no "recession" in any emitted string; `metric_name="extent_contraction"`; description explicitly disclaims hydrograph/discharge. df floor enforced. Units %/month, disagreement genuinely pp. **EDGE-CASE-BUG (minor):** `_end_dry_value` (`:96-105`) raises `ValueError` if end-dry month has no APSEC row in either composite — aborts an HY whose slope is otherwise computable. Degrade to `disagreement=NaN`/flag instead of raising.

**Reconnection timing — `dynamics.py:199-267`** — first month strictly after end-dry where series ≥ threshold, as integer month-lag; preference RC→LPSEC→LPI, `proxy_flag` only cleared for RC. Source spec §6.15/§6.29. **CORRECT.** No silent LPI fallback (§4.1). Minor edge: `_first_crossing` assumes sorted series (upstream contract, undefended) — sort on entry or assert.

**Refuge spatial stability (Jaccard) — `dynamics.py:282-305`** — `|R_y ∩ R_{y-1}|/|R_y ∪ R_{y-1}|` on two boolean footprints; first HY → None; empty∪empty → NaN. Source spec §6.16 option 1, Jaccard 1912. **CORRECT.** ✅ **Prior static-footprint bug (§6.16) RESOLVED:** compares each HY's own end-dry footprint vs *previous* HY's end-dry footprint — genuinely inter-annual, not a static long-term occurrence footprint. Minor: implements the water-footprint variant (not refuge-thresholded) — spec permits ("optionally"); verify caller passes intended footprint.

---

## Handoff to next stage

Science fixes before user-ready, ranked. **No blockers, no WRONG formulas.** The three previously-flagged blockers (occurrence denominator, refuge static footprint, dry-down recession naming) are all already fixed in code. Remaining items are minor code hardening + doc/spec reconciliation.

1. **[major — doc] Reconcile spec §6.17 occurrence formula with code.** Code ships season-stratified equal-weight mean (U2/Q1); §6.17 still prints pooled `Σ_t W/Σ_t V·100`. Different estimators. Update §6.17 or add `[AUDIT FIX]` noting U2/Q1 supersedes, else a reviewer diffing spec-vs-code calls it a discrepancy. `persistence.py:86-106` vs spec §6.17.

2. **[minor — code] Dry-down crashes on missing end-dry APSEC record.** `_end_dry_value` raises `ValueError`, aborting an otherwise-computable HY slope. Degrade to `disagreement=NaN`/flag. `dynamics.py:96-105, 161-165`.

3. **[minor — code] MESH unit label vs value.** Registry `km2`, code returns m². Currently harmless (MESH not emitted at runtime) but a trap if MESH is ever wired to output — convert `mesh_m2/1e6` or relabel `m2`. `patches.py:191`, `registry.py:142`.

4. **[minor — code] `_first_crossing` assumes sorted series.** Sort on entry or assert the ordering contract. `dynamics.py:199-212`.

5. **[minor — code] APSEC per-month coverage flag.** No minimum-valid-pixel guard per month (unlike occurrence's floor); sparse month still emits. Add coverage flag on `ApsecRecord`. `extent.py:94-106`.

6. **[minor — doc] Threshold/constant provenance.** Report θ=90% (RA) and `min_patch_pixels=3` (vs Paper 2's 2 px) alongside every derived claim; document cross-resolution non-comparability of N/LPI/MESH. `persistence.py:151`, `patches.py:217`.

7. **[minor — doc/verify] Unit-convention & redundancy notes.** Hydroperiod is fraction while occurrence/recurrence are percent (verify registry) `persistence.py:203`; recurrence ≈ occurrence-core (clarify difference) `:165-188`; AWRe all-or-nothing NaN (document) `patches.py:192-196`.

---

## Open questions / risks

- **Occurrence estimator is a published-paper divergence, not a bug.** Code's season-stratified equal-weight mean is scientifically *better* (corrects seasonal MNAR missingness) than Paper 2's pooled PP, but it means the shipped occurrence is not literally `WP/valid_obs`. This is a **user domain call**: is the stratified estimator the one to cite/publish, or should the pooled form be offered as an option? Affects how occurrence is described in the paper.

- **Buffer/cross-AOI sensitivity of APSEC/LPI/MESH (prior R6)** is a documentation/guard item not verifiable from code alone — fixed denominators kill within-series circularity but not cross-AOI arbitrariness (LPI depends on how much dry buffer the AOI includes). Confirm a guard/doc exists.

- **Empirical validation claims (prior V1–V8)** — AWRe⊥AWMSI orthogonality, LPI/MESH r>0.9 keep-both gate, max_water dry-down bias magnitude, RC/DCI benchmark vs riverconn — are all validate-before-*paper*, not fix-before-*code*. Formulas are sound; the interpretive/relational claims need the Gilbert data. Out of scope for this code audit; flag for the paper.

- **Reach-wet detection and pool length depend on skeleton quality** (`medial_axis`). Not a formula defect, but skeleton branching at meander bends can perturb width/length/gap — a known upstream sensitivity, worth a validation note.
