# HydroFragments v1.2 — Adversarial Synthesis

**Date:** 2026-07-10  
**Stance:** adversarial principal investigator + Reviewer 2 + senior maintainer  
**Constraint:** diagnosis and plan stress-test only; no source files edited  
**Verdict:** **NO-GO on the full planned migration as currently sequenced.** Conditional **GO** only on a stripped minimal credible v1.2 after locking open contract decisions.

> **Mandatory next-phase intake gate**
>
> Before writing `implementation_plan.md` or editing code, next phase must:
>
> 1. Enumerate every file under `docs/audit/` present at phase start.
> 2. Read each file in full (not summaries), including this report.
> 3. Record an intake manifest: filenames + unresolved cross-audit conflicts.
> 4. Stop if conflicts remain on input validity, compositing ownership, metric semantics, or implementation order.
>
> Prior audits ingested for this synthesis (order):
>
> 1. [`evidence_packet.md`](evidence_packet.md)
> 2. [`repo_triage.md`](repo_triage.md)
> 3. [`spec_compliance.md`](spec_compliance.md)
> 4. [`scientific_metrics_audit.md`](scientific_metrics_audit.md)
> 5. [`dask_cuda_audit.md`](dask_cuda_audit.md) and [`dask_cuda_audit_adversarial.md`](dask_cuda_audit_adversarial.md)
> 6. [`manager_interpretation_audit.md`](manager_interpretation_audit.md)
> 7. [`docs_audit.md`](docs_audit.md)
> 8. Planning context: [`../audit_implementation_plan.md`](../audit_implementation_plan.md)

---

## 0. Cross-audit conflict register (must resolve before Stage 8)

| Conflict | Side A | Side B | Why it blocks |
|---|---|---|---|
| **Rebrand timing** | `audit_implementation_plan.md` Recommended Order step 1: rebrand first | `spec_compliance.md` step 13 / `docs_audit.md` §5: rebrand after API stabilises | Early rename creates fourth identity while API still legacy |
| **CUDA ambition** | `dask_cuda_audit_adversarial.md`: cuCIM skeletonize + cuGraph BFS as near-term path | `dask_cuda_audit.md`: skeleton/igraph stay CPU; cuCIM not certified drop-in | Optimistic CUDA plan will burn schedule and risk silent metric drift |
| **Dry-down feasibility** | Spec + scientific audit treat dual-composite dry-down as core | WaterMask-TSFill already monthly; `spec_compliance.md` Q3: dual composite may be unrecoverable from upstream alone | Headline metric may be unshippable without upstream raw observations or both composites |
| **HY algorithm** | Spec claims persistence-based HY as contribution | Scientific audit R4/V8: Tayer 2025/2026 already published dynamic HY; algorithm unlocked (Q7) | Implement-before-lock risks duplicating published method |
| **Drainage contract** | Spec requires Zone 1, `L_ref`, gaps, RC/TCF | Current API: polygon + scalar `section_length` only (`spec_compliance.md` Blocker 5, Q6) | Half the secondary/connectivity register has no input |
| **Validity semantics** | Spec implies native `observed` denominator | Q1 unlocked: fill pixels may or may not count | Occurrence/RA/zones all change with this choice |
| **NNI fate** | Spec keeps Exploratory | Scientific + manager audits: cut from v1 / never surface to managers | Schema and docs will thrash if undecided |
| **DCI scope** | Spec Secondary/Core judgement call | Scientific audit: citation-minimum OK; implement only with benchmark | Connectivity module scope unbounded |
| **Legacy compatibility** | Plan says “backward compatibility where reasonable” | Spec drops PF/PLF/AWMP*; tests require them | Mixed schema is worse than clean break |
| **Docs honesty vs vaporware** | Docs audit: do not document v1.2 contracts as if they exist | Spec wants `input_format.md` / manager guide as v1.2 deliverables | Writing finished docs before contracts land recreates current README failure mode |

**Rule:** Stage 8 implementation plan must open with a decision table that closes every row above. Unresolved rows = stop.

---

## 1. Attack answers (the ten questions)

### 1.1 What is over-scoped?

The combined plan tries to ship, in one migration:

- Full scientific register (core + secondary + connectivity + zones + HY dynamics)
- True Dask-first spatial morphology (cross-chunk labels, component crops, Parquet partitioning)
- Optional CUDA path with capability registry and benchmarks
- Tidy long schema + config hashing + spatial rasters
- Manager guide with decision narratives
- Rebrand + packaging + CI lint/docs
- Validation studies that are paper-blocking, not software-blocking

That is three products: (A) a correct metric library, (B) a scalable compute engine, (C) a decision-support communication layer. Shipping A+B+C together is how this repo stays unshippable for another year.

Also over-scoped relative to evidence:

- Treating CUDA as near-term architecture work (`dask_cuda_audit_adversarial.md`) when the careful audit shows morphology cannot honestly go GPU yet.
- Treating full connectivity (RC/TCF/DCI) as same milestone as occurrence denominator fix.
- Treating JOSS/history packaging as adjacent work even while saying it is non-blocking — it still steals attention in checklists.

### 1.2 What is under-specified?

These are not “nice to clarify later.” They are load-bearing and currently unlocked (`spec_compliance.md` Questions before editing):

1. **Validity semantics (Q1):** is `observed=True` the sole valid denominator, or may filled/resolved pixels count?
2. **Canonical API object (Q2):** `WaterCube` vs separate `water_mask=` / `valid_obs=` parameters.
3. **Monthly ownership (Q3):** who composites — HydroFragments, caller, or upstream? Dual-composite dry-down dies if only one monthly product arrives.
4. **DCI decision (Q4):** implement vs cite-and-position.
5. **Compatibility policy (Q5):** shim vs clean break; never mixed columns.
6. **Drainage input contract (Q6):** line topology, CRS, per-AOI `L_ref`, fixed nodes.
7. **HY algorithm (Q7):** executable parameters, drought fallback, differentiation from Tayer 2025/2026.
8. **Validation fixtures (Q8):** synthetic + Gilbert subset; do not pretend `wmask_ts.nc` covers dry-down/stability.
9. **Config hash rules (Q9):** path exclusion / content addressing / cross-platform stability.
10. **Publication history (Q10):** graft vs link predecessor — irrelevant to code, but must not block v1.2.

Additional under-specs the audits surface but do not lock:

- Behaviour when `valid_obs_frac` below floor: mask pixel, flag row, or drop timestep?
- Default `state_flag_connectivity_metric` when RC/DCI absent in minimal v1.2.
- Whether legacy temporal fill (`fill_nodata`) survives at all once native validity exists.

### 1.3 What will fail scientifically?

| Failure | Evidence | Severity |
|---|---|---|
| Occurrence/RA with total-timestep denominator | Current code; scientific F-1; triage #2/#5 | Fatal if shipped |
| `255`/`254` collapsed to dry via `uint8→int8` | Spec compliance B2 | Fatal if shipped |
| Dry-down framed/sold as recession | Scientific R2; manager Danger 1 | High (paper + managers) |
| Dry-down claimed without dual-composite magnitude | Scientific R7/V3; upstream may not supply both composites | High |
| Interpretive claims in indicative mood | Scientific R1; F-5 | High for paper; medium for software |
| HY/zonation novelty collision with Tayer | Scientific R4/V8 | High for paper |
| LPI/MESH both kept without r-gate | Scientific V2 | Medium (software OK; paper not) |
| Width as morphology at 30 m without floor guard | Scientific R3/F-9 | Medium |
| Buffer-sensitive `A_total` undocumented | Scientific R6 | Medium |
| NNI retained and visible | Scientific R5; manager Danger 7 | Reputational |

Software can ship guarded hypotheses. A methods paper cannot. Conflating those bars is the recurring failure mode across audits.

### 1.4 What will fail computationally?

| Failure | Evidence | Severity |
|---|---|---|
| Spatial `apply_ufunc` core dims force whole-section chunks | `dask_cuda_audit.md` executive finding #1 | Blocks “Dask-first” claim |
| Nested `np.asarray` / `.values.item()` inside delayed | Same, #2; triage #12 | Worker blocking / repeated work |
| Client Pandas concat of all patches | Same, #3; memory table ~419 GiB logical for 500×10k | OOM on real catchments |
| Three eager validity `.compute()` barriers | Preprocess path | Read amplification |
| CUDA promised for skeleton/igraph | Adversarial Dask audit overclaims; careful audit forbids | False advertising |
| Per-label full-raster skeleton scans | `skeletonize_label` loop | Fragmentation death spiral |
| Export DAG recomputes metrics | Second `compute()` | Wasted wall time |

Honest computational v1.2: Dask-first for **I/O, validity, compositing, occurrence reductions**; CPU component crops for morphology; CUDA deferred to optional pixelwise tranche only.

### 1.5 What will confuse water managers?

From `manager_interpretation_audit.md`, ranked:

1. Dry-down as flow/recession / “days until dry”
2. Pool width as depth/storage
3. Refuge as permanent protected-area designation
4. Single-month APSEC as trend
5. Low-valid periods as confirmed dry
6. Cross-reach N/MESH comparisons across sensors
7. Any NNI appearance
8. RC vs DCI vs TCF shown as interchangeable “connectivity %”
9. Composite method invisible while changing dry-down
10. False precision in tables

Manager docs written before real Gilbert ranges, or written more confidently than §6.18 allows, will institutionalise these misreadings.

### 1.6 What will break existing users?

| Break | Who hurts | Mitigation |
|---|---|---|
| Drop PF/PLF/AWMPA/AWMPL/AWMPW from canonical output | Anyone with legacy pipelines/tests | Explicit legacy adapter behind flag; never mix into v1.2 table |
| Wide CSV → tidy long | Spreadsheet users, existing notebooks | Document migration; optional wide export of *new* metrics only |
| `calculate_metrics(da_wmask, …)` signature gains valid layer / config | Call sites | Thin deprecated shim returning legacy schema, or hard break with clear error |
| Package rename `ecofragments` → `hydrofragments` | All imports | Last, not first; temporary re-export |
| Sentinel/validity behaviour change | Anyone feeding uint8 masks | Versioned behaviour + loud changelog |
| Broken regression fixture already fails | Contributors | Fix path immediately as hygiene, not as v1.2 science |

Pretending compatibility while changing denominators is worse than a versioned break.

### 1.7 What would future journal / software reviewers reject?

**Methods paper (Reviewer 2):**

- “This is Tayer 2025/2026 software + metric appendix on Gilbert” (scientific O1)
- Dry-down dressed as recession (O2)
- Asserted orthogonality / refuge-risk / composite bias without figures (O7, V1–V3)
- RC as renamed DCI (O8)
- NNI on a river (O4)
- Overclaiming AWMPA as circular when it is redundant (O10)

**Software / JOSS-shaped review (even if JOSS deferred):**

- README that cannot install/run (`docs_audit.md` critical)
- Three package identities
- “Dask-first / CUDA-ready” claims while nested computes and whole-section gufuncs remain
- No LICENSE / broken tests
- One-commit local history vs six-month open development claim (`spec_compliance.md` F13)
- Docs advertising dropped metrics

### 1.8 What assumptions need evidence?

| Assumption | Needed evidence | Blocker for |
|---|---|---|
| Upstream `observed` is the correct valid denominator | Written contract + example Zarr + unit tests | Occurrence/RA |
| Dual-composite dry-down is computable from available inputs | Either raw sub-monthly stack or both monthly products from upstream | Core dry-down |
| Persistence-based HY ≠ Tayer rainfall HY | Algorithm note + Gilbert agreement table | Novelty / HY module |
| Drainage layer exists for target catchments | Schema + one real file | Zone 1, LPSEC quality, gaps, RC |
| Equal-area default EPSG:3577 acceptable outside AU | Config story + per-pixel area path tested | Non-AU users |
| Component-crop morphology matches current whole-image kernels | Parity tests on adversarial shapes | Dask morphology refactor |
| LPI/MESH both worth keeping | r-gate on real data | Paper keep-both claim |
| `max_water` bias magnitude | Dual-composite F4 | Paper headline finding |
| Config hash stable across machines | Golden hash tests with path exclusion | Reproducibility claim |
| Bundled `wmask_ts.nc` sufficient for contract tests | Inspect dims/CRS/time (currently unverified in audit envs) | Test design |

### 1.9 Which planned features should be deferred?

**Defer from minimal credible v1.2 (ship later):**

- CUDA / CuPy / cuCIM / cuGraph (keep interface stub only if cheap; default `accelerator="none"`)
- Full distributed cross-chunk morphology rewrite (do bounded CPU crops first)
- DCI implementation (cite + position; benchmark later)
- TCF / RC / reconnection timing / refuge stability (need HY + fixed graph)
- Inter-pool gap as required output (needs real channel model)
- Zones 1–4 full product (optional later; do not block core AOI-wide metrics)
- Persistence-based HY detection until algorithm locked and differentiated
- NNI (cut, do not demote-and-carry)
- MESH until LPI/MESH gate runs (or ship MESH as secondary with “unvalidated redundancy” flag)
- Pool width distribution as manager-facing output (optional secondary with floor guard)
- Vector shapefile export redesign
- Rebrand to `hydrofragments` (after API freeze)
- JOSS / Zenodo / six-month history packaging
- Finished `docs/for-managers.md` with real numbers (structure/glossary OK; numbers wait)
- Probabilistic mask thresholding polish beyond recording threshold metadata

**Do not defer (blocking science):**

- Valid-observation contract + sentinel decode
- Occurrence denominator + `min_valid_obs`
- Equal-area / AOI co-reprojection
- Drop circular/redundant metrics from canonical output
- Tidy long schema + config hash for what *does* ship
- Core AOI-wide metrics that do not need HY/drainage: occurrence, RA, APSEC, N, LPI, AWRe (and LPSEC only with honest `L_ref` / proxy flag)
- Contract tests replacing legacy schema assertions
- Honest README / quarantine wrong docs

### 1.10 What is the smallest credible v1.2?

See §5. Short form:

> Source-agnostic monthly (or pre-composited) water + valid cube → equal-area AOI metrics → non-circular core register → tidy long output with metadata/flags → CPU-correct Dask reductions → tests and honest docs.  
> No CUDA theatre. No connectivity module. No HY theatre until algorithm locked. No manager decision tables with invented numbers.

---

## 2. Top 10 adversarial objections

1. **Plan ships a second product (scalable GPU morphology) inside the first product (correct metrics).** Result: neither ships. Split milestones hard.

2. **Headline metric (dry-down) is not known to be computable from the stated upstream.** WaterMask-TSFill is monthly; dual-composite requirement may be fantasy without Q3 locked. Do not call dry-down “core” until input path exists.

3. **Open questions Q1–Q9 are treated as implementation details.** They are the implementation. Coding before answers recreates today’s sentinel disaster in a new module tree.

4. **“Dask-first” will be claimed while spatial morphology remains whole-section CPU.** Reviewers and users will treat that as false advertising. Scope the claim to stages that are actually lazy.

5. **`dask_cuda_audit_adversarial.md` is a liability if followed.** It proposes cuCIM skeletonize and cuGraph as if parity were free. Prefer the careful `dask_cuda_audit.md`. Mark the adversarial CUDA section non-normative.

6. **Rebrand-first sequencing in the planning doc is wrong.** Docs and compliance audits are right: stop bleeding with honesty banners; rename after API freeze.

7. **Legacy tests currently enforce the wrong science.** Extending `process_metrics` in place while keeping PF/PLF columns “for compatibility” will poison the v1.2 contract forever.

8. **Manager guide and paper novelty work are being scheduled as if they were the same phase as denominator fixes.** They are not. Wrong confidence in manager docs is an operational hazard, not a docs nicety.

9. **Novelty story still leans on HY detection and zonation without differentiation from Tayer 2023c/2025–2026.** Software can omit those modules; a paper that claims them without V8 will be desk-rejected.

10. **Smallest credible v1.2 is not “spec §5.1 Core set entire.”** Spec core includes dry-down, which pulls HY + dual composite + median path. That is a second milestone. Minimal shippable core is the fixed-denominator spatial register plus occurrence/RA, with dry-down gated on Q3/Q7.

---

## 3. Required plan changes

Stage 8 `implementation_plan.md` must change relative to `audit_implementation_plan.md` Recommended Order as follows:

1. **Insert Decision Gate 0** before any code milestone: close conflict register (§0) and `spec_compliance.md` Q1–Q9 in a short ADR or decision table committed under `docs/audit/`.

2. **Invert rebrand:** move namespace rename to after API/schema freeze. Immediate docs work = honesty banner + fix broken imports/paths only (`docs_audit.md` Tier 1).

3. **Split Milestone A (contracts) from Milestone B (core spatial metrics) from Milestone C (dynamics) from Milestone D (connectivity) from Milestone E (scale/CUDA).** No parallel “do all the modules” fantasy.

4. **Demote CUDA** to optional post-v1.2 tranche; Milestone E = remove nested computes + monthly Zarr checkpoint + chunk contracts, not GPU.

5. **Gate dry-down** on explicit dual-composite input availability. If unavailable, ship APSEC time series + documented “dry-down deferred” rather than a fake single-composite slope labelled as the headline metric.

6. **Gate LPSEC / gap / Zone 1** on drainage contract. Without drainage: skip or proxy-flag; do not invent Zone 1.

7. **Cut NNI from v1 plan** unless someone accepts explicit maintenance cost; default = cut.

8. **Legacy adapter policy:** canonical output = tidy v1.2 only; legacy wide CSV only via `legacy_output=True` emitting *legacy* metric names, never a hybrid.

9. **Test-first contract suite** before metric expansion: sentinels, grid align, occurrence denominator, CRS/AOI order, N=0/1 flags, schema/hash. Delete or quarantine assertions that require PF/PLF.

10. **Treat `dask_cuda_audit.md` as normative for compute; treat `dask_cuda_audit_adversarial.md` as stress input only** where it contradicts parity constraints.

11. **Manager docs:** glossary + negative scope early; decision table / narratives only after one real validation run. No invented thresholds.

12. **Publication / JOSS items:** remain explicitly out of the v1.2 critical path (already stated; enforce in checklist by deletion from “must”).

---

## 4. Deferrals recommended

| Item | Defer to | Condition to revive |
|---|---|---|
| CUDA tranche 1 (pixelwise) | post-minimal v1.2 | CPU monthly path correct; benchmarks exist |
| CUDA morphology / cuGraph | much later / never by default | Scientific parity suite green |
| Distributed cross-chunk labels at scale | after component-crop CPU path | Chunk-boundary label tests |
| HY detection module | after Q7 + V8 note | Algorithm differentiated from Tayer |
| Dry-down rate | after Q3 dual-composite path | Both composites or raw observations available |
| Zones product | after occurrence + drainage decisions | Circularity guards tested |
| Inter-pool gap, reconnection, refuge stability | after HY + channel model | Real drainage or validated skeleton |
| RC / TCF | after fixed-node graph design | Edge rule documented; RC≠DCI story written |
| DCI implementation | after citation-only docs | `riverconn`/Conefor benchmark |
| MESH keep-both | after V2 gate | r ≤ 0.9 or drop |
| NNI | cut | Only if planar non-river mode ever in scope |
| Width distribution manager surfacing | after F-9 floor guard + V5 | Resolution sensitivity shown |
| Package rename | after API freeze | Single import story |
| Full `for-managers.md` numbers | after Gilbert (or other) run | Real ranges only |
| JOSS / Zenodo / history graft | future | Stable API + demonstrated §6.18 rows |

---

## 5. Non-negotiable implementation gates

No merge to a “v1.2” branch/release without all of the following green:

### G0 — Decisions
- [ ] Conflict register (§0) closed in writing
- [ ] Q1–Q9 answered with owners and defaults

### G1 — Scientific correctness (software-blocking)
- [ ] Canonical WaterMask-TSFill Dataset/Zarr loads without manual reshape
- [ ] Sentinels `254`/`255` decoded before any signed cast; never counted as dry/water
- [ ] Aligned `valid_obs` required; mismatch raises
- [ ] Occurrence = `water_obs/valid_obs` with `min_valid_obs` floor
- [ ] PF/PLF/AWMPA/AWMPL/AWMPW absent from canonical v1.2 output
- [ ] Equal-area (or per-pixel area) path; AOI and raster co-reprojected before metrics
- [ ] N=0/1 edge semantics emit NaN + flags, not silent zeros for shape metrics

### G2 — Schema / reproducibility
- [ ] Tidy long table with required metadata columns for shipped metrics
- [ ] `config_hash` stable under locked hashing rules
- [ ] Composite method recorded when compositing occurs; comparison across composites refused or flagged

### G3 — Tests
- [ ] Contract tests for G1/G2 (synthetic fixtures)
- [ ] Legacy regression either fixed-and-scoped to kernels or retired; must not require dropped metrics
- [ ] CPU-only install passes without GPU extras

### G4 — Compute honesty
- [ ] No nested `compute` / `np.asarray` on Dask collections inside delayed metric tasks for new code paths
- [ ] Docs do not claim full-pipeline CUDA or out-of-core spatial morphology unless true
- [ ] Monthly/validity reductions are lazy with one explicit materialisation boundary

### G5 — Docs honesty
- [ ] README installs and imports the real package
- [ ] Status banner: migration state explicit
- [ ] `architecture.md` river-only (or clearly deprecated)
- [ ] No manager/paper indicative claims beyond §6.18 demonstrated rows

### G6 — Dynamics / connectivity (only if those modules are in the release)
- [ ] Dry-down only if dual-composite path real; rename to extent-contraction language
- [ ] HY only if algorithm locked + Tayer differentiation written
- [ ] Connectivity only with DCI/PC/IIC positioning and RC-vs-DCI statement
- [ ] NNI absent from release artifacts

**Failure of G1–G5 = no-go for calling it HydroFragments v1.2.**  
**Failure of G6 = those modules stay out; release can still be minimal v1.2.**

---

## 6. Minimal credible v1.2 scope

### In scope (ship)

**Identity / packaging (honesty, not full rebrand)**
- Document current importable name; banner that v1.2 migration is in progress
- Fix broken regression path; LICENSE decision tracked but not blocking science

**Contracts**
- `(water, valid_obs[, provenance])` input + WaterMask-TSFill adapter
- Sentinel map; grid/CRS/shape equality; configured equal-area default EPSG:3577 or per-pixel areas
- Typed config + `config_hash` + recorded thresholds/`min_patch_pixels`/`connectivity_rule`
- Monthly cadence validation; compositing **if and only if** sub-monthly input present; if input already monthly, require `monthly_composite` provenance metadata and do not invent a second composite

**Metrics (AOI-wide, fixed denominators)**
- Occurrence frequency raster + table summary
- Refuge Area (threshold recorded)
- APSEC
- N (default `min_patch_pixels=3`, `connectivity_rule` recorded)
- LPI
- AWRe with `awre_length_method` recorded (skeleton vs major-axis fallback)
- AWMSI (secondary OK if cheap; hypothesis-mood docs)
- LPSEC **only** with real `L_ref` or explicit `proxy_channel` flag

**Output**
- Tidy long metrics table
- Occurrence / valid-count rasters
- Edge/low-valid flags for shipped metrics
- Optional legacy wide adapter for old metric names only

**Compute**
- Dask reductions for validity/occurrence/composites
- Reuse existing label/skeleton/EDT kernels behind a patch table, without claiming spatial out-of-core
- Remove nested computes on the new path

**Tests / docs**
- Synthetic contract suite
- README + `docs/input_format.md` matching *implemented* contracts
- Metric register doc: Core shipped / Deferred clearly split
- Manager glossary stub + “what this tool does not measure” (no fake decision thresholds)

### Out of scope for minimal v1.2

- Dry-down as required core (unless Q3 satisfied in same release)
- HY anchors / refuge stability / reconnection
- Zones 1–4 product
- Inter-pool gap, RC, TCF, DCI implementation
- NNI
- CUDA
- Full Dask distributed morphology
- Finished manager decision narratives with numbers
- JOSS / paper results / history graft
- Premature `hydrofragments` rename

### Naming the release honestly

Call it **HydroFragments v1.2.0-contracts+core** (or similar) if dry-down/HY/connectivity are absent. Do not label a contracts-only release as full spec compliance. Spec §5.1 Core including dry-down remains the *target*; minimal ship is an explicit subset with a public deferral list.

---

## 7. Final go / no-go recommendation

### NO-GO

- Starting code from the full Recommended Order in `audit_implementation_plan.md` as written
- Treating all audit “absent” rows as one implementation sprint
- Following optimistic CUDA morphology advice as normative
- Writing finished v1.2 manager/API docs that describe unbuilt behaviour
- Claiming v1.2 complete when occurrence denominator, sentinels, or CRS/AOI ordering remain wrong

### CONDITIONAL GO

Proceed to Stage 8 implementation planning (still no production metric code until plan approved) **only if** the plan:

1. Opens with locked decisions for §0 conflicts and Q1–Q9  
2. Adopts the minimal credible scope in §6 as Milestone 1 release definition  
3. Places dry-down / HY / zones / connectivity / CUDA on explicitly gated later milestones  
4. Uses `dask_cuda_audit.md` (not the adversarial CUDA optimism) for compute sequencing  
5. Sequences docs honesty before docs completeness; rebrand last  
6. Makes G0–G5 non-negotiable gates for any branch named v1.2  

### Recommended immediate next action (still no source edits beyond audit docs)

1. Write `docs/audit/decisions.md` (or equivalent) closing Q1–Q9 + conflict register.  
2. Then Stage 8: `docs/audit/implementation_plan.md` scoped to §6.  
3. Then Stage 9 checklist.  
4. Only then touch `ecofragments/` — starting with input contract + tests, not metrics fashion or rename.

---

## 8. One-line maintainer summary

**Audits are strong; the combined plan is too big; dry-down/CUDA/connectivity/rebrand are the main schedule killers; lock contracts and ship a smaller non-circular core first, or expect another year of elegant non-compliance.**
