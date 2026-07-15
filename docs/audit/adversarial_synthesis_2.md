# HydroFragments v1.2 — Adversarial Synthesis (Second Pass)

**Date:** 2026-07-10
**Stance:** adversarial principal investigator + Reviewer 2 + senior maintainer, second pass
**Constraint:** diagnosis and plan stress-test only; no source files edited
**Verdict:** **NO-GO on the full planned migration** (concurs with `adversarial_synthesis.md`). **Second-pass finding: the first synthesis's own "minimal credible v1.2" is still over-scoped and still gated on unresolved contract questions it treats as closed.** Conditional GO only on a *contracts-and-reductions-first* release even smaller than §6 of the first synthesis, plus three time-sensitive actions the prior audits defer past the point where they can still be done.

> **Mandatory next-phase intake gate**
>
> Before writing `implementation_plan.md` (Stage 8) or editing code, the next phase must:
>
> 1. Enumerate every file under `docs/audit/` present at phase start.
> 2. Read each file in full (not summaries), including this report and `adversarial_synthesis.md`.
> 3. Record an intake manifest: filenames + unresolved cross-audit conflicts.
> 4. Stop if conflicts remain on input validity, compositing ownership, metric semantics, or implementation order.
>
> Prior audits ingested for this second-pass synthesis (order):
>
> 1. [`evidence_packet.md`](evidence_packet.md)
> 2. [`repo_triage.md`](repo_triage.md)
> 3. [`spec_compliance.md`](spec_compliance.md)
> 4. [`scientific_metrics_audit.md`](scientific_metrics_audit.md)
> 5. [`dask_cuda_audit.md`](dask_cuda_audit.md) and [`dask_cuda_audit_adversarial.md`](dask_cuda_audit_adversarial.md)
> 6. [`manager_interpretation_audit.md`](manager_interpretation_audit.md)
> 7. [`docs_audit.md`](docs_audit.md)
> 8. [`adversarial_synthesis.md`](adversarial_synthesis.md) — the first synthesis, now itself an audit target
> 9. Planning context: [`../audit_implementation_plan.md`](../audit_implementation_plan.md)

---

## 0. Why a second pass exists

The first synthesis (`adversarial_synthesis.md`) is strong and its verdict is correct: split the milestones, defer CUDA/connectivity/HY/rebrand, ship a non-circular core. This report does **not** re-argue that. It does three things the first pass did not do to itself:

1. **Attacks the first synthesis's own conclusions**, on the principle that a synthesis produced inside the same auto-staged LLM pipeline as the audits it summarises inherits their shared blind spots. The most dangerous assumptions are the ones *every* report agrees on without any one report having independently verified them.
2. **Finds the load-bearing facts that are still unverified across all eight prior reports** — not disagreements between audits, but things they all *assume* and none *checked*.
3. **Separates time-reversible from time-irreversible risk.** The first synthesis ranks by scope and severity. It does not flag that a small number of items get *permanently* harder the longer they wait — those must move to the front regardless of scope.

Bottom line up front: the first synthesis's minimal scope is still one milestone too big, its "publishable without data" claim for the paper spine is softer than it states, and at least three items it defers are the kind that cannot be un-deferred.

---

## 1. Attacking the first synthesis

### A1 — The "minimal credible v1.2" in §6 is still gated on Q1/Q3/Q6 and still ships partly-blind metrics

`adversarial_synthesis.md` §6 puts in the minimal ship list: occurrence, RA, APSEC, N, LPI, AWRe, AWMSI, and **LPSEC "only with real `L_ref` or explicit `proxy_channel` flag."** That list is internally inconsistent with its own §0 conflict register:

- **Occurrence and RA cannot be finalised until Q1 (validity semantics) is answered.** The denominator is `valid_obs` — but *what counts as a valid observation* (native `observed=True` only, vs. resolved/filled pixels under a confidence rule) is explicitly unlocked (Q1). Shipping occurrence "in minimal v1.2" means either (a) locking Q1 first, or (b) shipping a denominator that may be redefined in v1.3, breaking every occurrence-derived number and every refuge boundary drawn from it. The first synthesis lists Q1 as blocking in §1.2 but then lists occurrence as shippable in §6. Those cannot both be true. **Occurrence is only minimal-shippable *after* Q1 is closed in writing — it is not a free "do not defer" item.**
- **AWRe with `awre_length_method` recorded still requires the drainage decision (Q6).** The method switch is skeleton-length *if a channel exists*, major-axis otherwise. With no drainage contract (Q6 open), every AOI takes the fallback branch — so "AWRe with method recorded" ships as "AWRe, always major-axis, method column always says fallback." That is a defensible minimal metric, but the first synthesis presents it as if the skeleton branch were available. It is not, until Q6.
- **LPSEC with a `proxy_channel` flag is the single weakest inclusion.** `L_ref` is the drainage centreline length (Q6, open). The "proxy" is the maximum-wet skeleton — which is *derived from the water*, i.e. it reintroduces exactly the denominator-shares-numerator-features pattern the whole paper spine condemns for PF/PLF. A proxy `L_ref` built from wet extent is a soft circularity. Flagging it does not make it non-circular; it makes it *disclosed*. Shipping LPSEC-by-proxy in the flagship "non-circular core" release is an own-goal a hostile reviewer will enjoy. **Recommendation: LPSEC leaves the minimal ship entirely until a real drainage `L_ref` exists.** APSEC (area, fixed AOI polygon denominator — genuinely non-circular) carries the coverage story alone in v1.2.0.

**Net:** the truly non-circular, contract-closable-without-drainage minimal set is **occurrence (post-Q1), RA (post-Q1), APSEC, N, LPI, AWRe (fallback-only), AWMSI**. LPSEC and anything skeleton/channel-dependent go to the next milestone. This is smaller than §6.

### A2 — The paper spine ("publishable without data") is thinner than the first synthesis asserts

Both the scientific audit and the first synthesis lean hard on one comforting claim: the fixed-denominator / circularity argument is a *logical/algebraic* result, therefore publishable as reasoning without any dataset (scientific audit §16.2; synthesis §1.3 "software can ship guarded hypotheses"). Pressure-test that:

- A methods paper whose sole data-free contribution is "we removed five metrics because their denominators are circular, and replaced them with unweighted distributions" is a **Technical Note / Short Communication**, not a full methods paper, at most journals. The circularity argument is correct and worth stating — but "we identified that PF = N/ΣA is circular" is a paragraph, not a paper. Reviewer 2's O1 ("where is the novelty beyond Tayer 2025/2026?") is *not* answered by the circularity argument alone; it is answered by the circularity argument **plus** demonstrated non-redundancy of the replacements on real data (V1/V2) **plus** the composite-sensitivity finding (V3). Strip the data-dependent items and the remaining data-free spine may not clear the bar for a standalone paper at the target venue.
- Therefore the first synthesis's clean split — "ship guarded software now, validate for paper later" — quietly concedes that **the paper is blocked on Gilbert validation for its actual novelty**, not just its polish. That is fine for software but it means the publication timeline is bounded by validation-data work (V1–V3 minimum), and no amount of software milestone completion advances it. This should be stated plainly in the plan so no one schedules "write the paper" as a near-term deliverable.

### A3 — The first synthesis under-weights that dry-down failure poisons the *paper*, not just a metric

The first synthesis (§1.3, objection #2) correctly flags dry-down may be uncomputable if WaterMask-TSFill only emits one monthly composite. It files this as "high" severity and moves dry-down to a later milestone. But it misses the knock-on: **the composite-sensitivity result (V3) is one of only three legs of the paper's data-backed novelty, and V3 is *defined by* running dry-down under both `max_water` and `median`.** If the upstream cannot supply both composites (or raw sub-monthly observations to build them), then:

- dry-down is deferred (already noted), **and**
- V3 is not just deferred but *impossible from this data source*, **and**
- the paper loses one of three novelty legs, leaving circularity (thin, per A2) + non-redundancy (V1/V2).

So Q3 (monthly ownership) is not merely a "which milestone does dry-down land in" question. It is a **go/no-go input for whether the methods paper has enough data-backed novelty to exist in its current framing.** The first synthesis treats Q3 as a metric-scheduling issue. It is a publication-viability issue. This must be escalated: *before* committing to the paper's novelty framing, confirm whether both composites (or raw observations) are obtainable for the validation catchment.

### A4 — The conflict register (§0) is correct but has no owner or forcing function

`adversarial_synthesis.md` §0 lists ten cross-audit conflicts and says "Stage 8 must open with a decision table that closes every row." Good — but there is no mechanism that *prevents* Stage 8 from opening with hand-waving. The failure mode is predictable: an implementation planner writes "Decision: validity semantics = native observed (recommended)" for Q1 with no evidence, and the whole tree of occurrence/RA/zones inherits an unverified default dressed as a decision. **A decision table with default recommendations and no evidence column is how the original sentinel disaster happened in the first place** (someone assumed uint8 masks were safe to cast). The decisions doc must carry, per row: the decision, the *evidence* it rests on (a file, a contract, a test), and what breaks if the decision is wrong. No evidence column = not a decision, a guess.

### A5 — The first synthesis accepts every audit's framing of the upstream contract without anyone re-deriving `observed` semantics

All reports cite `WaterMask-TSFill/watermask_tsfill/contracts.py` and agree: `water_mask` 0/1/254/255, plus `confidence`, `method_flag`, `observed`. The evidence packet (§5) reads `observed` as "True if the pixel was natively observed." Every downstream audit then assumes `observed` is the correct occurrence denominator. **But no report verifies that `observed=True` pixels are the scientifically correct denominator versus, say, `method_flag==0` (observed) plus high-confidence temporal fills.** The spec itself leaves this open (Q1). This is the highest-leverage unverified assumption in the entire audit set: it silently determines occurrence, RA, refuge boundaries, zones, and every per-pixel temporal metric, and it was inherited identically through eight reports without independent challenge. Second-pass flag: **treat `observed`-as-denominator as an unverified hypothesis, not a documented fact, until confirmed against the upstream author's intent and a real Zarr sample.**

---

## 2. Load-bearing facts still unverified across all prior reports

These are not conflicts between audits. They are things every report assumes and none checked. Each one can invalidate a milestone if wrong.

| # | Unverified fact | Assumed by | What breaks if wrong |
|---|---|---|---|
| U1 | `tests/wmask_ts.nc` contains enough temporal/spatial variability to serve as a contract/regression fixture | repo_triage Q, spec_compliance F4/Q8, dask_cuda B1, every "synthetic + fixture test" plan | The entire contract-test-first strategy rests on a fixture no audit could open (NetCDF backend missing in audit envs). If it is 63 near-identical timesteps, it validates nothing about dry-down, stability, or occurrence-denominator behaviour. |
| U2 | `observed=True` is the correct valid-observation denominator | evidence_packet §5; all downstream | Occurrence/RA/zones all shift (see A5). |
| U3 | WaterMask-TSFill can supply both composites or raw sub-monthly observations | dry-down as core; V3 as paper finding | Dry-down and the composite-bias paper leg both die (see A3). |
| U4 | A drainage/centreline dataset exists for the validation catchment(s) with usable topology | Zone 1, `L_ref`, LPSEC, gap, RC, TCF, DCI | Half the secondary+connectivity register has no input; LPSEC ships as soft-circular proxy (A1). |
| U5 | EPSG:3577 equal-area is acceptable and upstream data is already in it | CRS guard "reproject only if needed" | Non-AU deployments and any non-3577 upstream need the per-pixel-area path *built*, not just documented — more work than "document the caveat." |
| U6 | Current label/skeleton/EDT kernels are numerically reusable behind a new patch table without re-derivation | spec_compliance "reuse kernels"; synthesis §6 "reuse existing kernels" | If the kernels bake in the wrong sentinel/validity handling (they do — B2), "reuse" means porting a bug. Reuse must be *after* the input contract is fixed, or the kernels inherit the corruption. |
| U7 | The bundled regression reference (`irm_metrics.csv`) is scientifically valid output, not itself a legacy artifact of the wrong denominators | spec_compliance F4 "regression as sanity" | If the reference CSV was produced by the naive-persistence / total-timestep-denominator code, regressing against it *locks in the bug* as the expected value. A "sanity regression" against wrong numbers is worse than none. |

U1, U2, U3, U4 are the four that can each independently sink a milestone. **None can be closed by more LLM analysis — each needs someone to open a file, run a script, or ask the upstream author.** The plan must front-load these four as evidence-gathering tasks *before* Decision Gate 0, not inside it.

---

## 3. Time-irreversible risk (front-load regardless of scope)

The first synthesis ranks by scope × severity. That ordering misses that some items get permanently harder with delay. These jump the queue:

1. **Six-month public-development history (F13).** Local history is one commit, "Initial commit: ecofragments package (clean start)," dated 2026-05-30. A publication narrative claiming preserved iRiverMetrics lineage and open development **cannot be manufactured retroactively** — every day that passes without a real public commit trail is a day that can never be recovered, and a "clean start" commit actively *destroyed* the graftable predecessor history if it squashed it. This is not a docs task to defer to Tier 3; it is a governance decision with a **shrinking window**. Whether or not the paper needs it, the maintainer must decide *now* whether to graft/restore predecessor history, because the option expires by neglect. First synthesis correctly notes it is "irrelevant to code" — but wrong to imply it can therefore wait. It is the *most* time-sensitive item in the whole audit and the only one that is strictly worse tomorrow.
2. **Deciding `observed`-denominator semantics (U2/Q1) before any occurrence code exists.** Not because it is urgent in calendar terms, but because every metric built on the wrong denominator must be *rebuilt*, and every downstream test *rewritten*. The cost of getting it wrong compounds with each metric added on top. Cheapest to fix at zero metrics built; catastrophic at twelve.
3. **The reference-output validity question (U7).** If the regression baseline is contaminated by the old denominators, that must be established *before* anyone writes a "regression must stay within 5%" test, or the test suite will enforce the bug in perpetuity and every future correct change will read as a regression failure.

Everything else in the first synthesis's defer list can genuinely wait. These three cannot.

---

## 4. Where the first synthesis is right and should not be softened

To keep this report honest and non-duplicative, the following first-synthesis conclusions are correct, load-bearing, and this pass explicitly endorses them without change:

- Split the three products (correct-metrics library / scalable engine / manager comms). Do not co-ship.
- CUDA is a post-v1.2 optional tranche; `dask_cuda_audit.md` is normative, `dask_cuda_audit_adversarial.md` is stress-input-only. cuCIM skeletonize / cuGraph BFS are not free parity.
- Rebrand last, after API freeze; honesty banner now.
- Canonical output = tidy long v1.2 only; legacy wide CSV only behind a flag emitting *legacy* names, never a hybrid schema.
- Cut NNI from v1 outright; never surface it to managers.
- Sentinel decode before any signed cast; `254`/`255` never counted as dry/water. (This is the clearest, most certain fix in the whole audit set.)
- Contract-test-first, before metric expansion.
- G0–G5 as non-negotiable gates.

None of §1–§3 above weakens these. They tighten scope *within* them.

---

## 5. Answers to the ten attack questions (second-pass deltas only)

Only where this pass adds to or corrects `adversarial_synthesis.md` §1.

1. **Over-scoped?** Beyond the first synthesis's list: *the first synthesis's own minimal set is over-scoped* by including LPSEC-by-proxy and by listing occurrence/AWRe as if their gating questions (Q1/Q6) were closed. Cut LPSEC; gate occurrence on Q1.
2. **Under-specified?** The decisions doc (§0 register) has no evidence column and no owner — under-specified *process*, not just under-specified contracts (A4).
3. **Fail scientifically?** New: LPSEC-by-proxy reintroduces soft circularity into the anti-circularity flagship (A1). And U7 — regressing against a possibly-contaminated baseline locks in the denominator bug.
4. **Fail computationally?** Concur with first synthesis; nothing to add. Add only: U6 — "reuse kernels" ports the sentinel bug unless sequenced after the input-contract fix.
5. **Confuse managers?** Concur. Add: shipping APSEC alone (without LPSEC) in v1.2.0 actually *reduces* manager confusion (one less length-vs-area caveat to hold) — a point in favour of A1's cut.
6. **Break existing users?** Concur. Add: if U7 holds and the baseline is contaminated, "backward-compatible regression" is impossible by definition — you cannot be compatible with wrong numbers and correct at once. Choose correct; document the break.
7. **Reviewers reject?** New and sharper: the data-free paper spine may be a Technical Note, not a methods paper (A2). The novelty rests on V1/V2/V3, all data-gated, and V3 may be *impossible* from a single-composite upstream (A3). The paper's viability is a Q3/U3 question, not a writing question.
8. **Assumptions needing evidence?** The four sink-a-milestone unknowns U1–U4, none closable by analysis (§2). Plus U2 (`observed` denominator) as the single highest-leverage inherited-unverified assumption.
9. **Defer?** Everything the first synthesis defers, plus LPSEC. **Un-defer** (pull forward) the three time-irreversible items in §3 — history, denominator decision, baseline-validity check.
10. **Smallest credible v1.2?** Smaller than first synthesis §6. See §7.

---

## 6. Top 10 adversarial objections (second pass)

Ordered by how much they change the plan relative to `adversarial_synthesis.md`.

1. **The "non-circular core" ships a soft-circular metric.** LPSEC-by-proxy uses a wet-derived skeleton as `L_ref`. Cut it or the flagship release contradicts its own thesis.
2. **Occurrence is listed as shippable while its denominator is undecided.** Q1/U2 must close, with evidence, before occurrence is anything but provisional. A metric whose denominator may change is not shippable in a release whose whole point is correct denominators.
3. **The paper's novelty is data-gated, and one leg may be impossible.** V3/composite-sensitivity requires two composites; a monthly-only upstream kills it. Confirm U3/Q3 before framing the paper. (Escalated from first synthesis's "high" to "publication-viability.")
4. **The decisions doc will launder guesses as decisions unless it carries an evidence column.** The sentinel disaster was a guess dressed as a safe assumption. Require evidence per decision row.
5. **Four sink-a-milestone facts (U1–U4) are unverified and unverifiable by analysis.** Front-load them as file-opening / author-asking tasks before Decision Gate 0. No more LLM passes will close them.
6. **The regression baseline may enforce the bug.** If `irm_metrics.csv` came from the naive-denominator code, a "stay within 5%" test canonises wrong numbers. Establish baseline provenance before writing regression gates. (U7.)
7. **Predecessor history is decaying now.** One-commit "clean start" may have already severed graftable lineage; every day narrows the window. Governance decision required immediately, independent of code. (§3.1.)
8. **"Reuse existing kernels" ports the sentinel/validity bug** unless kernel reuse is strictly sequenced after the input-contract fix. (U6.)
9. **Equal-area is treated as a doc caveat but is partly a build task.** Non-3577 / non-AU inputs need the per-pixel-area path *implemented*, not just a documented caveat. (U5.)
10. **The audit set shares a single point of failure: it was produced by one staged LLM pipeline cross-citing itself.** Eight reports agreeing is not eight independent confirmations. The `observed`-denominator assumption (U2) is the clearest inherited-without-check example. Independent human/upstream verification of U1–U4 is worth more than any further audit stage.

---

## 7. Minimal credible v1.2 scope (second pass — tighter than §6 of the first synthesis)

### Pre-conditions (evidence gathering, before Decision Gate 0)

Close by opening files / running scripts / asking the upstream author — **not** by analysis:

- **U1:** open `tests/wmask_ts.nc`; record dims, CRS, date range, wet-fraction variability. Decide if it can be a contract fixture or if a synthetic set is mandatory.
- **U2 / Q1:** confirm with the WaterMask-TSFill author and a real Zarr sample whether `observed=True` (or `method_flag==0`) is the intended valid-observation denominator.
- **U3 / Q3:** confirm whether both monthly composites (or raw sub-monthly observations) are obtainable for the validation catchment. This gates dry-down *and* the paper's V3 leg.
- **U4 / Q6:** confirm whether a drainage centreline dataset exists for the target catchment(s) and its topology.
- **U7:** establish the provenance of `irm_metrics.csv` — was it produced by the naive-denominator code? If yes, retire it as a correctness baseline.

### Decision Gate 0

Close every row of `adversarial_synthesis.md` §0 conflict register **and** Q1–Q9, in a committed `docs/audit/decisions.md`, **each row carrying: decision | evidence it rests on | what breaks if wrong | owner.** No evidence column = not closed.

### v1.2.0 ship set (smaller than first synthesis §6)

**Contracts (blocking):**
- `(water, valid_obs[, provenance])` input + WaterMask-TSFill Zarr adapter; sentinel decode before dtype narrowing; grid/CRS/shape equality raises on mismatch.
- Configured equal-area (EPSG:3577 default) *or* per-pixel-area path — whichever U5 shows is needed for the actual data; do not ship an unbuilt per-pixel path as a "documented caveat."
- Typed config + `config_hash` (rules locked per Q9) + recorded thresholds / `min_patch_pixels` / `connectivity_rule`.
- Monthly cadence validation; composite *only if* sub-monthly input present; if already monthly, require `monthly_composite` provenance and invent no second composite.

**Metrics (AOI-wide, genuinely non-circular, no drainage needed):**
- Occurrence frequency — **only after Q1/U2 closed**; raster + table summary.
- Refuge Area — **only after Q1/U2 closed**; threshold recorded.
- APSEC (fixed AOI-polygon denominator).
- N (`min_patch_pixels=3` default, `connectivity_rule` recorded).
- LPI.
- AWRe — **fallback (major-axis) only** until Q6/U4 supplies a channel; `awre_length_method` recorded as fallback.
- AWMSI (secondary; hypothesis-mood docs).

**Explicitly out of v1.2.0** (moved *out* of the first synthesis's minimal set):
- **LPSEC** — until a real drainage `L_ref` exists (soft-circular by proxy otherwise).
- Everything the first synthesis already defers: dry-down, HY, zones, gap, RC, TCF, DCI, NNI, CUDA, distributed morphology, manager decision numbers, JOSS/paper results, rebrand.

**Output / compute / tests / docs:** as `adversarial_synthesis.md` §6 (tidy long table, occurrence/valid-count rasters, edge/low-valid flags, Dask reductions for validity/occurrence, kernel reuse *after* input-contract fix, synthetic contract suite, honest README + `input_format.md` matching *implemented* contracts, manager glossary stub + negative scope). No change; endorsed.

### Release naming

Call it **`v1.2.0-contracts+core`** — and, given A1, note explicitly in the release notes that LPSEC and all channel/HY/connectivity metrics are deferred, with the deferral list public. Do not label a contracts-plus-AOI-core release as full §5.1 spec compliance.

---

## 8. Non-negotiable implementation gates (delta over first synthesis)

Adopt `adversarial_synthesis.md` §5 (G0–G6) in full. **Add:**

### G0+ — Evidence before decisions
- [ ] U1–U4 and U7 closed by real file inspection / upstream confirmation, recorded with artifacts, **before** Decision Gate 0 opens.
- [ ] `decisions.md` carries an evidence column per row; any row without evidence blocks the gate.

### G1+ — Correctness (additions)
- [ ] LPSEC absent from v1.2.0 canonical output (no wet-derived proxy `L_ref` in the flagship release).
- [ ] Occurrence/RA not merged to a v1.2 branch until Q1/U2 is closed with evidence.
- [ ] Regression baseline confirmed produced by *correct* denominators, or retired; no "stay within X%" gate against a contaminated reference.

### G7 — Time-irreversible (new)
- [ ] Predecessor-history graft/restore decision made and executed (or explicitly, publicly abandoned) — not deferred to post-v1.2, because the option decays.

**Failure of G0+/G1+ = no-go for calling it HydroFragments v1.2.**
**Failure of G7 = a permanent, unrecoverable hole in any publication-lineage claim; decide with eyes open, now.**

---

## 9. Final go / no-go recommendation

### NO-GO (concurs with and extends `adversarial_synthesis.md`)
- Full Recommended Order from `audit_implementation_plan.md`.
- The first synthesis's own §6 minimal set *as written* — it still ships LPSEC-by-proxy and lists Q1/Q6-gated metrics as free.
- Any Decision Gate 0 whose decisions doc lacks an evidence column.
- Committing to the methods-paper novelty framing before U3/Q3 confirms the composite-sensitivity leg is even computable from the available data.
- Deferring the predecessor-history decision — that is a NO-GO on *waiting*, not on the code.

### CONDITIONAL GO
Proceed to Stage 8 planning (no production metric code until the plan is approved) **only if** the plan:

1. Front-loads U1–U4 + U7 as evidence-gathering tasks *before* Decision Gate 0.
2. Opens with a `decisions.md` closing the §0 conflict register + Q1–Q9, **with an evidence column and owners**.
3. Adopts the tighter §7 scope (LPSEC out; occurrence/RA gated on Q1) as the v1.2.0 definition.
4. Escalates Q3/U3 as a **publication-viability** decision, not a metric-scheduling one.
5. Executes or explicitly abandons the predecessor-history graft **now** (G7), independent of code progress.
6. Otherwise follows `adversarial_synthesis.md` §3 required plan changes and §5 gates unchanged.

### Recommended immediate next action (still no source edits beyond audit docs)
1. Open the four files / ask the one author that close U1–U4 (and check U7's baseline provenance). This is human/tool work, not another LLM audit stage.
2. Make the predecessor-history decision (G7) — time-sensitive, code-independent.
3. Write `docs/audit/decisions.md` with the evidence column, closing Q1–Q9 + the §0 register.
4. Then Stage 8 `implementation_plan.md`, scoped to §7 here (not §6 of the first synthesis).
5. Only then touch `ecofragments/` — input contract + contract tests first; occurrence only after Q1 is closed.

---

## 10. One-line maintainer summary

**The first synthesis is right that the plan is too big — but its own "minimal" set is still one milestone too big (cut LPSEC-by-proxy, gate occurrence on the validity decision), the paper's novelty is data-gated with one leg possibly impossible from a monthly-only upstream, and three things — the observation-denominator decision, the regression-baseline provenance, and the decaying predecessor-history window — must be settled with real evidence *now*, because no further audit pass and no amount of waiting can settle them.**
