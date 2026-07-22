# HydroFragments v1.2 — Adversarial Scientific Metrics Audit

**Audit date:** 2026-07-10
**Reviewer stance:** Adversarial but fair. I am reading this as a hostile Reviewer 2 for a methods/software paper in *Environmental Modelling & Software* or *J. Hydrology*, who has also read Tayer et al. (2023a, 2023b, 2023c, 2025/2026).
**Scope:** scientific defensibility of the metric foundations *before* implementation. No source files changed; no code written.
**Contract audited:** [`docs/HydroFragments_v1.2_spec.md`](../HydroFragments_v1.2_spec.md)
**Inputs ingested:** the v1.2 spec, [`evidence_packet.md`](evidence_packet.md), [`spec_compliance.md`](spec_compliance.md), [`repo_triage.md`](repo_triage.md), and the paper summaries in [`paper-summary.md`](../paper-summary.md).

> **Mandatory next-phase intake gate**
>
> The implementation phase must read the prior audit markdown files **before** doing its own work, in this order:
>
> 1. [`docs/audit/evidence_packet.md`](evidence_packet.md)
> 2. [`docs/audit/repo_triage.md`](repo_triage.md)
> 3. [`docs/audit/spec_compliance.md`](spec_compliance.md)
> 4. this report — [`docs/audit/scientific_metrics_audit.md`](scientific_metrics_audit.md)
>
> The first three cover *code/spec compliance*. This one covers *scientific defensibility* and is orthogonal to them: a metric can be 100% spec-compliant and still be scientifically indefensible for publication. Both gates must pass.

---

## 0. How to read this report

The spec is already unusually self-critical — it carries its own audit trail (`[AUDIT FIX]`, `JUDGEMENT CALL`) and a validation-status table (§6.18). That is a strength, and this review does **not** re-litigate what the spec already concedes. Instead it does three things a hostile reviewer would do that the spec does not fully do to itself:

1. **Separates "must fix before implementation" from "must validate before paper."** The spec mixes these. Some risks are cheap to fix in the design now; others are empirical claims that can only be settled with data and must not be asserted in a paper until then. Conflating them is how a paper gets desk-rejected for over-claiming.
2. **Pressure-tests the novelty story against Tayer et al. (2025/2026) specifically** — the "Mapping resilience" Gilbert paper — because that is the single most dangerous citation in the project's own reference set.
3. **Names claims that are safe in software docs but unsafe in a peer-reviewed paper.** The bar is different. Docs may say "diagnostic of drying-recession mode"; a paper cannot, until a figure shows it.

Verdict up front: **the metric foundations are mostly defensible, and the circularity critique is genuinely publishable. The exposure is not in the individual formulas — it is in (a) unvalidated interpretive claims stated as fact, (b) the dry-down "rate" and pool-width framings inviting hydrological misreading, and (c) a novelty story that overlaps Tayer 2025/2026 more than the spec admits in places, even though §16 handles the headline overlap well.**

---

## 1. Major scientific risks

Ranked by how much damage a hostile reviewer could do with them.

### R1 — Interpretive claims are asserted as established, not demonstrated (highest risk for the paper)

The spec repeatedly states *what a metric means ecologically* as settled fact, when it is a design-time hypothesis:

- AWRe "tracks drying-recession mode and discriminates pool type" (§6.1) — asserted; no result shown.
- AWRe ⊥ AWMSI orthogonality — the spec itself flags this as asserted (§6.18 row 1), yet §5.1/§6.1 still lean on it as if settled.
- Pool width = "morphological confinement signal" behaving as designed in practice — asserted from EDT properties (§6.18 row 4).
- Dry-down rate as "the key refuge-risk indicator" (§4, §6.5) — a causal-ecological claim, not yet linked to any refuge outcome.

The spec's §6.18 table is the right instrument and it is honest. The risk is that **the rest of the document does not consistently defer to it** — the metric register (§4) and core-set framing (§5.1) speak in the indicative mood ("diagnostic of", "tracks", "discriminates") where §6.18 correctly uses "asserted." A reviewer who reads §4 first and §6.18 last will conclude the authors over-claim and then walk it back. **Fix: make §6.18's mood the governing mood everywhere.** This is cheap and it is the single highest-value change in this report.

### R2 — "Dry-down rate" risks reading as hydrological recession despite the disclaimer

The spec adds a terminology `[AUDIT FIX]` (§6.5) saying this is a "monthly-extent contraction rate, not a hydrograph recession-constant analysis." Good — but the mitigation is one sentence buried in §6.5, while the *headline framing* everywhere else ("recession limb", "dry-down", "the project's headline metric", module named `dynamics.py`/"recession" in §9) keeps invoking recession vocabulary. A hydrologist reviewer sees "recession limb" and "slope" and immediately expects a master recession curve / `k` constant / storage-discharge relationship (Brutsaert & Nieber-style). This metric is a linear slope of *areal extent* over ≤12 monthly points. See R-detail in §5 below. **This is a framing fix, not a method fix** — but it must propagate beyond one sentence, or a reviewer will accuse the paper of dressing a simple extent-decline slope in recession-analysis clothing.

### R3 — Pool width invites depth/storage misreading, and the spec's own guard is fragile

The spec is admirably explicit that width ≠ depth (§6.9, §5.4, §8 guard 6). The residual scientific risk is different from what the spec guards: it is that **planform width from a 30 m mask is a weak and possibly biased morphological signal in its own right**, independent of the depth confusion. At 30 m, a channel 1–2 pixels wide is at the resolution floor (the spec concedes this in §6.9 limitations). EDT-derived width on a stair-stepped raster boundary has a known positive bias at narrow widths and is sensitive to skeleton branching at bends. So even taken purely as morphology, the metric may not be measuring confinement so much as measuring rasterisation artefacts in exactly the narrow-channel regime where confinement matters most. The width≠depth guard is necessary but does not address this. **This is a validate-before-paper item**, not a fix-before-implementation item — but the paper must not present width distributions from 30 m WOfS as clean morphology without showing the resolution-sensitivity.

### R4 — Novelty overlap with Tayer et al. (2025/2026) is broader than §16 fully neutralises

§16 is genuinely good and identifies the salami-slice risk correctly: it forbids re-presenting "a framework applied to Gilbert showing pool dynamics." But the overlap is deeper than the framework + case study:

- **HY detection.** The spec claims "persistence-based HY detection + zonation" as a paper contribution (§16.1 item 2). But Tayer 2025/2026 already published a *dynamic hydrological-year algorithm* (k-means on rainfall + percentile refinement — see paper summary Paper 4). The spec's HY detection is described as "persistence-based" (mask-derived, not rainfall-derived), which *is* a genuine difference — but the spec never states this difference explicitly, and §16.1 item 2 as written ("as implemented") would let a reviewer read it as the same HY concept. If the persistence-based HY detector is actually the same algorithm ported, this is a duplication finding. **This must be pinned down before the paper and probably before implementation** (the HY algorithm is listed as unlocked — Q7 in `spec_compliance.md`).
- **The four hydrological zone types.** Tayer 2023c (clustering paper) already derived 4 hydrological zone types via clustering, validated against groundwater tracer surveys. HydroFragments' four *geomorphic* zones (§3) are conceptually distinct (defined by persistence thresholds, not clustering) — but "four zones for an intermittent river, validated on the same rivers" is a collision a reviewer will notice. Positioning against Tayer 2023c is currently **absent** from §16 (which only addresses Tayer 2025/2026).

### R5 — NNI remains too prominent for a quasi-1D system, even demoted

The spec demotes NNI to Exploratory and states the correct objection itself (2D CSR is the wrong null for a linear corridor; unstable at low N; §6.8). That is the right call. The residual risk: NNI still appears in the core output schema, edge-case handling (§7 `N2_unstable`), guards (§8 guard 10), and the module map (§9). A hostile reviewer flips to the metric list, sees Clark–Evans NNI on a river, and forms a negative prior about the authors' spatial-statistics literacy *before* reading the demotion rationale. The demotion is scientifically sufficient; the *presentation* still gives NNI more surface area than an exploratory-only fallback warrants. **Consider cutting NNI entirely** (as centrality and morphology-proxy Zone 1 were cut) rather than carrying it as a fallback that the spec itself says is never publication-grade. If retained, it must be walled off far more aggressively in any paper — ideally not mentioned at all.

### R6 — Fixed-denominator principle is sound but has one soft spot: `A_ref` = `A_total` = AOI polygon

The spec's fixed-denominator discipline is the intellectual core and it is correct (§1.2, §4). One scientific wrinkle a reviewer could probe: LPI and MESH use `A_total` = AOI polygon area (§6.3, §6.4), and APSEC uses `A_ref` = same AOI polygon area. For a river-corridor AOI (reach + buffer), the AOI polygon area is **largely dry floodplain** — so LPI and MESH are diluted by however much buffer the user drew. This is not circular (the denominator is fixed and independent of the water), so it satisfies the stated principle. But it makes LPI/MESH **sensitive to an arbitrary user choice (buffer width)** and therefore not comparable across AOIs drawn with different buffers. The spec flags cross-sensor incomparability (§8 guard 1) but not cross-AOI-geometry incomparability for fixed-`A_total` metrics. **This is a documentation/guard fix**, and it is defensible either way — but it must be stated, because "LPI depends on how much dry land you drew around the river" is an easy reviewer jab.

### R7 — Validation is asserted, not empirical, and the paper's own key result depends on it

The composite-sensitivity finding (§1.2.2, §6.18) — that `max_water` biases dry-down upward — is presented as a methodological *result* the paper will claim (§16.1 item 3). But §6.18 concedes the *magnitude* is "not yet measured on real data." A paper cannot lead with "we identify a composite bias that flattens the headline metric" and then not quantify it. Either the dual-composite comparison runs and produces F4 with real numbers, or this claim drops from a *finding* to a *caveat*. **This is the clearest validate-before-paper item in the document.**

---

## 2. Metric-by-metric scientific audit

Columns: **Defensible for intermittent-river surface-water masks?** / **Positioning vs literature** / **Key scientific risk** / **Verdict**.

| Metric | Defensible for intermittent-river masks? | Positioning vs literature | Key scientific risk | Verdict |
|---|---|---|---|---|
| **Occurrence frequency** | Yes — this is the JRC-GSW / DEA-WOfS occurrence layer | Correctly cited to Pekel et al. 2016, Mueller et al. 2016. Spec is honest that this is not novel | Denominator must be `valid_obs`, not total timesteps — the current code gets this wrong (`spec_compliance.md` A5). A *scientific* not just code error: naive denominator biases occurrence in cloud-prone wet seasons | **Sound. Fix denominator before any occurrence-derived claim.** |
| **Refuge Area (RA)** | Yes | "Minor adaptation" — fair | Threshold `t_refuge=90%` is arbitrary; refuge magnitude scales with it. Fine if reported *with* the threshold and ideally as a sensitivity curve | **Sound. Report threshold sensitivity.** |
| **APSEC** | Yes — fixed denominator, verbatim Tayer | Verbatim (Tayer 2023b). Correct | `A_ref` = AOI polygon → buffer-width sensitivity (R6). Not circular, but not cross-AOI comparable | **Sound. Document buffer sensitivity.** |
| **LPSEC** | Yes, with caveats | Verbatim (Tayer 2023b) | Length in an equal-area CRS (spec's own §1.1.1 caveat); can exceed 100% on braided/floodplain reaches (paper summary notes this) — must be documented or it reads as a bug | **Sound. Length-CRS + >100% caveats must be visible.** |
| **AWRe** | Yes, robust to edge noise | **Correctly downgraded to "adapted from Schumm 1956," not verbatim.** This is the right call — Schumm's elongation ratio is basin morphometry; area-weighting it over a pool population is the project's own construction | Interpretation ("diagnostic of drying-recession mode / pool type") is **asserted** (R1). Length-method switch (skeleton vs major-axis) can flip the reading for curved pools — spec locks this (§6.1), good | **Formula sound; interpretation unvalidated. Keep as Core; soften interpretive claims.** |
| **AWMSI** | Yes | Verbatim (McGarigal & Marks 1995). Correct | Edge-noise sensitive at 30 m (spec concedes). Orthogonality to AWRe **asserted** (R1) | **Sound as a shape index; its distinctness from AWRe is a validate-before-paper claim.** |
| **N (number of pools)** | Yes | Verbatim | Strongly resolution- and `min_patch_pixels`-dependent; not cross-sensor comparable (spec guards this, §8.1). Scientifically fine as long as never pooled across resolutions | **Sound.** |
| **LPI** | Yes — non-circular, fixed denominator | Verbatim (McGarigal & Marks 1995; PyLandStats) | `A_total` buffer sensitivity (R6). Captures only the largest patch — spec is honest | **Sound. Document `A_total` sensitivity.** |
| **MESH** | Yes — non-circular | Minor adaptation (Jaeger 2000). Correct | Hard gate at r(LPI,MESH) > 0.9 is **asserted-not-run** (§6.18 row 2). Correlated with LPI when one patch dominates — which is most of the dry season for a river. May fail its own gate | **Sound in principle; keep-both decision is unvalidated. Run the gate before claiming both.** |
| **Dry-down rate** | Conditionally — as an *extent-contraction slope*, yes; as *recession*, no | "Novel in this context"; contraction-rate framing adapted from Costigan et al. 2016, Gallart et al. 2012. Citations fine | **Recession-language misreading (R2); `max_water` bias flattens it (R7); ≤12 monthly points per HY → low-df slope; undefined in years with no clear peak** | **Method OK; framing and validation are the exposure. Headline metric carries the most risk.** |
| **Reconnection timing** | Yes, if defined off RC/LPSEC/DCI not LPI | Correctly warns against LPI-only (§4.1) | Depends on HY-anchor quality; threshold-dependent; "reconnection" is a network claim needing a network metric | **Sound given the RC/DCI preference. Do not fall back to LPI silently.** |
| **Refuge spatial stability** | Yes — end-dry Jaccard avoids the static-footprint bug | Minor adaptation (Jaccard 1912). Spec fixed the v1.1 static-footprint bug (§6.16) | Jaccard of small end-dry footprints is noisy at low N; inter-annual registration/co-registration error can masquerade as instability | **Sound. Flag low-N-footprint years.** |
| **Pool width distribution** | Weakly at 30 m — resolution floor (R3) | EDT lineage correctly cited (Pavelsky & Smith 2008; Yang et al. 2020); non-circular reformulation is genuinely the project's own | Depth misreading (guarded) + rasterisation bias at narrow widths (**not** guarded, R3) | **Non-circular reformulation is good; the morphology claim itself needs resolution validation.** |
| **Inter-pool gap** | Yes — correct 1D geometry for rivers | Adapted from waterhole-spacing literature (Sheldon et al. 2010; Fullerton et al. 2010). Correct, and rightly made the metric of record over NNI | Requires a real skeleton/centreline; skeleton quality drives it; falls back to NNI (which the spec says is never publication-grade) | **Best-positioned clustering metric. Correct choice as metric of record.** |
| **NNI (Clark–Evans)** | No, for publication — right call to demote | Verbatim (Clark & Evans 1954; Donnelly 1978). Spec states the correct objection | Wrong null (2D CSR) for a 1D corridor; unstable at low N — exactly end-dry (R5). Still too prominent | **Demotion is correct; consider full cut.** |
| **TCF (temporal connectivity frequency)** | Yes | **Rename from PCF is correct and important** (collision with PC). Positioning against DCI/PC/IIC now required (§1.1.11) | Genuine novelty (temporal frequency) is real but narrow; must not be presented as if no comparable literature exists. Edge-definition sensitive | **Sound and genuinely novel *as a temporal reduction*. Positioning is mandatory, not optional.** |
| **DCI** | Yes, and strategically wise to at least cite | Cote et al. 2009; intermittent-river zero-flow application cited to PNAS 2025. Correct anchor | If implemented as reach-length-weighted, must be validated against a reference (`riverconn`/Conefor) or it's a bespoke reimplementation with the same risk it was meant to cure | **Adopt at least as citation-anchor; if implemented, benchmark it.** |
| **RC (realised connectivity)** | Yes — snapshot, no pool-identity tracking | Positioned against DCI (§6.13). Sound | Edge rule (wet connection vs dry-gap threshold) is a modelling choice that drives the result; must be recorded and justified | **Sound. The edge rule needs a defensible default and sensitivity note.** |
| **Recurrence / seasonality / hydroperiod** | Yes | Correctly positioned as JRC/DEA-style, not novel (§6.12) | None major — honest positioning | **Sound.** |
| PF, PLF | Dropped — correct | Circular denominator (N / own wetted area/length) | — | **Correctly dropped. Circularity argument is right.** |
| AWMPA | Dropped — correct but argued as "redundant" not "invalid" | `= MESH/APSEC` after unit harmonisation | The spec's nuance here is *correct and important* — do not overclaim it as circular | **Correctly dropped; the "redundant not invalid" framing is scientifically honest.** |
| AWMPL | Dropped — weakly argued | "Less stable than LPSEC + gap" | "Less stable" is asserted, not shown. Defensible to drop for redundancy, but the *stability* claim is unvalidated | **Drop is fine; the stated reason is soft (see §3).** |
| AWMPW (original) | Dropped, replaced by width distribution — correct | Same circular-weighting pattern | — | **Correctly dropped; replacement is the right move.** |
| Connected-components count | Dropped — correct | Identical to raster N unless graph edges differ | — | **Correct.** |
| Largest-component fraction | Dropped for leanness — **correctly NOT claimed identical to LPI** | `max(a_i)/Σa_i` ≠ `max(a_i)/A_total` | The spec fixed a real v1.1 error here (they are not the same quantity). Good | **Correct and the reasoning is now right.** |
| Degree/betweenness centrality | Cut — correct | Trivial on linear reaches | — | **Correctly cut.** |

---

## 3. Dropped metrics — correctly dropped, or only weakly argued?

**Correctly and strongly dropped** (the argument is a genuine contribution):

- **PF, PLF** — circular denominator (numerator count over the same water features). This is the cleanest, most defensible drop and the intellectual core of the paper. A reviewer will accept it.
- **AWMPW (original)** — same circular-weighting pattern; replaced by an unweighted distribution. Strong.
- **AWMPA** — the spec's honesty is a strength: it says "redundant, algebraically `= MESH/APSEC`, recoverable not invalid." Do **not** let the paper overclaim this as circular; it isn't. Redundancy is a sufficient reason.
- **Largest-component fraction** — correctly *not* dropped-as-identical-to-LPI (the spec fixed the v1.1 error). Dropped for leanness. Defensible.
- **Connected-components count, centrality** — correctly dropped/cut.

**Weakly argued drops** (fine outcome, soft reasoning — a reviewer could poke, but nothing fatal):

- **AWMPL — "less stable than LPSEC + gap/wet-run metrics."** "Less stable" is asserted, never demonstrated. The honest defensible reason is *redundancy* with LPSEC + gap, not superior stability. Reframe the drop reason around redundancy, or show the stability claim. Low stakes.
- **NNI demotion (not a drop, but adjacent).** The demotion reasoning is strong; the problem is retention, not the argument (R5).

No metric is *wrongly* dropped. The drops are scientifically sound. The only exposure is that two of them (AWMPL stability, and implicitly the MESH-keep decision) rest on stability/redundancy claims that are asserted rather than shown — and a hostile reviewer will ask "show me."

---

## 4. Fixed denominators — scientifically justified?

**Yes, and this is the strongest and most publishable idea in the whole spec.** The principle — no metric whose denominator is composed of the same water features as its numerator — is correct, well-motivated, and directly fixes a real defect in the predecessor suite (PF/PLF/AWMP*).

Two scientific caveats a reviewer will raise, both **documentation/guard fixes, not method flaws**:

1. **Fixed ≠ comparable across AOIs (R6).** `A_ref` = `A_total` = AOI polygon means APSEC/LPI/MESH depend on how much dry buffer the user drew. Fixed denominator kills *within-series circularity* but not *cross-AOI arbitrariness*. State this explicitly; add a guard analogous to the cross-sensor guard.
2. **`L_ref` in an equal-area CRS** — the spec already concedes length distortion (§1.1.1, §2). At catchment scale this is real. The spec handles it honestly as a documented caveat; keep it visible.

The spec's own clarification (§4.1) that shape summaries (AWRe, AWMSI) may remain area-weighted *descriptive statistics* provided they are not read as fixed-area fragmentation indices is **correct and necessary** — it prevents the fixed-denominator rule from being applied where it doesn't belong. Good.

---

## 5. Does dry-down rate risk overclaiming as hydrological recession?

**Yes. This is the second-highest risk in the document (R2), and the spec's mitigation is under-propagated.**

What the metric is: a linear (or Theil–Sen) slope of monthly APSEC against month index between peak-wet and end-dry anchors, per hydrological year. That is a **decline rate of surface-water extent**, over at most ~12 monthly points.

What "recession" means to a hydrologist reviewer: the falling limb of a hydrograph, characterised by a recession constant `k` in a storage–discharge model (e.g. `Q = Q₀e^{-t/k}`; Brutsaert & Nieber 1977; Tallaksen 1995). It implies an exponential/nonlinear form, a baseflow-storage interpretation, and discharge — none of which this metric computes. It is areal extent, not flow, and linear, not exponential.

The spec's §6.5 `[AUDIT FIX]` says exactly this — but:
- the mitigation is one sentence in §6.5, while "recession limb," "dry-down," "headline metric," and a `dynamics.py`/"recession" module label recur throughout;
- the metric is called a **rate**, which compounds the expectation of a rate *constant*.

**Required fixes (framing, cheap, before paper — arguably before implementation for naming consistency):**
- Rename in user-facing text and ideally in code toward "surface-water contraction rate" / "extent-decline rate"; reserve "recession" for genuine hydrograph analysis or drop it.
- State the linear-vs-exponential and extent-vs-discharge distinctions wherever the metric is defined, not once.
- Report the degrees of freedom / number of monthly points per HY slope, and flag low-df years — a slope from 3 points is not a rate.

**Additional validation risk (R7):** the `max_water` composite bias is *claimed* to flatten and delay this signal, magnitude unmeasured. The dual-composite check (§1.2.2) must run and produce F4 before the paper claims composite bias as a finding.

---

## 6. Does pool width distribution risk being misread as depth/storage?

**The depth misreading is well-guarded; a *different* risk is not (R3).**

The spec guards width≠depth aggressively and correctly (§6.9, §5.4, §8 guard 6). A reviewer cannot claim the authors conflated width with depth — they explicitly forbid it. Credit where due.

The residual, unguarded scientific risk is that **planform width from a 30 m mask may be a poor morphology signal in its own right**, independent of depth:
- narrow channels (1–2 pixels) sit at the resolution floor — the spec concedes this in §6.9 limitations but does not carry it into interpretation or plots;
- EDT width on a stair-stepped raster boundary has a systematic positive bias at small widths;
- skeleton branching at meander bends distorts medial-axis width.

So even as pure morphology, the width distribution may partly encode rasterisation artefacts in exactly the confined-channel regime where the confinement signal is supposed to be most informative. **Validate-before-paper:** show width-vs-resolution behaviour, and/or restrict the morphology claim to widths comfortably above the resolution floor. Do not present 30 m WOfS width distributions as clean confinement morphology without this.

---

## 7. Does NNI remain too prominent given quasi-1D river geometry?

**Yes (R5).** The demotion to Exploratory is scientifically correct and the spec states the right reasons (2D CSR is the wrong null for a linear corridor; instability below N≈8–10 coincides with the end-dry regime that matters most). The problem is **surface area, not argument**: NNI still appears in the output schema, edge-case flags (`N2_unstable`, `Nlt10_NNI_unstable`), guard 10, the module map, and §6.18 row 3. A hostile reviewer scanning the metric inventory sees "Clark–Evans NNI on a river" and forms a negative prior before reaching the demotion.

**Recommendation:** treat NNI the way the spec treated centrality and morphology-proxy Zone 1 — **cut it from v1 outright.** The spec already says it is *never* publication-grade for river fragmentation and only a planar fallback where no skeleton exists; inter-pool gap is the metric of record. A metric that is never publishable and only fires in a degraded mode is a maintenance and reputational liability, not an asset. If it is retained for pure exploratory completeness, it must be **absent from any paper** and clearly quarantined in docs.

---

## 8. Are DCI / PC / IIC distinctions clear enough?

**Mostly yes — this is one of the spec's best recoveries — with two residual sharp edges.**

The spec correctly:
- renames PCF → TCF to avoid collision with PC (§1.1.11) — important and correct;
- requires every connectivity metric to state its relationship to DCI (Cote et al. 2009) and PC/IIC (Saura & Pascual-Hortal 2006, 2007) and cite them (§4.1, §6.11);
- distinguishes DCI (fragment-size / reachability, maps onto discrete pools) from PC/IIC (require a distance-decay dispersal kernel, do **not** map cleanly onto fixed discrete pools) and correctly declines to reimplement PC/IIC (§6.11a "not recommended").

Residual sharp edges a reviewer will probe:

1. **RC vs DCI near-equivalence.** The spec says `RC_pair` with reach-length-weighted nodes is "structurally close to a monthly DCI snapshot" (§6.11a, §6.13). If they are *that* close, a reviewer asks: why introduce RC as a separate named metric at all rather than just computing monthly DCI? The answer (RC's edge rule is more general; DCI is the citable special case) is defensible but **must be stated**, or RC looks like a renamed DCI — the exact sin the spec accuses v1.1 of committing with connectivity.
2. **TCF's novelty is narrow and must be scoped precisely.** "Temporal frequency of connectivity over a monthly series" is genuinely not what static DCI/PC give — but Rubio & Saura (2012) and related temporal-connectivity work exist (the spec cites Rubio & Saura). The claim must be "novel *operationalisation* as a monthly-frequency reduction on remotely-sensed masks," not "novel concept of temporal connectivity." The spec mostly gets this right; the paper must not drift.

---

## 9. Are validation claims empirical or still asserted?

**Still asserted — and the spec is commendably honest about this in §6.18, but the honesty is localised.**

§6.18 is the right instrument: it inventories asserted-vs-demonstrated and commits to move rows as checklist items resolve. Every one of the following is currently **asserted**, per the spec's own table:

- AWRe ⊥ AWMSI orthogonality — asserted (item 11 will test).
- LPI/MESH non-redundancy (keep-both) — asserted; hard gate at r>0.9 not run (item 12).
- NNI instability below N≈8–10 — demonstrated *in literature*, not on this pipeline's data (item 13).
- Pool width behaves as morphology not depth *in practice* — asserted (item 14).
- `max_water` composite bias magnitude on dry-down — mechanism understood, magnitude unmeasured (new item).
- RC/TCF sensible relative to DCI on a real network — not checked.
- Classification-error propagation to N/gap/MESH-tails/width — not characterised (explicitly out of scope for v1 as a formal model).

**Bottom line for the paper:** none of the interpretive/relational claims are empirical yet. A methods paper that *demonstrates* its keep/drop decisions is publishable; one that *argues* them is a desk-reject risk (the spec says this itself in §16.1 item 4). The §6.18 table must be driven to "demonstrated" on at least items 11, 12, and the composite-bias item before submission. **The circularity/fixed-denominator argument (§4) is the exception — it is a logical/algebraic result, not an empirical one, and is publishable as reasoning without a dataset.**

---

## 10. What would a hostile reviewer say about novelty vs Tayer et al. (2025/2026)?

The dangerous paper is Tayer et al. (2025/2026), "Mapping resilience" (Gilbert River, 1986–2023, WOfS Landsat) — it already publishes the four-step framework, the persistent-pool concept, dynamic hydrological-year detection, and Gilbert pool-dynamics trends. Also relevant: Tayer 2023c (clustering → 4 hydrological zone types, groundwater-validated) and Tayer 2023b (the original 11-metric ecohydrological suite).

**The strongest hostile objection:** *"This is the software implementation of Tayer 2025/2026 plus a metric-cleanup appendix, re-run on the same river. The framework, the HY detection, the zonation concept, and the Gilbert case study are all already published. What is left that is new?"*

**The best defensible response (and it is a good one, if disciplined):**
1. **The novel contribution is the metric-register reformulation and its circularity critique** (§4, §16.1 item 1) — eliminating PF/PLF/AWMPA/AWMPL/AWMPW on fixed-denominator grounds, and the non-circular replacements (unweighted width distribution, AWRe/AWMSI as descriptive shape axes, RC/TCF/DCI connectivity positioned against the standard literature). No prior Tayer paper makes this argument. This is real, and it is publishable *as reasoning*.
2. **The software artefact itself**, with reproducibility discipline (config hashing, composite-sensitivity flagging, source-agnostic input) — a JOSS-shaped contribution distinct from the science paper (§13, §16.4).
3. **The composite-sensitivity result** (`max_water` biases dry-down) as a genuine new methodological finding — *if quantified* (R7).

**Where the response is currently weak, and must be shored up before the paper:**
- **HY detection overlap (R4).** The spec claims HY detection as a contribution (§16.1 item 2) but Tayer 2025/2026 already published a dynamic HY algorithm. Unless the persistence-based detector is *demonstrably different* and stated as such, this is a duplication finding. **Pin down and articulate the difference, or drop HY detection from the claimed contributions and cite Tayer 2025/2026 as its source.**
- **Zonation overlap.** §16 addresses Tayer 2025/2026 but not Tayer 2023c's four groundwater-validated hydrological zones. The geomorphic four-zone scheme (§3) is conceptually distinct (persistence-threshold-defined, not clustering-derived) — but the paper must say so, or "four zones on the Fitzroy/Gilbert" reads as a rerun.
- **Same study system.** Using Gilbert (Tayer 2025/2026's exact dataset) as the validation case is efficient but hands the reviewer the salami-slice framing for free. §16.2 handles this correctly by using Gilbert **only to validate the register's specific claims** (orthogonality, redundancy, DCI benchmark) rather than to re-report ecology. That discipline must hold in the actual draft — the moment a results section describes Gilbert pool dynamics, the paper becomes a resubmission of Tayer 2025/2026.

**Net:** the novelty story is winnable, but only if the paper leads with the metric-register argument and the software, uses Gilbert strictly as a validation instrument, and explicitly cites Tayer 2023b/2023c/2025-2026 as the sources of the framework/HY/zonation/metric-suite it is *reforming*. The spec's §16 is 80% of the way there; R4 is the missing 20%.

---

## 11. Claims that MUST be validated before publication (empirical, cannot be asserted)

Each of these is currently asserted and must become a demonstrated result (a figure/number) before it appears as a claim in the paper.

| # | Claim | Required evidence | Spec hook |
|---|---|---|---|
| V1 | AWRe and AWMSI are orthogonal shape axes | Scatter + correlation on real Gilbert data; show occupancy of all four quadrants | §6.18 row 1; checklist 30; F6 |
| V2 | LPI and MESH are non-redundant enough to keep both | Correlation on real data; **hard gate: drop MESH if r > ~0.9** | §6.18 row 2; checklist 31 |
| V3 | `max_water` composite bias measurably flattens/delays dry-down | Dual-composite comparison on ≥1 catchment; report typical end-dry APSEC disagreement (pp) | §6.18 row 5; checklist 11; F4 |
| V4 | AWRe "tracks drying-recession mode / discriminates pool type" | Relate AWRe trajectory to independent drying/pool-type information; currently pure assertion | §6.1 (interpretation) |
| V5 | Pool width behaves as morphology, not artefact, at operational resolution | Width-vs-resolution sensitivity; comparison to any field/bathymetric data | §6.18 row 4; checklist 33 |
| V6 | RC/TCF/DCI behave sensibly vs a reference | Benchmark `RC_pair` (reach-length-weighted) against directly computed DCI (`riverconn`/Conefor) on Gilbert | §6.18 row 6; checklist 34; F8 |
| V7 | Dry-down rate is a meaningful "refuge-risk indicator" | Link dry-down to an actual refuge outcome (end-dry RA, pool survival); currently a named-not-shown causal claim | §6.5 interpretation |
| V8 | Persistence-based HY detection differs from Tayer 2025/2026's rainfall-based HY algorithm | Explicit algorithmic comparison; agreement/divergence on Gilbert | R4; Q7 in spec_compliance |

---

## 12. Claims safe for software docs but UNSAFE for the paper

The bar differs. These are fine to state in `docs/`/README/docstrings (as motivation/interpretation guidance), but must **not** appear as asserted fact in a peer-reviewed methods paper without the validation in §11.

| Claim | Safe in docs as… | Unsafe in paper because… |
|---|---|---|
| AWRe "diagnostic of drying-recession mode and pool type" | interpretation guidance for users | causal/ecological claim, unvalidated (V4) |
| AWRe ⊥ AWMSI ("two shape axes") | design rationale | asserted orthogonality (V1) |
| "Dry-down rate — the key refuge-risk indicator" | motivating framing | causal refuge link unshown (V7); "recession" misreads (R2) |
| Pool width = "morphological confinement signal" | interpretation-with-caveat | resolution-artefact risk unshown (V5) |
| Keep both LPI and MESH | reasonable default | redundancy gate not run (V2) |
| TCF is "novel temporal connectivity" | accurate as operationalisation | must be scoped as *operationalisation*, not concept (Rubio & Saura 2012 exists) |
| "max_water biases dry-down upward" | a documented, guarded caveat | as a *finding* needs magnitude (V3) |
| Persistence-based HY detection as a contribution | tool feature | duplication vs Tayer 2025/2026 until differentiated (V8) |

Rule of thumb: **docs may state a hypothesis as guidance; the paper may only state what §6.18 has moved to "demonstrated."**

---

## 13. Required citations / positioning fixes

Most are already in the spec — flagged here as a checklist a reviewer will verify.

**Already correct in spec (verify they land in code docstrings + paper):**
- Occurrence/recurrence/hydroperiod → Pekel et al. 2016; Mueller et al. 2016 (JRC-GSW / DEA-WOfS). §6.12.
- AWRe → Schumm 1956, **"adapted" not verbatim**. §6.1.
- AWMSI, LPI → McGarigal & Marks 1995; PyLandStats (Bosch 2019). §6.2/§6.3.
- MESH → Jaeger 2000. §6.4.
- Pool width EDT → Pavelsky & Smith 2008; Yang et al. 2020. §6.9.
- Inter-pool gap → Sheldon et al. 2010; Fullerton et al. 2010. §6.10.
- Connectivity → Cote et al. 2009 (DCI); Saura & Pascual-Hortal 2006, 2007 (IIC/PC); Baldan et al. 2022 (`riverconn`); Saura & Torné 2009 (Conefor); PNAS 2025 (zero-flow DCI). §1.1.11, §6.11a.
- Dry-down framing → Costigan et al. 2016; Gallart et al. 2012. §6.5.
- NNI → Clark & Evans 1954; Donnelly 1978. §6.8.
- Temporal connectivity → Rubio & Saura 2012. §6.11.

**Missing / must add (positioning gaps a reviewer will catch):**
- **Tayer et al. 2023c (clustering / four hydrological zone types, groundwater-validated)** — must be cited and differentiated from the geomorphic four-zone scheme (§3). Currently absent from §16.
- **Tayer et al. 2025/2026 (dynamic HY algorithm)** — must be cited *specifically as the source of dynamic HY detection*, with the persistence-based detector explicitly differentiated (R4/V8). §16 cites the paper for framework/Gilbert but not for HY.
- **Recession-analysis anchor** — when documenting dry-down, cite the recession literature being *distinguished from* (e.g. Brutsaert & Nieber 1977; Tallaksen 1995) precisely to disclaim it, so a reviewer sees the authors know the difference (R2).
- **Dry-down / drying-front remote-sensing precedents** — a reviewer will ask whether surface-water contraction rate from EO is truly novel; a short positioning against existing surface-water-dynamics EO work (beyond Costigan/Gallart, which are field hydrology) strengthens the "novel as a remote-sensing metric" claim.

---

## 14. Suggested validation analyses (minimal, mostly already in checklist 11C)

Ordered by value-per-effort. Most reuse the Gilbert dataset already in hand — no new data collection required.

1. **Dual-composite dry-down comparison (V3, F4).** Run `max_water` vs `median` dry-down on Gilbert; report typical end-dry APSEC disagreement (pp) and its effect on the slope. *This is the paper's own headline finding — highest priority.*
2. **LPI/MESH correlation gate (V2).** Compute r(LPI, MESH) across Gilbert months/sections; apply the r>0.9 hard gate. Cheap, binary outcome, decides a keep/drop.
3. **AWRe/AWMSI orthogonality (V1, F6).** Scatter + correlation; check quadrant occupancy. Cheap.
4. **RC/DCI benchmark (V6, F8).** `RC_pair` (reach-length-weighted) vs directly computed DCI on the Gilbert reach; report agreement. Needs `riverconn`/Conefor as reference.
5. **Pool-width resolution sensitivity (V5).** Width distribution vs resolution / vs narrow-channel floor; ideally against any field width or bathymetry. Guards R3.
6. **NNI stability vs N on this pipeline's data (checklist 32)** — confirms the exploratory-only tier (it should) and supports cutting it (R5).
7. **HY-algorithm comparison (V8).** Persistence-based HY vs Tayer 2025/2026 rainfall-based HY on Gilbert; report agreement/divergence. Settles the novelty question (R4) and is cheap given both are computable on the same record.
8. **Buffer-width sensitivity of APSEC/LPI/MESH (R6).** Recompute with 2–3 buffer widths; show and document the sensitivity. Small, and pre-empts an easy jab.

None of these require data the project does not already have (Gilbert WOfS 1986–2023/2024 is in hand). Item 4 needs an external reference tool, not new data.

---

## 15. Red-team reviewer objections and best responses

| # | Hostile objection | Best defensible response | Residual exposure |
|---|---|---|---|
| O1 | "This is Tayer 2025/2026 re-implemented on the same river — where's the novelty?" | Lead with the metric-register/circularity reformulation and the software artefact; use Gilbert only to *validate* the register (not re-report ecology); cite Tayer 2023b/c/2025-26 as the sources being reformed. | Real until R4/V8 are closed (HY + zonation differentiation). |
| O2 | "Your 'dry-down rate' is not a recession — it's a slope of extent over ≤12 points dressed as hydrology." | Rename to surface-water contraction/extent-decline rate; explicitly disclaim recession-constant modelling (cite Brutsaert & Nieber to distinguish); report per-HY df and flag low-df years. | Low if framing is fixed everywhere; high if "recession" language persists. |
| O3 | "Pool width from 30 m WOfS is rasterisation noise in the narrow-channel regime, not morphology." | Show width-vs-resolution sensitivity (V5); restrict the morphology claim above the resolution floor; keep the width≠depth guard. | Moderate until V5 runs. |
| O4 | "Clark–Evans NNI on a river shows you don't understand the geometry." | We agree — NNI is demoted to exploratory/cut; inter-pool gap (correct 1D geometry) is the metric of record, positioned on waterhole-spacing literature. | Low, *if NNI is cut or fully quarantined* (R5); otherwise its mere presence invites the prior. |
| O5 | "You reinvented connectivity indices without citing DCI/PC/IIC." | v1.2 explicitly cites and positions against Cote 2009 and Saura & Pascual-Hortal 2006/2007, renames PCF→TCF to avoid PC collision, and declines to reimplement PC/IIC. | Low — this is now a strength; residual is the RC-vs-DCI "why both" question (O8). |
| O6 | "LPI/MESH are redundant — you kept two metrics measuring the same thing." | We apply a pre-registered hard gate (drop MESH if r>0.9 on real data). | Real until V2 runs; if the gate fails and MESH is kept anyway, fatal. |
| O7 | "Your interpretive claims (AWRe diagnostic, refuge-risk) are asserted, not shown." | §6.18 inventories exactly these as asserted and commits to demonstrate them; the paper only states demonstrated rows. | Real until V1/V4/V7 run; mitigated by not asserting in the paper. |
| O8 | "RC is just a monthly DCI under another name — the sin you accuse v1.1 of." | RC's edge rule generalises beyond DCI's fragment-size form; DCI is the citable special case with reach-length nodes; both are reported and their relationship stated. | Low if the relationship is stated explicitly; otherwise it reads as renaming. |
| O9 | "Fixed denominators don't make metrics comparable — LPI depends on how much dry land you drew." | Correct; fixed denominator removes within-series circularity, not cross-AOI arbitrariness. We document buffer sensitivity and guard cross-AOI pooling. | Low once documented (R6). |
| O10 | "You dropped AWMPA/AWMPL as 'circular/unstable' but they're standard area-weighted summaries." | We do not call AWMPA circular — we drop it as *redundant* (`=MESH/APSEC`); AWMPL for redundancy with LPSEC+gap. The circularity charge is reserved for PF/PLF/AWMPW (numerator features in denominator). | Low — the spec's nuance here is correct; keep AWMPL's stated reason as redundancy, not "instability" (soft claim). |
| O11 | "Occurrence over total timesteps is biased in cloudy wet seasons." | Correct — occurrence uses `valid_obs` denominator with a `min_valid_obs` floor. | Low in v1.2 design; **but the current code violates this** (`spec_compliance.md` A5) — must be fixed in implementation. |

---

## 16. Must-fix-before-implementation vs must-validate-before-paper

The spec conflates these. Separating them is the single most useful output of this audit.

### 16.1 MUST FIX BEFORE / DURING IMPLEMENTATION (design & code — cheap, no data needed)

These are scientific-correctness or framing fixes that belong in the build, independent of any dataset:

- **F-1 — Occurrence denominator.** Use `valid_obs`, not total timesteps, with `min_valid_obs` floor. The current code is wrong (A5); this is a *scientific* error, not just schema drift. **Blocking.**
- **F-2 — Sentinel/validity preservation.** Preserve `254`/`255` and `observed` from WaterMask-TSFill; do not collapse invalid → dry (B2). Corrupts every spatial metric otherwise. **Blocking.**
- **F-3 — Equal-area CRS + AOI/raster co-reprojection** before any area/length metric; per-pixel area fallback (A11/A12). **Blocking for defensible areas.**
- **F-4 — Dry-down naming/framing.** Rename toward "contraction/extent-decline rate" in code and docs; disclaim recession-constant modelling; emit per-HY df. Propagate beyond the single §6.5 sentence (R2). **Cheap, do now.**
- **F-5 — Governing mood = §6.18's mood.** Rewrite §4/§5.1 interpretive language from indicative ("diagnostic of", "tracks") to hypothesis ("intended to indicate", "hypothesised") wherever §6.18 marks the claim asserted (R1). **Cheap, do now.**
- **F-6 — Buffer/cross-AOI guard.** Document that `A_ref`/`A_total`-normalised metrics (APSEC/LPI/MESH) are buffer-width-sensitive and not cross-AOI comparable; add a guard analogous to the cross-sensor guard (R6). **Cheap.**
- **F-7 — Decide NNI's fate.** Recommend cutting NNI from v1 (like centrality); if kept, quarantine it out of any publication surface (R5). **Design decision, do now.**
- **F-8 — RC-vs-DCI relationship stated in code docstrings** so RC is not later read as renamed DCI (O8). **Cheap.**
- **F-9 — Width resolution-floor guard.** Flag/withhold width statistics for pools at the 1–2-pixel width floor; document the artefact risk beyond the depth caveat (R3). **Cheap guard.**

### 16.2 MUST VALIDATE BEFORE THE PAPER (empirical — needs the Gilbert data, per §11/§14)

These may ship in software as guarded hypotheses, but **cannot be asserted as claims in a peer-reviewed paper** until demonstrated:

- **V1** AWRe ⊥ AWMSI orthogonality (F6/scatter).
- **V2** LPI/MESH r>0.9 hard gate (keep/drop MESH).
- **V3** `max_water` dry-down bias magnitude (F4) — *the paper's own headline finding.*
- **V4** AWRe "diagnostic of drying mode/pool type."
- **V5** Pool width behaves as morphology, not artefact, at 30 m.
- **V6** RC/TCF/DCI benchmark vs reference (F8).
- **V7** Dry-down rate linked to an actual refuge outcome.
- **V8** Persistence-based HY vs Tayer 2025/2026 rainfall HY — settles novelty (R4).

**Publishable without data (logical/algebraic results — the paper's spine):**
- The fixed-denominator/circularity argument and the PF/PLF/AWMPA/AWMPL/AWMPW drops (§4).
- The connectivity-positioning against DCI/PC/IIC and the PCF→TCF rename (§1.1.11).
- The reproducibility discipline (config hashing, composite-sensitivity flagging) as an engineering contribution.

---

## 17. Bottom line

- **The metric foundations are scientifically defensible.** The fixed-denominator/circularity reformulation is a genuine, publishable contribution and the individual formulas are correctly attributed (with the important corrections already made: AWRe "adapted," occurrence = JRC/DEA-style, largest-component-fraction ≠ LPI).
- **The exposure is not the formulas — it is (1) asserted interpretation stated as fact, (2) "recession"/depth framings inviting misreading, and (3) a novelty story that overlaps Tayer 2025/2026 (and 2023c) more than §16 currently neutralises for HY detection and zonation.**
- **Nothing here is fatal to implementation.** The must-fix items (§16.1) are cheap design/framing corrections plus the already-known code bugs (occurrence denominator, sentinels, CRS).
- **The paper's risk is real but manageable:** lead with the metric-register argument and the software; use Gilbert strictly to validate the register (not re-report ecology); drive §6.18 to "demonstrated" on V1/V2/V3 (and ideally V5/V6/V8) before submission; and differentiate persistence-based HY detection from Tayer 2025/2026 explicitly or drop it as a claim.
- **Single highest-value change:** make §6.18's honest "asserted vs demonstrated" mood the governing mood of the entire document, so the metric register stops speaking in the indicative about things the project has not yet shown.
