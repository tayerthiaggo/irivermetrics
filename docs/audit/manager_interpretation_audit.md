# HydroFragments v1.2 — Manager Interpretation Audit

**Audit date:** 2026-07-10
**Reviewer stance:** Water-resource manager interpretation reviewer. Not a scientific peer-review pass (see [`scientific_metrics_audit.md`](scientific_metrics_audit.md) for that) and not a code/spec compliance pass (see [`spec_compliance.md`](spec_compliance.md)). This report asks: *if a water manager, environmental-flows planner, or catchment officer reads a HydroFragments output table with no scientific training, what will they conclude — and where will that conclusion be wrong, incomplete, or dangerous?*
**Scope:** interpretation and communication only. No source files changed; no code written; no formulas invented; no numeric thresholds invented.
**Contract audited:** [`docs/HydroFragments_v1.2_spec.md`](../HydroFragments_v1.2_spec.md), §12.1 in particular (which already scopes a `docs/for-managers.md` deliverable this report expands into an implementable outline).

> **Mandatory next-phase intake gate**
>
> The next phase must read the prior audit markdown files **before** doing its own work, in this order:
>
> 1. [`docs/audit/evidence_packet.md`](evidence_packet.md)
> 2. [`docs/audit/repo_triage.md`](repo_triage.md)
> 3. [`docs/audit/spec_compliance.md`](spec_compliance.md)
> 4. [`docs/audit/scientific_metrics_audit.md`](scientific_metrics_audit.md)
> 5. [`docs/audit/dask_cuda_audit.md`](dask_cuda_audit.md)
> 6. this report — [`docs/audit/manager_interpretation_audit.md`](manager_interpretation_audit.md)
>
> This report assumes the metric register and caveats fixed by the scientific audit (`[AUDIT FIX]` items: AWRe framed as "adapted," NNI demoted, dry-down reframed away from "recession," etc.). Where the current spec's manager-facing language (§4, §5.1) still uses the un-audited indicative mood ("diagnostic of," "tracks"), this report treats [`scientific_metrics_audit.md`](scientific_metrics_audit.md) §16.1 F-5 as already governing — manager docs must never be *more* confident than the scientific audit allows, since managers are the audience least equipped to catch overclaiming.

---

## 0. How to read this report

A manager does not read the spec. A manager reads a CSV column header, a dashboard tile, or a two-paragraph summary someone hands them before a water-sharing plan meeting. The failure mode this report guards against is not "the manager misunderstands a formula" — it is **"the manager correctly reads the number and still draws the wrong management conclusion because the number means less, or something different, than its name suggests."**

Three different audiences get three different documents, and this report is careful not to blur them:

- **The scientific audit** ([`scientific_metrics_audit.md`](scientific_metrics_audit.md)) asks whether a metric is defensible in a journal.
- **This audit** asks whether a metric is safe to act on in a management decision when read without that scientific context.
- A metric can pass one and fail the other. Occurrence frequency is scientifically unimpeachable *and* dangerous for a manager who reads it as "how deep/permanent the water is." Dry-down rate is scientifically the highest-risk metric in the register *and* is exactly the number a manager most wants for environmental-flow timing — which is precisely why it needs the heaviest plain-language guarding, not exclusion.

---

## 1. Which metrics are meaningful to water managers?

Meaningful here means: answers a question a manager actually has, at a resolution they can act on, without requiring them to hold a caveat in their head while reading it.

### Directly meaningful (map to a management question with only light translation)

| Metric | Management question it answers |
|---|---|
| Refuge Area (RA) | How much dry-season refuge habitat is there, and is it shrinking year over year? |
| APSEC (wetted area fraction) | How much of this reach is holding water right now, relative to a fixed reference? |
| LPSEC (wetted length fraction) | Is the river still longitudinally connected as a visible wet corridor, or has it broken into disconnected pools? |
| Number of pools (N) | Is the reach one continuous water body or many fragments — a first-glance fragmentation signal |
| LPI (largest patch index) | Is there one dominant refuge/water body, or is the wetted area spread thin across many small ones? |
| Dry-down rate | How fast is this reach losing surface water through the dry season — the number most directly useful for timing an environmental water release |
| Refuge spatial stability | Do refuges reappear in the same places every year (reliable, protectable) or move around (harder to manage as fixed assets) |
| Reconnection timing | How long after the wet season starts does the river reconnect end-to-end |
| Pixel recurrence / hydroperiod | How reliable is water at a given location across years / how many months per year is a location typically wet |

These are the metrics worth putting in front of a manager as headline numbers, because each one maps to a single, specific decision question without requiring a statistics background.

### Meaningful only in combination or context (see §4)

Inter-pool gap, TCF, RC, MESH, pool width distribution — each is real signal, but none of them answers a management question *on its own*. They sharpen or contradict a headline metric. Presenting them standalone invites a manager to treat "gap increased" as bad news, when the actionable question is always "gap increased *relative to what else*."

### Not meaningful to managers as presented, and should not be surfaced as headline outputs

- **AWMSI, AWRe as raw shape-index numbers.** A value like "AWRe = 1.34" has no intuitive anchor for a non-specialist. These are useful as inputs to a narrative ("pools are becoming more elongated, consistent with active drying") but the number itself should not appear on a manager dashboard.
- **DCI/RC/graph-theoretic outputs**, when reported as fractions of network reachability without a plain-language wrapper — a manager cannot act on "RC = 0.42."
- **NNI**, in any form — see §3. It should not reach a manager audience at all.

---

## 2. Which metrics need plain-language translation?

Every metric that survives to a manager-facing document needs translation, but the following need the most work because their *names* actively mislead:

| Metric | Why the name misleads | Translation direction |
|---|---|---|
| **Dry-down rate** | "Rate" and "recession" (used elsewhere in the spec/code) sound like a hydrograph flow-recession constant. It is not a flow measurement at all — it is how fast the *visible wet area* is shrinking, month to month, in a given dry season. | "Surface-water shrinkage speed" or "how quickly the wetted area is contracting this dry season" — never "recession," never implied flow/discharge. |
| **AWRe (elongation ratio)** | Sounds geometric/abstract; managers will not connect "elongation" to anything actionable. | "Pool shape signal — long and thin vs. round and compact; shape changes can accompany active drying, but shape alone does not confirm it." Must carry an unvalidated-interpretation caveat (§3, §5). |
| **Pool width distribution** | "Width" invites an immediate mental leap to "how deep/how much water," i.e. storage. | "How wide the water is at its surface — tells you nothing about how deep it is or how much water is stored." Must be paired explicitly with the width≠depth warning every time it is shown (§5). |
| **Occurrence frequency / Refuge Area threshold** | Sounds like a fixed physical category ("this pixel is a refuge") rather than a modeled classification against an adjustable cutoff. | "Places that hold water almost every time we've observed this location — using a cutoff percentage the analysis team can adjust; changing the cutoff changes how much area counts as refuge." |
| **MESH / LPI** | Landscape-ecology jargon ("mesh size," "patch index") with no intuitive meaning outside FRAGSTATS literature. | LPI → "share of the total wetted area held by the single largest water body." MESH → "an overall fragmentation score that gives more weight to bigger water bodies — a second, complementary view to LPI, not a repeat of it." |
| **TCF (Temporal Connectivity Frequency)** | Acronym plus "connectivity" collides in the reader's mind with DCI and RC — three different-sounding but related-sounding numbers. | Never show TCF, RC, and DCI together without one sentence distinguishing them (§4). "TCF: what fraction of the time this specific spot stayed connected to the network." |
| **Composite type (`max_water` vs `median`)** | Not a metric but a hidden processing choice that silently changes dry-down and other outputs. Managers will not know this exists unless told. | "This number was calculated using the [maximum / typical] monthly water extent — the two methods can give different answers for how fast the reach is drying; see composite-sensitivity flag." |
| **Valid observation count / low-valid-observation flag** | Managers will not intuitively know that a metric can be *wrong*, not just *absent*, when satellite coverage was poor. | "This number is based on fewer clear satellite views than usual for this period — treat it as less reliable, not as a confirmed reading." |

---

## 3. Which metrics are dangerous if interpreted naively?

Ranked by how bad the naive misreading is, not by how good the metric is scientifically.

### Danger 1 — Dry-down rate read as a precise recession/flow measurement

**Naive reading:** "The dry-down rate is X% per month, so the river will fully dry in Y months — we can time the environmental flow release to the week."

**Why it's dangerous:** the metric is a slope fit to at most ~12 monthly points of *visible wetted-area* contraction, not a flow/discharge recession constant. It carries no storage or groundwater information, can be undefined or unstable in years without a clean drying trend, and is measurably sensitive to which monthly-compositing method was used (`max_water` vs `median`) — the scientific audit's own R7/V3 finding is that this composite-choice bias is asserted but not yet quantified. Treating it as flow-precise could drive an environmental-flow release timed against the wrong signal, or with false confidence in a slope built from three or four monthly points.

**Required guard:** always report alongside (a) the number of monthly points the slope was fit from, (b) which composite method was used, and (c) whether the composite-sensitivity flag is raised for that reach/year. Never state a drying "date" or "days remaining" projection from this metric without heavy hedging.

### Danger 2 — Pool width read as depth or water volume

**Naive reading:** "The pools are wide, so there's a lot of water / the refuge is well-stocked."

**Why it's dangerous:** width is a surface measurement only. A pool can be wide and centimetres deep, or narrow and metres deep. Confusing the two could lead a manager to deprioritise protection of a narrow-but-deep refuge that is actually more ecologically valuable than a wide-but-shallow one. The spec is explicit that this guard must hold (§6.9, §5.4) — this report's job is to make sure it survives translation into manager language, not just code guards.

**Required guard:** every appearance of pool width in manager-facing material must carry the sentence "width tells you nothing about depth or water volume" in the same breath, not as a footnote.

### Danger 3 — Occurrence frequency / Refuge Area read as a permanent, fixed category

**Naive reading:** "This is a refuge" (full stop) — treated as a stable, binary, permanent classification suitable for hard planning boundaries (e.g., a protected-area line on a map).

**Why it's dangerous:** occurrence and refuge status are both computed against a chosen probability/time-window and a chosen threshold (default around 90% occurrence, itself adjustable). They also depend on how many valid satellite observations were available — a location flagged as refuge in a data-sparse period is a weaker claim than one flagged in a data-rich period. A manager drawing a permanent boundary from one run of the tool risks locking in a boundary that shifts under a different threshold, time window, or when more/better observations come in later.

**Required guard:** always report the occurrence/refuge threshold used, the time window it was computed over, and the valid-observation support, alongside the classification. Frame refuge status as "how this location behaved over the period analysed," not as a permanent designation.

### Danger 4 — Any single-timestep number read as trend

**Naive reading:** "APSEC this month is low, so the reach is degrading" — without reference to season, hydrological year stage, or prior years.

**Why it's dangerous:** APSEC and most core-tier metrics are point-in-time or point-in-season values. A low value in the expected dry-season trough is normal; the same value outside the expected trough, or trending lower across multiple dry seasons at the same calendar point, is the actual signal. This is a combination problem (§4), not a single-metric problem, and is the single most common way time-series ecological data gets misread by a non-specialist audience.

**Required guard:** never present a single-timestep metric value without its seasonal/hydrological-year context, and prefer showing it against the same reach's own history over showing it in isolation.

### Danger 5 — Low-valid-observation periods read as "confirmed dry" or "confirmed low"

**Naive reading:** a gap-filled or low-confidence period shows a low wetted-area value, and the manager reads it as "the river was measurably dry that month."

**Why it's dangerous:** a low number from a period with poor satellite coverage (cloud, sensor gaps) is a *data-availability* artifact, not necessarily a *hydrological* one. Conflating "we don't have good data" with "we observed dryness" could trigger an unwarranted management response (or, just as bad, mask a real drying event behind an averaged-in low-confidence period that happens to look normal).

**Required guard:** low-valid-observation flags must be visually and textually distinct from the metric value itself wherever the metric appears — never let a flagged, low-confidence value sit next to a high-confidence value with no visual difference.

### Danger 6 — Cross-reach or cross-year comparison without matching resolution/sensor/CRS context

**Naive reading:** "Reach A has more pools (N) than Reach B, so Reach A is more fragmented" — where A and B were processed from different imagery sources, resolutions, or CRS choices.

**Why it's dangerous:** patch-count-dependent metrics (N, NNI, gap, MESH tails, width distribution) are not comparable across different sensor resolutions — a coarser sensor systematically merges small pools that a finer sensor resolves separately. The spec (§8, §5.4) flags this explicitly as a guard requirement. A manager comparing two reaches monitored by different programs, or the same reach before/after a sensor change, could draw a fragmentation-trend conclusion that is actually a sensor-resolution artifact.

**Required guard:** every comparison surfaced to a manager (across reaches, across years, across data sources) must state whether resolution/source is held constant. If not, the comparison should carry a visible "not directly comparable" flag rather than being presented side by side as if it were.

### Danger 7 — NNI, if it ever reaches a manager document

The scientific audit (R5) recommends cutting NNI from v1 outright, or at minimum fully quarantining it from any publication surface, because Clark–Evans NNI uses the wrong statistical null for a linear river corridor and is least stable in exactly the end-dry regime managers care about most. **This report goes further for the manager-facing surface specifically: NNI must never appear in `docs/for-managers.md`, a manager dashboard, or any manager-facing report, under any framing, even as a caveated fallback.** A manager cannot be expected to hold "this number uses the wrong statistical assumption for a river" in their head — the only safe answer is absence, not a footnote.

---

## 4. What combinations matter more than single metrics?

A manager who is only ever shown single numbers will chronically over- or under-react. The following pairs/combinations are where the actual decision-relevant signal lives.

| Combination | What it reveals that neither metric alone does |
|---|---|
| **Dry-down rate + end-dry Refuge Area** | Whether the reach is drying fast *and* has little refuge left to fall back on (high risk), versus drying fast but still landing on ample refuge (lower risk), versus drying slowly onto very little refuge (a different, quieter risk that a dry-down-only view would miss entirely). |
| **APSEC trend + LPSEC trend** | Whether the reach is shrinking in area while staying longitudinally connected (contracting but intact) versus shrinking in area *and* losing longitudinal connection (fragmenting) — these call for different management responses (habitat protection vs. connectivity/passage intervention). |
| **N (number of pools) + LPI** | Many small pools of similar size (evenly fragmented) reads very differently from many pools dominated by one large one (effectively one refuge plus scattered remnants) — same N, opposite management story. |
| **Inter-pool gap + reconnection timing** | Whether pools that are far apart during the dry season reconnect quickly once flow resumes (temporary fragmentation, lower concern) or stay disconnected for a long lag (fragmentation that persists into the wet season, higher concern for fish passage / recolonisation timing). |
| **Refuge spatial stability + Refuge Area trend** | A shrinking-but-stable refuge (same locations, less area each year) is a different management target — protect and possibly augment those specific sites — than a shrinking-and-shifting refuge (area moving location year to year), which argues for protecting a broader zone rather than fixed sites. |
| **Any metric + valid-observation flag + composite-sensitivity flag** | Whether an apparent trend is a real hydrological signal or an artifact of data gaps / processing-method choice. This is not optional context — see §5. |
| **TCF / RC / DCI shown together, with their relationship stated** | Each measures a related but distinct notion of connectivity (a fixed spot's connection frequency over time vs. a network snapshot vs. a standard fragmentation-weighted index). Shown alone, any one invites "connectivity is X%" as if it were a single settled number; shown together with one sentence on how they differ, they triangulate a much more defensible connectivity story. |

**General rule for the manager document:** wherever the spec or scientific audit pairs a "headline" metric with a companion metric that guards against misreading it (dry-down + Refuge Area; width + the depth disclaimer; APSEC + LPSEC), the manager document must show them together by default, not as an optional drill-down. A manager who has to click through to find the guardrail metric will usually not click through.

---

## 5. Communicating uncertainty, low-valid-observation flags, composite sensitivity, CRS/length caveats, and width-not-depth

These five caveats are scientifically load-bearing (per the spec, §8, and the scientific audit) and cannot be dropped from manager docs — but they must be translated, not reproduced verbatim. Principle: **state the practical consequence, not the mechanism.**

- **Uncertainty (general).** Don't explain confidence intervals or classification-error propagation. Do say, in one line near any number: "this number comes from satellite classification and can be wrong by a modest margin, especially in murky or vegetated water — use it to see direction and scale of change, not as an exact measurement." Never present a HydroFragments output number with the false precision of, e.g., three decimal places in a manager table.

- **Low-valid-observation flags.** Don't explain the 70%/95% thresholds or the temporal gap-fill window. Do say: "some periods have a flag showing the satellite view was unusually incomplete that month — treat flagged values as a rough indication, not a confirmed reading, and don't compare a flagged value directly against an unflagged one." Flags must be visually obvious (an icon or shaded cell), not a value in a footnote column a reader has to cross-reference.

- **Composite sensitivity (`max_water` vs `median`).** Don't explain monthly compositing rules. Do say: "there are two ways to summarise a month's worth of satellite images into one number, and they can disagree — especially for the dry-down number. Where the two methods give meaningfully different answers, this is flagged, and the dry-down number should be read as a range, not a single figure." This is the single most important caveat to get right, because dry-down rate is simultaneously the headline metric managers most want and the metric the scientific audit identifies as most exposed to this bias (R7/V3, currently unquantified).

- **CRS / length caveats.** Don't explain equal-area projections or geodesic length calculation. Do say: "area and length figures are calculated in a projection chosen to keep areas accurate across the whole catchment; length figures (like LPSEC) can be very slightly distorted by this choice, more so for reaches far from the projection's centre — treat length-based percentages near or above 100% as a sign to check with the analysis team rather than a literal reading." (LPSEC exceeding 100% on braided or floodplain reaches is a known, non-bug behaviour per the scientific audit — this must be pre-empted in plain language or it will read as an error and undermine trust in the whole tool.)

- **Width-not-depth.** As in §3 Danger 2 — this cannot be a caveat appended once in an introduction; it must travel with the metric every time pool width appears, in the same sentence or immediately adjacent, because the misreading is intuitive and repeats every time a reader is not actively holding the caveat in mind.

**Cross-cutting formatting rule:** every one of these five caveats should have a single consistent visual marker (icon, colour, or fixed phrase) used identically everywhere it applies, so a manager learns the marker once and recognises it everywhere, rather than re-reading prose each time.

---

## 6. What `docs/for-managers.md` should contain

This expands the spec's own §12.1 scope (glossary, 3–5 worked narratives, one decision-support table, explicit non-inclusions) into a concrete outline, without pre-writing its content from invented numbers.

1. **One-paragraph framing.** What HydroFragments measures (surface-water extent and fragmentation from satellite imagery over time) and what it does not measure (flow, discharge, water quality, groundwater, depth, ecological condition directly). Set the boundary of the tool before any metric is introduced.
2. **Plain-language glossary** — one sentence per core-tier metric (§5.1 of the spec), written to the translations in §2 of this report. No formulas. No secondary/exploratory-tier metrics unless a genuine institutional use case is confirmed for them (per spec §12.1's own instruction to drop token inclusions).
3. **Decision-support table** — the table in §7 below, adapted with real output ranges once a validation catchment (e.g., Gilbert) is available; placeholders only until then.
4. **3–5 worked narrative templates** — see §8 below, using placeholders, not invented numbers, until derived from a validation catchment run.
5. **The five caveats from §5**, each stated once in full plain language early in the document, then referenced by its consistent visual marker everywhere it recurs.
6. **A short "what this tool cannot tell you" section** — explicit negative scope: no flow/discharge, no direct water-quality signal, no groundwater information, no guaranteed depth information, no certainty about any single flagged/low-confidence reading, no cross-source/cross-resolution comparability guarantee. This section is as important as the glossary — it pre-empts the most likely category of naive misreading (§3) at the point of first contact with the document, rather than relying on caveats a busy reader may skim past later.
7. **A one-line escalation path.** Who to contact (the analysis/science team) before a management decision leans on a dry-down number, a composite-sensitivity-flagged value, or a low-valid-observation-flagged value. Managers should be explicitly invited to check back rather than resolve an ambiguous or flagged reading themselves.
8. **Explicit exclusions**, restated from the spec so the boundary is visible in the document itself, not just in this audit: no formulas, no circularity arguments, no connectivity-index positioning debates (DCI vs PC vs IIC), no CRS/compositing mechanism explanations — all of that belongs in the spec and companion paper, not here.

This document should be short — the spec calls it "a separate, short, plain-language document," and a two-to-four page PDF-equivalent is the right size. Length itself is a caveat-compliance risk: a manager document that grows past a few pages will not be read in the moment it's needed.

---

## 7. Decision-support table

Placeholders (`[VALUE]`, `[THRESHOLD]`, `[REACH]`, `[PERIOD]`) stand in for figures that must be derived from a validation catchment (e.g., Gilbert) once real output ranges exist — no numbers here are invented.

| Metric or metric pair | Concerning pattern | Management question | Caveat |
|---|---|---|---|
| Dry-down rate (alone) | Shrinkage speed for `[REACH]` in `[PERIOD]` is markedly faster than that reach's own history at the same point in the season | Should an environmental water release be brought forward for this reach? | Read only alongside monthly-point count and composite method used (§5); do not derive a "days until dry" estimate from this number alone |
| Dry-down rate + end-dry Refuge Area | Fast shrinkage **and** end-dry Refuge Area below `[REACH]`'s own recent-year range | Is this reach at elevated risk of losing dry-season refuge capacity this year, ahead of other reaches competing for a limited environmental water allocation? | Composite-sensitivity flag must be checked before comparing across years or reaches |
| APSEC trend + LPSEC trend | Wetted-area fraction (APSEC) declining while wetted-length fraction (LPSEC) declines faster / drops earlier | Is the reach losing longitudinal connectivity before it loses overall area — i.e. is fragmentation, not just drying, the active process? | Both must be read against the same hydrological-year stage, not compared across different calendar months |
| Number of pools (N) + LPI | N rising while LPI is falling | The reach is breaking into many smaller water bodies rather than retreating to one dominant refuge — does the refuge-protection plan need to shift from a single-site to a multi-site strategy? | N is not comparable across different imagery sources/resolutions (§3 Danger 6) |
| Refuge Area (RA) trend, multi-year | RA at end of dry season below `[REACH]`'s own multi-year baseline for `[THRESHOLD]`+ consecutive years | Is this a persistent decline warranting a management response, or a single dry year within normal variability? | Always compare like-for-like hydrological-year stage; a single low year is not yet a trend |
| Refuge spatial stability + Refuge Area trend | Stability index falling (refuges relocating) while RA is roughly flat | Refuges are not shrinking, but are they moving — does the protected-area boundary need to widen rather than shrink? | Stability computed only where enough dry-end years of data exist; low-sample-year results should be flagged, not treated as equal-confidence to well-sampled years |
| Inter-pool gap + reconnection timing | Gap distances increasing **and** reconnection lag lengthening year over year | Is fish/species passage window shrinking — does timing of any managed flow event need to shift earlier to preserve passage opportunity? | Requires a usable channel skeleton; degrades to a weaker planar fallback when no drainage layer is available, and that degraded mode must be flagged, not silently substituted |
| Pool width distribution (alone or trending) | Any presentation implying width changes track water volume/storage change | (None — this is the pattern to actively avoid presenting as a standalone management signal) | Width is a surface-only, morphology signal; never equate to depth or stored volume (§3 Danger 2) |
| Any metric with low-valid-observation flag raised | A concerning-looking value coincides with a low-valid-observation flag for that period | Should this reading trigger action, or should it wait for the next well-observed period / direct confirmation? | Flagged values should generally not trigger a standalone management decision without independent confirmation |
| TCF / RC / DCI (any) | Any of the three read in isolation as "connectivity is `[VALUE]`%" | Is the network functionally connected enough to support movement/recolonisation this season? | Always present with the one-sentence distinction between the three (§4); never headline just one without the others as context if more than one is available |

---

## 8. Worked narrative templates

Placeholders only — these are structures for the manager document, to be filled in once a validation catchment run produces real ranges. None of the bracketed values below should be treated as illustrative real numbers; they are slots.

**Template 1 — Elevated refuge risk (dry-down + Refuge Area combination)**

> In `[REACH]`, surface water has been contracting at `[RATE_DESCRIPTOR: faster than / in line with / slower than]` this reach's typical dry-season pace for `[PERIOD]`. At the same time, the area of reliable dry-season refuge (Refuge Area) is `[TREND_DESCRIPTOR: below / within / above]` its usual range for this stage of the year. Together, this suggests `[REACH]` may have `[RISK_DESCRIPTOR: less refuge capacity than usual to absorb continued drying / an adequate refuge buffer despite faster-than-usual drying]`. *This reading depends on which monthly-summary method was used — see the composite-sensitivity flag for this reach before acting on it.*

**Template 2 — Fragmentation without overall area loss (APSEC/LPSEC/N/LPI combination)**

> `[REACH]`'s total wetted area (APSEC) has stayed `[roughly stable / declined moderately]` over `[PERIOD]`, but the reach's longitudinal connection (LPSEC) and pool count (N vs. LPI) suggest the water is `[becoming more broken into separate pools / remaining a single connected stretch]`. This points to `[fragmentation as the active process, independent of overall drying / no fragmentation signal beyond the expected seasonal pattern]`, which is a different management concern than area loss alone — it speaks more to fish/species passage and reconnection timing than to total habitat area.

**Template 3 — Refuge location stability (spatial stability + Refuge Area combination)**

> Comparing end-of-dry-season conditions across `[N_YEARS]` years, the locations that qualify as refuge in `[REACH]` have been `[largely the same places each year / shifting location from year to year]`, while the total refuge area has `[stayed roughly constant / trended down]`. This suggests that a protection strategy built around `[a small number of fixed, known refuge sites / a wider protected zone that can accommodate refuge locations moving between years]` is the better fit for this reach. *Years with unusually few valid satellite observations are excluded/flagged in this comparison — see the low-valid-observation note.*

**Template 4 — Connectivity/passage timing window (gap + reconnection timing combination)**

> In `[REACH]`, the typical dry-gap distance between pools has been `[increasing / stable / decreasing]` over the monitored period, and the lag between the start of the wet season and full reconnection has been `[lengthening / stable / shortening]`. If a managed flow event is planned to support fish passage or recolonisation, `[REACH]`'s reconnection pattern suggests the effective passage window is `[narrower / about the same / wider]` than in previous years, which may affect the timing of any planned release. *This reading requires a usable channel reference for this reach — confirm with the analysis team whether the degraded (no-drainage-layer) fallback was used before relying on it for timing decisions.*

**Template 5 — Data-confidence caveat wrapper (used alongside any of the above)**

> The readings above for `[REACH]` in `[PERIOD]` are based on `[HIGH / REDUCED]` satellite observation coverage for that period. `[If REDUCED:]` Treat this period's values as indicative rather than confirmed, and avoid comparing them directly against periods with full coverage. Before this reading is used to justify a management action, confirm with the analysis team whether additional or higher-confidence data has since become available.

---

## 9. Warnings that must appear in manager docs

These are non-negotiable — every one maps to a danger identified in §3 or a caveat in §5, and omitting any of them re-opens a specific, already-identified misreading path.

1. **HydroFragments measures surface-water extent from satellite imagery, not streamflow, discharge, water quality, groundwater, or depth.** State this before any metric is introduced, not after.
2. **Dry-down rate is a speed of visible-area shrinkage, not a hydrograph recession measurement, and should never be used to predict an exact drying date.** Always paired with the composite-method and monthly-point-count caveats.
3. **Pool width describes surface width only and must never be read as depth or water volume.** Repeated at every appearance of width, not stated once.
4. **A "refuge" classification depends on a chosen threshold and time window, and is not a permanent designation.** State the threshold and window whenever refuge status is shown.
5. **Any value flagged for low valid observations is less reliable than an unflagged value and should not be compared directly against one.** Flags must be visually distinct, not buried in a footnote.
6. **Two different monthly-summary methods can give different dry-down answers; where they disagree meaningfully, this is flagged, and the number should be read as a range.** This is the single highest-priority warning given the scientific audit's own unresolved finding (R7/V3) that this bias is real but not yet quantified.
7. **Numbers from different reaches, years, or data sources are not automatically comparable** if resolution, sensor, or projection differs — check before comparing.
8. **A single reading, on its own, is rarely actionable — compare against the same reach's own history at the same point in the season, and check the paired guardrail metric (§4) before drawing a conclusion.**
9. **This tool supports a management decision; it does not make one.** Any reading approaching a decision threshold should be confirmed with the analysis team before action, especially where a flag (low-observation, composite-sensitivity, degraded-fallback) is present.

---

## 10. Claims to avoid

Claims that must **not** appear in manager-facing material, because they overstate what the tool or the current evidence supports — cross-referenced to the scientific audit's own asserted-vs-demonstrated findings (§9, §11 of [`scientific_metrics_audit.md`](scientific_metrics_audit.md)) so manager docs never assert more than the science currently backs.

- **"This metric predicts/proves refuge risk."** Dry-down rate is motivating framing for refuge risk, not a demonstrated causal link to refuge survival outcomes (scientific audit V7 — unvalidated). Manager docs may say a fast dry-down "may indicate elevated risk," never that it "predicts" or "confirms" it.
- **"The river will be fully dry by [date]."** No metric in the register supports a precise drying-date projection; dry-down rate is a slope over sparse monthly points, not a forecast model.
- **"Pool shape (AWRe) tells us the pool type / drying mode."** This interpretation is asserted in the spec's own current language but not yet empirically shown (scientific audit V1/V4) — manager docs should describe shape trends descriptively, not as a diagnosed mechanism.
- **"Wider pools mean more water / better refuge."** Width is not a volume or depth proxy — see §3 Danger 2, §9 item 3.
- **"This location is officially/permanently a refuge."** Refuge status is threshold- and window-dependent, not a fixed designation — see §3 Danger 3, §9 item 4.
- **"Fragmentation (N, MESH) is directly comparable between this reach and that other program's reach."** Not comparable across differing sensors/resolutions — see §3 Danger 6, §9 item 7.
- **"The tool measures ecological health / condition."** HydroFragments measures surface-water extent and fragmentation. It does not measure water quality, species presence, or ecological condition directly, and manager docs must not imply otherwise even by omission.
- **"A low reading this month confirms the river is drying faster than normal."** Not without checking the low-valid-observation flag first — a low reading during poor satellite coverage is a data-availability artifact until confirmed otherwise.
- **"Two connectivity numbers (e.g., RC and DCI) that disagree mean the tool is inconsistent."** They measure related but distinct things; disagreement is expected and informative, not an error — manager docs must pre-empt this reading rather than let a manager discover the "inconsistency" and lose trust in the tool.
- **Any number presented to more precision than the underlying satellite classification supports.** False precision (e.g., three decimal places) reads as certainty the tool does not have.

---

## 11. Summary

The metric register has a genuine manager-facing core (§1) — occurrence/refuge area, APSEC/LPSEC, N/LPI, dry-down rate, refuge stability, reconnection timing — but almost every one of them is dangerous in exactly the naive reading a manager without scientific training would default to (§3), and the danger is highest precisely for the metric managers most want (dry-down rate). The fix is not fewer metrics; it is that **no headline number should ever be shown without its paired guardrail metric and its relevant caveat marker** (§4, §5) — the single-number-in-isolation is the actual hazard, not any individual metric's science. `docs/for-managers.md` (§6) should be short, should lead with what the tool does *not* measure, and should treat the five caveats (uncertainty, low-valid-observation, composite-sensitivity, CRS/length, width-not-depth) as permanent visual companions to their metrics rather than a one-time introduction. Nothing in this report should be read as a science critique — see [`scientific_metrics_audit.md`](scientific_metrics_audit.md) for that — but manager docs must never claim more confidence than that audit's own asserted-vs-demonstrated table (§9, §11 there) currently supports.
