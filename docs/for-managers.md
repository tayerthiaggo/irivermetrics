# HydroFragments — a guide for water managers

**What this tool measures:** surface-water extent and fragmentation in a river
corridor, from satellite imagery over time.

**What this tool does not measure:** streamflow, discharge, water quality,
groundwater, depth, or ecological condition directly. Every number below
describes what was *visible from above*, not what was flowing, stored, or
alive underneath it.

This document is short on purpose. If a number here looks important enough
to act on, the last section tells you who to ask before you do.

---

## 1. Plain-language glossary

Each core metric answers one management question, always paired with the
caveat that guards against its most common misreading.

- **Occurrence frequency** — how often a location holds water, out of the
  times it was actually observed (not out of every possible month — some
  months have poor satellite coverage and are excluded, not counted as dry).
- **Refuge Area (RA)** — the area classified as holding water almost every
  time it was observed, using an adjustable cutoff percentage. This is *how
  this location behaved over the period analysed*, not a permanent
  designation — a different cutoff, time window, or more data can change
  which areas qualify.
- **APSEC (wetted area fraction)** — how much of the fixed reach area is
  holding water right now, relative to the whole corridor area (which
  includes dry land, not just the channel).
- **Number of pools (N)** — a first-glance fragmentation signal: is the
  reach one continuous water body, or many separate fragments?
- **LPI (largest patch index)** — the share of the total wetted area held by
  the single largest water body. Tells you whether there is one dominant
  refuge or many small, scattered ones.
- **AWRe (pool shape signal)** — long-and-thin vs. round-and-compact pools,
  averaged across the reach. Shape changes can accompany active drying, but
  shape alone does not confirm what is causing them — see the validation
  status note below.
- **Pool width** — how wide the water is at its surface. **Width tells you
  nothing about depth or how much water is stored** — a pool can be wide and
  centimetres deep, or narrow and metres deep.
- **Surface-water contraction rate** (formerly called "dry-down rate") — how
  quickly the wetted area is shrinking, month to month, during the dry
  season. This is a slope of *visible area*, not a flow or discharge
  measurement, and it does not give a precise date on which a reach will
  become fully dry.

---

## 2. Reading a number: five caveats that travel with every metric

1. **Uncertainty.** Every number comes from satellite classification and can
   be wrong by a modest margin, especially in murky or vegetated water. Use
   it to see direction and scale of change, not as an exact measurement.
2. **Low satellite coverage.** Some periods have unusually incomplete
   satellite coverage. A flagged value is a rough indication, not a
   confirmed reading — don't compare a flagged value directly against an
   unflagged one. When coverage across the whole analysis is too low, the
   run itself carries a warning recommending the data be pre-processed
   (gapfilled) before re-running — see the note at the end of this section.
3. **Two ways to summarise a month.** There is more than one way to turn a
   month of satellite passes into one number, and the choices can disagree —
   especially for the contraction rate. Where they disagree meaningfully,
   this is flagged, and the number should be read as a range, not a single
   figure.
4. **Projection choice.** Area and length figures use a projection chosen to
   keep areas accurate across the whole catchment. Length-based percentages
   near or above 100% are a sign to check with the analysis team, not
   necessarily an error.
5. **Width is not depth.** Repeated here because it is the easiest number to
   misread: pool width describes surface width only, never depth or stored
   volume.

**A note on gapfilling.** HydroFragments does not gapfill (fill in missing
satellite observations) itself. If a run's baseline coverage is too low, it
recommends pre-processing the imagery with the companion tool
WaterMask-TSFill before re-running, rather than silently working around the
gap. If your data has already been gapfilled upstream, tell the analysis
team so the run is configured with `gapfill: true` — that setting suppresses
the coverage recommendation because HydroFragments trusts the declaration
rather than re-checking it.

---

## 3. What this tool cannot tell you

- No flow, discharge, or streamflow measurement of any kind.
- No direct water-quality signal.
- No groundwater information.
- No guaranteed depth information — width and area are surface-only.
- No certainty about any single flagged or low-confidence reading.
- No comparability guarantee across different satellite sources or
  resolutions — a coarser sensor will systematically merge small pools that
  a finer sensor resolves separately.
- No causal or predictive claim: a fast contraction rate does not prove a
  refuge is at risk, and no number in this tool projects an exact drying
  date.

---

## 4. Reading combinations, not single numbers

A single number, read alone, is rarely actionable. Always check it against
the reach's own history at the same point in the season, and check the
paired guardrail metric before drawing a conclusion:

| Combination | What it reveals |
|---|---|
| Contraction rate + end-dry Refuge Area | Whether the reach is drying fast *and* has little refuge left (higher concern), or drying fast but still landing on ample refuge (lower concern). |
| APSEC trend + N/LPI trend | Whether the reach is shrinking in area while staying one connected body, or shrinking *and* breaking into fragments — these call for different management responses. |
| Any metric + low-coverage flag | Whether an apparent trend is a real signal or an artefact of poor satellite coverage that period. |

---

## 5. Current validation status

Full detail: [`docs/validation_status.md`](validation_status.md). Summary for
this document's audience:

- The claim that AWRe (pool shape) and AWMSI (a separate shape index) measure
  genuinely different things has been checked against real data for the
  Fitzroy (Kimberley) catchment — they behave as largely independent
  signals.
- The MESH fragmentation index failed its own pre-registered redundancy
  check against LPI for this catchment (they moved together too closely) and
  is not shown as an independent number here.
- The contraction-rate composite-sensitivity check, the check on whether
  pool width behaves as real morphology (not a satellite-resolution
  artefact — width is never a depth measurement either way), and the
  refuge-risk link have **not yet been run** for this catchment. Treat those
  readings as motivating context, not settled findings, until
  `docs/validation_status.md` is updated.

---

## 6. Escalation path

Before a management decision leans on a contraction-rate number, a
composite-sensitivity-flagged value, or a low-coverage-flagged value, check
back with the analysis team. This tool supports a management decision; it
does not make one.
