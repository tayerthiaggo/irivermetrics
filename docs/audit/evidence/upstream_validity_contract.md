# Upstream validity contract evidence (U2 / Q1 / U3)

**Captured:** 2026-07-10T21:10:00+08:00  
**Sources read:** `WaterMask-TSFill/watermask_tsfill/contracts.py`, `WaterMask-TSFill/docs/outputs.md`, audit chain `evidence_packet.md` → `spec_compliance.md` → `adversarial_synthesis_2.md`

## Canonical WaterMask-TSFill output (`water_cube.zarr`)

Four aligned variables on `(time, y, x)`:

| Variable | Dtype | Semantics |
|---|---|---|
| `water_mask` | uint8 | 0=dry, 1=water, 254=outside AOI, 255=invalid/unresolved |
| `confidence` | uint8 | 0–100 trust; 255=N/A |
| `method_flag` | uint8 | Provenance vocabulary (`MethodFlag` in contracts) |
| `observed` | bool | Whether pixel is a native observation per upstream rules |

**Spatial / temporal contract (from `DataContract` / `ZarrSchema`):**

- CRS: EPSG:3577 (Australian Albers equal-area), 30 m
- Cadence: **monthly** (`temporal_cadence: "monthly"`)
- Time chunk default: 12

## How `observed` is derived (upstream implementation)

From `derive_observed_mask()` in `contracts.py`:

> True where a native dry/water observation survived unchanged in `filled`.

Logic:

1. `raw_valid = (original == water) | (original == dry)`
2. `unchanged = (original water stays water) | (original dry stays dry)`
3. `observed = raw_valid & unchanged`

Post-processing method flags (`post_connectivity_filter`, `post_temporal_anomaly`, `post_morph_closing`, `post_preserved_water`) can yield **filled** pixels with `observed=False` even when `water_mask` is 0/1.

`validate_canonical_zarr_contract()` additionally rejects `observed=True` paired with non-observed `method_flag` values (except `observed` and `post_preserved_water`).

## Candidate validity denominators and reliability diagnostics for HydroFragments

| Policy ID | `valid_obs` definition | Evidence for | Evidence against |
|---|---|---|---|
| **P-native** | `observed == True` only | Matches upstream `observed` band intent; strict; avoids counting gap-filled pixels in occurrence denominator | Ignores high-confidence fills that may be scientifically usable; ~4.9% resolved pixels with `observed=False` cited in `implementation_plan.md` on a six-year test cube (not re-measured in M0) |
| **P-resolved** | `water_mask ∈ {0,1}` excluding 254/255 | Simple mask-based validity | Treats gap-filled and post-processed pixels as equally valid observations — conflicts with upstream `observed` semantics |
| **P-provenance** | `method_flag` in allowed production-fill set OR `observed` | Uses full provenance band | Requires locked allow-list; must not count `unresolved`/`outside_aoi`; needs author sign-off and sensitivity tables |

**Maintainer direction (2026-07-11):** validity must account for the provenance of observations, and outputs should make data reliability visible: how much was actually observed vs filled, and which fill methods contributed.

Minimum reliability diagnostics required before Q1 closure:

| Diagnostic | Purpose |
|---|---|
| `native_observed_fraction` | Share of reportable pixels/months from unchanged native observations |
| `filled_resolved_fraction` | Share of reportable pixels/months resolved by filling/post-processing |
| `method_flag_composition` | Breakdown by `observed`, production-fill, post-processing, unresolved, outside AOI |
| `confidence_distribution_by_method` | Whether high-confidence filled pixels dominate or whether outputs rely on low-confidence fills |
| `occurrence_policy_sensitivity` | Difference in occurrence/RA under P-native vs selected provenance-aware policy |

Open design question: the denominator policy may be strict native-only for science metrics while still emitting filled/provenance reliability diagnostics, or it may allow a restricted provenance-qualified denominator. This must be decided with sensitivity evidence.

**Real-data update (2026-07-14):** the sensitivity run and reliability diagnostics called for above have now
been executed against a real Fitzroy monthly cube — see
`docs/audit/evidence/validity_reliability_report.md`. Headline findings:

- The delivered `wofs_monthly_masks_1986_2026.zarr` artifact is **simpler** than the four-variable contract
  documented above: it has **only** `water_mask` (values `-2` outside AOI, `-1` unobserved, `0` dry, `1`
  wet). There is no per-pixel `confidence` or `method_flag` band in this delivery, so **P-provenance is not
  implementable against it** — it remains a documented target for a future upstream delivery, not a policy
  HydroFragments can adopt today.
- P-native (`{0,1}` denominator) vs P-resolved (`{-1,0,1}` denominator) wet-fraction diverges by a median of
  0.02 pp but up to **27.1 pp** in low-coverage months; divergence correlates with coverage (r = −0.31).
- The upstream `inserted_months` attribute **undercounts** unreliable months: 28 of 465 "source" months
  (6.0%) have `observed_frac_of_aoi < 0.50`, including 3 months at exactly 0% that are not flagged as
  inserted. Any reliability diagnostic must be computed empirically per period, not read from upstream
  attributes.
- Recommended (pending sign-off) policy: P-native denominator + mandatory `observed_frac_of_aoi`
  diagnostic + `low_confidence` flag below 0.70 coverage. Full rationale in the report above.

## Compositing ownership (U3 / Q3)

**Evidence from upstream docs (`outputs.md`):**

- Primary product is a **single filled monthly** `water_cube.zarr`.
- No second monthly composite (`median` vs `max_water`) is emitted as a standard variable.

**Implication for HydroFragments dry-down / V3:**

- Dual-composite dry-down **cannot** be reconstructed from `water_cube.zarr` alone.
- Requires either:
  - raw/sub-monthly observations (e.g. raw WOfS flags Zarr on disk at `.../raw_wofs_flags_gilbert_1986_2026.zarr`), **or**
  - caller/upstream supplying both `max_water` and `median` monthly products with provenance.
- Why this matters: the scientific audit's dry-down validation claim is not merely "compute one slope." It asks whether the slope changes when monthly water is summarised as `max_water` versus `median`. A single monthly product can support a provisional extent-contraction summary later, but it cannot quantify the composite-sensitivity claim.

**M0 status:** U3 **not closed** — dual-composite path not demonstrated on validation catchment.

**Real-data update (2026-07-14):** maintainer confirmed raw sub-monthly WOfS observations for the Fitzroy
catchment can be pulled directly from DEA STAC, giving a concrete path to build both `max_water` and
`median` monthly composites and test dry-down sensitivity empirically. This is not yet executed — it is a
dynamics-tranche (V3) task, not required to unblock the M0-scoped minimal core. Recorded here so the path
is not re-discovered later.

## Required evidence to close Q1/U2

1. Locked provenance-aware validity policy: native-only denominator plus diagnostics, or restricted provenance-qualified denominator.
2. Sensitivity run on a real `water_cube.zarr`: occurrence/RA under P-native vs selected policy; magnitude report archived under `docs/audit/evidence/`.
3. Reliability diagnostic schema approved (`native_observed_fraction`, fill-method composition, confidence summaries).
4. Locked `validity.policy` name + version in `decisions.md` with approval date.

**Status:** items 1–3 now have evidence — see `docs/audit/evidence/validity_reliability_report.md`. Item 4
(locking the policy name/version with an approval date) is the only remaining step, pending maintainer
sign-off on the recommended policy in that report.
