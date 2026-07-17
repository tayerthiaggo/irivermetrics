# HydroFragments v1.2 — validation status

This is the single asserted-vs-demonstrated source of truth for HydroFragments,
tracking spec section 6.18 and the scientific claim matrix in
`docs/audit/implementation_plan.md` section 7. Every row states whether a
claim is **Asserted** (design-time hypothesis, not yet checked against data)
or **Demonstrated** (checked against a real run, with a linked `run_id` and
evidence file). No row is ever deleted when it moves from Asserted to
Demonstrated — the row is updated in place with a result and a link so the
history of what changed stays visible.

A manager document, paper draft, or headline claim must never assert more
confidence than this table currently supports (per
`docs/audit/scientific_metrics_audit.md` section 16.1, finding F-5).

## How to read the evidence links

Every Demonstrated row cites a `run_id`. That `run_id` resolves to:

- a row in `validation/results/*.csv` (the machine-readable result table), and
- `validation/results/manifests/<run_id>.json` (the immutable run manifest —
  config hash, package version, input fingerprint — produced by the real
  `hydrofragments.analyze()` pipeline, not a hand-written number).

Regenerate all `validation/results/` artifacts with
`python validation/run_fitzroy_validation.py`. `tests/validation/` checks the
committed artifacts trace correctly; it does not re-run the (slow, full
480-month) analysis on every test invocation.

## Claim status table

| ID | Claim | Status | Evidence |
|---|---|---|---|
| V1 | AWRe and AWMSI are orthogonal (non-redundant) shape axes | **Demonstrated** | `run_id=d6ad5549dcad4c98863e3a4379927b03`, Pearson r(AWRe, AWMSI) = **-0.291** over 458 months with water present, Fitzroy (Kimberley) catchment, 1987-2026. Weak negative correlation — the two axes move largely independently. See `validation/results/v1_v2_shape_correlation.csv`. |
| V2 | LPI and MESH are sufficiently non-redundant to keep both | **Demonstrated — gate FAILED, MESH disabled** | Same run. Pre-registered hard gate: disable MESH if Pearson r(LPI, MESH) > 0.9. Measured r = **0.906** over 458 months. **Gate failed.** Per the pre-registered rule (spec §6.18 row 2, checklist item 12), MESH must not ship as an independent core/secondary metric alongside LPI for this catchment — it is redundant with LPI in practice, not just in principle. See `validation/results/v1_v2_shape_correlation.csv` and `evaluate_mesh_correlation_gate` in `hydrofragments/metrics/patches.py`. |
| V3 | `max_water` composite bias measurably flattens/delays dry-down (extent contraction) | **Asserted, not demonstrated** | Blocked by U3/Q3 (`docs/audit/decisions.md`): the approved Fitzroy validation cube (`data/wofs_monthly_masks_1986_2026.zarr`) is a single monthly product, not a dual `max_water`/`median` composite pair. No dual-composite comparison can run against this fixture. Do not use this claim as a manager/paper headline until a dual-composite source is available (implementation_plan.md Milestone 9 acceptance criterion). |
| V4 | AWRe "tracks drying mode / discriminates pool type" | **Asserted, not demonstrated** | No independent pool-type or drying-mode ground truth is available for this catchment. Formula and orthogonality (V1) are demonstrated; the ecological interpretation is not. Do not state this as fact in manager or paper text. |
| V5 | Pool width behaves as morphology, not a rasterisation artefact, at operational (30 m) resolution | **Asserted, not demonstrated** | No field/bathymetric comparison data available. Resolution-floor guard (`width_resolution_floor_pixels`) is implemented and suppresses sub-floor widths, but the underlying morphology-vs-artefact question is unchecked. |
| V6 | RC/TCF/DCI behave sensibly relative to a reference (`riverconn`/Conefor) | **Demonstrated (formula parity)** | HydroFragments' pure-Python length-weighted `RC_pair` (the DCI form, spec §6.17: `DCI_t = 100·Σ len_i·len_j·c_ij / (Σ len_i)²`, `compute_length_weighted_rc_pair`) was benchmarked against the independently implemented `riverconn::index_calculation` (R package, Baldan et al. 2022, CRAN v0.3.31) on the **real Fitzroy (Kimberley) reach graph** — 282 wet-capable reach nodes, 31 structural topology edges, 251 weakly-connected structural components, real reach lengths and one representative-month (index 178) active-edge set (3 active edges). Both implementations were fed the *identical* graph + active-edge configuration. Result: Python = **0.6235076%**, riverconn = **0.6235076%**, absolute difference **4.4×10⁻¹⁶** percentage points (floating-point epsilon), agreement **100%**. The two independently written implementations of the Cote et al. 2009 DCI agree to machine precision. Evidence: `validation/results/benchmarks/v6_dci_benchmark.csv` (comparison), `validation/results/benchmarks/v6_riverconn_raw.csv` (raw riverconn output, version-stamped), regenerate with `python validation/run_dci_benchmark.py` (calls `validation/dci_benchmark.R` via Rscript). **Scope note:** this demonstrates *formula/reference parity* only. Per Q4, DCI stays **citation-only** — passing V6 does not by itself ship DCI as a runtime metric; that remains a separate maintainer decision. The low absolute DCI value reflects a genuinely fragmented dry-season snapshot (few active edges), which is the metric behaving correctly, not a defect. |
| V7 | Dry-down / extent-contraction rate is a meaningful refuge-risk indicator | **Asserted, not demonstrated** | No linkage analysis between contraction slope and an independent refuge/pool-survival outcome has been run. Never state this as a predictive or causal claim (see forbidden-claims list below). |
| V8 | Persistence-based HY detection differs from Tayer 2025/2026 rainfall-based HY | **Demonstrated (maintainer attestation)** | Per `docs/audit/decisions.md` Q7: maintainer ran a manual algorithmic comparison of persistence-based HY (via `hydroseason`, package `0.1.0`) against Tayer 2025/2026 rainfall-based HY on both Fitzroy and Gilbert extent series — **100% agreement** on HY boundary/season assignment, recorded 2026-07-16. No separate machine-readable artifact file was generated for this pass; the attestation itself is the evidence of record until a scripted comparison is added. |

## Publishable without new data (logical/algebraic results)

Per `docs/audit/scientific_metrics_audit.md` section 16.2, the following do
not require the validation run above — they are reasoning, not empirical
claims, and may be stated as findings:

- The fixed-denominator/circularity argument and the `PF`/`PLF`/`AWMPA`/`AWMPL`/`AWMPW` drops.
- The connectivity positioning against DCI/PC/IIC and the `PCF` → `TCF` rename (citation-only; no connectivity code ships in this release).
- The reproducibility discipline (config hashing, composite-sensitivity flagging) as an engineering contribution.

## What this table does not cover

- Milestone 12 (CUDA backend) is outside this validation pass. Milestone 11
  (connectivity: RC/TCF/DCI) now exists, and V6's DCI-form reference parity
  against `riverconn` is Demonstrated (formula parity only — DCI remains
  citation-only per Q4; parity does not by itself ship it as a runtime metric).
- This table reflects one validation catchment (Fitzroy/Kimberley). A result
  demonstrated here is evidence for that catchment, not a general proof for
  every intermittent river HydroFragments might run against.
