# HydroFragments v1.2 — Audit Intake Manifest (Milestone 0)

**Recorded:** 2026-07-10T21:10:00+08:00  
**Reader / model:** Composer (Cursor agent)  
**Scope:** Milestone 0 evidence and Decision Gate 0 intake only  
**Gate status:** **OPEN** — see unresolved references below; do not treat Decision Gate 0 as closed.

## Ingestion rule

Every Markdown file under `docs/audit/` was read in full at intake time, plus `docs/HydroFragments_v1.2_spec.md` and `docs/audit_implementation_plan.md`. Summaries were not substituted for raw audit files.

## Files ingested

| Path | SHA-256 | Read timestamp (UTC+8) | Notes |
|---|---|---|---|
| `docs/audit/adversarial_synthesis.md` | `c0060f55e97dcd9877a04d6d56bea3d17d600b3ae97908a7959b60e851158500` | 2026-07-10T21:10:00+08:00 | First-pass synthesis; minimal v1.2 scope |
| `docs/audit/adversarial_synthesis_2.md` | `466e67b1204cc5931d0143e2c5800d7aa6f34bb4e3c9fcaf4c38221ce0301473` | 2026-07-10T21:10:00+08:00 | Tighter scope; G0+ evidence-before-decisions |
| `docs/audit/dask_cuda_audit.md` | `ce201952f4ce1d9d1c3d52744b3069985614869766adeb9b12a1010a797d5666` | 2026-07-10T21:10:00+08:00 | Normative compute audit |
| `docs/audit/dask_cuda_audit_adversarial.md` | `f1e5340aab91ca7a0fb1f13631f29e0857f5efede428c1293e5590e1bb7356ed` | 2026-07-10T21:10:00+08:00 | Non-normative CUDA stress input |
| `docs/audit/docs_audit.md` | `f563cdd2ef664c55b15cfeb9621371c4fbf148ff3c5b509cb3e041d680eda13b` | 2026-07-10T21:10:00+08:00 | Docs/packaging drift |
| `docs/audit/evidence_packet.md` | `32874658c77f251c381056a453d824244db4ab4222f57040f52754d0ea6b092f` | 2026-07-10T21:10:00+08:00 | Repo evidence packet |
| `docs/audit/execution_checklist.md` | `823275db8498d555be6a6bf75152b1aa05835dc4b3bf6b5ef60fa604af847413` | 2026-07-10T21:10:00+08:00 | Agent execution checklist |
| `docs/audit/implementation_plan.md` | `f0050fb9a3fdaeb33ac206cce8a99a2f37d7c80bd71d040928deab05c59826ec` | 2026-07-10T21:10:00+08:00 | Stage 8 implementation plan |
| `docs/audit/manager_interpretation_audit.md` | `2638f42734cfcc3ce53de66bc5818e78c69b91c3698dc2b74ff4b19bed0c3652` | 2026-07-10T21:10:00+08:00 | Manager-facing interpretation |
| `docs/audit/repo_triage.md` | `3d5a239054b2d04a5f1251ebe8522fb92eaaa1c6488ea9e3a62073b969c95964` | 2026-07-10T21:10:00+08:00 | First-pass triage |
| `docs/audit/scientific_metrics_audit.md` | `871dcd5378b133e6f43f3cd737da35cb978acfa234a21d2145d9e7615edab7e7` | 2026-07-10T21:10:00+08:00 | Scientific defensibility |
| `docs/audit/spec_compliance.md` | `bfa8f3e0d57bcdb6ccce508eeb245a1e0ac8d6961f42d789c7a1a46d0edae360` | 2026-07-10T21:10:00+08:00 | Code/spec compliance |
| `docs/HydroFragments_v1.2_spec.md` | `a0be1d68646e21e13848bcd923530a96aa5006476a78956fba04e51bf04c3b17` | 2026-07-10T21:10:00+08:00 | Locked v1.2 contract |
| `docs/audit_implementation_plan.md` | `74af1a6642aa45188d222eab0af567641988a6fb2d9dcd644685c7cc6e3074bc` | 2026-07-10T21:10:00+08:00 | Planning artifact; hydroyear external |

## Cross-audit conflicts registered (not closed here)

| ID | Conflict | Blocking? |
|---|---|---|
| C1 | Rebrand timing vs API freeze | Yes — see `decisions.md` Q5/Q10 |
| C2 | CUDA ambition (`dask_cuda_audit` vs adversarial CUDA doc) | Yes — compute sequencing |
| C3 | Dry-down / dual-composite feasibility vs monthly-only upstream | Yes — U3/Q3 |
| C4 | HY algorithm ownership (spec vs external `hydroseason` repo) | Yes — Q7 |
| C5 | Drainage contract missing vs channel metrics in spec | Yes — U4/Q6 |
| C6 | Validity semantics (`observed` denominator) | Yes — U2/Q1 |
| C7 | NNI fate (exploratory vs cut) | No for M0 — audits converge on cut |
| C8 | DCI implement vs cite-only | Partial — Q4 |
| C9 | Legacy compatibility vs clean v1.2 schema | Yes — Q5 |
| C10 | Docs honesty vs vaporware | Process — Milestone 8 |

## Unresolved decision references (Decision Gate 0)

Updated 2026-07-14 after real Fitzroy `water_mask` cube and drainage dataset evidence was ingested (see
`docs/audit/evidence/validity_reliability_report.md`, `docs/audit/evidence/drainage_inventory.md`).
U1, U4/Q6, U7, Q2, Q4, Q5, Q8, Q10 are now `approved` in `docs/audit/decisions.md`. Remaining rows:

- **U2 / Q1** — evidence delivered (real-cube sensitivity + reliability diagnostics + seasonal MNAR analysis showing coverage is lowest exactly when wetness peaks); only maintainer sign-off on the recommended P-native + `observed_frac_of_aoi` + 0.70-threshold + season-stratified-temporal-estimator policy remains
- **U3 / Q3** — minimal-core deferral approved; DEA STAC identified as viable raw sub-monthly evidence source for dynamics/V3 dual-composite testing, not yet executed (non-blocking for M0/minimal core)
- **Q7** — HY algorithm authority (external package is sibling `hydroseason`; HydroFragments only adapts/calls it; version pin + V8 comparison not locked; blocked on `hydroseason` contract, not on local HY implementation)
- **Q9** — config hash golden rules (design proposed; cross-platform golden tests not run; blocked on test infra, not on maintainer action)

## Real data assets ingested (2026-07-14)

| Path | SHA-256 (`.zmetadata` for the zarr) | Notes |
|---|---|---|
| `data/wofs_monthly_masks_1986_2026.zarr` | `c69f7e8b0706496...9f7f3785420c794bf459fa9c157e73446ba2e4ced1a36e790` | Confirmed byte-identical to maintainer-supplied `D:\RLH\5.6\data_local\raw\WaterMask-TSFill\cache\wofs_monthly_masks_1986_2026.zarr` |
| `data/fitzroy_kimberley_drainage.gpkg` | `004442d0a65a7eeb51a335dbaa621e281f610080b31e7ae05ee9980a46dc3b3a` | AHGF-style drainage extract for the Fitzroy AOI |
| `data/fitzroy_kimberley_aoi.geojson` | not hashed (small, non-decision-bearing AOI reference polygon) | WGS84 AOI polygon; reproject to EPSG:3577 at load time |

## Milestone 0 artifacts produced

| Artifact | Path |
|---|---|
| Intake manifest | `docs/audit/intake_manifest.md` (this file) |
| Decision register | `docs/audit/decisions.md` |
| Fixture inventory evidence | `docs/audit/evidence/fixture_inventory.md` |
| Upstream validity evidence | `docs/audit/evidence/upstream_validity_contract.md` |
| Real-data validity/reliability report | `docs/audit/evidence/validity_reliability_report.md` |
| Per-month reliability raw data | `docs/audit/evidence/validity_reliability_per_month.csv` |
| Calendar-month seasonal MNAR raw data | `docs/audit/evidence/validity_reliability_by_calendar_month.csv` |
| Drainage inventory evidence | `docs/audit/evidence/drainage_inventory.md` |
| Regression baseline evidence | `docs/audit/evidence/regression_baseline.md` |
| Fixture inspection tests | `tests/contracts/test_fixture_characterisation.py` |
| Read-only inspector | `tests/contracts/fixture_inspector.py` |

## Digest invalidation rule

If any ingested file digest changes after this manifest, affected milestones must re-read the file and re-approve decisions that cite it.

---

## Milestone 7 intake refresh

**Recorded:** 2026-07-15T10:02:41+08:00  
**Reader / model:** Codex GPT-5  
**Scope:** Milestone 7 tidy outputs, manifests, comparison guards, and export isolation  
**Decision status:** Decision Gate 0 is closed. Q7 does not affect M7. Q9 still awaits formal register approval; its required cross-platform golden hash tests exist in `tests/contracts/test_hashing.py` and pass.

M7-relevant sections were read directly from the raw files; summaries were not used in their place.

| Path | SHA-256 | M7 relevance |
|---|---|---|
| `docs/audit/implementation_plan.md` | `f0050fb9a3fdaeb33ac206cce8a99a2f37d7c80bd71d040928deab05c59826ec` | Normative architecture, schema, M7 acceptance |
| `docs/audit/execution_checklist.md` | `823275db8498d555be6a6bf75152b1aa05835dc4b3bf6b5ef60fa604af847413` | Exact M7 prompt and test requirements |
| `docs/audit/decisions.md` | `282df696b96372ffa41deb69970342d81e2838191328b101c04eb74f52073596` | Q5 legacy policy and Q9 hashing status |
| `docs/HydroFragments_v1.2_spec.md` | `a0be1d68646e21e13848bcd923530a96aa5006476a78956fba04e51bf04c3b17` | Output schema and comparison guards |
| `docs/audit_implementation_plan.md` | `74af1a6642aa45188d222eab0af567641988a6fb2d9dcd644685c7cc6e3074bc` | Raw planning constraints |
| `docs/audit/spec_compliance.md` | `bfa8f3e0d57bcdb6ccce508eeb245a1e0ac8d6961f42d789c7a1a46d0edae360` | Output/config/test gaps |
| `docs/audit/repo_triage.md` | `3d5a239054b2d04a5f1251ebe8522fb92eaaa1c6488ea9e3a62073b969c95964` | Legacy wide-output and dropped-metric risks |
| `docs/audit/adversarial_synthesis.md` | `c0060f55e97dcd9877a04d6d56bea3d17d600b3ae97908a7959b60e851158500` | No-hybrid schema and reproducibility gates |
| `docs/audit/adversarial_synthesis_2.md` | `466e67b1204cc5931d0143e2c5800d7aa6f34bb4e3c9fcaf4c38221ce0301473` | Tight core-output scope |
| `docs/audit/dask_cuda_audit.md` | `ce201952f4ce1d9d1c3d52744b3069985614869766adeb9b12a1010a797d5666` | Export isolation and geometry-memory guard |
| `docs/audit/manager_interpretation_audit.md` | `2638f42734cfcc3ce53de66bc5818e78c69b91c3698dc2b74ff4b19bed0c3652` | Cross-source/resolution comparison refusal |
