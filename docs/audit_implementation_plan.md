# HydroFragments v1.2 - Audit and Implementation Planning

Status: planning artifact. No implementation instructions in this file should be treated as approval to edit code.

Source read:

- `docs/HydroFragments_v1.2_spec.md`
- `README.md`
- `pyproject.toml`
- `docs/architecture.md`
- `docs/module2.md`
- `ecofragments/main.py`
- `ecofragments/utils/calc_metrics.py`
- `tests/test_unit_metrics.py`
- `tests/test_integration.py`

Likely upstream input source:

- `D:\RLH\5.6\repos\WaterMask-TSFill`
- Expected role: produces binary water-mask time series for HydroFragments metric calculation.
- Treat as a reference integration point during audit. Do not assume its output schema without inspecting it in a later phase.

## Scope Decision: hydroyear and hydrozones

- **hydroyear** — **out of scope for this repo.** Will be developed as its own standalone package/repo (reused across ≥2 repos; not to be duplicated here). Do not implement `hydroyear/` in this codebase. The v1.2 spec's suggested module layout (`docs/HydroFragments_v1.2_spec.md`, module layout section) lists `hydroyear/` as an internal module — that line is now stale against this decision and needs reconciliation (likely: HydroFragments consumes hydroyear as an external dependency/import rather than owning the module) before Stage 8 implementation planning treats it as buildable here.
- **hydrozones** (in-channel / off-channel / seasonally-flooded / marginal-floodplain classification, spec §3 `zones/`) — **not yet developed.** Keep in this repo's plan as-is for now (see `zones/` in the module layout and the Recommended Minimal v1.2 Implementation Order, workstream 5). Extraction to its own repo is a candidate for later, once the logic is proven and reused, but is not a current decision.

## Current Repo Snapshot

Code is still mostly `ecofragments` / iRiverMetrics-era implementation. Current public API is `calculate_metrics()`, with core work in one large file: `ecofragments/utils/calc_metrics.py`.

Important mismatch against v1.2 spec:

- Package/docs still use old naming: iRiverMetrics / EcoFragments.
- Current output is wide CSV (`ecof_metrics.csv`), not v1.2 tidy long schema.
- Current metrics still emit dropped/reworked metrics: `PF`, `PLF`, `AWMPA`, `AWMPL`, `AWMPW`.
- Current code uses Dask partially (`@delayed`, `apply_ufunc`, chunks), but many operations convert to NumPy/Pandas/GeoPandas/igraph. Needs real Dask audit, not assumption.
- No visible config hash, run metadata, monthly composite metadata, valid-observation layer contract, edge flags, or manager-oriented docs yet.
- v1.2 spec is much stronger than implementation. Treat work as refactor + scientific migration, not small patch.

## Publication Priority

JOSS is not a near-term priority. Keep publication readiness as a future option, but do not let JOSS artifacts block v1.2 implementation.

Near-term priority:

- Scientific metric defensibility.
- Dask-first processing and CUDA-ready optional path.
- Outputs useful for hydrology/ecology analysis.
- Manager/practitioner documentation.
- Reproducibility metadata that helps real users, not journal compliance.

Future optional:

- JOSS `paper.md`.
- Six-month public-development evidence packaging.
- Zenodo/JOSS submission checklist.
- Formal JOSS-specific docs review.

## Token-Efficient Model Strategy

Use cheap/wide models for discovery and indexing. Use stronger models only after scope is narrowed and evidence packet is small.

Codex tier convention used here:

- `Codex 5.6 high (terra)`: cheapest Codex pass; use for repo indexing, grep-style triage, and low-risk summaries.
- `Codex 5.6 high (luna)`: balanced Codex pass; use for normal engineering audits where cost still matters.
- `Codex 5.6 high (sol)`: strongest Codex pass; use for spec compliance, Dask/CUDA architecture, adversarial engineering review, and final implementation planning.

| Etapa | Objetivo | Modelo sugerido | Por que esse modelo |
|---|---|---|---|
| 0 | Build evidence packet: repo map, spec summary, entry points | **Gemini 3.5 Flash high** or **Codex 5.4 medium** | Cheap, fast, good for broad scan |
| 1 | Cheap repo triage | **Codex 5.4 medium** or **Kimi 2.7 high** | Finds files, flows, obvious gaps with low token cost |
| 2 | Spec compliance audit | **Codex 5.6 high (sol)** | Best fit for repo-grounded engineering audit and exact file references |
| 3 | Scientific metrics audit | **Claude Opus high** or **Claude Fable high** | Stronger conceptual critique; use small evidence packet only |
| 4 | Dask/CUDA processing audit | **Codex 5.6 high (sol)** then **Gemini Pro 3.1 high** adversarial pass | Code path + performance reasoning, then second-opinion stress test |
| 5 | Manager interpretation audit | **Claude Sonnet high** or **Gemini Pro 3.1 medium** | Good at audience framing, caveats, decision-support language |
| 6 | Documentation audit | **Claude Sonnet medium** | Good docs critique; focus README, quickstart, input format, manager docs |
| 7 | Adversarial synthesis | **Claude Opus high** or **Grok 4.5 high** | Challenge assumptions, novelty, hidden risk |
| 8 | Implementation plan, no edits | **Codex 5.6 high (sol)** | Convert findings into safe, ordered code plan |
| 9 | Plan review / compression | **Kimi 2.7 medium** or **Gemini 3.5 Flash high** | Cheap dedupe, sequencing, token reduction |

Default recommendation: run stages 0-2-4-8 with Codex/Gemini first. Use Opus/Fable only for scientific/adversarial review after evidence is compressed.

## Workflow Gates

Do not implement until these gates pass:

1. Audit report approved.
2. Metric register decisions approved: keep/drop/add/rename.
3. Dask/CUDA architecture approved.
4. Output schema approved.
5. Validation dataset and benchmark plan approved.
6. Implementation sequence approved.

JOSS approval is not a workflow gate for this phase.

## Stage 0 - Evidence Packet

Model: `Gemini 3.5 Flash high` or `Codex 5.4 medium`

Token budget: 20k-60k input, 3k-6k output.

Output file suggested: `docs/audit/evidence_packet.md`

Prompt:

```text
You are preparing a compact evidence packet for a later senior audit.

Goal:
Summarize this repository and the v1.2 spec so a stronger model can audit it without rereading everything.

Read:
- docs/HydroFragments_v1.2_spec.md
- README.md
- pyproject.toml
- docs/architecture.md
- docs/module2.md
- ecofragments/main.py
- ecofragments/utils/calc_metrics.py
- tests/

Reference upstream, if accessible:
- D:\RLH\5.6\repos\WaterMask-TSFill

Use it only to identify likely HydroFragments input contracts. Do not assume compatibility without evidence.

Do not edit source files. Create this markdown report: `docs/audit/evidence_packet.md`.
The next phase must ingest that file before doing its own work.

Output:
1. Repository structure summary
2. Main public API and execution path
3. Current metric outputs
4. v1.2 required metric outputs
5. Expected upstream input from WaterMask-TSFill, if inspectable
6. Explicit mismatches between current code and v1.2 spec
7. Current Dask usage: where lazy, where compute happens, where NumPy/Pandas/GeoPandas/igraph break laziness
8. Current tests and what they prove
9. Missing tests required by v1.2
10. Documentation drift
11. Exact file/function references for all evidence

Constraints:
- Evidence only. Do not propose fixes yet.
- Be concise.
- Do not invent behavior not visible in code/spec.
```

## Stage 1 - Cheap Repo Triage

Model: `Codex 5.4 medium` or `Kimi 2.7 high`

Token budget: 40k-100k input, 5k-8k output.

Prompt:

```text
You are acting as a senior software engineer doing first-pass repo triage.

Goal:
Find high-signal risks before deeper audit. First phase only.

Do not edit source files. Create this markdown report: `docs/audit/repo_triage.md`.
The next phase must ingest `docs/audit/evidence_packet.md` and this file before doing its own work.

Review scope:
1. Package structure and naming drift
2. Main execution flow
3. Data input/output assumptions
4. Current metric formulas
5. Dask usage and scalability
6. CRS/nodata/valid-observation handling
7. Tests and regression coverage
8. Documentation drift from HydroFragments v1.2 spec
9. Likely compatibility with upstream WaterMask-TSFill binary water-mask time series, if inspectable

Special focus:
- Current code still emits PF/PLF/AWMPA/AWMPL/AWMPW
- v1.2 requires source-agnostic binary mask + valid-observation layer
- v1.2 requires fixed denominators, edge flags, config hash, tidy long schema
- Dask-based processing is mandatory
- CUDA-ready path should be possible if available, but not required at runtime

Output:
- Project overview
- Main execution pathways
- Top 20 risks, ordered by severity
- Evidence table: finding | file/function | why it matters | how to verify | likely fix area | priority
- Suggested deeper-audit questions

Constraints:
- No edits.
- Specific and evidence-based.
- If unsure, say what remains uncertain.
```

## Stage 2 - Spec Compliance Audit

Model: `Codex 5.6 high (sol)`

Token budget: 80k-180k input, 8k-14k output.

Prompt:

```text
You are acting as a senior software engineer and implementation-contract auditor.

Goal:
Audit the current repository against docs/HydroFragments_v1.2_spec.md.

First phase only: audit and diagnosis. Do not change source files.
Create this markdown report: `docs/audit/spec_compliance.md`.
The next phase must ingest `docs/audit/evidence_packet.md`, `docs/audit/repo_triage.md`, and this file before doing its own work.

Spec priorities:
1. Source-agnostic input contract: binary/probabilistic water mask + valid-observation layer
2. Monthly compositing rules and dry-down dual-composite check
3. CRS/unit guard: equal-area for area, length-distortion caveat for length metrics
4. Fixed denominators and removal/reformulation of circular/redundant metrics
5. Zone schema and no-drainage fallback
6. Core metrics: occurrence, RA, APSEC, LPSEC, N, LPI, AWRe, dry-down
7. Secondary metrics: AWMSI, MESH, pool width distribution, inter-pool gap, reconnection, refuge stability, TCF, optional DCI
8. Guardrails: edge flags, min_patch_pixels, connectivity_rule, valid obs floors, no PCF, no morphology-proxy Zone 1
9. Output schema: tidy long table + metadata columns
10. Tests required by v1.2 checklist
11. Documentation and optional future publication notes
12. Compatibility with WaterMask-TSFill outputs as likely upstream inputs, if inspectable

Instructions:
1. Build current implementation map from code.
2. Build spec requirement checklist.
3. Mark each requirement as:
   - implemented
   - partially implemented
   - absent
   - contradicted by current code
   - unclear
4. For each gap, include exact file/function affected.
5. Identify dependencies between gaps.
6. Propose implementation order, but do not write code.

Output:
- Executive summary
- Compliance matrix
- Contradictions and blockers
- Implementation dependency graph
- Test gap matrix
- Minimal refactor sequence
- Questions before editing

Constraints:
- Evidence-based only.
- Do not invent missing code.
- Prefer minimal robust migration over rewrite unless spec requires new module boundary.
```

## Stage 3 - Scientific Metrics Audit

Model: `Claude Opus high` first choice; `Claude Fable high` if available and strong on scientific writing/critique.

Token budget: compressed evidence only, 25k-60k input, 6k-10k output.

Prompt:

```text
You are acting as an adversarial scientific reviewer for a hydrology / ecohydrology software method.

Goal:
Audit HydroFragments v1.2 metric foundations before implementation.

Inputs:
- HydroFragments v1.2 spec
- Evidence packet from repo audit
- Current metric list and planned metric list

Do not edit source files. Do not write code.
Create this markdown report: `docs/audit/scientific_metrics_audit.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Review these scientific risks:
1. Are metric definitions defensible for intermittent river surface-water masks?
2. Are dropped metrics correctly dropped or only weakly argued?
3. Are fixed denominators scientifically justified?
4. Are AWRe, AWMSI, LPI, MESH, N, RA, APSEC, LPSEC, dry-down, RC/TCF/DCI positioned correctly against literature?
5. Does dry-down rate risk overclaiming as hydrological recession?
6. Does pool width distribution risk being misread as depth/storage?
7. Does NNI remain too prominent despite quasi-1D river geometry?
8. Are DCI/PC/IIC distinctions clear enough?
9. Are validation claims empirical or still asserted?
10. What would a hostile reviewer say about novelty versus Tayer et al. 2025?

Output:
- Major scientific risks
- Metric-by-metric audit table
- Claims that need validation before publication
- Claims safe for docs but unsafe for paper
- Required citations or positioning fixes
- Suggested validation analyses
- Red-team reviewer objections and best responses

Constraints:
- Be adversarial but fair.
- Separate "must fix before implementation" from "must validate before paper".
- Do not ask for more data unless essential.
```

## Stage 4 - Dask/CUDA Processing Audit

Model: `Codex 5.6 high (sol)`; adversarial second pass with `Gemini Pro 3.1 high`.

Token budget: 50k-120k input, 6k-12k output.

Prompt:

```text
You are acting as a senior geospatial performance engineer.

Goal:
Audit the repo for Dask-first scalability and CUDA readiness.

First phase only. Do not edit source files.
Create this markdown report: `docs/audit/dask_cuda_audit.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Requirements:
- Processing must be Dask-based.
- CUDA-ready if available, but CPU must remain default and correct.
- Avoid requiring GPU-only dependencies.
- Avoid pretending GPU acceleration exists where libraries force CPU.

Inspect:
1. xarray/dask chunking strategy
2. Places where `.compute()`, NumPy conversion, Pandas concat, GeoPandas, scikit-image, scipy.ndimage, rasterio, shapely, igraph force eager/CPU work
3. `dask.delayed` task granularity and scheduler overhead
4. Memory risk for large rasters and long time series
5. Connected components, skeletonization, EDT, regionprops, polygon export
6. Valid-observation and monthly compositing pipeline
7. Zarr/NetCDF/GeoTIFF stack I/O
8. Candidate CuPy/CUDA paths: which ops can use CuPy, which cannot, and where feature detection should live
9. Benchmark design for CPU and GPU-available environments

Output:
- Current processing graph summary
- Eager/CPU choke points table
- Dask risks table
- CUDA-ready design proposal
- CPU fallback design
- Benchmark plan: datasets, metrics, expected outputs
- Implementation sequence

Constraints:
- Be precise about what is actually GPU-capable.
- Do not introduce GPU dependency as hard requirement.
- Prefer clear interfaces and feature flags over scattered conditionals.
```

## Stage 5 - Manager Interpretation Audit

Model: `Claude Sonnet high` or `Gemini Pro 3.1 medium`.

Token budget: 20k-50k input, 4k-8k output.

Prompt:

```text
You are acting as a water-resource manager interpretation reviewer.

Goal:
Audit HydroFragments metric outputs for practical decision-support meaning.

Do not edit source files.
Create this markdown report: `docs/audit/manager_interpretation_audit.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Focus:
1. Which metrics are meaningful to water managers?
2. Which metrics need plain-language translation?
3. Which metrics are dangerous if interpreted naively?
4. What combinations matter more than single metrics?
5. How should uncertainty, low-valid-observation flags, composite sensitivity, CRS/length caveats, and width-not-depth be communicated?
6. What should docs/for-managers.md contain?

Required output:
- Manager-facing metric glossary, one sentence each
- Decision-support table: metric or metric pair | concerning pattern | management question | caveat
- 3-5 worked narrative templates using placeholders, not invented numbers
- Warnings that must appear in manager docs
- Claims to avoid

Constraints:
- No formulas unless unavoidable.
- No fake thresholds. Use placeholders or say "derive from validation catchment".
- Keep science caveats but translate them.
```

## Stage 6 - Documentation Audit

Model: `Claude Sonnet medium`.

Token budget: 30k-70k input, 5k-9k output.

Prompt:

```text
You are acting as a documentation reviewer for a scientific Python package.

Goal:
Audit docs and packaging against HydroFragments v1.2 spec and near-term user needs.

Do not edit source files.
Create this markdown report: `docs/audit/docs_audit.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Review:
1. README
2. docs/
3. pyproject.toml
4. tests/
5. package naming and citation story
6. input data format docs
7. practitioner quickstart
8. manager guide
9. API docs
10. basic packaging/readme/install clarity
11. reproducibility docs
12. future publication notes, but JOSS is not current priority

Output:
- Docs drift summary
- Missing docs matrix
- README rewrite outline
- docs/input_format.md outline
- docs/for-managers.md outline
- API docs outline
- Future publication notes, clearly marked non-blocking

Constraints:
- Do not treat JOSS as an implementation blocker.
- Do not overclaim publication readiness.
- Keep implementation docs aligned with v1.2 spec.
```

## Stage 7 - Adversarial Synthesis

Model: `Claude Opus high` or `Grok 4.5 high`.

Token budget: compressed reports only, 20k-60k input, 5k-10k output.

Prompt:

```text
You are acting as an adversarial principal investigator, reviewer 2, and senior maintainer combined.

Goal:
Stress-test the audit reports and proposed implementation plan before any code changes.

Inputs:
- Repo triage
- Spec compliance audit
- Scientific metrics audit
- Dask/CUDA audit
- Manager interpretation audit
- Documentation audit

Do not edit source files.
Create this markdown report: `docs/audit/adversarial_synthesis.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Attack the plan:
1. What is over-scoped?
2. What is under-specified?
3. What will fail scientifically?
4. What will fail computationally?
5. What will confuse water managers?
6. What will break existing users?
7. What would future journal/software reviewers likely reject?
8. What assumptions need evidence?
9. Which planned features should be deferred?
10. What is the smallest credible v1.2?

Output:
- Top 10 adversarial objections
- Required plan changes
- Deferrals recommended
- Non-negotiable implementation gates
- Minimal credible v1.2 scope
- Final go/no-go recommendation

Constraints:
- Be severe but constructive.
- Prefer smaller shippable scope over heroic rewrite.
- Do not invent requirements beyond spec unless risk is explicit.
```

## Stage 8 - Implementation Plan, No Code

Model: `Codex 5.6 high (sol)`.

Token budget: all final audit reports, 50k-120k input, 8k-14k output.

Prompt:

```text
You are acting as a senior implementation planner.

Goal:
Create a no-code implementation plan for migrating current EcoFragments/iRiverMetrics repo to HydroFragments v1.2.

Do not edit source files.
Create this markdown report: `docs/audit/implementation_plan.md`.
The next phase must ingest prior audit markdown files before doing its own work.

Inputs:
- HydroFragments v1.2 spec
- Audit reports
- Adversarial synthesis

Plan must optimize for:
- Scientific defensibility
- Dask-first processing
- CUDA-ready optional acceleration
- Water-manager interpretation
- Documentation and future publication readiness
- Minimal safe increments
- Backward compatibility where reasonable

Scope constraint:
- `hydroseason` (sibling repo; formerly planned as `hydroyear`) is out of scope for this repo — standalone package owns HY detection, season mapping, and related HY metrics. HydroFragments must import/adapt that API only; do not design or build HY/season algorithms here. Milestone 9 may add a thin adapter, not a detector.
- `hydrozones` stays in scope for this repo (not yet developed; keep as planned internal `zones/` module per spec §3).

Output:
1. Target architecture
2. Module boundaries
3. Public API proposal
4. Config schema proposal
5. Output schema proposal
6. Implementation milestones
7. Each milestone:
   - objective
   - files likely touched
   - tests to write first
   - acceptance criteria
   - rollback risk
8. Validation plan
9. Benchmark plan
10. Documentation plan
11. Release plan
12. Explicit deferrals

Constraints:
- No code.
- Order work so tests and schema stabilize before metric expansion.
- Separate core v1.2 from optional DCI/CUDA/future publication polish.
- Keep existing regression tests useful, but do not preserve dropped metrics as v1.2 outputs.
```

## Stage 9 - Plan Compression

Model: `Kimi 2.7 medium` or `Gemini 3.5 Flash high`.

Token budget: 20k-50k input, 2k-5k output.

Prompt:

```text
Compress this implementation plan into an execution checklist for coding agents.

Do not edit source files.
Create this markdown report: `docs/audit/execution_checklist.md`.
Future coding phases must ingest `docs/audit/implementation_plan.md` and this file before editing source code.

Keep:
- Milestone order
- Acceptance criteria
- Test-first requirements
- Risk gates
- Deferrals
- Model-specific follow-up prompts if needed

Remove:
- Repeated rationale
- Long prose
- Anything not actionable

Output:
- One-page executive plan
- Detailed checklist
- Stop/go gates
```

## Recommended Minimal v1.2 Implementation Order

Use after audits pass.

| Order | Workstream | Why first/now |
|---|---|---|
| 1 | Rebrand namespace/docs references carefully | Low algorithm risk, high clarity, preserves history |
| 2 | Config schema + metadata/hash + output schema | Stabilizes contracts before metrics |
| 3 | Input contract: water mask + valid-observation layer + grid alignment errors | Foundation for all scientific metrics |
| 4 | Dask-first I/O and chunking architecture | Avoids baking eager design into new metrics |
| 5 | CRS/unit guards + min patch/connectivity config | Prevents invalid metric values |
| 6 | Core metric migration: occurrence, RA, APSEC, LPSEC, N, LPI, AWRe | Smallest publishable scientific core |
| 7 | Dry-down + monthly composite + composite sensitivity | Headline metric, high scientific risk |
| 8 | Edge flags and hydrological state flags | Needed for interpretation/modeling |
| 9 | Secondary metrics: AWMSI, MESH, width distribution, gap | Add after core stable |
| 10 | Connectivity: RC/TCF; DCI decision gate | Most conceptual risk; do after graph design review |
| 11 | Manager guide + practitioner quickstart | Must use real output ranges, not invented examples |
| 12 | CI + docs quality | Practical maintainability; JOSS-specific artifacts deferred |

## Dask/CUDA Design Rules For Later Implementation

- CPU correctness is baseline.
- CUDA is optional feature detection, not required dependency.
- No metric should silently change numerical semantics between CPU and GPU.
- `cupy`/GPU path only where array operations support it cleanly.
- GeoPandas, Shapely, rasterio polygonization, scikit-image skeletonization, and igraph likely remain CPU choke points unless replaced or isolated.
- Benchmark both scheduler overhead and peak memory, not just wall time.
- Export-heavy operations should be separate from metric computation.

## Acceptance Criteria For Audit Phase

Audit phase done when these files exist:

- `docs/audit/evidence_packet.md`
- `docs/audit/repo_triage.md`
- `docs/audit/spec_compliance.md`
- `docs/audit/scientific_metrics_audit.md`
- `docs/audit/dask_cuda_audit.md`
- `docs/audit/manager_interpretation_audit.md`
- `docs/audit/docs_audit.md`
- `docs/audit/adversarial_synthesis.md`
- `docs/audit/implementation_plan.md`
- `docs/audit/execution_checklist.md`

Implementation should not start before `docs/audit/implementation_plan.md` is reviewed and approved.

Optional future publication artifacts:

- `paper/paper.md`
- Zenodo DOI checklist
- JOSS eligibility/public-history checklist
