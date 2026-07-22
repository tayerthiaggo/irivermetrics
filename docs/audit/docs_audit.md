# HydroFragments v1.2 — Documentation & Packaging Audit

**Audit date:** 2026-07-10
**Reviewer stance:** Documentation/packaging reviewer. Not a scientific pass ([`scientific_metrics_audit.md`](scientific_metrics_audit.md)), not a code/spec compliance pass ([`spec_compliance.md`](spec_compliance.md)), not a manager-interpretation pass ([`manager_interpretation_audit.md`](manager_interpretation_audit.md)), not a Dask/CUDA pass ([`dask_cuda_audit.md`](dask_cuda_audit.md), [`dask_cuda_audit_adversarial.md`](dask_cuda_audit_adversarial.md)). This report asks: *does the documentation and packaging surface honestly represent what the repository currently is, and what does it need to say to serve a v1.2 practitioner and a v1.2 manager?*
**Constraint:** no source files edited. This markdown report is the only deliverable.
**Contract audited:** [`docs/HydroFragments_v1.2_spec.md`](../HydroFragments_v1.2_spec.md), current `README.md`, `docs/`, `pyproject.toml`, `tests/`.

> **Mandatory next-phase intake gate**
>
> This audit is downstream of five prior reports. The next phase (implementation or further docs work) must read all of the following, in this order, before doing its own work:
>
> 1. [`docs/audit/evidence_packet.md`](evidence_packet.md)
> 2. [`docs/audit/repo_triage.md`](repo_triage.md)
> 3. [`docs/audit/spec_compliance.md`](spec_compliance.md)
> 4. [`docs/audit/scientific_metrics_audit.md`](scientific_metrics_audit.md)
> 5. [`docs/audit/manager_interpretation_audit.md`](manager_interpretation_audit.md)
> 6. [`docs/audit/dask_cuda_audit.md`](dask_cuda_audit.md) and [`docs/audit/dask_cuda_audit_adversarial.md`](dask_cuda_audit_adversarial.md)
> 7. this report — [`docs/audit/docs_audit.md`](docs_audit.md)
>
> This report assumes all prior findings as given and does not re-argue them. It translates them into a documentation/packaging work plan. Where this report recommends specific doc wording or structure, it defers to `scientific_metrics_audit.md` §16.1 F-5 ("governing mood") and `manager_interpretation_audit.md` on tone — no doc content proposed here should overclaim beyond what those two reports license.

---

## 0. Framing: what "docs audit" means here, given the code is pre-migration

The repository's code is, per `spec_compliance.md`, "not v1.2 compliant" and "a working legacy EcoFragments/iRiverMetrics-style snapshot pipeline." That fact bounds what a documentation audit can responsibly ask for:

- It is not useful to write `docs/input_format.md` describing the v1.2 `(water, valid_obs)` contract as if it exists today — that would be documenting vaporware.
- It is also wrong to leave the current docs describing `iRiverMetrics`/`irivermetrics` imports that do not exist in this repository at all — that is documenting a package that cannot be installed as documented.
- The correct target state for **this phase** is: docs that are honest about current (legacy) behaviour, clearly labelled as pre-v1.2, plus **outlines** (not finished prose) for the v1.2-target documents, so the implementation phase has a drafting scaffold that already reflects the locked spec and the prior audits' caveats.

This is why the outlines below are structural (headings, required content, source-of-truth pointers) rather than filled-in text with invented numbers — consistent with how `manager_interpretation_audit.md` §6–§8 handled the same problem for `docs/for-managers.md`.

---

## 1. README audit

**File:** [`README.md`](../../README.md)

| Finding | Evidence | Severity |
|---|---|---|
| Wrong package identity throughout | Title is "iRiverMetrics" (`README.md:2`); clone URL is `github.com/tayerthiaggo/irivermetrics.git` (`README.md:29`); conda env named `irivermetrics` (`README.md:34`) | Critical |
| Install instructions install a package that cannot produce the documented import | `pip install -e path/to/clone/irivermetrics` (`README.md:37`) installs whatever is at that path; the actual package in *this* repo is `ecofragments` (`pyproject.toml:2`), and neither `irivermetrics` nor `ecofragments` exposes `irivermetrics.irm_main` | Critical |
| Usage example imports a module that does not exist in this repository | `from irivermetrics.irm_main import waterdetect_batch, calculate_metrics` (`README.md:49`) — no `irivermetrics` package exists here at all; `waterdetect_batch` does not exist in `ecofragments` either (only `calculate_metrics` is implemented per `repo_triage.md` §2) | Critical — a new user following the README verbatim gets `ModuleNotFoundError` on line 1 of the example |
| Describes a two-module toolkit; only one module is implemented | README §"Modules" describes `waterdetect_batch` and `calculate_metrics` as both present; `repo_triage.md` §1 states "the only implemented execution path is `calculate_metrics()`" | High |
| Citation story is present but incomplete and not yet updated for the v1.2 rename/lineage | Cites the two 2023 J. Hydrology papers (`README.md:71-77`); does not cite Tayer et al. 2025/2026 (the Gilbert resilience paper) or Tayer 2023c (clustering paper), both of which `scientific_metrics_audit.md` R4 identifies as directly relevant to novelty positioning | Medium |
| No mention of `HydroFragments` anywhere | Entire file uses legacy naming | Critical (same root cause as above, listed separately because it affects search/discoverability, not just import correctness) |
| No statement of current implementation status / v1.2 migration state | A new reader cannot tell this is a pre-migration snapshot | High — this is the single most load-bearing missing sentence in the whole docs tree |

**Net assessment:** the README does not describe an installable, runnable package as it exists in this repository. It describes a *different, non-existent* package (`irivermetrics`) with a two-module surface that is only half-built even under its own legacy name (`ecofragments`). This is not a stale-docs problem in the ordinary sense — it is a **wrong-repository-pointer problem**: the README's `git clone` URL and import path point at a GitHub project that is not this codebase.

---

## 2. `docs/` audit

| File | Status | Key issues |
|---|---|---|
| [`docs/index.md`](../index.md) | Legacy, wrong imports | `from irivermetrics import waterdetect_batch, calculate_metrics` (`index.md:27`) — same nonexistent-module problem as README. Cites 4 papers including the 2026 resilience paper (good — more complete than README) but frames it as "Scientific foundation" for a still-legacy toolkit. |
| [`docs/project-overview.md`](../project-overview.md) | Legacy, wrong status table | "Current status" table (`project-overview.md:56-66`) marks `waterdetect_batch` and `calculate_metrics` both "✅ Operational" and QA thresholds "✅ Implemented" — but `spec_compliance.md` A5/A6 shows the QA thresholds are hard-coded legacy values, not the v1.2 `min_valid_obs`/`min_valid_fraction_month` contract, and `evidence_packet.md` shows only one of the two modules exists in code. This table actively misrepresents both v1.2 compliance and even legacy completeness (no `waterdetect_batch` in this repo). |
| [`docs/module1.md`](../module1.md) | Documents a module not present in this repository | Full usage guide for `waterdetect_batch`, imported from `irivermetrics.irm_main` (`module1.md:70`). Not verified to exist anywhere in `ecofragments/`. Should either be marked "planned / not yet ported" or removed until ported. |
| [`docs/module2.md`](../module2.md) | Documents the legacy metric register, including metrics v1.2 explicitly drops | Metric table (`module2.md:110-130`) lists `AWMPA`, `AWMPL`, `AWMPW`, `PF`, `PLF` — all dropped/reworked per spec §4. Does not mention any v1.2 metric (`LPI`, `MESH`, `dry_down_rate`, `TCF`, `DCI`, `RC`, inter-pool gap, pool width distribution, recurrence, hydroperiod). Import example uses nonexistent `irivermetrics.irm_main` (`module2.md:81`). This is the most consequential wrong-doc in the tree because it is the one a practitioner would actually run code from. |
| [`docs/architecture.md`](../architecture.md) | Contradicts locked v1.2 scope | Frames the tool as domain-agnostic — "the generalised architecture supports any domain... Aquatic... Terrestrial... Urban" (`architecture.md:5-12`). Spec §0 is explicit and deliberate that v1.2 is river/surface-water-focused only, not a generic patch-metrics tool, and that genericising would "add nothing novel." This is a direct, named contradiction, not a drift. |
| [`docs/paper-summary.md`](../paper-summary.md) | Internal research artifact, not really a "doc" for external users, but usefully accurate and complete | Correctly identifies pool-unit tracking and dynamic HY detection as "new scope — not v1", matching spec §"Explicitly out of scope for v1." Good source-of-truth for the citation story; should be the seed for README/index citation updates, not deleted. |
| [`docs/HydroFragments_v1.2_spec.md`](../HydroFragments_v1.2_spec.md) | Authoritative, current | Locked spec. All other docs should be checked against this, not the reverse. |
| [`docs/audit_implementation_plan.md`](../../docs/audit_implementation_plan.md) *(repo root, not under `docs/audit/`)* | Planning artifact, explicitly non-authoritative for implementation | States "No implementation instructions in this file should be treated as approval to edit code." Confirms JOSS is explicitly deprioritized ("JOSS is not a near-term priority... do not let JOSS artifacts block v1.2 implementation") — this directly supports the non-blocking framing required for §12 of this report. |

**Pattern across `docs/`:** every user-facing doc (README, index, project-overview, module1, module2) inherits the same two defects — (a) wrong/nonexistent package identity and import path, (b) legacy metric register presented as current/complete. `architecture.md` additionally contradicts the locked scope decision, which is a content error, not just a staleness error.

---

## 3. `pyproject.toml` audit

**File:** [`pyproject.toml`](../../pyproject.toml)

| Field | Current value | Issue |
|---|---|---|
| `name` | `"ecofragments"` | Neither `iRiverMetrics` (README) nor `HydroFragments` (spec, target). A `pip install` of this repo installs `ecofragments`, matching neither doc set. |
| `description` | `"EcoFragments: spatial patch dynamics metrics from binary classification time series"` | Repeats the domain-agnostic framing `architecture.md` uses, which spec §0 rejects. Also doesn't mention rivers/surface water/intermittent rivers at all — a user scanning PyPI metadata would not know this is a river tool. |
| `dependencies` | Lists `dask-image`, `dask-regionprops` | Per `spec_compliance.md` runtime note and `dask_cuda_audit.md`, `dask_image` import currently breaks `pytest` collection in the audit environment, and both packages are imported but "do not participate in executed kernels" (`dask_cuda_audit.md` "Unused imports give false confidence"). This is a packaging-hygiene issue: declared-but-unused-or-broken dependencies inflate install surface without benefit. |
| No `[project.urls]`, no `LICENSE` reference, no `classifiers` | — | `spec_compliance.md` F11 already flags absence of LICENSE/CONTRIBUTING/CODE_OF_CONDUCT at the repo level; `pyproject.toml` itself has no homepage/repository URL, so `pip show` / PyPI metadata would be a dead end for a user trying to find docs or file an issue. |
| No `[project.optional-dependencies]` for docs or lint, only `test` | `pyproject.toml:28-32` | Consistent with `spec_compliance.md` F10 finding that CI has no lint/docs-build step — the packaging config doesn't even define the extras that would make such a step meaningful later. |

**Net assessment:** `pyproject.toml` is internally consistent with the `ecofragments` package directory (i.e., it does not lie about what's importable, unlike the docs), but it is the third distinct identity in play (`ecofragments` vs. docs' `iRiverMetrics` vs. spec's `HydroFragments`), and its description actively encodes the scope claim the spec has locked out.

---

## 4. `tests/` audit (packaging/docs-relevant angle only — see `spec_compliance.md` §F for full test-contract gap analysis)

This audit does not re-run the full test-gap analysis (already done exhaustively in `spec_compliance.md` Compliance Matrix §F and Test Gap Matrix). From a **documentation-and-packaging** lens specifically:

| Finding | Evidence | Why it's a docs/packaging concern, not just a code concern |
|---|---|---|
| Regression fixture path is broken | `tests/conftest.py:34` reads `TEST_DIR / "results_ecofragments" / "metrics" / "ecof_metrics.csv"`; actual directory is `tests/results_iRiverMetrics/metrics/irm_metrics.csv` (confirmed present in repo file listing) | A new contributor following `docs/architecture.md`'s own test-directory listing (`architecture.md:30-35`, which itself still says `wmask_ts.nc` / `rcor_extent.shp` — accurate — but doesn't mention `results_iRiverMetrics/`) would hit an undocumented `FileNotFoundError` on the one regression test that's supposed to prove the pipeline works end to end. |
| No `docs/testing.md` or CONTRIBUTING guidance on how to run tests, what `slow` marker means beyond one line in `pyproject.toml`, or what fixtures exist | `pyproject.toml:48-49` defines the `slow` marker but no doc explains the two-tier fast/slow test strategy to a contributor | Medium — affects onboarding, not correctness. |
| Test directory itself carries the same three-identity confusion | `tests/results_iRiverMetrics/` (legacy name) vs. `conftest.py`'s reference to `results_ecofragments` (current package name, but wrong on disk) vs. neither matching `hydrofragments` (target name) | Confirms the naming drift isn't just docs — it's baked into fixture paths, so a docs rewrite alone cannot fix this; it is correctly flagged as a code-side fix in `repo_triage.md` item 17, and this report defers to that. |

**No test-content documentation exists** describing what `tests/wmask_ts.nc` and `tests/rcor_extent.shp` actually contain (dimensions, CRS, date range, section count) outside of one line in `docs/architecture.md` ("63-timestep test water mask", "7-section test corridor") and passing mentions in `evidence_packet.md`/`spec_compliance.md` (which note this content could not be independently verified in the audit environment due to missing NetCDF backend deps). This matters for the practitioner quickstart (§7 below): any worked example needs to either use this fixture honestly-described, or a synthetic dataset built for the quickstart specifically.

---

## 5. Package naming and citation story

**Naming.** Three names are live simultaneously in this repository right now:

1. `iRiverMetrics` / `irivermetrics` — README, docs/index.md, docs/module1.md, docs/module2.md, example notebook imports, test directory name (`results_iRiverMetrics`)
2. `EcoFragments` / `ecofragments` — `pyproject.toml` package name, actual `ecofragments/` directory, `docs/architecture.md`
3. `HydroFragments` — the locked target name per spec title and repo name itself (`D:\RLH\5.6\repos\HydroFragments`)

None of the three is fully self-consistent even on its own terms (e.g., `iRiverMetrics`-branded docs describe modules and metrics that don't match the `ecofragments` code that actually runs). This is not a cosmetic branding issue — a new user cannot form a correct mental model of "what do I `pip install`, what do I `import`" from any single doc in the current tree.

**Recommendation for next phase (not performed here — no source edited):** rebrand is correctly sequenced *last* in `spec_compliance.md`'s "Minimal refactor sequence" (step 13, after API stabilises) and in `repo_triage.md`'s risk list (item 16, Medium priority, not Critical) — this audit agrees with that sequencing. Documentation should not be renamed to `hydrofragments` prematurely if the underlying `import ecofragments` still works differently; that would just add a fourth inconsistent identity. Instead, the immediate documentation fix (before any rename) is to **stop the bleeding**: clearly label current docs as describing the legacy `ecofragments` (née iRiverMetrics) implementation, pending v1.2 migration — see §11 README rewrite outline.

**Citation story.** The predecessor lineage is genuinely strong and should be foregrounded, not buried:

- Tayer et al. 2023a (GIScience & RS) — Water Detect calibration
- Tayer et al. 2023b (J. Hydrology 617) — the 11/16-metric ecohydrological suite (the direct ancestor of this codebase)
- Tayer et al. 2023c (J. Hydrology 626) — hydrological clustering / 4 zone types, groundwater-validated
- Tayer et al. 2025/2026 (J. Hydrology 666) — Mapping Resilience framework, Gilbert River

`docs/index.md` cites all four; `README.md` cites only the first two. `scientific_metrics_audit.md` R4 flags that the *novelty positioning* against papers 3 and 4 specifically is currently weak in the spec's own §16 — that is a spec/paper concern, not a docs concern, but the **citation list itself** (as opposed to the novelty argument) should be complete and consistent everywhere a citation appears. Currently it is not even consistent between README and docs/index.md.

Also relevant per intake-gate report `manager_interpretation_audit.md`: the spec supersedes "internal `iRiverMetrics` metric suite (Tayer et al. 2023a, 2023b, 2025/2026)" (spec header line 10) — so the spec's own front matter already treats this as settled lineage. Docs should mirror the spec's citation list exactly, not a subset.

---

## 6. Docs drift summary

One consolidated table, synthesising §1–§5 above plus prior-audit findings F6–F9 (`spec_compliance.md`) and the docs findings already logged in `evidence_packet.md` §10 and `repo_triage.md` items 19–20 — restated here specifically as **doc-vs-reality drift**, not repeated as new findings:

| Drift | Where | Ground truth it contradicts |
|---|---|---|
| Package name/import path point at a nonexistent external repo | README, docs/index.md, docs/module1.md, docs/module2.md, examples/irm_example.ipynb | `pyproject.toml` (`name = "ecofragments"`), actual `ecofragments/` package directory |
| Two-module toolkit claimed; one module implemented | README §"Modules", docs/index.md, docs/project-overview.md status table | `ecofragments/__init__.py` / `ecofragments/main.py` (`calculate_metrics` only, per `repo_triage.md` §1) |
| Metric register documents 5 metrics v1.2 explicitly drops, omits all v1.2-new metrics | docs/module2.md metric table, docs/paper-summary.md (accurately, as historical record — not itself wrong, but not flagged as historical) | spec §4 metric register |
| "Operational ✅" status claimed for QA thresholds that don't meet v1.2 semantics | docs/project-overview.md status table | spec §1.1.5 (`min_valid_obs`/`min_valid_fraction_month`), `spec_compliance.md` A5/A6 |
| Domain-agnostic architecture framing | docs/architecture.md, pyproject.toml description | spec §0 (river-focused, deliberate, non-negotiable) |
| Test regression fixture path documented nowhere and broken on disk | tests/conftest.py vs. tests/results_iRiverMetrics/ | `repo_triage.md` item 17, `evidence_packet.md` §10.4 |
| Citation list incomplete/inconsistent across docs | README (2 papers) vs. docs/index.md (4 papers) | spec header lineage, `docs/paper-summary.md` |
| No v1.2 input-format doc exists at all | entire `docs/` tree | spec §1.1 (mask+valid-obs contract), `spec_compliance.md` F8, F12 |
| No manager-facing doc exists at all | entire `docs/` tree | spec §12.1, fully specified as an outline in `manager_interpretation_audit.md` §6 |
| No API reference doc beyond one function's docstring-style usage guide | docs/module2.md (describes only `calculate_metrics`, legacy signature) | spec's full v1.2 metric/output/config surface |
| No packaging clarity: LICENSE, CONTRIBUTING, repo URL, classifiers absent | pyproject.toml, repo root | `spec_compliance.md` F11 |
| Reproducibility discipline (config hash, run metadata) described in spec as a genuine strength "foreground it in docs" but zero docs currently mention it | spec §1.1.7 | No corresponding doc section anywhere |
| Future-publication material (JOSS, six-month history, paper) has no docs home and is not marked non-blocking anywhere doc-facing | — | `docs/audit_implementation_plan.md` root file states JOSS is explicitly deprioritized, but this framing exists only in a planning artifact, not in any doc a contributor would naturally read |

---

## 7. Missing docs matrix

What v1.2 needs vs. what exists today. "Exists (legacy)" means present but describing pre-v1.2 behaviour; "Absent" means no file/section addresses this at all.

| Required v1.2 documentation surface | Status | Spec/audit hook |
|---|---|---|
| README describing the actual installable package, correct import path, current migration status | **Wrong** (describes a different, nonexistent package) | §1 above |
| `docs/input_format.md` — water mask + valid-observation contract, WaterMask-TSFill Zarr schema, sentinel semantics, monthly compositing, CRS/units | **Absent** | spec §1.1(1,2,6), §2; `spec_compliance.md` A1–A16, B1–B5 |
| `docs/for-managers.md` — plain-language glossary, decision-support table, worked narratives, non-inclusions | **Absent** (but fully outlined already in `manager_interpretation_audit.md` §6–§9 — this report does not re-outline it, see §9 below) | spec §12.1 |
| Practitioner quickstart (install → run → read output) | **Wrong** (legacy, broken import path; no v1.2 equivalent) | spec §14 (implied — "repository refactor" contract needs a runnable entry point doc) |
| API reference covering the full v1.2 metric register, config schema, output schema | **Absent / severely incomplete** — only legacy `calculate_metrics()` documented, and incorrectly (wrong metric list) | spec §4, §7, §11 (config), §6 (per-metric detail) |
| Metric register doc distinguishing Core / Secondary / Exploratory / Dropped, with citations | **Wrong** (legacy register only, no tier system, dropped metrics presented as current) | spec §4, §5 |
| Zonation / no-drainage fallback doc | **Absent** | spec §3 |
| Connectivity module positioning doc (RC/TCF/DCI vs. DCI/PC/IIC) | **Absent** — this is flagged by the scientific audit as a *locked documentation requirement*, not optional (§1.1.11) | spec §1.1.11, §6.11, §6.11a; `scientific_metrics_audit.md` §8, §13 |
| Validation status doc (asserted-vs-demonstrated inventory, spec §6.18) surfaced outside the spec itself | **Absent as a standalone doc** — currently lives only inside the spec | spec §6.18; `scientific_metrics_audit.md` §9, §11 (this is the single highest-value "don't overclaim" artifact and it isn't discoverable from README/docs/index) |
| Reproducibility / config-hash doc | **Absent** | spec §1.1.7 ("Foreground it in docs") |
| CRS / length-distortion caveat doc | **Absent** as a standalone note; buried only in spec | spec §1.1.1, §2 |
| Architecture doc matching locked river-only scope | **Wrong** (actively contradicts scope) | spec §0 |
| CONTRIBUTING / test-running guide | **Absent** | `spec_compliance.md` F11 |
| LICENSE reference in README/pyproject | **Absent** | `spec_compliance.md` F11 |
| Future publication notes (JOSS, six-month history, paper scoping) — explicitly non-blocking | **Absent as docs**; exists only as a planning-artifact aside in `docs/audit_implementation_plan.md` | spec §10, §13, §16; this report's own constraint: "do not treat JOSS as a blocker" |

**Reading this matrix:** every "Absent" row is new-document work; every "Wrong" row is correction-of-existing-document work. The "Wrong" rows are higher priority than the "Absent" rows in the near term, because a wrong doc actively misleads a user who trusts it (broken imports, invalid metrics, contradicted scope), whereas an absent doc merely leaves a gap. This ordering is reflected in §12.

---

## 8. README rewrite outline

Not a rewritten README — a structural outline for the next phase to draft against, once the underlying package identity/import path question is settled by implementation (this report cannot itself decide whether the package will be renamed to `hydrofragments` before or after this rewrite; see §5). Two versions are given: what changes **now** (honest legacy state) vs. what the **v1.2-target** shape looks like, so the outline survives the eventual rename.

**Section order:**

1. **Title + one-line description** — must name the actual importable package (currently `ecofragments`; target `hydrofragments`) and state river/surface-water scope explicitly (not "spatial patch dynamics" — see §3 pyproject finding).
2. **Status banner** — a single, prominent, unmissable line: this repository is mid-migration from a legacy `EcoFragments`/`iRiverMetrics`-lineage implementation to the locked HydroFragments v1.2 specification; link to `docs/HydroFragments_v1.2_spec.md` and to this audit directory. This is the single highest-value addition — it converts every other stale claim in the doc tree from "misleading" to "known-legacy, clearly labelled," without requiring the rewrite to solve the underlying migration first.
3. **What it does** — one paragraph, scoped to what is *actually implemented right now* (currently: `calculate_metrics` computing legacy metrics from a binary water mask + polygon sections). Explicitly note what it does not yet do (v1.2 zonation, HY dynamics, connectivity module, tidy output) rather than silently omitting.
4. **Install** — corrected clone URL (this repository, not `tayerthiaggo/irivermetrics`), corrected package name, corrected conda/pip instructions matching `pyproject.toml`.
5. **Quickstart usage example** — corrected import (`from ecofragments import calculate_metrics` or whatever the true public API is per `ecofragments/__init__.py`), using either the bundled test fixture or a minimal synthetic example; must not reference `waterdetect_batch` unless/until that module is confirmed present.
6. **Current metric output** — link to a corrected `docs/module2.md` (or its replacement), explicitly labelled "legacy metric set — see the v1.2 metric register in the spec for what is changing."
7. **Where this project is going (v1.2)** — short paragraph + link to spec, `docs/input_format.md` (once drafted), `docs/for-managers.md` (once drafted). This is where forward-looking material belongs, kept clearly separate from "what works today."
8. **Citation** — full four-paper list matching `docs/index.md` and the spec's own front-matter lineage (§5 above), not the current two-paper subset.
9. **Future publication note** — one short paragraph, explicitly marked non-blocking (see §12 below for exact framing) — do not let this section imply JOSS submission is imminent or gating.
10. **Contributing / License** — pointers, even if the underlying files still need to be created (`spec_compliance.md` F11); a README section that says "License: TBD, see issue #N" is more honest than silence.

**Explicit non-goals for this rewrite:** do not invent example output values; do not claim v1.2 compliance anywhere; do not remove the legacy citation/attribution even though the metric set is changing — the predecessor papers remain the scientific foundation being reformed, not replaced (per `scientific_metrics_audit.md` §10's recommended framing).

---

## 9. `docs/for-managers.md` outline

This report does not re-derive this outline — `manager_interpretation_audit.md` §6 already produced a complete, spec-aligned structural outline (8 numbered sections: framing paragraph, plain-language glossary, decision-support table, 3–5 worked narrative templates, the five caveats, "what this tool cannot tell you," escalation path, explicit exclusions), backed by a full decision-support table skeleton (§7 there) and five worked-narrative templates with placeholder-only values (§8 there). Duplicating it here would risk drift between two "authoritative" outlines.

**This report's contribution is scope-gating, not re-outlining:**

- `docs/for-managers.md` **must not be drafted with real numbers until a validation catchment run exists** (Gilbert, per the scientific audit's own recommendation) — `manager_interpretation_audit.md` is explicit that all placeholders (`[VALUE]`, `[THRESHOLD]`, `[REACH]`, `[PERIOD]`) must stay placeholders until then. Any implementation phase that fills these in from imagined numbers would violate this audit's own no-overclaim constraint.
- The document must not be written before the metric-naming and framing fixes land (`scientific_metrics_audit.md` §16.1 F-4, F-5 — "contraction rate" not "recession", hypothesis-mood language) — otherwise the manager doc would need a second rewrite immediately after the code/spec-language fix.
- Sequencing recommendation for the next phase: draft `docs/for-managers.md`'s structure and glossary sentences (which don't need real numbers) now or early; leave the decision-support table and worked narratives as literal placeholder text until a validation run exists.

**Cross-reference, not duplication:** implementation phase should treat [`manager_interpretation_audit.md`](manager_interpretation_audit.md) §6–§9 as the outline of record for this file.

---

## 10. `docs/input_format.md` outline

No such file exists today (§7). This is a genuine gap, not a drift-correction. Outline, derived directly from spec §1.1(1,2,3,4,5,6), §2, and the WaterMask-TSFill contract already inspected in `evidence_packet.md` §5:

1. **Scope statement.** HydroFragments accepts any binary (or pre-thresholded probabilistic) water-mask time series **plus an aligned valid-observation layer**. WOfS is the reference/default source used in development, not a hard requirement (spec title block, §1 principle 7). State this up front so users with non-WOfS sources (Water Detect on Sentinel-2/PlanetScope, thresholded NDWI/MNDWI) know they're supported, not accommodated as an afterthought.
2. **The canonical upstream contract (WaterMask-TSFill).** Document the four-variable Zarr schema verified in `evidence_packet.md` §5: `water_mask` (uint8: 0=dry, 1=water, 254=outside AOI, 255=invalid/unresolved), `confidence` (uint8, 255=N/A), `method_flag` (uint8, provenance), `observed` (bool). State plainly, per `spec_compliance.md` B2, that sentinel values `254`/`255` must never be silently cast/collapsed to dry — this is the single most consequential input-format fact in the whole spec, because getting it wrong corrupts every downstream metric.
3. **Generic (non-WaterMask-TSFill) input contract.** For users bringing their own GeoTIFF/NetCDF/Zarr mask, specify the minimum required pair: a binary/probabilistic mask array and a valid-observation array, aligned in transform/CRS/shape/time — and that misalignment must raise, not silently resample (spec §1.1.4, `spec_compliance.md` A4).
4. **Probabilistic masks.** If input is probabilistic, document that it must be thresholded once, and that `water_threshold`, `threshold_method`, `probability_source` become mandatory output metadata (spec §1.1.6).
5. **CRS and units.** Equal-area CRS requirement, EPSG:3577 default for Australian deployments, per-pixel area array as the alternative for non-Australian AOIs, and the length-vs-area distortion caveat (spec §1.1.1) stated plainly, not just for area but explicitly warning that length quantities (`L_ref`, gap distances, skeleton length) are not equally protected by an equal-area CRS choice.
6. **Monthly compositing.** Explain `max_water` as default, `median` as the mandatory secondary composite for dry-down/end-dry detection, and that both must be recorded in output metadata (spec §1.1.2). This section should explicitly flag that this is *input-adjacent* (a required transformation before metrics run), not a metric itself, so users understand where in their pipeline this sits.
7. **Minimum mapping unit and connectivity rule.** `min_patch_pixels` default (3 for Landsat/WOfS-class 30 m), `connectivity_rule` (8-neighbour default), and that both are recorded in output (spec §1.1.3, §1.1.4).
8. **Validity floors.** Distinguish `min_valid_obs` (per-pixel long-term floor) from `min_valid_fraction_month` (per-month AOI/zone reportability floor) — spec is explicit these must not share one parameter name (§1.1.5).
9. **AOI polygon and optional drainage layer.** What's required (AOI polygon, always) vs. optional (drainage centreline — enables Zone 1, real `L_ref`, gap/TCF/DCI; absence triggers documented fallbacks per spec §3).
10. **Worked example.** A minimal, runnable snippet once the v1.2 input adapter exists — explicitly deferred, not written with placeholder/invented code against an API that doesn't exist yet.
11. **Cross-references.** Link to `docs/for-managers.md` (plain-language version of caveats 5–6 above) and to the connectivity positioning doc (§7 matrix) for anything DCI/PC/IIC-related that a data-format reader might stumble into.

This document should **not** be written by copy-pasting spec prose verbatim — the spec is the implementation contract for engineers; `input_format.md` is the practitioner-facing distillation. But its content must trace 1:1 to spec §1.1/§2/§3, and every mandatory-metadata claim above must match `spec_compliance.md`'s compliance IDs (A1–A16, B1–B5) so a future compliance re-check can verify the doc against the same IDs.

---

## 11. Practitioner quickstart outline

Distinct from the README quickstart snippet (§8 item 5, a few lines) and from the full API doc (§13) — this is a short, task-oriented document (or README section, if kept combined) aimed at "I have water mask data, how do I get metrics out."

1. **Prerequisites** — Python version, conda/GDAL setup (matches current README's conda instructions, which are otherwise correct and can be preserved), and which input shape is expected (point to `docs/input_format.md`, §10).
2. **Minimal working example** — smallest possible call: mask + AOI polygon → metrics table. Must use a *real* runnable import path once the package identity question is settled (§5); must not repeat the current README's broken `irivermetrics.irm_main` import.
3. **Reading the output** — for v1.2, this means explaining the tidy long schema (spec §7: `catchment_id | aoi_id | zone | window_id | date | hy | hy_anchor | metric | value | ...`), not the legacy wide CSV. This is a genuinely different mental model from the current wide-CSV-per-section format documented in `module2.md`, and the quickstart should say so explicitly for anyone coming from the legacy tool.
4. **Interpreting your first metrics** — a short pointer to which metrics are "safe to read directly" per the Core tier (spec §5.1) with a one-line caution that shape/connectivity metrics need the fuller glossary (link to for-managers.md and/or the API doc's per-metric detail).
5. **Common gotchas** — sentinel handling if bringing your own mask, CRS requirement, what happens with no drainage layer (Zone 1 skipped, not faked), what an `edge_flag` means. Each gotcha should link to the fuller explanation rather than re-explain it (avoid drift between two versions of the same caveat, per the manager audit's "single consistent visual marker" principle §5 — the written-word analogue here is "single source of truth per caveat, quickstart only summarizes and links").
6. **Where to go next** — links to `docs/input_format.md`, `docs/for-managers.md`, the API reference, and the spec itself for anyone who wants the full contract.

**Sequencing note:** this document cannot be usefully written until the v1.2 API stabilises (its own worked example depends on a real call signature). It is listed here as an outline so the shape is ready, matching the precedent already set in `spec_compliance.md`'s refactor sequence step 13 ("current quickstart... after API stabilises").

---

## 12. API docs outline

Today's only API doc is `docs/module2.md`, documenting exactly one legacy function (`calculate_metrics`) with a wrong import path and an outdated metric register (§2 above). The v1.2 API surface is considerably larger per the spec's module boundaries (`io`, `zones`, `hydroyear`, `metrics`, `aggregate`, `guards` — named in `spec_compliance.md`'s "Minimal refactor sequence" note). Outline for the eventual API reference:

1. **Input contract reference** — the `(water, valid_obs, [confidence, method_flag])` object/parameters; cross-reference `docs/input_format.md` rather than duplicate it.
2. **Configuration reference** — every named config field the spec locks as mandatory: `crs`, `area_unit`, `length_unit`, `min_patch_pixels`, `connectivity_rule`, `monthly_composite`, `water_threshold`, `threshold_method`, `min_valid_obs`, `min_valid_fraction_month`, `t_persist`, `t_season`, `state_flag_connectivity_metric`, `state_flag_connectivity_threshold`, `awre_length_method` — each with default, meaning, and which output column it controls (spec §1.1, §3, §6.1). This table doubles as the reproducibility/config-hash documentation the spec asks to be foregrounded (§1.1.7) — see item 6 below.
3. **Metric register reference**, organized by tier exactly as spec §5:
   - **Core** (Occurrence frequency, RA, APSEC, LPSEC, N, LPI, AWRe, Dry-down rate)
   - **Secondary** (AWMSI, MESH, Pool width distribution, Inter-pool gap, Reconnection timing, Refuge spatial stability, TCF, DCI)
   - **Exploratory** (NNI — with the exploratory-only framing intact; see note below)
   - **Dropped** (PF, PLF, AWMPA, AWMPL, AWMPW, connected-components count, largest-component fraction, centrality) — kept as a documented table with the reason for each drop (per spec §4.1's "redundant not invalid" nuance for AWMPA — this distinction must survive into the API docs, not just the spec, per `scientific_metrics_audit.md`'s explicit warning against overclaiming the circularity argument for metrics that are actually just redundant).
   - Every metric entry must carry: definition, tier, citation/positioning (verbatim / adapted / novel, per spec §4 column), and — critically, per `scientific_metrics_audit.md` §16.1 F-5 — **hypothesis-mood language for any interpretive claim not yet in spec §6.18's "demonstrated" column.** The API doc must not be more confident than the spec's own validation-status table.
   - **NNI note:** `scientific_metrics_audit.md` R5 recommends considering cutting NNI from v1 entirely; if that decision is taken before this doc is written, the API reference should simply omit it rather than document a metric that no longer ships. If NNI is retained, this doc must carry its exploratory-only, `N<10`-unstable framing without softening (and it must never appear in `docs/for-managers.md` regardless of this decision, per `manager_interpretation_audit.md` §3 Danger 7).
4. **Connectivity module positioning** — a dedicated subsection (not folded into the metric table) explicitly stating TCF/RC/DCI's relationship to PC/IIC/DCI literature, per the spec's locked documentation requirement (§1.1.11) and `scientific_metrics_audit.md` §8's residual sharp edges (RC-vs-DCI "why both," TCF's narrow-not-absent novelty). This is not optional polish — the spec treats silence here as "the single most exposed weakness the audit found in the whole document" (§1.1.11).
5. **Output schema reference** — the tidy long table columns (spec §7): identifiers, `zone`, `hy_anchor`, `edge_flag` enum, `hy_confidence`, `source`/`resolution_m`, distributional-metric row convention (one row per summary statistic). Include the zone circularity guard as a documented *behavior*, not just a warning (persistence metrics never emit `zone ∈ {2,3,4}`).
6. **Reproducibility metadata reference** — `run_id`, `config_hash`, `package_version`, `git_sha`, and how they're computed (once that judgment call from `spec_compliance.md` Q9 is resolved). Spec §1.1.7 explicitly asks for this to be foregrounded in docs as "a genuine strength of the design."
7. **Validation status appendix** — reproduce (or link live to) spec §6.18's asserted-vs-demonstrated table. This is the anti-overclaiming backstop for the whole API doc; per `scientific_metrics_audit.md` §17, this table's mood should govern the entire document, so making it visible and current in the API reference (not buried only in the spec) is high-value and cheap.

**What this doc must not do:** it must not be written before the metric register and output schema stabilize in code — an API reference that drifts from the actual function signatures is worse than no API reference (it actively misleads, exactly the failure mode this whole audit documents for the current `module2.md`). Draft the structure now; fill in signatures once implementation lands.

---

## 13. Basic packaging / README / install clarity — consolidated recommendations

Distinct from §1/§3's findings (what's wrong) — this section is the minimal fix list for "can a new user get from zero to a running example" clarity, ranked by how cheap the fix is relative to how much confusion it currently causes:

| Fix | Cost | Confusion it currently causes |
|---|---|---|
| Add a status banner to README (§8 item 2) stating this is a pre-v1.2 migration snapshot, with a link to the spec | Trivial (a few sentences) | Prevents every other doc issue from reading as "broken project" rather than "known migration state" |
| Correct the `git clone` URL and import path to match this actual repository/package | Trivial once package identity is confirmed (§5) | Currently the single most damaging error — literally cannot follow the README as written |
| Remove or clearly quarantine `docs/module1.md` (`waterdetect_batch`) until confirmed present in code | Trivial (add a "not yet ported" banner, or remove) | A practitioner will otherwise try to import a function that isn't there |
| Correct `docs/module2.md`'s metric table to match what `ecofragments/utils/calc_metrics.py` actually emits today (16 legacy metrics, not a mix of legacy-plus-invented) | Low (it's already close — mostly needs the import-path fix and a "legacy, pre-v1.2" label) | A practitioner reading it as current-and-correct will use dropped metrics in downstream analysis |
| Fix `docs/architecture.md`'s scope claim to match spec §0 (river-focused, not generic) — or, if this is deferred to the rebrand, add a note flagging the contradiction | Trivial (a note); Low (full rewrite) | Confuses contributors about what kinds of PRs are in-scope |
| Add `[project.urls]` (repository, documentation) and a `classifiers` list to `pyproject.toml` | Trivial | Currently `pip show ecofragments` gives a dead end for finding docs/issues |
| Reconcile `pyproject.toml` `description` field with actual river-focused scope | Trivial | PyPI/pip metadata currently claims a different tool than the one in this repo |
| Add LICENSE file + reference it from README and `pyproject.toml` | Low (needs a license decision, not just docs work — flagged as F11 in `spec_compliance.md`, this report does not make that decision) | Currently unclear whether/how this can be reused or contributed to |
| Fix `tests/conftest.py`'s regression fixture path (code fix, not docs — listed here only because it blocks the "run the tests to see it work" onboarding step a good README should point to) | Trivial (one path string, per `repo_triage.md` item 17) | A new contributor following any "run the test suite" instruction hits an undocumented failure |

None of these require the v1.2 metric/architecture migration to be complete first — they are honest-labelling and correctness fixes to *existing* content, separable from and much cheaper than drafting the new v1.2 documents in §9–§12.

---

## 14. Reproducibility docs

Spec §1.1.7 is explicit: "Configuration is part of the scientific result... This reproducibility discipline is a genuine strength of the design — it directly satisfies criteria JOSS now weighs heavily... **Foreground it in docs and in the companion paper.**" Current state: **zero documentation exists on this topic anywhere in the repo.**

What's needed, once implemented (per `spec_compliance.md` E9, Q9):

1. A short conceptual doc (or a section within the API reference, §12 item 6) explaining *why* HydroFragments emits `run_id`/`config_hash`/`package_version`/`git_sha` with every run — the practitioner-facing "why should I care" framing, not just the schema.
2. A worked example showing two runs with identical config producing identical `config_hash`, and one run with a changed threshold producing a different hash — this is the single most convincing demonstration of the reproducibility claim and costs little once the hashing mechanism exists.
3. An explicit statement of what is *and isn't* covered by the hash (per `spec_compliance.md` Q9's open question — e.g., are absolute file paths excluded, is it content-addressed) — this must be resolved as an implementation decision first; the doc should not paper over an unresolved design question with vague language.
4. Cross-reference from `docs/for-managers.md`'s escalation-path section (`manager_interpretation_audit.md` §6 item 7) — a manager who needs to check back with the analysis team benefits from knowing a specific run can be exactly reproduced and audited.

**This is currently the cleanest "genuine strength, zero doc coverage" gap in the whole audit** — unlike the metric-interpretation caveats (which require careful hedging), the reproducibility story is a straightforward engineering fact once built, and documenting it needs no validation data, no Gilbert run, no hedged language. It should be one of the first *new* (not corrective) docs written once the config/hash mechanism lands in code.

---

## 15. Future publication notes — non-blocking

**This section is explicitly informational, not a task list, and nothing here should be read as gating v1.2 implementation or documentation work.** Per this audit's own constraint and per `docs/audit_implementation_plan.md`'s existing framing: *"JOSS is not a near-term priority... do not let JOSS artifacts block v1.2 implementation."* This report extends that same non-blocking status to documentation work specifically.

- **JOSS.** Spec §13 (referenced across multiple prior audits) ties reproducibility discipline to JOSS review criteria, but no prior audit or this one treats JOSS submission as scheduled or imminent. `spec_compliance.md` F12 lists "Hosted docs, practitioner quickstart, manager guide, paper, Zenodo" together as **absent**, correctly bundled as future-state deliverables contingent on "Stable API and validated outputs" — i.e., blocked on the science and code migration, not on docs effort. No JOSS `paper.md`, Zenodo metadata, or submission checklist should be drafted before the v1.2 metric register and output schema are implemented and validated (§6.18's asserted→demonstrated work, per `scientific_metrics_audit.md`).
- **Companion methods paper.** Spec §10/§16 scope a paper distinct from JOSS (the metric-register/circularity-reformulation argument). `scientific_metrics_audit.md` §16.2 is explicit about which claims are "publishable without data" (the circularity argument itself) vs. which need the Gilbert validation run first (V1–V8). Documentation work on this front should track the same split: a docs page *summarizing the paper's intended scope and framing* is fine to draft early (low cost, matches spec §16 content already written); a docs page presenting *results* must wait for the validation runs.
- **Six-month public-development-history evidence.** `spec_compliance.md` F13 notes local git history currently shows one commit ("Initial commit: ecofragments package (clean start)"), dated 2026-05-30, which does not by itself demonstrate any predecessor-history preservation or public development timeline. This is a repository/governance decision (whether/how to graft predecessor history), not a documentation-drafting task, and is explicitly out of scope for this docs audit to resolve.
- **Recommended framing if/when any publication-facing doc is drafted:** state plainly, in the doc itself, that publication readiness (JOSS or the methods paper) is a future milestone contingent on (a) v1.2 implementation completing, (b) the §6.18 validation checklist advancing from "asserted" to "demonstrated" on at least the items `scientific_metrics_audit.md` §16.2 flags as paper-blocking (V1, V2, V3 minimum), and (c) a governance decision on repository history. Do not present a submission timeline. Do not claim any current output is publication-ready.

**Net position for this audit:** future publication material is real, tracked, and already well-scoped by prior audits — it simply has no current documentation home, and per this audit's constraints, creating one is optional, low-priority, cost-effective only if drafted as scope-and-framing (not results), and must never be allowed to compete for priority against the corrective work in §6/§13 or the new-document work in §9–§12, §14.

---

## 16. Priority summary for the next phase

Ordered by (severity of current harm) × (cheapness of fix), synthesising §6, §13, §14 above. This is a priority ordering only — not a plan, not an approval to edit source, consistent with this audit's own constraint.

**Tier 1 — corrective, cheap, high-harm-if-left:**
1. README status banner + corrected install/import path (§8 items 2, 4, 5; §13 row 1–2)
2. Quarantine/remove `docs/module1.md` until `waterdetect_batch` is confirmed to exist in code (§13 row 3)
3. Label `docs/module2.md` as legacy/pre-v1.2; fix its import path (§13 row 4)
4. Fix or flag `docs/architecture.md`'s scope contradiction (§13 row 5)
5. `pyproject.toml` metadata hygiene: description, urls, classifiers (§13 rows 6–7)

**Tier 2 — new documents, structurally ready, content gated on implementation:**
6. `docs/input_format.md` (§10) — draftable once the input adapter contract is locked (spec Q1/Q2 resolved)
7. API reference skeleton (§12) — draftable once metric register/output schema land in code
8. `docs/for-managers.md` (§9) — structure/glossary draftable now per `manager_interpretation_audit.md`; numbers wait for a validation run
9. Practitioner quickstart (§11) — waits on stable v1.2 call signature

**Tier 3 — new, low urgency, no science/code dependency:**
10. Reproducibility docs (§14) — can be written as soon as config-hash mechanism exists; no validation data needed
11. Future publication notes (§15) — optional, non-blocking, scope-only framing if drafted at all

This ordering deliberately puts all Tier 1 items before any Tier 2/3 item, because Tier 1 fixes active harm (a user or contributor being actively misled) at near-zero cost, while Tier 2/3 items require either implementation progress or validation data this docs audit cannot itself produce.
