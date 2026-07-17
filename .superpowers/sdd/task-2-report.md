# Task 2 report: equation and citation metric inventories

**Date:** 2026-07-17
**Artifact:** `docs/metric_comparison_report.html`

## Implementation

- Replaced the stale static report tables with one inline `const metricRecords` data model containing the exact 39 required IDs (21 legacy/context/deprecated and 18 current registry metrics).
- Every record includes an equation or explicit descriptive/context measurement, denominator/reference, meaning, dependency gate, side, family, status/tier, author-year citation, repository source, caveat, and lineage links where applicable.
- Added the required formulas: fixed-reference APSEC/LPSEC/LPI/MESH, valid-observation occurrence/recurrence/hydroperiod/TCF, refuge threshold with valid support, AWRe/AWMSI, legacy circular PF/PLF/AWMPA/AWMPL/AWMPW, EDT width distribution, Jaccard refuge stability, dual-composite extent contraction, reconnection lag, and RC edge/reachable-pair forms.
- Added a citation-only DCI anchor (`status: "citation_anchor"`) outside the registry, with positioning notes for RC, reconnection timing, and TCF.
- Added visible executive summary, legacy inventory, current inventory, deprecated/quarantined inventory, and citation/source note. All rows are rendered from `metricRecords`; no external scripts, stylesheets, fetches, or runtime data files are used.

## Verification

Command:

```text
python -m pytest tests/docs/test_metric_comparison_report.py -q
```

Result: **2 passed**.

The focused test confirms the standalone/offline contract and finds every required ID with `equation`, `citation`, and `source` fields. The report also explicitly carries the fixed-vs-moving denominator, valid-observation, resolution, width-not-depth, extent-contraction-not-hydrograph, and RC/TCF positioning caveats.

## Self-review

- [x] Exact 39 metric IDs represented once in `metricRecords`.
- [x] Current statuses follow registry tiers/dependencies rather than stale deferred labels.
- [x] Legacy circular metrics and deprecated names are retained with replacement/deprecation lineage.
- [x] All citations are author-year form only; source fields point to repository files/sections.
- [x] HTML remains self-contained and offline.
- [x] Unrelated worktree changes were preserved.

## Review-fix evidence (2026-07-17)

- Corrected `occurrence` to document the locked U2/Q1 season-stratified estimator implemented in `hydrofragments/metrics/persistence.py`: equal-weight mean of supported calendar-month ratios `mean_m[Σ_y(W & V)/Σ_y V] × 100`, while retaining the canonical pooled `Σ_t W/Σ_t V × 100` as an explicit distinction. The valid-observation support floor and unsupported-month rule are stated in the denominator/caveat fields.
- Set current AWMSI to `status: "secondary"`, `tier: "secondary"` per spec §6.2, with an explicit note that the registry `MetricSpec` default `tier="core"` is stale and the scientific specification governs this report.
- Corrected NNI source to `docs/HydroFragments_v1.2_spec.md §6.8` as a design reference, explicitly recording that no production calculator exists. Normalized legacy source paths to `docs/...`, `hydrofragments/metrics/...`, or fixture paths.
- Re-ran `python -m pytest tests/docs/test_metric_comparison_report.py -q`: **2 passed**. SVG/filter assertions remain deferred to Task 3.
