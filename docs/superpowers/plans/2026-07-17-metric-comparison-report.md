# Metric Comparison Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `docs/metric_comparison_report.html` with a self-contained, equation-first report that documents every legacy/current/deprecated metric and shows lineage in an offline SVG map.

**Architecture:** Keep one standalone HTML file with inline CSS, an inline `metricRecords` data model, and vanilla JavaScript that renders tables, filters, and an SVG lineage graph from that model. Add a small Python documentation test that validates record completeness and offline constraints without requiring a browser or new dependency.

**Tech Stack:** HTML5, inline CSS, vanilla JavaScript, inline SVG, Python 3.10+, pytest, repository Markdown/spec/code sources.

## Global Constraints

- Keep `docs/metric_comparison_report.html` usable from `file:///...`.
- No external scripts, stylesheets, fonts, chart libraries, runtime JSON, or network requests.
- Inline citations use author-year text only; no bibliography section.
- Equations use readable Unicode/plain text; no MathJax or external renderer.
- Do not change metric algorithms, registry contents, schemas, or production outputs.
- Preserve the current repository's unrelated working-tree changes.

---

### Task 1: Add report completeness tests and inventory contract

**Files:**
- Create: `tests/docs/test_metric_comparison_report.py`
- Read: `docs/HydroFragments_v1.2_spec.md:236-454`
- Read: `hydrofragments/metrics/registry.py:35-178`
- Read: `docs/metric_comparison_report.html`

**Interfaces:**
- Produces a test contract for the HTML data model: `metricRecords` must expose stable `id`, `side`, `status`, `equation`, `citation`, and `source` fields.
- The required id set contains legacy/context ids (`section_area_km2`, `section_length_km`, `wet_area_km2`, `wet_length_km`, `wet_perimeter_km`, `npools`, `awmsi_legacy`, `awre_legacy`, `awmpa`, `awmpl`, `awmpw`, `apsec_legacy`, `lpsec_legacy`, `pf`, `plf`, `pp_mean`, `ra_area`, `pixel_persistence`, `nni`, `pcf`, `centrality`) and current ids (`occurrence`, `refuge_area`, `apsec`, `number_of_pools`, `lpi`, `awre`, `awmsi`, `recurrence`, `hydroperiod`, `extent_contraction`, `reconnection_timing`, `refuge_spatial_stability`, `lpsec`, `inter_pool_gap`, `mesh`, `pool_width`, `realised_connectivity`, `tcf`).

- [ ] **Step 1: Write failing completeness tests**

```python
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT = REPO_ROOT / "docs" / "metric_comparison_report.html"
REQUIRED_IDS = {
    "section_area_km2", "section_length_km", "wet_area_km2", "wet_length_km",
    "wet_perimeter_km", "npools", "awmsi_legacy", "awre_legacy", "awmpa",
    "awmpl", "awmpw", "apsec_legacy", "lpsec_legacy", "pf", "plf",
    "pp_mean", "ra_area", "pixel_persistence", "nni", "pcf", "centrality",
    "occurrence", "refuge_area", "apsec", "number_of_pools", "lpi", "awre",
    "awmsi", "recurrence", "hydroperiod", "extent_contraction",
    "reconnection_timing", "refuge_spatial_stability", "lpsec", "inter_pool_gap",
    "mesh", "pool_width", "realised_connectivity", "tcf",
}

def _report() -> str:
    return REPORT.read_text(encoding="utf-8")

def test_report_is_self_contained_and_has_metric_records():
    text = _report()
    assert "const metricRecords" in text
    assert '<script src=' not in text.lower()
    assert '<link rel="stylesheet"' not in text.lower()
    assert "fetch(" not in text

def test_all_metric_ids_and_required_fields_are_present():
    text = _report()
    for metric_id in REQUIRED_IDS:
        assert f'id: "{metric_id}"' in text
    assert text.count("equation:") >= len(REQUIRED_IDS)
    assert text.count("citation:") >= len(REQUIRED_IDS)
    assert text.count("source:") >= len(REQUIRED_IDS)

```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `python -m pytest tests/docs/test_metric_comparison_report.py -q`

Expected: FAIL because the existing report has no `metricRecords` model and no lineage SVG/filter hooks.

- [ ] **Step 3: Commit the test contract**

```powershell
git add tests/docs/test_metric_comparison_report.py
git commit -m "test: define metric report completeness contract"
```

### Task 2: Replace report content with complete metric data model and equation tables

**Files:**
- Modify: `docs/metric_comparison_report.html`
- Test: `tests/docs/test_metric_comparison_report.py`

**Interfaces:**
- `const metricRecords` is the literal array containing one object for every record listed in Task 2 and is the single source for all visible metric rows and lineage nodes.
- Record shape:

```javascript
{
  id: "occurrence",
  label: "Occurrence frequency",
  side: "current",
  family: "persistence",
  status: "kept",
  equation: "OCCₚ = Σₜ Wₚ,ₜ / Σₜ Vₚ,ₜ × 100",
  meaning: "Per-pixel wet frequency using valid observations.",
  denominator: "valid observations per pixel",
  dependencies: "valid-observation mask",
  citation: "(Pekel et al., 2016; Mueller et al., 2016)",
  source: "HydroFragments_v1.2_spec.md §6.17; metrics/persistence.py",
  links: ["pixel_persistence"]
}
```

- [ ] **Step 1: Define all legacy/context/deprecated records**

Use the legacy wide CSV names and current report rows. For formulas, encode `PF = N / WA`, `PLF = N / WL`, `AWMPA = Σ(aᵢ²) / Σaᵢ`, `AWMPL = Σ(lᵢaᵢ) / Σaᵢ`, `AWMPW = Σ(wᵢaᵢ) / Σaᵢ`; mark raw area/length/perimeter/reference fields as descriptive/context measurements, not invented response equations. Cite legacy suite entries as `(Tayer et al., 2023)` and preserve the circular-denominator explanation.

- [ ] **Step 2: Define all current registry records**

Include equations and sources for occurrence, refuge area, APSEC, pool count, LPI, AWRe, AWMSI, recurrence, hydroperiod, extent contraction, reconnection timing, refuge spatial stability, LPSEC, inter-pool gap, MESH, pool width distribution, realised connectivity, and TCF. Use exact spec equations: `Nₜ`, `WAₜ`, `APSECₜ`, `LPSECₜ`, `OCCₚ`, `RAθ`, AWRe, AWMSI, LPI, MESH, RC edge/reachable-pair form, TCF active-month fraction, reconnection lag, and Jaccard stability.

- [ ] **Step 3: Add citation-anchor/deprecated records**

Represent DCI as `status: "citation_anchor"` because it is required for positioning but is not in `METRIC_REGISTRY`; represent NNI as exploratory-only, PCF as renamed/deprecated, and centrality as cut. Use author-year citations only: `(Cote et al., 2009)`, `(Saura & Pascual-Hortal, 2006; Saura & Pascual-Hortal, 2007)`, `(Rubio & Saura, 2012)`, `(Clark & Evans, 1954; Donnelly, 1978)`, `(Pavelsky & Smith, 2008; Yang et al., 2020)`, `(Sheldon et al., 2010; Fullerton et al., 2010)`, `(Costigan et al., 2016; Gallart et al., 2012)`, and `(Jaeger, 2000)`.

- [ ] **Step 4: Render inventory sections from `metricRecords`**

Add equation cards or table columns for equation, denominator/reference, meaning, status, dependency, citation, and source. Split sections by `side`/status without duplicating record content. Include explicit caveats for width-not-depth, extent contraction-not-hydrograph recession, fixed denominators, valid observations, resolution sensitivity, and connectivity positioning.

- [ ] **Step 5: Run completeness tests and commit**

Run: `python -m pytest tests/docs/test_metric_comparison_report.py -q`

Expected: PASS for record ids and required fields. SVG/filter assertions are added in Task 3.

```powershell
git add docs/metric_comparison_report.html tests/docs/test_metric_comparison_report.py
git commit -m "docs: add equation and citation metric inventories"
```

### Task 3: Add offline lineage SVG, filters, and responsive styling

**Files:**
- Modify: `docs/metric_comparison_report.html`
- Test: `tests/docs/test_metric_comparison_report.py`

**Interfaces:**
- `renderLineage(records)` draws `#lineageGraph` from `record.links`; edge classes are status values `kept`, `reworked`, `replaced`, `decomposed`, and `deprecated`.
- `renderTables(records, query, status, family)` updates visible rows without network calls.
- `applyFilters()` reads `#metricSearch`, `#statusFilter`, and `#familyFilter`.

- [ ] **Step 1: Add filter controls and status/family legend**

Place search and select controls before inventories. Populate status/family options from the record model. Show filtered-record count and a reset button.

- [ ] **Step 2: Draw inline SVG lineage map**

Use a fixed `viewBox` with left/right columns. Create one `<g class="lineage-node status-kept">`-style group per record and one `<path class="lineage-edge edge-kept">`-style path per link, substituting the record's actual status class. Use text labels, marker-end arrows, status colors, and an accessible `<title>`/`aria-label>`. Add a textual lineage list below the SVG so print and screen readers retain all mappings.

- [ ] **Step 3: Add responsive and print CSS**

Keep tables horizontally scrollable below 780px, stack summary panels, allow SVG horizontal scrolling, preserve status colors in print, and keep equation cards readable with monospace or system fallback.

- [ ] **Step 4: Extend tests for SVG and filters**

Add `test_lineage_svg_and_filter_hooks_exist()` to assert `<svg id="lineageGraph"`, `metricSearch`, `statusFilter`, `familyFilter`, and `aria-label` are present. This test belongs here because the hooks do not exist until this task.

- [ ] **Step 5: Run complete report tests and commit**

Run: `python -m pytest tests/docs/test_metric_comparison_report.py -q`

Expected: PASS, including self-contained HTML, all ids/fields, SVG, accessibility label, and filter hooks.

```powershell
git add docs/metric_comparison_report.html tests/docs/test_metric_comparison_report.py
git commit -m "feat: add offline metric lineage graph and filters"
```

### Task 4: Verify rendered report and documentation hygiene

**Files:**
- Modify: `docs/metric_comparison_report.html` only if verification finds a defect.
- Test: `tests/docs/test_metric_comparison_report.py` if assertions need tightening.

**Interfaces:**
- No production-code changes. Verification checks the committed report as a local artifact.

- [ ] **Step 1: Parse HTML and check no external dependencies**

Run:

```powershell
python -c "from pathlib import Path; import re; p=Path('docs/metric_comparison_report.html'); t=p.read_text(encoding='utf-8'); assert '<!doctype html>' in t.lower(); assert not re.search(r'<script[^>]+src=|<link[^>]+href=', t, re.I); assert 'fetch(' not in t; print('offline HTML checks passed')"
```

Expected: `offline HTML checks passed`.

- [ ] **Step 2: Run targeted documentation tests**

Run: `python -m pytest tests/docs/test_metric_comparison_report.py tests/docs/test_vocabulary_scan.py -q`

Expected: PASS. The vocabulary scan must accept necessary negative caveats and reject unsupported width/depth, recession-as-flow, or permanence claims.

- [ ] **Step 3: Perform local visual smoke check**

Open `docs/metric_comparison_report.html` via the local file path. Verify the header, equation inventories, search/status/family filters, SVG left/right mapping, textual fallback, and narrow-window horizontal scrolling. If no browser is available, inspect the SVG and table DOM with a local HTML parser and record that visual inspection was unavailable.

- [ ] **Step 4: Commit verification-only fixes**

```powershell
git add docs/metric_comparison_report.html tests/docs/test_metric_comparison_report.py
git commit -m "test: verify metric comparison report artifact"
```

## Self-review

- Spec coverage: standalone HTML and no network dependencies are covered by Tasks 2 and 4; complete equations/citations/statuses by Task 2; SVG lineage and accessibility by Task 3; filters and responsive layout by Task 3; source reconciliation and stale-status correction by Task 2; verification by Task 4.
- Placeholder scan: no `TBD`, `TODO`, or unspecified implementation steps appear in this plan.
- Type consistency: `metricRecords`, `renderLineage(records)`, `renderTables(records, query, status, family)`, and `applyFilters()` are named consistently across Tasks 2–3.
