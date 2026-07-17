# Metric comparison report design

## Goal

Refresh `docs/metric_comparison_report.html` into a self-contained offline HTML report that documents every legacy and current metric with its equation and an author-year citation, while making legacy-to-current lineage visually obvious.

## Scope

The report covers:

- every legacy metric currently represented by the report and legacy fixture headers;
- every metric registered in `hydrofragments/metrics/registry.py`;
- deprecated, dropped, exploratory, or quarantined metrics named by the v1.2 specification;
- one or more current descendants for each legacy metric where a replacement or decomposition exists.

Inline citations use plain author-year text only, for example `(Pekel et al., 2016)`. No separate bibliography is required. Repository source sections remain the traceability basis and can be named in a compact source note.

## Architecture

Use one standalone HTML file. Keep CSS and JavaScript inline; do not add network dependencies, external fonts, chart libraries, or runtime data files. Existing local-file behavior must continue to work from `file:///.../docs/metric_comparison_report.html`.

Represent report content as an inline data model in JavaScript or HTML data attributes. Each metric record contains:

- stable id and display name;
- side (`legacy`, `current`, or `deprecated`);
- family and status (`kept`, `reworked`, `replaced`, `context`, `secondary`, `deferred`, `exploratory`, `dropped`);
- equation text;
- plain-language meaning;
- denominator/reference and dependency notes;
- caveats and implementation status;
- author-year citation;
- source section/file for audit traceability.

Derive visible tables, filters, and SVG nodes from this model so labels do not drift between sections.

## Report sections

1. Header with purpose, version context, offline badge, and status legend.
2. Executive summary explaining moving-denominator circularity, valid-observation denominators, fixed references, dependency gates, and why the refactor matters.
3. Legacy metric inventory. Every row shows metric, equation or explicit “descriptive/context only”, meaning, decision, replacement/current link, citation, and caveat.
4. Current metric inventory. Every row shows metric, equation, denominator/reference, unit/statistic, registry tier/status, dependency gate, interpretation, citation, and caveat.
5. Deprecated and quarantined inventory, including PF, PLF/PFL, AWMPA, AWMPL, AWMPW, NNI, centrality, and PCF. Explain whether each was dropped, renamed, replaced, or exploratory-only.
6. Legacy-to-current lineage map rendered as inline SVG. Legacy nodes appear on the left, current/deprecated nodes on the right. Curved paths encode `kept`, `reworked`, `replaced`, `decomposed`, and `deprecated`; one legacy metric may connect to multiple current metrics. Include a legend and accessible text fallback.
7. Citation/source note naming the local specification, audit, registry, and implementation modules used to reconcile formulas and statuses.

## Equation policy

Use readable Unicode/plain-text equations inside equation cards; no MathJax or external renderer. Preserve mathematical distinctions:

- fixed denominators (`A_ref`, `L_ref`, `A_total`);
- valid-observation denominators for occurrence, recurrence, and hydroperiod;
- distributional metrics as unweighted summaries where required;
- dual-composite and HY-anchor conditions for extent contraction;
- graph and reachable-pair forms for RC/TCF positioning;
- Jaccard form for refuge spatial stability.

When a metric is descriptive context rather than a scalar response, show its measured quantity and mark denominator as “not applicable” instead of inventing an equation.

## Interaction and accessibility

Provide client-side search plus filters for side, status, and metric family. Filter state must not require a server. Tables remain usable on narrow screens via horizontal scrolling. SVG nodes have text labels, high-contrast status colors, a legend, and a hidden/visible textual lineage list for screen readers and print.

## Source reconciliation

Treat the v1.2 specification as the equation and citation contract, implementation modules as calculation/status evidence, and `registry.py` as current tier/dependency evidence. Reconcile stale prose in the existing HTML against current code; do not copy old “deferred” labels when the registry and implementation now show a metric as implemented.

## Verification

- Parse the resulting HTML to confirm it is self-contained and has no external URLs, scripts, or stylesheet dependencies.
- Check every metric record has equation/descriptive measurement, status, citation, and source fields.
- Check every legacy record appears in the SVG lineage map or is explicitly marked context-only.
- Open the file locally or use a headless browser if available; verify tables, filters, SVG, and narrow viewport behavior.
- Run targeted documentation tests if they cover report/source traceability.

## Non-goals

- No changes to metric algorithms, registry contents, schemas, or production outputs.
- No external citation download or bibliography management.
- No interactive graph framework or network-backed assets.
