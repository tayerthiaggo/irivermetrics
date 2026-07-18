/caveman

# Stage 1a — Science Audit (HydroFragments)

## Task

Audit the HydroFragments codebase to make it user-ready. THIS stage covers the **science axis only**: verify every equation, its source/citation, the math as coded, and the overall scientific soundness. A separate parallel stage handles performance — do not touch efficiency here.

HydroFragments is a remote-sensing hydrology package computing surface-water **fragmentation metrics** over satellite image cubes: extent, patch morphology, clustering, persistence/dynamics, and river/landscape connectivity (including DCI — Dendritic Connectivity Index — and a length-weighted RC_pair form). Metric definitions live in `hydrofragments/metrics/*.py` and `hydrofragments/spatial/connectivity*.py`.

## What you must do

Work metric by metric. For EACH metric implemented in the code:

1. **Locate the formula as coded** — cite `file:line`. Write out the equation the code actually computes (not what the docstring claims).
2. **Find the claimed source** — docstring citation, `docs/` reference, or spec. Load and read:
   - `docs/HydroFragments_v1.2_spec.md`, `docs/paper-summary.md`, `docs/metrics/*.md`
   - `docs/audit/scientific_metrics_audit.md` (prior audit — verify or refute its findings, don't trust blindly)
3. **Verify the math matches the source.** Check: correct formula, correct units, correct normalization, correct handling of edge cases (empty patches, single-pixel patches, division by zero, all-water / all-dry frames, NaN/masked pixels, zero-length segments in DCI).
4. **Verify citations are real and correctly attributed.** Flag any equation attributed to a paper that doesn't define it, or any ungrounded "magic" constant.
5. Check unit consistency across the pipeline (pixel_size_m propagation, m vs m², area vs count).

Prioritize connectivity/DCI and dynamics/persistence — those carry the most formula risk. Verify against primary literature where a source is named.

Read the code and docs directly. Do not assume the prior audit is correct.

## Output contract — write to `pipelines/hydrofragments-audit/out-1a-science.md`

```markdown
# Science Audit output

## Task
{one-line restatement}

## What this stage did
{which modules + docs read; which metrics verified}

## Findings
{Per metric — a row/block each:
- Metric name + `file:line`
- Formula as coded
- Claimed source
- Verdict: CORRECT | WRONG | UNGROUNDED | EDGE-CASE-BUG
- Severity: blocker | major | minor
- Fix (if any), one line}

## Handoff to next stage
{The specific science fixes that must land before user-ready, ranked by severity — concrete enough to act on without re-reading the code}

## Open questions / risks
{Formulas you could not verify against a source; anything needing the user's domain call}
```
