/caveman

# Stage 2 — Reconcile + Roadmap (HydroFragments)

## Task

Audit the HydroFragments codebase to make it user-ready. Two parallel audits already ran — a science audit (equations/sources/math) and an efficiency audit (bottlenecks). THIS stage is the arbiter: merge both into ONE ranked, actionable roadmap that says what to fix, in what order, to reach user-ready. Where a science fix and a perf fix conflict (e.g. a formula correction that undoes a vectorization, or a perf rewrite that changes numerical results), you decide which wins and say why.

You have zero conversation history. Everything you need is pasted below.

## Output from Stage 1a (Science Audit) — paste `out-1a-science.md` here before running

{PASTE CONTENTS OF pipelines/hydrofragments-audit/out-1a-science.md HERE}

## Output from Stage 1b (Efficiency Audit) — paste `out-1b-perf.md` here before running

{PASTE CONTENTS OF pipelines/hydrofragments-audit/out-1b-perf.md HERE}

## What you must do

1. **Deduplicate + cross-check.** Where both audits touch the same module, reconcile. Flag any perf finding that would break a science finding (or vice versa).
2. **Rank everything by blocker → major → minor for user-ready.** Science correctness blockers outrank perf unless a perf issue makes the tool unusable at real data scale (then it's also a blocker).
3. **Resolve conflicts explicitly** — name the tradeoff, pick a winner, one line why.
4. **Produce the punch list** — an ordered fix list a developer can execute top-down, each with file:line, the fix, and why it's at that rank.
5. Note what is NOT blocking (nice-to-have) so the user can ship without it.

## Output contract — write to `pipelines/hydrofragments-audit/audit-report.md`

```markdown
# HydroFragments Audit — User-Ready Roadmap

## Verdict
{Is it user-ready today? If not, the N blockers between here and ready.}

## Blockers (must fix)
{Ordered. Each: rank, science|perf, file:line, the defect, the fix, why blocking}

## Major (should fix before wide release)
{same shape}

## Minor / nice-to-have
{same shape}

## Conflicts resolved
{Each science-vs-perf tradeoff: what conflicted, who won, why}

## Not blocking — safe to ship without
{Explicit list so the user knows what they can defer}
```
