# HydroFragments User-Ready Audit — Pipeline

Run top to bottom. Stages 1a and 1b are independent — run in either order or parallel. Stage 2 needs both outputs pasted in.

All stage prompts open with `/caveman` to hold each model to terse, high-signal output.

---

## Stage 1a — Science Audit

Verify equations, sources, math, edge cases. Prompt: [`stage-1a-science-audit.md`](stage-1a-science-audit.md). Output → `out-1a-science.md`.

**Model (ranked):**
1. **Claude Fable 5 (high)** — hardest reasoning; science is highest-stakes axis. `claude --model claude-fable-5` (set effort high via `/model`)
2. **Claude Opus 4.8 (xhigh)** — cheaper, still strong; use if Fable budget tight. `claude --model claude-opus-4-8`

Run: `cat stage-1a-science-audit.md` → paste into the chosen CLI. Save reply to `out-1a-science.md`.

---

## Stage 1b — Efficiency Audit

Find bottlenecks, CPU/CUDA parity, chunk/dask misuse. Prompt: [`stage-1b-efficiency-audit.md`](stage-1b-efficiency-audit.md). Output → `out-1b-perf.md`.

**Model (ranked) — cross-family from 1a on purpose:**
1. **GPT-5.5 (high)** — best on infra/shell/dask-heavy reasoning. `codex --model gpt-5.5 -e high "$(cat stage-1b-efficiency-audit.md)"`
2. **Claude Opus 4.8 (high)** — use if fixes turn out algorithmic not infra. `claude --model claude-opus-4-8`

Run: paste the prompt into the chosen CLI. Save reply to `out-1b-perf.md`.

---

## Stage 2 — Reconcile + Roadmap

Merge both audits into one ranked punch list; resolve science-vs-perf conflicts. Prompt: [`stage-2-reconcile.md`](stage-2-reconcile.md). Output → `audit-report.md`.

**Before running:** open `stage-2-reconcile.md`, paste `out-1a-science.md` and `out-1b-perf.md` into the two marked placeholder sections.

**Model (ranked):**
1. **Claude Opus 4.8 (xhigh)** — standard arbiter; wrong verdict propagates. `claude --model claude-opus-4-8`
2. **Claude Fable 5 (high)** — use if science findings dominate the conflict. `claude --model claude-fable-5`

Run: `cat` the edited `stage-2-reconcile.md` → paste into the chosen CLI. Save reply to `audit-report.md`.

---

## Checklist

- [ ] Stage 1a → `out-1a-science.md`
- [ ] Stage 1b → `out-1b-perf.md`
- [ ] Paste both into `stage-2-reconcile.md`
- [ ] Stage 2 → `audit-report.md`
