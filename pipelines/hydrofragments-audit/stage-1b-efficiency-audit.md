/caveman

# Stage 1b — Efficiency Audit (HydroFragments)

## Task

Audit the HydroFragments codebase to make it user-ready. THIS stage covers the **performance axis only**: identify bottlenecks and improve efficiency where possible. A separate parallel stage handles the science/math — do not verify equations here; treat metric formulas as fixed and only judge how efficiently they compute.

HydroFragments computes surface-water fragmentation metrics over satellite image cubes. Compute path spans: `hydrofragments/compute/` (policy, chunks, backends cpu/cuda, capabilities), `hydrofragments/spatial/` (windows, zones, connectivity_context), `hydrofragments/patches/` (labels, components, morphology), `hydrofragments/temporal/` (composites, cadence, hydroyear), and `hydrofragments/pipeline.py` / `api.py`.

## What you must do

1. **Profile the hot paths by reading.** Find:
   - Python-level loops over pixels/patches/frames that should be vectorized (numpy/xarray/scipy.ndimage).
   - Needless materialization — `.compute()` / `.values` / `np.asarray` forcing a dask graph early, whole-cube loads that could stay lazy/chunked.
   - O(n²) or worse patch/connectivity ops (pairwise segment loops in DCI, label relabeling, morphology).
   - Repeated recomputation of the same intermediate (masks, labels, distance transforms).
   - Chunk-size / dask-graph misuse — bad chunk alignment, rechunking storms, per-chunk overhead.
2. **Check CPU/CUDA parity** — read `compute/backends/cpu.py` and `cuda.py`. Flag where CUDA path diverges, silently falls back, or the scaffold is incomplete. Load `docs/audit/dask_cuda_audit.md` and `dask_cuda_audit_adversarial.md` — verify or refute their findings, don't trust blindly.
3. **Check the benchmark harness** — `hydrofragments/benchmarks/cpu_baseline.py` — does it measure the real hot path? Is there a baseline to compare fixes against?
4. For each bottleneck: state location (`file:line`), estimated cost / scaling, the concrete fix, and the risk/effort of the fix.

Read the code and benchmarks directly. Do not assume the prior dask/cuda audit is correct.

## Output contract — write to `pipelines/hydrofragments-audit/out-1b-perf.md`

```markdown
# Efficiency Audit output

## Task
{one-line restatement}

## What this stage did
{which modules + benchmarks + audit docs read}

## Findings
{Per bottleneck — a block each:
- Location `file:line`
- What's slow + why (loop / materialization / O(n²) / chunk misuse / recompute)
- Cost + scaling (with N patches / frames / pixels)
- Fix, concrete
- Risk + effort: low | med | high}

## Handoff to next stage
{The bottleneck fixes that most move user-ready perf, ranked by impact/effort — concrete enough to act on without re-reading the code. Note any that conflict with a likely science change.}

## Open questions / risks
{Anything needing a real profiler run to confirm; CPU/CUDA parity gaps needing the user's call}
```
