# CUDA acceleration: the evidence gate

HydroFragments treats CPU as the certified scientific reference. An optional
CuPy/CUDA backend exists (`hydrofragments/compute/backends/cuda.py`) for a
handful of reduction stages, but **no CUDA stage runs by default, and none
will, until benchmark evidence proves both that it computes the same answer
as CPU and that it is actually faster once host&#8596;device transfer cost is
included.** This document explains how that evidence gate works end to end,
how to (re)generate the evidence, and what is enabled today (nothing).

## The pipeline: candidate to enabled

```
CUDA_CANDIDATE_STAGES            benchmarks/results/cuda_baseline.json     enabled_cuda_stages
(capabilities.py, hardcoded,  →  (per-stage parity_pass +              →   (evidence-gated result,
 5 stages, never auto-enabled)    net_speedup_pass evidence)                only non-empty if a
                                                                             baseline file says so)
```

1. **`CUDA_CANDIDATE_STAGES`** (`hydrofragments/compute/capabilities.py`)
   lists the 5 stages eligible to ever run on CUDA:
   `sentinel_normalization`, `masks`, `valid_counts`, `monthly_reduction`,
   `occurrence`. This list is a wishlist, not a promise -- being a candidate
   confers no execution rights.

2. **`hydrofragments/benchmarks/cuda_parity.py`** runs each candidate stage
   on both `CPUBackend` and `CUDABackend` against fixture cubes and checks
   numeric agreement within `FLOATING_TOLERANCES` (`float32: 1e-5`,
   `float64: 1e-12`). It records a `parity_pass: true/false` (or `null` for
   stages without a `CUDABackend` method yet) per stage, and is safe to run
   without CuPy installed -- it lazily imports CuPy and returns a clean
   `skipped: true` result instead of raising.

3. **`hydrofragments/benchmarks/cuda_transfer_cost.py`** measures wall time
   -- CPU vs CUDA, CUDA time *including* the host&#8596;device transfer --
   across a sweep of input sizes, and records the **crossover size**: the
   smallest swept size at which CUDA (with transfer cost included) beats
   CPU. It also records `net_speedup_at_max_size`. Same CuPy-optional
   import safety as the parity harness.

4. Both harnesses produce standalone evidence reports (JSON + Markdown,
   following `cpu_baseline.py`'s schema conventions: `schema_version`,
   `created_at`, per-stage `timing_seconds`-style summaries). Neither
   harness's raw output is read directly by `detect_capabilities` --
   instead, a human (or a future automation) distills both reports into
   the compact gate-evidence file described next.

5. **`benchmarks/results/cuda_baseline.json`** (repo root, alongside the
   existing `benchmarks/results/cpu_baseline.json`) is the file
   `hydrofragments.compute.capabilities.detect_capabilities` actually reads.
   Its schema is intentionally small -- one boolean pair per stage:

   ```json
   {
     "schema_version": "1.0.0",
     "baseline": "cuda_gate_evidence",
     "created_at": "2026-07-21T00:00:00+00:00",
     "stages": {
       "valid_counts": {
         "parity_pass": true,
         "net_speedup_pass": true,
         "crossover_size": 64,
         "net_speedup_at_max_size": 1.8
       },
       "monthly_reduction": {
         "parity_pass": true,
         "net_speedup_pass": false
       }
     }
   }
   ```

   `parity_pass` comes from `cuda_parity.py`'s report; `net_speedup_pass`
   is `true` only if `cuda_transfer_cost.py` found a `crossover_size` at or
   below the scale you intend to run at (i.e. CUDA nets faster than CPU
   including transfer). `crossover_size` / `net_speedup_at_max_size` are
   optional extra context carried through from the transfer-cost report,
   not required for the gate decision itself.

6. `hydrofragments.compute.capabilities.gated_stages_from_baseline()`
   reads that file and returns exactly the `CUDA_CANDIDATE_STAGES` entries
   whose evidence has **both** `parity_pass: true` **and**
   `net_speedup_pass: true`. Any of the following degrade to "no evidence"
   (empty tuple) rather than raising: the file doesn't exist, it's
   malformed JSON, it's an empty object, or a stage name isn't recognized.

7. `detect_capabilities(probe_cuda=True)` calls this only inside its
   CUDA-detected branch (after CuPy imports and the runtime smoke test
   pass) and populates `BackendCapabilities.enabled_cuda_stages` with the
   result. **If no baseline file exists, `enabled_cuda_stages` stays `()`
   -- this is the default-safe behavior and must never regress.**

## Who honors `enabled_cuda_stages`

Two separate mechanisms consume this evidence -- know which is which:

- **`hydrofragments.compute.capabilities.resolve_execution_plan`** -- the
  evidence-gated per-stage planner actually used by the live `analyze()`
  path (`hydrofragments/api.py`). It already falls back to CPU per-stage
  whenever a stage isn't in `enabled_cuda_stages`; this was true before
  this document existed and is unchanged by it.

- **`hydrofragments.compute.policy.ComputePolicy`** -- a separate, stricter
  execution-lane object used by `assemble_monthly_pipeline`
  (`hydrofragments/pipeline.py`) and `hydrofragments/benchmarks/cpu_baseline.py`.
  Historically this raised `ComputePolicyError` on *any*
  `accelerator="cuda"` request, unconditionally. It now accepts an optional
  `capabilities: BackendCapabilities | None = None` constructor parameter:
  - Omitted (every caller today) &#8594; resolves to a safe "no CUDA stages
    enabled" default and `accelerator="cuda"` is refused exactly as before,
    same error message ("CUDA execution is not certified for the Milestone
    4 pipeline"). **No existing caller's behavior changes.**
  - Passed with a `capabilities` whose `enabled_cuda_stages` is non-empty
    &#8594; `accelerator="cuda"` is accepted. `ComputePolicy` does not track
    stages individually (it is a single execution lane, not a per-stage
    planner like `resolve_execution_plan`), so it gates on "at least one
    CUDA stage has evidence" rather than a specific stage list; per-stage
    enforcement for the actual pipeline still runs through
    `resolve_execution_plan`.

## Running the benchmarks locally

Both harnesses are function-only modules (no `__main__` block), matching
`hydrofragments/benchmarks/cpu_baseline.py`'s existing convention. Invoke
them from Python directly:

```python
from hydrofragments.benchmarks.cuda_parity import write_cuda_parity
from hydrofragments.benchmarks.cuda_transfer_cost import write_cuda_transfer_cost

write_cuda_parity("benchmarks/results")
write_cuda_transfer_cost("benchmarks/results")
```

or, matching `docs/performance.md`'s one-liner style:

```powershell
python -c "from hydrofragments.benchmarks.cuda_parity import write_cuda_parity; write_cuda_parity('benchmarks/results')"
python -c "from hydrofragments.benchmarks.cuda_transfer_cost import write_cuda_transfer_cost; write_cuda_transfer_cost('benchmarks/results')"
```

Compatibility shims also exist at `benchmarks/cuda_parity.py` and
`benchmarks/cuda_transfer_cost.py` (repo root), re-exporting the same
functions, matching `benchmarks/cpu_baseline.py`'s existing shim pattern.

Without CuPy installed, both calls return immediately with
`skipped: true` and write a report saying so -- this is the expected
CPU-only-CI behavior, not a failure. With CuPy and a real GPU, they run the
actual parity/timing sweeps and write the full reports described above.

Producing `benchmarks/results/cuda_baseline.json` itself (the compact gate
file `detect_capabilities` reads) is a manual/curatorial step today: read
the `parity_pass` per stage out of `cuda_parity.json` and the
crossover/speedup evidence out of `cuda_transfer_cost.json`, and write the
compact per-stage `{parity_pass, net_speedup_pass}` summary shown above.
This is intentional -- graduating a stage to run on GPU by default should
require a human to look at both reports and make the call, not an
unattended script.

## Current status: nothing is enabled

No `benchmarks/results/cuda_baseline.json` ships in this repository. No
GPU host has recorded parity + net-speedup evidence for any
`CUDA_CANDIDATE_STAGES` entry. As a direct consequence:

- `detect_capabilities(probe_cuda=True)` always returns
  `enabled_cuda_stages=()` on this codebase as shipped, even on a machine
  with a working GPU and CuPy installed -- the CUDA runtime probe passing
  is necessary but not sufficient.
- `resolve_execution_plan` always falls back every stage to CPU.
- `ComputePolicy(accelerator="cuda")` (no `capabilities` argument) always
  raises `ComputePolicyError`.

**This is expected, not a bug.** It will remain true until someone runs
the benchmarks above on real CUDA hardware, evaluates the parity and
transfer-cost evidence, and commits a `benchmarks/results/cuda_baseline.json`
that shows a stage passing both gates.

## CI

`.github/workflows/ci.yml`'s CPU-only job runs `cuda_parity.py`'s harness
on every push/PR and asserts it returns a clean `skipped: true` result --
this exercises the "CuPy unavailable, degrade gracefully" code path, not
real GPU parity evidence (there is no GPU runner in this repo's CI matrix).
A separate `cuda-benchmarks` job describes the shape a real GPU-runner job
would take (`runs-on: [self-hosted, cuda]`) but is gated `if: false` since
no such runner exists yet; it is inert and does not affect CI status.
