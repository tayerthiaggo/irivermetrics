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

## Numba acceleration: the evidence gate (same rule, no hardware step)

HydroFragments also has an optional Numba JIT backend for hot Python-level
loops (`hydrofragments/metrics/clustering_numba.py`). It follows **exactly
the same evidence-gate rule as CUDA above**: a kernel is not enabled just
because Numba is installed. It is enabled only once a benchmark proves it
computes the same answer as the existing pure-Python/NumPy implementation
*and* is actually faster. The mechanics differ in one respect: Numba has no
hardware-availability probe step. A JIT compiler works on any CPU, so there
is no "is a device visible" check equivalent to CUDA's `cupy.is_available()`
+ device count probe -- the gate is pure evidence-from-file, no
probe-then-gate two-step.

### Which hot loops were considered, and which one got a kernel

The plan named two loop candidates:

1. **Per-crop EDT/width extraction**
   (`hydrofragments/patches/morphology.py::_measure_component`, the
   `include_width` branch: `skimage.morphology.medial_axis(mask,
   return_distance=True)` then `(2.0 * dist[axis]).max()`). This was
   profiled directly: across mask sizes from 16x16 to 128x128,
   `medial_axis()` itself costs **14-16 milliseconds per call**, while the
   surrounding post-processing (`(2.0 * dist[axis]).max()`, a single
   vectorized NumPy reduction, not a Python loop) costs **2-5
   *microseconds* per call** -- a ratio of roughly 3,000x to 7,500x.
   `medial_axis` is already a compiled skimage function; there is no
   Python-level loop left to accelerate around it, and reimplementing
   `medial_axis` itself in Numba is a large, separate undertaking that was
   explicitly out of scope for this task. **No Numba prototype was built
   for this candidate.** This is a profiled, evidence-based decision, not
   an oversight -- see `hydrofragments/compute/capabilities.py`'s
   `NUMBA_CANDIDATE_KERNELS` docstring for the same note in code.

2. **Inter-pool gap run-length loop**
   (`hydrofragments/metrics/clustering.py::compute_inter_pool_gaps`) -- a
   pure-Python `while` loop over a 1-D boolean array finding dry runs
   bounded by wet runs on both sides. No external library calls, a tight
   scalar loop over a NumPy array: an ideal `@njit` target. **This is the
   one kernel this task built**, candidate name `inter_pool_gap_runs`.

(A third named candidate, per-label regionprops aggregation in
`morphology.py`, was already resolved by prior work: `_bulk_major_axis_lengths`
replaced that loop with a single bulk `regionprops_table` call. No loop
remains there to accelerate.)

### The pipeline: candidate to enabled

```
NUMBA_CANDIDATE_KERNELS          benchmarks/results/numba_baseline.json    numba_enabled_kernels
(capabilities.py, hardcoded, →   (per-kernel parity_pass +              →  (evidence-gated result,
 1 kernel today, never             speedup_pass evidence)                   non-empty only if the
 auto-enabled)                                                              baseline says so)
```

1. **`NUMBA_CANDIDATE_KERNELS`** (`hydrofragments/compute/capabilities.py`)
   lists the kernel(s) eligible to ever run via Numba: today, just
   `inter_pool_gap_runs`. Being a candidate confers no execution rights, same
   as `CUDA_CANDIDATE_STAGES`.

2. **`hydrofragments/metrics/clustering_numba.py`** holds the `@njit`
   prototype (`_compute_inter_pool_gaps_numba`) alongside a pure-Python
   reference (`_pure_python_inter_pool_gaps`) it is diffed against. It
   lazily imports `numba` (mirroring
   `CUDABackend.__init__`'s `importlib.import_module("cupy")` pattern) --
   the module imports cleanly with Numba absent, and calling the kernel
   transparently falls back to the pure-Python path instead of raising
   `ImportError`. `hydrofragments/metrics/clustering.py::compute_inter_pool_gaps`,
   the certified reference implementation, is completely unmodified by this
   module -- the Numba path is an additive alternative, not a replacement.

3. **`hydrofragments/benchmarks/numba_kernels.py`** runs the Numba kernel
   against the pure-Python/NumPy baseline on fixture inputs of varied size
   and checks numeric agreement. Unlike CUDA's float-tolerance comparison,
   both candidate kernels here are integer/boolean/float-sum driven with no
   floating-point algorithm divergence between implementations, so **exact
   equality is the bar** (`max_abs_diff == 0.0`), not a tolerance. It also
   measures wall time: **warm** (the kernel already JIT-compiled, repeated
   calls, minimum of several samples) and **cold** (the very first call
   against a freshly-built, never-yet-called dispatcher, so the one-time JIT
   compilation cost is fully included) are both reported -- a kernel that is
   only faster after amortizing a compile cost over many calls is a
   different, weaker claim than one that is faster immediately, so both
   numbers are always shown rather than one being hidden behind a flag. Same
   CuPy-optional-style import safety: self-skips cleanly with `skipped: true`
   when Numba is not importable/usable, never raises.

4. This harness's raw output (`numba_kernels.json` + `.md`, named after the
   module, matching `cuda_parity.py` -> `cuda_parity.json`) is evidence
   input, not the gate file itself -- same relationship as
   `cuda_parity.json`/`cuda_transfer_cost.json` to `cuda_baseline.json`.
   `gate_evidence_from_report()` distills it into the compact per-kernel gate
   summary described next; a human should still review the report before
   that distillation is committed, same "review before it graduates"
   principle as CUDA.

5. **`benchmarks/results/numba_baseline.json`** (repo root, alongside
   `cpu_baseline.json` and the -- still absent -- `cuda_baseline.json`) is
   the file `hydrofragments.compute.capabilities.detect_capabilities` reads.
   Its schema:

   ```json
   {
     "schema_version": "1.0.0",
     "baseline": "numba_gate_evidence",
     "created_at": "2026-07-21T12:35:55.053347+00:00",
     "kernels": {
       "inter_pool_gap_runs": {
         "parity_pass": true,
         "speedup_pass": true,
         "speedup_ratio": 49.4078880029646
       }
     }
   }
   ```

   `parity_pass` and `speedup_pass` are the two required gates;
   `speedup_ratio` is optional extra context, not required for the gate
   decision itself (mirroring `crossover_size`/`net_speedup_at_max_size`'s
   role in `cuda_baseline.json`).

6. `hydrofragments.compute.capabilities.gated_kernels_from_baseline()` reads
   that file and returns exactly the `NUMBA_CANDIDATE_KERNELS` entries whose
   evidence has **both** `parity_pass: true` **and** `speedup_pass: true`.
   Missing file, malformed JSON, an empty object, or an unrecognized kernel
   name all degrade to "no evidence" (empty tuple) rather than raising --
   same never-crash contract as `gated_stages_from_baseline`.

7. `detect_capabilities()` calls `gated_kernels_from_baseline()`
   unconditionally on every call path -- including the `probe_cuda=False`
   default -- and populates `BackendCapabilities.numba_enabled_kernels` with
   the result. There is no equivalent to CUDA's "only gated inside the
   CUDA-detected branch" placement, because there is no hardware probe for
   Numba to be gated behind in the first place.

### Current status: `inter_pool_gap_runs` is evidence-graduated

Unlike the CUDA gate, this repo **does** ship
`benchmarks/results/numba_baseline.json` with real, reproducible evidence:
the Numba benchmark harness was run in this development environment (Numba
0.66.0, CPU-only, no special hardware required -- this is the key difference
from CUDA, where no GPU host has ever been available to record evidence in
this repo's history) and recorded:

- **Parity**: exact (`max_abs_diff: 0.0`) across small (64), medium (2,000),
  and large (50,000) element synthetic inputs.
- **Speedup**: consistently **40-70x** warm-call speedup over the
  pure-Python baseline at 50,000 elements across repeated benchmark runs.
  Cold-call (first-ever call, JIT compilation included) time is
  correspondingly higher (roughly 0.1-0.2 seconds for this kernel size) --
  worth knowing if this kernel is ever called on a very small input exactly
  once, but irrelevant for any workload calling it more than a handful of
  times.

As a consequence, `detect_capabilities().numba_enabled_kernels ==
("inter_pool_gap_runs",)` **by default** in this repository as shipped.

**This evidence-graduated status intentionally does *not* mean the kernel
runs in the live pipeline yet.** `hydrofragments/metrics/clustering.py::compute_inter_pool_gaps`
is not wired to consult `numba_enabled_kernels` or dispatch to
`_compute_inter_pool_gaps_numba` -- that call-site wiring was judged to be
outside this task's scope (the brief explicitly allows leaving "gate
infrastructure complete and tested... without wiring it into `analyze()`'s
hot path yet" when live-path wiring would require more than trivial
surgery). Wiring `compute_inter_pool_gaps` to honor the gate is a
straightforward follow-up: check `"inter_pool_gap_runs" in
capabilities.numba_enabled_kernels`, and if so, call
`_compute_inter_pool_gaps_numba` for the `gaps` array instead of the inline
`while` loop, before computing the same downstream summary statistics
unchanged. It was not done in this task to avoid threading a
`BackendCapabilities`/config parameter through `compute_inter_pool_gaps`'s
call sites without also updating its callers' tests, which was judged
broader than "wire an already-tested kernel behind an already-tested gate."

If a re-benchmark on different hardware or a future Numba version ever
shows regressed parity or speedup, re-running
`hydrofragments.benchmarks.numba_kernels.write_numba_benchmark` and
reviewing the new report before updating (or reverting)
`benchmarks/results/numba_baseline.json` is how that gets caught and fixed
-- the gate is only as current as the last time someone looked.

### Running the benchmark locally

```python
from hydrofragments.benchmarks.numba_kernels import write_numba_benchmark, gate_evidence_from_report
import json

report = write_numba_benchmark("benchmarks/results")  # writes numba_kernels.json/.md
evidence = gate_evidence_from_report(report)           # distill to gate shape
with open("benchmarks/results/numba_baseline.json", "w", encoding="utf-8") as f:
    json.dump(evidence, f, indent=2)
    f.write("\n")
```

A compatibility shim also exists at `benchmarks/numba_kernels.py` (repo
root), re-exporting the same functions, matching the CUDA harnesses' shim
pattern.

Without Numba installed, `run_numba_benchmark()`/`write_numba_benchmark()`
return immediately with `skipped: true` and write a report saying so -- this
is the expected Numba-absent behavior, not a failure.

### Who honors `numba_enabled_kernels`

Nothing does yet in the live pipeline -- see "Current status" above. The
field is populated and tested (`hydrofragments/compute/capabilities.py`,
`tests/compute/test_capabilities.py`'s Numba gate tests), and is available
on `BackendCapabilities` for any future caller, but no `analyze()` call site
currently reads it. This is an explicit, documented scoping decision, not an
oversight.

### Optional dependency

`numba` is an optional dependency, installed via the `accel` extra:

```
pip install -e ".[accel]"
```

Installing this extra does not, by itself, enable any kernel -- see the
evidence-gate rule at the top of this section. The package imports and all
existing tests pass without Numba installed; this is a hard CI requirement
(mirrors CUDA's CPU-only-CI-stays-green constraint) and is exercised by
`tests/metrics/test_clustering_numba.py::test_kernel_importable_and_falls_back_cleanly_without_numba`
and `tests/benchmarks/test_numba_kernels.py::test_numba_benchmark_skips_cleanly_without_numba`.
