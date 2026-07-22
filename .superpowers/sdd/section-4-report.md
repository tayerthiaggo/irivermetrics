# Section 4 report — Numba prototyping (benchmark-gated)

## What was built

Section 4 mirrors Section 3's CUDA evidence-gate pattern for Numba JIT
kernels: nothing is enabled by default, a kernel only activates once a
benchmark proves it is both numerically identical to the existing
pure-Python/NumPy implementation and actually faster.

### 1. Capability gate infrastructure (`hydrofragments/compute/capabilities.py`)

- `NUMBA_CANDIDATE_KERNELS: tuple[str, ...] = ("inter_pool_gap_runs",)` —
  twin of `CUDA_CANDIDATE_STAGES`.
- `BackendCapabilities.numba_enabled_kernels: tuple[str, ...] = ()` — new
  field, empty by default, mirroring `enabled_cuda_stages`. Included in
  `to_mapping()`.
- `_DEFAULT_NUMBA_BASELINE_PATH` — points at
  `benchmarks/results/numba_baseline.json` (repo root), twin of
  `_DEFAULT_BASELINE_PATH`.
- `gated_kernels_from_baseline(baseline_path=None) -> tuple[str, ...]` —
  twin of `gated_stages_from_baseline`. Same never-crash contract: missing
  file, malformed JSON, empty object, or an unrecognized kernel name all
  degrade to `()` rather than raising.
- Wired into `detect_capabilities()`: unlike CUDA, Numba has no
  hardware-probe step (a JIT compiler works on any CPU — there is no device
  to detect), so `gated_kernels_from_baseline()` is called unconditionally
  at the top of `detect_capabilities()` and threaded through **every**
  return path in that function (including the `probe_cuda=False` default
  path, and all four early-return branches in the CUDA-probing logic), not
  gated behind a CUDA-detected branch the way `enabled_cuda_stages` is via
  `_resolve_cuda_capabilities_from_probe`.

### 2. The Numba kernel decision

Two candidates were named in the brief. One was built, one was rejected
with evidence:

- **Rejected: per-crop EDT/width extraction**
  (`hydrofragments/patches/morphology.py::_measure_component`). Profiled
  directly (`skimage.morphology.medial_axis` vs the surrounding
  `(2.0 * dist[axis]).max()` post-processing) across mask sizes 16×16
  through 128×128:

  | Size | `medial_axis()` per call | post-processing per call |
  |---|---:|---:|
  | 16×16 | 14.9 ms | 2.0 µs |
  | 32×32 | 15.6 ms | 2.3 µs |
  | 64×64 | 16.5 ms | 2.9 µs |
  | 128×128 | 14.6 ms | 5.1 µs |

  `medial_axis` dominates wall time by roughly 3,000×–7,500×. The
  surrounding code is already a single vectorized NumPy reduction, not a
  Python loop — there is nothing Numba-shaped left to accelerate without
  reimplementing `medial_axis` itself, which the brief explicitly placed
  out of scope. **No Numba prototype was built for this candidate.** This
  is documented in `NUMBA_CANDIDATE_KERNELS`'s docstring in
  `capabilities.py` and in `docs/acceleration.md`.

- **Built: inter-pool-gap run-length loop**
  (`hydrofragments/metrics/clustering.py::compute_inter_pool_gaps`) — a
  pure-Python `while` loop over a 1-D boolean array, no external library
  calls, a textbook `@njit` target. Candidate name: `inter_pool_gap_runs`.

- **Already resolved (per the brief, not re-checked from scratch but
  confirmed by reading the referenced docstring)**: the third named
  candidate, per-label regionprops aggregation in `morphology.py`, was
  already replaced by a bulk `regionprops_table` call in prior work
  (`_bulk_major_axis_lengths`, `morphology.py:44-76`). No loop remains
  there.

### 3. The kernel itself (`hydrofragments/metrics/clustering_numba.py`, new file)

- `_pure_python_inter_pool_gaps(wet, lengths) -> np.ndarray` — a
  structurally-identical free-function reference implementation (same
  while-loop shape as `compute_inter_pool_gaps`'s internals), used both as
  the Numba-absent fallback and as the parity baseline.
- `_build_numba_kernel() -> Callable | None` — lazily imports `numba`
  (mirrors `CUDABackend.__init__`'s `importlib.import_module("cupy")`
  pattern exactly) and constructs a fresh `@njit`-decorated dispatcher.
  Returns `None` on any import or compile failure — never raises.
- `numba_available() -> bool`, `_ensure_kernel()` — module-level
  build-once caching (`_NUMBA_KERNEL`, `_NUMBA_IMPORT_ATTEMPTED` globals),
  same shape as other lazy-import caches in this codebase.
- `_compute_inter_pool_gaps_numba(wet, lengths) -> np.ndarray` — the public
  entry point. Falls back transparently to
  `_pure_python_inter_pool_gaps` if Numba is unavailable. **Never raises
  ImportError to the caller.**
- `hydrofragments/metrics/clustering.py` is **completely unmodified** —
  `compute_inter_pool_gaps`'s existing while-loop and all its numeric
  behavior are untouched, per the brief's explicit constraint.

### 4. Benchmark harness (`hydrofragments/benchmarks/numba_kernels.py`, new file)

Twin of `cuda_parity.py` + `cuda_transfer_cost.py` combined into one
harness, since a same-process `@njit` call has no "transfer cost" concept
to benchmark separately (per the brief's simplification note).

- `run_numba_benchmark(cases=DEFAULT_CASES, repeats=3, kernel_override=None)`
  — runs parity (exact equality, `max_abs_diff == 0.0` — both candidate
  kernels are boolean/float-sum driven with no cross-implementation
  floating-point divergence, so exact equality is the correct bar per the
  brief, not `FLOATING_TOLERANCES`) and speedup checks. Self-skips cleanly
  (`skipped: true`) when Numba is unavailable — same CuPy-optional-style
  import safety, verified not to raise.
- **Cold vs warm timing**: reports both, and a real bug was caught and
  fixed during development — the initial "cold" measurement accidentally
  reused the same cached kernel object the parity-checking loop had
  already warmed up, so it was silently measuring a warm call. Fixed by
  adding a `cold_probe` parameter that calls `_build_numba_kernel()`
  directly to get a brand-new, never-yet-called dispatcher for the cold
  measurement, fully independent of the module-level cache used for
  parity/warm timings. Verified the fix: cold time went from a suspicious
  ~0.16ms to a believable ~130ms (JIT compilation genuinely costs
  something), while warm/parity numbers were unaffected.
- `gate_evidence_from_report(payload) -> dict` — distills the raw
  per-case/per-kernel-detail report (`kernels` as a *list*, matching the
  CUDA harnesses' `cases`-list convention) into the compact
  `{"kernels": {name: {"parity_pass", "speedup_pass"}}}` dict shape
  `gated_kernels_from_baseline()` actually reads — analogous to the human
  curatorial step `docs/acceleration.md` describes for `cuda_baseline.json`,
  offered here as a helper function since there's no second report
  (transfer-cost) to cross-reference.
- `write_numba_benchmark(output_dir)` — writes the raw report to
  `numba_kernels.json` / `.md` (named after this module, matching
  `cuda_parity.py` → `cuda_parity.json`; deliberately *not*
  `numba_baseline.json`, to avoid colliding with the gate file's name —
  this was a real naming bug caught mid-development, see Design decisions
  below).
- Repo-root compatibility shim added at `benchmarks/numba_kernels.py`,
  matching `benchmarks/cuda_parity.py`'s shim pattern.

### 5. Real benchmark evidence, committed

The harness was run in this development environment (Numba 0.66.0,
Python 3.14.5, Windows, CPU-only — no special hardware, unlike CUDA):

- **Parity**: exact (`max_abs_diff: 0.0`) across small (64), medium
  (2,000), and large (50,000) element synthetic inputs, every run.
- **Speedup**: consistently **40×–70×** warm-call speedup at 50,000
  elements across five separate runs (measured: 69.7×, 53.0×, 55.5×,
  39.9×, 49.4×). Cold-call (JIT-compile-included) time is correspondingly
  higher (~0.13–0.39 s at this size).

Unlike CUDA (where no GPU host has ever been available in this repo's
history, so `cuda_baseline.json` was never committed),
**`benchmarks/results/numba_baseline.json` is committed with real,
reproducible evidence**, since Numba requires no special hardware. As a
direct consequence, `detect_capabilities().numba_enabled_kernels ==
("inter_pool_gap_runs",)` by default in this repository as shipped. This
is documented explicitly and prominently in `docs/acceleration.md`'s
"Current status" section, including the honest caveat that graduation is
current only as of the last time someone re-ran the benchmark.

Also committed: `benchmarks/results/numba_kernels.json` / `.md` (the raw
report this evidence was distilled from).

### 6. Live-path wiring — explicit scoping decision: NOT done

Per the brief's explicit latitude ("this task does not require wiring the
enabled-kernel flag into the live call sites' dispatch logic if doing so
would require broader pipeline changes outside this task's scope"):
`compute_inter_pool_gaps` does **not** consult `numba_enabled_kernels` or
dispatch to `_compute_inter_pool_gaps_numba`. This was judged to require
threading a `BackendCapabilities`/config parameter through
`compute_inter_pool_gaps`'s call sites and updating their tests — broader
surgery than "wire an already-tested kernel behind an already-tested
gate." The gate infrastructure is complete, tested, and evidence-graduated;
the live pipeline still always uses the certified pure-Python
implementation. This is documented explicitly (not left ambiguous) in
`docs/acceleration.md`'s "Current status" section, including a concrete
sketch of what the follow-up wiring would look like.

### 7. Docs (`docs/acceleration.md`)

Added a full `## Numba acceleration` section (parallel to the existing
`# CUDA acceleration` H1, since the doc's structure treats CUDA as the
document's title-level topic and Numba as a sibling top-level section) with
the same sub-structure as the CUDA section: which loops were considered and
why one was rejected (with the real profiling table), the
candidate→evidence→enabled pipeline, current status (evidence-graduated,
but not live-wired), running the benchmark locally, who honors the gate
(nobody yet, by design), and the optional `accel` extra.

### 8. `pyproject.toml`

Added `accel = ["numba>=0.59"]` to `[project.optional-dependencies]`,
matching the brief's suggested pin and current PyPI availability (latest
is 0.66.0, which is what's installed in this dev environment).

### 9. CI (`.github/workflows/ci.yml`)

Added a "Run Numba benchmark harness (Numba-absent self-skip path)" step,
mirroring the existing CUDA parity self-skip step — the `accel` extra is
not installed by the `test` extra CI uses, so this exercises the
Numba-unavailable → clean-skip path on every push/PR. Verified the
underlying self-skip logic directly (not just trusting the CI YAML) by
simulating `numba` import failure via `importlib.import_module`
monkeypatching, matching the technique the actual pytest tests use.
Note: `.github/` is covered by a stray `.gitignore` rule in this worktree,
but `ci.yml` is already git-tracked (added via `-f` in Section 3), so this
commit followed the same `git add -f` precedent.

## Test results

Command used (per the brief's junitxml workaround for the `-q` summary
quirk):

```
python -m pytest tests/ -m "not slow" --junitxml=<path>
```

Final full-suite run:

```
<testsuite name="pytest" errors="0" failures="1" skipped="3" tests="575" .../>
```

**571 passed, 3 skipped (intentional), 1 failed, 575 total.**

The 1 failure is `tests/contracts/test_fixture_characterisation.py::test_fitzroy_zarr_exists_and_checksum_is_stable`
— confirmed **pre-existing and unrelated** to this task: reproduced via
`git stash` (temporarily removing all Section 4 changes) and re-running
just that test, which failed identically before any of this section's code
existed. Not touched or investigated further, as it is outside this
section's scope (a fixture checksum drift, likely an environment/package
version difference, not a Section 4 regression).

Section-4-specific test files, isolated:

```
tests/compute/test_capabilities.py   (Numba gate tests only, -k numba)  3 passed
tests/metrics/test_clustering_numba.py                                 15 passed, 1 skipped
tests/benchmarks/test_numba_kernels.py                                 6 passed
```

Combined Section-4-relevant run
(`tests/benchmarks/ tests/metrics/test_clustering_numba.py tests/compute/`):

```
<testsuite name="pytest" errors="0" failures="0" skipped="1" tests="70" .../>
```

All new tests pass. The 1 skip is intentional
(`test_numba_kernel_matches_public_api_gaps[empty]` — skipped because
`compute_inter_pool_gaps` itself doesn't accept empty input in practice;
the empty-array case is still covered directly against the hand-rolled
reference in `test_numba_kernel_matches_pure_python_reference[empty]`,
which passes).

## Design decisions where the brief left room for judgment

1. **EDT/width kernel: skipped, with evidence.** The brief explicitly
   invited this outcome ("a single well-justified kernel... is an
   acceptable, honest outcome"). Profiling gave an unambiguous answer
   (3,000×–7,500× dominance by the compiled `medial_axis` call), so no
   kernel was built for that candidate. See the table above.

2. **Exact equality, not tolerance, for parity.** Both `wet` (boolean) and
   `lengths` (float64, but the parity operation is pure summation with no
   algorithmic divergence between the pure-Python and `@njit` versions —
   same operation order, same accumulation) inputs produce bit-identical
   results in every test run. The brief anticipated this
   ("only fall back to tolerance-based comparison if you find a genuine
   floating-point path and justify it") — no such path was found, so
   `FLOATING_TOLERANCES` is not used by this harness.

3. **Numba gating placement in `detect_capabilities()`.** Because Numba
   has no hardware-probe step, `gated_kernels_from_baseline()` is called
   unconditionally at the top of `detect_capabilities()`, independent of
   the `probe_cuda` flag and independent of which CUDA branch is reached.
   This required touching every `BackendCapabilities(...)` construction
   site inside `detect_capabilities()` (5 return points) to thread
   `numba_enabled_kernels` through, rather than a single seam like
   `_resolve_cuda_capabilities_from_probe`. This is the most structurally
   different piece from the CUDA twin, and is called out explicitly in
   both the code comments and `docs/acceleration.md`.

4. **Committing `benchmarks/results/numba_baseline.json` as real,
   evidence-graduated (not empty).** This is the biggest divergence from
   the CUDA precedent, where the gate file was never committed because no
   GPU host was ever available. Numba has no such hardware dependency —
   the benchmark ran, repeatably, in this very development environment, on
   plain CPU. Given the whole point of the evidence-gate mechanism is "graduate once you have real proof," and real, reproducible proof was in
   hand (5 separate runs, consistent 40×–70× speedup, exact parity every
   time), shipping the gate closed anyway would have been dishonest
   theater rather than honoring the mechanism's actual purpose. This is
   flagged prominently in `docs/acceleration.md`'s "Current status"
   section so nobody mistakes evidence-graduated for live-pipeline-wired
   (see point 6 above — those are two different things, and the doc says
   so explicitly).

5. **Raw report filename vs gate filename collision, caught and fixed.**
   Initially `write_numba_benchmark(output_dir)` wrote its raw report to
   `<output_dir>/numba_baseline.json` — the *same basename* as the gate
   file `_DEFAULT_NUMBA_BASELINE_PATH` points at. Since a natural workflow
   is `write_numba_benchmark("benchmarks/results")`, this would have
   silently produced a `benchmarks/results/numba_baseline.json` in the
   wrong (raw-report, list-shaped) schema, which `gated_kernels_from_baseline`
   would then silently fail to parse (a list where a dict is expected —
   `isinstance(kernels, dict)` check returns `False`, degrading to "no
   evidence" rather than crashing, but silently wrong). Caught this by
   actually inspecting the generated JSON's shape rather than trusting the
   docstring I'd written. Fixed by renaming the raw-report output to
   `numba_kernels.json` / `.md` (matching the module name, like
   `cuda_parity.py` → `cuda_parity.json`), freeing `numba_baseline.json`
   to be exclusively the gate file, mirroring `cuda_baseline.json`'s role
   exactly. Added `test_gate_evidence_from_report_is_readable_by_the_capability_gate`
   as an end-to-end regression test for this exact failure mode.

6. **Cold-timing measurement bug, caught and fixed.** See item 4 under
   "What was built" above — the first version of the speedup benchmark
   silently measured a warm call as "cold" because Numba caches compiled
   dispatchers on the function object itself, and the parity-checking loop
   had already called that same object. Caught by manually inspecting the
   actual numbers (~0.16ms for a "cold, JIT-compile-included" call looked
   too fast to believe) rather than accepting a passing test at face
   value. Fixed with a `cold_probe` seam that builds a genuinely
   never-called dispatcher via `_build_numba_kernel()` for the cold
   measurement only.

## Self-review

- **Structural fidelity to Section 3**: `NUMBA_CANDIDATE_KERNELS` /
  `numba_enabled_kernels` / `gated_kernels_from_baseline` are direct
  structural twins of `CUDA_CANDIDATE_STAGES` / `enabled_cuda_stages` /
  `gated_stages_from_baseline` — same never-crash-on-bad-input contract,
  same "evidence file absent ⇒ empty tuple" default, same lazy-import
  pattern for the optional dependency, same benchmark-harness-writes-JSON-
  and-Markdown convention. Verified by re-reading both side by side while
  writing tests, not just by memory of having read Section 3 once.
- **No modification to certified reference implementations**: verified
  `git diff` shows `hydrofragments/metrics/clustering.py` and
  `hydrofragments/patches/morphology.py` are untouched by this section (I
  did not edit either file at all — only read them for profiling and
  reference).
- **Import safety without Numba**: verified directly, not just via mocked
  tests — ran `python -c "import hydrofragments.metrics.clustering_numba as m; assert 'numba' not in sys.modules"`
  and confirmed the module import itself never triggers a Numba import;
  only calling `numba_available()` or the kernel does.
- **A genuine risk I want to flag rather than bury**: this environment
  happens to have Numba 0.66.0 pre-installed and working on Python 3.14.5
  (a very new combination). The committed benchmark evidence was recorded
  here. If a different CI/production environment's Numba version or
  Python version behaves differently for this specific kernel (unlikely,
  given how simple the kernel is — a scalar loop with no advanced NumPy
  features — but not something I can rule out without testing on other
  Python/Numba versions), the evidence in `benchmarks/results/numba_baseline.json`
  would be stale for that environment. This is exactly the scenario
  `docs/acceleration.md`'s closing paragraph in the Numba section
  addresses ("re-running... and reviewing the new report... is how that
  gets caught and fixed"), but it's worth a human's attention before any
  future work builds on this kernel being enabled by default.
- **What I did not do, deliberately**: did not wire the gate into
  `compute_inter_pool_gaps`'s call path (scoping decision, documented);
  did not attempt an EDT/width Numba prototype (profiling-based rejection,
  documented); did not add `numba` to CI's `test` extra (would defeat the
  point of the CI self-skip check, which needs Numba genuinely absent to
  be meaningful).

## Status

DONE

## Commits (first..last)

```
f064f02 feat: add Numba evidence-gate fields to BackendCapabilities
d00f92c feat: add @njit prototype for inter-pool-gap run-length loop
26e9a59 feat: add Numba benchmark harness; graduate inter_pool_gap_runs kernel
3d3e4c7 docs: add Numba evidence-gate section to acceleration.md
f22b3de ci: run Numba benchmark self-skip check
```

## One-line test summary

571 passed, 3 skipped, 1 pre-existing unrelated failure (575 total, `pytest -m "not slow" --junitxml=...`); all Section 4 tests pass (70/70, 1 intentional skip, isolated run).

## Concerns

- Evidence in `benchmarks/results/numba_baseline.json` was recorded on
  Numba 0.66.0 / Python 3.14.5 / Windows in this dev environment only —
  not cross-validated on another OS/Python/Numba combination. The kernel
  is simple enough that this is a low risk, but it's the one place this
  section made a judgment call beyond what Section 3 needed to make (Section 3 never got to commit real evidence at all, for lack of GPU
  hardware).
- Live-pipeline wiring (`compute_inter_pool_gaps` actually calling the
  Numba kernel when enabled) is explicitly not done — see scoping decision
  above. Anyone picking this up next has a documented, concrete sketch of
  what that follow-up looks like in `docs/acceleration.md`.
