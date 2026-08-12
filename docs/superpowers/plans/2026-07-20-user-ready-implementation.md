# HydroFragments — User-Ready Implementation Plan (2026-07-20)

Derived from repo audit against 6 readiness items. Scope = wiring + tests + docs. No metric-math changes.

Decisions locked:
- **DEA path:** consume WaterMask-TSFill output only. No `odc.stac`/`pystac` dep added here. TSFill stays the single DEA+gapfill entry point.
- **Numba:** prototype on hot loops, benchmark-gated (same evidence-gate pattern as CUDA).
- **Gapfilling:** NOT implemented in this repo. Exposed as a workflow config flag (`gapfill: bool`, default `False` for efficiency) — the user declares whether their input is already gapfilled; HydroFragments never gapfills itself. It only detects+flags baseline quality and, when insufficient and `gapfill=False`, prescribes (recommends) running WaterMask-TSFill.
- **Input structure:** no manual user input describing their data layout. HydroFragments inspects the input against the documented §14 contract and either (a) auto-normalizes safe structural mismatches (variable naming, dtype/domain coercion, dim order), or (b) raises a specific, actionable error naming the exact mismatch — never silently resamples/reprojects to fix a grid or CRS mismatch, per the existing locked spec guard (§14: "must raise an explicit error, not silently resample").

---

## Section 1 — Generic binary mask adapter + DEA/TSFill handoff

**State:** `io/adapters.py` has one parser (`parse_watermask_tsfill`, hardcoded 0/1/254/255). `api.open_water_cube` has an `input_kind` param w/ a thin `generic_binary` branch, but `.zarr` path *always* assumes TSFill; `variable_map` reserved-unused; no adapter registry; no §14 input-contract validation.

**No manual "describe your data" step.** The pipeline inspects the input and either auto-normalizes what's safe or raises a specific, actionable error. `input_kind` becomes an optional override/hint, not a required user declaration.

**Build:**
1. `io/adapters.py` → adapter registry keyed by `input_kind`, plus **auto-detection** (variable names present, dtype/value domain, band names) that picks the right adapter when `input_kind` is omitted:
   - `watermask_tsfill` (existing, refactor into registry) — detected via the 0/1/254/255 uint8 signature.
   - `generic_binary` — `{0,1}`/bool + explicit nodata + optional paired `valid_obs` layer; honor `variable_map`.
   - `raw_wofs` — WOfS `water`/`frequency` band naming + WOfS nodata convention → binary via `water_threshold`; detected via band names.
2. `io/input_contract.py` — implements the inspect-then-act rule:
   - **Auto-fix (safe, structural only):** rename a single unambiguous data variable to the expected name, coerce `{0,1}` int/float → bool, reorder dims to the expected order. Every auto-fix is logged to run provenance so it's never silent.
   - **Raise (never auto-fix):** grid/transform mismatch between water and valid-obs layers, undefined or degrees CRS, ambiguous multi-variable Dataset with no `water`-like candidate. Error message names the exact field and expected vs actual value — actionable without reading source.
   - Also records `water_threshold`/`threshold_method`/`probability_source` for probabilistic input per §14.
3. `open_water_cube`: drop the `.zarr ⇒ tsfill` assumption; run auto-detection → contract check → adapter; wire `variable_map`; populate `source`/`provenance` (including any auto-fixes applied) from the resolved adapter.
4. `config.InputConfig`: `input_kind` becomes optional (`None` = auto-detect); keep as an override for ambiguous cases. Add `variable_map`.

**Tests:** per-adapter round-trip; auto-detection picks correct adapter per fixture; grid-mismatch raises with actionable message; degrees-CRS refusal; safe auto-fixes applied + logged; TSFill parity vs current output (no regression); raw-WOfS threshold path.

**Docs:** `docs/input_format.md` (§14 standalone, checklist item 27) — the documented contract auto-detection checks against, a section per adapter, which mismatches auto-fix vs raise, and the TSFill handoff diagram (TSFill exports canonical uint8 cube → HydroFragments `watermask_tsfill` adapter).

---

## Section 2 — Baseline quality analysis → gapfill prescription (no gapfilling here)

**State:** `min_valid_obs` / `min_valid_fraction_month` guards flag low coverage (`edge_flag=low_valid_obs`) but nothing *analyzes baseline quality up front* or *recommends* remediation. Gapfill code lives only in legacy `ecofragments/utils/calc_metrics.py` — not ported, and per decision stays out.

**Build:**
1. `guards/quality.py` — `assess_baseline_quality(cube, config)` → report: per-pixel valid-obs coverage, per-month valid fraction, seasonal MNAR gap summary (reuse the season-stratified logic already in `metrics/persistence.py`), fraction of AOI/months below floors.
2. Prescription logic: when coverage is below configurable thresholds and `config.gapfill is False`, emit a structured **recommendation** (`recommend_gapfill=True`, with reason + pointer to WaterMask-TSFill) — surfaced in `ValidationReport.warnings` and run manifest. Never mutates data, never gapfills itself.
3. `config`: new top-level workflow field **`gapfill: bool = False`** — explicit user switch, defaults off for efficiency (matches current behavior). `True` means "I've already gapfilled upstream" and suppresses the recommendation (HydroFragments trusts the declaration, doesn't re-verify). Recorded in provenance/manifest either way so it's auditable per run.

**Tests:** `gapfill=False` + low-coverage cube → recommendation fires + reason correct; `gapfill=True` → recommendation suppressed regardless of coverage; MNAR summary numbers hand-checked on a fixture; config default confirmed `False`.

**Docs:** manager guide + input_format note — "HydroFragments does not gapfill. Set `gapfill: true` in config once your input is pre-filled (e.g. via WaterMask-TSFill); leave `false` (default) to run on raw data and get quality flags instead."

---

## Section 3 — CUDA benchmark suite (unlock the evidence gate)

**State:** `compute/backends/cuda.py` (CuPy reductions), `capabilities.py` gates stages on benchmark evidence — `enabled_cuda_stages` deliberately empty ("no stage has transfer-cost benefit evidence"). `policy.py` hard-refuses `accelerator="cuda"`. `CUDA_CANDIDATE_STAGES` = sentinel_normalization, masks, valid_counts, monthly_reduction, occurrence.

**Build:**
1. `benchmarks/cuda_parity.py` — for each candidate stage: CPU vs CUDA numeric parity within `FLOATING_TOLERANCES`; assert equality on fixture cubes of varied size.
2. `benchmarks/cuda_transfer_cost.py` — wall-time incl. host↔device transfer, CPU vs CUDA, across sizes → find crossover where GPU wins net of transfer.
3. Results → `benchmarks/results/` as machine-readable baseline (JSON) that `capabilities.detect_capabilities` can read to populate `enabled_cuda_stages` per stage that *proved* benefit.
4. Wire: a stage graduates from `CUDA_CANDIDATE_STAGES` to enabled only when its baseline JSON shows parity-pass + net speedup. `policy.py` relaxes the blanket `cuda` refusal → gated by enabled stages.
5. CI: run parity (correctness) always; transfer-cost only on GPU runners (skip-if-no-CUDA), guarded so CPU-only CI stays green.

**Tests:** parity harness self-test on CPU (CuPy mocked/absent path); gate logic — empty baseline ⇒ still CPU; populated baseline ⇒ stage enabled.

**Docs:** `docs/acceleration.md` — how the evidence gate works, how to run benchmarks, current enabled stages.

---

## Section 4 — Numba prototyping (benchmark-gated)

**State:** zero Numba references repo-wide.

**Build:**
1. Add `numba` as optional dep (`[accel]` extra).
2. Identify hot Python-loop kernels (not already vectorized/pylandstats): per-pool EDT width loop (`metrics/morphology.py`), inter-pool gap run-length (`metrics/clustering.py`), any per-label regionprops aggregation loop.
3. `@njit` prototypes behind a capability flag mirroring CUDA: a `numba_enabled_kernels` gate, empty until benchmark proves speedup.
4. `benchmarks/numba_kernels.py` — Numba vs pure-numpy/vectorized baseline per kernel; parity + speedup.
5. Enable per-kernel only where benchmark wins; fall back to current impl otherwise.

**Tests:** parity Numba-vs-baseline per kernel; import works without Numba installed (fallback path).

**Docs:** fold into `docs/acceleration.md`.

---

## Section 5 — Notebooks (plain-language, step-by-step) + optional CLI

**State:** `examples/irm_example.ipynb`, `examples/STAC_query.ipynb` — legacy iRivermetrics, not wired to current `hydrofragments` API (one is malformed JSON). No plain-language new-user walkthrough. No CLI anywhere in `hydrofragments/`.

**Build:**
1. `examples/01_quickstart.ipynb` — plain-language, new-user: load a bundled small fixture cube → `open_water_cube` → `analyze` → read tidy table → one plot. Markdown-heavy, minimal jargon, runs in <2 min. (checklist item 26 / §12.2)
2. `examples/02_dea_via_tsfill.ipynb` — the real DEA workflow: point at a WaterMask-TSFill output, run HydroFragments. Documents the handoff, not STAC internals.
3. `examples/03_metrics_walkthrough.ipynb` — the 4 metric groups (see below), one section each, interpreting output.
4. Retire/fix legacy notebooks (move to `examples/legacy/` or delete the broken one — confirm before deleting).
5. **CLI** (`hydrofragments/cli.py`, `argparse` or `typer`, `[cli]` extra): `hydrofragments analyze --config cfg.yaml --input ... --aoi ... --out ...` mirroring `api.analyze`. Entry point in `pyproject.toml`. "Notebook for learning, CLI for efficiency."

**Tests:** `tests/docs/test_examples.py` executes quickstart end-to-end (nbmake or papermill); CLI smoke test on fixture.

**Docs:** README quickstart section points at notebook 01 + CLI one-liner.

---

## Zones module — highlight, don't rebuild

**`hydrofragments/spatial/zones.py::build_zones` is already a complete, correct, spec-locked implementation of the §3 four-zone schema** (in-channel / persistent off-channel / seasonally-flooded / marginal, with the single persistence-proxy no-drainage fallback and no morphology-proxy path — matching the audit's hard cut). It's not part of the gap list; it just isn't surfaced anywhere user-facing yet.

**Action (documentation/visibility only, not a build item):**
- Give it its own section in `docs/input_format.md` or a new `docs/zones.md`: the table below, worked example (with vs without drainage layer), and the circularity guard explanation (why persistence metrics never stratify by zone 2/3/4).
- Feature it in notebook 03 (metrics walkthrough, Section 5) as the spatial-stratification step before the 4 analysis sections run per-zone.
- Reference it explicitly in the paper's F2 figure (already planned in spec §10).

| Zone | Name | Default definition | Drainage layer required? |
|---|---|---|---|
| 1 | In-channel | Drainage channel mask + adjacent high-frequency pixels | **Yes** (else Zone 1 is not emitted) |
| 2 | Persistent off-channel | RAW freq > `t_persist` (default 50%), spatially isolated from channel | No |
| 3 | Seasonally flooded floodplain | RAW freq `t_season`–`t_persist` (default 10–50%) | No |
| 4 | Marginal / extreme-event floodplain | RAW freq < `t_season` (default 10%), within max wet extent | No |

---

## The 4 analysis-section delineation (for the spec doc)

Grouping spec §6 metric families (output `metric_family` taxonomy, §7.1) into 4 analysis sections:

1. **Extent & Persistence** — APSEC, LPSEC, occurrence, recurrence, hydroperiod, Refuge Area. *"How much water, how reliably."*
2. **Morphology & Fragmentation** — N, LPI, MESH, AWRe, AWMSI, pool-width distribution. *"How the water is shaped and broken up."*
3. **Clustering & Connectivity** — inter-pool gap (primary), NNI (exploratory), RC, TCF, DCI. *"How pools relate in space and network."*
4. **Dynamics** — dry-down rate, reconnection timing, refuge spatial stability. *"How it changes through the hydrological year."* (this is where the hydroseason link is load-bearing.)

Each maps 1:1 to a `metric_family` set, so it's a clean documentation + notebook-03 + paper-figure spine, not a new code axis.

---

## Sequencing

1. **Section 1** (generic adapter) — unblocks real DEA/TSFill + other-source input; everything else consumes cubes.
2. **Section 2** (quality→prescription) — small, sits on adapter provenance.
3. **Section 5** (notebooks + CLI) — needs 1 & 2 to demo the real path; delivers the visible user-ready win.
4. **Section 3** (CUDA benchmarks) — independent; can run in parallel.
5. **Section 4** (Numba) — after 3's benchmark harness exists (reuse it).

Sections 1–2, 5 = user-ready. 3–4 = performance, benchmark-gated, no correctness risk to the default CPU path.
