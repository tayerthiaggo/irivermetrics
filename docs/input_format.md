# Input format (HydroFragments v1.2)

## Canonical object

`open_water_cube()` returns a `WaterCube` with:

- `water` — boolean or encoded water state
- `valid_obs` — boolean valid-observation mask
- `cadence` — `monthly` or `submonthly`
- `provenance` — key/value tuples recording the resolved adapter, the
  requested `input_kind` (or `"auto"`), chunking, and any auto-fixes applied
  (see "Auto-detection and the input contract" below)

## Supported sources

`open_water_cube(source, ...)` accepts an `xr.DataArray`, an `xr.Dataset`, or
a path (currently `.zarr`). `input_kind` defaults to `None`: the shape of
`source` is auto-detected and routed to the matching adapter below. Passing
an explicit `input_kind` (one of `watermask_tsfill`, `raw_wofs`,
`generic_binary`) skips detection and forces that adapter — use this when
detection would be ambiguous, or when the caller already knows the source's
shape.

### WaterMask-TSFill Zarr / Dataset (`watermask_tsfill`)

Variables: `water_mask`, optional `observed`, `confidence`, `method_flag`.

Sentinel values:

| Value | Meaning |
|-------|---------|
| `0` | dry |
| `1` | water |
| `254` | outside AOI |
| `255` | unobserved |

Sentinels must be decoded **before** any signed cast. They are never counted as
dry or water.

**Auto-detection signature:** a variable literally named `water_mask`, or a
`uint8` array whose only values are a subset of `{0, 1, 254, 255}` and
includes at least one of the TSFill-specific sentinels (`254`/`255`) —
distinguishes a TSFill export from a plain `{0,1}` binary mask that merely
happens to be `uint8`.

**TSFill handoff.** TSFill (a separate tool) owns DEA/STAC access and
gapfilling; HydroFragments only consumes its output. The handoff is a single
export/import boundary:

```
DEA/STAC source imagery
        |
        v
  WaterMask-TSFill  (gapfills, encodes 0/1/254/255 sentinels)
        |
        v
  canonical uint8 Zarr/Dataset (water_mask + optional observed/
  confidence/method_flag)
        |
        v
  open_water_cube(path)  --auto-detect-->  watermask_tsfill adapter
        |
        v
  WaterCube(water, valid_obs)
```

HydroFragments never talks to DEA/STAC directly (no `odc.stac`/`pystac`
dependency) and never gapfills (see the note at the end of "Auto-detection
and the input contract" below).

### Raw DEA WOfS (`raw_wofs`)

A `water` or `frequency`-named band/variable following DEA WOfS naming
conventions, values in `{0, 1}` (already-binary) or a probability/frequency
in `[0, 1]` requiring `water_threshold` to binarize (raise `ValueError` if
non-binary and no `water_threshold` is supplied — this adapter never
guesses a threshold). Raw WOfS does not bundle a separate valid-observation
layer by convention: if the caller does not supply `valid_obs`, an all-`True`
mask is used. Pass `water_threshold` to `open_water_cube(..., input_kind=
"raw_wofs", water_threshold=0.5)` to threshold a probability/frequency band.
`HydroConfig.input.water_threshold`/`input.kind`/`input.variable_map` are
recorded for provenance and comparison-guard purposes only — today's CLI and
`analyze()` do not read them back into `open_water_cube`; they auto-detect
every time. Set `water_threshold` as a direct `open_water_cube` kwarg (or via
your own analyze-time integration) until that config-driven wiring lands.

**Auto-detection signature:** a variable named `water` or `frequency`.

### Generic binary pair (`generic_binary`)

The fallback adapter for source-agnostic masks with no WOfS-specific or
TSFill-specific naming/sentinel convention: `bool` arrays pass straight
through; `{0, 1}` int/float arrays are coerced to bool. An optional paired
`valid_obs` array and/or an explicit `nodata` sentinel value are honored (a
pixel excluded by either is invalid and never counted as water). With
neither, an all-`True` valid mask is used.

**Auto-detection signature:** this is the fallback — anything not matching
`watermask_tsfill`'s or `raw_wofs`'s signature above, provided the values are
`bool` or `{0, 1}`.

### Generic probability (config-gated)

Requires `input.kind = generic_probability` plus threshold provenance fields in
`HydroConfig`. Not yet wired through `open_water_cube`'s adapter registry —
this is a known gap tracked outside this document's scope, not a supported
`open_water_cube` code path today.

## Auto-detection and the input contract

`open_water_cube` never asks the user to manually describe their data's
layout. Instead it runs an inspect-then-act sequence:

1. **Normalize structure** (always safe, always logged) — rename a single
   unambiguous data variable to the expected name, apply `variable_map`
   renames, coerce an already-binary `{0,1}` int/float array to `bool`,
   reorder dims to `(time, y, x)`. Every fix applied is recorded as a
   human-readable string in `WaterCube.provenance["auto_fixes"]`.
2. **Detect the adapter** (unless `input_kind` was given explicitly) — see
   each adapter's "Auto-detection signature" above. A `Dataset` with more
   than one data variable and no recognizable water-like candidate is
   **never guessed** — it raises `ValueError` naming the ambiguous variables,
   asking the caller to supply `variable_map` or an explicit `input_kind`.
3. **Check the contract, never silently fix a grid/CRS problem** — the
   following always raise `InputContractError` (never a silent
   resample/reproject):
   - Grid/transform mismatch between the water layer and a caller-supplied
     `valid_obs` layer (shape, dim order, or coordinate values differ).
   - CRS mismatch between the water and `valid_obs` layers.
   - A defined-but-geographic (degrees) CRS on the water layer — a
     projected CRS in metres is required (spec §8 guard 8). An *undefined*
     CRS is not itself an error (many valid in-memory/generic_binary inputs
     carry no georeferencing at all); only a CRS that is set and in degrees
     is refused.
   - An ambiguous multi-variable `Dataset` with no water-like candidate (see
     step 2).
4. **Dispatch to the adapter** — the resolved/normalized array is parsed by
   the chosen adapter into `(water, valid_obs)`.

| Mismatch | Outcome |
|---|---|
| Single unnamed/renamable data variable | auto-fix (rename), logged |
| `variable_map` rename | auto-fix (rename), logged |
| `{0,1}` int/float dtype | auto-fix (coerce to bool), logged |
| Dims out of `(time, y, x)` order | auto-fix (reorder), logged |
| Grid/transform mismatch (water vs. valid_obs) | raise `InputContractError` |
| CRS mismatch (water vs. valid_obs) | raise `InputContractError` |
| Geographic (degrees) CRS | raise `InputContractError` |
| Ambiguous multi-variable Dataset | raise `ValueError`/`InputContractError` |

HydroFragments never gapfills; it only consumes already-gapfilled input
(typically via WaterMask-TSFill, see above). Set `gapfill: true` in config
once your input is pre-filled (e.g. via WaterMask-TSFill); leave it `false`
(the default) to run on raw data and get quality flags instead — see
"Baseline quality assessment and the `gapfill` flag" below.

## Baseline quality assessment and the `gapfill` flag

Before running metrics, `validate_inputs`/`analyze` assess how much valid
observation coverage the input cube actually has
(`hydrofragments.guards.quality.assess_baseline_quality`):

- Overall valid-observation coverage, per-calendar-month valid fraction, and
  a season-stratified occurrence summary (reusing the same MNAR-corrected
  estimator locked in for the `occurrence` metric, Decision U2/Q1).
- The fraction of pixels/pixel-months below the existing
  `validity.min_valid_obs` / `validity.min_valid_fraction_month` floors.

When coverage is below those floors and `HydroConfig.gapfill` is `False`
(the default), a recommendation is appended to `ValidationReport.warnings`
(and threaded into the run manifest) pointing at WaterMask-TSFill:
"insufficient baseline coverage; consider pre-processing with
WaterMask-TSFill before running HydroFragments, or set gapfill=true if
already gapfilled."

Set `gapfill: true` once the input has already been gapfilled upstream (for
example via WaterMask-TSFill) to suppress this recommendation. HydroFragments
trusts that declaration outright and does **not** re-check coverage against
it — the whole point of the flag is that the caller already knows. Either
way, `gapfill` is recorded in the run manifest's `execution_config` so it is
auditable per run. This assessment is purely advisory: it never mutates
`cube.water`/`cube.valid_obs`, and HydroFragments itself never gapfills,
interpolates, or otherwise fills data — that remains WaterMask-TSFill's job.

## Validity policy

Locked policy name: `p_native_season_stratified_v1` (Decision U2/Q1).

Temporal aggregates (occurrence, refuge area, recurrence, hydroperiod) use
season-stratified valid-observation denominators — not total timestep count.

## Spatial alignment

- Raster, AOI, and optional drainage must agree in shape, CRS, transform, and time.
- Misalignment raises; no silent resampling.
- Geographic CRS inputs require explicit reprojection or per-pixel area policy.

## Monthly compositing

Already-monthly upstream products keep supplied provenance (`composite_owner:
upstream`). HydroFragments does not invent a second monthly composite from a single
upstream monthly mask.

Dual-composite dry-down metrics remain blocked without raw observations or both
required composites (Decision U3/Q3).

## Configuration traceability

Scientific settings hash to `config_hash` (see `hydrofragments.config.HydroConfig`).
Execution paths, worker counts, and accelerator choice are excluded from
`config_hash` but recorded separately in the run manifest.
