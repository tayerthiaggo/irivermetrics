# Input format (HydroFragments v1.2)

## Canonical object

`open_water_cube()` returns a `WaterCube` with:

- `water` — boolean or encoded water state
- `valid_obs` — boolean valid-observation mask
- `cadence` — `monthly` or `submonthly`
- optional provenance metadata

## Supported sources

### WaterMask-TSFill Zarr / Dataset

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

### Generic binary pair

`water` + `valid_obs` arrays or a single binary mask (`1` = water) with implicit
`valid_obs=True` everywhere.

### Generic probability (config-gated)

Requires `input.kind = generic_probability` plus threshold provenance fields in
`HydroConfig`.

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
