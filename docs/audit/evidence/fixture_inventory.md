# Fixture inventory evidence (U1 / Q8)

**Captured:** 2026-07-10T21:10:00+08:00  
**Method:** read-only inspection via `tests/contracts/fixture_inspector.py`  
**Verification:** `python -m pytest tests/contracts/test_fixture_characterisation.py -v` → **6 passed** (2026-07-10)

## Bundled HydroFragments fixtures

### `tests/wmask_ts.nc`

| Field | Value |
|---|---|
| SHA-256 | `8866c7737a33fa078a9daf74b8f08435ada24096188ee1f76c3cc1487ef698d6` |
| Size | 6,064,558 bytes |
| Variable | `water` |
| Dtype | `int16` |
| Dimensions | time=63, y=145, x=331 |
| CRS | EPSG:28351 (GDA94 / MGA zone 51) — **not** equal-area EPSG:3577 |
| Time range | 2018-01-01 → 2020-12-16 |
| Cadence | **sub_monthly_irregular** (includes e.g. 2018-02-16 between monthly anchors) |
| Value domain | `-1` (nodata)=1,956,402; `0` (dry)=936,826; `1` (water)=130,457 |
| Sentinel 254/255 | **absent** (legacy int16 nodata, not WaterMask-TSFill uint8 contract) |
| Valid-observation layer | **absent** |
| Wet-fraction per timestep | min≈0.0118, max≈0.1034, std≈0.0156, **61 unique** wet fractions |

**Suitable uses**

- Legacy integration smoke (with `tests/rcor_extent.shp`)
- Low-level kernel characterisation when paired with **synthetic** `valid_obs`
- APSEC/N/LPI/AWRe/AWMSI shape tests after CRS policy is explicit

**Unsuitable uses (without new fixtures)**

- v1.2 occurrence/RA denominator contract (no `valid_obs`)
- WaterMask-TSFill sentinel decode tests (wrong encoding)
- Dry-down, HY anchors, refuge stability, recurrence/hydroperiod
- Cross-sensor or equal-area contract tests (CRS is UTM MGA)

### `tests/rcor_extent.shp` (+ sidecars)

| Field | Value |
|---|---|
| SHA-256 (`.shp`) | `2140ca8e91cc68e0855ebadd54d25296f1354b18c298245a3beca5284e4790cc` |
| Features | 7 polygons |
| CRS | EPSG:28351 |
| Geometry | Polygon corridors only |
| Drainage centreline | **not present** (polygon AOI only; `len` attribute exists but geometry is areal) |

**Suitable uses:** legacy seven-section smoke AOI  
**Unsuitable uses:** real `L_ref`, Zone 1 channel, inter-pool gap ordering, fixed-node graph

## Upstream / validation fixtures (outside HydroFragments repo)

| Asset | Location | Status at M0 |
|---|---|---|
| WaterMask-TSFill `contracts.py` | `D:/RLH/5.6/repos/WaterMask-TSFill/watermask_tsfill/contracts.py` | **read** — canonical schema documented |
| Canonical `water_cube.zarr` (filled monthly cube) | Not present in HydroFragments workspace at M0 | **not characterised** — required for U2/Q1 sensitivity |
| Raw sub-monthly flags Zarr (Gilbert) | `D:/RLH/5.6/data_local/raw/WaterMask-TSFill/cache/raw_wofs_flags_gilbert_1986_2026.zarr` exists on disk | **not characterised in M0** — candidate for U3 dual-composite evidence |
| Gilbert validation catchment drainage lines | Not found in HydroFragments repo inventory | **missing** — blocks U4 |

## Q8 fixture strategy (evidence-backed proposal)

| Tier | Fixture | Role |
|---|---|---|
| A | Tiny analytic masks (to be added Milestone 1) | Formula truth, chunk boundaries, sentinels |
| B | `tests/wmask_ts.nc` + `rcor_extent.shp` | Legacy smoke only |
| C | Real `water_cube.zarr` + validation catchment subset | Occurrence validity policy, V1–V3 science |
| D | Raw sub-monthly or dual monthly composites | Dry-down / composite-sensitivity (U3) |

**U1 conclusion for Decision Gate 0:** fixture **characterised**; **cannot** close Q8 as “bundled NC alone is sufficient for v1.2 contract tests.” Maintainer must approve Tier A+C as mandatory before occurrence/dry-down milestones.
