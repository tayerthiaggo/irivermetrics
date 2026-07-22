# Drainage / centreline inventory evidence (U4 / Q6)

**Captured:** 2026-07-10T21:10:00+08:00  
**Scope:** files discoverable from HydroFragments repository and bundled test fixtures

## Search performed

| Location | Pattern | Result |
|---|---|---|
| `HydroFragments/` repo | `*drain*`, `*centreline*`, `*centerline*`, `*channel*.shp` | **0 files** |
| `tests/rcor_extent.shp` | polygon corridors | 7 polygon features, EPSG:28351 |
| `HydroFragments` codebase grep | `drainage`, `L_ref`, `centreline` | spec/docs only — **no runtime drainage loader** |

## What exists today

### `tests/rcor_extent.shp`

- Geometry: **Polygon** (7 sections)
- CRS: EPSG:28351
- Attributes include `len`, `Shape_Leng`, `Shape_Area`, `Feat_type`
- **Not** a centreline/network layer
- Cannot supply per-AOI `L_ref`, ordered gap path, Zone 1 geomorphic channel, or fixed graph nodes without additional data

### Current API (`ecofragments/main.py`)

- Accepts `rcor_extent` polygons and optional scalar `section_length`
- No drainage line input parameter
- Matches `spec_compliance.md` Blocker 5

## What v1.2 requires (when channel profile enabled)

Per `HydroFragments_v1.2_spec.md` §2–§3 and implementation plan Milestone 10:

| Need | Minimum contract |
|---|---|
| Centreline / drainage | `LineString`/`MultiLineString` in same CRS as raster after co-reprojection |
| `L_ref` | Centreline length clipped to AOI — **not** wet-derived skeleton in core release |
| Topology | Ordered reach direction for inter-pool gap; node sources for RC/TCF |
| Versioning | Dataset ID, CRS, source, snapshot date in manifest |

## External assets (filesystem search completed 2026-07-10)

Recursive search under `D:\RLH\5.6` for `*drain*`, `*centreline*`, `*centerline*`, `*channel*.shp` (first 25 hits, ~23 min runtime):

| Hit | Verdict |
|---|---|
| `HydroFragments/docs/audit/evidence/drainage_inventory.md` | This evidence file — not data |
| `suaimob-mvp/.../node_modules/...` (lucide icons, test JS) | False positives — not geospatial data |

**No drainage shapefile or centreline dataset found** under `D:\RLH\5.6` in this search.

Gilbert/Fitzroy validation catchments referenced in audits remain **without** an inventoried centreline layer in this workspace.

## U4 / Q6 status

| Question | M0 answer |
|---|---|
| Does a drainage dataset exist for validation catchment(s)? | **Yes** — `data/fitzroy_kimberley_drainage.gpkg`, hashed and inspected (below) |
| Can LPSEC, Zone 1, gap, RC, TCF proceed? | **Design/implementation can proceed** for the Fitzroy catchment; LPSEC still excluded from v1.2.0 core per `adversarial_synthesis_2.md` §7 regardless of drainage availability |
| Can polygon-only core (APSEC, N, LPI, …) proceed? | **Yes** — no drainage required |

**Closed:** maintainer supplied a centreline dataset with CRS/extent matching the water-mask grid and
complete topology. See "Real drainage dataset supplied" section below for hash, CRS, and contract check.

## Proxy-channel fallback idea (not accepted as core `L_ref`)

Maintainer suggested developing an alternative when no drainage dataset is supplied, for example skeletonizing representative mid-season water between peak flow and mid-dry.

M0 interpretation:

- This is a plausible **proxy-channel research fallback** for degraded/no-drainage mode.
- It must be explicitly flagged (`proxy_channel=true`) and validated before any channel-derived metric becomes reportable.
- It must not silently replace publication-grade `L_ref` in the core release.
- Candidate design inputs: peak-flow month, mid-dry month, persistence surface, maximum-wet extent, and stability across years.
- Candidate validation: compare proxy skeleton against supplied basis drainage once available; quantify length bias, topology errors, and downstream LPSEC/gap sensitivity.

**Status:** superseded — real basis drainage was supplied (below) and validated. Proxy-channel fallback
remains documented as a future degraded-mode design for catchments where no AHGF-equivalent drainage
exists, but is **not needed** for the Fitzroy validation catchment.

## Real drainage dataset supplied (2026-07-14)

**Path:** `data/fitzroy_kimberley_drainage.gpkg`
**SHA-256:** `004442d0a65a7eeb51a335dbaa621e281f610080b31e7ae05ee9980a46dc3b3a`
**Source:** AHGF (Australian Hydrological Geospatial Fabric)-style drainage extract, clipped to the Fitzroy/Kimberley AOI

| Property | Value |
|---|---|
| Geometry type | `MultiLineString` (291 features, single geometry type — homogeneous) |
| CRS | **EPSG:3577** — identical to `water_mask` grid CRS, no reprojection needed |
| Extent (EPSG:3577) | x: [-856100.7, -823133.0], y: [-1967276.4, -1951940.6] — matches `water_mask` grid extent (x: [-856125, -822615], y: [-1967895, -1951725]) |
| Total reach length (`GeodesLen`) | 525,129 m (~525 km) across 291 reaches |
| Hierarchy | 70 `Major`, 221 `Minor` |
| Perennial flag | 21 `Perennial`, 270 `Non Perennial` |
| Topology | `From_Node`, `To_Node`, `NextDownID` fully populated — **0 nulls** across all 291 features; supports ordered reach traversal for inter-pool gap and RC/TCF |
| Other attributes | `AHGFFType`, `SegmentNo`, `DrainID`, `FlowDir`, `UpstrDArea`, `ConCatID`, `StrOutlet`, `Shape_Length` |

**Verdict:** this dataset satisfies the v1.2 minimum contract in the table above — `MultiLineString`
centreline geometry, matching CRS/extent to the water-mask grid, and complete upstream/downstream topology.
It supports `L_ref` (centreline length clipped to AOI), ordered gap-path derivation, and RC/TCF node
sourcing for the Fitzroy validation catchment.

**Open item before Milestone 10 implementation (not M0-blocking):** confirm `AHGFFType` codes (`1`, `3`)
against the AHGF data dictionary to decide whether both types are in-scope for `L_ref`, or only one
(e.g. mapped stream vs artificial channel).
