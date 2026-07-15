# HydroFragments v1.2 Implementation Plan

> **For agentic workers:** Before any implementation work, enumerate and read every Markdown file under `docs/audit/`, then read `docs/HydroFragments_v1.2_spec.md` and `docs/audit_implementation_plan.md`. Record the intake in `docs/audit/intake_manifest.md`. No production code may be changed until Milestone 0 evidence and Decision Gate 0 are complete. Implement this plan task-by-task using test-first changes and reviewer gates.

**Goal:** Migrate the current EcoFragments/iRiverMetrics implementation to a scientifically defensible, source-agnostic HydroFragments v1.2 release through small, reversible increments.

**Architecture:** Stabilise input, configuration, output, and quality contracts before expanding metrics. Keep Dask/xarray lazy for I/O, validity, temporal reductions, and occurrence; use bounded CPU component work for exact morphology. Treat CUDA, connectivity, and publication polish as optional tranches with independent acceptance gates.

**Tech stack:** Python 3.10+, xarray, Dask Array/Distributed, rioxarray/odc-geo/rasterio, NumPy/SciPy/scikit-image, GeoPandas/Shapely, Pandas/Parquet, pytest, optional CuPy after CPU parity exists.

## Global constraints

- No implementation phase may rely on an audit summary in place of reading the audit files.
- CPU results are the scientific reference. GPU packages remain optional.
- Canonical v1.2 output is tidy and contains no `PF`, `PLF`, `AWMPA`, `AWMPL`, or original `AWMPW` metrics.
- Tests and schemas stabilise before metric expansion.
- Sentinel values `254` and `255` are decoded before any signed cast and never treated as dry or water.
- Occurrence and Refuge Area remain blocked until valid-observation semantics are supported by evidence.
- LPSEC and channel-dependent metrics require a real drainage/centreline reference; a wet-derived skeleton is not accepted as `L_ref` for the core release.
- Monthly input must carry composite provenance. HydroFragments must not invent a second composite from an already monthly mask.
- Dry-down/extent-contraction metrics remain blocked unless raw sub-monthly observations or both required monthly composites are available.
- Dask-first means lazy, chunk-aware array stages with explicit materialisation boundaries; it does not imply out-of-core or GPU execution for every spatial kernel.
- Documentation must distinguish asserted interpretations from empirically demonstrated claims.
- Manager-facing outputs must not imply flow, discharge, groundwater, depth, water quality, or ecological condition.
- Existing user changes and historical fixtures are preserved unless a milestone explicitly retires them through a reviewed migration.

---

## 0. Intake record and decisions

### Audit intake manifest

This plan ingested the following files present at planning time:

1. `docs/audit/adversarial_synthesis.md`
2. `docs/audit/adversarial_synthesis_2.md`
3. `docs/audit/dask_cuda_audit.md`
4. `docs/audit/dask_cuda_audit_adversarial.md`
5. `docs/audit/docs_audit.md`
6. `docs/audit/evidence_packet.md`
7. `docs/audit/manager_interpretation_audit.md`
8. `docs/audit/repo_triage.md`
9. `docs/audit/scientific_metrics_audit.md`
10. `docs/audit/spec_compliance.md`
11. `docs/HydroFragments_v1.2_spec.md`
12. `docs/audit_implementation_plan.md`

Every later phase must create a fresh manifest because audit files may be added or revised after this plan. Manifest must include path, SHA-256 digest, read timestamp, reader/agent identifier, and unresolved-decision list. A changed digest invalidates prior approval for affected milestones.

### Evidence learned during planning

- Upstream `WaterMask-TSFill/watermask_tsfill/contracts.py` confirms monthly `water_mask`, `confidence`, `method_flag`, and `observed` variables, with `water_mask` values `0`, `1`, `254`, and `255`.
- Visible six-year canonical test Zarr contains 72 months, Dask chunks `(12, 512, 512)`, all four sentinel values, and both native and filled resolved pixels. About 4.9% of all pixels are resolved dry/water values where `observed=False`. Choosing native-only versus confidence/provenance-qualified filled observations will materially alter occurrence-derived results.
- Canonical upstream output is one monthly reconstructed mask. It does not itself provide both `max_water` and `median` composites. Dry-down dual-composite analysis therefore requires a separate raw/sub-monthly source or upstream dual products.
- No drainage/centreline dataset was found in the visible HydroFragments or WaterMask-TSFill file inventories. Channel-dependent work remains blocked pending a supplied dataset and contract.
- `tests/results_iRiverMetrics/metrics/irm_metrics.csv` exists, has 441 rows, contains naive `pp_mean_%`, and contains dropped metrics. It is unsuitable as a v1.2 correctness oracle. It may remain a historical smoke fixture for invariant low-level kernels only.
- `tests/wmask_ts.nc` could not be characterised in the active environment because NetCDF backends were unavailable. U1 remains open.

### Conflict resolution table

| Conflict | Plan decision | Evidence/gate | What breaks if wrong |
|---|---|---|---|
| Rebrand timing | Freeze API/schema first; expose `hydrofragments` publicly only after core contracts pass. Keep `ecofragments` as a temporary facade. | Adversarial and docs audits | Early rename multiplies identity drift; late internal design creates needless moves. |
| CUDA ambition | `dask_cuda_audit.md` is normative. CUDA starts after CPU benchmarks and is initially limited to eligible array stages. | CPU parity suite and benchmark gate | Silent metric drift, hard GPU dependency, wasted schedule. |
| Dry-down feasibility | Defer from minimal v1.2 core until dual-composite/raw input evidence exists. Call it surface-water extent contraction in user-facing text. | U3/Q3 evidence | Headline metric becomes irreproducible or scientifically overstated. |
| HY algorithm | Out of repo: consume sibling package `hydroseason` for HY/season mapping and HY metrics. Thin adapter only in HydroFragments. Close Q7/V8 against that package’s API + version, not a local detector. | Q7/V8; `../hydroseason` | Duplicate novelty claim and unstable anchors if we reimplement; broken integration if API/version not pinned. |
| Drainage contract | Core release does not emit LPSEC, Zone 1, gap, RC, TCF, or DCI without a real drainage reference. | U4/Q6 | Wet-derived denominator undermines fixed-denominator thesis. |
| Validity semantics | Remains unresolved. No occurrence/RA implementation merge until upstream intent and sensitivity evidence close Q1/U2. | Upstream author/ADR plus real-cube comparison | Occurrence, RA, zones, recurrence, and hydroperiod all change. |
| NNI fate | Cut from v1.2 runtime and manager/publication surfaces. | Scientific and manager audits | Wrong planar null creates maintenance and reputational risk. |
| DCI scope | Citation and conceptual positioning are required; runtime DCI is optional and later. | RC/DCI design and reference benchmark | Connectivity scope expands before graph semantics stabilise. |
| Legacy compatibility | Canonical v1.2 is never hybrid. Compatibility facade may pivot retained v1.2 metrics, but requests for dropped metrics fail with migration guidance. | Schema tests | Hybrid tables preserve scientifically rejected outputs. |
| Docs honesty | Correct status/import/scope immediately; finish v1.2 API and manager narratives only after implemented outputs exist. | Docs build and runnable examples | Documentation becomes vaporware. |
| Regression baseline | Retire legacy CSV as correctness baseline; preserve only for historical kernel/smoke comparisons with explicit exclusions. | Baseline provenance review | Correct changes fail because tests canonise old denominator defects. |
| Publication history | Maintainer must decide now whether to restore/graft predecessor history or explicitly abandon that claim. This does not block numerical core, but blocks lineage/publication claims. | Governance record | Lost public-history evidence cannot be recreated later. |

### Decision Gate 0

Before production code changes, create `docs/audit/decisions.md`. Each row must contain: decision, evidence artifact, owner, approval date, consequence if wrong, and affected milestone. It must close:

- U1: bundled fixture dimensions, CRS, time range, wet-fraction variability, and suitable uses;
- U2/Q1: valid-observation denominator policy;
- U3/Q3: compositing ownership and dry-down input availability;
- U4/Q6: drainage/centreline availability and topology contract;
- U7: legacy baseline disposition;
- Q2: canonical input object;
- Q4: DCI citation-only versus implementation;
- Q5: compatibility policy;
- Q7: HY authority via external `hydroseason` (API contract, version pin, drought/fallback);
- Q8: validation fixtures;
- Q9: canonical config hashing rules;
- Q10/governance: predecessor-history decision.

Missing evidence is not a default. If validity, compositing, drainage, or metric semantics remain unresolved, only milestones independent of that decision may proceed.

---

## 1. Target architecture

### Processing flow

1. **Open and decode:** adapters load generic water/valid pairs or canonical WaterMask-TSFill datasets. Sentinel and provenance semantics become explicit boolean/typed arrays before narrowing.
2. **Validate contracts:** water, validity, AOI, and optional drainage must agree in dimensions, coordinates, transform, CRS, and time. Misalignment raises; no silent resampling.
3. **Normalise space:** raster, AOI, and drainage are co-reprojected into the configured equal-area CRS, or a tested per-pixel-area path supplies areas. Length distortion remains explicit metadata.
4. **Normalise time:** cadence is validated. Sub-monthly inputs may be composited under a declared rule; already-monthly inputs retain supplied provenance and are not recomposited implicitly.
5. **Checkpoint temporal products:** monthly water/valid arrays and compact diagnostics form the deliberate materialisation boundary, preferably Zarr for large runs.
6. **Compute lazy reductions:** valid counts, wet counts, occurrence, RA support arrays, monthly validity, and pixel-temporal products use Dask/xarray.
7. **Build bounded patch work:** one month and AOI/window at a time receives globally reconciled `int32` connected-component labels. Component bounding boxes feed CPU NumPy crops for exact region properties, skeleton, EDT, and path work.
8. **Apply metric registry and guards:** each metric declares dependencies, tier, unit, statistic, validity policy, and edge-case behavior. Missing dependencies cause explicit skipped/NaN rows with flags, never fabricated proxy results.
9. **Emit results:** tidy metric rows, run manifest, diagnostics, config, and spatial products write incrementally. Vector export is a separate bounded DAG and cannot trigger metric recomputation.
10. **Compare safely:** comparison APIs reject mismatched resolution, source, AOI definition, monthly composite, or other load-bearing scientific settings unless an explicit override is recorded.

### Architectural invariants

- Scientific functions receive validated domain objects, not file paths or loosely structured dictionaries.
- Scientific functions do not call `.compute()`, `.persist()`, `.values`, or `np.asarray()` on Dask collections.
- Materialisation is owned by pipeline/orchestration modules and visible in diagnostics.
- Configuration separates scientific settings from execution settings.
- CPU and optional GPU backends implement the same stage contracts and emit actual-backend metadata.
- Patch geometry/path objects are opt-in side products, not columns in the default metric table.
- Output schema versioning is independent from package versioning.
- Every metric has a formula test, dependency test, edge-state test, and metadata test before registry activation.

---

## 2. Module boundaries

Target paths below describe final architecture. Early milestones may create them before public package metadata switches from `ecofragments` to `hydrofragments`; no temporary duplicate scientific implementation is allowed.

| Module | Responsibility | Consumes | Produces / must not own |
|---|---|---|---|
| `hydrofragments/api.py` | Small public facade and orchestration entry points | Validated config and user inputs | `HydroResult`; no metric formulas |
| `hydrofragments/config.py` | Typed immutable configuration, defaults, validation, canonical serialization | JSON/YAML/user mappings | `HydroConfig`, scientific/execution hash inputs; no filesystem opening |
| `hydrofragments/schema.py` | Output column types, enums, metric identifiers, schema versions | Metric records and manifests | Schema validation; no computation |
| `hydrofragments/models.py` | `WaterCube`, `SpatialContext`, `PatchTable`, `HydroResult`, validation report | xarray/geospatial objects | Stable domain interfaces; no I/O policy |
| `hydrofragments/io/adapters.py` | Zarr, NetCDF, GeoTIFF-stack, in-memory adapters | Paths/objects and variable mapping | Undecoded source cube plus source metadata |
| `hydrofragments/io/validity.py` | Sentinel decoding, validity-policy application, provenance preservation | Source cube and locked validity decision | Canonical water/valid/provenance arrays |
| `hydrofragments/io/alignment.py` | Grid, CRS, shape, coordinate, time checks | Canonical arrays | Validation result or explicit error; no resampling |
| `hydrofragments/spatial/context.py` | AOI, drainage, `A_ref`, `A_total`, optional real `L_ref`, IDs | Geometry and CRS policy | `SpatialContext`; no wet-derived core length reference |
| `hydrofragments/spatial/crs.py` | Co-reprojection, units, per-pixel area, length caveat metadata | Cube/AOI/drainage | Normalised spatial inputs |
| `hydrofragments/spatial/zones.py` | Static persistence zones and no-drainage behavior | Occurrence plus optional drainage | Zone mask; never computes persistence metrics |
| `hydrofragments/spatial/windows.py` | Fixed channel windows or regular grids | Spatial context | Stable window IDs/geometries |
| `hydrofragments/temporal/cadence.py` | Cadence detection and validation | Time coordinate | Monthly/sub-monthly classification |
| `hydrofragments/temporal/composites.py` | Declared monthly composite generation | Sub-monthly water/valid observations | Monthly products and provenance |
| `hydrofragments/temporal/hydroyear.py` | Thin adapter to external `hydroseason` (HY labels, seasons, anchors/confidence, related HY metrics) | Monthly extent/series + pinned `hydroseason` version/config | Mapped HY/season labels for downstream metrics; **no local HY algorithm** (Q7/V8) |
| `hydrofragments/compute/policy.py` | Scheduler/materialisation policy | Execution config | Explicit stage execution plan |
| `hydrofragments/compute/chunks.py` | Named chunk contracts and byte-budget validation | Dask arrays and hardware budget | Rechunk plan/diagnostics |
| `hydrofragments/compute/capabilities.py` | Optional backend detection and certified kernel registry | Runtime environment | Immutable planned/actual backend record |
| `hydrofragments/patches/labels.py` | 2-D global component labels, connectivity, minimum mapping unit | One monthly mask | `int32` labels and component counts |
| `hydrofragments/patches/components.py` | Bounding boxes, crop bucketing, compact patch table | Labels and masks | Numeric patch table; no global Pandas accumulation |
| `hydrofragments/patches/morphology.py` | Region properties, major-axis/skeleton length, EDT width | Concrete bounded component crops | Numeric patch properties; CPU reference |
| `hydrofragments/metrics/registry.py` | Metric IDs, tiers, dependencies, units, statistics | Metric implementations | Selected metric plan and dependency skips |
| `hydrofragments/metrics/extent.py` | APSEC; later LPSEC when real `L_ref` exists | Counts/areas/spatial context | Metric records |
| `hydrofragments/metrics/persistence.py` | Occurrence, RA, recurrence, hydroperiod | Water/valid reductions | Raster and summary records |
| `hydrofragments/metrics/patches.py` | N, LPI, AWRe, AWMSI; gated MESH | Patch table and fixed AOI area | Metric records |
| `hydrofragments/metrics/dynamics.py` | Extent contraction, reconnection, refuge stability | HY anchors and prerequisite metrics | HY summary records; later tranche |
| `hydrofragments/metrics/clustering.py` | Inter-pool gap only | Real channel reference | Gap summaries; no NNI in v1.2 |
| `hydrofragments/metrics/connectivity.py` | RC, TCF, optional DCI | Fixed graph/node source | Connectivity records; later optional tranche |
| `hydrofragments/guards/scientific.py` | Circularity, CRS, AOI comparability, width-floor, dependency guards | Config/context/records | Errors, suppressions, warning flags |
| `hydrofragments/guards/comparison.py` | Cross-run compatibility checks | Two manifests/result sets | Approved comparison or explicit refusal |
| `hydrofragments/output/tables.py` | Incremental Parquet/CSV rows and optional retained-metric pivot | Valid records | Tidy table and compatibility export |
| `hydrofragments/output/rasters.py` | Occurrence/valid/refuge/zone products | xarray products | Zarr/GeoTIFF outputs |
| `hydrofragments/output/manifest.py` | Config, hashes, versions, input fingerprints, backend use | Run context | Machine-readable run manifest |
| `hydrofragments/pipeline.py` | Stage graph assembly and materialisation boundaries | All deep modules | `HydroResult`; no scientific formulas |
| `ecofragments/__init__.py`, `ecofragments/main.py` | Temporary compatibility facade | Legacy-shaped calls | Deprecation/migration path; no duplicate kernels |

The current `ecofragments/utils/calc_metrics.py` becomes a source of characterised kernels, not a destination for new behavior. Functions move only after parity or replacement tests exist.

---

## 3. Public API proposal

### Canonical API

- `open_water_cube(source, *, valid_obs=None, variable_map=None, chunks=None) -> WaterCube`
  - Accepts canonical WaterMask-TSFill Zarr/Dataset, generic aligned pairs, or file adapters.
  - Does not decide scientific validity policy silently.
- `validate_inputs(cube, aoi, *, drainage=None, config) -> ValidationReport`
  - Runs schema, sentinel, grid, CRS, cadence, and dependency checks without computing metrics.
- `analyze(cube, aoi, *, config, drainage=None) -> HydroResult`
  - Only high-level scientific execution entry point.
- `compare_results(left, right, *, overrides=None) -> ComparisonResult`
  - Refuses incompatible scientific settings by default and records any override.
- `HydroResult.write(path, *, formats=(...))`
  - Writes metrics, manifest, diagnostics, config, and requested spatial outputs without recomputing the analysis DAG.

### Domain objects

- `WaterCube`: water state, valid-observation mask, confidence/method provenance, time/grid/CRS, source identity, cadence/composite provenance.
- `SpatialContext`: AOI/catchment/window IDs, projected geometries, fixed areas, optional drainage and real `L_ref`, unit metadata.
- `HydroConfig`: frozen validated scientific and execution configuration.
- `HydroResult`: lazy or materialised metric table handle, spatial products, diagnostics, manifest, and schema version.
- `ValidationReport`: errors, warnings, resolved capabilities, skipped metric dependencies, and estimated compute plan.

### Metric selection

Use named profiles plus explicit additions:

- `contracts_core`: occurrence and RA after Q1, APSEC, N, LPI, AWRe, AWMSI.
- `pixel_temporal`: recurrence and hydroperiod after validity/year policy stabilises.
- `dynamics`: extent contraction, reconnection, refuge stability after composite/HY gates.
- `channel`: LPSEC and inter-pool gaps after drainage gate.
- `secondary`: width distribution and MESH only after their validation/guard gates.
- `connectivity`: RC and TCF after fixed-node graph design; DCI only when explicitly certified.

Profile expansion is visible in resolved config and output manifest. No default profile includes a metric whose required dependency is unavailable.

### Compatibility API

- Keep `ecofragments.calculate_metrics` for one documented deprecation cycle.
- Adapter translates legacy AOI and patch parameters into `HydroConfig`, then calls canonical pipeline.
- Default adapter output is a wide pivot of retained v1.2 metrics only, with warnings and a pointer to tidy output.
- Requests for `PF`, `PLF`, `AWMPA`, `AWMPL`, or original `AWMPW` raise a migration error explaining why each metric was removed.
- Existing legacy engine may remain test-only during migration but is not installed as a public v1.2 execution path.
- No `legacy_output=True` mode may append dropped columns to canonical v1.2 rows.

---

## 4. Config schema proposal

Configuration should be immutable, reject unknown fields, serialize canonically to JSON, and optionally load YAML. Every field below must have type, allowed values, default status, scientific/execution classification, and output traceability documented.

### Identity and schema

| Field | Proposal | Hash class |
|---|---|---|
| `config_schema_version` | Required semantic schema version | Scientific |
| `run_label` | Optional human label, excluded from scientific comparisons | Neither |
| `metric_profiles` | Ordered canonical set of enabled profiles | Scientific |
| `metric_overrides` | Explicit additions/removals with reasons | Scientific |

### Input and validity

| Field | Proposal | Default/status |
|---|---|---|
| `input.kind` | `watermask_tsfill`, `generic_binary`, `generic_probability` | Required |
| `input.variable_map` | Names for water, valid, confidence, method, observed | Adapter-specific |
| `input.water_threshold` | Probability cutoff | Required only for probability input |
| `input.threshold_method` | Threshold provenance/method | Required only for probability input |
| `input.probability_source` | Source/model identifier | Required only for probability input |
| `validity.policy` | Locked policy name and version | **Blocked on Q1/U2** |
| `validity.min_valid_obs` | Per-pixel floor | `20` |
| `validity.min_valid_fraction_month` | Monthly AOI/zone reportability floor | Required decision; do not inherit legacy 70/95% values |
| `validity.low_support_behavior` | `suppress_value`, `emit_flagged_value`, or metric-specific approved policy | Must be locked in Decision Gate 0 |

### Spatial and patch settings

| Field | Proposal | Default/status |
|---|---|---|
| `spatial.target_crs` | Equal-area computation CRS | `EPSG:3577` for Australian deployments |
| `spatial.area_method` | `projected` or `per_pixel` | `projected` |
| `spatial.length_crs_note` | Required manifest note for area-optimised CRS | Always emitted |
| `spatial.windowing.mode` | `none`, `channel_length`, `regular_grid` | `none` |
| `spatial.windowing.length_m` | Fixed channel window length | `5000` only when channel mode enabled |
| `patches.min_patch_pixels` | Minimum mapping unit | `3` |
| `patches.connectivity_rule` | `4` or `8` | `8` |
| `patches.width_resolution_floor_pixels` | Suppression/flag threshold | Must be scientifically approved; no invented default |

### Persistence, temporal, and dynamics

| Field | Proposal | Default/status |
|---|---|---|
| `persistence.refuge_threshold` | Occurrence cutoff for RA | `0.90` |
| `zones.t_persist` | Zone 2/3 threshold | `0.50` |
| `zones.t_season` | Zone 3/4 threshold | `0.10` |
| `temporal.input_cadence` | Declared/validated source cadence | Required or inferred with validation |
| `temporal.monthly_composite` | `max_water`, `median`, `mode`, `end_of_month_nearest`, or supplied provenance | Recorded always |
| `temporal.composite_owner` | `hydrofragments`, `upstream`, `caller` | Required |
| `dynamics.composite_sensitivity_tolerance_pp` | End-dry APSEC disagreement | `10` percentage points |
| `hydroyear.provider` | External package name | `hydroseason` (sibling repo; not implemented here) |
| `hydroyear.package_version` | Pinned `hydroseason` version recorded in manifest | Required for any HY/season-dependent run |
| `hydroyear.config` | `HydroYearConfig` (or equivalent) fields passed through to `hydroseason` | No local algorithm params; no silent override of package defaults without recording them |
| `dynamics.contraction_method` | `linear` or `theil_sen` | Must be locked before implementation |
| `dynamics.minimum_points` | Minimum monthly points for a reportable slope | Must be justified and recorded |

### Channel, state, and graph

| Field | Proposal | Default/status |
|---|---|---|
| `channel.source` | External centreline/network identity | Required for channel profile |
| `channel.node_source` | `channel_windows`, `fixed_refuge_nodes`, `skeleton_segments`, `external_network` | Required for graph profile |
| `connectivity.edge_rule` | Versioned wet-link/gap rule and threshold | Required for RC/TCF/DCI |
| `state.enabled` | Whether hydrological state labels emit | `false` in minimal core |
| `state.connectivity_metric` | `RC`, `LPSEC`, `LPI`, or later `DCI` | Required when state enabled |
| `state.connectivity_threshold` | Declared threshold | Required when state enabled |
| `reconnection.metric` | `RC`, `DCI`, `LPSEC`, or explicit `LPI` proxy | Required for dynamics profile |

### Compute and output

| Field | Proposal | Default/status |
|---|---|---|
| `compute.accelerator` | `none`, `auto`, `cuda` | `none` |
| `compute.cuda_strict` | Fail when requested certified CUDA stage unavailable | `false` |
| `compute.target_chunk_bytes` | Byte budget, not fixed pixels | Hardware/config dependent |
| `compute.worker_memory_fraction` | Chunk-planning input | Validated range |
| `compute.checkpoint` | `none`, `persist`, `zarr` | `zarr` recommended for large runs |
| `compute.scheduler` | Local/thread/process/distributed policy | Execution-only |
| `output.formats` | Parquet primary; optional CSV/Zarr/GeoTIFF/GeoParquet | Parquet + manifest |
| `output.include_patch_table` | Debug/analysis side product | `false` |
| `output.include_vectors` | Separate export DAG | `false` |

### Hashing rules

- `config_hash` is SHA-256 over canonical UTF-8 JSON containing scientifically meaningful settings only: input semantics, validity, CRS/area policy, AOI definition identifier, thresholds, temporal rules, metric profiles, and graph rules.
- Normalize enum spelling, numeric representation, key order, CRS representation, and path separators before hashing.
- Exclude output directories, temporary/checkpoint paths, scheduler address, worker counts, accelerator choice, and human labels from `config_hash` because they must not change scientific meaning.
- Record execution settings separately as `execution_hash` and `execution_config`.
- Record source data separately as `input_fingerprint`; do not hash multi-terabyte data into config. Fingerprint rules must be adapter-specific and documented.
- Golden tests must prove identical scientific configs hash identically across Windows/Linux and changes to any scientific threshold change the hash.

---

## 5. Output schema proposal

### Canonical metrics table

Primary storage is partitioned Parquet. Flattened CSV is an export, not schema authority.

| Group | Columns | Type/constraint |
|---|---|---|
| Schema/run | `schema_version`, `run_id`, `config_hash`, `package_version`, `git_sha` | Non-null strings |
| Spatial identity | `catchment_id`, `aoi_id`, `zone`, `window_id` | Stable strings; `zone` is `AOI`, `channel`, `1`-`4`, or null by metric contract |
| Time | `date`, `hy`, `hy_anchor` | Monthly timestamp; nullable HY/anchor |
| Metric | `metric`, `metric_family`, `statistic`, `value`, `unit`, `value_type` | Registry-controlled metric IDs; nullable float value |
| State/support | `state`, `n_pools`, `n_valid_pixels`, `n_water_pixels`, `valid_fraction_month`, `min_valid_fraction_month` | Nullable typed counts/fractions |
| Quality | `edge_flag`, `warning_flags`, `is_reportable`, `hy_confidence`, `composite_sensitive` | Enum/list/bool; CSV flattens flag list with semicolons |
| Source/spatial | `source`, `resolution_m`, `crs`, `area_unit`, `length_unit` | Mandatory where relevant |
| Input/science config | `monthly_composite`, `water_threshold`, `threshold_method`, `min_patch_pixels`, `min_patch_area_m2`, `connectivity_rule` | Nullable only when structurally inapplicable |
| Dependencies | `metric_dependency`, `proxy_channel`, `awre_length_method`, `node_source` | Registry-controlled values |
| State/reconnection | `connected_wet_metric`, `connected_wet_threshold`, `reconnection_metric_used`, `proxy_reconnection_flag` | Nullable unless associated output enabled |

Metric names are canonical snake case. Dropped metrics are forbidden by schema validation. Distributional metrics emit separate rows using `statistic = mean|median|max|cv|q10|q90`, never compound list values.

`valid_fraction_month` is the canonical monthly support field and replaces the ambiguous draft name `valid_obs_frac`. Schema documentation must record this resolution so downstream consumers do not assume the two names carry distinct quantities.

### Edge and warning semantics

- `N0`: water state is observed dry; patch/configuration values are NaN, not missing rows.
- `N1`: clustering/connectivity and distribution CV are NaN; extent/LPI remain valid.
- `N2_unstable`: only metrics with an approved instability rule are flagged; NNI does not exist in v1.2.
- `low_valid_obs`: value suppression/reportability follows locked validity policy.
- `no_channel`: channel-dependent metric emits an explicit skipped diagnostic, not a proxy core value.
- `composite_sensitive`, `proxy_reconnection`, `width_resolution_floor`, `aoi_not_comparable`, and `length_crs_caveat` are composable warnings.

### Normalised run artifacts

Each run directory contains:

- `metrics/`: partitioned tidy Parquet;
- `run_manifest.json`: versions, input fingerprint, full resolved config, scientific/execution hashes, backend planned/actual by stage, warnings, skipped metrics, timings;
- `config.json`: canonical resolved scientific config;
- `diagnostics/`: compact table/JSON for validation and Dask execution;
- `rasters/occurrence`, `valid_count`, and configured refuge masks for minimal core;
- later optional recurrence, hydroperiod, zone mask, end-dry refuge masks, and graph geometries;
- optional `metrics.csv` flattened export;
- optional retained-metric wide pivot for legacy consumers, clearly labelled non-canonical.

Spatial vector output should prefer partitioned GeoParquet. Shapefile export remains a compatibility option and runs after metric completion from checkpoints.

---

## 6. Implementation milestones

### Milestone 0: Evidence, intake, and binding decisions

**Objective:** Close load-bearing unknowns before scientific behavior is encoded.

**Files likely touched:**

- Create `docs/audit/intake_manifest.md`
- Create `docs/audit/decisions.md`
- Create `docs/audit/evidence/fixture_inventory.md`
- Create `docs/audit/evidence/upstream_validity_contract.md`
- Create `docs/audit/evidence/drainage_inventory.md`
- Create `docs/audit/evidence/regression_baseline.md`

**Tests to write first:**

- Add a repeatable fixture-inspection test/command specification that reports dimensions, CRS, cadence, value domain, wet-fraction variability, and checksum without changing data.
- Compare occurrence/RA under native-only and candidate provenance-qualified validity policies on a real upstream cube; report magnitude, not only formulas.
- Verify whether raw sub-monthly data or both monthly composites exist for validation catchment.
- Validate proposed drainage topology on at least one real catchment if available.
- Trace legacy CSV generator/provenance or formally retire it as a correctness reference.

**Acceptance criteria:**

- Fresh audit manifest complete.
- Decision Gate 0 table has evidence and owner for every row.
- No unresolved assumption is presented as a default.
- Maintainer has recorded predecessor-history decision.
- Approved release scope states exactly which profiles can proceed.

**Rollback risk:** Low. Documentation-only. Failure to complete blocks dependent milestones.

### Milestone 1: Characterisation suite and historical baseline quarantine

**Objective:** Preserve useful current behavior as evidence while preventing legacy schema and denominator defects from controlling v1.2.

**Files likely touched:**

- Modify `tests/conftest.py`
- Split/modify `tests/test_unit_metrics.py`
- Split/modify `tests/test_integration.py`
- Create `tests/legacy/test_legacy_kernels.py`
- Create `tests/fixtures/README.md`
- Create `tests/contracts/test_fixture_characterisation.py`
- Create `docs/testing.md`

**Tests to write first:**

- Fixture path and checksum tests.
- Characterisation tests for current component count, areas, skeleton path, EDT width, APSEC, AWRe, and AWMSI on tiny analytic masks.
- Explicit test that legacy CSV is never loaded by canonical contract tests.
- Tests documenting which current results are invalid because they use total timesteps or dropped formulas.

**Acceptance criteria:**

- Fast and slow suites collect successfully in supported Python environments.
- Historical CSV is limited to smoke comparisons for approved invariant kernels.
- No canonical test requires dropped metric columns or naive `pp_mean_%` equivalence.
- Tiny analytic fixtures cover diagonal connectivity, one-pixel noise, empty/full masks, holes, long bars, and components crossing future chunk boundaries.

**Rollback risk:** Low-medium. Test reorganisation may hide coverage; reviewer must map each removed assertion to replacement evidence.

### Milestone 2: Config, schema, registry, and public-contract freeze

**Objective:** Freeze names, types, hashes, metric dependencies, and output behavior before metric implementation.

**Files likely touched:**

- Create `hydrofragments/config.py`
- Create `hydrofragments/schema.py`
- Create `hydrofragments/models.py`
- Create `hydrofragments/metrics/registry.py`
- Create `tests/contracts/test_config.py`
- Create `tests/contracts/test_schema.py`
- Create `tests/contracts/test_registry.py`
- Create `tests/contracts/test_hashing.py`
- Create `docs/configuration.md`

**Tests to write first:**

- Unknown config keys rejected; required conditional fields enforced.
- Scientific versus execution config separated.
- Cross-platform canonical hash golden tests.
- Exact output columns/types/enums and schema version.
- Forbidden metric IDs (`PF`, `PLF`, `AWMPA`, `AWMPL`, original `AWMPW`, `PCF`, NNI, centrality) rejected.
- Dependency resolution skips unavailable channel/HY/graph metrics with explicit reason.
- N0/N1 and low-valid record construction rules.

**Acceptance criteria:**

- API/config/output design review approved.
- Schema can represent every shipped metric and documented flag without adding metric-specific ad hoc columns.
- Minimal profile resolves to only approved core metrics.
- Config hash is stable and sensitive to all scientific fields.
- No numerical kernel has been expanded yet.

**Rollback risk:** Medium-high. Schema churn affects all later work. Tag schema-contract commit and require migration note for any later change.

### Milestone 3: Input adapters, validity, spatial alignment, and cadence

**Objective:** Build one correct source-agnostic `WaterCube` boundary and eliminate sentinel/CRS ambiguity.

**Files likely touched:**

- Create `hydrofragments/io/adapters.py`
- Create `hydrofragments/io/validity.py`
- Create `hydrofragments/io/alignment.py`
- Create `hydrofragments/spatial/crs.py`
- Create `hydrofragments/spatial/context.py`
- Create `hydrofragments/temporal/cadence.py`
- Create `tests/io/test_watermask_tsfill.py`
- Create `tests/io/test_generic_inputs.py`
- Create `tests/spatial/test_crs_alignment.py`
- Create `tests/temporal/test_cadence.py`

**Tests to write first:**

- Canonical upstream Dataset/Zarr parsing including optional scalar `spatial_ref`.
- `254`/`255` decoded before signed cast; neither counted dry/water.
- Locked validity-policy truth table across observed, filled, post-processed, unresolved, and outside-AOI flags.
- Generic binary/probability input and threshold provenance.
- Shape/transform/CRS/time mismatch raises; no silent resampling.
- Raster, AOI, and drainage co-reprojection ordering.
- Geographic CRS refusal without configured reprojection/per-pixel areas.
- Monthly versus sub-monthly cadence detection and missing-month behavior.

**Acceptance criteria:**

- Real upstream test Zarr validates and remains Dask-backed.
- All input paths produce the same canonical domain object for equivalent data.
- Scientific validity policy is named/versioned in resolved config and manifest.
- CRS/unit metadata complete before any metric computes.
- Input normalisation contains no full-cube eager materialisation.

**Rollback risk:** High. Wrong boundary corrupts every metric. Keep legacy API untouched until this milestone passes real-cube integration.

### Milestone 4: Dask temporal graph, chunk contracts, and explicit checkpoint

**Objective:** Establish genuinely lazy validity/compositing/reduction stages before spatial metric expansion.

**Files likely touched:**

- Create `hydrofragments/compute/policy.py`
- Create `hydrofragments/compute/chunks.py`
- Create `hydrofragments/temporal/composites.py`
- Create `hydrofragments/pipeline.py`
- Create `tests/compute/test_chunk_contracts.py`
- Create `tests/compute/test_laziness.py`
- Create `tests/temporal/test_composites.py`
- Create `tests/integration/test_monthly_checkpoint.py`

**Tests to write first:**

- Storage-aligned inputs retain appropriate chunks.
- Chunk byte budget rejects unsafe layouts and reports estimated live-array multiplier.
- Exact `max_water`, `median`, `mode`, and end-of-month-nearest examples where applicable.
- Already-monthly input is not recomposited and must carry provenance.
- One explicit materialisation boundary; no nested `compute`, `.values`, or `np.asarray` on lazy arrays.
- Reusing checkpoint does not reread/recompute raw observations.

**Acceptance criteria:**

- Valid counts, monthly valid fractions, compositing, and diagnostic reductions stay lazy until pipeline-owned checkpoint.
- Dask graph and chunk diagnostics appear in manifest.
- Current three eager validity scans are replaced by one shared graph/compact diagnostic boundary on new path.
- CPU-only execution works without CUDA packages.

**Rollback risk:** Medium. Performance changes can preserve values but worsen memory. Retain stage benchmarks and old path until acceptance matrix passes.

### Milestone 5: Persistence and fixed-area core reductions

**Objective:** Ship occurrence/RA only after validity approval, plus APSEC and required spatial outputs.

**Files likely touched:**

- Create `hydrofragments/metrics/persistence.py`
- Create `hydrofragments/metrics/extent.py`
- Create `hydrofragments/guards/scientific.py`
- Create `hydrofragments/output/rasters.py`
- Create `tests/metrics/test_occurrence.py`
- Create `tests/metrics/test_refuge_area.py`
- Create `tests/metrics/test_apsec.py`
- Create `tests/guards/test_scientific_guards.py`

**Tests to write first:**

- Uneven valid-observation denominators, zero-valid pixels, and exact `min_valid_obs` boundary.
- RA threshold boundary and threshold sensitivity examples.
- APSEC formula against fixed AOI area, including all-dry/all-wet and clipped AOI.
- Low monthly validity suppression/flag behavior.
- Occurrence never stratified by occurrence-defined zones.
- AOI definition mismatch causes comparison refusal.

**Acceptance criteria:**

- Occurrence equals approved `water_obs/valid_obs`, never total timesteps.
- Occurrence and valid-count rasters plus RA/APSEC tidy rows carry full provenance.
- All-dry state emits informative zero extent and NaN patch values with correct flags.
- No LPSEC or wet-derived `L_ref` appears in minimal core.

**Rollback risk:** High. Validity changes alter all downstream persistence products. Version validity policy and force a schema/run change if policy changes.

### Milestone 6: Exact CPU patch engine and minimal patch metrics

**Objective:** Replace whole-section gufunc morphology with correct global labels plus bounded component crops; activate N, LPI, AWRe, and AWMSI.

**Files likely touched:**

- Create `hydrofragments/patches/labels.py`
- Create `hydrofragments/patches/components.py`
- Create `hydrofragments/patches/morphology.py`
- Create `hydrofragments/metrics/patches.py`
- Create `tests/patches/test_labels.py`
- Create `tests/patches/test_component_crops.py`
- Create `tests/patches/test_morphology.py`
- Create `tests/metrics/test_patch_metrics.py`
- Modify characterised portions of `ecofragments/utils/calc_metrics.py` only when removing duplication is safe

**Tests to write first:**

- Cross-chunk component membership with label-ID normalization.
- 4- versus 8-neighbour behavior.
- `min_patch_pixels=3` applied after global reconciliation.
- `int32` labels on highly fragmented masks.
- Component crop padding parity for EDT/skeleton/region properties.
- Exact N/LPI/AWRe/AWMSI formulas, N0/N1 cases, curved-pool major-axis fallback, and `awre_length_method` metadata.
- No path/geometry object retained when optional vector output is disabled.

**Acceptance criteria:**

- Component membership and core metrics match analytic truth across chunk layouts.
- No full-raster scan per label.
- Component tasks are bucketed toward useful durations and return compact numeric rows.
- AWRe uses major axis in no-channel core and says so; skeleton method activates only with approved real channel context.
- MESH and width distribution remain disabled pending validation/guard milestones.

**Rollback risk:** High. Topology/geometry changes affect scientific values. Preserve analytic parity suite and stage checksums; do not use legacy CSV as oracle.

### Milestone 7: Tidy output, manifests, comparison guards, and export isolation

**Objective:** Complete reproducible core result packaging without recomputation or client-memory growth.

**Files likely touched:**

- Create `hydrofragments/output/tables.py`
- Create `hydrofragments/output/manifest.py`
- Create `hydrofragments/guards/comparison.py`
- Extend `hydrofragments/models.py`
- Create `tests/output/test_tables.py`
- Create `tests/output/test_manifest.py`
- Create `tests/output/test_exports.py`
- Create `tests/guards/test_comparison.py`

**Tests to write first:**

- Exact schema, types, partitioning, nullability, units, and flag composition.
- Manifest contains resolved config, hashes, input fingerprint, versions, skipped metrics, actual backend, and warnings.
- Comparison rejects resolution, source, AOI, validity-policy, and composite mismatches by default.
- CSV flag flattening round-trips.
- Export DAG consumes checkpoints and does not rerun metrics.
- Driver memory does not grow with optional patch geometry when vector export is off.

**Acceptance criteria:**

- One end-to-end core run writes a self-contained reproducible bundle.
- Canonical table contains no forbidden metrics.
- Reopening output validates schema and manifest without source data.
- Export failures do not invalidate or recompute completed metric output.

**Rollback risk:** Medium. Storage layout is externally visible. Version schema and keep a reader migration for prior pre-release outputs.

### Milestone 8: Public namespace, compatibility facade, honest docs, and core release candidate

**Objective:** Expose stable HydroFragments identity after API/schema freeze while giving current users a bounded migration path.

**Files likely touched:**

- Create/modify `hydrofragments/__init__.py`
- Modify `pyproject.toml`
- Modify `ecofragments/__init__.py`
- Modify `ecofragments/main.py`
- Modify `README.md`
- Modify `docs/index.md`
- Modify `docs/architecture.md`
- Modify/quarantine `docs/module1.md`
- Modify/retire `docs/module2.md`
- Create `docs/migration_v1_2.md`
- Create `docs/input_format.md`
- Modify `.github/workflows/ci.yml`
- Create `tests/compat/test_ecofragments_facade.py`
- Create `tests/docs/test_examples.py`

**Tests to write first:**

- `import hydrofragments` and minimal real call.
- Legacy import emits one deprecation warning and maps retained parameters correctly.
- Dropped metric requests fail clearly.
- README/quickstart examples execute in clean environment.
- Package metadata, version, URLs, and extras are consistent.
- CPU-only install passes without GPU packages.

**Acceptance criteria:**

- One public identity and runnable install/import story.
- Compatibility output is clearly non-canonical and never includes dropped metrics.
- README states exact shipped profile and explicit deferrals.
- Docs no longer claim generic terrestrial/urban scope or missing `waterdetect_batch` functionality.
- Core release candidate passes G0-G5 and benchmark smoke gates.

**Rollback risk:** High for users because imports/package names change. Publish deprecation timeline, migration table, and pre-release wheels before final tag.

### Milestone 9: Pixel-temporal and HY/dynamics tranche

**Objective:** Add recurrence/hydroperiod and extent-contraction metrics; obtain HY/season labels and HY-side metrics by calling external package `hydroseason` (sibling repo `../hydroseason`). Do not reimplement HY detection or season algorithms here.

**External dependency:**

- Package: `hydroseason` (installable; development path `D:\RLH\5.6\repos\hydroseason` / sibling `../hydroseason`).
- Consume public API such as `HydroYearConfig`, `detect_hydrological_years`, `label_hydrological_months`, `monthly_water_extent`, and any exported season/stress metrics — map results into HydroFragments temporal labels and dynamics inputs.
- Record `hydroseason` version and passed config in run config/manifest.

**Files likely touched:**

- Extend `hydrofragments/metrics/persistence.py` (recurrence, hydroperiod)
- Create thin `hydrofragments/temporal/hydroyear.py` **adapter** (imports/calls `hydroseason`; no detector logic)
- Create `hydrofragments/metrics/dynamics.py`
- Declare `hydroseason` in `pyproject.toml` (or optional extra) when Q7 closes
- Create `tests/metrics/test_recurrence_hydroperiod.py`
- Create `tests/temporal/test_hydroyear_adapter.py` (integration with `hydroseason`, not algorithm unit tests of detection)
- Create `tests/metrics/test_extent_contraction.py`
- Create `docs/metrics/dynamics.md`

**Tests to write first:**

- Recurrence denominator uses valid years; hydroperiod uses valid observed months.
- Adapter maps monthly series through `hydroseason` and yields HY/season labels; drought/no-clear-anchor / missing-month behaviour matches package contract (mock or pinned fixture — do not fork algorithm tests).
- Tayer-algorithm comparison (V8) targets `hydroseason` outputs / paper positioning, not a local reimplementation.
- Exact contraction slope, sign, units, number of points, missing-anchor behavior, and low-degrees-of-freedom flag.
- Dual-composite 10 percentage-point boundary and `composite_sensitive` propagation.
- Already-monthly single-composite input refuses dual-composite metric rather than fabricating it.

**Acceptance criteria:**

- Q3/U3 and Q7/V8 approved against `hydroseason` contract + version pin.
- No HY/season detection code lives in HydroFragments beyond the thin adapter.
- Metric is named/described as surface-water extent contraction; no flow/recession-constant claim.
- Both composites, slope sample count, and `hydroseason` version/config appear in output/manifest.
- V3 magnitude analysis completed on at least one validation catchment before manager headline use.

**Rollback risk:** High scientific/reputational risk. Feature flag and separate profile allow removal without destabilising core. Optional `hydroseason` extra keeps core install free of HY if needed.

### Milestone 10: Real channel context, zones, and secondary morphology

**Objective:** Add channel-dependent and validated secondary metrics without weakening fixed-denominator rules.

**Files likely touched:**

- Extend `hydrofragments/spatial/context.py`
- Create `hydrofragments/spatial/zones.py`
- Create `hydrofragments/spatial/windows.py`
- Create `hydrofragments/metrics/clustering.py`
- Extend `hydrofragments/metrics/extent.py`
- Extend `hydrofragments/metrics/patches.py`
- Create `tests/spatial/test_channel_context.py`
- Create `tests/spatial/test_zones.py`
- Create `tests/metrics/test_lpsec_gap_width_mesh.py`

**Tests to write first:**

- Drainage topology, CRS, AOI clipping, real `L_ref`, and channel-window stability.
- No-drainage mode emits Zones 2-4 only; Zone 1 absent; no morphology proxy.
- Persistence-by-occurrence-zone requests rejected.
- LPSEC formula and documented >100% braided behavior.
- Inter-pool gap order/run-length truth.
- Width mean/median/max/CV and resolution-floor suppression.
- LPI/MESH correlation gate artifact; MESH disabled or dropped when `r > 0.9` under approved validation rule.
- AWRe skeleton versus major-axis behavior never mixed silently.

**Acceptance criteria:**

- U4/Q6 closed with a real dataset and versioned contract.
- Core without drainage continues unchanged.
- Channel-dependent missing data yields explicit skips.
- Width is never manager-facing without width-not-depth and resolution-floor warnings.
- NNI remains absent.

**Rollback risk:** High. Drainage topology and skeleton algorithms can alter many values. Keep this profile optional through at least one release cycle.

### Milestone 11: Connectivity tranche; DCI is separate decision

**Objective:** Add fixed-node RC/TCF only after graph semantics stabilise; add DCI only after independent reference validation.

**Files likely touched:**

- Create `hydrofragments/metrics/connectivity.py`
- Extend `hydrofragments/metrics/dynamics.py`
- Create `tests/connectivity/test_fixed_graph.py`
- Create `tests/connectivity/test_rc.py`
- Create `tests/connectivity/test_tcf.py`
- Create `tests/connectivity/test_dci_reference.py` only if DCI approved
- Create `docs/metrics/connectivity.md`

**Tests to write first:**

- Stable node source and edge rule across months.
- RC edge fraction and reachable-pair analytic graphs.
- TCF valid-month denominator and chronically isolated/always-connected nodes.
- No transient monthly patch identity.
- RC-versus-DCI relationship documented and tested on linear examples.
- If DCI enabled, parity against `riverconn`/Conefor or independently verified reference fixture.
- Reconnection preference order and proxy flag.

**Acceptance criteria:**

- DCI/PC/IIC positioning appears in docstrings/docs before feature activation.
- Node source, edge rule, thresholds, and actual dependency status appear in every result manifest.
- RC/TCF can be disabled without affecting core schemas or results.
- DCI stays citation-only unless V6/reference parity passes.

**Rollback risk:** High conceptual risk. Separate profile and minor release; never backport silently into core defaults.

### Milestone 12: Optional CUDA backend

**Objective:** Accelerate only certified stages after CPU correctness and performance baselines exist.

**Files likely touched:**

- Create/extend `hydrofragments/compute/capabilities.py`
- Create `hydrofragments/compute/backends/cpu.py`
- Create `hydrofragments/compute/backends/cuda.py`
- Modify `pyproject.toml` optional extras
- Create `tests/compute/test_capabilities.py`
- Create `tests/compute/test_cpu_cuda_parity.py`
- Create `benchmarks/`
- Create `docs/performance.md`

**Tests to write first:**

- CPU-only import/install with no CuPy/RAPIDS.
- `accelerator=cuda` strict failure and `auto` truthful fallback.
- Exact integer/count parity and declared floating tolerance for eligible stages.
- Actual backend recorded stage-by-stage.
- Host-device transfer and VRAM limits included in benchmark output.
- Unsupported skeleton/graph/vector stages stay CPU.

**Acceptance criteria:**

- First enabled CUDA set is limited to sentinel normalization, masks, valid counts, monthly reductions, occurrence, and other certified block reductions.
- No kernel enabled without end-to-end or stage improvement after transfer cost.
- CPU and mixed runs produce scientifically equivalent bundles.
- cuCIM morphology remains experimental/off until adversarial-shape and real-data parity pass.

**Rollback risk:** Medium if isolated, catastrophic if backend logic leaks into metric formulas. Keep central capability registry and optional extra.

### Milestone 13: Validation evidence, manager deliverables, and publication readiness

**Objective:** Turn guarded hypotheses into demonstrated claims where data permits, then complete interpretation and publication-facing material.

**Files likely touched:**

- Create `validation/` analyses and machine-readable result tables
- Create `docs/validation_status.md`
- Create `docs/for-managers.md`
- Create/complete practitioner quickstart and API reference
- Create `CHANGELOG.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, license and templates after governance decisions
- Create `paper/` only when publication gate opens

**Tests to write first:**

- Validation analyses are reproducible from immutable manifests/configs.
- Manager numbers trace to actual validation rows and run IDs.
- Prohibited claims/terms scan: depth/volume inference, recession-as-flow, permanent refuge, false precision, unsupported novelty.
- Docs examples execute; links and docs build pass.

**Acceptance criteria:**

- Validation status table says asserted or demonstrated with linked evidence; no silent deletion of unresolved rows.
- Manager guide leads with negative scope, pairs headline metrics with guardrail metrics, and uses real ranges only.
- V1-V3 are demonstrated before methods-paper claims; V5/V6/V8 completed before associated claims.
- Publication timeline remains separate from v1.2 numerical release and respects public-history decision.

**Rollback risk:** Low for core software, high reputational risk for claims. Publication and manager docs remain independently versioned and reviewable.

---

## 7. Validation plan

### Validation layers

1. **Contract validation:** sentinel truth tables, variable/dtype/domain validation, grid equality, cadence, CRS, config/hash, and schema.
2. **Analytic numerical validation:** tiny masks with hand-calculable occurrence, RA, APSEC, N, LPI, AWRe, AWMSI, later MESH/gap/RC/TCF/DCI.
3. **Chunk invariance:** same result across storage/chunk layouts, including components crossing two and four chunks.
4. **Legacy kernel characterisation:** compare only unchanged low-level areas, perimeters, skeleton paths, and EDT behavior. Do not validate v1.2 occurrence or schema against legacy output.
5. **Real upstream integration:** canonical WaterMask-TSFill Zarr including sentinels/provenance and at least one non-WaterMask generic pair.
6. **Scientific validation:** real Gilbert or approved equivalent catchment, with immutable input fingerprint and run manifest.
7. **Audience validation:** water-manager review of language, paired metrics, warning comprehension, and escalation rules.

### Scientific claim matrix

| ID | Claim | Required evidence | Release consequence |
|---|---|---|---|
| V1 | AWRe and AWMSI are distinct axes | Scatter, correlation, quadrant occupancy | Until shown, both may ship but interpretation stays hypothesis language |
| V2 | LPI and MESH are non-redundant | Pre-registered correlation gate; drop MESH if `r > 0.9` under approved scope | MESH disabled until result |
| V3 | Composite choice affects extent contraction | Dual-composite result with typical disagreement magnitude | No manager/paper headline before result |
| V4 | AWRe indicates drying mode/pool type | Independent pool/drying evidence | No causal/diagnostic claim before result |
| V5 | Width is meaningful above raster floor | Resolution sensitivity and field/bathymetric comparison where possible | Width secondary and guarded; no clean-morphology claim before result |
| V6 | RC/TCF/DCI behave against reference | Direct DCI reference benchmark | DCI runtime disabled until pass |
| V7 | Extent contraction indicates refuge risk | Link to end-dry RA/pool outcome | No predictive claim before result |
| V8 | Persistence-based HY differs from prior rainfall HY | Algorithmic comparison and novelty statement | HY novelty claim blocked; module may remain deferred |

Also run AOI buffer-width sensitivity for APSEC/LPI/MESH, width-versus-resolution tests, classification-error perturbations for edge-sensitive metrics, and cross-sensor sensitivity. Formal error-propagation modelling remains deferred, but unresolved sensitivity must be stated.

---

## 8. Benchmark plan

### Datasets

- **B0 analytic:** 32-512 pixel masks; empty/full, diagonal, chunk-crossing, one-pixel neck, holes, long bars, nodata, multiple observations/month.
- **B1 bundled:** `tests/wmask_ts.nc` plus seven sections after U1 characterisation; smoke/legacy only.
- **B2 synthetic scale:** 24/120/500 times and 2,048/8,192/10,000 pixels per side, generated lazily.
- **B3 fragmentation stress:** equal wet fractions with few large versus many tiny components.
- **B4 AOI stress:** 1/10/100 non-overlapping AOIs plus overlap case.
- **B5 I/O:** equivalent Zarr, chunked NetCDF, and tiled GeoTIFF/COG with aligned/misaligned chunk layouts.
- **B6 real:** approved Gilbert validation inputs, including raw/both composites only if actually available.

### Environments

- CPU reference: single worker/single thread.
- CPU scale: local distributed threads and processes on recorded hardware.
- GPU available: same host with recorded NVIDIA device; optional eligible stages only.
- CPU-only clean install without CuPy, cuCIM, dask-cuda, RAPIDS, or CUDA toolkit.

Each timed case gets one warm-up and at least three measured runs. Record versions, hardware, storage, cache state, scheduler, worker/thread counts, chunks, compression, and backend planned/actual.

### Measurements

- Correctness checksums and metric/flag parity.
- End-to-end and per-stage time, first-result latency, pixel-observations/s, section-months/s.
- Dask task count, graph bytes, median/p95 task duration, scheduler overhead, occupancy, retries, spill, shuffle/transfer.
- Client/worker peak RSS, managed/unmanaged memory, VRAM, host-device transfer.
- Bytes read/written, bandwidth, file opens, read amplification, output count/size.
- Scaling slopes versus pixels, times, AOIs, component count, and skeleton pixels.

### Gates

- CPU-only correctness always passes.
- Label IDs may differ; component membership and count may not.
- Integer/count products exact; floating tolerance declared before runs.
- Skeleton/graph/vector outputs identical between CPU and mixed runs because those stages remain CPU.
- Median heavy task normally exceeds 100 ms and scheduler overhead target is below 10%.
- Peak memory stays below configured limits without avoidable spill; no client accumulation proportional to patch geometry when vectors are off.
- CUDA is enabled only when measured benefit survives transfer cost and parity checks. No promised speedup.

Benchmark output is machine-readable JSON/Parquet plus a short human report committed or attached to release artifacts.

---

## 9. Documentation plan

### Stage A: honesty before completeness

- README status banner, actual import/install path, river-only scope, one real feature list.
- Quarantine missing `waterdetect_batch` docs and mark legacy metric docs historical.
- Explain that current legacy outputs are not v1.2 correctness references.
- Complete citation lineage without asserting new novelty.

### Stage B: contract documentation

- `docs/input_format.md`: canonical upstream schema, generic pairs, sentinel rules, validity policy, grid alignment, probability thresholding, CRS, cadence/composite provenance.
- `docs/configuration.md`: every field/default, hash inclusion, metric profiles, dependency skips.
- `docs/output_schema.md`: tidy table, flags, manifest, spatial products, comparison guards.
- `docs/testing.md`: fixtures, fast/slow markers, benchmark/validation distinctions.

### Stage C: stable user API

- Runnable quickstart only after `analyze` stabilises.
- API reference generated/maintained from real public objects.
- Migration guide mapping legacy parameters and retained metrics; explicit table for dropped metrics and replacements.
- Performance guide accurately states lazy stages, CPU islands, checkpoints, and optional accelerator behavior.

### Stage D: manager interpretation

`docs/for-managers.md` remains short and must contain:

- what HydroFragments measures and does not measure;
- plain-language core glossary;
- paired interpretation patterns, not isolated headline numbers;
- low-validity, composite-sensitivity, source/resolution, length, and width-not-depth warnings;
- real validation ranges only, with run IDs;
- no NNI, raw AWRe/AWMSI headline tiles, invented thresholds, exact drying-date prediction, or causal refuge-risk claim;
- escalation path to analysis team for flagged results.

### Stage E: publication readiness

- Maintain `docs/validation_status.md` as asserted-versus-demonstrated source of truth.
- Complete licensing, contribution, conduct, support, CI, hosted docs, release archive, and AI disclosure only when governance/publication milestones open.
- Methods paper leads with metric-register reformulation; Gilbert is validation, not repeated ecology.
- JOSS/Zenodo work never blocks core correctness or Dask release gates.

---

## 10. Release plan

### Release slices

1. **Internal contract tag:** Decision Gate 0, characterisation suite, config/schema/registry. No public v1.2 claim.
2. **`1.2.0a1` contracts preview:** input, validity, CRS, cadence, Dask temporal graph, schema. For technical adopters only.
3. **`1.2.0b1` contracts+AOI-core:** occurrence/RA after Q1, APSEC, N, LPI, AWRe major-axis fallback, AWMSI, tidy output, manifests. Explicitly excludes LPSEC, HY, contraction, zones, connectivity, and CUDA.
4. **`1.2.0` “contracts + AOI core” release:** only after G0-G5, compatibility/docs, CPU benchmark, and real upstream integration pass. Release notes state this is the adversarially scoped v1.2 profile, not full spec section 5.1 coverage.
5. **`1.2.x` optional profiles:** pixel-temporal, dynamics, channel/zones, and validated secondary metrics land independently when their gates pass.
6. **Connectivity minor release:** RC/TCF after graph review; DCI remains separately gated and may wait for a later minor version.
7. **CUDA extra release:** optional accelerator after benchmark/parity publication; does not change scientific schema or defaults.

### Versioning and compatibility

- Package semantic version and `schema_version` are separate. Any incompatible table change increments schema major version even within package development.
- Support reading immediately prior pre-release schema only when transformation is lossless.
- Deprecate `ecofragments.calculate_metrics` for at least one documented minor cycle; removal requires a major package version unless project governance chooses a clean break before first stable HydroFragments release.
- Never reintroduce dropped metrics to satisfy compatibility. Provide formulas/replacements in migration guide.
- Changelog separates scientific changes, schema changes, performance changes, and documentation claims.

### Release gates

- G0/G0+: evidence and decisions complete.
- G1/G1+: sentinel, validity, CRS, fixed denominators, edge semantics, and baseline quarantine pass.
- G2: schema/config/hash stable.
- G3: contract/core/CPU-only tests pass.
- G4: Dask materialisation and compute claims are truthful.
- G5: docs/install/import and audience claims are honest.
- G6: required only for dynamics/connectivity profiles.
- G7: history decision recorded before publication-lineage claims.

Every release candidate includes resolved config, benchmark summary, validation-status snapshot, known limitations, and explicit deferrals.

---

## 11. Explicit deferrals

### Deferred from minimal `1.2.0` core

- LPSEC until real drainage `L_ref` exists.
- Persistence-based zones and Zone 1 until validity/drainage contracts pass.
- HY detection/season mapping (owned by `hydroseason`) and all HY-derived metrics until Q7/V8 close against that package.
- Surface-water extent contraction until both composites/raw inputs exist and V3 begins.
- Recurrence/hydroperiod until validity/year denominators stabilise.
- MESH until LPI redundancy gate.
- Pool width distribution until resolution-floor guard and V5 plan.
- Inter-pool gap until real channel reference.
- Reconnection and refuge stability until HY/channel prerequisites.
- RC and TCF until fixed-node/edge-rule design.

### Optional later work

- Runtime DCI; citation/positioning is mandatory, implementation is not.
- CUDA for pixelwise/reduction stages.
- cuCIM label/EDT/region properties after parity and benefit.
- Distributed morphology beyond exact CPU component crops.
- Large vector export redesign and GeoParquet optimisation.
- Probabilistic-mask calibration beyond threshold recording and validation.
- Windowed catchment analysis beyond core AOI support.

### Cut from v1.2

- PF, PLF, AWMPA, AWMPL, original area-weighted AWMPW.
- NNI.
- Degree and betweenness centrality.
- Morphology-proxy Zone 1.
- Transient pool identity tracking, lineage, merge/split histories, per-pool recession/survival models.
- Full PC/IIC reimplementation.

### Publication and polish deferrals

- Finished manager narratives with numeric thresholds before validation.
- Claims that AWRe diagnoses pool type/drying mode before V4.
- Claims that contraction predicts refuge risk before V7.
- Methods-paper results before V1-V3.
- JOSS/Zenodo submission until stable API, public-history/governance requirements, licensing, hosted docs, and validation are satisfied.
- Formal classification-error propagation model; qualitative sensitivity and warnings remain required.

---

## 12. Plan self-review

- **Spec coverage:** all locked input, schema, guard, core/secondary/connectivity, Dask, manager, documentation, validation, and publication requirements map to milestones or explicit deferrals.
- **Audit precedence:** second adversarial synthesis controls minimal scope; careful Dask audit controls compute/CUDA choices.
- **Scientific ordering:** validity, sentinel, CRS, config, schema, and tests precede occurrence or metric expansion.
- **Compute ordering:** temporal Dask graph precedes patch engine; CPU reference precedes CUDA.
- **Compatibility:** useful kernel regressions retained; legacy table cannot control v1.2 or restore dropped metrics.
- **No hidden proxies:** no wet-derived core `L_ref`, morphology Zone 1, LPI-only reconnection without proxy flag, or invented second composite.
- **Audience honesty:** manager and paper claims stay below validation evidence.
- **No placeholders:** unresolved decisions are named blockers with required evidence, owners, and affected milestones rather than silent defaults.

Approval of this plan authorises planning only. Implementation begins with Milestone 0 and stops at every unmet gate.
