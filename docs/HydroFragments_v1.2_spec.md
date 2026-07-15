# HydroFragments v1.2 — Metric Specification & Refactor Guide

**Status:** Locked for implementation. Revision incorporating external audit (2026-07-09).
**Supersedes:** v1.1 (`HydroFragments_v1_spec_reviewed_v1_1.md`)
**Formerly:** iRivermetrics → EcoFragments → **HydroFragments**
**Scope:** AOI- and catchment-scale surface-water extent, persistence, fragmentation, clustering, and connectivity metrics computed from **binary (or thresholded probabilistic) water-mask time series from any source** — WOfS, Water Detect on Sentinel-2/PlanetScope, thresholded NDWI/MNDWI, or any other classified water raster. WOfS is the reference/default data source used in development and validation, not a hard requirement — every metric operates on a binary mask plus a valid-observation count, and is agnostic to how that mask was produced.
**Domain scope (deliberate):** river / surface-water systems specifically. Not intended as a generic patch-metrics tool for vegetation, urban, or other land-cover domains — see §0.
**Explicitly out of scope for v1 (architectural):** pool-unit-level identity tracking through time (lineage, per-pool recession curves, per-pool survival models). Deferred to a later release.
**Newly deferred from v1 (audit-driven, not architectural — see §5.5):** node centrality (degree/betweenness), morphology-proxy Zone 1.
**Supersedes:** internal `iRivermetrics` metric suite (Tayer et al. 2023a, 2023b, 2025/2026).

---

## How to read this document

This revision was produced by walking a critical audit of v1.1 line by line and resolving every actionable finding into a locked decision, a renamed/repositioned metric, an added citation, or an explicit deferral — rather than leaving them as open commentary. Two conventions:

- **`[AUDIT FIX]`** marks a place where v1.1 is changed as a direct consequence of the audit.
- **`JUDGEMENT CALL:`** marks a place where the audit identified a genuine choice rather than a determinate fix. The recommendation given is a reasoned default, not a settled fact — revisit if the refactor surfaces new information.

This document is written to serve two downstream uses directly:
1. **Repository refactor** — §§1–9, 11, 13, 14 are the implementation contract.
2. **Paper scoping** — §10 and §16 define what a companion methods paper can defensibly claim, and how it stays clear of Tayer et al. (2025).

---

## 0. Why river-focused, not a generic patch-metrics tool

Unchanged from v1.1. The individual patch-shape/fragmentation metrics (LPI, MESH, NNI, AWMSI, AWRe) are domain-agnostic — FRAGSTATS-family metrics are already used on vegetation, urban growth, sea ice, anything with binary/categorical patches. Genericising the tool around them would add nothing novel.

What makes HydroFragments valuable — and publishable — is the water-specific interpretation layer that a generic tool cannot have:
- **Zones** are hydro-geomorphic (in-channel / persistent off-channel / seasonally flooded / marginal floodplain) — meaningless for urban or vegetation patches.
- **Hydrological-year (HY) detection** is a wet/dry seasonal-cycle concept specific to flow regimes, not phenology.
- **Refuge Area, dry-down rate, reconnection timing** are aquatic-ecology concepts with no cross-domain analogue.
- **The circularity critique** that eliminated PF/PLF/AWMPA/AWMPL is specific to how water fragments longitudinally along a channel.

A tool trying to serve water + vegetation + urban domains at once ends up serving none well and has no coherent publishable narrative. Domain focus is retained as a design decision, not a limitation.

---

## 1. Design principles (locked)

Unchanged from v1.1 — these held up under audit.

1. **Monthly cadence is the base unit.** Every metric is computed at every available monthly time step. The full monthly series feeds downstream Bayesian modelling and prediction. Hydrological-year (HY) anchors are an *aggregation view* on top of the monthly series, never a separate computation path.
2. **Fixed denominators only.** No metric whose denominator is composed of the same water features as its numerator. This is why PF, PLF, and the original area-weighted AWMPA/AWMPL/AWMPW formulations are dropped or reformulated — see §4. Normalising quantities must be spatially fixed references (AOI area, channel length, landscape area) — or, for distributional metrics, simply left unweighted (see §6.9).
3. **Zone stratification is an optional layer, not a redefinition.** The same metric definitions apply whether run AOI-wide or per zone. Zones stratify the spatial domain; they do not change formulas.
4. **Graceful degradation without a drainage layer.** Everything must run with persistence-derived zoning alone. Metrics that structurally require a drainage layer or a fixed pool-node set are flagged and skipped (not faked) in fallback mode.
5. **Leanness over coverage.** Every metric must earn its place by adding non-redundant information. Redundant, trivial, or geometrically meaningless metrics are dropped or demoted to exploratory.
6. **Uncertainty propagates.** Valid-observation counts, HY-detection confidence, and edge-case flags travel with every metric value so the Bayesian layer can treat missingness and low-confidence correctly rather than as noise.
7. **Source-agnostic input.** All metrics accept any binary (or pre-thresholded probabilistic) mask time series with an accompanying valid-observation layer. WOfS-specific defaults (e.g. 16-day nominal revisit, `frequency` band naming) are configuration, not architecture.

---

## 1.1 Implementation decisions (locked)

These decisions affect reproducibility, testing, and cross-catchment comparability. Items 1, 2, 7, and 8 are revised from v1.1 per audit; items 9–12 are new.

1. **Equal-area geometry is mandatory for metric computation — and equal-area is not the same guarantee as equidistant.** `[AUDIT FIX]` All areas and lengths must be computed in a projected CRS suitable for the AOI. If the input is geographic coordinates, the pipeline must either reproject to a configured equal-area CRS or use an explicit per-pixel area array. Output must record `crs`, `area_unit`, and `length_unit`.
   - Default for Australian deployments: **EPSG:3577 (Australian Albers)** — this matches DEA's own equal-area convention and should be the config default, not left as "a configured equal-area CRS" with no example.
   - **Caveat that must be documented, not just satisfied:** an equal-area CRS minimises area distortion but does not minimise *length* distortion. `LPSEC`, inter-pool gap distances, and skeleton length are length quantities computed in an area-optimised projection. This is negligible at reach scale and non-trivial at catchment scale spanning several degrees of longitude. Document this explicitly in user-facing docs; do not imply equal-area solves length metrics too.

2. **Monthly compositing must be explicit and recorded — and `max_water`'s bias on dry-down and refuge metrics must be actively managed, not just documented.** `[AUDIT FIX]` HydroFragments accepts any water-mask source, but monthly masks must be produced by a documented rule.
   - **`max_water` remains the default** for general monthly extent series (APSEC, LPI, N, MESH, AWRe, AWMSI) — a pixel is water in a month if it was confidently water in at least one valid observation. This is retained for continuity with iRivermetrics precedent and because it is the conservative choice for detecting *any* wet presence.
   - **Audit finding, now a locked requirement:** `max_water` is maximally sensitive to a single false-positive water observation within a month, and systematically biases occurrence, Refuge Area, and APSEC upward. This directly flattens and delays the **dry-down rate** signal — the project's headline metric — because it inflates the apparent extent at the point where the mask should be most contracted.
   - **Mitigation (locked for v1):** for **end-dry anchor detection and dry-down rate specifically**, compute and report a secondary composite (default: `median`) alongside `max_water`. Flag disagreement between `max_water`-derived and `median`-derived end-dry APSEC beyond a configurable tolerance (default: 10 percentage points) as `composite_sensitive` on the affected HY-anchor rows (see §7.1, §8 guard 11).
   - Alternatives (`median`, `mode`, `end_of_month_nearest`) remain allowed generally and must be written to output metadata regardless of which is used.

3. **Raster connectivity rule is locked.** Default patch delineation uses **8-neighbour connectivity** to match the existing iRivermetrics/Tayer-style pool delineation. A 4-neighbour option can exist for sensitivity analysis, but the output must record `connectivity_rule`.

4. **Minimum mapping unit is mandatory.** Default `min_patch_pixels = 3` for Landsat/WOfS-class 30 m products, configurable by source. Patches below the threshold are removed or flagged before patch metrics are calculated. Output must record `min_patch_pixels` and `min_patch_area_m2`.

5. **Monthly validity and per-pixel validity are separate.** Use `min_valid_obs` for per-pixel long-term occurrence estimates, and `min_valid_fraction_month` for whether a monthly AOI/zone metric is reportable. These should not share one parameter name.

6. **All binary thresholds must be recorded.** For probabilistic water masks, patch/configuration metrics require thresholding. Output must include `water_threshold`, `threshold_method`, and `probability_source` where applicable.

7. **Configuration is part of the scientific result.** Every run should emit a machine-readable config file and a config hash. Recommended output columns: `run_id`, `config_hash`, `package_version`, `git_sha`, `source`, `resolution_m`, `crs`, `water_threshold`, `min_patch_pixels`, `connectivity_rule`, and `monthly_composite`. This reproducibility discipline is a genuine strength of the design — it directly satisfies criteria JOSS now weighs heavily (see §13). Foreground it in docs and in the companion paper.

8. **State flags should be emitted alongside metrics — and must name which connectivity definition they use.** `[AUDIT FIX]` For each AOI/zone/month, emit a simple hydrological state: `dry` (`WA=0`), `fragmented_wet` (`WA>0` and connectivity below threshold), or `connected_wet` (connectivity above threshold). v1.1 left "connectivity" undefined here, which makes the label irreproducible. **Locked:** the config must declare `state_flag_connectivity_metric ∈ {RC, LPSEC, LPI, DCI}` and `state_flag_connectivity_threshold`, and both must be recorded in output metadata (`connected_wet_metric`, `connected_wet_threshold`). This is a QA/modelling label, not an extra metric family.

9. **No-drainage channel proxies must be opt-in.** The draft allows a maximum-wet-extent skeleton fallback for `L_ref`. That is useful but should be flagged as `proxy_channel`. For publication-grade longitudinal metrics, prefer `LPSEC`, dry gaps, and channel NNI only when a real drainage layer or validated channel skeleton is available. *(Note: "channel NNI" here is now superseded — see §5.3/§6.8; inter-pool gap is the metric of record.)*

10. **Graph metrics should be split into safe snapshot metrics and fixed-node metrics.** Snapshot realised connectivity can be implemented now without pool-unit tracking. Node centrality is **cut from v1 entirely** (see §5.5) — it is not merely deferred as "fixed-node output," it is not built at all in v1.

11. **`[NEW]` Connectivity naming and literature positioning is locked.** The audit found that v1.1's connectivity module reinvented, without citation, a large existing literature: the **Dendritic Connectivity Index (DCI; Cote, Kehler, Bourne & Wiersma 2009, *Landscape Ecology*)**, the **Integral Index of Connectivity / Probability of Connectivity (IIC/PC; Pascual-Hortal & Saura 2006; Saura & Pascual-Hortal 2007)**, and the existing software **`riverconn`** (Baldan et al. 2022, *Environmental Modelling & Software*) and **Conefor** (Saura & Torné 2009). Two consequences, both locked:
    - **Naming.** `PCF` is renamed **`TCF` (Temporal Connectivity Frequency)** throughout the codebase, schema, and docs. `PCF` is retired — it is one letter from `PC` (Probability of Connectivity), the most-used graph connectivity index in the field, and using it invites exactly the wrong association.
    - **Positioning.** Every place the spec or docs describe realised connectivity (RC), reconnection timing, or TCF must state explicitly how the metric relates to DCI/PC/IIC, and cite them. Silence on this point is the single most exposed weakness the audit found in the whole document (see §6.11).

12. **`[NEW]` Morphology-proxy Zone 1 is cut from v1, not merely discouraged.** v1.1 listed it as "experimental... not recommended for publication without validation," which left it implemented but gated by a warning label. **Locked change:** it is not implemented in v1 at all. If a drainage layer is unavailable, Zone 1 is simply not emitted (persistence-proxy fallback only — §3). Revisit in v2 only alongside a real validation study.

---

## 2. The spatial normalising unit

**The normalising unit is a user-supplied AOI polygon** (catchment boundary, reach-plus-buffer, or arbitrary AOI). All extent and configuration metrics normalise against quantities derived from this polygon and, where available, the drainage layer:

| Quantity | With drainage layer | No-drainage fallback |
|---|---|---|
| **Reference area** (`A_ref`) | AOI polygon area | AOI polygon area (unchanged) |
| **Reference length** (`L_ref`) | Channel centreline length within AOI | Long-axis length of the maximum-wet-extent skeleton within AOI |
| **Landscape area** (`A_total`, for MESH/LPI) | AOI polygon area | AOI polygon area (unchanged) |

**Optional windowed sectioning.** For within-AOI spatial variation, the AOI may be split into fixed-length channel windows (default 5 km, configurable) along the centreline, or into a regular grid in no-drainage mode. Metrics are then computed per window as well as AOI-wide.

**`[AUDIT FIX]` CRS distortion caveat for length quantities.** `A_ref`, `A_total` are areas and are well-served by the mandatory equal-area CRS (§1.1.1). `L_ref`, gap distances, and skeleton lengths are *length* quantities, which an area-optimised projection does not guarantee to preserve, especially for AOIs spanning a wide longitude range. For catchment-scale runs, note the projection's length-distortion characteristics in the output metadata (`crs`) and consider a catchment-appropriate conformal or equidistant alternative if `L_ref`/gap comparisons across a wide AOI are load-bearing for a specific analysis. This is a documentation and QA requirement, not a blocking guard — equal-area remains the default.

**Decision rule for the user:**
- Catchment-scale comparison → run AOI-wide (one value per catchment per month).
- Within-catchment spatial pattern → enable windowed sectioning.
- Both are just different `groupby` domains over the same monthly rasters.

---

## 3. Zone configuration schema

Static four-zone mask, defined once from the long-term RAW persistence surface (any binary mask source) plus (optionally) a drainage layer. **Thresholds are user-configurable; defaults below.**

| Zone | Name | Default definition | Drainage layer required? |
|---|---|---|---|
| 1 | In-channel | Drainage channel mask + adjacent high-frequency pixels | **Yes** (else Zone 1 is not emitted — see below) |
| 2 | Persistent off-channel | RAW freq > `t_persist` (default 50%), spatially isolated from channel | No |
| 3 | Seasonally flooded floodplain | RAW freq `t_season`–`t_persist` (default 10–50%) | No |
| 4 | Marginal / extreme-event floodplain | RAW freq < `t_season` (default 10%), within max wet extent | No |

```python
zone_config = {
    "t_persist": 0.50,      # Zone 2 lower bound / Zone 3 upper bound
    "t_season": 0.10,       # Zone 3 lower bound / Zone 4 upper bound
    "min_valid_obs": 20,    # per-pixel floor for a valid frequency estimate
    "drainage_layer": None, # path or None -> triggers fallback
}
```

**No-drainage fallback for Zone 1 — `[AUDIT FIX]` single option only.** When `drainage_layer is None`, Zone 1 cannot be defined geomorphically. v1.2 implements **one** fallback:
- **Persistence proxy (only option in v1):** treat Zone 2 (freq > `t_persist`) as the in-channel + persistent surface collectively; Zone 1 is not emitted; metrics that require Zone 1 are skipped with a flag.

The v1.1 "morphology proxy" option (approximating Zone 1 from elongated, narrow, high-frequency patches) is **removed from v1** per §1.1.12. It is unvalidated and was gated by a warning in v1.1 rather than actually excluded — that gate is now a hard cut. Revisit only alongside a dedicated validation study.

**Circularity guard.** Zones are defined from the RAW persistence surface. Therefore **persistence metrics (occurrence frequency, Refuge Area) must be reported AOI-wide or against the drainage layer — NOT stratified by Zone 2/3/4** (that would be circular: defining a zone by frequency then reporting frequency within it). Zones stratify **configuration** metrics only (fragmentation, morphology, clustering, connectivity). This separation is enforced in the API.

---

## 4. Metric register — keep / drop / add / rename

`[AUDIT FIX]` throughout: AWRe adaptation-level language corrected, NNI demoted, PCF renamed and repositioned, DCI/PC/IIC added as citation-anchors (and optionally implemented — see §6.11), centrality moved from "exploratory" to "cut."

| Metric | Status | Adaptation level | One-line definition | Why |
|---|---|---|---|---|
| Occurrence frequency (per pixel) | **Keep (replaces Pixel Persistence)** | Verbatim (Pekel et al. 2016, *Nature*; Mueller et al. 2016, DEA WOfS) | water obs / valid obs, per pixel | Standard, comparable, replaces bespoke PP. **`[AUDIT FIX]`** explicitly cite Pekel/Mueller — this is a JRC-GSW/DEA-WOfS-style layer, not a novel construction. |
| Refuge Area (RA) | **Keep** | Minor adaptation | area of pixels with occurrence > `t_refuge` (default 90%) | Direct refuge magnitude |
| APSEC | **Keep** | Verbatim (Tayer) | wetted area / `A_ref` | Extent, fixed denominator |
| LPSEC | **Keep** | Verbatim (Tayer) | wetted length / `L_ref` | Longitudinal extent, fixed denominator |
| **AWRe** | **Keep — Core** | **`[AUDIT FIX]` Adapted from Schumm (1956)** — Schumm's elongation ratio is a *basin morphometry* index; repurposing it to a pool population and area-weighting it is HydroFragments' own construction, not a verbatim reuse | area-weighted elongation ratio | Elongation/compactness signal; robust to edge noise; diagnostic of drying-recession mode and pool type |
| AWMSI | **Keep — Secondary** | Verbatim (McGarigal & Marks 1995) | area-weighted mean shape index | Boundary complexity/crenulation — orthogonal to elongation, but more edge-noise-sensitive |
| Number of pools (N) | **Keep** | Verbatim | count of connected water components | Fundamental count |
| **LPI** | **Keep — Core** | Verbatim (McGarigal & Marks 1995) | largest patch area / `A_total` × 100 | Non-circular fragmentation; dominance of largest waterbody |
| **MESH** | **Keep — Secondary** | Minor adaptation (Jaeger 2000) | Σ(aᵢ²) / `A_total` | Fragmentation across full patch-size distribution. Hard gate unchanged: drop if `r(LPI, MESH) > ~0.9` on validation data. |
| **Dry-down rate** | **Keep — Core** | Novel (in this context) | slope of APSEC over recession limb per HY, dual-composite checked (§1.1.2) | Stress/contraction signal — the key refuge-risk indicator |
| **Reconnection timing** | **Keep — Secondary** | Novel | lag from dry-minimum to network re-merge, preferring RC/LPSEC/DCI over LPI-only | Wet-season reconnection |
| **Refuge spatial stability** | **Keep — Secondary** | Minor adaptation (Jaccard 1912) | inter-annual overlap of end-dry >`t_refuge` footprint | Whether refuges are fixed or migrating |
| **Pool width distribution** | **Keep — Secondary** | Novel (non-circular reformulation); EDT method per Pavelsky & Smith (2008, RivWidth) / Yang et al. (2020, RivWidthCloud) | unweighted mean/median/max per-pool width via EDT | Morphological confinement/shape signal — NOT a depth proxy |
| **Inter-pool gap** | **Keep — Secondary, metric of record for clustering** | Verbatim/adapted (waterhole-spacing literature: Sheldon et al. 2010; Fullerton et al. 2010) | mean dry-gap between pools along channel/skeleton | Correct 1D geometry for rivers; directly ecological |
| **NNI (Clark–Evans)** | **`[AUDIT FIX]` Demoted — Exploratory only, planar AOI, not publication-grade for river fragmentation** | Verbatim (Clark & Evans 1954; Donnelly 1978 edge correction) | pool-spacing clustering vs 2D CSR | 2D CSR is the wrong null for a quasi-1D river corridor; unstable below N≈8–10, which is exactly the end-dry regime that matters most. Gap (above) is the metric of record wherever a skeleton exists; NNI is retained only as a planar-AOI fallback with no skeleton, and is barred from publication-grade fragmentation claims. |
| **TCF (Temporal Connectivity Frequency)** — formerly PCF | **Keep — Secondary (zone-dependent)** | Novel, but must be positioned against DCI/PC/IIC (§1.1.11, §6.11) | fraction of months a fixed pool-node has ≥1 connection | **`[AUDIT FIX]` Renamed from PCF** to avoid collision with PC (Probability of Connectivity, Saura & Pascual-Hortal 2007). The genuine novelty is *temporal frequency over a monthly series*, which static DCI/PC do not give — say so explicitly wherever this metric is documented. |
| **DCI — Dendritic Connectivity Index** | **`[NEW]` Add — see §6.11a; JUDGEMENT CALL on implementation vs citation-only** | Adapted from Cote et al. (2009, *Landscape Ecology*); intermittent-river zero-flow-fragment application per PNAS 2025 (doi:10.1073/pnas.2421046122) | fraction of network reachable under current fragmentation, reduces to reach-length-weighted node "size" on a linear/dendritic graph | Positions RC/TCF/reconnection against the field's standard connectivity index instead of leaving them unanchored |
| PF (Pool Fragmentation) | **Drop** | — | N / wetted area | Circular denominator; ≈ 1/mean area |
| PLF (Pool Long. Fragmentation) | **Drop** | — | N / wetted length | Circular denominator; ≈ 1/mean length |
| AWMPA (area-weighted mean pool area) | **Drop** | — | Σ(aᵢ²)/Σaᵢ | Algebraically related to MESH/APSEC (`AWMPA = MESH / APSEC` after unit harmonisation); recoverable, not invalid, just redundant |
| AWMPL (area-weighted mean pool length) | **Drop** | — | Σ(lᵢaᵢ)/Σaᵢ | Less stable than LPSEC + gap/wet-run metrics |
| AWMPW (original) | **Drop, replaced** | — | Σ(wᵢaᵢ)/Σaᵢ | Same circular-weighting pattern as PF/PLF; replaced by unweighted **Pool width distribution** (§6.9) |
| Connected-components count | **Drop** | — | count of graph components | Identical to raster N unless graph edges differ from raster contiguity |
| Largest-component fraction | **Drop, for leanness — not identity** | — | largest comp / total water area | `max(aᵢ)/Σaᵢ` (dominance within water) is *not* the same quantity as `max(aᵢ)/A_total` (LPI, absolute extent); they correlate but are not identical. Excluded for leanness, not because it duplicates LPI. |
| Degree / betweenness centrality | **`[AUDIT FIX]` CUT FROM V1** | Verbatim (graph theory) | node centrality on pool graph | v1.1 marked this "exploratory, catchment-only," which left it implemented behind a soft label. Audit verdict: make the cut hard. Trivial on linear reaches; meaningful only on dendritic multi-tributary networks HydroFragments doesn't yet target; overlaps DCI/PC territory better served by adopting the standard index (see DCI row above) than hand-rolling a graph-theory metric. Not built in v1 at all — see §5.5. |

### 4.1 Review corrections to the metric register

Carried forward from v1.1, plus one addition:

| Item | Correction |
|---|---|
| **AWMPA / AWMPL / AWMPW language** | Do not describe all area-weighted means as "invalid" or strictly circular. Area-weighted summaries are standard. The issue is redundancy and interpretation: AWMPA is algebraically related to MESH and APSEC (`AWMPA = MESH / APSEC` after unit harmonisation), AWMPL is less stable than LPSEC + gap/wet-run metrics, and AWMPW is better represented as an unweighted width distribution. |
| **Fixed denominator principle** | Clarify that the fixed-denominator rule applies to abundance, fragmentation, and connectivity ratios. Shape summaries such as AWRe and AWMSI may remain area-weighted descriptive statistics, provided they are not interpreted as fixed-area fragmentation indices. |
| **Largest-component fraction** | Do not state that largest-component fraction is approximately LPI by definition. `max(a_i)/sum(a_i)` measures dominance within water; `max(a_i)/A_total` measures absolute largest-patch extent. They may correlate but are not the same. Keep it excluded for leanness, not because it is identical. |
| **Connected-components count** | Raster connected-component count is identical to `N`. Graph connected-components are not necessarily identical if edges are defined by channel adjacency, threshold gaps, or fixed-node reachability. Keep raster `N`; add graph components only inside the connectivity module if graph edges are used. |
| **Reconnection timing** | Avoid defining reconnection from `LPI >= 80%` alone. High LPI can reflect one large off-channel/floodplain patch rather than longitudinal river reconnection. Prefer `RC >= t_reconnect`, `LCC >= t_lcc`, `LPSEC >= t_lpsec`, or `DCI` recovery depending on available channel/graph data. |
| **TCF (formerly PCF)** | `[AUDIT FIX]` Define TCF only for fixed nodes derived from a long-term refuge/persistence layer or fixed channel windows. Do not compute TCF on transient monthly patches; that would reintroduce pool-unit tracking. Always cite and distinguish from PC (Probability of Connectivity). |
| **Recurrence / seasonality** | Add these as secondary pixel-temporal outputs. They are source-agnostic, do not require pool tracking, and are explicitly JRC/DEA-GSW-style layers (§6.12) — not a novel construction. |
| **Realised connectivity (RC)** | Add as a v1 connectivity metric. It is safer than node centrality because it can be computed from fixed channel windows or fixed refuge nodes without asserting dynamic pool identity. Note its structural relationship to DCI (§6.11a) when node "size" is reach length. |
| **`[NEW]` Connectivity module positioning** | Every connectivity metric (RC, TCF, reconnection timing) must, in both code docstrings and any paper text, name its relationship to DCI (Cote et al. 2009) and PC/IIC (Saura & Pascual-Hortal 2006, 2007). This is a documentation requirement enforced at review, not merely a suggestion — see §1.1.11. |

---

## 5. Recommended metric framework

### 5.1 Core set (robust, interpretable, publishable, feasible now)
Run AOI-wide and optionally per window. All computable from a thresholded binary mask + AOI polygon.

- **Occurrence frequency** (pixel) → summarised to AOI mean
- **Refuge Area** (RA)
- **APSEC** — wetted area fraction
- **LPSEC** — wetted length fraction
- **Number of pools (N)**
- **LPI** — largest patch index
- **AWRe** — elongation ratio (drying-mode / pool-type signal)
- **Dry-down rate** — contraction slope of APSEC per HY, dual-composite checked

### 5.2 Secondary set (ecological / analytical depth)
- **AWMSI** — boundary complexity (orthogonal to AWRe, pending empirical check — checklist item)
- **MESH** — full-distribution fragmentation (pending redundancy check against LPI)
- **Pool width distribution** — unweighted morphological confinement signal
- **Inter-pool gap** — metric of record for spatial clustering
- **Reconnection timing** — preferring RC/LPSEC/DCI
- **Refuge spatial stability** (inter-annual Jaccard, end-dry footprint version)
- **TCF** — temporal connectivity frequency (Zone 2 fixed nodes; skipped in no-drainage/no-persistent-pool mode)
- **DCI** — see §6.11a; may be Core or Secondary depending on the implementation-vs-citation judgement call

### 5.3 Exploratory (limited applicability, not publication-grade as-is)
- **NNI** — planar-AOI fallback only, when no channel skeleton exists. `[AUDIT FIX]` demoted here from Secondary; do not use for publication-grade fragmentation claims.

### 5.4 Avoid / treat cautiously
- **PF, PLF, area-weighted AWMPA/AWMPL/AWMPW** — circular; use LPI/MESH/width-distribution instead.
- **Any cross-sensor comparison** of patch-count-dependent metrics (N, NNI, gap, MESH tails, width distribution) — see §8 resolution caveat.
- **Zone-stratified persistence metrics** — circular against the zone definition.
- **Treating pool width as a depth proxy** — width indicates morphological confinement (narrow/wide), not water depth; a pool can be wide and shallow. Do not conflate the two in interpretation (see §6.9).
- **`[NEW]` Bespoke connectivity indices built without positioning against DCI/PC/IIC** — see §1.1.11.

### 5.5 `[NEW]` Deferred from v1 (audit-driven — not implemented, not exploratory)

Distinct from the architectural "explicitly out of scope" list at the top of this document (pool-unit tracking), these were implemented-but-gated in v1.1 and are now cut outright pending further work:

- **Degree / betweenness centrality.** Cut entirely — see §4 table. Revisit only if/when a genuinely dendritic multi-tributary network use case arises, and prefer adopting DCI/PC/IIC machinery over hand-rolled centrality at that point.
- **Morphology-proxy Zone 1.** Cut entirely — see §3, §1.1.12. Revisit only alongside a dedicated ground-truth validation study.

---

## 6. Per-metric detail (additions and structurally-affected metrics)

The retained verbatim metrics (APSEC, LPSEC, occurrence frequency, RA, N) follow their published definitions unchanged; formulas in §6.17.

### 6.1 AWRe — Area-weighted Elongation Ratio (adapted from Schumm 1956)

- **Definition:** `AWRe = Σ_i[(2·√(a_i/π) / l_i) · (a_i / Σa_i)]` — ratio of the diameter of a circle with the same area as the pool to the pool's length `l_i`.
- **`[AUDIT FIX]` Length definition is now locked, not left as an open choice.** v1.1 allowed `l_i` to be either the `regionprops` major-axis length or the channel-skeleton length without specifying which, despite noting they diverge sharply for curved pools (a bent channel pool has a short major axis but a long skeleton, giving opposite elongation readings for the same pool). **Locked default:** use **skeleton-based length** whenever a channel skeleton is available (real drainage layer or validated skeleton, not the proxy-channel fallback); fall back to **regionprops major-axis length** only when no skeleton exists. `JUDGEMENT CALL:` this default favours the more geomorphically accurate measurement over the cheaper one; if performance at scale becomes a constraint, major-axis-only may be revisited, but the two methods must never be silently mixed within one run. Record which was used per pool/run as `awre_length_method ∈ {skeleton, major_axis}`.
- **Measures:** elongation vs compactness of pools, independent of boundary crenulation.
- **Interpretation:** low AWRe = elongated (typical of in-channel pools, especially late in recession as they contract along the channel axis into slivers); high AWRe (→1) = compact/circular (typical of off-channel or floodplain refuges). Tracks drying-recession mode and discriminates pool type.
- **Input:** binary mask, patch length (major axis or skeleton length per the locked rule above).
- **Scale:** AOI or window; monthly.
- **Calculation:** `skimage.measure.regionprops` (major axis length) or channel-skeleton-based length; area-weighted aggregation across pools.
- **Strengths:** more robust to pixel-edge/classification noise than AWMSI; directly diagnostic for the project's drying/refuge questions.
- **Limitations:** area-weighted (large pools dominate the AOI-mean); says nothing about boundary complexity — pair with AWMSI if both axes matter. The AWRe⊥AWMSI orthogonality claim is **asserted, not yet demonstrated** — see §6.18 and checklist item 11.
- **Sensitivity:** low-to-moderate resolution sensitivity; more robust than AWMSI at 30 m.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Core.**
- **Ref:** Schumm 1956, *GSA Bulletin* — **adapted**, not verbatim; this is a basin-morphometry index repurposed to a pool population (see §4).

### 6.2 AWMSI — Area-weighted Mean Shape Index

- **Definition:** `AWMSI = Σ_i[(0.25·p_i/√a_i)·(a_i/Σa_i)]`.
- **Measures:** boundary complexity/crenulation, independent of elongation.
- **Interpretation:** high AWMSI = dissected, many-fingered, complex-edged patches; low AWMSI = smooth-edged patches, regardless of whether they are elongated or compact.
- **Input:** binary mask.
- **Scale:** AOI or window; monthly.
- **Calculation:** `pylandstats` shape index (area-weighted), or manual perimeter/area computation from labelled regions.
- **Strengths:** captures a genuinely distinct shape axis from AWRe, pending empirical confirmation (§6.18).
- **Limitations:** more sensitive to pixel-edge classification noise and stair-stepping than AWRe, especially at 30 m; can conflate coastline-style crenulation with sensor noise.
- **Sensitivity:** moderate-to-high resolution sensitivity.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Secondary** (demoted from core; not redundant with AWRe, but AWRe is the more robust and more ecologically diagnostic of the two).
- **Ref:** McGarigal & Marks 1995. See also Fernández-i-Marín et al. 2024 (*Landscape Ecology*) on complexity-weighted-area extensions to MESH/IIC/PC, for context on where shape-weighting sits in current landscape-metrics work — not directly used here, but relevant background.

### 6.3 LPI — Largest Patch Index

- **Definition:** `LPI = max_i(a_i) / A_total × 100`, where `A_total` is the fixed AOI/landscape area.
- **Measures:** dominance of the single largest connected waterbody.
- **Interpretation:** high LPI = one large connected pool dominates; falls as the main body breaks up.
- **Input:** binary mask, AOI polygon.
- **Scale:** AOI or window; monthly.
- **Calculation:** `pylandstats.Landscape.largest_patch_index()`, or `skimage.measure.regionprops` → max area / `A_total`.
- **Strengths:** non-circular (fixed denominator), interpretable, published, cheap.
- **Limitations:** captures only the largest patch, ignores the rest of the distribution (pair with MESH if that matters).
- **Sensitivity:** robust to classification noise. Resolution-sensitive only at pool edges.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Core.**
- **Ref:** McGarigal & Marks 1995; Bosch 2019 (PyLandStats).

### 6.4 MESH — Effective Mesh Size

- **Definition:** `MESH = Σ_i(a_i²) / A_total`. `A_total` is the fixed landscape area (**not** total water area).
- **Measures:** probability that two randomly chosen water pixels lie in the same patch, scaled to area.
- **Interpretation:** collapses as water fragments into many small patches; integrates the whole size distribution, unlike LPI.
- **Input:** binary mask, AOI polygon.
- **Scale:** AOI or window; monthly.
- **Calculation:** `pylandstats.Landscape.effective_mesh_size()`, or manual `sum(a**2)/A_total`.
- **Strengths:** non-circular, sensitive to full distribution, published.
- **Limitations:** correlates with LPI when one patch dominates — validate on real data; **hard gate: drop if r > ~0.9** (checklist item 12; not a soft suggestion).
- **Sensitivity:** as LPI.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Secondary.**
- **Ref:** Jaeger 2000.

### 6.5 Dry-down rate

- **Definition:** slope of APSEC (or wetted area) against time over the recession limb of each HY: `dry_down = (APSEC_enddry − APSEC_peak) / Δt_months` (negative = drying).
- **`[AUDIT FIX]` Terminology.** This is a **monthly-extent contraction rate**, not a hydrograph recession-constant analysis (which "recession" can imply to a hydrologist). Document it as such in user-facing text to avoid implying a `k`-parameterised recession model that the metric does not compute.
- **Measures:** rate of surface-water contraction through the drying season.
- **Interpretation:** the primary refuge-risk signal — two reaches with identical end-dry extent are ecologically very different if one dried in 2 months vs 6.
- **Input:** monthly APSEC series + HY anchors; both `max_water`- and `median`-composited series per §1.1.2.
- **Scale:** AOI or window; per HY.
- **Calculation:** linear regression (or robust Theil–Sen) of APSEC on month index between peak-wet and end-dry anchors, per HY. Compute on both composite series; flag `composite_sensitive` if end-dry APSEC disagrees by more than the configured tolerance (default 10 pp) between composites.
- **Strengths:** cheap, directly ecological, uses data already computed.
- **Limitations:** depends on HY anchor quality — propagate HY confidence flag; undefined in years with no clear peak/end-dry; sensitive to the `max_water` bias described in §1.1.2 unless the dual-composite check is run.
- **Sensitivity:** robust (uses aggregate extent, not patch delineation) — except for the composite-choice sensitivity above.
- **Binary/probabilistic:** both (APSEC works on either once extent is defined).
- **Tier:** **Core.**
- **Ref:** contraction-rate framing adapted from intermittent-flow hydrology (Costigan et al. 2016; Gallart et al. 2012) — novel as a remote-sensing surface-water metric.

### 6.6 Reconnection timing — see revised definition in §6.15

### 6.7 Refuge spatial stability — see revised definition in §6.16

### 6.8 NNI (Clark–Evans) — clustering, exploratory only

- **Definition:** `NNI = D_obs / D_exp`, `D_exp = 0.5 / √(N / A)`; `D_obs` = mean nearest-neighbour distance among pool centroids.
- **`[AUDIT FIX]` Tier change.** v1.1 offered NNI as a Secondary metric to "pick" alongside inter-pool gap depending on skeleton availability. The audit's reading of the spec's own stated limitations — 2D CSR is the wrong null model for a quasi-1D river corridor, and NNI is unstable below N≈8–10, which is precisely the low-water, high-fragmentation regime where the metric matters most — means it should not be treated as an equal alternative to gap. **NNI is now Exploratory only**: usable as a fallback descriptive statistic when no channel skeleton exists at all (so gap cannot be computed), never as a publication-grade fragmentation claim, and always reported alongside its `N < 10` instability flag.
- **Measures:** whether pools are clustered (<1), random (≈1), or dispersed (>1) in space, under a planar-CSR null that is a poor model for linear river corridors.
- **Input:** pool centroids, AOI area.
- **Scale:** AOI or zone; monthly.
- **Calculation:** `pointpats` (PySAL) or `scipy.spatial.KDTree`; Donnelly edge correction for bounded/elongated AOIs.
- **Strengths:** published, normalises for density and area.
- **Limitations:** unstable at low N (< ~8–10) — common at end-dry, exactly when it matters most; 2D CSR expectation is a poor fit for linear river corridors. The correct alternative — network-constrained point-pattern analysis (distance-along-network rather than Euclidean; e.g. `spatstat.linnet`-style methods) — is not implemented in v1; inter-pool gap (§6.10) serves this role instead.
- **Sensitivity:** strongly resolution-sensitive (pool count drives N); do not compare across sensors.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Exploratory — planar-AOI fallback only, no channel skeleton available.** Not publication-grade for river fragmentation.
- **Ref:** Clark & Evans 1954; Donnelly 1978.

### 6.9 Pool width distribution (non-circular reformulation of AWMPW)

- **Definition:** for each pool at each time step, compute width via the Euclidean Distance Transform (EDT) skeleton method (max or mean of `2 × EDT` along the pool's medial axis). Report **unweighted** distributional summaries across all N pools: **mean width, median width, max width, CV of width**. This is **not** area-weighted — each pool contributes one observation to the distribution regardless of its area, removing the circular-denominator pattern of the original AWMPW (`Σ(wᵢaᵢ)/Σaᵢ`).
- **Measures:** the morphological confinement/shape of pools — how narrow (channel-confined) vs wide (open, floodplain-style) the population of pools is at a given time.
- **Interpretation — important caveat:** **width is a confinement/morphology signal, not a depth proxy.** A pool can be wide and shallow, or narrow and deep; width alone does not indicate water depth or storage volume. Interpret alongside AWRe (elongation) and zone membership: narrow + elongated + in-channel = confined channel pool morphology; wide + compact + off-channel = open floodplain/billabong morphology. Do not use width as a depth or persistence proxy in ecological interpretation.
- **Input:** binary mask.
- **Scale:** AOI or zone; monthly; reported as a distribution, not a single scalar.
- **Calculation:** `scipy.ndimage.distance_transform_edt` on each labelled pool mask; width = 2 × max(EDT) (widest cross-section) or 2 × mean(EDT) along the skeleton (average cross-section); aggregate unweighted across pools into distributional statistics.
- **Strengths:** non-circular; recovers the morphological information the original AWMPW targeted, without conflating it with pool size; distributional form preserves information a single scalar discarded.
- **Limitations:** EDT-based width is sensitive to raster resolution (planform measurement only, cannot be a depth proxy) and to pool shape irregularity (a bent pool's EDT width can be misleading at bends). Unweighted mean can be pulled by many small transient pools during high-N months — median and max are often more stable summaries.
- **Sensitivity:** resolution-sensitive; strongest at coarse resolution (30 m) — narrow channels near the 1–2 pixel width limit are unreliable.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Secondary — morphological confinement indicator.**
- **Ref:** `[AUDIT FIX]` EDT-based river-width estimation lineage now cited explicitly: **Pavelsky & Smith 2008 (RivWidth)**, **Yang et al. 2020 (RivWidthCloud)**. The non-circular distributional reformulation for pool populations is novel to this project.

### 6.10 Inter-pool gap — clustering, metric of record

- **Definition:** distribution of dry-channel distances between consecutive pools along the channel (Zone 1 / skeleton). Report **mean gap**, **max gap**, **CV of gaps**, and **% gaps > T** (T ecological, e.g. 5 km).
- **Measures:** longitudinal fragmentation as experienced by an organism moving along the channel.
- **Interpretation:** the most ecologically direct fragmentation metric for linear systems.
- **Input:** binary mask within Zone 1 / channel skeleton.
- **Scale:** Zone 1 (or skeleton); monthly.
- **Calculation:** 1D run-length analysis along the channel skeleton.
- **Strengths:** directly ecological, correct 1D geometry for rivers, richer than a single clustering scalar.
- **Limitations:** requires a channel skeleton/centreline — not available in full no-drainage mode (falls back to NNI, §6.8, in that case only).
- **Sensitivity:** resolution-sensitive; skeleton quality matters.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Secondary — metric of record for clustering wherever a channel skeleton exists.** `[AUDIT FIX]` promoted from "preferred over NNI" to the default; NNI is now the fallback, not a co-equal option.
- **Ref:** adapted from waterhole-spacing / fish-passage literature (Sheldon et al. 2010; Fullerton et al. 2010); operationalisation novel.

### 6.11 TCF — Temporal Connectivity Frequency (formerly PCF)

- **`[AUDIT FIX]` Renamed from PCF.** `PCF` collided with `PC` (Probability of Connectivity, Saura & Pascual-Hortal 2007), the most commonly used graph-based connectivity index in landscape ecology. Use `TCF` everywhere: code, schema (`metric` values), docs, and any paper text.
- **Definition:** for each fixed pool node (Zone 2 persistent-pool set), the fraction of months it has ≥1 active connection to an adjacent node.
- **Measures:** temporal reliability of a pool's connectivity — the connectivity analogue of pixel occurrence frequency.
- **Interpretation:** distinguishes reliably-linked from chronically-isolated refuges.
- **Positioning against the field (locked requirement, §1.1.11):** DCI and PC/IIC are *static* structural indices computed on one network snapshot (or on barrier passability). TCF's actual novelty is running an RC-style snapshot connectivity check *every month over a multi-decadal series* and reducing it to a per-node frequency — something neither DCI nor PC/IIC, as conventionally applied, does. State this explicitly wherever TCF is documented; do not present it as if no comparable literature exists.
- **Input:** monthly binary masks + fixed node set (Zone 2 persistent pools).
- **Scale:** per node → summarised to AOI/zone; over full series.
- **Calculation:** build fixed-node graph from Zone 2; mark active edges per month; TCF = active months / valid months per node.
- **Strengths:** fills the temporal-connectivity gap; no cross-time pool identity needed because nodes are fixed by the zone mask.
- **Limitations:** requires the fixed persistent-pool node set — not computable in no-drainage/no-persistent-pool fallback; edge-definition choice is sensitive.
- **Sensitivity:** depends on node-set stability and edge rule.
- **Binary:** yes. **Probabilistic:** threshold first.
- **Tier:** **Secondary — zone-mask-dependent; skip cleanly when unavailable.**
- **Ref:** temporal-connectivity concept related to Rubio & Saura 2012; positioned against Cote et al. 2009 (DCI) and Saura & Pascual-Hortal 2006/2007 (IIC/PC); novel operationalisation as a monthly-frequency reduction.

### 6.11a `[NEW]` DCI — Dendritic Connectivity Index, and its relationship to RC/TCF

`JUDGEMENT CALL:` the audit identified that HydroFragments must, at minimum, *cite and conceptually position* against DCI/PC/IIC (§1.1.11). Whether it should also *implement* DCI as a first-class metric is a genuine choice, not a determinate fix. This section gives a recommended default and states the alternative.

- **What DCI is:** for a dendritic (or, degenerately, linear) river network with barriers or fragmentation events at known locations, DCI is the probability that two random points in the network are connected, computed from the sizes and connectivity of the resulting fragments (Cote et al. 2009). On a single-reach linear network, DCI reduces to a formula close to HydroFragments' `RC_pair` (reachable-pair) definition in §6.13, with "node size" = reach length rather than pool count. A revised, zero-flow-fragment DCI has already been applied to intermittent-river connectivity at large scale (PNAS 2025, doi:10.1073/pnas.2421046122).
- **Recommended default (implement):** because RC/TCF's underlying graph machinery (§6.13, `networkx`, fixed nodes/edges) already computes almost everything DCI needs, implement DCI as an additional monthly output using reach-length-weighted nodes, computed from the same fixed-node graph used for RC. This gives HydroFragments a standard, citable connectivity number for free, strengthens the "build vs. reuse" story for JOSS (§13), and gives the companion paper a direct benchmark (§16).
- **Alternative (citation-only):** if implementation time is constrained, DCI/PC/IIC may instead be used purely as a **conceptual and citation-level positioning** for RC/TCF, with a one-off benchmark comparison run for the paper (e.g., using `riverconn` in R, or a hand-computed DCI on the Gilbert reach) rather than a maintained code path. This satisfies the minimum bar from the audit (cite and position) but forgoes the stronger, cheaper option of just implementing it.
- **Not recommended:** implementing PC or IIC in full. Both require a distance-decay dispersal-probability kernel between patches, which does not map cleanly onto discrete pools with fixed node identity in the way DCI's fragment-size formulation does. If a benchmark is wanted for PC/IIC specifically, use `riverconn`/Conefor externally rather than reimplementing.
- **Output requirement if implemented:** `DCI_t` per AOI/window per month, alongside `RC_t`; both in the tidy long table under `metric_family = connectivity`.

### 6.12 Pixel recurrence and seasonality / hydroperiod

- **Definition — recurrence:** for each pixel, `REC_p = count(years where pixel is wet at least once) / count(valid years) × 100`.
- **Definition — annual seasonality / hydroperiod:** for each pixel and year, `SEAS_{p,y} = count(months wet in year y)` or `HP_{p,y} = valid wet months / valid observed months`.
- **Measures:** whether water reliably returns year after year, and how long it remains wet within a hydrological year.
- **Interpretation:** occurrence captures long-term wet frequency; recurrence captures inter-annual reliability; seasonality/hydroperiod captures within-year duration. These are complementary and do not require pool identity tracking.
- **`[AUDIT FIX]` Positioning.** These are **JRC Global Surface Water / DEA WOfS-style layers** (Pekel et al. 2016; Mueller et al. 2016), re-derived on an arbitrary source mask rather than a novel construction. Document them as such — "aligns HydroFragments with JRC/DEA-style products," not as a new contribution.
- **Input:** monthly binary mask and valid-observation mask.
- **Scale:** pixel → AOI/window summary.
- **Calculation:** `xarray` reductions grouped by HY or calendar year.
- **Tier:** **Secondary pixel-temporal output.**
- **Output recommendation:** emit rasters (`occurrence`, `recurrence`, `mean_hydroperiod`) plus AOI/window summary statistics (`mean`, `median`, `q90`, area above thresholds).

### 6.13 Realised connectivity (RC) — safe snapshot graph metric

- **Definition:** build a fixed graph `G = (V, E_max)` from either fixed channel windows, fixed refuge nodes, or fixed skeleton segments. At month `t`, each possible edge `(i,j)` is active if the water mask indicates a wet connection, or if the dry gap is below a configured threshold.
- **Equation:** `RC_t = 100 × Σ_{(i,j)∈E_max} e_{ij,t} / |E_max|`, where `e_{ij,t}=1` when edge `(i,j)` is active at time `t`.
- **Alternative reachable-pair equation:** `RC_pair_t = 100 × Σ_{i<j} I(component_i(t)=component_j(t)) / choose(|V|,2)`.
- **Measures:** the fraction of possible network links or reachable pairs realised at a given month.
- **Interpretation:** high RC = the system is connected/reconnected; low RC = wet habitat exists but is fragmented.
- **`[AUDIT FIX]` Relationship to DCI.** `RC_pair_t` with reach-length-weighted nodes is structurally close to a monthly DCI snapshot (§6.11a). State this explicitly in docs; it is the cleanest way to position RC against the standard literature.
- **Input:** monthly mask, fixed node/segment set, edge rule.
- **Scale:** AOI, channel, or window network.
- **Calculation:** `networkx` on fixed graph with monthly active edges.
- **Tier:** **Secondary/Core connectivity module** depending on whether the paper emphasises connectivity.
- **Critical guard:** fixed nodes/segments are allowed; transient monthly patch identity is not.

### 6.14 Graph components and largest connected component — only within graph module

- **Graph components:** `C_graph_t = count(connected_components(G_t))`.
- **Largest connected component fraction:** `LCC_t = max_c |V_{c,t}| / |V| × 100`.
- **Interpretation:** unlike raster `N`, these describe connectivity among fixed nodes or segments under a chosen edge rule. They are not the same as patch count unless graph nodes are identical to monthly patches and edges are only raster contiguity.
- **Tier:** optional support metrics for realised connectivity.

### 6.15 Revised reconnection timing

- **Preferred definition:** `reconnection_lag_y = first month after end-dry where RC_t ≥ t_RC` (or `DCI_t ≥ t_DCI` if implemented), minus the end-dry month.
- **Fallback when graph is unavailable:** use `LPSEC_t ≥ t_LPSEC` for channel-dominated AOIs, or `LPI_t ≥ t_LPI` only as a coarse proxy.
- **Required output:** `reconnection_metric_used = RC | DCI | LPSEC | LPI`, `t_reconnect`, and `proxy_reconnection_flag`. `[AUDIT FIX]` `DCI` added to the enum if §6.11a is implemented.
- **Reason for change:** LPI alone can indicate one dominant patch but not necessarily longitudinal reconnection.

### 6.16 Revised refuge spatial stability

Use one of the following (both avoid the static-footprint bug of using the same long-term occurrence footprint for every year):

1. **End-dry footprint stability (recommended for v1):** `R_y = water footprint at the end-dry month of HY y`, optionally intersected with the long-term refuge mask. Then `J_y = |R_y ∩ R_{y-1}| / |R_y ∪ R_{y-1}|`.
2. **Rolling-occurrence stability:** compute occurrence over a rolling window, e.g. 5 years, then threshold each rolling occurrence raster and compare successive windows with Jaccard overlap.

The end-dry footprint version is recommended for v1 because it uses monthly masks and avoids pool identity tracking.

### 6.17 Minimum metric formulas for retained "verbatim" metrics

- `N_t = count(label(W_t = 1, connectivity = 8, min_patch_pixels))`.
- `WA_t = cell_area × count(W_t = 1)` or `Σ_i a_i`.
- `APSEC_t = WA_t / A_ref × 100`.
- `WL_t = length of wet channel/skeleton cells`.
- `LPSEC_t = WL_t / L_ref × 100`.
- `OCC_p = Σ_t W_{p,t} / Σ_t V_{p,t} × 100`.
- `RA_θ = cell_area × count(OCC_p ≥ θ and valid_count_p ≥ min_valid_obs)`.
- `[NEW]` `DCI_t` (if implemented) `= 100 × Σ_{i<j} (len_i × len_j × c_ij,t) / (Σ_i len_i)²`, where `c_ij,t = 1` if fragments `i,j` are connected at month `t` under the chosen edge rule, else 0 — the standard DCI form (Cote et al. 2009) with reach-length node weights.

These formulas make the implementation contract testable without relying on external papers.

### 6.18 `[NEW]` Validation and accuracy — what is asserted vs demonstrated

The audit's clearest gap for a scientific reader: v1.1 had no section distinguishing metrics whose properties are *asserted by design* from those *empirically demonstrated*. This section is that inventory, and it is a locked deliverable — it must ship with v1, even if some rows read "not yet run."

| Claim | Status at spec-lock | Where it gets resolved |
|---|---|---|
| AWRe and AWMSI are orthogonal shape axes | **Asserted**, not demonstrated | Checklist item 11 — scatter + correlation on real data |
| LPI and MESH are sufficiently non-redundant to keep both | **Asserted**, not demonstrated | Checklist item 12 — hard gate at r > 0.9 |
| NNI is unstable below N≈8–10 | **Demonstrated in literature** (Clark & Evans; general point-pattern theory), not yet checked against this pipeline's own data | Checklist item 13 |
| Pool width behaves as a morphology signal, not a depth proxy in practice (not just in theory) | **Asserted** from EDT method properties | Checklist item 14 — compare against any available field/bathymetric data |
| `max_water` composite bias measurably affects dry-down rate | **Identified by audit, mechanism understood, magnitude not yet measured on real data** | New checklist item — run dual-composite comparison (§1.1.2) on at least one validation catchment and report the typical disagreement magnitude |
| RC/TCF behave sensibly relative to DCI on a real network | **Not yet checked** | New checklist item — benchmark RC_pair (reach-length-weighted) against a directly computed DCI on the Gilbert reach |
| Classification error in the input mask propagates to N/gap/MESH-tails/EDT-width in a bounded, characterisable way | **Not characterised** | Out of scope for v1 as a formal error-propagation model; document qualitatively (which metrics are edge-sensitive, per §8) and flag this explicitly as unresolved rather than silent |

This table should be maintained, not just created once — as checklist items resolve, move rows from "asserted" to "demonstrated" with a one-line result and a link to the analysis, rather than deleting the row.

---

## 7. Output data structure

```
catchment_id | aoi_id | zone | window_id | date | hy | hy_anchor | metric | value |
n_pools | valid_obs_frac | hy_confidence | edge_flag | source | resolution_m
```

- `zone`: `AOI`, `1`, `2`, `3`, `4`. Persistence metrics only ever emit `zone = AOI` or `zone = channel` (never 2/3/4 — circularity guard). Zone `1` is only emitted when a real drainage layer is present (§3 — morphology-proxy fallback removed).
- `hy_anchor`: `null` for plain monthly rows; `peak_wet` / `mid_dry` / `end_dry` where the month is an anchor.
- `edge_flag`: `ok` / `N0` / `N1` / `N2_unstable` / `no_channel` / `low_valid_obs`.
- `hy_confidence`: propagated from HY detection; low-confidence years flagged.
- `source` / `resolution_m`: mandatory — records the mask source (WOfS / Water Detect-S2 / Water Detect-PS / other) and native resolution; enables the cross-sensor guard (§8).
- Distributional metrics (pool width, inter-pool gap) emit one row per summary statistic (`mean`, `median`, `max`, `cv`) tagged under the same `metric` family name.

**Edge-case handling (mandatory):**
- `N = 0` → all patch/config metrics = NaN, `edge_flag = N0`. Informative zero-water state, not missing data.
- `N = 1` → clustering/connectivity/width-distribution CV = NaN, `edge_flag = N1`; extent/LPI still valid.
- `N = 2` → NNI computed but `edge_flag = N2_unstable`; width-distribution CV flagged low-confidence.
- `valid_obs_frac < min_valid_obs` → `edge_flag = low_valid_obs`; occurrence/RA suppressed for that pixel/month.

### 7.1 Output schema amendments

Add these columns to the long-format table:

```
run_id | config_hash | package_version | git_sha | crs | area_unit | length_unit |
monthly_composite | water_threshold | threshold_method | min_patch_pixels |
connectivity_rule | metric_family | statistic | unit | value_type |
n_valid_pixels | n_water_pixels | valid_fraction_month | min_valid_fraction_month |
proxy_channel | metric_dependency | warning_flag |
awre_length_method | composite_sensitive | connected_wet_metric | connected_wet_threshold |
reconnection_metric_used
```

`[AUDIT FIX]` new columns explained:
- `awre_length_method`: `skeleton` | `major_axis` — see §6.1.
- `composite_sensitive`: boolean — flags HY-anchor rows where `max_water`- and `median`-derived end-dry APSEC disagree beyond tolerance (§1.1.2).
- `connected_wet_metric`, `connected_wet_threshold`: names and thresholds the connectivity definition backing the `connected_wet` state flag (§1.1.8).
- `reconnection_metric_used`: now includes `DCI` as an allowed value (§6.15).

Recommended conventions (unchanged):
- `metric_family`: extent, persistence, morphology, fragmentation, clustering, connectivity, dynamics, diagnostic.
- `statistic`: null for scalar metrics; mean/median/max/cv/q10/q90 for distributional outputs.
- `unit`: %, km2, km, count, month, dimensionless, etc.
- `value_type`: monthly, HY_anchor, HY_summary, raster_summary, diagnostic.
- `metric_dependency`: e.g. `requires_channel`, `requires_fixed_nodes`, `requires_HY_anchor`, `proxy_allowed`.
- `warning_flag`: allow multiple semicolon-separated flags, e.g. `low_valid_obs;Nlt10_NNI_unstable;proxy_channel;composite_sensitive`.

Also emit spatial outputs separately:
- occurrence raster
- recurrence raster
- mean/median hydroperiod raster
- valid-observation-count raster
- zone mask
- refuge mask at each configured threshold
- optional fixed-node/fixed-window graph geometries

---

## 8. Cross-cutting caveats (must be enforced, not just documented)

1. **Resolution is not comparable across sensors.** Patch-count-dependent metrics (N, NNI, gap, MESH tails, pool width) are **not** cross-sensor comparable. Enforce: tag every row with `source`/`resolution_m`; the comparison API refuses to pool across resolutions unless explicitly overridden.
2. **Probabilistic masks feed only pixel-level occurrence natively.** Every patch/config metric needs a binary mask, so a probabilistic mask is thresholded first (record the threshold).
3. **HY confidence must propagate to every aggregate.** Anchor-based metrics (dry-down, reconnection, refuge stability) inherit the HY confidence flag.
4. **Per-pixel valid-obs floor.** Occurrence = water / valid obs; enforce `min_valid_obs` (default 20).
5. **Zones stratify configuration, never persistence.** Enforced in the API (§3).
6. **Width ≠ depth.** Pool width distribution metrics must never be labelled or interpreted as depth/volume/storage proxies in documentation, plots, or downstream modelling without explicit caveat (§6.9).
7. **Minimum mapping unit guard.** Patch metrics must not be computed on unfiltered one-pixel noise unless the user explicitly disables `min_patch_pixels`.
8. **CRS/unit guard.** Refuse area/length metrics when CRS units are degrees unless a per-pixel area/length correction is supplied.
9. **Monthly-composite guard.** Refuse to merge or compare metric series generated with different `monthly_composite` rules unless explicitly overridden.
10. **NNI reporting floor.** Report NNI as NaN at `N < 2`; flag `Nlt10_NNI_unstable` for `2 ≤ N < 10`. `[AUDIT FIX]` NNI is Exploratory-only regardless of N (§6.8); this floor governs when it may even be reported, not when it becomes "publication-grade."
11. **`[NEW]` Monthly-composite bias guard.** Dry-down rate and end-dry-anchor-derived outputs must carry the `composite_sensitive` diagnostic (§1.1.2, §7.1). Do not present dry-down rate to a non-technical audience without checking this flag first.
12. **`[NEW]` Length-metric CRS-distortion note.** For catchment-scale windowed analyses, note in output metadata that the equal-area CRS does not guarantee minimal length distortion for `L_ref`/gap comparisons (§2).
13. **`[NEW]` Naming-collision guard.** `TCF` is reserved for temporal connectivity frequency; `PCF` must not reappear anywhere in code, schema, or docs (§1.1.11).
14. **`[NEW]` Graph-node guard.** Fixed-node graph outputs must declare the node source: `channel_windows`, `fixed_refuge_nodes`, `skeleton_segments`, or `external_network`.
15. **`[NEW]` Reconnection proxy guard.** If reconnection is derived from LPI rather than RC/LPSEC/DCI, mark it as `proxy_reconnection`.
16. **`[NEW]` Occurrence/zone circularity guard.** Persistence metrics can be summarised by AOI and by externally defined channel mask, but not by zones whose definition uses the same occurrence thresholds.

---

## 9. Python implementation map

| Metric group | Primary tools |
|---|---|
| Raster I/O, masking, time series | `xarray`, `dask`, `rioxarray`, `rasterio` |
| Patch delineation (N, areas, LPI, AWRe, AWMSI) | `scikit-image` (`label`, `regionprops`), `pylandstats` |
| MESH, LPI, area-weighted shape | `pylandstats` |
| Vector / AOI / windows / gaps | `geopandas`, `shapely` |
| Channel skeleton / inter-pool gap / pool width EDT | `scikit-image` (`skeletonize`), `scipy.ndimage.distance_transform_edt` |
| Clustering (inter-pool gap primary; NNI fallback) | `pointpats` (PySAL), `scipy.spatial` |
| Graph / RC / TCF / DCI | `networkx`. `[AUDIT FIX]` cite and, where useful for testing, cross-check against `riverconn` (R; Baldan et al. 2022) or Conefor for DCI/PC/IIC formulas — not a runtime dependency, a validation reference. |
| Occurrence / recurrence | `xarray` reductions (native) |
| Trend / dry-down / recession | `numpy`/`scipy` regression, `statsmodels` (STL), `pymannkendall` |
| Bayesian downstream | `pymc` / `numpyro` (consumes the tidy monthly table) |
| `[NEW]` Testing | `pytest`, `pytest-cov`; CI via GitHub Actions (lint + test + docs build matrix) |
| `[NEW]` Docs | `sphinx` or `mkdocs` + hosted build (Read the Docs / GitHub Pages) |

**Suggested module layout for the refactor:**
```
hydrofragments/
  io/            # mask loading, harmonisation, valid-obs (source-agnostic)
  zones/         # zone mask build + fallback logic (persistence-proxy only, v1)
  hydroyear/     # persistence-based HY detection + anchors + confidence
  metrics/
    extent.py        # APSEC, LPSEC, occurrence, RA
    patches.py       # N, LPI, MESH, AWRe, AWMSI
    morphology.py    # pool width distribution (EDT-based, non-circular)
    dynamics.py      # dry-down, reconnection, refuge stability
    clustering.py    # inter-pool gap (primary), NNI (exploratory fallback)
    connectivity.py  # RC, TCF, DCI (if implemented) — docstrings must cite Cote 2009 / Saura & Pascual-Hortal 2006,2007
  aggregate/     # monthly -> HY anchors, windowing
  guards/        # circularity, resolution, edge-case, composite-bias flags
tests/
docs/            # sphinx/mkdocs source; input-format spec (§14) lives here
.github/workflows/   # CI: lint, test, docs
CONTRIBUTING.md
CODE_OF_CONDUCT.md
LICENSE
paper/
  paper.md       # JOSS short paper — see §13
```

---

## 10. Suggested paper figures & tables

`[AUDIT FIX]` framing revised to avoid overlap with Tayer et al. (2025); one figure added for connectivity positioning.

- **F1** Conceptual: pixel → pool → window → AOI/catchment scale hierarchy + monthly/HY temporal axis.
- **F2** The four-zone mask over a study reach, with and without drainage layer.
- **F3** Monthly time series of core metrics (APSEC, N, LPI, AWRe) for a catchment, with HY anchors marked.
- **F4** Dry-down rate vs end-dry Refuge Area across HYs — the refuge-risk plane, computed on both `max_water` and `median` composites to show the composite-sensitivity result directly (§1.1.2, §6.18).
- **F5** Refuge spatial-stability (Jaccard) map/trend — fixed vs migrating refuges.
- **F6** AWRe vs AWMSI scatter — demonstrating the two shape axes are genuinely orthogonal (compact/complex vs elongated/smooth quadrants). This is a *results* figure, not a design assertion — it only exists once checklist item 11 has run.
- **F7** Pool width distribution by zone — confinement signal across in-channel vs off-channel populations, explicitly not framed as depth.
- **`[NEW]` F8** RC/TCF vs DCI positioning — monthly RC_pair (or implemented DCI) time series against a directly computed static DCI, demonstrating what the temporal-frequency framing adds beyond a snapshot index.
- **T1** Full metric register (the keep/drop/add/rename table, §4) — publishable contribution in itself.
- **T2** Redundancy/adaptation-level table.
- **`[NEW]` T3** Validation status table (§6.18) — what's asserted vs demonstrated, reproduced directly for the paper.

**Publishability framing — `[AUDIT FIX]`, this is the differentiation statement, read it before drafting the paper:** the novel contribution is the *integrated framework* — persistence-based HY detection + zonation + the dynamics metrics (dry-down, reconnection, refuge stability) + the **non-circular morphology and connectivity reformulations** (AWRe, pool width, RC/TCF positioned against DCI/PC/IIC) + the metric-register circularity critique itself — not the individual indices (LPI, MESH, NNI, AWMSI, AWRe are all independently published) and **not** a restatement of the four-step framework / HY-detection concept / Gilbert-reach case study that Tayer et al. (2025) already published. See §16 for the explicit scope boundary against that paper.

---

## 11. Implementation checklist

Restructured into three tracks: **11A** core metrics (v1.1's items, revised), **11B** open-source/JOSS readiness (new — see §13), **11C** validation (existing items, reframed against §6.18).

### 11A. Core metrics

1. [ ] Rebrand repo/package: `iRivermetrics` → `HydroFragments`. **Preserve git history — do not squash the rebrand commit** (this matters for §13's public-history requirement). Update PyPI/GitHub naming, module namespace, docs.
2. [ ] Generalise all I/O and docs language from WOfS-specific to source-agnostic binary-mask input.
3. [ ] Lock the AOI-polygon + `A_ref`/`L_ref` conventions (§2); implement optional windowing; document the equal-area-vs-equidistant length caveat.
4. [ ] Build zone module with configurable thresholds + persistence-proxy no-drainage fallback **only** (morphology proxy removed, §3) + Zone-1 skip logic.
5. [ ] Wire persistence-based HY detection → anchors + confidence flags into the aggregation layer.
6. [ ] Implement core metrics (occurrence, RA, APSEC, LPSEC, N, LPI, AWRe [with locked `awre_length_method`, §6.1], dry-down [with dual-composite check, §1.1.2]).
7. [ ] Implement secondary metrics (AWMSI, MESH, pool width distribution, inter-pool gap [primary] / NNI [exploratory fallback], reconnection, refuge stability, TCF [renamed from PCF], DCI [if implementing per §6.11a]).
8. [ ] Implement guards: circularity (zone×persistence), resolution tagging, edge-case flags, valid-obs floor, width≠depth documentation guard, composite-sensitivity flag, naming-collision guard (no `PCF` anywhere).
9. [ ] Emit the tidy long-format table (§7, §7.1) with all flags populated, including the new columns.
10. [ ] Unit-test edge cases: N=0/1/2, empty zone-month, no-drainage mode, low-valid-obs, missing end-dry anchor.
11. [ ] Define and test monthly compositing rules (`max_water`, `median`, `mode`, `nearest_enddry`) and write the chosen rule to metadata; **run the dual-composite dry-down comparison and report the typical disagreement magnitude on at least one validation catchment.**
12. [ ] Enforce projected/equal-area CRS or per-pixel area arrays for all area/length metrics.
13. [ ] Implement `min_patch_pixels`, `connectivity_rule`, and patch-filter tests.
14. [ ] Add recurrence and hydroperiod rasters/summaries, documented explicitly as JRC/DEA-style (§6.12).
15. [ ] Add realised connectivity (`RC`) using fixed channel windows, fixed refuge nodes, or skeleton segments, with node-source metadata; implement `DCI` alongside it if the §6.11a judgement call resolves toward implementation.
16. [ ] Replace LPI-only reconnection timing with RC/DCI-based timing where possible; mark LPI-only reconnection as a proxy.
17. [ ] Revise refuge spatial stability so `R_y` varies by HY: end-dry footprint or rolling-occurrence footprint.
18. [ ] Emit config hash, run metadata, metric units, statistic labels, and all threshold values.
19. [ ] Unit-test CRS refusal, min-patch filtering, different connectivity rules, monthly compositing, graph-node guards, and the new `composite_sensitive` / `awre_length_method` / naming-collision guards.
20. [ ] Regression-test new suite against published iRivermetrics outputs on the Gilbert reach (sanity, not equivalence — PF/PLF and the old AWMPW are gone).

### 11B. `[NEW]` Open-source / JOSS readiness (see §13 for full rationale)

21. [ ] Add an OSI-approved `LICENSE` file (MIT/BSD/Apache/GPL — pick one; not just the paper's CC-BY).
22. [ ] Plan and execute a **≥6-month public open-development window** with real tagged releases and real issues/PRs before submission — do not treat this as a formality; JOSS's 2026 scope change makes it a pre-review screening gate, and a fresh repo dropped at the rebrand is exactly the pattern it screens against (§13).
23. [ ] Write `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, and a stated support channel.
24. [ ] Set up CI (GitHub Actions): lint, test matrix, docs build.
25. [ ] Write hosted, rendered API docs (Sphinx/mkdocs), not just docstrings.
26. [ ] Write a README quickstart: install instructions, minimal runnable example, example dataset.
27. [ ] Write the input-data-format specification (§14) as a standalone doc practitioners can follow without reading this whole spec.
28. [ ] Draft `paper/paper.md`: Summary, Statement of need, **State of field** (explicit comparison to `pylandstats`, `riverconn`/Conefor, DEA/JRC GSW — §13), references, **AI usage disclosure** (this spec itself was produced with AI assistance — disclose accurately).
29. [ ] Archive a release on Zenodo and mint a DOI at/near submission.

### 11C. Validation (reframed against §6.18)

30. [ ] Validate AWRe vs AWMSI orthogonality on real data (scatter, correlation) — confirm both axes are worth retaining. Update §6.18 row from "asserted" to "demonstrated" with the result.
31. [ ] Validate LPI vs MESH correlation on real data — **hard gate**: drop MESH if r > ~0.9 everywhere. Update §6.18.
32. [ ] Validate NNI stability vs N on this pipeline's own data; confirm the exploratory-only tier is warranted (it should be). Update §6.18.
33. [ ] Validate pool width distribution against any available field/bathymetric data to confirm it behaves as a morphology signal, not spuriously correlating with depth in ways that invite misinterpretation. Update §6.18.
34. [ ] Benchmark RC_pair (or implemented DCI) against a directly computed DCI on the Gilbert reach; report agreement/divergence. Update §6.18 and produce F8 (§10).

---

## 12. `[NEW]` Audience-specific deliverables

The audit found the spec serves implementers reasonably and scientists partially, but has nothing for water-resource managers and is missing the practical layer implementers need to actually run the tool. These are separate deliverables, not sections of this spec — but their scope is locked here so they get built.

### 12.1 Manager / stakeholder interpretation guide

**Not a JOSS artifact, not the companion paper — a separate, short, plain-language document** (suggested: `docs/for-managers.md` or a 2-page PDF), scoped as:

- A plain-language glossary: what each core-tier metric (§5.1) means in one sentence, with no formulas.
- 3–5 **worked interpretation narratives**, e.g.: *"A dry-down rate of X% per month combined with an end-dry Refuge Area below Y indicates elevated refuge risk — the reach is drying faster than its refuges can absorb."* These narratives should be drawn from real output ranges on a validation catchment (e.g., Gilbert), not hypothetical numbers.
- A single **decision-support summary table**: metric → what a concerning value looks like → what management question it speaks to (e.g., environmental flow timing, refuge protection prioritisation).
- Explicitly **not** included: formulas, circularity arguments, connectivity-index positioning, CRS/compositing caveats — all of that stays in this spec and the companion paper. If a caveat is load-bearing for correct interpretation (e.g., width≠depth, `composite_sensitive` on dry-down), translate it into plain language rather than omitting it.
- If, on review, there is no real institutional audience for this document, drop the claim that HydroFragments serves managers rather than shipping a token version of it.

### 12.2 Practitioner quickstart (feeds into §14 and README, checklist item 26)

- A minimal runnable example against a small bundled or linked example dataset (a short reach, a few HYs) that exercises the full pipeline end-to-end in under a few minutes.
- CLI or API signature sketch — concrete function/class names and call patterns, not just the module map in §9.
- Explicit troubleshooting notes for the most likely first-run failures: grid misalignment between mask and valid-obs layer (§14), CRS-in-degrees refusal (§8 guard 8), missing drainage layer behaviour (§3).

---

## 13. `[NEW]` JOSS readiness and open-development timeline

JOSS's criteria changed materially in **January 2026** ("Preparing JOSS for a generative AI future," blog post 2026-01-05): the journal now evaluates human creativity, design thinking, and demonstrable research impact rather than effort-by-code-volume, in direct response to AI-assisted codegen. This section locks the plan for meeting the *current* criteria — do not plan against older JOSS guidance.

### 13.1 The hard blocker

JOSS requires: *"Projects developed privately are not eligible until there is a public record of open development: at least six months of public history prior to submission, with evidence of releases, public issues/pull requests."* This is now explicitly a **pre-review desk-rejection screening gate**, not a soft preference — reviewers are instructed to flag repos "made public days before submission" or with "commit history concentrated into a short window" immediately to the handling editor.

**Locked plan:**
1. Rebrand the repo now (`iRivermetrics` → `HydroFragments`), but **do not squash or discard the `iRivermetrics` git history** — preserve it so the public record predates the refactor. iRivermetrics is already published-and-used (Tayer 2023a, 2023b, 2025), which is itself strong evidence of "sufficiently useful... likely to be cited."
2. Develop the v1.2 refactor **in the open**, with real tagged releases (not one mega-commit at the end) and real issues/PRs, even if the team is small — single-author projects are acceptable to JOSS but need community-engagement evidence *somewhere* (repo or paper).
3. **Do not submit earlier than 6 months** after the repo/rebrand goes public with active development, and ideally have at least a few genuine external interactions (an issue from someone outside the immediate team, a PR, a cited use) by submission time.

### 13.2 Other JOSS artifacts (mapped to checklist §11B)

- OSI-approved `LICENSE` (item 21).
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates, support channel (item 23).
- CI (item 24), hosted docs (item 25), README quickstart (item 26).
- **AI usage disclosure** in `paper.md` (item 28) — required as of the 2026 policy; this spec itself was produced with AI assistance during drafting/audit, and that must be disclosed accurately rather than omitted.
- **State-of-field comparison**, drafted here so it's ready for `paper.md`:

| Tool | What it does | How HydroFragments differs |
|---|---|---|
| `pylandstats` (Bosch 2019) | General FRAGSTATS-family landscape metrics (LPI, MESH, shape indices) in Python | HydroFragments *uses* pylandstats for some of these, but adds the water-specific interpretation layer (zones, HY detection, dynamics metrics) pylandstats has no concept of |
| `riverconn` (Baldan et al. 2022) / Conefor (Saura & Torné 2009) | DCI, PC, IIC and related river/landscape connectivity indices, static snapshots | HydroFragments' RC/TCF/DCI module is positioned against these explicitly (§6.11, §6.11a); the differentiator is monthly-frequency temporal connectivity, not a replacement for static structural connectivity analysis |
| JRC Global Surface Water / DEA WOfS | Pixel-level occurrence, recurrence, seasonality from Earth observation | HydroFragments re-derives these layers on any source mask (source-agnostic) and builds the pool/patch/connectivity/dynamics layer on top, which JRC/DEA do not provide |
| iRivermetrics (Tayer et al. 2023a) | The direct predecessor — pool-scale ecohydrological metrics on Sentinel-2 | HydroFragments generalises the input source, fixes the circular-denominator metrics, adds zonation/HY-detection/dynamics, and adds the reproducibility/config-hashing discipline (§1.1.7) |

- Zenodo archive + DOI (item 29).

### 13.3 Residual gap after checklist completion

Completing all of §11A does **not** make the repo JOSS-ready by itself — §11B is not optional scaffolding, it is roughly half of what JOSS actually reviews. Track 11B to completion with the same seriousness as 11A.

---

## 14. `[NEW]` Input data format specification

Practitioner-facing; intended to become its own doc (`docs/input_format.md`) rather than live only here, per checklist item 27. This is a genuine gap in v1.1 — nothing there specified exact input contracts.

| Input | Format | Requirements |
|---|---|---|
| Water mask time series | GeoTIFF stack, NetCDF, or Zarr, readable via `xarray`/`rioxarray` | Boolean or `{0,1}` with an explicit nodata value (`255` or masked); time dimension named `time`, monthly or sub-monthly cadence; CRS must be defined in the file (not assumed) |
| Valid-observation layer | Same grid as the water mask | Boolean: `1` = observed/clear, `0` = cloud/no-data/outside swath/outside AOI; **must share identical transform, CRS, and shape with the water mask** |
| AOI polygon | GeoJSON, Shapefile, or GeoPackage | Single polygon or multipolygon; CRS must be defined; will be reprojected to the configured equal-area CRS (default EPSG:3577 for AU deployments, §1.1.1) |
| Drainage layer (optional) | GeoJSON/Shapefile line features | `LineString`/`MultiLineString`; used to build Zone 1, `L_ref`, and the channel skeleton. Absence triggers the persistence-proxy fallback (§3) — no morphology-proxy fallback exists in v1 |
| Config file | YAML or JSON | Schema per §1.1 parameters (`t_persist`, `t_season`, `min_valid_obs`, `monthly_composite`, `connectivity_rule`, `min_patch_pixels`, `water_threshold`, `crs`, `state_flag_connectivity_metric`, `state_flag_connectivity_threshold`, etc.) |

**Grid alignment is a hard requirement, not a convenience.** The pipeline must **raise an explicit error**, not silently resample, if the water mask and valid-observation layer do not share an identical grid (same transform, CRS, and shape). Silent resampling is a common failure mode in source-agnostic tools and would corrupt every downstream metric without a visible symptom.

**Probabilistic input:** if the water mask is probabilistic rather than binary, it must additionally carry (or be paired with a config specifying) `water_threshold`, `threshold_method`, and `probability_source` per §1.1.6 — thresholding happens once, upstream, and is recorded, not silently re-applied per metric.

---

## 15. Deferred to a later release

**Architectural deferral (unchanged from v1.1):** pool-unit-level identity tracking through time — lineage through merges/splits, per-pool recession curves, per-pool survival/hysteresis models. None are required for v1 and none block the above.

**`[NEW]` Audit-driven deferral (implemented-in-v1.1, cut-in-v1.2):** node centrality (degree/betweenness) and morphology-proxy Zone 1 — see §5.5, §1.1.12, §3, §4. Unlike the architectural deferral, these were built-but-gated in v1.1; v1.2 removes them outright pending, respectively, a genuine dendritic-network use case and a ground-truth validation study.

---

## 16. `[NEW]` Companion paper scope

JOSS papers are deliberately short (Summary, Statement of need, references — no methods depth, per JOSS's own submission guidelines) and explicitly must not contain the kind of methodological argument this project has built. That argument needs a home; §10's "Suggested paper figures & tables" already gestures at one. This section locks the **minimal defensible scope** for that companion methods paper, and — the audit's sharpest finding on this point — draws an explicit line against Tayer et al. (2025), which is in the project's own reference set and already covers adjacent ground.

### 16.1 What the companion paper claims

1. **The metric register and its circularity/redundancy reasoning** (T1/T2, §4) — this is the intellectual core and the most novel thing in the whole document. No prior Tayer paper makes this argument explicitly; it is HydroFragments' own contribution.
2. **Persistence-based HY detection + zonation**, as implemented (not just as a framework claim — see boundary note below).
3. **The dynamics metrics** (dry-down, reconnection, refuge stability) as the headline ecological contribution, including the composite-sensitivity result (§1.1.2, §6.18, F4) as a methodological finding in its own right — this is a genuinely new result, not just an implementation detail.
4. **One validation case** (Gilbert reach — data already used in Tayer 2025) that *demonstrates rather than asserts*: the AWRe⊥AWMSI orthogonality result, the LPI/MESH redundancy check, and the RC/DCI positioning benchmark (F8). Without these results, the paper argues its keep/drop decisions instead of proving them.

### 16.2 `[AUDIT FIX]` Explicit differentiation from Tayer et al. (2025)

**Salami-slice risk is real and specific, not generic.** Tayer et al. (2025, in this project's reference set) already publishes: the four-step framework (scope, data, processing, analysis), the persistent-pool concept and definition, dynamic hydrological-year detection, and a Gilbert River (1986–2023) case study demonstrating trends in pool number/size/fragmentation against rainfall and discharge. A companion paper that re-presents "a framework, applied to Gilbert, showing pool dynamics" **will read as a resubmission of that paper** to any reviewer who has seen both, and risks a novelty/duplication desk-reject.

**The clean differentiator, stated as the paper's actual thesis:** *the software itself, and the non-circular reformulation of the metric suite* — the circularity critique that eliminates PF/PLF/AWMPA/AWMPL/AWMPW, the connectivity-module positioning against DCI/PC/IIC, and the reproducibility discipline (config hashing, composite-sensitivity flagging) — not "here is a framework and a case study, again." Concretely:
- Lead with the metric-register argument (§16.1 item 1), not the framework or the case study.
- Use Gilbert only as a **validation** dataset for the register's specific claims (orthogonality, redundancy, DCI positioning) — not as a standalone results section re-describing pool dynamics Tayer 2025 already reported.
- Cite Tayer 2025 explicitly as the source of the framework/HY-detection concept and the Gilbert dataset, and state plainly that this paper's contribution is the metric-register reformulation and its software implementation, applied to that same system to validate the reformulation rather than to re-report its ecology.

### 16.3 Suggested venues

*Environmental Modelling & Software*, *Ecohydrology*, *HESS*, or *J. Hydrology* — in roughly that order of fit, given the software/methods framing above (Environmental Modelling & Software is also `riverconn`'s home, which reinforces the connectivity-positioning argument if submitted there).

### 16.4 Relationship to the JOSS submission

JOSS explicitly accommodates a companion science paper submitted alongside a software submission, and asks authors to disclose related publications (in review or nearing submission) at JOSS submission time. Plan the two submissions as a pair: JOSS carries the software citation and the reproducibility/engineering story (§13); the companion paper carries the methodological argument (§16.1–16.2). Do not attempt to fit the methodological argument into the JOSS paper itself — JOSS paper.md explicitly excludes API/methods depth.

---

## Summary of changes from v1.1

For quick reference during refactor:

- **Renamed:** `PCF` → `TCF` (§1.1.11, §6.11).
- **Demoted:** NNI, from co-equal Secondary option to Exploratory/fallback-only (§4, §5.3, §6.8).
- **Cut from v1:** node centrality, morphology-proxy Zone 1 (§3, §4, §5.5, §1.1.12).
- **Added:** DCI as citation-anchor and optional implementation (§6.11a); dual-composite dry-down check (§1.1.2); locked AWRe length method (§6.1); validation-status table (§6.18); manager/practitioner deliverable scopes (§12); JOSS readiness plan incl. 6-month public-history requirement (§13); input-format spec (§14); companion-paper scope and Tayer-2025 differentiation (§16).
- **Corrected citations/attribution:** occurrence frequency (Pekel 2016, Mueller 2016), AWRe ("adapted" not "verbatim," Schumm 1956), pool-width EDT method (Pavelsky & Smith 2008, Yang et al. 2020), connectivity module (Cote et al. 2009; Saura & Pascual-Hortal 2006, 2007; Baldan et al. 2022).
- **New schema fields:** `awre_length_method`, `composite_sensitive`, `connected_wet_metric`, `connected_wet_threshold`; `reconnection_metric_used` extended with `DCI`.
- **New guards:** composite-bias, length-metric CRS-distortion note, naming-collision.
- **Checklist restructured** into 11A (core), 11B (JOSS/open-source readiness — new), 11C (validation, reframed).
