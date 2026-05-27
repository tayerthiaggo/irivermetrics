# Project Overview

## What iRiverMetrics does

iRiverMetrics is a Python toolkit for quantifying the surface-water dynamics of **intermittent and non-perennial rivers** from multispectral satellite imagery. It produces 16 ecohydrological metrics describing pool morphology, fragmentation, persistence, and refuge conditions — metrics that cannot be measured adequately from flow gauges alone because many intermittent rivers have no gauges.

The toolkit operates on two inputs:
- **Multispectral satellite images** (e.g., Sentinel-2, Landsat, Planetscope, or WOfS products)
- **A river corridor extent shapefile** defining the area of interest and optional section boundaries

It produces:
- A binary water mask time series (Module 1)
- Per-section, per-timestep metric tables in CSV format (Module 2)
- Pixel persistence rasters and optional shapefiles of wetted features

## Modules

### Module 1 — `waterdetect_batch`

Runs the Water Detect algorithm (Cordeiro et al., 2021; Tayer et al., 2023) on a batch of multispectral images to produce a binary water mask time series. Water Detect uses normalised spectral water indices and agglomerative clustering — no labelled training data required.

Key parameters to calibrate for local conditions:
- `max_cluster` — maximum number of spectral clusters (dominates accuracy; default 6 for VNIR)
- `reg` — regularisation of normalised spectral indices (default 0.07 for VNIR)

See [Module 1 documentation](./module1.md) and [Tayer et al. (2023) GIScience](https://doi.org/10.1080/15481603.2023.2168676) for calibration guidance.

### Module 2 — `calculate_metrics`

Computes 16 ecohydrological metrics from any binary water mask time series. The algorithm:
1. Validates and clips input data to river corridor sections
2. Fills missing observations (cloud/shadow) via temporal interpolation (±2 timesteps)
3. Applies QA thresholds: excludes timesteps with <70% valid pixels before fill, <95% after fill
4. Reprojects data to UTM for accurate area/distance calculations
5. Labels connected water pools (8-neighbour connectivity)
6. Skeletonizes pools; finds longest path in each skeleton using igraph BFS (double-BFS on undirected graph)
7. Computes width via Euclidean distance transform along the skeleton path
8. Aggregates metrics per section per timestep

See [Module 2 documentation](./module2.md) and [Tayer et al. (2023) J. Hydrology](https://doi.org/10.1016/j.jhydrol.2023.129087) for full definitions.

## Scientific context

iRiverMetrics was developed and validated on:
- **Fitzroy River, WA** (5–400 km reaches; Sentinel-2 NBART 10 m; 2017–2021)
- **Gilbert River, QLD** (100 km reach; WOfS Landsat 30 m; 1986–2023)

It is designed to support:
- Ecological connectivity assessments (refuge pool mapping)
- Hydrological trend analysis over decadal timescales
- Section-level hydrological clustering (see [Tayer et al., 2023 J. Hydrology 626](https://doi.org/10.1016/j.jhydrol.2023.130266))
- The 4-step resilience mapping framework (see [Tayer et al., 2026 J. Hydrology 666](https://doi.org/10.1016/j.jhydrol.2025.134750))

## Current status

| Component | Status |
|-----------|--------|
| Package installable (`pip install -e .`) | ✅ |
| `waterdetect_batch` | ✅ Operational |
| `calculate_metrics` | ✅ Operational |
| igraph BFS pool-length algorithm | ✅ Implemented |
| QA thresholds (70%/95%) | ✅ Implemented |
| Dask lazy evaluation + batch processing | ✅ Implemented |
| Unit tests (16 tests) | ✅ |
| Integration tests | ✅ (marked `slow`) |
| GitHub Actions CI | ✅ |

## Citation

If you use iRiverMetrics, please cite:

> Tayer, T.C., Beesley, L.S., Douglas, M.M., Bourke, S.A., Meredith, K., McFarlane, D. (2023). Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river. *Journal of Hydrology*, 617, 129087. https://doi.org/10.1016/j.jhydrol.2023.129087

[Back to docs index](./index.md)
