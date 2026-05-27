# Documentation Index

This documentation covers the **iRiverMetrics** toolkit — an open-source Python package for quantifying surface-water dynamics in intermittent rivers from multispectral satellite imagery.

## Entry points

| Audience | Start here |
|----------|-----------|
| New users | [Project overview](./project-overview.md) — scope, quick-start, citations |
| Module users | [Module 1 — waterdetect_batch](./module1.md) · [Module 2 — calculate_metrics](./module2.md) |
| Contributors | [Architecture](./architecture.md) — algorithms, data-flow, design decisions |

## Quick install

```bash
git clone https://github.com/tayerthiaggo/irivermetrics.git
cd irivermetrics
conda create -n irivermetrics python=3.10
conda activate irivermetrics
conda install conda-forge::gdal
pip install -e .
```

## Quick use

```python
from irivermetrics import waterdetect_batch, calculate_metrics

# Module 1 — generate water mask time series from images
da_wmask = waterdetect_batch("path/to/images", "path/to/rcor_extent.shp",
                             return_da_array=True)

# Module 2 — compute 16 ecohydrological metrics
metrics_df = calculate_metrics(da_wmask, rcor_extent="path/to/rcor_extent.shp",
                               section_length=0.484)
```

## Module documentation

- [waterdetect_batch](./module1.md) — batch water detection via the Water Detect algorithm
- [calculate_metrics](./module2.md) — 16 ecohydrological metrics from water mask time series

## Scientific foundation

iRiverMetrics is grounded in four peer-reviewed publications by Tayer et al.:

1. **Water Detect accuracy** — Tayer et al. (2023) *GIScience & Remote Sensing* — parameter calibration for `waterdetect_batch` ([DOI](https://doi.org/10.1080/15481603.2023.2168676))
2. **Ecohydrological metrics** — Tayer et al. (2023) *J. Hydrology* — the 16-metric algorithm ([DOI](https://doi.org/10.1016/j.jhydrol.2023.129087))
3. **Hydrological clustering** — Tayer et al. (2023) *J. Hydrology* — scale-up to 400 km reaches ([DOI](https://doi.org/10.1016/j.jhydrol.2023.130266))
4. **Resilience mapping framework** — Tayer et al. (2026) *J. Hydrology* — 38-year WOfS analysis ([DOI](https://doi.org/10.1016/j.jhydrol.2025.134750))

[Back to README](../README.md)
