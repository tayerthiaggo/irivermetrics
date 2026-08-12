# HydroFragments Project Overview

HydroFragments is a Python library for quantifying surface-water landscape metrics and patch dynamics in intermittent rivers from satellite earth observation data.

## Scientific Scope

Intermittent rivers and ephemeral streams exhibit dynamic wetting and drying cycles. HydroFragments analyzes time series of satellite-derived water masks to measure surface-water fragmentation, persistence, connectivity, and morphology.

## Core Capabilities

- **Extent & Persistence:** Calculate annual and monthly water occurrence, APSEC (area percentage of section covered), and refuge pool location/area.
- **Fragmentation & Geometry:** Measure pool counts, largest patch index (LPI), area-weighted shape metrics (AWRe, AWMSI), and channel-confined pool widths.
- **River Connectivity:** Measure structural connectivity along river network centrelines (RC, TCF, DCI).
- **Provenanced Pipeline:** Generate reproducible output packages with Parquet/CSV metric tables and SHA256-verified JSON manifests.

## Citation

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river, *Journal of Hydrology*, DOI [10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087).
