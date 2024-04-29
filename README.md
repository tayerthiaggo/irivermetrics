# iRiverMetrics
![Alt text](docs/logo.jpg)
## Overview

**iRiverMetrics** is an open-source Python toolkit designed for analysing the surface water dynamics of intermittent rivers. It offers a set of modules to help researchers and environmental professionals to detect water and compute ecohydrological metrics from multispectral satellite imagery efficiently.

## Key Features

- **Modular Design:** Divided into two modules, each serving a specific purpose. This modular approach allows you to use only the components relevant to your project.

- **Remote Sensing Integration:** Leverages multispectral and multitemporal satellite imagery, enabling you to analyse surface water features and assess river characteristics. It supports common satellite sensors and data formats.

- **Efficient Processing:** Employs Dask for distributed computing, ensuring rapid processing of large-scale datasets.

- **User-Friendly:** Suitable for users with varying levels of expertise. It includes detailed documentation and code comments to guide you through the process.

## Modules

iRiverMetrics consists of two main modules:

1. Water Detection ([`waterdetect_batch`)](docs/module1.md)): Generate water masks from multispectral imagery using the Water Detect package. It integrates spectral water indices and clustering techniques to delineate and map aquatic bodies accurately.

2. Calculate Metrics ([`calculate_metrics`)](docs/module2.md)): Utilises the water masks to compute a range of metrics comprising various aspects of river surface water, such as morphological characteristics, water persistence, and fragmentation.

## Getting Started

To get started with iRiverMetrics, follow these steps:

1. **Clone the Repository:** Clone the iRiverMetrics repository from GitHub to your local machine.

```bash
git clone https://github.com/tayerthiaggo/iRiverMetrics.git
```

2. **Requirements:** Ensure Python 3.x is installed. Set up the environment and install dependencies using:

(install GDAL with conda for less headache)
```bash
conda create -n irivermetrics python=3.x
conda activate irivermetrics

conda install conda-forge::gdal
pip install -r requirements.txt
```

3. **Explore the Modules:** Dive into the documentation for each module to understand their functionality and usage.

4. **Example Usage:** Review example use cases and code snippets in the documentation of each module ([wd_batch](docs/module1.md) and [calc_metrics](docs/module2.md)) apply iRiverMetrics effectively to your projects.

5. **Contribute:** Contributions are welcome! If you have enhancements or additional features, please consider contributing back to the project via GitHub.

## Usage Example
```python
# Add cloned directory to path
import sys
sys.path.append(r'path\to\clone\irivermetrics')

# Import modules
from src.irm_main import waterdetect_batch, calculate_metrics

## Module 1
# Define input non-optional parameters

# Path to a directory containing multispectral images (e.g., TIFF files)
input_img = "path/to/images"
# Path to the river corridor extent shapefile (.shp)
r_lines = "path/to/rcor_extent.shp"
# Generate a DataArray containing water masks based on the specified parameters
da_wmask = waterdetect_batch(input_img, r_lines)

## Module 2
# Path to the river corridor extent (sections) shapefile (.shp)
rcor_extent = "path/to/rcor_extent.shp"
# Section length in km
section_length = 0.484 #Adjust as needed
# Define minimum pool size in pixels
min_pool_size=4 #Adjust as needed

# Calculate river metrics
calculate_metrics(da_wmask, rcor_extent, section_length, min_pool_size)
```

## Citation

If you use iRiverMetrics in your research or projects, please consider citing the original paper:

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river, Journal of Hydrology, Volume 617, Part C, DOI:[10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087)

and 

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Identifying intermittent river sections with similar hydrology using remotely sensed metrics, Journal of Hydrology, Volume 626, Part A, DOI:[10.1016/j.jhydrol.2023.130266](https://doi.org/10.1016/j.jhydrol.2023.130266)