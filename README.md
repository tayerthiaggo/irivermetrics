# iRiverMetrics

## Overview

**iRiverMetrics** is an open-source Python toolkit designed for analysing intermittent rivers surface water patterns. It offers a comprehensive set of modules to help researchers and environmental professionals analyse intermittent rivers characteristics based on remote sensing data. With iRiverMetrics, you can efficiently process multispectral imagery to detect water and calculate meaningful metrics.

## Key Features

- **Modular Design:** iRiverMetrics is divided into modules, each serving a specific purpose, from generating water masks to calculating river metrics. This modular approach allows you to use only the components relevant to your project.

- **Remote Sensing Integration:** The toolkit leverages multispectral and multitemporal satellite imagery, enabling you to analyse surface water features and assess river characteristics. It supports common satellite sensors and data formats.

- **Efficient Processing:** iRiverMetrics is designed for efficiency. It employs Dask for parallel processing, making it suitable for analysing large datasets quickly.

- **User-Friendly:** While powerful, iRiverMetrics is approachable for users with varying levels of expertise. It includes detailed documentation and code comments to guide you through the process.

## Modules

iRiverMetrics consists of two main modules:

1. Water Detection ([`wd_batch`)](docs/module1.md)): Generate water masks from multispectral imagery using the Water Detect package. This module identifies open water features, combines spectral water indices, and clusters them to map water bodies.

2. Calculate Metrics ([`calc_metrics`)](docs/module2.md)): Calculate a variety of river metrics based on water masks. These metrics provide insights into surface water characteristics, such as morphology, persistence and fragmentation.

## Getting Started

To get started with iRiverMetrics, follow these steps:

1. **Clone the Repository:** Clone the iRiverMetrics repository from GitHub to your local machine.

```bash
git clone https://github.com/tayerthiaggo/iRiverMetrics.git
```

2. **Requirements:** Make sure you have Python 3.x installed on your system. Install the required dependencies by running the following command:

```bash
pip install -r requirements.txt
```

3. **Explore the Modules:** Dive into the documentation for each module to understand its functionality and usage.

4. **Example Usage:** Review example use cases and code snippets in the documentation of each module ([wd_batch](docs/module1.md) and [calc_metrics](docs/module2.md))to see how iRiverMetrics can be applied to your specific research or projects. 

5. **Contribute:** If you find iRiverMetrics useful and have improvements or contributions to make, consider contributing to the open-source project on GitHub.

## Citation

If you use iRiverMetrics in your research or projects, please consider citing the original paper:

Tayer T.C., Beesley L.S., Douglas M.M., Bourke S.A., Meredith K., McFarlane D. (2023) Ecohydrological metrics derived from multispectral images to characterize surface water in an intermittent river, Journal of Hydrology, Volume 617, Part C, DOI:[10.1016/j.jhydrol.2023.129087](https://doi.org/10.1016/j.jhydrol.2023.129087)