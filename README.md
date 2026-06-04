# TGR — Three Gorges Reservoir Chlorophyll-a Estimation & Algal Bloom Driver Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![R 4.0+](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

Satellite-based chlorophyll-a (Chl-a) concentration estimation for tributary bays of the Three Gorges Reservoir (TGR) using Sentinel-2 imagery, combined with statistical driver analysis and future climate scenario predictions of algal blooms.

## Overview

This repository implements a complete workflow for:

1. **Data Acquisition** — Download Sentinel-2 L2A imagery from Google Earth Engine and Microsoft Planetary Computer
2. **Satellite–In-Situ Matching** — Spatially and temporally match satellite pixels with field-measured Chl-a data
3. **Algorithm Development** — Build and validate NDCI–Chl-a empirical models (linear, polynomial, log-linear)
4. **Spatial Inference** — Apply cloud/water/TI masking and invert Chl-a from SR imagery
5. **Time-Series Statistics** — Compute seasonal/yearly Chl-a medians and per-polygon trend slopes
6. **Driver Analysis & Future Prediction** — GAMM-based algal bloom driver identification + CMIP6 future projections

## Repository Structure

```
TGR/
├── script/
│   ├── 1_download/            # Sentinel-2 image acquisition
│   │   ├── gee_download.py          # GEE-based download
│   │   ├── PC_download.py           # Planetary Computer STAC download
│   │   ├── convert_sentinel2_dn_to_sr.py  # DN → Surface Reflectance
│   │   └── three_gorges_watermask.geojson  # ROI water mask
│   ├── 2_match/               # Satellite–in-situ matching
│   │   └── match.py                 # Pixel extraction + NDCI computation
│   ├── 3_modeling/            # NDCI–Chl-a algorithm
│   │   └── NDCI_ChlA_Algorithm.py   # Model training, CV, evaluation
│   ├── 4_inference/           # Chl-a spatial inversion
│   │   └── mask_with_MNDWI_TI_NDCI_ploy_inversion.py
│   ├── 5_statistics/          # Time-series & statistics
│   │   ├── calculate_chla_median_by_year.py
│   │   ├── calculate_folder_tif_median.py
│   │   ├── calculate_river_season_mean.py
│   │   ├── plot_chla_median_all_river.py
│   │   ├── polygon_slope_by_point_index.py
│   │   ├── tif_to_plot.py
│   │   └── watermask_partitions_20pts.geojson
│   └── 6_analysis/            # Statistical analysis & prediction
│       ├── 1_data_processing/       # Data merging & preprocessing
│       ├── 2_train_test/            # GAMM cross-validation
│       ├── 3_model_test/            # Model testing
│       ├── 4_driver_analysis/       # Algal bloom driver analysis
│       └── 5_future_prediction/     # CMIP6 future scenario predictions
├── result/                    # Output data & figures
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

## Pipeline

```
Sentinel-2 L2A  ──►  DN → SR Conversion  ──►  Pixel Sampling (9-point)
                                                      │
                                          In-situ Chl-a Matching
                                                      │
                                          NDCI Computation + CV Filter
                                                      │
                                          NDCI–Chl-a Model (Train/Test)
                                                      │
                                          Spatial Chl-a Inversion
                                            (MNDWI + Cloud + TI masks)
                                                      │
                                    ┌─────────────────┴─────────────────┐
                                    ▼                                   ▼
                          Time-Series Statistics              Driver Analysis (GAMM)
                          (Yearly/Seasonal Medians,           + Future Prediction
                           Per-Polygon Trend Slopes)          (CMIP6: 3 GCMs × 2 SSPs)
```

## Data Sources

| Data | Source | Description |
|------|--------|-------------|
| Sentinel-2 L2A | [GEE](https://earthengine.google.com/) / [Planetary Computer](https://planetarycomputer.microsoft.com/) | Multispectral imagery at 10–60 m resolution |
| In-situ Chl-a | Field measurements | Three Gorges Reservoir tributary bays |
| NEX-GDDP-CMIP6 | [NASA Earth Exchange](https://www.nasa.gov/nex/) | Downscaled climate projections (EC-Earth3, GFDL-ESM4, MRI-ESM2-0) |
| Reservoir attributes | TGR operation records | Water level, hydrodynamics, land use |

## Setup

### Prerequisites

- Python 3.9+ (with Conda recommended)
- R 4.0+ (for driver analysis and future prediction scripts)
- Google Earth Engine account (for `gee_download.py`)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/TGR.git
cd TGR

# Create and activate Conda environment
conda create -n tgr python=3.9
conda activate tgr

# Install Python dependencies
pip install -r requirements.txt
```

### R Package Dependencies

The analysis scripts in `script/6_analysis/` require the following R packages:

```r
install.packages(c("dplyr", "tidyr", "mgcv", "ggplot2", "patchwork", "readxl", "scales"))
```

### Configuration

1. **Google Earth Engine**: Update `PROJECT_ID` in `script/1_download/gee_download.py` with your GEE project ID.

2. **Proxy Settings**: Both `gee_download.py` and `PC_download.py` have proxy configurations — adjust `PROXY_PORT` or comment out proxy setup as needed.

3. **Model Coefficients**: The Chl-a inversion coefficients in `script/4_inference/mask_with_MNDWI_TI_NDCI_ploy_inversion.py` are derived from the best-fit polynomial model. Re-run `script/3_modeling/NDCI_ChlA_Algorithm.py` to update coefficients for new data.

## Usage

### 1. Download Sentinel-2 Imagery

```bash
# Using Google Earth Engine
python script/1_download/gee_download.py

# Or using Microsoft Planetary Computer
python script/1_download/PC_download.py
```

### 2. Convert DN to Surface Reflectance

```bash
python script/1_download/convert_sentinel2_dn_to_sr.py
```

### 3. Match Satellite Pixels with In-Situ Data

```bash
python script/2_match/match.py --match-max-hours 48
```

### 4. Build NDCI–Chl-a Algorithm

```bash
python script/3_modeling/NDCI_ChlA_Algorithm.py
```

### 5. Spatial Chl-a Inversion

```bash
python script/4_inference/mask_with_MNDWI_TI_NDCI_ploy_inversion.py
```

### 6. Time-Series Statistics

```bash
python script/5_statistics/calculate_chla_median_by_year.py
python script/5_statistics/polygon_slope_by_point_index.py
```

### 7. Driver Analysis & Future Prediction

```bash
# R scripts (run in order)
Rscript script/6_analysis/1_data_processing/match_chla_history.R
Rscript script/6_analysis/2_train_test/CV_GAMM_v1.R
Rscript script/6_analysis/3_model_test/test_model_bundle.R
Rscript script/6_analysis/4_driver_analysis/driver_analysis_4groups.R
Rscript script/6_analysis/5_future_prediction/future_prediction.R
```

## Key Methods

### NDCI (Normalized Difference Chlorophyll Index)

```
NDCI = (B5 − B4) / (B5 + B4)
```

where B4 (665 nm) and B5 (705 nm) are Sentinel-2 red-edge bands.

### Pixel Sampling & CV Filtering

For each site polygon, 9 uniformly distributed pixels are sampled. Groups with spatial CV ≥ 0.3 undergo iterative outlier removal; groups with fewer than 3 remaining valid pixels are discarded.

### Masking (Spatial Inference)

- **Water mask**: Dynamic MNDWI > 0.3 (B3 − B11 / B3 + B11)
- **Cloud mask**: SCL classes 3/8/9/10 + opaque + cirrus bands
- **TI (Turbidity Index) mask**: B4 − B3 − (B8 − B3) × 105/273 > 0

### Model Selection

Four candidate models with stratified 5-fold CV (3 repeats):
- M1: Linear regression
- M2: Quadratic polynomial
- M3: Cubic polynomial
- M4: Log-linear with Duan's smearing correction

Best model selected by lowest CV-RMSE.

### Driver Analysis (GAMM)

Generalized Additive Mixed Models (GAMM) with basin-level random effects to identify key environmental drivers of Chl-a variability.

### Future Prediction

CMIP6 downscaled projections (NEX-GDDP) under SSP2-4.5 and SSP5-8.5 scenarios, using three climate models (EC-Earth3, GFDL-ESM4, MRI-ESM2-0) for ensemble prediction.

## Study Area

The Three Gorges Reservoir (三峡水库) is one of the world's largest reservoirs, located on the Yangtze River in China. This study focuses on its tributary bays, which experience recurring algal blooms influenced by hydrological operations, nutrient loads, and climatic factors.

## Output Description

| Directory | Content |
|-----------|---------|
| `result/2_matched/` | Matched satellite–in-situ data tables |
| `result/3_model/` | Model CV results, equation, test evaluation |
| `result/4_inference/` | Chl-a GeoTIFF maps |
| `result/5_statistics_visualization/` | Yearly/seasonal statistics, trend plots |
| `result/6_analysis/` | Driver analysis, future predictions, variable importance |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{TGR_chla,
  author = {Wang, Ziyi},
  title = {TGR: Three Gorges Reservoir Chlorophyll-a Estimation \& Algal Bloom Driver Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/TGR}
}
```
