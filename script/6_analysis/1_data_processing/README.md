# Future Climate Data Merge - README

## Overview
This script merges stable basin attribute data with future climate projections (2025-2100) to create analysis-ready datasets for studying algal bloom drivers under different climate scenarios.

## Input Files

### 1. Stable Data (Stable data.xlsx)
23 tributary basins with constant attributes across all years.

### 2. Climate Projection CSV Files (6 files)
- nexgddp_EC-Earth3_ssp245_bc.csv
- nexgddp_EC-Earth3_ssp585_bc.csv
- nexgddp_GFDL-ESM4_ssp245_bc.csv
- nexgddp_GFDL-ESM4_ssp585_bc.csv
- nexgddp_MRI-ESM2-0_ssp245_bc.csv
- nexgddp_MRI-ESM2-0_ssp585_bc.csv

## Output Files (6 Excel files)
- Future_merged_EC-Earth3_ssp245.xlsx
- Future_merged_EC-Earth3_ssp585.xlsx
- Future_merged_GFDL-ESM4_ssp245.xlsx
- Future_merged_GFDL-ESM4_ssp585.xlsx
- Future_merged_MRI-ESM2-0_ssp245.xlsx
- Future_merged_MRI-ESM2-0_ssp585.xlsx

Each file contains:
- 1748 rows (76 years × 23 basins)
- 27 columns

## Parameter Description

### Climate Variables (from CSV)

| Column Name | Chinese | Unit | Description |
|-------------|---------|------|-------------|
| basin_id | 流域ID | - | Unique basin identifier |
| basin_name | 流域名称 | - | Basin name |
| year | 年份 | - | Year (2025-2100) |
| model | 气候模式 | - | Climate model name |
| scenario | 排放情景 | - | SSP scenario (ssp245/ssp585) |
| tas_celsius | 年平均气温 | °C | Annual mean air temperature |
| pr_annual_mm | 年降水量 | mm | Annual precipitation |
| hurs | 相对湿度 | % | Relative humidity |
| rsds | 表面短波辐射 | W/m² | Surface downwelling shortwave radiation |
| sfcWind | 表面风速 | m/s | Surface wind speed |

### Basin Attribute Variables (from Stable data.xlsx)

#### Basin Intrinsic Properties

| Column Name | Chinese | Unit | Description |
|-------------|---------|------|-------------|
| water_area_km2 | 水体面积 | km² | Water body area |
| slope_promille | 坡降 | ‰ | River slope (per mille) |
| shoreline_development_factor_SL | 岸线发育系数 | - | Shoreline development coefficient |

#### Reservoir Hydrodynamic Indicators

| Column Name | Chinese | Unit | Description |
|-------------|---------|------|-------------|
| WL_mean | 平均水位 | m | Mean water level |
| WL_daily_change | 日水位变化 | m | Daily water level change |
| WL_stability | 水位稳定性 | - | Water level stability index |
| N_rapid_rise | 快速上升次数 | count | Number of rapid water level rises |
| N_rapid_drop | 快速下降次数 | count | Number of rapid water level drops |
| N_pulse | 脉冲次数 | count | Number of water level pulses |
| Max_rise_days | 最大上升天数 | days | Maximum consecutive rise days |
| Max_drop_days | 最大下降天数 | days | Maximum consecutive drop days |

#### Human Activity Indicators

| Column Name | Chinese | Unit | Description |
|-------------|---------|------|-------------|
| cropland_area_km2 | 耕地面积 | km² | Cropland area |
| impervious_area_km2 | 不透水面积 | km² | Impervious surface area |
| HAI | 人类活动强度 | - | Human activity index |
| ratio_ci_imp | 建设用地比例 | - | Ratio of constructed + impervious area |
| ratio_imp | 不透水比例 | - | Impervious area ratio |

## Python Code

```python
import pandas as pd
import os

csv_files = [
    'nexgddp_EC-Earth3_ssp245_bc.csv',
    'nexgddp_EC-Earth3_ssp585_bc.csv',
    'nexgddp_GFDL-ESM4_ssp245_bc.csv',
    'nexgddp_GFDL-ESM4_ssp585_bc.csv',
    'nexgddp_MRI-ESM2-0_ssp245_bc.csv',
    'nexgddp_MRI-ESM2-0_ssp585_bc.csv'
]

column_mapping = {
    '水体面积_km2': 'water_area_km2',
    '坡降（‰）': 'slope_promille',
    '岸线发育系数_SL': 'shoreline_development_factor_SL',
    'ratio_ci+imp': 'ratio_ci_imp'
}

stable_df = pd.read_excel('Stable data.xlsx', header=None)

header_row = stable_df.iloc[1].tolist()
stable_df.columns = header_row
stable_df = stable_df.iloc[2:].reset_index(drop=True)

for old_col, new_col in column_mapping.items():
    if old_col in stable_df.columns:
        stable_df.rename(columns={old_col: new_col}, inplace=True)

stable_df['basin_id'] = stable_df['basin_id'].astype(int)

id_cols = ['basin_id', 'basin_name']
attribute_cols = [col for col in stable_df.columns if col not in id_cols]

print(f"Stable data loaded: {len(stable_df)} basins, {len(attribute_cols)} attributes")
print(f"Attribute columns: {attribute_cols}")

for csv_file in csv_files:
    print(f"\nProcessing: {csv_file}")
    climate_df = pd.read_csv(csv_file)
    print(f"  Climate data: {len(climate_df)} rows, columns: {climate_df.columns.tolist()}")

    merged_df = climate_df.merge(stable_df, on='basin_id', how='left')
    print(f"  Merged data: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    model_name = csv_file.split('_')[1]
    scenario = csv_file.split('_')[2].replace('.csv', '')
    output_file = f'Future_merged_{model_name}_{scenario}.xlsx'

    merged_df.to_excel(output_file, index=False)
    print(f"  Saved: {output_file}")

    missing_count = merged_df.isnull().sum().sum()
    print(f"  Missing values: {missing_count}")

print("\n=== All files processed successfully ===")
```

## How to Reuse

### For New Climate Models
1. Add new CSV filename to `csv_files` list
2. Modify `model_name` and `scenario` extraction if filename format differs
3. Update output filename pattern as needed

### For Different Stable Data Structure
1. If header row changes, adjust `header_row = stable_df.iloc[1]` to correct row
2. If column names change, update `column_mapping` dictionary
3. If data starts at different row, adjust `stable_df.iloc[2:]`

### For Additional Attributes
1. Add new column mapping to `column_mapping` dictionary
2. The merge will automatically include all columns from stable_df except id_cols

## Verification
Each output file should have:
- 1748 rows (76 years × 23 basins)
- 27 columns (10 from CSV + 17 from stable data including basin_name)
- 0 missing values

Run `verify.py` to check all outputs:
```bash
python verify.py
```