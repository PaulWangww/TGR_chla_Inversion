"""Merge future climate projections with stable basin attributes for algae bloom driver analysis."""

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