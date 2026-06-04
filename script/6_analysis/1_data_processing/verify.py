import pandas as pd
import os

files = [
    'Future_merged_EC-Earth3_ssp245.xlsx',
    'Future_merged_EC-Earth3_ssp585.xlsx',
    'Future_merged_GFDL-ESM4_ssp245.xlsx',
    'Future_merged_GFDL-ESM4_ssp585.xlsx',
    'Future_merged_MRI-ESM2-0_ssp245.xlsx',
    'Future_merged_MRI-ESM2-0_ssp585.xlsx'
]

print('=== Verification Results ===')
for f in files:
    if os.path.exists(f):
        df = pd.read_excel(f)
        print(f)
        print('  Rows:', len(df), ', Columns:', len(df.columns))
        print('  Years:', df['year'].min(), '-', df['year'].max())
        print('  Basins:', df['basin_id'].nunique())
        print('  Missing:', df.isnull().sum().sum())
        print()
    else:
        print('  ' + f + ': NOT FOUND!')