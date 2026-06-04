"""Calculate CHLA median and quantile statistics by basin, year, and polygon, then export to CSV."""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import warnings
from rasterio.mask import mask

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ROOT_DIR = REPO_ROOT / "data" / "chla_TIFF"
GEOJSON_FILE = REPO_ROOT / "data" / "watermask" / "watermask_partitions_20pts.geojson"
OUTPUT_DIR = REPO_ROOT / "result" / "5_statistics_visualization" / "test"
OUTPUT_CSV = OUTPUT_DIR / "chla_median_results_20pts.csv"

warnings.filterwarnings("ignore")

VALUE_COLUMNS = ["CHLA_Min", "CHLA_P25", "CHLA_Median", "CHLA_P75", "CHLA_P90", "CHLA_Max"]


def extract_year_from_filename(filename):
    import re

    match = re.search(r"(\d{4})", filename)
    if match:
        return match.group(1)
    return "unknown"


def normalize_name(value):
    if value is None:
        return ""
    return "".join(str(value).lower().split())


def get_river_folder_from_path(tif_path):
    relative_path = os.path.relpath(tif_path, ROOT_DIR)
    parts = relative_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "unknown"


def main():
    if not os.path.exists(GEOJSON_FILE):
        print(f"Error: GeoJSON file not found: {GEOJSON_FILE}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("1. Loading polygon vector data...")
    gdf = gpd.read_file(GEOJSON_FILE)
    print(f"   Loaded {len(gdf)} polygons.")

    print("2. Scanning TIF files and organizing by folder/year...")
    tif_files_by_folder_year = defaultdict(lambda: defaultdict(list))
    for subdir_root, _, files in os.walk(ROOT_DIR):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(subdir_root, file)
                river_folder = get_river_folder_from_path(tif_path)
                year = extract_year_from_filename(file)
                tif_files_by_folder_year[river_folder][year].append(tif_path)

    if not tif_files_by_folder_year:
        print("   No .tif files found!")
        return

    total_tif_count = sum(len(tif_files) for year_map in tif_files_by_folder_year.values() for tif_files in year_map.values())
    print(f"   Found {total_tif_count} .tif files across {len(tif_files_by_folder_year)} folders.")
    print(f"   Folders: {sorted(tif_files_by_folder_year.keys())}")

    print("3. Computing yearly median per polygon and writing to CSV...")

    prop_columns = gdf.columns.drop("geometry").tolist()
    fieldnames = ["Year", "Polygon_Index", "TIF_Count", "Valid_Pixel_Count"] + VALUE_COLUMNS + prop_columns

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        total_records = 0
        folder_items = sorted(tif_files_by_folder_year.items(), key=lambda item: item[0])
        total_folders = len(folder_items)

        for folder_idx, (river_folder, year_map) in enumerate(folder_items, start=1):
            if "basin_name" in gdf.columns:
                folder_polygons = gdf[gdf["basin_name"].fillna("").map(normalize_name) == normalize_name(river_folder)]
            else:
                folder_polygons = gdf
            if folder_polygons.empty:
                folder_polygons = gdf

            print(f"\n   [{folder_idx}/{total_folders}] Processing folder: {river_folder} ({len(folder_polygons)} polygons)")

            year_items = sorted(year_map.items(), key=lambda item: item[0])
            total_years = len(year_items)
            for year_idx, (year, tif_files) in enumerate(year_items, start=1):
                print(f"\n     [{year_idx}/{total_years}] Processing year: {year} ({len(tif_files)} TIF files)")

                for poly_pos, (poly_idx, row) in enumerate(folder_polygons.iterrows(), start=1):
                    geom = [row.geometry]
                    all_valid_data = []
                    tif_count = 0

                    for tif_pos, tif_path in enumerate(tif_files, start=1):
                        sys.stdout.write(
                            f"\r     Folder {river_folder} | Year {year} [{year_idx}/{total_years}] | TIF {tif_pos}/{len(tif_files)} | Polygon {poly_pos}/{len(folder_polygons)}"
                        )
                        sys.stdout.flush()
                        try:
                            with rasterio.open(tif_path) as src:
                                if gdf.crs is not None and src.crs is not None:
                                    if gdf.crs != src.crs:
                                        geom_proj = gpd.GeoSeries([row.geometry], crs=gdf.crs).to_crs(src.crs)
                                        geom_to_use = [geom_proj.iloc[0]]
                                    else:
                                        geom_to_use = geom
                                else:
                                    geom_to_use = geom

                                nodata = src.nodata
                                try:
                                    out_image, _ = mask(src, geom_to_use, crop=True)
                                    data = out_image[0]
                                    if nodata is not None:
                                        valid_data = data[(data != nodata) & (~np.isnan(data))]
                                    else:
                                        valid_data = data[~np.isnan(data)]
                                    if len(valid_data) > 0:
                                        all_valid_data.extend(valid_data.flatten())
                                        tif_count += 1
                                except ValueError:
                                    pass
                                except Exception:
                                    pass
                        except Exception:
                            pass

                    if len(all_valid_data) > 0:
                        arr = np.array(all_valid_data)
                        chla_min = np.min(arr)
                        chla_p25 = np.percentile(arr, 25)
                        chla_median = np.median(arr)
                        chla_p75 = np.percentile(arr, 75)
                        chla_p90 = np.percentile(arr, 90)
                        chla_max = np.max(arr)
                        pixel_count = len(all_valid_data)
                    else:
                        chla_min = np.nan
                        chla_p25 = np.nan
                        chla_median = np.nan
                        chla_p75 = np.nan
                        chla_p90 = np.nan
                        chla_max = np.nan
                        pixel_count = 0

                    row_props = row.drop("geometry").to_dict()
                    record = {
                        "Year": year,
                        "Polygon_Index": poly_idx,
                        "TIF_Count": tif_count,
                        "Valid_Pixel_Count": pixel_count,
                        "CHLA_Min": chla_min,
                        "CHLA_P25": chla_p25,
                        "CHLA_Median": chla_median,
                        "CHLA_P75": chla_p75,
                        "CHLA_P90": chla_p90,
                        "CHLA_Max": chla_max,
                    }
                    record.update(row_props)
                    writer.writerow(record)
                    out_file.flush()
                    total_records += 1
                    del all_valid_data
                    sys.stdout.write("\n")

    print(f"\nAll calculations complete! Generated {total_records} records.")
    print(f"Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
