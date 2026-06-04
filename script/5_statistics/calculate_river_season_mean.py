"""Summarize CHLA median by river basin and season from TIF images and export to CSV."""

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_ROOT = REPO_ROOT / "data" / "chla_TIFF"
OUTPUT_CSV = REPO_ROOT / "result" / "5_statistics_visualization" / "test" / "river_season_chla_median.csv"


def extract_date_from_filename(filename):
    match = re.search(r"(\d{8})", filename)
    if match:
        return match.group(1)
    return None


def parse_year_month(filename):
    date_text = extract_date_from_filename(filename)
    if not date_text:
        return None, None

    year = int(date_text[:4])
    month = int(date_text[4:6])
    return year, month


def get_season_info(year, month):
    if month in (3, 4, 5):
        return year, "Spring"
    if month in (6, 7, 8):
        return year, "Summer"
    if month in (9, 10, 11):
        return year, "Autumn"
    if month in (12,):
        return year, "Winter"
    if month in (1, 2):
        return year - 1, "Winter"
    return None, None


def get_river_folder_from_path(tif_path):
    relative_path = os.path.relpath(tif_path, os.fspath(INPUT_ROOT))
    parts = relative_path.split(os.sep)
    if len(parts) > 1:
        return parts[0]
    return "unknown"


def collect_tif_files_by_river_season(root_dir):
    tif_files_by_river_season = defaultdict(lambda: defaultdict(list))
    for subdir_root, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(".tif"):
                continue

            tif_path = os.path.join(subdir_root, file)
            river_folder = get_river_folder_from_path(tif_path)
            year, month = parse_year_month(file)
            if year is None or month is None:
                continue

            season_year, season_name = get_season_info(year, month)
            if season_year is None or season_name is None:
                continue

            season_key = f"{season_year}_{season_name}"
            tif_files_by_river_season[river_folder][season_key].append(tif_path)

    return tif_files_by_river_season


def compute_season_mean_for_river(tif_files, river_name, season_key):
    all_valid_data = []
    tif_count = len(tif_files)
    tif_with_valid = 0

    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as src:
                data = src.read(1)
                nodata = src.nodata

                if nodata is not None:
                    valid_data = data[(data != nodata) & (~np.isnan(data))]
                else:
                    valid_data = data[~np.isnan(data)]

                if len(valid_data) > 0:
                    all_valid_data.extend(valid_data.flatten())
                    tif_with_valid += 1
        except Exception:
            continue

    if len(all_valid_data) > 0:
        arr = np.array(all_valid_data)
        return {
            "River_Folder": river_name,
            "Season": season_key,
            "TIF_Count": tif_count,
            "TIF_With_Valid_Count": tif_with_valid,
            "Valid_Pixel_Count": len(all_valid_data),
            "CHLA_Median": float(np.median(arr)),
        }

    return {
        "River_Folder": river_name,
        "Season": season_key,
        "TIF_Count": tif_count,
        "TIF_With_Valid_Count": tif_with_valid,
        "Valid_Pixel_Count": 0,
        "CHLA_Median": np.nan,
    }


def sort_season_key(season_key):
    season_year_text, season_name = season_key.split("_", 1)
    season_year = int(season_year_text)
    season_order = {"Spring": 1, "Summer": 2, "Autumn": 3, "Winter": 4}
    return season_year, season_order.get(season_name, 99)


def main():
    print("1. Scanning root directory for river folders and TIF files...")
    tif_files_by_river_season = collect_tif_files_by_river_season(os.fspath(INPUT_ROOT))
    if not tif_files_by_river_season:
        print("Error: No TIF files found.")
        return

    print("2. Aggregating per-river per-season median and writing CSV...")
    fieldnames = [
        "River_Folder",
        "Season",
        "TIF_Count",
        "Valid_Pixel_Count",
        "CHLA_Median",
    ]

    total_records = 0
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for river_name, season_map in sorted(tif_files_by_river_season.items(), key=lambda item: item[0]):
            print(f"\nProcessing river folder: {river_name}")
            for season_key, tif_files in sorted(season_map.items(), key=lambda item: sort_season_key(item[0])):
                print(f"  Season {season_key}, TIF count: {len(tif_files)}")
                record = compute_season_mean_for_river(tif_files, river_name, season_key)
                writer.writerow(record)
                out_file.flush()
                total_records += 1

    print(f"\nAll done. Wrote {total_records} rows.")
    print(f"CSV saved to: {os.fspath(OUTPUT_CSV)}")


if __name__ == "__main__":
    main()