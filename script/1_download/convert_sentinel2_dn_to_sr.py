"""Batch convert Sentinel-2 DN images to surface reflectance while preserving the original band structure."""

from __future__ import annotations

import csv
import re
from datetime import date
from pathlib import Path

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = REPO_ROOT / "data" / "DN_TIFF"
OUTPUT_ROOT = REPO_ROOT / "data" / "SR_TIFF" 
SUMMARY_CSV = OUTPUT_ROOT / "conversion_summary.csv"

BASELINE_DATE = date(2022, 1, 25)
SCIENCE_BANDS = {"B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"}


def _parse_date_from_name(path: Path) -> date:
    match = re.search(r"(\d{8})", path.name)
    if not match:
        raise ValueError(f"Cannot parse YYYYMMDD date from file name: {path.name}")
    value = match.group(1)
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def _output_path_for(input_path: Path) -> Path:
    relative_path = input_path.relative_to(INPUT_ROOT)
    return OUTPUT_ROOT / relative_path.parent / f"{input_path.stem}_reflectance{input_path.suffix}"


def _convert_science_band(arr: np.ndarray, capture_date: date) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan

    if capture_date >= BASELINE_DATE:
        out = np.maximum(out, 1000.0) - 1000.0

    out = out / 10000.0
    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32, copy=False)


def _copy_aux_band(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = np.nan
    return out


def convert_file(input_path: Path, output_path: Path) -> dict[str, object]:
    capture_date = _parse_date_from_name(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        descriptions = tuple(desc or f"band_{i}" for i, desc in enumerate(src.descriptions, start=1))
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=np.nan, compress="deflate", tiled=True)

        converted = []
        converted_band_count = 0
        for band_index, band_name in enumerate(descriptions, start=1):
            arr = src.read(band_index, masked=True).astype(np.float32).filled(np.nan)
            if band_name in SCIENCE_BANDS:
                converted.append(_convert_science_band(arr, capture_date))
                converted_band_count += 1
            else:
                converted.append(_copy_aux_band(arr))

        stack = np.stack(converted, axis=0).astype(np.float32)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stack)
        for band_index, band_name in enumerate(descriptions, start=1):
            dst.set_band_description(band_index, band_name)

    science_values = stack[[i for i, band_name in enumerate(descriptions) if band_name in SCIENCE_BANDS]]
    finite = science_values[np.isfinite(science_values)]

    return {
        "source": str(input_path),
        "output_tif": str(output_path),
        "date": capture_date.isoformat(),
        "baseline_rule": "post_2022_01_25_subtract_1000" if capture_date >= BASELINE_DATE else "pre_2022_01_25_divide_10000",
        "band_count": int(stack.shape[0]),
        "converted_science_bands": int(converted_band_count),
        "science_min": float(np.min(finite)) if finite.size else None,
        "science_max": float(np.max(finite)) if finite.size else None,
        "science_mean": float(np.mean(finite)) if finite.size else None,
    }


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(INPUT_ROOT.rglob("*.tif"))
    if not tif_files:
        print(f"No tif files found under: {INPUT_ROOT}")
        return

    results = []
    failed = []
    for i, input_path in enumerate(tif_files, start=1):
        output_path = _output_path_for(input_path)
        try:
            result = convert_file(input_path, output_path)
            results.append(result)
            print(f"[{i}/{len(tif_files)}] Saved: {output_path.relative_to(OUTPUT_ROOT)}")
        except Exception as exc:
            failed.append({"source": str(input_path), "error": str(exc)})
            print(f"[{i}/{len(tif_files)}] Failed: {input_path.name} -> {exc}")

    fieldnames = [
        "status",
        "source",
        "output_tif",
        "date",
        "baseline_rule",
        "band_count",
        "converted_science_bands",
        "science_min",
        "science_max",
        "science_mean",
        "error",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out = {"status": "success", **row, "error": ""}
            writer.writerow(out)
        for row in failed:
            writer.writerow(
                {
                    "status": "failed",
                    "source": row["source"],
                    "output_tif": "",
                    "date": "",
                    "baseline_rule": "",
                    "band_count": "",
                    "converted_science_bands": "",
                    "science_min": "",
                    "science_max": "",
                    "science_mean": "",
                    "error": row["error"],
                }
            )

    print(f"Done. Success={len(results)}, Failed={len(failed)}")
    print(f"Summary saved: {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
