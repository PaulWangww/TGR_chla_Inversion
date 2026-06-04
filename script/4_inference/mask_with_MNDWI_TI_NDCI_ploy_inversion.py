"""Invert Chl-a from SR Sentinel-2 imagery using cloud, water, and TI masks before inversion."""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import rasterio


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
INPUT_ROOT = REPO_ROOT / "data" / "SR_TIFF"
OUTPUT_ROOT = REPO_ROOT / "data" / "chla_TIFF" / "test"
SUMMARY_CSV = OUTPUT_ROOT / "mask_summary.csv"
MASKED_PIXEL_COUNTS_CSV = OUTPUT_ROOT / "masked_pixel_counts.csv"

CHLA_COEF_0 = 3.441597
CHLA_COEF_1 = 127.299647
CHLA_COEF_2 = 1498.584858

MNDWI_THRESHOLD = 0.3
USE_CLOUD_MASK = True
NODATA_VALUE = -9999.0
CHLA_MIN = 0.0
CHLA_MAX = 200.0
TI_WEIGHT = 105.0 / 273.0


@dataclass(frozen=True)
class InferenceResult:
    source: str
    output_tif: str
    valid_pixels_before_ti: int
    masked_pixels_ti_gt_0: int
    valid_pixels_after_ti: int
    min_value: float | None
    max_value: float | None
    mean_value: float | None


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = a / b
    out[~np.isfinite(out)] = np.nan
    return out


def _read_band(src: rasterio.io.DatasetReader, band_index: int) -> np.ndarray:
    return src.read(band_index, masked=True).astype(np.float32).filled(np.nan)


def _compute_dynamic_water_mask_on_src_grid(src: rasterio.io.DatasetReader) -> np.ndarray:
    if src.count < 11:
        return np.zeros((src.height, src.width), dtype=bool)

    b3 = _read_band(src, 3)
    b11 = _read_band(src, 11)

    mndwi = _safe_divide(b3 - b11, b3 + b11)
    return np.isfinite(mndwi) & (mndwi > MNDWI_THRESHOLD)


def _load_cloud_mask_on_src_grid(src: rasterio.io.DatasetReader) -> np.ndarray:
    cloud_type = np.zeros((src.height, src.width), dtype=np.uint8)

    if src.count >= 15:
        scl = src.read(
            15,
            out_shape=(src.height, src.width),
            resampling=rasterio.enums.Resampling.nearest,
            masked=True,
        ).filled(0)
        scl = scl.astype(np.int16)
        cloud_type[np.isin(scl, [3])] = 1
        cloud_type[np.isin(scl, [8, 9])] = 2
        cloud_type[np.isin(scl, [10])] = 3

    if src.count >= 21:
        opaque = src.read(
            21,
            out_shape=(src.height, src.width),
            resampling=rasterio.enums.Resampling.nearest,
            masked=True,
        ).filled(0)
        cloud_type[(opaque > 0) & (cloud_type == 0)] = 2

    if src.count >= 22:
        cirrus = src.read(
            22,
            out_shape=(src.height, src.width),
            resampling=rasterio.enums.Resampling.nearest,
            masked=True,
        ).filled(0)
        cloud_type[(cirrus > 0) & (cloud_type == 0)] = 3

    return cloud_type > 0


def _infer_chla_polynomial(
    src: rasterio.io.DatasetReader,
    water_mask: np.ndarray,
    cloud_mask: np.ndarray | None,
) -> np.ndarray:
    if src.count < 5:
        raise ValueError(f"{src.name} does not have required bands B4/B5, count={src.count}")

    b4 = _read_band(src, 4)
    b5 = _read_band(src, 5)

    ndci = _safe_divide(b5 - b4, b5 + b4)
    pred = CHLA_COEF_0 + CHLA_COEF_1 * ndci + CHLA_COEF_2 * (ndci ** 2)

    pred = np.where(np.isfinite(pred), pred, np.nan)
    pred = np.clip(pred, CHLA_MIN, CHLA_MAX)
    if cloud_mask is not None:
        pred[cloud_mask] = np.nan
    pred[~water_mask] = np.nan
    return pred


def _save_single_channel_tif(src: rasterio.io.DatasetReader, output_path: Path, chla_map: np.ndarray) -> None:
    profile = src.profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA_VALUE, compress="lzw")

    output_arr = np.where(np.isfinite(chla_map), chla_map, NODATA_VALUE).astype(np.float32)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_arr, 1)


def _compute_ti_on_src_grid(src: rasterio.io.DatasetReader) -> np.ndarray:
    if src.count < 8:
        return np.zeros((src.height, src.width), dtype=bool)

    b3 = _read_band(src, 3)
    b4 = _read_band(src, 4)
    b8 = _read_band(src, 8)

    ti = b4 - b3 - (b8 - b3) * TI_WEIGHT
    ti = np.where(np.isfinite(ti), ti, np.nan)
    return np.isfinite(ti) & (ti > 0)


def _output_path_for(source_path: Path) -> Path:
    return OUTPUT_ROOT / source_path.relative_to(INPUT_ROOT)


def _build_chla_map_for_path(tif_path: Path) -> tuple[np.ndarray, rasterio.io.Profile, int, int, int]:
    with rasterio.open(tif_path) as src:
        water_mask = _compute_dynamic_water_mask_on_src_grid(src)
        cloud_mask = _load_cloud_mask_on_src_grid(src) if USE_CLOUD_MASK else None
        ti_mask = _compute_ti_on_src_grid(src)
        pre_ti_mask = water_mask if cloud_mask is None else (water_mask & ~cloud_mask)
        before_valid = int(np.count_nonzero(pre_ti_mask))
        masked_pixels = int(np.count_nonzero(pre_ti_mask & ti_mask))
        combined_mask = pre_ti_mask & ~ti_mask
        after_valid = int(np.count_nonzero(combined_mask))
        chla_map = _infer_chla_polynomial(src, combined_mask, None)
        profile = src.profile.copy()
    return chla_map, profile, before_valid, masked_pixels, after_valid


def _write_output(src_profile: rasterio.io.Profile, output_path: Path, chla_map: np.ndarray) -> None:
    profile = src_profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA_VALUE, compress="lzw")
    output_arr = np.where(np.isfinite(chla_map), chla_map, NODATA_VALUE).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_arr, 1)
        dst.set_band_description(1, "Chl_a_dynamic_pc_ti_masked")


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(INPUT_ROOT.rglob("*.tif"))
    if not tif_files:
        print(f"No tif files found under: {INPUT_ROOT}")
        return

    print(f"Input root: {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Found {len(tif_files)} Sentinel2 tif files")
    print("Using polynomial model only")
    print(f"Using dynamic water mask: MNDWI > {MNDWI_THRESHOLD}")
    if USE_CLOUD_MASK:
        print("Using cloud mask")
    else:
        print("Cloud mask disabled")

    results: list[InferenceResult] = []
    failed_items: list[dict[str, str]] = []

    for i, tif_path in enumerate(tif_files, start=1):
        try:
            output_path = _output_path_for(tif_path)
            chla_map, profile, before_valid, masked_pixels, after_valid = _build_chla_map_for_path(tif_path)
            _write_output(profile, output_path, chla_map)

            finite_values = chla_map[np.isfinite(chla_map)]
            results.append(
                InferenceResult(
                    source=str(tif_path),
                    output_tif=str(output_path),
                    valid_pixels_before_ti=before_valid,
                    masked_pixels_ti_gt_0=masked_pixels,
                    valid_pixels_after_ti=after_valid,
                    min_value=float(np.min(finite_values)) if finite_values.size else None,
                    max_value=float(np.max(finite_values)) if finite_values.size else None,
                    mean_value=float(np.mean(finite_values)) if finite_values.size else None,
                )
            )
            print(f"[{i}/{len(tif_files)}] Saved: {output_path.name}")
        except Exception as exc:
            failed_items.append({"source": str(tif_path), "error": str(exc)})
            print(f"[{i}/{len(tif_files)}] Failed: {tif_path.name} -> {exc}")

    fieldnames = [
        "status",
        "source",
        "output_tif",
        "valid_pixels_before_ti",
        "masked_pixels_ti_gt_0",
        "valid_pixels_after_ti",
        "min_value",
        "max_value",
        "mean_value",
        "error",
    ]
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = asdict(item)
            row["status"] = "success"
            row["error"] = ""
            writer.writerow(row)
        for item in failed_items:
            writer.writerow(
                {
                    "status": "failed",
                    "source": item["source"],
                    "output_tif": "",
                    "valid_pixels_before_ti": "",
                    "masked_pixels_ti_gt_0": "",
                    "valid_pixels_after_ti": "",
                    "min_value": "",
                    "max_value": "",
                    "mean_value": "",
                    "error": item["error"],
                }
            )

    masked_count_fieldnames = [
        "source",
        "output_tif",
        "masked_pixels_ti_gt_0",
        "valid_pixels_before_ti",
        "valid_pixels_after_ti",
    ]
    with MASKED_PIXEL_COUNTS_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=masked_count_fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow({field: getattr(item, field) for field in masked_count_fieldnames})

    print(
        f"Done. Success={len(results)}, Failed={len(failed_items)}, "
        f"ValidPixels={int(sum(item.valid_pixels_after_ti for item in results))}"
    )
    print(f"Summary saved: {SUMMARY_CSV}")
    print(f"Masked pixel counts saved: {MASKED_PIXEL_COUNTS_CSV}")


if __name__ == "__main__":
    main()
