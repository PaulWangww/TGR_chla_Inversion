import json
import os
import re
import time
from pathlib import Path

import ee
import geemap
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom


PROJECT_ID = "ivory-haven-475508-f6"
COLLECTION = "COPERNICUS/S2_SR"
SATELLITE_TAG = "S2"
DATA_SOURCE_TAG = "DN"
START_DATE = "1900-01-01"
# Earth Engine filterDate uses an exclusive end date, so this includes 2021-01-01.
END_DATE_EXCLUSIVE = "2026-05-02"
SCALE = 10
AUTO_RESTART_DELAY_SECONDS = 20
PROXY_PORT = 10090
PC_SCIENCE_BAND_PAIRS = [
    ("AOT", "AOT"),
    ("B1", "B01"),
    ("B2", "B02"),
    ("B3", "B03"),
    ("B4", "B04"),
    ("B5", "B05"),
    ("B6", "B06"),
    ("B7", "B07"),
    ("B8", "B08"),
    ("B9", "B09"),
    ("B11", "B11"),
    ("B12", "B12"),
    ("B8A", "B8A"),
    ("SCL", "SCL"),
    ("WVP", "WVP"),
]
OUTPUT_BAND_NAMES = [dst for _, dst in PC_SCIENCE_BAND_PAIRS] + ["preview", "visual"]


def configure_proxy(port: int):
    proxy = f"http://127.0.0.1:{port}"
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy

if not os.environ.get("HTTP_PROXY") and not os.environ.get("HTTPS_PROXY"):
    configure_proxy(PROXY_PORT)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
GEOJSON_PATH = REPO_ROOT / "data" / "watermask" / "three_gorges_watermask.geojson"
OUT_BASE_DIR = BASE_DIR / "test"
OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)


def initialize_ee():
    try:
        ee.Initialize(project=PROJECT_ID)
        print("Earth Engine initialized.")
    except Exception as exc:
        proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "not set"
        raise RuntimeError(
            f"Earth Engine initialization failed: {exc}. Current proxy: {proxy}"
        ) from exc


def load_geojson_features(geojson_path: Path):
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")

    with geojson_path.open("r", encoding="utf-8") as f:
        geojson = json.load(f)

    features = geojson.get("features", [])
    if not features:
        raise ValueError("GeoJSON contains no features.")

    return features


def get_river_name(feature: dict) -> str:
    props = feature.get("properties", {})
    return props.get("basin_name") or props.get("name") or f"river_{props.get('basin_id', 'unknown')}"


def get_existing_time_keys(directory: Path, satellite_tag: str, source_tag: str):
    existing_time_keys = set()
    pattern = re.compile(
        rf"^(\d{{8}})_.+_{re.escape(satellite_tag)}_{re.escape(source_tag)}\.tif$",
        re.IGNORECASE,
    )

    for path in directory.glob("*.tif"):
        match = pattern.match(path.name)
        if match and tif_has_expected_bands(path, OUTPUT_BAND_NAMES):
            existing_time_keys.add(match.group(1))

    return existing_time_keys


def tif_has_expected_bands(path: Path, expected_band_names):
    try:
        with rasterio.open(path) as src:
            return (
                src.count == len(expected_band_names)
                and tuple(src.descriptions) == tuple(expected_band_names)
            )
    except Exception:
        return False


def build_roi(geometry: dict):
    return ee.Feature(ee.Geometry(geometry)).geometry()


def build_collection(roi):
    return (
        ee.ImageCollection(COLLECTION)
        .filterBounds(roi)
        .filterDate(START_DATE, END_DATE_EXCLUSIVE)
        .sort("system:time_start")
    )


def build_pc_style_band_stack(image):
    science_bands = [
        image.select(src_name).rename(out_name)
        for src_name, out_name in PC_SCIENCE_BAND_PAIRS
    ]
    visual_red = image.select("TCI_R")
    return ee.Image.cat(
        science_bands
        + [
            visual_red.rename("preview"),
            visual_red.rename("visual"),
        ]
    )


def prepare_image_for_export(image, roi):
    roi_mask = ee.Image.constant(1).clip(roi).mask()
    return (
        build_pc_style_band_stack(image)
        .toFloat()
        .updateMask(roi_mask)
    )


def apply_geometry_mask(tif_path: Path, geometry: dict):
    tmp_path = tif_path.with_name(f"{tif_path.stem}.tmp{tif_path.suffix}")
    try:
        with rasterio.open(tif_path) as src:
            if not src.crs:
                raise ValueError(f"Downloaded GeoTIFF has no CRS: {tif_path}")

            data = src.read().astype(np.float32)
            geom_proj = transform_geom("EPSG:4326", src.crs, geometry)
            mask_arr = rasterize(
                [(geom_proj, 1)],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0,
                dtype="uint8",
            )
            data[:, mask_arr != 1] = np.nan

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                count=data.shape[0],
                dtype="float32",
                nodata=np.nan,
                compress="lzw",
            )

        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(data)
            for index, band_name in enumerate(OUTPUT_BAND_NAMES, start=1):
                dst.set_band_description(index, band_name)

        tmp_path.replace(tif_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def download_image(image, out_path: Path, roi, geometry: dict):
    export_image = prepare_image_for_export(image, roi)
    export_crs = image.select("B2").projection().crs().getInfo()
    geemap.download_ee_image(
        export_image,
        str(out_path),
        scale=SCALE,
        crs=export_crs,
        region=roi.bounds(1),
        overwrite=True,
    )
    apply_geometry_mask(out_path, geometry)


def main():
    initialize_ee()
    features = load_geojson_features(GEOJSON_PATH)

    for feature_index, feature in enumerate(features, start=1):
        river = get_river_name(feature)
        geometry = feature.get("geometry")
        if not geometry:
            print(f"[{feature_index}/{len(features)}] Skip {river}: missing geometry.")
            continue

        out_dir = OUT_BASE_DIR / river
        out_dir.mkdir(parents=True, exist_ok=True)
        existing_keys = get_existing_time_keys(out_dir, SATELLITE_TAG, DATA_SOURCE_TAG)

        roi = build_roi(geometry)
        collection = build_collection(roi)
        count = collection.size().getInfo()

        print(
            f"\n=== River: {river} | images before {END_DATE_EXCLUSIVE}: {count} | "
            f"existing dates: {len(existing_keys)} ==="
        )
        if count == 0:
            continue

        image_list = collection.toList(count)
        for image_index in range(count):
            image = ee.Image(image_list.get(image_index))
            capture_date = image.date().format("YYYYMMdd").getInfo()

            if capture_date in existing_keys:
                print(f"[{image_index + 1}/{count}] Skip existing date: {capture_date}")
                continue

            out_path = out_dir / f"{capture_date}_{river}_{SATELLITE_TAG}_{DATA_SOURCE_TAG}.tif"
            if out_path.exists():
                if tif_has_expected_bands(out_path, OUTPUT_BAND_NAMES):
                    existing_keys.add(capture_date)
                    print(f"[{image_index + 1}/{count}] Skip existing file: {out_path.name}")
                    continue
                print(f"[{image_index + 1}/{count}] Replacing incompatible file: {out_path.name}")

            print(f"[{image_index + 1}/{count}] Downloading: {out_path.name}")
            try:
                download_image(image, out_path, roi, geometry)
                if out_path.exists():
                    size_mb = out_path.stat().st_size / 1024 / 1024
                    existing_keys.add(capture_date)
                    print(f"   Done. Size: {size_mb:.1f} MB")
            except Exception as exc:
                print(f"   Error: {exc}")
                if out_path.exists():
                    out_path.unlink()


def run_with_auto_restart():
    attempt = 0
    while True:
        try:
            main()
            return
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            attempt += 1
            print(
                f"\nDownload failed. Restarting in {AUTO_RESTART_DELAY_SECONDS}s. "
                f"Attempt: {attempt}. Error: {exc}"
            )
            time.sleep(AUTO_RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    run_with_auto_restart()
