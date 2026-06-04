"""Compute per-pixel median across TIF images grouped by folder, then output results."""

import os
from pathlib import Path
from typing import List, Tuple
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
import tempfile
import shutil


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_ROOT = REPO_ROOT / "data" / "chla_TIFF"
OUTPUT_ROOT = REPO_ROOT / "result" / "5_statistics_visualization" / "test"


def find_tif_groups(root_dir: str, output_root: str) -> List[Tuple[str, List[str]]]:
    groups: List[Tuple[str, List[str]]] = []
    output_root_abs = os.path.abspath(output_root)

    for current_root, _, files in os.walk(root_dir):
        current_root_abs = os.path.abspath(current_root)
        if current_root_abs.startswith(output_root_abs):
            continue

        tif_files = [
            os.path.join(current_root, f)
            for f in files
            if f.lower().endswith(".tif")
        ]
        if tif_files:
            tif_files.sort()
            groups.append((current_root, tif_files))

    groups.sort(key=lambda x: x[0])
    return groups


def is_same_grid(ds: rasterio.io.DatasetReader, ref: rasterio.io.DatasetReader) -> bool:
    return (
        ds.width == ref.width
        and ds.height == ref.height
        and ds.transform == ref.transform
        and ds.crs == ref.crs
    )


def iter_windows(width: int, height: int, block_size: int = 1024):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def choose_block_size(tif_count: int, max_block: int = 1024) -> int:
    # Limit stack elements per window to avoid memory spikes with many input images.
    target_stack_elements = 20_000_000
    if tif_count <= 0:
        return max_block

    block = int((target_stack_elements / tif_count) ** 0.5)
    block = max(128, min(max_block, block))

    # Align to multiples of 16 to reduce small-window overhead.
    block = (block // 16) * 16
    if block < 128:
        block = 128
    return block


def compute_group_median_tif(tif_files: List[str], output_tif: str) -> bool:
    datasets: List[rasterio.io.DatasetReader] = []
    reproj_readers: List[WarpedVRT] = []
    try:
        for path in tif_files:
            datasets.append(rasterio.open(path))

        if not datasets or datasets[0].count < 1:
            print("  Skipping: no usable bands")
            return False

        ref = datasets[0]
        aligned_readers = []
        reprojected_count = 0

        for ds in datasets:
            if ds.count < 1:
                continue

            if is_same_grid(ds, ref):
                aligned_readers.append(ds)
                continue

            # Auto-reproject to reference grid when grids differ
            if ref.crs is None:
                print(f"  Warning: reference TIF has no CRS; cannot reproject, skipping {ds.name}")
                continue

            try:
                vrt = WarpedVRT(
                    ds,
                    crs=ref.crs,
                    transform=ref.transform,
                    width=ref.width,
                    height=ref.height,
                    resampling=Resampling.bilinear,
                )
                reproj_readers.append(vrt)
                aligned_readers.append(vrt)
                reprojected_count += 1
            except Exception as exc:
                print(f"  Warning: reprojection failed, skipping {ds.name}, reason: {exc}")

        if not aligned_readers:
            print("  Skipping: no valid TIF files in this group")
            return False

        if reprojected_count > 0:
            print(f"  Auto-reprojected {reprojected_count} TIF(s) to reference grid")

        out_nodata = ref.nodata if ref.nodata is not None else -9999.0
        tif_count = len(aligned_readers)
        block_size = choose_block_size(tif_count)
        print(f"  Using window size: {block_size} x {block_size}")

        # Write via temp file; fall back to different write options (e.g. no compression) on first failure
        os.makedirs(os.path.dirname(output_tif), exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tif", dir=os.path.dirname(output_tif))
        os.close(tmp_fd)

        write_attempts = [
            {"compress": "lzw", "predictor": 2, "BIGTIFF": "YES"},
            {"compress": None, "BIGTIFF": "YES"},
        ]

        for attempt in write_attempts:
            try:
                profile = ref.profile.copy()
                profile.update(
                    dtype="float32",
                    count=1,
                    nodata=float(out_nodata),
                )

                # Apply write parameters
                if attempt.get("compress") is not None:
                    profile["compress"] = attempt.get("compress")
                else:
                    profile.pop("compress", None)
                if "predictor" in attempt:
                    profile["predictor"] = attempt["predictor"]
                if "BIGTIFF" in attempt:
                    profile["BIGTIFF"] = attempt["BIGTIFF"]

                with rasterio.open(tmp_path, "w", **profile) as dst:
                    for window in iter_windows(ref.width, ref.height, block_size=block_size):
                        stack = []
                        for reader in aligned_readers:
                            try:
                                arr = reader.read(1, window=window).astype(np.float32)
                                nodata = reader.nodata
                                if nodata is not None:
                                    arr = np.where(arr == nodata, np.nan, arr)
                            except Exception:
                                # Fill NaN on single-input-block error; continue with remaining inputs
                                arr = np.full((int(window.height), int(window.width)), np.nan, dtype=np.float32)
                            stack.append(arr)

                        stack_arr = np.stack(stack, axis=0)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                            median_arr = np.nanmedian(stack_arr, axis=0)

                        median_arr = np.where(np.isnan(median_arr), float(out_nodata), median_arr).astype(np.float32)
                        dst.write(median_arr, 1, window=window)

                # Write succeeded; move temp to target
                try:
                    if os.path.exists(output_tif):
                        os.remove(output_tif)
                    shutil.move(tmp_path, output_tif)
                except Exception as exmove:
                    print(f"  Failed to move output file: {exmove}")
                    if os.path.exists(tmp_path) and not os.path.exists(output_tif):
                        shutil.move(tmp_path, output_tif)

                return True
            except Exception as exc:
                print(f"  Failed (write attempt with {attempt}): {exc}")
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

        # All write strategies failed; return False
        return False
    except Exception as exc:
        print(f"  Failed: {exc}")
        return False
    finally:
        for vrt in reproj_readers:
            vrt.close()
        for ds in datasets:
            ds.close()


def build_output_path(root_dir: str, folder_path: str, output_root: str) -> str:
    rel = os.path.relpath(folder_path, root_dir)
    folder_name = os.path.basename(folder_path.rstrip(os.sep))

    if rel == ".":
        return os.path.join(output_root, f"{folder_name}_median.tif")

    rel_parent = os.path.dirname(rel)
    return os.path.join(output_root, rel_parent, f"{folder_name}_median.tif")


def main():
    root_dir = os.fspath(INPUT_ROOT)
    output_root = os.fspath(OUTPUT_ROOT)
    os.makedirs(output_root, exist_ok=True)

    print(f"Root directory: {root_dir}")
    print(f"Output directory: {output_root}")
    print("Scanning for TIF files...")

    groups = find_tif_groups(root_dir, output_root)
    if not groups:
        print("No TIF files found.")
        return

    print(f"Found {len(groups)} folder(s) containing TIF files.")

    ok_count = 0
    fail_count = 0
    for idx, (folder_path, tif_files) in enumerate(groups, start=1):
        output_tif = build_output_path(root_dir, folder_path, output_root)
        print(f"\n[{idx}/{len(groups)}] Folder: {folder_path}")
        print(f"  TIF count: {len(tif_files)}")
        print(f"  Output file: {output_tif}")

        if compute_group_median_tif(tif_files, output_tif):
            ok_count += 1
            print("  Done")
        else:
            fail_count += 1

    print("\nAll processing complete")
    print(f"Succeeded: {ok_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
