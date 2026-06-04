"""Render NDCI/CHLA raster results as color PNGs for quick visual inspection."""

from __future__ import annotations

import csv
import os
from pathlib import Path

MPL_CACHE_DIR = Path(__file__).resolve().parent / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio


BASE_DIR = Path(__file__).resolve().parent
INPUT_ROOT = BASE_DIR / "data" / "chla_TIFF"
PLOT_OUTPUT_ROOT = BASE_DIR / "data" / "chla_plot"

NODATA_VALUE = -9999.0
CHLA_MAX = 200.0
DPI = 300
FIG_MARGIN = 0.12
CB_WIDTH_RATIO = 0.06
CB_GAP = 0.04
BASE_FIG_WIDTH = 10.0


def _get_chla_cmap_norm():
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "chla_blue_to_red",
        ["#1e90ff", "#1fae66", "#f0dd3f", "#f59d2a", "#d7191c"],
        N=256,
    )
    cmap = cmap.copy()
    cmap.set_bad("white")
    cmap.set_under("white")
    cmap.set_over("darkred")
    norm = mcolors.LogNorm(vmin=0.1, vmax=CHLA_MAX, clip=True)
    return cmap, norm


def _read_chla_tif(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        arr = src.read(1, masked=True).astype(np.float32).filled(np.nan)

    arr[arr == NODATA_VALUE] = np.nan
    return arr


def _save_chla_png(png_path: Path, chla_map: np.ndarray, cmap, norm) -> None:
    plot_arr = np.where(np.isfinite(chla_map) & (chla_map > 0), chla_map, np.nan).astype(np.float32, copy=False)
    img_h, img_w = plot_arr.shape

    fig_height = BASE_FIG_WIDTH * (img_h / img_w)
    fig = plt.figure(figsize=(BASE_FIG_WIDTH, fig_height), dpi=DPI)
    try:
        fig.patch.set_facecolor("white")

        ax_img = fig.add_axes(
            [
                FIG_MARGIN,
                FIG_MARGIN,
                1 - FIG_MARGIN * 2 - CB_GAP - CB_WIDTH_RATIO,
                1 - FIG_MARGIN * 2,
            ]
        )
        im = ax_img.imshow(plot_arr, cmap=cmap, norm=norm, interpolation="nearest")
        ax_img.set_axis_off()

        ax_cbar = fig.add_axes(
            [
                1 - FIG_MARGIN - CB_WIDTH_RATIO,
                FIG_MARGIN,
                CB_WIDTH_RATIO,
                1 - FIG_MARGIN * 2,
            ]
        )
        cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
        cbar.set_ticks([0.1, 0.5, 1, 5, 10, 50, 100, 200])
        cbar.set_ticklabels([0.1, 0.5, 1, 5, 10, 50, 100, 200])
        cbar.set_label("Chla concentration (ug/L)", labelpad=12)

        png_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, facecolor="white", dpi=DPI)
    finally:
        plt.close(fig)


def main() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input folder not found: {INPUT_ROOT}")

    PLOT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(INPUT_ROOT.rglob("*.tif"))
    if not tif_files:
        print(f"No tif files found under: {INPUT_ROOT}")
        return

    print(f"Input root: {INPUT_ROOT}")
    print(f"Plot output root: {PLOT_OUTPUT_ROOT}")
    print(f"Found {len(tif_files)} tif files")

    cmap, norm = _get_chla_cmap_norm()
    rows: list[dict[str, str]] = []

    for i, tif_path in enumerate(tif_files, start=1):
        try:
            chla_map = _read_chla_tif(tif_path)
            png_path = PLOT_OUTPUT_ROOT / tif_path.relative_to(INPUT_ROOT).with_suffix(".png")
            _save_chla_png(png_path, chla_map, cmap, norm)

            rows.append({"status": "success", "source_tif": str(tif_path), "output_plot": str(png_path), "error": ""})
            print(f"[{i}/{len(tif_files)}] Saved: {png_path.name}")
        except Exception as exc:
            rows.append({"status": "failed", "source_tif": str(tif_path), "output_plot": "", "error": str(exc)})
            print(f"[{i}/{len(tif_files)}] Failed: {tif_path.name} -> {exc}")

    summary_path = PLOT_OUTPUT_ROOT / "plot_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["status", "source_tif", "output_plot", "error"])
        writer.writeheader()
        writer.writerows(rows)

    success_count = sum(row["status"] == "success" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    print(f"Done. Success={success_count}, Failed={failed_count}")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
