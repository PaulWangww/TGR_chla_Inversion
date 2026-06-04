"""Plot annual CHLA median distribution across TGR tributaries with interpolated point-index stripes."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
INPUT_CSV = REPO_ROOT / "result" / "5_statistics_visualization" / "chla_median_results_20pts.csv"
OUTPUT_DIR = REPO_ROOT / "result" / "5_statistics_visualization"
OUTPUT_PNG = OUTPUT_DIR / "chla_median_results_20pts_plot.png"

VALUE_COLUMN = "CHLA_Median"
X_COLUMN = "basin_name"
Y_COLUMN = "Year"
POINT_COLUMN = "point_index"

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
    }
)

VMIN = 0
VMAX = 30
MIN_STRIP_COUNT = 40
MAX_STRIP_COUNT = 180
STRIPES_PER_POINT = 10


def prepare_data(df):
    required_columns = {X_COLUMN, Y_COLUMN, POINT_COLUMN, VALUE_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {sorted(missing_columns)}")

    data = df.copy()
    data[Y_COLUMN] = pd.to_numeric(data[Y_COLUMN], errors="coerce")
    data[POINT_COLUMN] = pd.to_numeric(data[POINT_COLUMN], errors="coerce")
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")
    data = data.dropna(subset=[X_COLUMN, Y_COLUMN, POINT_COLUMN]).copy()
    data[Y_COLUMN] = data[Y_COLUMN].astype(int)
    data[POINT_COLUMN] = data[POINT_COLUMN].astype(int)
    return data


def get_group_profile(group):
    group = group.sort_values(POINT_COLUMN).copy()
    valid = group[[POINT_COLUMN, VALUE_COLUMN]].dropna()

    if valid.empty:
        return None

    point_values = valid[POINT_COLUMN].to_numpy(dtype=float)
    chla_values = valid[VALUE_COLUMN].to_numpy(dtype=float)

    if len(valid) == 1:
        return np.full(MIN_STRIP_COUNT, chla_values[0], dtype=float)

    point_span = max(1, int(point_values.max() - point_values.min() + 1))
    dense_count = int(np.clip(point_span * STRIPES_PER_POINT, MIN_STRIP_COUNT, MAX_STRIP_COUNT))
    dense_points = np.linspace(point_values.min(), point_values.max(), dense_count)
    dense_values = np.interp(dense_points, point_values, chla_values)
    return dense_values


def draw_cell(ax, x_center, y_center, cell_width, cell_height, values, cmap, norm, edge_color="#d9d9d9"):
    x0 = x_center - cell_width / 2.0
    y0 = y_center - cell_height / 2.0

    if values is None:
        ax.add_patch(
            Rectangle((x0, y0), cell_width, cell_height, facecolor="#f2f2f2", edgecolor=edge_color, linewidth=0.7)
        )
        return

    stripe_height = cell_height / len(values)
    for idx, value in enumerate(values):
        clipped_value = float(np.clip(value, VMIN, VMAX))
        color = cmap(norm(clipped_value))
        stripe_y = y0 + idx * stripe_height
        ax.add_patch(Rectangle((x0, stripe_y), cell_width, stripe_height, facecolor=color, edgecolor=color, linewidth=0))

    ax.add_patch(Rectangle((x0, y0), cell_width, cell_height, facecolor="none", edgecolor=edge_color, linewidth=0.7))


def main():
    if not INPUT_CSV.exists():
        print(f"Error: Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    data = prepare_data(df)
    data = data[(data[Y_COLUMN] >= 2019) & (data[Y_COLUMN] <= 2025)].copy()

    if data.empty:
        print("Error: No data to plot after filtering years 2019-2025.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    basin_order = list(dict.fromkeys(data[X_COLUMN].tolist()))
    year_order = sorted(data[Y_COLUMN].unique().tolist())

    basin_to_x = {basin: idx for idx, basin in enumerate(basin_order)}
    year_to_y = {year: idx for idx, year in enumerate(year_order)}

    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)

    fig_width = max(14, len(basin_order) * 0.8)
    fig_height = max(8, len(year_order) * 0.55)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    cell_width = 0.72
    cell_height = 0.88

    for basin_name in basin_order:
        basin_data = data[data[X_COLUMN] == basin_name]
        for year in year_order:
            group = basin_data[basin_data[Y_COLUMN] == year]
            values = get_group_profile(group)
            draw_cell(ax, basin_to_x[basin_name], year_to_y[year], cell_width, cell_height, values, cmap, norm)

    ax.set_xlim(-0.5, len(basin_order) - 0.5)
    ax.set_ylim(-0.5, len(year_order) - 0.5)
    ax.set_xticks(range(len(basin_order)))
    ax.set_xticklabels(basin_order, rotation=45, ha="right")
    ax.set_yticks(range(len(year_order)))
    ax.set_yticklabels(year_order)
    ax.set_xlabel("TGR Tributaries")
    ax.set_ylabel("Year")
    ax.set_title("Annual Median chl-a Distribution of TGR Tributaries")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.86, 0.22, 0.018, 0.56])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("chl-a_Median (clipped to 0-20)", labelpad=14)
    cbar.set_ticks(np.linspace(VMIN, VMAX, 9))
    cbar.ax.text(0.5, 1.14, "Upstream\n↓\nDownstream", transform=cbar.ax.transAxes, va="bottom", ha="center", fontsize=9)

    ax.grid(False)
    plt.subplots_adjust(right=0.83)
    plt.tight_layout(rect=[0.0, 0.0, 0.83, 1.0])
    plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Plot saved to: {os.fspath(OUTPUT_PNG)}")


if __name__ == "__main__":
    main()
