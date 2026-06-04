"""Compute per-polygon linear regression slopes of CHLA over time and plot by point_index."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "chla_TIFF")
OUTPUT_CSV = os.path.join(ROOT_DIR, "chla_median_results_20pts.csv")

YEAR_COLUMN = "Year"
POLYGON_COLUMN = "Polygon_Index"
BASIN_COLUMN = "basin_name"
POINT_COLUMN = "point_index"
VALUE_COLUMN = "CHLA_Median"

YEAR_MIN = 2019
YEAR_MAX = 2025

OUTPUT_SLOPE_CSV = os.path.join(ROOT_DIR, "polygon_slope_results.csv")
OUTPUT_SLOPE_PNG = os.path.join(ROOT_DIR, "polygon_slope_results.png")
MIN_STRIP_COUNT = 40
MAX_STRIP_COUNT = 220
STRIPES_PER_POLYGON = 12

plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
    }
)


def prepare_data(df):
    required_columns = {YEAR_COLUMN, POLYGON_COLUMN, BASIN_COLUMN, POINT_COLUMN, VALUE_COLUMN}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV missing required columns: {sorted(missing_columns)}")

    data = df.copy()
    data[YEAR_COLUMN] = pd.to_numeric(data[YEAR_COLUMN], errors="coerce")
    data[POLYGON_COLUMN] = pd.to_numeric(data[POLYGON_COLUMN], errors="coerce")
    data[POINT_COLUMN] = pd.to_numeric(data[POINT_COLUMN], errors="coerce")
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")
    data = data.dropna(subset=[YEAR_COLUMN, POLYGON_COLUMN, BASIN_COLUMN, POINT_COLUMN, VALUE_COLUMN]).copy()
    data[YEAR_COLUMN] = data[YEAR_COLUMN].astype(int)
    data[POLYGON_COLUMN] = data[POLYGON_COLUMN].astype(int)
    data[POINT_COLUMN] = data[POINT_COLUMN].astype(int)
    return data


def aggregate_polygon_yearly(data):
    grouped = (
        data.groupby([POLYGON_COLUMN, BASIN_COLUMN, POINT_COLUMN, YEAR_COLUMN], as_index=False)[VALUE_COLUMN]
        .median()
        .sort_values([BASIN_COLUMN, POINT_COLUMN, POLYGON_COLUMN, YEAR_COLUMN])
        .reset_index(drop=True)
    )
    return grouped


def fit_least_squares(years, values):
    x = np.asarray(years, dtype=float)
    y = np.asarray(values, dtype=float)

    if x.size < 2:
        return np.nan, np.nan

    design_matrix = np.column_stack([x, np.ones_like(x)])
    slope, intercept = np.linalg.lstsq(design_matrix, y, rcond=None)[0]
    return float(slope), float(intercept)


def build_results(grouped_data):
    records = []
    polygon_order = grouped_data[[BASIN_COLUMN, POINT_COLUMN, POLYGON_COLUMN]].drop_duplicates()
    polygon_order = polygon_order.sort_values([BASIN_COLUMN, POINT_COLUMN, POLYGON_COLUMN]).reset_index(drop=True)

    for _, row in polygon_order.iterrows():
        polygon_id = int(row[POLYGON_COLUMN])
        basin_name = str(row[BASIN_COLUMN])
        point_index = int(row[POINT_COLUMN])

        polygon_data = grouped_data[grouped_data[POLYGON_COLUMN] == polygon_id].sort_values(YEAR_COLUMN)
        slope, intercept = fit_least_squares(polygon_data[YEAR_COLUMN], polygon_data[VALUE_COLUMN])

        records.append(
            {
                POLYGON_COLUMN: polygon_id,
                BASIN_COLUMN: basin_name,
                POINT_COLUMN: point_index,
                "slope": slope,
                "intercept": intercept,
                "n_years": int(polygon_data.shape[0]),
                "year_start": int(polygon_data[YEAR_COLUMN].min()),
                "year_end": int(polygon_data[YEAR_COLUMN].max()),
                "median_chla": float(polygon_data[VALUE_COLUMN].median()),
            }
        )

    return pd.DataFrame.from_records(records)


def build_dense_profile(basin_data):
    basin_data = basin_data.sort_values(POINT_COLUMN).copy()
    valid = basin_data[[POINT_COLUMN, "slope"]].dropna()

    if valid.empty:
        return None

    point_values = valid[POINT_COLUMN].to_numpy(dtype=float)
    slope_values = valid["slope"].to_numpy(dtype=float)

    if len(valid) == 1:
        return np.full(MIN_STRIP_COUNT, slope_values[0], dtype=float)

    point_span = max(1, int(point_values.max() - point_values.min() + 1))
    dense_count = int(np.clip(point_span * STRIPES_PER_POLYGON, MIN_STRIP_COUNT, MAX_STRIP_COUNT))
    dense_points = np.linspace(point_values.min(), point_values.max(), dense_count)
    dense_values = np.interp(dense_points, point_values, slope_values)
    return dense_values


def plot_results(results_df):
    if results_df.empty:
        raise ValueError("No slope results to plot.")

    basin_order = list(dict.fromkeys(results_df[BASIN_COLUMN].tolist()))
    basin_to_rows = {
        basin: results_df[results_df[BASIN_COLUMN] == basin].sort_values(POINT_COLUMN)
        for basin in basin_order
    }

    vmin = -3.0
    vmax = 3.0

    cmap = plt.get_cmap("jet")
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    cell_width = 0.78
    cell_height = 0.92
    fig_width = max(14, len(basin_order) * 0.8)
    fig_height = 4.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=300)

    for idx, basin_name in enumerate(basin_order):
        basin_data = basin_to_rows[basin_name].reset_index(drop=True)
        dense_values = build_dense_profile(basin_data)

        x_center = idx
        x0 = x_center - cell_width / 2.0
        y0 = -cell_height / 2.0

        if dense_values is None:
            ax.add_patch(
                Rectangle((x0, y0), cell_width, cell_height, facecolor="#f2f2f2", edgecolor="#d9d9d9", linewidth=0.8)
            )
            ax.text(x_center, 0, "NA", ha="center", va="center", fontsize=9, color="#666666")
            continue

        stripe_height = cell_height / len(dense_values)
        for stripe_idx, value in enumerate(dense_values):
            clipped_value = float(np.clip(value, vmin, vmax))
            color = cmap(norm(clipped_value))
            stripe_y = y0 + stripe_idx * stripe_height
            ax.add_patch(
                Rectangle((x0, stripe_y), cell_width, stripe_height, facecolor=color, edgecolor=color, linewidth=0)
            )

        ax.add_patch(Rectangle((x0, y0), cell_width, cell_height, facecolor="none", edgecolor="#d9d9d9", linewidth=0.8))

    ax.set_xlim(-0.5, len(basin_order) - 0.5)
    ax.set_ylim(-0.7, 0.7)
    ax.set_xticks(range(len(basin_order)))
    ax.set_xticklabels(basin_order, rotation=45, ha="right")
    ax.set_yticks([])
    ax.set_xlabel("TGR Tributaries")
    ax.set_title("Per-Polygon Linear Regression Slopes Ordered by point_index")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.86, 0.24, 0.018, 0.52])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Slope of chl-a_Median per Year", labelpad=12)
    cbar.ax.text(0.5, 1.14, "point_index\nsmall → large", transform=cbar.ax.transAxes, va="bottom", ha="center", fontsize=9)

    plt.subplots_adjust(right=0.83)
    plt.savefig(OUTPUT_SLOPE_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.exists(OUTPUT_CSV):
        print(f"Error: Input file not found: {OUTPUT_CSV}")
        return

    df = pd.read_csv(OUTPUT_CSV)
    data = prepare_data(df)
    data = data[(data[YEAR_COLUMN] >= YEAR_MIN) & (data[YEAR_COLUMN] <= YEAR_MAX)].copy()

    if data.empty:
        print(f"Error: No data after filtering years {YEAR_MIN}-{YEAR_MAX}.")
        return

    grouped_data = aggregate_polygon_yearly(data)
    results_df = build_results(grouped_data)
    results_df.to_csv(OUTPUT_SLOPE_CSV, index=False, encoding="utf-8-sig")
    plot_results(results_df)

    print(f"Regression complete. Slope results saved to: {OUTPUT_SLOPE_CSV}")
    print(f"Slope plot saved to: {OUTPUT_SLOPE_PNG}")


if __name__ == "__main__":
    main()