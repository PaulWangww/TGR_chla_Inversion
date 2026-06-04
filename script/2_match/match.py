"""Sentinel-2 processing pipeline for matching satellite pixels with in-situ data.

Data flow:
1. Raw satellite stage
    Input: data/SR_TIFF/XiangxiRiver|ShennongRiver|PengxiRiver/*.tif and data/watermask/site.geojson
    Processing:
    - Read capture time and cloud info from each image
    - Extract 9 uniformly distributed pixels within each site polygon from site.geojson
    - Convert Sentinel bands to radiance/reflectance; retain per-point samples
    - Match insitu data by site + river + nearest time

2. Aggregation stage
    Input: Point-level samples from step 1 (in-memory only, no intermediate CSV)
    Processing:
    - Deduplicate by site + river + image_name + image_datetime + sample_point_id,
      keeping higher-priority match results
    - Compute NDCI for each 9-point group; filter by spatial CV
    - If CV < 0.3: use the mean NDCI of retained points as the group result
    - If CV >= 0.3: iteratively remove the point with the largest deviation from mean
      until CV < 0.3; discard the group if fewer than 3 points remain

Final output: result/2_matched/test/matched.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import bounds as geom_bounds
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


# ============================================================
# Path configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'

SITE_GEOJSON_FILE = DATA_DIR / 'watermask' / 'site.geojson'

SENTINEL_TIFF_BASE_DIR = DATA_DIR / 'SR_TIFF'
RIVER_FOLDERS = ['XiangxiRiver', 'ShennongRiver', 'PengxiRiver']
RIVER_FOLDER_TO_CODE = {
    'XiangxiRiver': 'XXR',
    'ShennongRiver': 'SNR',
    'PengxiRiver': 'PXR',
}

INSITU_FILE = DATA_DIR / 'insitu' / 'institu_20260603.csv'
DEFAULT_MATCH_MAX_HOURS = 48

MATCHED_DIR = BASE_DIR / 'result' / '2_matched' 
FINAL_OUTPUT = MATCHED_DIR / 'matched.csv'

MATCHED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# General utilities
# ============================================================
def _normalize_site(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _normalize_river(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def load_site_polygons(geojson_path: Path):
    """Load polygons from site.geojson."""
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get('features', [])
    out = []
    for ft in features:
        props = ft.get('properties', {}) or {}
        geom = ft.get('geometry')
        site = props.get('site')
        river = props.get('river')
        if not site or not river or not geom:
            continue
        if geom.get('type') not in {'Polygon', 'MultiPolygon'}:
            continue
        out.append({'site': site, 'river': str(river).upper(), 'geometry': geom})

    return out


site_polygons = load_site_polygons(SITE_GEOJSON_FILE)
allowed_rivers = set(RIVER_FOLDER_TO_CODE.values())
site_polygons = [sp for sp in site_polygons if sp.get('river') in allowed_rivers]
print(f"Loaded {len(site_polygons)} polygons from {SITE_GEOJSON_FILE.name} (river codes: {sorted(allowed_rivers)})")


def extract_datetime_from_filename(img_name: str):
    """Extract date from filename, defaulting to UTC+8."""
    try:
        m = re.search(r'(19|20)\d{6}', img_name)
        date_str = m.group(0) if m else None
        if date_str is None:
            return None

        img_date = datetime.strptime(date_str, '%Y%m%d')
        img_datetime_utc = datetime.combine(img_date.date(), datetime.min.time().replace(hour=12))
        return img_datetime_utc + timedelta(hours=8)
    except Exception as e:
        print(f"  Warning: Failed to parse datetime from {img_name}: {e}")
        return None


def extract_datetime_and_cloud_from_metadata(img_path: Path):
    """Read capture time and cloud coverage from image metadata."""
    try:
        with rasterio.open(img_path) as src:
            md = src.tags() or {}

            sensing_time = None
            cloud = None

            product_id = md.get('PRODUCT_ID')
            if product_id:
                m = re.search(r'(19|20)\d{6}T\d{6}', product_id)
                if m:
                    dt_utc = datetime.strptime(m.group(0), '%Y%m%dT%H%M%S')
                    sensing_time = dt_utc + timedelta(hours=8)

            if 'CLOUD_COVERAGE_ASSESSMENT' in md:
                cloud = float(md['CLOUD_COVERAGE_ASSESSMENT'])
            elif 'CLOUDY_PIXEL_PERCENTAGE' in md:
                cloud = float(md['CLOUDY_PIXEL_PERCENTAGE'])

            return sensing_time, cloud
    except Exception as e:
        print(f"  Warning: Failed to read metadata from {img_path.name}: {e}")
        return None, None


def estimate_cloud_ratio(src: rasterio.io.DatasetReader, cloud_from_metadata: float | None) -> float | None:
    """Estimate cloud pixel percentage (0-100). Uses metadata first; falls back to pixel mask."""
    if cloud_from_metadata is not None and np.isfinite(cloud_from_metadata):
        return float(cloud_from_metadata)

    cloud_mask = np.zeros((src.height, src.width), dtype=bool)

    if src.count >= 15:
        scl = src.read(15, out_shape=(src.height, src.width), masked=True).filled(0).astype(np.int16)
        cloud_mask |= np.isin(scl, [3, 8, 9, 10])

    if src.count >= 21:
        opaque = src.read(21, out_shape=(src.height, src.width), masked=True).filled(0)
        cloud_mask |= (opaque > 0)

    if src.count >= 22:
        cirrus = src.read(22, out_shape=(src.height, src.width), masked=True).filled(0)
        cloud_mask |= (cirrus > 0)

    total = cloud_mask.size
    if total == 0:
        return None
    return float(cloud_mask.sum() * 100.0 / total)


def _normalize_band_name(name: str | None) -> str | None:
    if name is None:
        return None
    out = str(name).strip().upper()
    m = re.fullmatch(r'B0([1-9])', out)
    if m:
        out = f'B{m.group(1)}'
    return out if out else None


def _find_band_index(band_names: list[str], candidates: list[str]) -> int | None:
    for cand in candidates:
        if cand in band_names:
            return band_names.index(cand)
    return None


def _get_band_names(src: rasterio.io.DatasetReader) -> list[str]:
    names = []
    if src.descriptions and any(d is not None for d in src.descriptions):
        for i, desc in enumerate(src.descriptions, start=1):
            norm = _normalize_band_name(desc)
            names.append(norm if norm is not None else f'BAND_{i}')
    else:
        names = [f'BAND_{i}' for i in range(1, src.count + 1)]
    return names


def build_cloud_mask(src: rasterio.io.DatasetReader) -> tuple[np.ndarray, bool]:
    """Build cloud mask (True=cloud/shadow/cirrus) and report whether reliable cloud layers exist."""
    cloud_mask = np.zeros((src.height, src.width), dtype=bool)
    has_cloud_layer = False

    band_names = _get_band_names(src)

    scl_idx = _find_band_index(band_names, ['SCL'])
    if scl_idx is not None:
        scl = src.read(scl_idx + 1, out_shape=(src.height, src.width), masked=True).filled(0).astype(np.int16)
        cloud_mask |= np.isin(scl, [3, 8, 9, 10])
        has_cloud_layer = True

    opaque_idx = _find_band_index(band_names, ['MSK_OPAQUE'])
    if opaque_idx is not None:
        opaque = src.read(opaque_idx + 1, out_shape=(src.height, src.width), masked=True).filled(0)
        cloud_mask |= (opaque > 0)
        has_cloud_layer = True

    cirrus_idx = _find_band_index(band_names, ['MSK_CIRRUS'])
    if cirrus_idx is not None:
        cirrus = src.read(cirrus_idx + 1, out_shape=(src.height, src.width), masked=True).filled(0)
        cloud_mask |= (cirrus > 0)
        has_cloud_layer = True

    cldprb_idx = _find_band_index(band_names, ['MSK_CLDPRB', 'CLDPRB'])
    if cldprb_idx is not None:
        cldprb = src.read(cldprb_idx + 1, out_shape=(src.height, src.width), masked=True).filled(0).astype(np.float32)
        cloud_mask |= (cldprb >= 50.0)
        has_cloud_layer = True

    return cloud_mask, has_cloud_layer


def _find_required_band_indices(band_names: list[str]) -> tuple[list[int], int]:
    required = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
    idx_map = {name: idx for idx, name in enumerate(band_names)}
    present_indices = [idx_map[b] for b in required if b in idx_map]
    absent_count = len(required) - len(present_indices)
    return present_indices, absent_count


def _build_missing_channel_count(
    src: rasterio.io.DatasetReader,
    required_band_indices: list[int],
) -> np.ndarray:
    """Count missing required multispectral channels per pixel."""
    miss_count = np.zeros((src.height, src.width), dtype=np.int16)
    for b_idx in required_band_indices:
        arr = src.read(b_idx + 1, masked=True)
        mask_arr = np.ma.getmaskarray(arr)
        data = arr.astype(np.float32).filled(np.nan)
        miss = mask_arr | (~np.isfinite(data))
        miss_count += miss.astype(np.int16)
    return miss_count


def _bbox_intersects(src_bounds, bbox):
    return not (
        bbox[2] < src_bounds.left
        or bbox[0] > src_bounds.right
        or bbox[3] < src_bounds.bottom
        or bbox[1] > src_bounds.top
    )


def _pick_9_points_from_mask(poly_mask: np.ndarray) -> list[tuple[int, int]]:
    """Select 9 uniformly distributed points within a polygon mask."""
    rows, cols = np.where(poly_mask)
    if rows.size == 0:
        return []

    r_min, r_max = rows.min(), rows.max()
    c_min, c_max = cols.min(), cols.max()
    r_edges = np.linspace(r_min, r_max + 1, 4)
    c_edges = np.linspace(c_min, c_max + 1, 4)

    chosen = []
    chosen_set = set()

    for i in range(3):
        for j in range(3):
            r0, r1 = r_edges[i], r_edges[i + 1]
            c0, c1 = c_edges[j], c_edges[j + 1]

            in_cell = (rows >= r0) & (rows < r1) & (cols >= c0) & (cols < c1)
            if not np.any(in_cell):
                continue

            cell_rows = rows[in_cell]
            cell_cols = cols[in_cell]
            target_r, target_c = (r0 + r1) / 2, (c0 + c1) / 2
            dists = (cell_rows - target_r) ** 2 + (cell_cols - target_c) ** 2
            best_idx = np.argmin(dists)

            rc = (int(cell_rows[best_idx]), int(cell_cols[best_idx]))
            if rc not in chosen_set:
                chosen.append(rc)
                chosen_set.add(rc)

    if len(chosen) < 9:
        all_points = list(zip(rows.tolist(), cols.tolist()))
        step_indices = np.linspace(0, len(all_points) - 1, min(18, len(all_points))).astype(int)

        for idx in step_indices:
            rc = (int(all_points[idx][0]), int(all_points[idx][1]))
            if rc not in chosen_set:
                chosen.append(rc)
                chosen_set.add(rc)
            if len(chosen) == 9:
                break

    return chosen[:9]


def _find_valid_neighbor(
    r: int,
    c: int,
    poly_mask: np.ndarray,
    cloud_mask: np.ndarray,
    missing_channel_count: np.ndarray,
    use_cloud_mask: bool,
    use_missing_filter: bool,
) -> tuple[int, int] | None:
    """If the center pixel is invalid, search a 5x5 neighborhood for a valid one."""
    h, w = poly_mask.shape
    candidates = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            rr = r + dr
            cc = c + dc
            if rr < 0 or rr >= h or cc < 0 or cc >= w:
                continue
            if not poly_mask[rr, cc]:
                continue
            if use_cloud_mask and cloud_mask[rr, cc]:
                continue
            if use_missing_filter and (missing_channel_count[rr, cc] > 0):
                continue
            candidates.append((dr * dr + dc * dc, rr, cc))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    _, rr, cc = candidates[0]
    return rr, cc


def sample_polygon_points(
    src: rasterio.io.DatasetReader,
    site_name: str,
    geom_wgs84: dict,
    river_code: str,
    image_name: str,
    image_datetime: datetime,
    cloud_mask: np.ndarray,
    missing_channel_count: np.ndarray,
    use_cloud_mask: bool,
    use_missing_filter: bool,
    absent_required_channels: int,
    total_required_channels: int,
):
    """Extract sample points within a single polygon and read all band values."""
    try:
        if src.crs is None:
            geom_src = geom_wgs84
        elif str(src.crs).upper() in {'EPSG:4326', 'OGC:CRS84', 'WGS84'}:
            geom_src = geom_wgs84
        else:
            try:
                geom_src = transform_geom('EPSG:4326', src.crs, geom_wgs84, antimeridian_cutting=False)
            except TypeError:
                geom_src = transform_geom('EPSG:4326', src.crs, geom_wgs84)
    except Exception:
        return []

    bbox = geom_bounds(geom_src)
    if not _bbox_intersects(src.bounds, bbox):
        return []

    try:
        poly_mask = geometry_mask(
            [geom_src],
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
            all_touched=True,
        )
    except Exception:
        return []

    points_rc = _pick_9_points_from_mask(poly_mask)
    if not points_rc:
        return []

    poly_pixels = int(poly_mask.sum())
    if poly_pixels <= 0:
        return []

    if use_cloud_mask:
        poly_cloud_ratio = float(cloud_mask[poly_mask].mean() * 100.0)
        if poly_cloud_ratio > 50.0:
            return []

    if use_missing_filter:
        missing_channels_in_present = float(missing_channel_count[poly_mask].sum())
        missing_channels_absent = float(poly_pixels * absent_required_channels)
        total_channels = float(poly_pixels * total_required_channels)
        poly_missing_ratio = ((missing_channels_in_present + missing_channels_absent) / total_channels * 100.0) if total_channels > 0 else 100.0
        if poly_missing_ratio > 50.0:
            return []

    repaired_points = []
    for r, c in points_rc:
        is_cloud_free = (not cloud_mask[r, c]) if use_cloud_mask else True
        is_channel_ok = (missing_channel_count[r, c] == 0) if use_missing_filter else True
        if is_cloud_free and is_channel_ok:
            repaired_points.append((r, c))
            continue
        repl = _find_valid_neighbor(r, c, poly_mask, cloud_mask, missing_channel_count, use_cloud_mask, use_missing_filter)
        if repl is not None:
            repaired_points.append(repl)

    if not repaired_points:
        return []
    points_rc = repaired_points[:9]

    rows = [rc[0] for rc in points_rc]
    cols = [rc[1] for rc in points_rc]
    xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset='center')
    coords = list(zip(xs, ys))

    band_names_raw = src.descriptions if src.descriptions else None
    if not band_names_raw or all(b is None for b in band_names_raw):
        band_names_raw = [f'Band_{i + 1}' for i in range(src.count)]
    band_names = [_normalize_band_name(b) or f'BAND_{i + 1}' for i, b in enumerate(band_names_raw)]

    rrs_bands_sentinel = {'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'}
    sampled_vals = list(src.sample(coords))

    # Determine whether to use the new formula (post-2022-01-25)
    cutoff_date = datetime(2022, 1, 25)
    use_new_formula = image_datetime >= cutoff_date if image_datetime is not None else True
    
    rows_out = []
    for p_idx, vals in enumerate(sampled_vals, start=1):
        row_data = {}
        for b_i, val in enumerate(vals):
            b_name = band_names[b_i] if b_i < len(band_names) else f'BAND_{b_i + 1}'
            v = float(val)
            if b_name in rrs_bands_sentinel:
                # Sentinel-2 Level-2A: choose SR formula based on date
                if use_new_formula:
                    # Post-2022-01-25: SR = (DN_{L2A} - 1000) * 0.0001
                    sr = (v - 1000.0) * 0.0001
                else:
                    # Pre-2022-01-25: SR = DN_{L2A} * 0.0001
                    sr = v * 0.0001
                row_data[b_name] = sr if np.isfinite(sr) else np.nan
            else:
                row_data[b_name] = v

        row_data['satellite'] = 'Sentinel'
        row_data['image_name'] = image_name
        row_data['image_datetime'] = image_datetime
        row_data['site'] = site_name
        row_data['river'] = river_code
        row_data['sample_point_id'] = p_idx
        rows_out.append(row_data)

    return rows_out


def process_satellite_step1(river_folders):
    """Read Sentinel images, extract sample points, and build the base match table."""
    print("\nProcessing Sentinel...")
    all_rows = []

    for folder_name in river_folders:
        folder_path = SENTINEL_TIFF_BASE_DIR / folder_name
        if not folder_path.exists():
            print(f"  Folder {folder_path} not found, skipping.")
            continue

        image_files = sorted(list(folder_path.glob('*.tif')))
        print(f"  Found {len(image_files)} images in {folder_name}")
        river_code = RIVER_FOLDER_TO_CODE.get(folder_name, folder_name.upper())
        folder_polygons = [sp for sp in site_polygons if sp.get('river') == river_code]
        print(f"  River={river_code}, matched polygons={len(folder_polygons)}")

        for img_idx, img_path in enumerate(image_files):
            if img_idx % 10 == 0 and img_idx > 0:
                print(f"    Processed {img_idx}/{len(image_files)} images")

            img_name = img_path.stem
            with rasterio.open(img_path) as src:
                md_datetime, cloud_meta = extract_datetime_and_cloud_from_metadata(img_path)
                cloud_mask, has_cloud_layer = build_cloud_mask(src)
                if cloud_meta is not None and np.isfinite(cloud_meta):
                    cloud_ratio = float(cloud_meta)
                elif has_cloud_layer:
                    cloud_ratio = float(cloud_mask.mean() * 100.0)
                else:
                    cloud_ratio = None
                filtered_by_cloud = bool((cloud_ratio is not None) and (cloud_ratio > 50.0))

                band_names_norm = _get_band_names(src)
                required_indices, absent_required = _find_required_band_indices(band_names_norm)
                total_required = 12
                use_missing_filter = bool(required_indices)
                missing_channel_count = _build_missing_channel_count(src, required_indices) if use_missing_filter else np.zeros((src.height, src.width), dtype=np.int16)
                if not use_missing_filter:
                    absent_required = 0

                img_datetime = md_datetime if md_datetime is not None else extract_datetime_from_filename(img_name)

                sample_count = 0
                if not filtered_by_cloud and img_datetime is not None:
                    for sp in folder_polygons:
                        site_rows = sample_polygon_points(
                            src=src,
                            site_name=sp['site'],
                            geom_wgs84=sp['geometry'],
                            river_code=river_code,
                            image_name=img_name,
                            image_datetime=img_datetime,
                            cloud_mask=cloud_mask,
                            missing_channel_count=missing_channel_count,
                            use_cloud_mask=has_cloud_layer,
                            use_missing_filter=use_missing_filter,
                            absent_required_channels=absent_required,
                            total_required_channels=total_required,
                        )
                        if site_rows:
                            sample_count += 1
                            all_rows.extend(site_rows)

    print(f"  Sentinel: Total sampled points = {len(all_rows)}")

    df = pd.DataFrame(all_rows) if all_rows else None
    return df


def run_extract_stage():
    print('=' * 60)
    print('Stage 1: Reading raw Sentinel images and sampling')
    print('=' * 60)

    sentinel_df = process_satellite_step1(RIVER_FOLDERS)

    if sentinel_df is None or sentinel_df.empty:
        print('No Sentinel data to save')
        return None

    sentinel_df = _clean_and_reorder_columns(sentinel_df)
    print(f"Stage 1 completed in memory: {len(sentinel_df)} rows")
    return sentinel_df


# ============================================================
# Secondary matching: re-match insitu against existing Sentinel results
# ============================================================
def _pick_base_satellite_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Extract base satellite sample rows (excluding extension rows)."""
    base_df = df.copy()

    if 'source' in base_df.columns:
        source = base_df['source']
        source_str = source.astype(str).str.strip().str.lower()
        base_mask = source.isna() | (source_str == '') | (~source_str.isin(['insitu']))
        base_df = base_df.loc[base_mask].copy()

    stale_cols = [
        'datetime', 'chla', 'source', 'time_diff_hours',
        'insitu_datetime', 'insitu_chla', 'insitu_site', 'insitu_river',
    ]
    base_df = base_df.drop(columns=[c for c in stale_cols if c in base_df.columns], errors='ignore')
    return base_df


def _build_group_index(df: pd.DataFrame) -> dict:
    """Build a (site, river) group index sorted by time."""
    out = {}
    for (site, river), grp in df.groupby(['site', 'river'], dropna=False):
        grp_sorted = grp.sort_values('datetime').copy()
        time_ns = grp_sorted['datetime'].values.astype('datetime64[ns]').astype(np.int64)
        out[(site, river)] = {'df': grp_sorted, 'time_ns': time_ns}
    return out


def _nearest_row_in_group(group_pack: dict, target_time: pd.Timestamp):
    """Find the record nearest to target_time within a sorted time group."""
    t_ns = int(target_time.to_datetime64().astype('datetime64[ns]').astype(np.int64))
    time_ns = group_pack['time_ns']

    pos = np.searchsorted(time_ns, t_ns)
    cand_pos = []
    if pos > 0:
        cand_pos.append(pos - 1)
    if pos < len(time_ns):
        cand_pos.append(pos)

    if not cand_pos:
        return None, None

    best_pos = min(cand_pos, key=lambda p: abs(time_ns[p] - t_ns))
    best_row = group_pack['df'].iloc[best_pos]
    diff_hours = abs(time_ns[best_pos] - t_ns) / 3_600_000_000_000
    return best_row, float(diff_hours)


def _match_aux_data(
    satellite_df: pd.DataFrame,
    aux_df: pd.DataFrame,
    max_hours: float,
) -> pd.DataFrame | None:
    """Match insitu data to satellite samples by site+river+nearest time."""
    if satellite_df is None or satellite_df.empty or aux_df is None or aux_df.empty:
        return None

    aux_index = _build_group_index(aux_df)
    rows = []

    for idx, sat_row in satellite_df.iterrows():
        if idx % 500 == 0 and idx > 0:
            print(f"  insitu: Processed {idx}/{len(satellite_df)} rows")

        sat_site = sat_row['site']
        sat_river = sat_row['river']
        sat_time = sat_row['image_datetime']

        group_pack = aux_index.get((sat_site, sat_river))
        if group_pack is None:
            continue

        best_row, diff_hours = _nearest_row_in_group(group_pack, sat_time)
        if best_row is None or diff_hours > max_hours:
            continue

        new_row = sat_row.copy()
        new_row['insitu_datetime'] = best_row['datetime']
        new_row['insitu_chla'] = best_row.get('chla', np.nan)
        new_row['insitu_site'] = best_row.get('site', None)
        new_row['insitu_river'] = best_row.get('river', None)
        new_row['source'] = 'insitu'

        new_row['time_diff_hours'] = diff_hours
        rows.append(new_row)

    if not rows:
        return None
    return pd.DataFrame(rows)


def _combine_results(base_df: pd.DataFrame, insitu_df: pd.DataFrame | None) -> pd.DataFrame:
    parts = [base_df]
    if insitu_df is not None and not insitu_df.empty:
        parts.append(insitu_df)
    return pd.concat(parts, ignore_index=True, sort=False)


def _clean_and_reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and reorder columns for consistent output."""
    cols_to_drop = ['QA10', 'QA20', 'QA60', 'satellite', 'longitude', 'latitude']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    merge_pairs = [
        ('datetime', ['insitu_datetime']),
        ('chla', ['insitu_chla']),
        ('site', ['insitu_site']),
        ('river', ['insitu_river']),
    ]

    for target_col, source_cols in merge_pairs:
        existing_sources = [c for c in source_cols if c in df.columns]
        if not existing_sources:
            continue

        if target_col not in df.columns:
            df[target_col] = None

        for source_col in existing_sources:
            mask = df[target_col].isna()
            df.loc[mask, target_col] = df.loc[mask, source_col]

        df = df.drop(columns=existing_sources, errors='ignore')

    rename_map = {}
    for col in df.columns:
        if isinstance(col, str) and col.startswith('SR_'):
            new_col = col[3:]
            if new_col not in df.columns:
                rename_map[col] = new_col
    if rename_map:
        df = df.rename(columns=rename_map)

    default_band_cols = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r'(?i)band_\d+', c)]
    if default_band_cols:
        df = df.drop(columns=default_band_cols, errors='ignore')

    priority_cols = ['site', 'image_datetime', 'datetime', 'chla', 'source', 'time_diff_hours', 'river']
    front_cols = [c for c in priority_cols if c in df.columns]
    other_cols = [c for c in df.columns if c not in front_cols]
    return df[front_cols + other_cols]


def run_rematch_stage(
    base_df: pd.DataFrame | None = None,
    match_max_hours: float = DEFAULT_MATCH_MAX_HOURS,
):
    print('=' * 60)
    print('Stage 2: Re-matching insitu against Sentinel results')
    print('=' * 60)

    if base_df is None:
        raise ValueError('base_df is required for in-memory rematching')

    base_df = base_df.copy()
    if 'image_datetime' in base_df.columns:
        base_df['image_datetime'] = pd.to_datetime(base_df['image_datetime'], errors='coerce')
        base_df = base_df.dropna(subset=['image_datetime']).copy()
    if 'site' in base_df.columns:
        base_df['site'] = _normalize_site(base_df['site'])
    if 'river' in base_df.columns:
        base_df['river'] = _normalize_river(base_df['river'])
    base_df = _pick_base_satellite_rows(base_df)

    print(f'Base satellite rows (for matching): {len(base_df)}')

    insitu_csv = pd.read_csv(INSITU_FILE)
    insitu_csv['datetime'] = pd.to_datetime(insitu_csv['datetime'], errors='coerce')
    print(f'Loaded insitu csv rows: {len(insitu_csv)}')

    insitu_df = insitu_csv.copy()
    insitu_df = insitu_df.dropna(subset=['datetime']).copy()
    insitu_df['site'] = _normalize_site(insitu_df['site'])
    insitu_df['river'] = _normalize_river(insitu_df['river'])
    print(f'Loaded insitu rows after clean: {len(insitu_df)}')

    print('\n' + '=' * 60)
    print('Matching insitu')
    print('=' * 60)

    sentinel_insitu = _match_aux_data(base_df, insitu_df, match_max_hours)
    print(f"insitu matched rows: {0 if sentinel_insitu is None else len(sentinel_insitu)}")

    combined = _combine_results(base_df, sentinel_insitu)
    combined = _clean_and_reorder_columns(combined)

    print('\n' + '=' * 60)
    print('Stage 2 completed in memory')
    print('=' * 60)
    print(f'Rows: {len(combined)}')
    print(f'Columns: {len(combined.columns)}')

    return combined


# ============================================================
# NDCI + cubic polynomial fitting
# ============================================================
def compute_ndci(df: pd.DataFrame) -> pd.DataFrame:
    """Compute NDCI = (B5 - B4) / (B5 + B4)."""
    out = df.copy()
    denom = out['B5'] + out['B4']
    out['NDCI'] = np.where(denom != 0, (out['B5'] - out['B4']) / denom, np.nan)
    out['NDCI'] = out['NDCI'].replace([np.inf, -np.inf], np.nan)
    return out


def _deduplicate_point_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by point-level key, keeping higher-priority match results."""
    out = df.copy()
    if 'source' not in out.columns:
        out['source'] = None

    source_priority = {'insitu': 3}
    out['_source_priority'] = out['source'].astype(str).str.strip().str.lower().map(source_priority).fillna(1)

    key_cols = [c for c in ['site', 'river', 'image_name', 'image_datetime', 'sample_point_id'] if c in out.columns]
    if len(key_cols) < 5:
        raise KeyError('Missing required point-level key columns for deduplication')

    out = out.sort_values(key_cols + ['_source_priority'], ascending=[True, True, True, True, True, False])
    out = out.drop_duplicates(subset=key_cols, keep='first').copy()
    out = out.drop(columns=['_source_priority'], errors='ignore')
    return out


def _summarize_group_by_cv(group: pd.DataFrame) -> pd.Series | None:
    """Filter and summarize a single 9-point group by the CV rule."""
    current = group.copy()
    current = current.loc[current['NDCI'].notna()].copy()
    if len(current) < 3:
        return None

    original_count = len(current)

    while True:
        values = current['NDCI'].astype(float).to_numpy()
        if len(values) < 3:
            return None

        mean_value = float(np.mean(values))
        std_value = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        denom = abs(mean_value)
        cv_value = float(np.inf if denom <= 1e-12 and std_value > 0 else (0.0 if denom <= 1e-12 else std_value / denom))

        if cv_value < 0.3:
            summary = current.iloc[0].copy()

            numeric_cols = [
                c for c in current.columns
                if pd.api.types.is_numeric_dtype(current[c]) and c not in {'sample_point_id', 'NDCI'}
            ]
            for col in numeric_cols:
                summary[col] = float(current[col].mean())

            summary['NDCI'] = mean_value
            summary['NDCI_std'] = std_value
            summary['NDCI_cv'] = cv_value
            summary['NDCI_points_used'] = int(len(current))
            summary['NDCI_points_total'] = int(original_count)
            summary['NDCI_removed_points'] = int(original_count - len(current))
            summary['sample_point_id'] = None
            return summary

        worst_idx = (current['NDCI'].astype(float) - mean_value).abs().idxmax()
        current = current.drop(index=worst_idx)
        if len(current) < 3:
            return None


def run_finalize_stage(input_df: pd.DataFrame):
    print('=' * 60)
    print('Stage 3: NDCI CV filtering and group aggregation')
    print('=' * 60)

    point_df = _clean_and_reorder_columns(input_df)
    point_df = _deduplicate_point_rows(point_df)
    point_df = compute_ndci(point_df)

    group_keys = ['site', 'river', 'image_name', 'image_datetime']
    summaries = []

    for _, group in point_df.groupby(group_keys, dropna=False):
        summary = _summarize_group_by_cv(group)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise ValueError('No valid groups remained after CV filtering.')

    final_df = pd.DataFrame(summaries)
    final_df = _clean_and_reorder_columns(final_df)
    final_df = final_df.reset_index(drop=True)
    return final_df


# ============================================================
# Command-line entry point
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Unified Sentinel matching pipeline.')
    parser.add_argument('--match-max-hours', type=float, default=DEFAULT_MATCH_MAX_HOURS, help='Max hours for insitu matching.')
    parser.add_argument('--output', type=Path, default=FINAL_OUTPUT, help='Final output CSV path.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extract_df = run_extract_stage()
    if extract_df is None or extract_df.empty:
        raise ValueError('No Sentinel rows were extracted.')

    rematch_df = run_rematch_stage(
        base_df=extract_df,
        match_max_hours=args.match_max_hours,
    )

    final_df = run_finalize_stage(rematch_df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(args.output, index=False, encoding='utf-8-sig')
    print('\n' + '=' * 60)
    print('Final output saved')
    print('=' * 60)
    print(f'Output: {args.output}')
    print(f'Rows: {len(final_df)}')
    print(f'Columns: {len(final_df.columns)}')


if __name__ == '__main__':
    main()