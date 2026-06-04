import os
import re
import time
from datetime import datetime
from pathlib import Path
import inspect
import numpy as np
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.features import shapes
from rasterio.features import rasterize
from rasterio.warp import reproject, transform_bounds, transform_geom
import requests
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- 配置 ---
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
# 输出文件名标签配置
SATELLITE_TAG = "S2"
DATA_SOURCE_TAG = "DN"
AUTO_RESTART_DELAY_SECONDS = 20
PROXY_PORT = 7897

def configure_proxy(port: int):
    proxy = f"http://127.0.0.1:{port}"
    os.environ["HTTP_PROXY"] = proxy
    os.environ["HTTPS_PROXY"] = proxy


if not os.environ.get("HTTP_PROXY") and not os.environ.get("HTTPS_PROXY"):
    configure_proxy(PROXY_PORT)

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[1]
GEOJSON_PATH = REPO_ROOT / "data" / "watermask" / "three_gorges_watermask.geojson"
OUT_BASE_DIR = REPO_ROOT / "test"
OUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# --- 工具函数 ---

def build_network_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=6, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    if os.environ.get("HTTP_PROXY"):
        session.proxies = {"http": os.environ["HTTP_PROXY"], "https": os.environ["HTTPS_PROXY"]}
    return session

def create_stac_client() -> Client:
    session = build_network_session()
    
    # 检查当前环境的 StacApiIO 是否支持 session 参数
    init_sig = inspect.signature(StacApiIO.__init__)
    if "session" in init_sig.parameters:
        stac_io = StacApiIO(session=session)
    else:
        # 兼容旧版：手动创建一个带 session 的 StacApiIO 实例
        stac_io = StacApiIO()
        stac_io.session = session 

    return Client.open(STAC_URL, stac_io=stac_io)

def load_geojson_features(geojson_path: Path):
    """读取 GeoJSON 并返回要素列表，包含属性和几何体（保持为 GeoJSON dict）。"""
    import json

    if not geojson_path.exists():
        raise FileNotFoundError(f"未找到 GeoJSON: {geojson_path}")

    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    if not features:
        raise ValueError("GeoJSON 中未发现任何要素")

    return features


def geometry_bounds(geom: dict):
    """计算 GeoJSON geometry 的 bbox [minx, miny, maxx, maxy]（支持 Polygon/MultiPolygon）。"""
    coords = []

    def collect(c):
        if isinstance(c[0], (float, int)):
            coords.append(tuple(c))
        else:
            for v in c:
                collect(v)

    collect(geom.get("coordinates", []))
    if not coords:
        raise ValueError("无法解析几何坐标")
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return [min(xs), min(ys), max(xs), max(ys)]

def get_existing_time_keys(directory: Path, satellite_tag: str, source_tag: str):
    """返回目录中已存在的日期键（YYYYMMDD），基于命名规则 *_<satellite>_<source>.tif"""
    existing_time_keys = set()
    pattern = re.compile(rf"^(\d{{8}})_.+_{re.escape(satellite_tag)}_{re.escape(source_tag)}\.tif$", re.IGNORECASE)
    for p in directory.glob("*.tif"):
        m = pattern.match(p.name)
        if m:
            existing_time_keys.add(m.group(1))
    return existing_time_keys

def band_sort_key(name: str):
    # ✅ 修正：下载所有波段，仅按名称自然排序
    return name

# --- 核心下载逻辑 ---

def download_and_clip(item, out_path, bbox_4326, geometry):
    signed_item = planetary_computer.sign(item)
    
    # 筛选有效的栅格波段
    assets = {k: v for k, v in signed_item.assets.items() 
              if (v.href.lower().endswith(".tif") or "geotiff" in (v.media_type or ""))}
    selected_keys = sorted(assets.keys(), key=band_sort_key)
    if not selected_keys:
        raise ValueError(f"未找到可下载的栅格资产: {item.id}")

    # 1. 以 B02 作为参考坐标系和分辨率基准
    ref_key = "B02" if "B02" in assets else selected_keys[0]
    with rasterio.open(assets[ref_key].href) as ref_src:
        # 将经纬度 BBox 转为参考影像坐标系
        ref_left, ref_bottom, ref_right, ref_top = transform_bounds("EPSG:4326", ref_src.crs, *bbox_4326)
        # 获取窗口
        ref_window = from_bounds(ref_left, ref_bottom, ref_right, ref_top, ref_src.transform)
        ref_transform = ref_src.window_transform(ref_window)
        ref_width = int(ref_window.width)
        ref_height = int(ref_window.height)
        ref_crs = ref_src.crs

    # 2. 将 GeoJSON 几何体投影到参考 CRS 并 rasterize 为掩膜
    geom_proj = transform_geom("EPSG:4326", ref_crs, geometry)
    mask_arr = rasterize(
        [(geom_proj, 1)],
        out_shape=(ref_height, ref_width),
        transform=ref_transform,
        fill=0,
        dtype="uint8",
    )
    final_mask = (mask_arr == 1)

    # 3. 创建输出文件 (流式写入)
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": len(selected_keys),
        "dtype": "float32",
        "crs": ref_crs,
        "transform": ref_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": np.nan,
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, key in enumerate(selected_keys, start=1):
            asset_href = assets[key].href
            is_qa = key.upper() in ["SCL", "QA60"]
            
            try:
                with rasterio.open(asset_href) as src:
                    # 转换 BBox 到当前波段坐标系（Sentinel-2 不同波段分辨率不同）
                    b_l, b_b, b_r, b_t = transform_bounds("EPSG:4326", src.crs, *bbox_4326)
                    b_window = from_bounds(b_l, b_b, b_r, b_t, src.transform)
                    
                    # 仅读取窗口内数据
                    data = src.read(1, window=b_window, boundless=True, fill_value=0)
                    win_trans = src.window_transform(b_window)

                    # 重采样到参考网格（统一大小）
                    band_data = np.full((ref_height, ref_width), np.nan, dtype=np.float32)
                    reproject(
                        source=data,
                        destination=band_data,
                        src_transform=win_trans,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.nearest if is_qa else Resampling.bilinear,
                    )
                    
                    # 应用掩膜裁剪
                    band_data[~final_mask] = np.nan
                    
                    dst.write(band_data, i)
                    dst.set_band_description(i, key)
                    print(f"   -> Band {key} 写入完成")
            except Exception as e:
                print(f"   -> Band {key} 读取失败: {e}")

def main():
    client = create_stac_client()
    features = load_geojson_features(GEOJSON_PATH)

    for feat in features:
        props = feat.get("properties", {})
        river = props.get("basin_name") or props.get("name") or f"river_{props.get('basin_id','unknown')}"
        geom = feat.get("geometry")
        if not geom:
            print(f"跳过无几何的要素: {river}")
            continue

        out_dir = OUT_BASE_DIR / river
        out_dir.mkdir(parents=True, exist_ok=True)

        bbox = geometry_bounds(geom)

        search = client.search(collections=[COLLECTION], bbox=bbox, max_items=500)
        items = sorted(list(search.items()), key=lambda x: x.datetime or datetime.min)

        print(f"\n=== 处理河流: {river} | 发现影像: {len(items)} ===")
        existing_keys = get_existing_time_keys(out_dir, SATELLITE_TAG, DATA_SOURCE_TAG)

        for idx, item in enumerate(items, start=1):
            if not item.datetime:
                continue
            capture_date = item.datetime.strftime("%Y%m%d")
            if capture_date in existing_keys:
                continue

            out_path = out_dir / f"{capture_date}_{river}_{SATELLITE_TAG}_{DATA_SOURCE_TAG}.tif"
            print(f"[{idx}/{len(items)}] 正在下载: {out_path.name}")
            
            try:
                download_and_clip(item, out_path, bbox, geom)
                print(f"   Done. 大小: {out_path.stat().st_size/1024/1024:.1f} MB")
            except Exception as e:
                print(f"   Error: {e}")
                if out_path.exists(): out_path.unlink()

def run_with_auto_restart():
    attempt = 0
    while True:
        try:
            main()
            return
        except KeyboardInterrupt:
            raise
        except Exception as e:
            attempt += 1
            print(
                f"\n下载进程异常退出，{AUTO_RESTART_DELAY_SECONDS} 秒后自动重启"
                f"（第 {attempt} 次）：{e}"
            )
            time.sleep(AUTO_RESTART_DELAY_SECONDS)


if __name__ == "__main__":
    run_with_auto_restart()
