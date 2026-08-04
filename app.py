import os
import base64
from io import BytesIO

import numpy as np
import streamlit as st
import rasterio
from rasterio.warp import (reproject, Resampling, transform_bounds,
                           calculate_default_transform)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import folium
from streamlit_folium import st_folium


CLASS_NAMES  = ["Vegetation", "Impervious/Built", "Bare soil/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#fee08b", "#4575b4"]

DATA_DIR = "data"


DRIVE_IDS = {
    "LANDCOVER_PRED_10M.tif": "1BhMgtE7KLFHklS2NJn_TvS4zBEJFrY4-",
    "LST_BASELINE_10M.tif":   "1dvBsiPy9LTbcNXtcb6jOWQcEbRDhQMwH",
    "LST_SCENARIO_10M.tif":   "1D3ougnYocq_2zV1ueiuvUum3AzJVtxsR",
    "LST_DELTA_10M.tif":      "1Xv3RvIinFUibHHmf4TiAp2WDwKySzjxI",
}

SPECS = [
    dict(f="LANDCOVER_PRED_10M.tif", name="U-Net land cover",
         cmap=None,      vmin=None, vmax=None, nodata=255,     discrete=True,  show=True),
    dict(f="LST_BASELINE_10M.tif",   name="LST baseline (°C)",
         cmap="inferno", vmin=30,   vmax=58,   nodata=-9999.0, discrete=False, show=False),
    dict(f="LST_SCENARIO_10M.tif",   name="LST scenario (°C)",
         cmap="inferno", vmin=30,   vmax=58,   nodata=-9999.0, discrete=False, show=False),
    dict(f="LST_DELTA_10M.tif",      name="Cooling Δ (°C)",
         cmap="Blues",   vmin=0,    vmax=2.5,  nodata=-9999.0, discrete=False, show=False),
]

MAP_CENTER = [25.10, 55.30]
MAP_ZOOM   = 10
MAX_PX     = 2000


LC_CMAP = ListedColormap(CLASS_COLORS)


def ensure_files():
    """Download from Drive if IDs provided, else expect local DATA_DIR files."""
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname, fid in DRIVE_IDS.items():
        dest = os.path.join(DATA_DIR, fname)
        if fid and not os.path.exists(dest):
            try:
                import gdown
                gdown.download(id=fid, output=dest, quiet=False)
            except Exception as e:
                st.warning(f"Could not download {fname}: {e}")
    return DATA_DIR



@st.cache_data(show_spinner=False)
def raster_to_overlay(path, cmap, vmin, vmax, nodata, discrete, max_px=MAX_PX):
    with rasterio.open(path) as src:
        dst_crs = "EPSG:4326"
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        scale = min(1.0, max_px / max(w, h))
        w, h = max(1, int(w * scale)), max(1, int(h * scale))
        transform, w, h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            dst_width=w, dst_height=h)
        data = np.full((h, w), np.nan, "float32")
        reproject(
            source=src.read(1), destination=data,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=dst_crs,
            src_nodata=nodata, dst_nodata=np.nan,
            resampling=Resampling.nearest if discrete else Resampling.bilinear)
        left, bottom, right, top = transform_bounds(src.crs, dst_crs, *src.bounds)

    alpha = np.isfinite(data)
    if nodata is not None:
        alpha &= (data != nodata)

    if discrete:
        idx = np.clip(np.nan_to_num(data, nan=0).astype(int), 0, len(CLASS_COLORS) - 1)
        rgba = (LC_CMAP(idx / max(1, len(CLASS_COLORS) - 1)) * 255).astype("uint8")
    else:
        norm = np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgba = (plt.get_cmap(cmap)(np.nan_to_num(norm)) * 255).astype("uint8")

    rgba[..., 3] = np.where(alpha, 255, 0)
    buf = BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    bounds = [[bottom, left], [top, right]]
    return url, bounds



def colorbar_png(cmap, vmin, vmax, label):
    """Return a base64 PNG of a horizontal colorbar."""
    fig, ax = plt.subplots(figsize=(3.2, 0.4))
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=[vmin, vmax, 0, 1])
    ax.set_yticks([]); ax.set_xlabel(label, fontsize=8)
    ax.tick_params(labelsize=7)
    plt.tight_layout(pad=0.2)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def land_cover_legend_html():
    rows = "".join(
        f'<div style="display:flex;align-items:center;margin:2px 0;">'
        f'<span style="width:14px;height:14px;background:{c};display:inline-block;'
        f'margin-right:6px;border:1px solid #333;"></span>'
        f'<span style="font-size:12px;">{n}</span></div>'
        for n, c in zip(CLASS_NAMES, CLASS_COLORS))
    return (f'<div style="background:white;padding:8px;border:1px solid #999;'
            f'border-radius:4px;"><b style="font-size:12px;">Land cover</b>{rows}</div>')



st.set_page_config(page_title="Dubai Urban Heat Mitigation", layout="wide")
st.title("🛰️ Satellite-Driven Urban Heat Mitigation — Dubai")
st.caption("U-Net land cover • LST baseline • What-if scenario • Cooling Δ")

data_dir = ensure_files()

st.sidebar.header("Layers")
active = {}
for s in SPECS:
    active[s["name"]] = st.sidebar.checkbox(s["name"], value=s["show"])

basemap = st.sidebar.selectbox("Basemap",
                               ["Esri.WorldImagery", "OpenStreetMap", "CartoDB positron"])

tiles = None if basemap == "Esri.WorldImagery" else basemap
fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM,
                  tiles=tiles, control_scale=True)
if basemap == "Esri.WorldImagery":
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri.WorldImagery").add_to(fmap)

missing, first_bounds = [], None
for s in SPECS:
    if not active[s["name"]]:
        continue
    path = os.path.join(data_dir, s["f"])
    if not os.path.exists(path):
        missing.append(s["f"]); continue
    url, bounds = raster_to_overlay(
        path, s["cmap"], s["vmin"], s["vmax"], s["nodata"], s["discrete"])
    folium.raster_layers.ImageOverlay(
        image=url, bounds=bounds, name=s["name"], opacity=0.85).add_to(fmap)
    first_bounds = first_bounds or bounds

if first_bounds:
    fmap.fit_bounds(first_bounds)
folium.LayerControl(collapsed=False).add_to(fmap)

col_map, col_key = st.columns([4, 1])
with col_map:
    st_folium(fmap, width=None, height=620, returned_objects=[])

with col_key:
    st.markdown("#### Legends")
    if active["U-Net land cover"]:
        st.markdown(land_cover_legend_html(), unsafe_allow_html=True)
    if active["LST baseline (°C)"] or active["LST scenario (°C)"]:
        b64 = colorbar_png("inferno", 30, 58, "LST (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)
    if active["Cooling Δ (°C)"]:
        b64 = colorbar_png("Blues", 0, 2.5, "Cooling Δ (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)

if missing:
    st.error("Missing file(s): " + ", ".join(missing) +
             f"\n\nExpected in `{data_dir}/`. Add local files or Drive IDs in CONFIG.")

with st.expander("ℹ️ About this map"):
    st.write(
        "- **U-Net land cover** — semantic segmentation (4 classes).\n"
        "- **LST baseline** — 10 m downscaled summer Land Surface Temperature.\n"
        "- **LST scenario** — baseline after applying green-roof + high-albedo interventions.\n"
        "- **Cooling Δ** — baseline − scenario (°C); only treated built-up pixels change.")
