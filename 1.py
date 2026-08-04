# -*- coding: utf-8 -*-
"""
Streamlit app — Satellite-Driven Urban Heat Mitigation (Dubai)
FAST interactive planning tool.

Speed strategy
--------------
The two costly steps are done ONCE and cached, so moving a slider is instant:
  1) Reproject land cover + baseline LST to the EPSG:4326 *display grid* once.
  2) Pre-sort each target class's pixels by temperature once.
Then each slider change is just: slice the pre-sorted indices, subtract a
coefficient, and recolour — pure NumPy on the small display array (no argsort,
no reprojection).

USAGE
-----
    pip install streamlit streamlit-folium folium rasterio numpy matplotlib pillow gdown
    streamlit run streamlit_app.py
"""

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

# ============================================================================
# CONFIG  — EDIT THIS SECTION
# ============================================================================
CLASS_NAMES  = ["Vegetation", "Impervious/Built", "Bare soil/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#fee08b", "#4575b4"]   # 0,1,2,3
# indices: 0=Vegetation 1=Impervious/Built 2=Bare soil/Sand 3=Water

COOL = {
    'green_roof':           1.45,
    'green_roof_hotarid':   1.83,
    'cool_roof_albedo':     2.00,
    'high_albedo_pavement': 2.50,
    'veg_buffer':           1.00,
}

DATA_DIR = "data"
DRIVE_IDS = {
    "LANDCOVER_PRED_10M.tif": "",   # <-- REQUIRED  Drive ID
    "LST_BASELINE_10M.tif":   "",   # <-- REQUIRED  Drive ID
}

LC_FILE  = "LANDCOVER_PRED_10M.tif"
LST_FILE = "LST_BASELINE_10M.tif"

MAP_CENTER = [25.10, 55.30]
MAP_ZOOM   = 10
DISPLAY_PX = 1100          # display-grid resolution (lower = faster)
# ============================================================================

LC_CMAP = ListedColormap(CLASS_COLORS)


# ---------------------------------------------------------------------------
# Data acquisition
# ---------------------------------------------------------------------------
def ensure_files():
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


# ---------------------------------------------------------------------------
# ONE-TIME heavy prep (cached): reproject to display grid + pre-sort classes
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparing rasters (one-time)…")
def prepare(lc_path, lst_path, max_px):
    """Reproject both rasters to a common EPSG:4326 display grid ONCE, and
    pre-compute the temperature-sorted pixel order for each target class."""
    def reproject_to_grid(path, nodata, resampling):
        with rasterio.open(path) as src:
            dst_crs = "EPSG:4326"
            transform, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds)
            scale = min(1.0, max_px / max(w, h))
            w, h = max(1, int(w * scale)), max(1, int(h * scale))
            transform, w, h = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds,
                dst_width=w, dst_height=h)
            dst = np.full((h, w), np.nan, "float32")
            reproject(source=src.read(1), destination=dst,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=transform, dst_crs=dst_crs,
                      src_nodata=nodata, dst_nodata=np.nan,
                      resampling=resampling)
            b = transform_bounds(src.crs, dst_crs, *src.bounds)
        return dst, b

    lc, bounds  = reproject_to_grid(lc_path,  255,     Resampling.nearest)
    lst, _      = reproject_to_grid(lst_path, -9999.0, Resampling.bilinear)

    lc = np.where(np.isfinite(lc), lc, 255).astype("int16")

    # Pre-sort pixels of each target class by baseline temperature (descending).
    # Applying an intervention later is then just an index-slice — no argsort.
    orders = {}
    for cls in (1, 2):                      # 1=built, 2=bare (veg buffers)
        flat = np.where((lc == cls) & np.isfinite(lst))[0] if lc.ndim == 1 else \
               np.where(((lc == cls) & np.isfinite(lst)).ravel())[0]
        if flat.size:
            temps = lst.ravel()[flat]
            orders[cls] = flat[np.argsort(-temps)]   # hottest first
        else:
            orders[cls] = np.array([], dtype=np.int64)

    return lc, lst, bounds, orders


# ---------------------------------------------------------------------------
# FULL-RESOLUTION prep (cached separately, only built when high-accuracy is on)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading full-resolution rasters (one-time)…")
def prepare_fullres(lc_path, lst_path):
    """Load both rasters at native 10 m resolution (source CRS) and pre-sort
    each target class by temperature. Used ONLY for high-accuracy metrics."""
    with rasterio.open(lc_path) as s:
        lc = s.read(1).astype("int16")
    with rasterio.open(lst_path) as s:
        lst = s.read(1).astype("float32")
    lst = np.where(lst == -9999.0, np.nan, lst)
    lc  = np.where(lc == 255, 255, lc).astype("int16")

    orders = {}
    for cls in (1, 2):
        flat = np.where(((lc == cls) & np.isfinite(lst)).ravel())[0]
        if flat.size:
            orders[cls] = flat[np.argsort(-lst.ravel()[flat])]
        else:
            orders[cls] = np.array([], dtype=np.int64)
    return lc, lst, orders


# ---------------------------------------------------------------------------
# FAST scenario (no argsort — just slices the pre-sorted order)
# ---------------------------------------------------------------------------
def run_scenario_fast(lst, orders, interventions):
    """Reproduces the thesis 'hottest-fraction of remaining eligible' logic,
    but using pre-sorted indices so it runs in microseconds."""
    out    = lst.copy()
    flat   = out.ravel()
    offset = {}                              # pixels already consumed per class
    for iv in interventions:
        tgt  = iv['target_class']
        order = orders.get(tgt)
        if order is None or order.size == 0 or iv['fraction'] <= 0:
            continue
        off  = offset.get(tgt, 0)
        rem  = order[off:]                   # still-eligible, hottest first
        n    = int(iv['fraction'] * rem.size)
        sel  = rem[:n]
        flat[sel] -= COOL[iv['name']]
        offset[tgt] = off + n
    return out


# ---------------------------------------------------------------------------
# Fast colourisers (operate on the display grid — no reprojection)
# ---------------------------------------------------------------------------
def _png_url(rgba):
    buf = BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def continuous_overlay(arr, cmap, vmin, vmax):
    norm = np.clip((arr - vmin) / (vmax - vmin + 1e-9), 0, 1)
    rgba = (plt.get_cmap(cmap)(np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[..., 3] = np.where(np.isfinite(arr), 255, 0)
    return _png_url(rgba)

def discrete_overlay(arr):
    idx = np.clip(np.nan_to_num(arr, nan=0).astype(int), 0, len(CLASS_COLORS)-1)
    rgba = (LC_CMAP(idx / max(1, len(CLASS_COLORS)-1)) * 255).astype("uint8")
    rgba[..., 3] = np.where(arr != 255, 255, 0)
    return _png_url(rgba)


# ---------------------------------------------------------------------------
# Legend / colorbar helpers
# ---------------------------------------------------------------------------
def colorbar_png(cmap, vmin, vmax, label):
    fig, ax = plt.subplots(figsize=(3.2, 0.4))
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=[vmin, vmax, 0, 1])
    ax.set_yticks([]); ax.set_xlabel(label, fontsize=8); ax.tick_params(labelsize=7)
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


# ===========================================================================
# UI
# ===========================================================================
st.set_page_config(page_title="Dubai Urban Heat Mitigation", layout="wide")
st.title("🛰️ Satellite-Driven Urban Heat Mitigation — Dubai")
st.caption("Interactive what-if planning — sliders update instantly.")

data_dir = ensure_files()
lc_path, lst_path = os.path.join(data_dir, LC_FILE), os.path.join(data_dir, LST_FILE)
if not (os.path.exists(lc_path) and os.path.exists(lst_path)):
    st.error(f"Need `{LC_FILE}` and `{LST_FILE}` in `{data_dir}/`. "
             "Add local files or Drive IDs in CONFIG.")
    st.stop()

# --- heavy prep runs ONCE (cached) -----------------------------------------
landcover, lst10, bounds, orders = prepare(lc_path, lst_path, DISPLAY_PX)
map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

# --- baseline land-cover / LST overlays computed ONCE (cached) -------------
@st.cache_data(show_spinner=False)
def base_overlays(_lc, _lst):
    return discrete_overlay(_lc), continuous_overlay(_lst, "inferno", 30, 58)
lc_url, base_url = base_overlays(landcover, lst10)

# ---------------------------------------------------------------------------
# Sidebar — scenario controls
# ---------------------------------------------------------------------------
st.sidebar.header("🎛️ Scenario controls")

st.sidebar.markdown("**Green roofs** (built-up)")
gr_on   = st.sidebar.checkbox("Enable green roofs", value=True)
gr_frac = st.sidebar.slider("Green-roof coverage (% of built)", 0, 100, 20, 5) / 100
gr_coef = st.sidebar.select_slider("Green-roof coefficient (°C)",
                                   options=[1.0, 1.45, 1.83, 2.0], value=1.83)

st.sidebar.markdown("**High-albedo paving** (built-up)")
al_on   = st.sidebar.checkbox("Enable high-albedo paving", value=True)
al_frac = st.sidebar.slider("Albedo coverage (% of built)", 0, 100, 30, 5) / 100
al_coef = st.sidebar.select_slider("Albedo coefficient (°C)",
                                   options=[1.5, 2.0, 2.5, 3.0], value=2.5)

st.sidebar.markdown("**Vegetation buffers** (bare soil/sand)")
vb_on   = st.sidebar.checkbox("Enable veg buffers", value=False)
vb_frac = st.sidebar.slider("Veg-buffer coverage (% of bare)", 0, 100, 0, 5) / 100

COOL['green_roof_hotarid']   = gr_coef
COOL['high_albedo_pavement'] = al_coef

interventions = []
if gr_on and gr_frac > 0:
    interventions.append({'name': 'green_roof_hotarid',   'fraction': gr_frac, 'target_class': 1})
if al_on and al_frac > 0:
    interventions.append({'name': 'high_albedo_pavement', 'fraction': al_frac, 'target_class': 1})
if vb_on and vb_frac > 0:
    interventions.append({'name': 'veg_buffer',           'fraction': vb_frac, 'target_class': 2})

st.sidebar.divider()
hi_acc = st.sidebar.checkbox(
    "🎯 High-accuracy metrics (full 10 m)", value=False,
    help="Recompute the metric cards at native 10 m resolution — matches your "
         "thesis numbers exactly. The map stays on the fast display grid.")

st.sidebar.divider()
st.sidebar.header("🗺️ Display layer")
layer = st.sidebar.radio("Show", ["Cooling Δ (°C)", "LST scenario (°C)",
                                  "LST baseline (°C)", "U-Net land cover"], index=0)
basemap = st.sidebar.selectbox("Basemap",
                               ["Esri.WorldImagery", "OpenStreetMap", "CartoDB positron"])

# ---------------------------------------------------------------------------
# FAST live computation for the MAP (microseconds, on the display grid)
# ---------------------------------------------------------------------------
lst_scn = run_scenario_fast(lst10, orders, interventions)
delta   = lst10 - lst_scn
v       = np.isfinite(delta)   # used by the map overlay below


def compute_metrics(lc_arr, lst_arr, ord_dict):
    scn = run_scenario_fast(lst_arr, ord_dict, interventions)
    d   = lst_arr - scn
    vv  = np.isfinite(d)
    bt  = (lc_arr == 1) & vv
    tr  = bt & (d > 0)
    return (float(np.nanmean(d[vv]))  if vv.any() else 0.0,
            float(np.nanmean(d[bt]))  if bt.any() else 0.0,
            float(np.nanmean(d[tr]))  if tr.any() else 0.0,
            float(np.nanmax(d[vv]))   if vv.any() else 0.0,
            100 * tr.sum() / max(1, bt.sum()),
            float(np.nanmean(lst_arr)), float(np.nanmean(scn)))

if hi_acc:
    lc_fr, lst_fr, orders_fr = prepare_fullres(lc_path, lst_path)
    (m_all, m_built, m_treat, max_c, pct_tr,
     base_mean, scn_mean) = compute_metrics(lc_fr, lst_fr, orders_fr)
    res_note = "full 10 m resolution"
else:
    (m_all, m_built, m_treat, max_c, pct_tr,
     base_mean, scn_mean) = compute_metrics(landcover, lst10, orders)
    res_note = f"~{DISPLAY_PX}px display grid (approx.)"

st.subheader("📊 Live scenario metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("City-wide mean cooling", f"{m_all:.3f} °C")
c2.metric("Built-up mean cooling",  f"{m_built:.3f} °C",
          help="Planner-relevant number (excludes desert dilution).")
c3.metric("Treated-pixel cooling",  f"{m_treat:.3f} °C")
c4.metric("Built-up treated",       f"{pct_tr:.1f} %")
st.caption(f"Baseline city mean: {base_mean:.2f} °C → scenario: {scn_mean:.2f} °C "
           f"• max local cooling: {max_c:.2f} °C  •  metrics @ {res_note}")

# ---------------------------------------------------------------------------
# Pick the overlay for the chosen layer (only the dynamic ones recompute)
# ---------------------------------------------------------------------------
if layer == "Cooling Δ (°C)":
    overlay_url = continuous_overlay(np.where(v & (delta > 0), delta, np.nan), "Blues", 0, 2.5)
elif layer == "LST scenario (°C)":
    overlay_url = continuous_overlay(lst_scn, "inferno", 30, 58)
elif layer == "LST baseline (°C)":
    overlay_url = base_url
else:
    overlay_url = lc_url

# ---------------------------------------------------------------------------
# Map (single overlay -> lighter + faster than stacking 4)
# ---------------------------------------------------------------------------
tiles = None if basemap == "Esri.WorldImagery" else basemap
fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=tiles, control_scale=True)
if basemap == "Esri.WorldImagery":
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri").add_to(fmap)
folium.raster_layers.ImageOverlay(overlay_url, bounds=map_bounds,
                                  name=layer, opacity=0.85).add_to(fmap)
fmap.fit_bounds(map_bounds)

col_map, col_key = st.columns([4, 1])
with col_map:
    st_folium(fmap, width=None, height=600, returned_objects=[],
              key="uhi_map")            # stable key => no full component remount
with col_key:
    st.markdown("#### Legend")
    if layer == "U-Net land cover":
        st.markdown(land_cover_legend_html(), unsafe_allow_html=True)
    elif layer == "Cooling Δ (°C)":
        b64 = colorbar_png("Blues", 0, 2.5, "Cooling Δ (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)
    else:
        b64 = colorbar_png("inferno", 30, 58, "LST (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)

st.caption("💡 Tip: keep **Cooling Δ (°C)** selected while dragging sliders — "
           "it's the layer where changes are most visible.")
