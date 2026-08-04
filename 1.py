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
CLASS_INDEX  = {n: i for i, n in enumerate(CLASS_NAMES)}

COOL = {
    'green_roof':           1.45,
    'green_roof_hotarid':   1.83,
    'cool_roof_albedo':     2.00,
    'high_albedo_pavement': 2.50,
    'veg_buffer':           1.00,
}

DATA_DIR = "data"

DRIVE_IDS = {
    "LANDCOVER_PRED_10M.tif": "1mMTNbQAD9eKKKqwtibf-VA5uiiZBznvV",
    "LST_BASELINE_10M.tif":   "1teySbI4X0UcjVs-oL07cnfInywGnG1mY",
    "LST_SCENARIO_10M.tif":   "1ODuEu7UjTdUFiQiiVyHfEIm33u10pAhZ",
    "LST_DELTA_10M.tif":      "1--mYkjmkw1fvgK9UkK9RWgDfDLvzw85I",
}

LC_FILE   = "LANDCOVER_PRED_10M.tif"
LST_FILE  = "LST_BASELINE_10M.tif"

MAP_CENTER = [25.10, 55.30]
MAP_ZOOM   = 10

LC_CMAP = ListedColormap(CLASS_COLORS)

STRATEGIES = {
    "Hottest first (worst hotspots)": "hottest",
    "Coolest first (comparison)":     "coolest",
    "Area-based (all areas fairly)":  "area",
}


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


@st.cache_data(show_spinner=False)
def load_array(path, nodata=None):
    """Read band 1 as float32; return array, src profile, and lat/lon bounds."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        prof = {"crs": src.crs, "transform": src.transform,
                "width": src.width, "height": src.height, "bounds": src.bounds}
        bounds4326 = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return arr, prof, bounds4326


def run_scenario(lst, lc, interventions):
    """Apply cooling coefficients to eligible pixels of each target class.
    A `done` mask ensures no pixel is treated twice.

    Each intervention may specify a selection `strategy`:
      * 'hottest' -> treat the hottest frac share first (default; worst hotspots)
      * 'coolest' -> treat the coolest frac share first (comparison mode)
      * 'area'    -> treat an evenly-spread random frac share regardless of
                     temperature, so every part of the class gets a fair chance
    """
    out  = lst.copy().astype("float32")
    done = np.zeros_like(lc, bool)
    rng  = np.random.default_rng(0)
    for iv in interventions:
        coef, frac, tgt = COOL[iv['name']], iv['fraction'], iv['target_class']
        strat = iv.get('strategy', 'hottest')
        cand  = (lc == tgt) & np.isfinite(out) & (~done)
        idx   = np.where(cand.ravel())[0]
        if idx.size == 0 or frac <= 0:
            continue
        n = int(frac * idx.size)
        if n <= 0:
            continue
        if strat == 'hottest':
            order = idx[np.argsort(-out.ravel()[idx])][:n]
        elif strat == 'coolest':
            order = idx[np.argsort(out.ravel()[idx])][:n]
        else:
            order = rng.choice(idx, size=n, replace=False)
        rr, cc = np.unravel_index(order, lc.shape)
        out[rr, cc] -= coef
        done[rr, cc] = True
    return out


@st.cache_data(show_spinner=False)
def array_to_overlay(_arr, _prof, cmap, vmin, vmax, discrete, key, max_px):
    """_arr in the source CRS; `key` makes the cache unique per logical layer."""
    src_crs, src_transform = _prof["crs"], _prof["transform"]
    src_w, src_h, bounds = _prof["width"], _prof["height"], _prof["bounds"]
    dst_crs = "EPSG:4326"
    transform, w, h = calculate_default_transform(
        src_crs, dst_crs, src_w, src_h, *bounds)
    scale = min(1.0, max_px / max(w, h))
    w, h = max(1, int(w * scale)), max(1, int(h * scale))
    transform, w, h = calculate_default_transform(
        src_crs, dst_crs, src_w, src_h, *bounds, dst_width=w, dst_height=h)

    data = np.full((h, w), np.nan, "float32")
    reproject(source=np.ascontiguousarray(_arr), destination=data,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=transform, dst_crs=dst_crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest if discrete else Resampling.bilinear)
    left, bottom, right, top = transform_bounds(src_crs, dst_crs, *bounds)

    alpha = np.isfinite(data)
    if discrete:
        idx = np.clip(np.nan_to_num(data, nan=0).astype(int), 0, len(CLASS_COLORS)-1)
        rgba = (LC_CMAP(idx / max(1, len(CLASS_COLORS)-1)) * 255).astype("uint8")
    else:
        norm = np.clip((data - vmin) / (vmax - vmin + 1e-9), 0, 1)
        rgba = (plt.get_cmap(cmap)(np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[..., 3] = np.where(alpha, 255, 0)

    buf = BytesIO(); Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    return url, [[bottom, left], [top, right]]


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


st.set_page_config(page_title="Dubai Urban Heat Mitigation", layout="wide")
st.title("🛰️ Satellite-Driven Urban Heat Mitigation — Dubai")
st.caption("Interactive what-if planning: choose surfaces, strategy, and intensity.")

data_dir = ensure_files()
lc_path, lst_path = os.path.join(data_dir, LC_FILE), os.path.join(data_dir, LST_FILE)

if not (os.path.exists(lc_path) and os.path.exists(lst_path)):
    st.error(f"Need `{LC_FILE}` and `{LST_FILE}` in `{data_dir}/`. "
             "Add local files or Drive IDs in CONFIG.")
    st.stop()

landcover, lc_prof, bounds = load_array(lc_path, nodata=255)
landcover = np.nan_to_num(landcover, nan=255).astype("int16")
lst10, lst_prof, _         = load_array(lst_path, nodata=-9999.0)


st.sidebar.header("🎛️ Scenario controls")

strat_label = st.sidebar.selectbox(
    "Selection strategy", list(STRATEGIES.keys()), index=0,
    help="How pixels are chosen WITHIN a target class. 'Hottest first' targets "
         "the worst hotspots; 'Area-based' spreads treatment fairly across the "
         "whole class so newer/cooler buildings also get treated.")
strategy = STRATEGIES[strat_label]

st.sidebar.divider()

st.sidebar.markdown("**Green roofs**")
gr_on   = st.sidebar.checkbox("Enable green roofs", value=True)
gr_tgt  = st.sidebar.selectbox("Green-roof target surface", CLASS_NAMES,
                               index=1, key="gr_tgt")
gr_frac = st.sidebar.slider("Green-roof coverage (%)", 0, 100, 20, 5) / 100
gr_coef = st.sidebar.select_slider("Green-roof coefficient (°C)",
                                   options=[1.0, 1.45, 1.83, 2.0], value=1.83)

st.sidebar.markdown("**High-albedo paving**")
al_on   = st.sidebar.checkbox("Enable high-albedo paving", value=True)
al_tgt  = st.sidebar.selectbox("Albedo target surface", CLASS_NAMES,
                               index=1, key="al_tgt")
al_frac = st.sidebar.slider("Albedo coverage (%)", 0, 100, 30, 5) / 100
al_coef = st.sidebar.select_slider("Albedo coefficient (°C)",
                                   options=[1.5, 2.0, 2.5, 3.0], value=2.5)

st.sidebar.markdown("**Vegetation buffers**")
vb_on   = st.sidebar.checkbox("Enable veg buffers", value=False)
vb_tgt  = st.sidebar.selectbox("Veg-buffer target surface", CLASS_NAMES,
                               index=2, key="vb_tgt")
vb_frac = st.sidebar.slider("Veg-buffer coverage (%)", 0, 100, 0, 5) / 100

COOL['green_roof_hotarid']   = gr_coef
COOL['high_albedo_pavement'] = al_coef

interventions = []
if gr_on and gr_frac > 0:
    interventions.append({'name': 'green_roof_hotarid', 'fraction': gr_frac,
                          'target_class': CLASS_INDEX[gr_tgt], 'strategy': strategy})
if al_on and al_frac > 0:
    interventions.append({'name': 'high_albedo_pavement', 'fraction': al_frac,
                          'target_class': CLASS_INDEX[al_tgt], 'strategy': strategy})
if vb_on and vb_frac > 0:
    interventions.append({'name': 'veg_buffer', 'fraction': vb_frac,
                          'target_class': CLASS_INDEX[vb_tgt], 'strategy': strategy})

st.sidebar.divider()
st.sidebar.header("🗺️ Display layers")
show_lc     = st.sidebar.checkbox("U-Net land cover", value=False)
show_base   = st.sidebar.checkbox("LST baseline (°C)", value=False)
show_scn    = st.sidebar.checkbox("LST scenario (°C)", value=False)
show_delta  = st.sidebar.checkbox("Cooling Δ (°C)", value=True)
basemap     = st.sidebar.selectbox("Basemap",
                                   ["Esri.WorldImagery", "OpenStreetMap", "CartoDB positron"])

st.sidebar.divider()
st.sidebar.header("🎨 Appearance")
opacity = st.sidebar.slider("Layer opacity", 0.0, 1.0, 0.85, 0.05,
    help="Raise toward 1.0 so the basemap stops bleeding through gaps between buildings.")
res_px = st.sidebar.select_slider("Display resolution (px)",
    options=[800, 1100, 1500, 2000, 2500], value=1500,
    help="Higher = sharper overlays and more solid built-up blocks, but slower.")


lst_scn = run_scenario(lst10, landcover, interventions)
delta   = lst10 - lst_scn


v          = np.isfinite(delta)
target_ids = {iv['target_class'] for iv in interventions} or {1}
in_targets = np.isin(landcover, list(target_ids)) & v
treated    = in_targets & (delta > 0)
m_all   = float(np.nanmean(delta[v]))            if v.any()          else 0.0
m_tgt   = float(np.nanmean(delta[in_targets]))   if in_targets.any() else 0.0
m_treat = float(np.nanmean(delta[treated]))      if treated.any()    else 0.0
max_c   = float(np.nanmax(delta[v]))             if v.any()          else 0.0
pct_tr  = 100 * treated.sum() / max(1, in_targets.sum())

st.subheader("📊 Live scenario metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("City-wide mean cooling", f"{m_all:.3f} °C")
c2.metric("Target-surface cooling", f"{m_tgt:.3f} °C",
          help="Mean cooling over the surface(s) you targeted (excludes untreated classes).")
c3.metric("Treated-pixel cooling",  f"{m_treat:.3f} °C")
c4.metric("Target treated",         f"{pct_tr:.1f} %")
st.caption(f"Strategy: **{strat_label}**  •  baseline city mean {np.nanmean(lst10):.2f} °C → "
           f"scenario {np.nanmean(lst_scn):.2f} °C  •  max local cooling {max_c:.2f} °C")


tiles = None if basemap == "Esri.WorldImagery" else basemap
fmap = folium.Map(location=MAP_CENTER, zoom_start=MAP_ZOOM, tiles=tiles, control_scale=True)
if basemap == "Esri.WorldImagery":
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Esri.WorldImagery").add_to(fmap)

scn_key = (f"{strategy}{gr_on}{gr_tgt}{gr_frac}{gr_coef}"
           f"{al_on}{al_tgt}{al_frac}{al_coef}{vb_on}{vb_tgt}{vb_frac}{res_px}")

if show_lc:
    url, b = array_to_overlay(landcover.astype("float32"), lc_prof,
                              None, None, None, True, key=f"lc{res_px}", max_px=res_px)
    folium.raster_layers.ImageOverlay(url, bounds=b, name="U-Net land cover",
                                      opacity=opacity).add_to(fmap)
if show_base:
    url, b = array_to_overlay(lst10, lst_prof, "inferno", 30, 58, False,
                              key=f"base{res_px}", max_px=res_px)
    folium.raster_layers.ImageOverlay(url, bounds=b, name="LST baseline (°C)",
                                      opacity=opacity).add_to(fmap)
if show_scn:
    url, b = array_to_overlay(lst_scn, lst_prof, "inferno", 30, 58, False,
                              key="scn"+scn_key, max_px=res_px)
    folium.raster_layers.ImageOverlay(url, bounds=b, name="LST scenario (°C)",
                                      opacity=opacity).add_to(fmap)
if show_delta:
    url, b = array_to_overlay(np.where(v, delta, np.nan), lst_prof, "Blues", 0, 2.5,
                              False, key="delta"+scn_key, max_px=res_px)
    folium.raster_layers.ImageOverlay(url, bounds=b, name="Cooling Δ (°C)",
                                      opacity=opacity).add_to(fmap)

fmap.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
folium.LayerControl(collapsed=False).add_to(fmap)

col_map, col_key = st.columns([4, 1])
with col_map:
    st_folium(fmap, width=None, height=600, returned_objects=[])

with col_key:
    st.markdown("#### Legends")
    if show_lc:
        st.markdown(land_cover_legend_html(), unsafe_allow_html=True)
    if show_base or show_scn:
        b64 = colorbar_png("inferno", 30, 58, "LST (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)
    if show_delta:
        b64 = colorbar_png("Blues", 0, 2.5, "Cooling Δ (°C)")
        st.markdown(f'<img src="data:image/png;base64,{b64}" width="100%">',
                    unsafe_allow_html=True)

with st.expander("ℹ️ How the scenario works"):
    st.write(
        "- **Target surface** — pick which land-cover class each intervention applies to.\n"
        "- **Selection strategy** — *Hottest first* treats the worst hotspots (best "
        "cooling per area); *Area-based* spreads treatment fairly so newer/cooler "
        "buildings are also cooled; *Coolest first* is a comparison mode.\n"
        "- A `done` mask prevents any pixel being treated twice when interventions share "
        "a target class.\n"
        "- Cooling is a first-order **constant subtraction** using empirical coefficients "
        "(Alaa et al. 2025 and related) — not a physical energy-balance simulation.")
