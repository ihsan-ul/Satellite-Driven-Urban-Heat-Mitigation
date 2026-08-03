
import os
import io
import base64
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import matplotlib
import streamlit as st
import folium
import leafmap.foliumap as leafmap
import gdown
from PIL import Image

st.set_page_config(layout="wide", page_title="Dubai UHI Decision-Support")


DATA_IN = os.environ.get("UHI_DATA", "/tmp/uhi_data")
DATA_OUT = os.environ.get("UHI_OUT", "/tmp/uhi_out")
os.makedirs(DATA_IN, exist_ok=True)
os.makedirs(DATA_OUT, exist_ok=True)

DRIVE_FILE_IDS = {
    "LST_BASELINE_10M.tif":   "1dvBsiPy9LTbcNXtcb6jOWQcEbRDhQMwH",
    "LANDCOVER_PRED_10M.tif": "1BhMgtE7KLFHklS2NJn_TvS4zBEJFrY4-",
}


@st.cache_data(show_spinner="Downloading rasters from Google Drive…")
def fetch_inputs():
    for name, fid in DRIVE_FILE_IDS.items():
        dst = os.path.join(DATA_IN, name)
        if not os.path.exists(dst) or os.path.getsize(dst) == 0:
            if not fid or fid.startswith("PASTE_"):
                st.error(f"Missing Google Drive ID for {name}. "
                         f"Edit DRIVE_FILE_IDS in app.py.")
                st.stop()
            gdown.download(id=fid, output=dst, quiet=True)
    return True


fetch_inputs()

CLASS_NAMES = ["Vegetation", "Impervious/Built", "Bare/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#fee08b", "#4575b4"]

COOL = {
    "green_roof": 1.45,
    "green_roof_hotarid": 1.83,
    "cool_roof_albedo": 2.0,
    "high_albedo_pavement": 2.5,
    "veg_buffer": 1.0,
}


@st.cache_data
def load(name):
    with rasterio.open(os.path.join(DATA_IN, name)) as s:
        return s.read(1), s.profile


lst, prof = load("LST_BASELINE_10M.tif")
lc, _ = load("LANDCOVER_PRED_10M.tif")

st.sidebar.title("Dubai UHI Mitigation Simulator")
gr = st.sidebar.slider("Green roofs on built-up (%)", 0, 100, 20)
al = st.sidebar.slider("High-albedo on bare/paved (%)", 0, 100, 30)
st.sidebar.markdown("---")
show_lc = st.sidebar.checkbox("Land cover", True)
show_base = st.sidebar.checkbox("LST baseline", True)
show_scn = st.sidebar.checkbox("LST scenario", True)
show_delta = st.sidebar.checkbox("Cooling delta", True)


def scenario(lst, lc, gr, al):
    out = lst.copy().astype("float32")
    done = np.zeros_like(lc, bool)
    for frac, coef, tgt in [
        (gr / 100, COOL["green_roof_hotarid"], 1),
        (al / 100, COOL["high_albedo_pavement"], 2),
    ]:
        cand = (lc == tgt) & np.isfinite(out) & (~done)
        idx = np.where(cand.ravel())[0]
        if idx.size and frac > 0:
            order = idx[np.argsort(-out.ravel()[idx])][: int(frac * idx.size)]
            r, c = np.unravel_index(order, lc.shape)
            out[r, c] -= coef
            done[r, c] = True
    return out


scn = scenario(lst, lc, gr, al)
delta = lst - scn


def latlon_bounds(profile):
    """Return [[south, west], [north, east]] in EPSG:4326 for folium."""
    w, s, e, n = rasterio.transform.array_bounds(
        profile["height"], profile["width"], profile["transform"]
    )
    src_crs = profile.get("crs")
    if src_crs and src_crs.to_epsg() != 4326:
        w, s, e, n = transform_bounds(src_crs, "EPSG:4326", w, s, e, n)
    return [[s, w], [n, e]]


BOUNDS = latlon_bounds(prof)


def _to_png_url(rgba):
    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def colorize_continuous(arr, cmap_name, alpha=200):
    a = arr.astype("float32")
    finite = np.isfinite(a)
    if finite.any():
        vmin = np.nanpercentile(a[finite], 2)
        vmax = np.nanpercentile(a[finite], 98)
    else:
        vmin, vmax = 0.0, 1.0
    norm = np.clip((a - vmin) / (vmax - vmin + 1e-9), 0, 1)
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = (cmap(np.nan_to_num(norm)) * 255).astype("uint8")
    rgba[..., 3] = np.where(finite, alpha, 0).astype("uint8")
    return rgba


def colorize_categorical(arr, colors, alpha=200):
    a = arr.astype("int")
    rgba = np.zeros((*a.shape, 4), dtype="uint8")
    for i, hexc in enumerate(colors):
        h = hexc.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        m = a == i
        rgba[m] = [r, g, b, alpha]
    return rgba


def add_overlay(m, rgba, name, show=True):
    folium.raster_layers.ImageOverlay(
        image=_to_png_url(rgba),
        bounds=BOUNDS,
        name=name,
        opacity=1.0,
        show=show,
    ).add_to(m)



st.title("AI-Driven Urban Heat Island Mitigation — Dubai")
c1, c2 = st.columns([3, 1])
with c2:
    st.metric("Mean city cooling (C)", f"{np.nanmean(delta):.3f}")
    st.metric("Max local cooling (C)", f"{np.nanmax(delta):.2f}")
    st.metric("Baseline mean LST (C)", f"{np.nanmean(lst):.2f}")
    st.metric("Scenario mean LST (C)", f"{np.nanmean(scn):.2f}")
with c1:
    m = leafmap.Map(center=[25.10, 55.30], zoom=10)
    m.add_basemap("HYBRID")
    if show_lc:
        add_overlay(m, colorize_categorical(lc, CLASS_COLORS),
                    "Land cover", show=True)
    if show_base:
        add_overlay(m, colorize_continuous(lst, "inferno"),
                    "LST baseline", show=True)
    if show_scn:
        add_overlay(m, colorize_continuous(scn, "inferno"),
                    "LST scenario", show=True)
    if show_delta:
        add_overlay(m, colorize_continuous(delta, "Blues"),
                    "Cooling delta", show=True)
    m.add_legend(title="Land cover", labels=CLASS_NAMES, colors=CLASS_COLORS)
    folium.LayerControl(collapsed=False).add_to(m)
    m.to_streamlit(height=720)
