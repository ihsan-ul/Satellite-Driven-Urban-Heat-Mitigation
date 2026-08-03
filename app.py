"""
Dubai Urban Heat Island — Predictive Decision-Support Web App
=============================================================
Interactive Streamlit + leafmap website. Planners move sliders to run
"what-if" mitigation scenarios and instantly see the re-simulated cooling
map, with toggleable layers (pan / zoom / scroll built in).

Run locally:      streamlit run app.py
Data location:    set env var UHI_DATA to the folder holding the *_10M.tif
                  outputs (defaults to the current directory).
Deploy:           push app.py + requirements.txt + the four *_10M.tif files
                  to a GitHub repo, then deploy on share.streamlit.io.
"""
import os
import numpy as np
import rasterio
import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide", page_title="Dubai UHI Decision-Support")
DATA = os.environ.get("UHI_DATA", ".")

CLASS_NAMES = ["Vegetation", "Impervious/Built", "Bare/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#fee08b", "#4575b4"]

# Literature cooling coefficients (deg C local LST reduction) — editable inputs.
#   green_roof_hotarid : Alaa et al. (2025), hot-arid pedestrian level
#   green_roof         : Sanchez-Cordero et al. (2025), semi-arid average LST
#   high_albedo_*      : Santamouris (2014) / Fork et al. (2025) reflective range
COOL = {
    "green_roof": 1.45,
    "green_roof_hotarid": 1.83,
    "cool_roof_albedo": 2.0,
    "high_albedo_pavement": 2.5,
    "veg_buffer": 1.0,
}


@st.cache_data
def load(name):
    with rasterio.open(os.path.join(DATA, name)) as s:
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


def write(name, arr):
    p = prof.copy()
    p.update(count=1, dtype="float32", nodata=float("nan"))
    fp = os.path.join(DATA, name)
    with rasterio.open(fp, "w", **p) as d:
        d.write(arr.astype("float32"), 1)
    return fp


write("_scn.tif", scn)
write("_delta.tif", np.nan_to_num(delta))

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
        m.add_raster(os.path.join(DATA, "LANDCOVER_PRED_10M.tif"), colormap="tab10", layer_name="Land cover")
    if show_base:
        m.add_raster(os.path.join(DATA, "LST_BASELINE_10M.tif"), colormap="inferno", layer_name="LST baseline")
    if show_scn:
        m.add_raster(os.path.join(DATA, "_scn.tif"), colormap="inferno", layer_name="LST scenario")
    if show_delta:
        m.add_raster(os.path.join(DATA, "_delta.tif"), colormap="Blues", layer_name="Cooling delta")
    m.add_legend(title="Land cover", labels=CLASS_NAMES, colors=CLASS_COLORS)
    m.to_streamlit(height=720)
