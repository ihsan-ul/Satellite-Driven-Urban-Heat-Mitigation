
import os
import numpy as np
import rasterio
import streamlit as st
import leafmap.foliumap as leafmap
import gdown

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
    """Download each input raster into DATA_IN once (cached across reruns)."""
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


def write(name, arr):
    p = prof.copy()
    p.update(count=1, dtype="float32", nodata=float("nan"))
    fp = os.path.join(DATA_OUT, name)
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
        m.add_raster(os.path.join(DATA_IN, "LANDCOVER_PRED_10M.tif"),
                     colormap="tab10", layer_name="Land cover")
    if show_base:
        m.add_raster(os.path.join(DATA_IN, "LST_BASELINE_10M.tif"),
                     colormap="inferno", layer_name="LST baseline")
    if show_scn:
        m.add_raster(os.path.join(DATA_OUT, "_scn.tif"),
                     colormap="inferno", layer_name="LST scenario")
    if show_delta:
        m.add_raster(os.path.join(DATA_OUT, "_delta.tif"),
                     colormap="Blues", layer_name="Cooling delta")
    m.add_legend(title="Land cover", labels=CLASS_NAMES, colors=CLASS_COLORS)
    m.to_streamlit(height=720)
