!pip install rasterio scikit-learn scipy -q

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom, uniform_filter
from sklearn.ensemble import RandomForestRegressor


STACK_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
OUT_DIR    = "/content/phase1b_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

BANDS = ["B2", "B3", "B4", "B8", "B11", "B12",
         "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL", "LABEL"]
def bidx(name):
    return BANDS.index(name) + 1

PREDICTOR_BANDS = ["NDVI", "NDBI", "NDWI", "B4", "B8", "B11", "B12"]

COARSE_FACTOR = 10


with rasterio.open(STACK_PATH) as src:
    profile = src.profile
    lst = src.read(bidx("LST")).astype("float32")
    preds = np.stack([src.read(bidx(b)) for b in PREDICTOR_BANDS],
                     axis=-1).astype("float32")

finite_lst = np.isfinite(lst)
lst_clean = np.nan_to_num(lst, nan=np.nanmean(lst[finite_lst]))
preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

H, W = lst.shape
print(f"Loaded LST + {len(PREDICTOR_BANDS)} predictors. Scene: {H}x{W}")
print(f"Raw LST (C): mean {np.nanmean(lst[finite_lst]):.2f}, "
      f"std {np.nanstd(lst[finite_lst]):.3f}, "
      f"min {np.nanmin(lst[finite_lst]):.1f}, max {np.nanmax(lst[finite_lst]):.1f}")


local_mean = uniform_filter(lst_clean, size=COARSE_FACTOR)
local_texture = lst_clean - local_mean
print("\n[DIAGNOSTIC] Local LST texture (pixel - local mean):")
print(f"  std of local texture = {local_texture[finite_lst].std():.4f} C")
print("  (Very low std here = LST is smooth / lacks fine detail.)")

p2, p98 = np.nanpercentile(lst[finite_lst], [2, 98])
regional_mean = np.nanmean(lst[finite_lst])
anomaly = np.where(finite_lst, lst - regional_mean, np.nan)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))
im0 = ax[0].imshow(np.where(finite_lst, lst, np.nan), cmap="inferno")
ax[0].set_title("RAW LST (min-max stretch)\n[looks smooth]")
fig.colorbar(im0, ax=ax[0], fraction=0.046)

im1 = ax[1].imshow(np.where(finite_lst, lst, np.nan), cmap="inferno",
                   vmin=p2, vmax=p98)
ax[1].set_title(f"RAW LST (2-98% stretch)\n[more contrast]")
fig.colorbar(im1, ax=ax[1], fraction=0.046)

im2 = ax[2].imshow(anomaly, cmap="RdBu_r", vmin=-4, vmax=4)
ax[2].set_title("SUHI ANOMALY (LST - regional mean)\n[hotspots vs coolspots]")
fig.colorbar(im2, ax=ax[2], fraction=0.046)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "A_diagnostic_views.png"), dpi=150)
plt.show()

def coarsen(arr, factor):
    """Block-mean downsample to a coarse grid (handles 2D or 3D)."""
    h, w = arr.shape[:2]
    hc, wc = h // factor, w // factor
    arr = arr[:hc*factor, :wc*factor]
    if arr.ndim == 2:
        return arr.reshape(hc, factor, wc, factor).mean(axis=(1, 3))
    else:
        c = arr.shape[2]
        return arr.reshape(hc, factor, wc, factor, c).mean(axis=(1, 3))

lst_c   = coarsen(lst_clean, COARSE_FACTOR)
preds_c = coarsen(preds, COARSE_FACTOR)
hc, wc = lst_c.shape
print(f"\n[DOWNSCALE] Coarse grid: {hc}x{wc} (factor {COARSE_FACTOR}).")

Xc = preds_c.reshape(-1, preds_c.shape[-1])
yc = lst_c.reshape(-1)
mask_c = np.isfinite(yc)
rf = RandomForestRegressor(n_estimators=100, max_depth=14,
                           n_jobs=-1, random_state=42)
rf.fit(Xc[mask_c], yc[mask_c])
r2 = rf.score(Xc[mask_c], yc[mask_c])
print(f"[DOWNSCALE] Coarse regression R^2 = {r2:.3f}")

Xf = preds.reshape(-1, preds.shape[-1])
lst_pred_fine = rf.predict(Xf).reshape(H, W).astype("float32")

lst_pred_coarse = rf.predict(Xc).reshape(hc, wc)
residual_coarse = lst_c - lst_pred_coarse
residual_fine = zoom(residual_coarse, (H / hc, W / wc), order=1)[:H, :W]

lst_downscaled = lst_pred_fine + residual_fine
lst_downscaled = np.where(finite_lst, lst_downscaled, np.nan)


new_texture = (np.nan_to_num(lst_downscaled) -
               uniform_filter(np.nan_to_num(lst_downscaled), size=COARSE_FACTOR))

hc2 = H // COARSE_FACTOR; wc2 = W // COARSE_FACTOR
trim   = lst_downscaled[:hc2*COARSE_FACTOR, :wc2*COARSE_FACTOR]
blocks = trim.reshape(hc2, COARSE_FACTOR, wc2, COARSE_FACTOR)
lst_ds_recoarse = np.nanmean(blocks, axis=(1, 3))
valid_count = np.sum(np.isfinite(blocks), axis=(1, 3))
full_cells  = valid_count == (COARSE_FACTOR * COARSE_FACTOR)
compare = full_cells & np.isfinite(lst_c) & np.isfinite(lst_ds_recoarse)
rmse = np.sqrt(np.nanmean((lst_ds_recoarse[compare] - lst_c[compare])**2))

print("\n================ DOWNSCALING VALIDATION ================")
print(f"  Local texture std BEFORE : {local_texture[finite_lst].std():.4f} C")
print(f"  Local texture std AFTER  : {new_texture[finite_lst].std():.4f} C  "
      f"(higher = more urban detail recovered)")
print(f"  Re-aggregation RMSE      : {rmse:.3f} C  (target <= 2.7 C)")
if rmse <= 2.7:
    print("  PASS: consistent with Andriambololonaharisoamalala et al. 2025.")
else:
    print("  Above 2.7 C -> try COARSE_FACTOR 8, or add DEM/GHSL predictors.")
print("=======================================================")


fig, ax = plt.subplots(1, 2, figsize=(14, 7))
im0 = ax[0].imshow(np.where(finite_lst, lst, np.nan), cmap="inferno",
                   vmin=p2, vmax=p98)
ax[0].set_title("RAW LST (smooth)")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(lst_downscaled, cmap="inferno", vmin=p2, vmax=p98)
ax[1].set_title("DOWNSCALED LST (sharpened, 10 m texture)")
fig.colorbar(im1, ax=ax[1], fraction=0.046)
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "B_raw_vs_downscaled.png"), dpi=150)
plt.show()


out_path = os.path.join(OUT_DIR, "dubai_LST_downscaled.tif")
prof = profile.copy()
prof.update(count=1, dtype="float32")
with rasterio.open(out_path, "w", **prof) as dst:
    dst.write(np.nan_to_num(lst_downscaled, nan=-9999).astype("float32"), 1)
    dst.nodata = -9999
print(f"\nSaved downscaled LST -> {out_path}")

