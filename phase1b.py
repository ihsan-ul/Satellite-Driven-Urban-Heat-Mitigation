!pip install rasterio scikit-learn scipy -q

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

STACK_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
OUT_DIR    = "/content/drive/MyDrive/thesis_phase1_exports"
OUT_PATH   = os.path.join(OUT_DIR, "dubai_LST_downscaled.tif")
os.makedirs(OUT_DIR, exist_ok=True)

BAND_ORDER = ["B2","B3","B4","B8","B11","B12","NDVI","NDBI","NDWI","LST","DEM","GHSL","LABEL"]
PREDICTOR_BANDS = ["NDVI","NDBI","NDWI","B4","B8","B11","B12","DEM","GHSL"]

COARSE_FACTOR = 3
CV_BLOCK      = 20
SEED = 42

def bidx(name): return BAND_ORDER.index(name) + 1

with rasterio.open(STACK_PATH) as src:
    profile = src.profile
    lst   = src.read(bidx("LST")).astype("float32")
    preds = np.stack([src.read(bidx(b)) for b in PREDICTOR_BANDS], axis=-1).astype("float32")

H, W = lst.shape
finite = np.isfinite(lst)
lst_clean = np.where(finite, lst, np.nan)
preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Loaded LST + {len(PREDICTOR_BANDS)} predictors. Scene {H}x{W}.")
print(f"Baseline LST (C): mean {np.nanmean(lst_clean):.2f}, "
      f"min {np.nanmin(lst_clean):.1f}, max {np.nanmax(lst_clean):.1f}")

def coarsen(arr, f):
    h, w = arr.shape[:2]
    hc, wc = h // f, w // f
    arr = arr[:hc*f, :wc*f]
    if arr.ndim == 2:
        blk = arr.reshape(hc, f, wc, f)
        return np.nanmean(blk, axis=(1, 3))
    c = arr.shape[2]
    blk = arr.reshape(hc, f, wc, f, c)
    return np.nanmean(blk, axis=(1, 3))

lst_c   = coarsen(lst_clean, COARSE_FACTOR)
preds_c = coarsen(np.where(finite[..., None], preds, np.nan), COARSE_FACTOR)
hc, wc = lst_c.shape
print(f"Coarse grid: {hc}x{wc} (factor {COARSE_FACTOR} -> ~{10*COARSE_FACTOR} m).")

Xc = preds_c.reshape(-1, preds_c.shape[-1])
yc = lst_c.reshape(-1)
ok = np.isfinite(yc) & np.isfinite(Xc).all(axis=1)
Xc, yc = Xc[ok], yc[ok]

from sklearn.ensemble import HistGradientBoostingRegressor

import time
from sklearn.ensemble import HistGradientBoostingRegressor

rr, cc = np.meshgrid(np.arange(hc), np.arange(wc), indexing="ij")
blocks_full = ((rr // CV_BLOCK) * (wc // CV_BLOCK + 1) + (cc // CV_BLOCK)).reshape(-1)[ok]

print(f"CV on {len(yc):,} coarse samples, {len(np.unique(blocks_full))} spatial blocks.")

def make_model():
    return HistGradientBoostingRegressor(
        max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, random_state=SEED)

gkf = GroupKFold(n_splits=3)                 # 3 folds is plenty and faster
cv_rmse, cv_r2 = [], []
for i, (tr, te) in enumerate(gkf.split(Xc, yc, groups=blocks_full), 1):
    t0 = time.time()
    m = make_model().fit(Xc[tr], yc[tr])
    p = m.predict(Xc[te])
    rmse = np.sqrt(mean_squared_error(yc[te], p)); r2 = r2_score(yc[te], p)
    cv_rmse.append(rmse); cv_r2.append(r2)
    print(f"  fold {i}/3  RMSE={rmse:.3f} C  R2={r2:.3f}  ({time.time()-t0:.0f}s)")

print(f"  Spatial-CV RMSE : {np.mean(cv_rmse):.3f} +/- {np.std(cv_rmse):.3f} C")
print(f"  Spatial-CV R^2  : {np.mean(cv_r2):.3f} +/- {np.std(cv_r2):.3f}")

t0 = time.time()
rf = make_model().fit(Xc, yc)
print(f"  Final fit done ({time.time()-t0:.0f}s). "
      f"In-sample R^2 = {rf.score(Xc, yc):.3f}  [training fit only]")

from sklearn.inspection import permutation_importance
samp = np.random.default_rng(SEED).choice(len(yc), size=min(5000, len(yc)), replace=False)
pi = permutation_importance(rf, Xc[samp], yc[samp], n_repeats=2, random_state=SEED)
imp = sorted(zip(PREDICTOR_BANDS, pi.importances_mean), key=lambda x: -x[1])
print("  Predictor importance:", [(n, round(float(v), 3)) for n, v in imp])

Xf = preds.reshape(-1, preds.shape[-1])
print("  Predicting on all fine pixels...")
t0 = time.time()
lst_pred_fine = rf.predict(Xf).reshape(H, W).astype("float32")
print(f"  Fine prediction done ({time.time()-t0:.0f}s).")

lst_pred_coarse = rf.predict(preds_c.reshape(-1, preds_c.shape[-1]))
lst_pred_coarse = np.where(np.isfinite(lst_c.reshape(-1)),
                           lst_pred_coarse, np.nan).reshape(hc, wc)
residual_coarse = np.nan_to_num(lst_c - lst_pred_coarse, nan=0.0)
residual_fine = zoom(residual_coarse, (H/hc, W/wc), order=1)[:H, :W]
lst_downscaled = np.where(finite, lst_pred_fine + residual_fine, np.nan).astype("float32")


def recoarsen_mean(a, f):
    hc2, wc2 = H // f, W // f
    blk = a[:hc2*f, :wc2*f].reshape(hc2, f, wc2, f)
    return np.nanmean(blk, axis=(1, 3))

ds_recoarse = recoarsen_mean(lst_downscaled, COARSE_FACTOR)
cmp = np.isfinite(lst_c) & np.isfinite(ds_recoarse)
consistency_rmse = np.sqrt(np.nanmean((ds_recoarse[cmp] - lst_c[cmp])**2))
print(f"\n  Mass-conservation consistency RMSE: {consistency_rmse:.4f} C "
      f"(guaranteed low — sanity check, NOT accuracy)")

def local_texture(a, f):
    from scipy.ndimage import uniform_filter
    m = np.isfinite(a).astype("float32")
    a0 = np.where(np.isfinite(a), a, 0.0)
    num = uniform_filter(a0, size=f); den = uniform_filter(m, size=f)
    local_mean = np.where(den > 0, num / den, np.nan)
    return (a - local_mean)

tex_before = local_texture(lst_clean, COARSE_FACTOR)[finite].std()
tex_after  = local_texture(lst_downscaled, COARSE_FACTOR)[finite].std()
print(f"\n  Local texture std BEFORE: {tex_before:.4f} C")
print(f"  Local texture std AFTER : {tex_after:.4f} C  "
      f"(higher = fine urban detail injected from optical predictors)")
print("=========================================================")

p2, p98 = np.nanpercentile(lst_clean[finite], [2, 98])
fig, ax = plt.subplots(1, 2, figsize=(14, 7))
im0 = ax[0].imshow(np.where(finite, lst_clean, np.nan), cmap="inferno", vmin=p2, vmax=p98)
ax[0].set_title("Baseline LST (30 m native, resampled)"); ax[0].axis("off")
fig.colorbar(im0, ax=ax[0], fraction=0.046)
im1 = ax[1].imshow(lst_downscaled, cmap="inferno", vmin=p2, vmax=p98)
ax[1].set_title("Downscaled LST (10 m, thermally sharpened)"); ax[1].axis("off")
fig.colorbar(im1, ax=ax[1], fraction=0.046)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "phase1b_raw_vs_downscaled.png"),
                                dpi=150, bbox_inches="tight"); plt.show()

prof = profile.copy()
prof.update(count=1, dtype="float32", nodata=-9999)
with rasterio.open(OUT_PATH, "w", **prof) as dst:
    dst.write(np.nan_to_num(lst_downscaled, nan=-9999), 1)
print(f"\nSaved downscaled LST -> {OUT_PATH}")
