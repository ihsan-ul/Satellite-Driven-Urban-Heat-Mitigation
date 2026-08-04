!pip -q install "earthengine-api>=1.4.0" geemap leafmap rasterio rioxarray \
     geopandas localtileserver scikit-learn tensorflow tqdm matplotlib psutil

import os, glob, time, warnings, gc, base64
from io import BytesIO
import numpy as np
import rasterio
import ee, geemap
warnings.filterwarnings("ignore")

from google.colab import userdata
GEE_PROJECT = userdata.get('GEEID')

USE_DRIVE = True
DRIVE_DIR = "/content/drive/MyDrive/UHI_Dubai"
LOCAL_DIR = "/content/UHI_Dubai"

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)
print("Earth Engine:", ee.String("ready").getInfo(), " project:", GEE_PROJECT)

AOI  = ee.Geometry.Rectangle([55.10, 24.80, 55.55, 25.40])
UTM  = "EPSG:32640"
YEARS = list(range(2016, 2026))
SUMMER_MONTHS = [6, 7, 8]
S2_SCALE, LST_SCALE = 10, 30

NUM_CLASSES  = 4
CLASS_NAMES  = ["Vegetation", "Impervious/Built", "Bare soil/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#fee08b", "#4575b4"]

os.makedirs(LOCAL_DIR, exist_ok=True)
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    os.makedirs(DRIVE_DIR, exist_ok=True)
    OUT = DRIVE_DIR
else:
    OUT = LOCAL_DIR
print("outputs will be written to:", OUT)

def prep_landsat(img):
    qa = img.select('QA_PIXEL')
    clear = (qa.bitwiseAnd(1 << 1).eq(0)
             .And(qa.bitwiseAnd(1 << 2).eq(0))
             .And(qa.bitwiseAnd(1 << 3).eq(0))
             .And(qa.bitwiseAnd(1 << 4).eq(0)))
    lst = (img.select('ST_B10').multiply(0.00341802).add(149.0)
           .subtract(273.15).rename('LST'))
    return lst.updateMask(clear).copyProperties(img, ['system:time_start'])

def landsat_lst():
    merged = ee.ImageCollection([])
    for cid in ['LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2']:
        c = (ee.ImageCollection(cid).filterBounds(AOI)
             .filter(ee.Filter.calendarRange(YEARS[0], YEARS[-1], 'year'))
             .filter(ee.Filter.calendarRange(SUMMER_MONTHS[0], SUMMER_MONTHS[-1], 'month'))
             .map(prep_landsat))
        merged = merged.merge(c)
    return merged.select('LST').median().clip(AOI).rename('LST')

lst_baseline = landsat_lst()

def build_s2():
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(AOI)
          .filter(ee.Filter.calendarRange(YEARS[0], YEARS[-1], 'year'))
          .filter(ee.Filter.calendarRange(SUMMER_MONTHS[0], SUMMER_MONTHS[-1], 'month'))
          .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs']))

    def mask(img):
        img = img.updateMask(img.select('cs').gte(0.6))
        return (img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).divide(10000)
                .copyProperties(img, ['system:time_start']))

    comp = s2.map(mask).median().clip(AOI)
    ndvi = comp.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndbi = comp.normalizedDifference(['B11', 'B8']).rename('NDBI')
    ndwi = comp.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return comp.addBands([ndvi, ndbi, ndwi])

s2_features = build_s2()
dem = ee.Image('USGS/SRTMGL1_003').select('elevation').clip(AOI).rename('elevation')

wc = ee.ImageCollection('ESA/WorldCover/v200').first().select('Map')
labels = wc.remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                  [ 0,  0,  0,  0,  1,  2,  2,  3,  0,  0,   0]).rename('class').clip(AOI)

FEAT_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDBI', 'NDWI', 'elevation']
feat_stack = s2_features.addBands(dem).select(FEAT_BANDS).toFloat()
print("Composites built. Feature bands:", feat_stack.bandNames().getInfo())

def export_drive(image, name, scale):
    task = ee.batch.Export.image.toDrive(
        image=image, description=name, folder=os.path.basename(DRIVE_DIR),
        fileNamePrefix=name, region=AOI, scale=scale, crs=UTM,
        maxPixels=int(1e13), fileFormat='GeoTIFF')
    task.start()
    print(f"[{name}] batch export started (scale {scale} m)…", end='', flush=True)
    while task.active():
        time.sleep(20); print(".", end='', flush=True)
    print(" ->", task.status().get('state'))

def load_raster(name, retries=24, delay=5):
    pattern = os.path.join(OUT, name + '*.tif')
    files = sorted(glob.glob(pattern))
    for _ in range(retries):
        if files:
            break
        time.sleep(delay)
        try:
            from google.colab import drive
            drive.flush_and_unmount()
            drive.mount('/content/drive')
        except Exception:
            pass
        files = sorted(glob.glob(pattern))
    assert files, (f"No file found for {name} in {OUT} after {retries*delay}s. "
                   f"Check the export finished AND that OUT points to the mounted folder.")
    with rasterio.open(files[0]) as s:
        return s.read(), s.profile

def get_layer(image, name, scale):
    if USE_DRIVE:
        export_drive(image, name, scale)
    else:
        geemap.download_ee_image(image, os.path.join(OUT, name + '.tif'),
                                 region=AOI, scale=scale, crs=UTM)
    return load_raster(name)

_ = get_layer(feat_stack,   'S2_FEATURES_10M', S2_SCALE)
_ = get_layer(labels,       'LABELS_10M',      S2_SCALE)
_ = get_layer(lst_baseline, 'LST_30M',         LST_SCALE)
del _; gc.collect()
print("Exports finished.")

feat_arr,  feat_prof  = load_raster('S2_FEATURES_10M')
label_arr, label_prof = load_raster('LABELS_10M')
lst30_arr, lst30_prof = load_raster('LST_30M')

BANDS = feat_arr.shape[0]
H, W  = feat_arr.shape[1], feat_arr.shape[2]

X = np.moveaxis(feat_arr, 0, -1).astype('float32')
del feat_arr; gc.collect()
Y = label_arr[0].astype('int32')
del label_arr; gc.collect()

valid = np.isfinite(X).all(-1) & np.isin(Y, np.arange(NUM_CLASSES))
np.nan_to_num(X, copy=False)

mean = X[valid].mean(0); std = X[valid].std(0) + 1e-6
X -= mean; X /= std
Xn = X
np.save(os.path.join(OUT, 'feat_mean.npy'), mean)
np.save(os.path.join(OUT, 'feat_std.npy'),  std)

PATCH, STRIDE = 256, 192
def starts(n, p, s):
    xs = list(range(0, max(1, n - p + 1), s))
    if n > p and xs[-1] != n - p:
        xs.append(n - p)
    return xs

def make_patches(img, msk, lab):
    xs, ys, ws = [], [], []
    for r in starts(img.shape[0], PATCH, STRIDE):
        for c in starts(img.shape[1], PATCH, STRIDE):
            mm = msk[r:r+PATCH, c:c+PATCH]
            if mm.shape != (PATCH, PATCH) or mm.mean() < 0.25:
                continue
            xs.append(img[r:r+PATCH, c:c+PATCH, :])
            ys.append(lab[r:r+PATCH, c:c+PATCH])
            ws.append(mm)
    return (np.asarray(xs, 'float16'),
            np.asarray(ys, 'uint8'),
            np.asarray(ws, 'float16'))

split = int(W * 0.75)
Xtr, Ytr, Wtr = make_patches(Xn[:, :split], valid[:, :split], Y[:, :split])
Xva, Yva, Wva = make_patches(Xn[:, split:], valid[:, split:], Y[:, split:])
print("Train patches:", Xtr.shape, " Val patches:", Xva.shape)

uniq, cnt = np.unique(Y[valid], return_counts=True)
print("Class balance:", {CLASS_NAMES[u]: int(c) for u, c in zip(uniq, cnt)})
del uniq, cnt; gc.collect()

import tensorflow as tf
from tensorflow.keras import layers, Model

for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass
print("GPU:", tf.config.list_physical_devices('GPU'))

USE_MIXED_PRECISION = bool(tf.config.list_physical_devices('GPU'))
if USE_MIXED_PRECISION:
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')
    print("Mixed precision enabled (float16 compute, float32 softmax).")

def conv_block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    return x

def build_unet(bands, nclass, base=32):
    inp = layers.Input((None, None, bands))
    c1 = conv_block(inp, base);      p1 = layers.MaxPool2D()(c1)
    c2 = conv_block(p1, base*2);     p2 = layers.MaxPool2D()(c2)
    c3 = conv_block(p2, base*4);     p3 = layers.MaxPool2D()(c3)
    c4 = conv_block(p3, base*8);     p4 = layers.MaxPool2D()(c4)
    bn = conv_block(p4, base*16)
    u4 = layers.Conv2DTranspose(base*8, 2, strides=2, padding='same')(bn)
    c5 = conv_block(layers.concatenate([u4, c4]), base*8)
    u3 = layers.Conv2DTranspose(base*4, 2, strides=2, padding='same')(c5)
    c6 = conv_block(layers.concatenate([u3, c3]), base*4)
    u2 = layers.Conv2DTranspose(base*2, 2, strides=2, padding='same')(c6)
    c7 = conv_block(layers.concatenate([u2, c2]), base*2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2, padding='same')(c7)
    c8 = conv_block(layers.concatenate([u1, c1]), base)
    out = layers.Conv2D(nclass, 1, activation='softmax', dtype='float32')(c8)
    return Model(inp, out)

model = build_unet(BANDS, NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Params:", f"{model.count_params():,}")

cbs = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
       tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor='val_loss')]
hist = model.fit(Xtr, Ytr, sample_weight=Wtr,
                 validation_data=(Xva, Yva, Wva),
                 epochs=60, batch_size=4, callbacks=cbs, verbose=1)
model.save(os.path.join(OUT, 'unet_dubai.keras'))
print("Model saved to", os.path.join(OUT, 'unet_dubai.keras'))
del Xtr, Ytr, Wtr; gc.collect()

from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt

pv   = model.predict(Xva, batch_size=4).argmax(-1)
mask = Wva.astype(bool)
yt, yp = Yva[mask], pv[mask]
iou = jaccard_score(yt, yp, average=None, labels=range(NUM_CLASSES))
f1  = f1_score(yt, yp, average=None, labels=range(NUM_CLASSES))

print(f"{'Class':18s}{'IoU':>8s}{'F1':>8s}")
for i, n in enumerate(CLASS_NAMES):
    print(f"{n:18s}{iou[i]:8.3f}{f1[i]:8.3f}")
print("-" * 34)
print(f"mean IoU = {iou.mean():.3f}   macro-F1 = {f1_score(yt,yp,average='macro'):.3f}   "
      f"OA = {(yt==yp).mean():.3f}")

cm = confusion_matrix(yt, yp, labels=range(NUM_CLASSES), normalize='true')
plt.figure(figsize=(5, 4)); plt.imshow(cm, cmap='Blues', vmin=0, vmax=1)
plt.xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha='right')
plt.yticks(range(NUM_CLASSES), CLASS_NAMES)
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        plt.text(j, i, f"{cm[i,j]:.2f}", ha='center', va='center')
plt.title('Normalised confusion matrix'); plt.colorbar(); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_matrix.png'), dpi=120); plt.show()
del pv, mask, yt, yp, Xva, Yva, Wva; gc.collect()

prob = np.zeros((H, W, NUM_CLASSES), 'float32'); cnt = np.zeros((H, W), 'float32')
for r in starts(H, PATCH, STRIDE):
    for c in starts(W, PATCH, STRIDE):
        p = model.predict(Xn[r:r+PATCH, c:c+PATCH, :][None], verbose=0)[0]
        prob[r:r+PATCH, c:c+PATCH] += p
        cnt[r:r+PATCH,  c:c+PATCH] += 1
prob /= np.maximum(cnt[..., None], 1)
landcover = prob.argmax(-1).astype('uint8'); landcover[~valid] = 255

prof_lc = feat_prof.copy(); prof_lc.update(count=1, dtype='uint8', nodata=255)
with rasterio.open(os.path.join(OUT, 'LANDCOVER_PRED_10M.tif'), 'w', **prof_lc) as d:
    d.write(landcover, 1)

frac = {CLASS_NAMES[i]: float((landcover == i).sum()) / valid.sum() for i in range(NUM_CLASSES)}
print("Saved LANDCOVER_PRED_10M.tif  | class fractions:",
      {k: round(v, 3) for k, v in frac.items()})
del prob, cnt; gc.collect()

from rasterio.warp import reproject, Resampling
from sklearn.ensemble import RandomForestRegressor

def bidx(name): return FEAT_BANDS.index(name)
print("H,W =", H, W, " total 10 m px =", f"{H*W:,}", " NUM_CLASSES =", NUM_CLASSES)

P10 = np.stack([X[..., bidx('NDVI')], X[..., bidx('NDBI')],
                X[..., bidx('NDWI')], X[..., bidx('elevation')]], -1).astype('float32')
onehot = np.eye(NUM_CLASSES, dtype='float32')[np.clip(landcover, 0, NUM_CLASSES-1)]
onehot[landcover == 255] = 0
P10 = np.concatenate([P10, onehot], -1).astype('float32')
del onehot; gc.collect()
del X
N_FEAT = P10.shape[-1]
print("Predictor stack:", P10.shape, " features =", N_FEAT)

lst30 = lst30_arr[0].astype('float32')
agg_pred = np.zeros((N_FEAT, lst30.shape[0], lst30.shape[1]), 'float32')
for b in range(N_FEAT):
    reproject(P10[..., b], agg_pred[b],
              src_transform=feat_prof['transform'], src_crs=feat_prof['crs'],
              dst_transform=lst30_prof['transform'], dst_crs=lst30_prof['crs'],
              resampling=Resampling.average)
Ptr = np.moveaxis(agg_pred, 0, -1).reshape(-1, N_FEAT)
ytr = lst30.reshape(-1)
ok  = np.isfinite(ytr) & np.isfinite(Ptr).all(-1) & (ytr > 0)
del agg_pred; gc.collect()
print("30 m train grid =", lst30.shape, " valid train rows =", f"{int(ok.sum()):,}")

Xok, yok = Ptr[ok], ytr[ok]
print(f"Training on ALL {Xok.shape} valid 30 m rows")

rf = RandomForestRegressor(
    n_estimators=200, max_depth=20, max_samples=0.3, max_features=0.5,
    min_samples_leaf=5, n_jobs=-1, random_state=0, verbose=1)
t = time.time()
rf.fit(Xok, yok)
print(f"fit done in {time.time()-t:.1f}s  R^2 (30 m fit): {rf.score(Xok, yok):.3f}")
del Ptr, ytr, Xok, yok; gc.collect()

rf.set_params(n_jobs=-1)
flat = P10.reshape(-1, N_FEAT)
okf  = np.isfinite(flat).all(-1)
lst10 = np.full(flat.shape[0], np.nan, 'float32')
idx = np.where(okf)[0]
CHUNK = 500_000
t = time.time()
for i in range(0, idx.size, CHUNK):
    sl = idx[i:i+CHUNK]
    lst10[sl] = rf.predict(flat[sl]).astype('float32')
    print(f"  predicted {min(i+CHUNK, idx.size):,}/{idx.size:,} "
          f"({time.time()-t:.0f}s elapsed)", flush=True)

lst10 = lst10.reshape(H, W)
lst10[~valid] = np.nan
del flat, idx; gc.collect()

back = np.zeros_like(lst30)
reproject(lst10, back, src_transform=feat_prof['transform'], src_crs=feat_prof['crs'],
          dst_transform=lst30_prof['transform'], dst_crs=lst30_prof['crs'],
          resampling=Resampling.average)
mm = np.isfinite(back) & np.isfinite(lst30) & (lst30 > 0)
rmse = float(np.sqrt(np.mean((back[mm] - lst30[mm])**2)))
mae  = float(np.mean(np.abs(back[mm] - lst30[mm])))
bias = float(np.mean(back[mm] - lst30[mm]))
print(f"\nRMSE vs 30 m baseline = {rmse:.2f} °C  (target <= 2.7 °C)   "
      f"MAE = {mae:.2f} °C  bias = {bias:+.2f} °C")

prof1 = feat_prof.copy(); prof1.update(count=1, dtype='float32', nodata=float('nan'))
with rasterio.open(os.path.join(OUT, 'LST_BASELINE_10M.tif'), 'w', **prof1) as d:
    d.write(lst10, 1)
print("Saved LST_BASELINE_10M.tif")
del back, mm; gc.collect()

COOL = {
    'green_roof':           1.45,
    'green_roof_hotarid':   1.83,
    'cool_roof_albedo':     2.00,
    'high_albedo_pavement': 2.50,
    'veg_buffer':           1.00,
}

def run_scenario(lst, lc, interventions):
    out  = lst.copy().astype('float32')
    done = np.zeros_like(lc, bool)
    for iv in interventions:
        coef, frac, tgt = COOL[iv['name']], iv['fraction'], iv['target_class']
        cand = (lc == tgt) & np.isfinite(out) & (~done)
        idx  = np.where(cand.ravel())[0]
        if idx.size == 0 or frac <= 0:
            continue
        order = idx[np.argsort(-out.ravel()[idx])][:int(frac * idx.size)]
        rr, cc = np.unravel_index(order, lc.shape)
        out[rr, cc] -= coef
        done[rr, cc] = True
    return out

interventions = [
    {'name': 'green_roof_hotarid',   'fraction': 0.20, 'target_class': 1},
    {'name': 'high_albedo_pavement', 'fraction': 0.30, 'target_class': 1},
]

lst_scn = run_scenario(lst10, landcover, interventions)
delta   = lst10 - lst_scn

with rasterio.open(os.path.join(OUT, 'LST_SCENARIO_10M.tif'), 'w', **prof1) as d:
    d.write(lst_scn, 1)
with rasterio.open(os.path.join(OUT, 'LST_DELTA_10M.tif'), 'w', **prof1) as d:
    d.write(np.nan_to_num(delta), 1)

v       = np.isfinite(delta)
built   = (landcover == 1) & v
treated = built & (delta > 0)
print(f"City mean LST baseline      : {np.nanmean(lst10):.2f} °C")
print(f"City mean LST scenario      : {np.nanmean(lst_scn):.2f} °C")
print(f"Mean cooling (whole AOI)    : {np.nanmean(delta[v]):.3f} °C")
print(f"Mean cooling (built-up)     : {np.nanmean(delta[built]):.3f} °C")
print(f"Mean cooling (treated only) : {np.nanmean(delta[treated]):.3f} °C")
print(f"Built-up pixels treated     : {treated.sum():,} / {built.sum():,} "
      f"({100*treated.sum()/max(1,built.sum()):.1f}%)")
print(f"Max local cooling           : {np.nanmax(delta[v]):.2f} °C")
print("Saved LST_SCENARIO_10M.tif and LST_DELTA_10M.tif")
del lst_scn; gc.collect()

fr = np.linspace(0, 1, 11)
plt.figure(figsize=(7, 4))
for iv in interventions:
    curve = [np.nanmean(lst10 - run_scenario(
                lst10, landcover,
                [{'name': iv['name'], 'fraction': f, 'target_class': iv['target_class']}]))
             for f in fr]
    plt.plot(fr * 100, curve, 'o-',
             label=f"{iv['name']} → class {iv['target_class']}")
plt.xlabel('% of target surface treated')
plt.ylabel('City-wide mean cooling (°C)')
plt.title('Sensitivity of city cooling to intervention intensity')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sensitivity.png'), dpi=120); plt.show()

def fix_nan_nodata(path, nodata=-9999.0):
    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()
    data = np.where(np.isnan(data), nodata, data).astype('float32')
    profile.update(dtype='float32', nodata=nodata, compress='deflate')
    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data, 1)
    with rasterio.open(path) as chk:
        print(f"{os.path.basename(path):26s} nodata={nodata} hasNaN={np.isnan(chk.read(1)).any()}")

for f in ['LST_BASELINE_10M.tif', 'LST_SCENARIO_10M.tif', 'LST_DELTA_10M.tif']:
    fix_nan_nodata(os.path.join(OUT, f))

import shutil
LOCAL = '/content/UHI_Dubai'; os.makedirs(LOCAL, exist_ok=True)
for f in os.listdir(OUT):
    if f.endswith('.tif'):
        shutil.copy(os.path.join(OUT, f), os.path.join(LOCAL, f))
print("copied to", LOCAL)

import leafmap
from ipyleaflet import ImageOverlay
from PIL import Image
from matplotlib.colors import ListedColormap
from rasterio.warp import transform_bounds, calculate_default_transform

SRC = LOCAL
LC_CMAP = ListedColormap(CLASS_COLORS)

def raster_to_overlay(path, cmap, vmin=None, vmax=None, nodata=None,
                      discrete=False, max_px=2000):
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

specs = [
    dict(f="LANDCOVER_PRED_10M.tif", name="U-Net land cover",
         cmap=None, discrete=True, nodata=255),
    dict(f="LST_BASELINE_10M.tif",   name="LST baseline (°C)",
         cmap="inferno", vmin=30, vmax=58, nodata=-9999.0),
    dict(f="LST_SCENARIO_10M.tif",   name="LST scenario (°C)",
         cmap="inferno", vmin=30, vmax=58, nodata=-9999.0),
    dict(f="LST_DELTA_10M.tif",      name="Cooling Δ (°C)",
         cmap="Blues", vmin=0, vmax=2.5, nodata=-9999.0),
]

m = leafmap.Map(center=[25.10, 55.30], zoom=10)
m.add_basemap("HYBRID")

first = True
for s in specs:
    path = os.path.join(SRC, s["f"])
    if not os.path.exists(path):
        print(f"missing {s['f']}"); continue
    url, bounds = raster_to_overlay(
        path, s["cmap"], s.get("vmin"), s.get("vmax"),
        s.get("nodata"), s.get("discrete", False))
    ov = ImageOverlay(url=url, bounds=bounds, name=s["name"])
    m.add_layer(ov)
    if first:
        m.fit_bounds(bounds); first = False
    print(f"added {s['name']}  bounds={bounds}")

legend = {n: c for n, c in zip(CLASS_NAMES, CLASS_COLORS)}
m.add_legend(title="Land cover", legend_dict=legend)
try:
    m.add_colorbar(cmap="inferno", vmin=30, vmax=58, label="LST (°C)")
except TypeError:
    cols = [plt.get_cmap("inferno")(i/9) for i in range(10)]
    m.add_colorbar(colors=cols, vmin=30, vmax=58, label="LST (°C)")
m.add_layer_control()
m
