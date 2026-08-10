!pip -q install "earthengine-api>=1.4.0" geemap leafmap rasterio rioxarray localtileserver scikit-learn tensorflow tqdm matplotlib psutil osmnx shapely

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
AOI_LONLAT = (55.10, 24.80, 55.55, 25.40)
UTM  = "EPSG:32640"
YEARS = list(range(2023, 2026))
SUMMER_MONTHS = [6, 7, 8]
S2_SCALE, LST_SCALE = 10, 30

NUM_CLASSES  = 5
CLASS_NAMES  = ["Vegetation", "Building/Roof", "Road/Pavement",
                "Bare soil/Sand", "Water"]
CLASS_COLORS = ["#1a9850", "#d73027", "#4d4d4d", "#fee08b", "#4575b4"]

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
                  [0,  0,  0,  0,  2,  3,  3,  4,  0,  0,   3]).rename('class').clip(AOI)

FEAT_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'NDVI', 'NDBI', 'NDWI', 'elevation']
feat_stack = s2_features.addBands(dem).select(FEAT_BANDS).toFloat()
print("Composites built. Feature bands:", feat_stack.bandNames().getInfo())

def export_drive(image, name, scale):
    task = ee.batch.Export.image.toDrive(
        image=image, description=name, folder=os.path.basename(DRIVE_DIR),
        fileNamePrefix=name, region=AOI, scale=scale, crs=UTM,
        maxPixels=int(1e13), fileFormat='GeoTIFF')
    task.start()
    print(f"[{name}] batch export started ({scale} m)...", end='', flush=True)
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

import osmnx as ox
import geopandas as gpd
from shapely.geometry import box
from rasterio.features import rasterize

ROAD_HALF_WIDTH = {
    'motorway': 12, 'motorway_link': 8, 'trunk': 10, 'trunk_link': 7,
    'primary': 9, 'primary_link': 6, 'secondary': 7, 'secondary_link': 5,
    'tertiary': 6, 'residential': 4, 'living_street': 3, 'service': 3,
    'unclassified': 4,
}
DEFAULT_HALF_WIDTH = 4


def _first(x):
    return x[0] if isinstance(x, list) else x


def build_osm_masks(prof, aoi_lonlat):
    west, south, east, north = aoi_lonlat
    poly  = box(west, south, east, north)
    H, W  = prof['height'], prof['width']
    tform = prof['transform']
    crs   = prof['crs']

    tags_b = {'building': True}
    gdf_b = ox.features_from_polygon(poly, tags_b)
    gdf_b = gdf_b[gdf_b.geometry.type.isin(['Polygon', 'MultiPolygon'])].to_crs(crs)
    building_mask = rasterize(
        ((g, 1) for g in gdf_b.geometry),
        out_shape=(H, W), transform=tform, fill=0, dtype='uint8').astype(bool)

    tags_r = {'highway': True}
    gdf_r = ox.features_from_polygon(poly, tags_r)
    gdf_r = gdf_r[gdf_r.geometry.type.isin(['LineString', 'MultiLineString'])].to_crs(crs)
    hw = gdf_r['highway'].map(lambda h: ROAD_HALF_WIDTH.get(_first(h), DEFAULT_HALF_WIDTH))
    gdf_r = gdf_r.assign(buf=gdf_r.geometry.buffer(hw.values))
    road_mask = rasterize(
        ((g, 1) for g in gdf_r['buf']),
        out_shape=(H, W), transform=tform, fill=0, dtype='uint8').astype(bool)

    return building_mask, road_mask


print("Refining labels with OpenStreetMap (buildings vs roads)...")
lab = label_arr[0].astype('int16').copy()

wc_builtup = (lab == 2)

building_mask, road_mask = build_osm_masks(label_prof, AOI_LONLAT)

lab[road_mask]     = 2
lab[building_mask] = 1
leftover_builtup = wc_builtup & ~road_mask & ~building_mask
lab[leftover_builtup] = 1
print(f"Reclassified {int(leftover_builtup.sum()):,} leftover built-up pixels "
      f"from Road -> Building/Roof (FIX 1).")

label_arr[0] = lab
uniq, cnt = np.unique(lab, return_counts=True)
print("Refined label balance:",
      {CLASS_NAMES[u]: int(c) for u, c in zip(uniq, cnt) if u < NUM_CLASSES})

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


uniq, cnt = np.unique(Y[valid], return_counts=True)
freq = np.zeros(NUM_CLASSES, 'float64')
freq[uniq] = cnt / cnt.sum()
CLS_W = np.where(freq > 0, 1.0 / (freq + 1e-6), 0.0)
CLS_W = (CLS_W / CLS_W[freq > 0].mean()).astype('float32')
print("Class weights (FIX 2):", {CLASS_NAMES[i]: round(float(CLS_W[i]), 2)
                                  for i in range(NUM_CLASSES)})

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
            lb = lab[r:r+PATCH, c:c+PATCH]
            w  = mm.astype('float32') * CLS_W[np.clip(lb, 0, NUM_CLASSES-1)]
            xs.append(img[r:r+PATCH, c:c+PATCH, :])
            ys.append(lb)
            ws.append(w)
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



def dice_ce_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    ce = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    ce = tf.reduce_mean(ce)

    yt = tf.one_hot(y_true, NUM_CLASSES)
    yt = tf.cast(yt, y_pred.dtype)
    inter = tf.reduce_sum(yt * y_pred, axis=[1, 2])
    union = tf.reduce_sum(yt + y_pred, axis=[1, 2])
    dice  = 1.0 - tf.reduce_mean((2.0 * inter + 1e-6) / (union + 1e-6))
    return ce + dice


model = build_unet(BANDS, NUM_CLASSES)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=dice_ce_loss, metrics=['accuracy'])
print("Params:", f"{model.count_params():,}")

cbs = [tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor='val_loss'),
       tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, monitor='val_loss')]
hist = model.fit(Xtr, Ytr, sample_weight=Wtr,
                 validation_data=(Xva, Yva, Wva),
                 epochs=60, batch_size=4, callbacks=cbs, verbose=1)
model.save(os.path.join(OUT, 'unet_dubai.keras'))
print("Model saved to", os.path.join(OUT, 'unet_dubai.keras'))
del Xtr, Ytr, Wtr; gc.collect()

from sklearn.metrics import confusion_matrix, f1_score

y_true_all, y_pred_all = [], []
EVAL_BATCH = 4
for i in range(0, len(Xva), EVAL_BATCH):
    xb   = Xva[i:i+EVAL_BATCH].astype('float32')
    pred = model.predict(xb, verbose=0).argmax(-1).astype('uint8')
    yb   = Yva[i:i+EVAL_BATCH].astype('uint8')
    wb   = Wva[i:i+EVAL_BATCH].astype('float32') > 0
    y_true_all.append(yb[wb])
    y_pred_all.append(pred[wb])

y_true = np.concatenate(y_true_all)
y_pred = np.concatenate(y_pred_all)
del y_true_all, y_pred_all; gc.collect()

_labels = np.arange(NUM_CLASSES)
cm      = confusion_matrix(y_true, y_pred, labels=_labels)
inter = np.diag(cm).astype('float64')
union = cm.sum(0) + cm.sum(1) - inter
iou   = np.where(union > 0, inter / union, np.nan)
f1_per = f1_score(y_true, y_pred, labels=_labels, average=None, zero_division=0)

print("\n===== U-Net validation metrics (patch-level) =====")
print(f"{'Class':<16}{'IoU':>8}{'F1':>8}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<16}{iou[i]:>8.3f}{f1_per[i]:>8.3f}")
print("-" * 32)
print(f"{'mIoU':<16}{np.nanmean(iou):>8.3f}")
print(f"{'Macro-F1':<16}{f1_per.mean():>8.3f}")
print(f"{'Pixel accuracy':<16}{np.trace(cm)/cm.sum():>8.3f}")

_target = 0.57
print(f"\nBenchmark (proposal §2.4.3): mIoU > 57%  ->  "
      f"{'PASS' if np.nanmean(iou) > _target else 'BELOW TARGET'}")
del y_true, y_pred, cm; gc.collect()

prob = np.zeros((H, W, NUM_CLASSES), 'float32'); cnt = np.zeros((H, W), 'float32')
for r in starts(H, PATCH, STRIDE):
    for c in starts(W, PATCH, STRIDE):
        p = model.predict(Xn[r:r+PATCH, c:c+PATCH, :][None], verbose=0)[0]
        prob[r:r+PATCH, c:c+PATCH] += p
        cnt[r:r+PATCH,  c:c+PATCH] += 1
prob /= np.maximum(cnt[..., None], 1)
landcover = prob.argmax(-1).astype('uint8'); landcover[~valid] = 255

if 'building_mask' not in globals() or 'road_mask' not in globals():
    print("Rebuilding OSM masks for the override...")
    building_mask, road_mask = build_osm_masks(feat_prof, AOI_LONLAT)

landcover[road_mask     & valid] = 2
landcover[building_mask & valid] = 1

print("OSM authority override applied.")
_ufr = {CLASS_NAMES[i]: round(float((landcover == i).sum()) / max(1, valid.sum()), 3)
        for i in range(NUM_CLASSES)}
print("Post-override class fractions:", _ufr)

import matplotlib.pyplot as plt

val_cols = np.zeros(W, bool); val_cols[split:] = True
val_mask = valid & val_cols[None, :] & (landcover != 255)

yt_dep = Y[val_mask].astype('uint8')
yp_dep = landcover[val_mask].astype('uint8')

cm_dep   = confusion_matrix(yt_dep, yp_dep, labels=_labels)
inter_d  = np.diag(cm_dep).astype('float64')
union_d  = cm_dep.sum(0) + cm_dep.sum(1) - inter_d
iou_dep  = np.where(union_d > 0, inter_d / union_d, np.nan)
f1_dep   = f1_score(yt_dep, yp_dep, labels=_labels, average=None, zero_division=0)

print("\n===== As-deployed metrics (post-OSM override, validation cols) =====")
print(f"{'Class':<16}{'IoU':>8}{'F1':>8}")
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<16}{iou_dep[i]:>8.3f}{f1_dep[i]:>8.3f}")
print("-" * 32)
print(f"{'mIoU':<16}{np.nanmean(iou_dep):>8.3f}")
print(f"{'Macro-F1':<16}{f1_dep.mean():>8.3f}")
print(f"{'Pixel accuracy':<16}{np.trace(cm_dep)/cm_dep.sum():>8.3f}")

cm_norm = cm_dep.astype('float64') / cm_dep.sum(axis=1, keepdims=True).clip(min=1)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel('Predicted class'); ax.set_ylabel('True class')
ax.set_title('As-deployed Confusion Matrix (post-OSM override, row-normalised)')
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm_dep[i, j]:,})",
                ha='center', va='center', fontsize=8,
                color='white' if cm_norm[i, j] > 0.5 else 'black')
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label('Proportion of true-class pixels')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'confusion_matrix_deployed.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved confusion_matrix_deployed.png to", OUT)
del yt_dep, yp_dep, cm_dep, cm_norm; gc.collect()

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
print(f"Training on ALL {Xok.shape[0]:,} valid 30 m rows")

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

!pip -q install osmnx shapely

import osmnx as ox
from shapely.geometry import box
from rasterio.features import rasterize

ROAD_HALF_WIDTH = {
    'motorway': 12, 'motorway_link': 8, 'trunk': 10, 'trunk_link': 7,
    'primary': 9, 'primary_link': 6, 'secondary': 7, 'secondary_link': 5,
    'tertiary': 6, 'residential': 4, 'living_street': 3, 'service': 3,
    'unclassified': 4}
DEFAULT_HALF_WIDTH = 4
_first = lambda x: x[0] if isinstance(x, list) else x


def load1(name):
    f = sorted(glob.glob(os.path.join(OUT, name + '*.tif')))
    assert f, f"missing {name} in {OUT}"
    with rasterio.open(f[0]) as s:
        return s.read(1), s.profile, f[0]


def build_osm_masks(prof):
    poly  = box(*AOI_LONLAT)
    H, W  = prof['height'], prof['width']
    tform, crs = prof['transform'], prof['crs']

    tags_b = {'building': True}
    gdf_b = ox.features_from_polygon(poly, tags_b)
    gdf_b = gdf_b[gdf_b.geometry.type.isin(['Polygon', 'MultiPolygon'])].to_crs(crs)
    building_mask = rasterize(
        ((g, 1) for g in gdf_b.geometry),
        out_shape=(H, W), transform=tform, fill=0, dtype='uint8').astype(bool)

    tags_r = {'highway': True}
    gdf_r = ox.features_from_polygon(poly, tags_r)
    gdf_r = gdf_r[gdf_r.geometry.type.isin(['LineString', 'MultiLineString'])].to_crs(crs)
    hw = gdf_r['highway'].map(lambda h: ROAD_HALF_WIDTH.get(_first(h), DEFAULT_HALF_WIDTH))
    gdf_r = gdf_r.assign(buf=gdf_r.geometry.buffer(hw.values))
    road_mask = rasterize(
        ((g, 1) for g in gdf_r['buf']),
        out_shape=(H, W), transform=tform, fill=0, dtype='uint8').astype(bool)
    return building_mask, road_mask


pred, pred_prof, pred_path = load1('LANDCOVER_PRED_10M')
pred = pred.astype('int16')
valid = (pred != 255)
landcover = pred

lst_arr, lst_prof, lst_path = load1('LST_BASELINE_10M')
lst10 = lst_arr.astype('float32')
lst10 = np.where(lst10 == -9999.0, np.nan, lst10)

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
    {'name': 'high_albedo_pavement', 'fraction': 0.30, 'target_class': 2},
]

lst_scn = run_scenario(lst10, landcover, interventions)
delta   = lst10 - lst_scn

prof1 = lst_prof.copy(); prof1.update(count=1, dtype='float32', nodata=-9999.0,
                                      compress='deflate')


def save_nodata(path, arr):
    a = np.where(np.isnan(arr), -9999.0, arr).astype('float32')
    with rasterio.open(path, 'w', **prof1) as d:
        d.write(a, 1)


save_nodata(os.path.join(OUT, 'LST_SCENARIO_10M.tif'), lst_scn)
save_nodata(os.path.join(OUT, 'LST_DELTA_10M.tif'),    delta)

v       = np.isfinite(delta)
roofs   = (landcover == 1) & v
roads   = (landcover == 2) & v
treated = (roofs | roads) & (delta > 0)
print(f"\nCity mean LST baseline      : {np.nanmean(lst10):.2f} °C")
print(f"City mean LST scenario      : {np.nanmean(lst_scn):.2f} °C")
print(f"Mean cooling (whole AOI)    : {np.nanmean(delta[v]):.3f} °C")
print(f"Mean cooling (roofs)        : {np.nanmean(delta[roofs]):.3f} °C")
print(f"Mean cooling (roads)        : {np.nanmean(delta[roads]):.3f} °C")
print(f"Mean cooling (treated only) : {np.nanmean(delta[treated]):.3f} °C")
print(f"Treated pixels              : {treated.sum():,} / {(roofs|roads).sum():,} "
      f"({100*treated.sum()/max(1,(roofs|roads).sum()):.1f}%)")
print(f"Max local cooling           : {np.nanmax(delta[v]):.2f} °C")
print("\nSaved LST_SCENARIO_10M.tif and LST_DELTA_10M.tif")

import matplotlib.pyplot as plt

fr = np.linspace(0, 1, 11)
plt.figure(figsize=(7, 4))
for iv in interventions:
    curve = [np.nanmean(lst10 - run_scenario(
        lst10, landcover, [
            {'name': iv['name'], 'fraction': f, 'target_class': iv['target_class']}]))
        for f in fr]
    plt.plot(fr * 100, curve, 'o-',
             label=f"{iv['name']} → {CLASS_NAMES[iv['target_class']]}")
plt.xlabel('% of target surface treated')
plt.ylabel('City-wide mean cooling (°C)')
plt.title('Sensitivity of city cooling to intervention intensity')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'sensitivity.png'), dpi=120); plt.show()
