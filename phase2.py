
!pip install rasterio scikit-learn tensorflow -q

import os, gc
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))

TIF_PATH    = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
MODEL_OUT   = "/content/drive/MyDrive/thesis_phase2_exports/unet_dubai.keras"
PREDMAP_OUT = "/content/drive/MyDrive/thesis_phase2_exports/dubai_prediction_map.tif"

PATCH_SIZE, STRIDE = 256, 128
GUTTER       = PATCH_SIZE
NUM_CLASSES  = 4
BATCH_SIZE   = 8
EPOCHS       = 60
LR           = 1e-3
VAL_FRACTION, TEST_FRACTION = 0.15, 0.15

CLASS_NAMES = ["Bare/Sand", "Vegetation", "Built-up", "Water"]

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12",
              "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL", "LABEL"]
INPUT_BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12",
                    "NDVI", "NDBI", "NDWI", "DEM", "GHSL"]
input_idx = [BAND_ORDER.index(b) for b in INPUT_BAND_NAMES]
label_idx = BAND_ORDER.index("LABEL")
N_CH = len(INPUT_BAND_NAMES)

with rasterio.open(TIF_PATH) as src:
    stack   = src.read()
    profile = src.profile
    print(f"Loaded {src.count} bands, {src.width}x{src.height}, CRS {src.crs}")

stack = np.transpose(stack, (1, 2, 0)).astype("float32")
assert stack.shape[-1] == len(BAND_ORDER), \
    f"Expected {len(BAND_ORDER)} bands, got {stack.shape[-1]} — check Phase 1."

X_full = stack[:, :, input_idx]
label_raw = stack[:, :, label_idx]

finite_inputs = np.isfinite(X_full).all(axis=-1)
valid = (np.isfinite(label_raw) &
         (label_raw >= 0) & (label_raw < NUM_CLASSES) &
         finite_inputs)

y_full = np.where(valid, np.round(label_raw), 0).astype("int32")
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

H, W = y_full.shape
del stack; gc.collect()
print(f"Scene {H}x{W}. Valid pixels: {valid.sum():,} "
      f"({100*valid.sum()/valid.size:.1f}%).  Classes present: "
      f"{np.unique(y_full[valid])}")


train_end  = int(W * (1 - VAL_FRACTION - TEST_FRACTION))
val_start  = train_end + GUTTER
val_end    = int(W * (1 - TEST_FRACTION))
test_start = val_end + GUTTER

def make_coords(c0, c1):
    """Patch top-left coords whose FULL width fits inside [c0, c1)."""
    coords = []
    for r in range(0, H - PATCH_SIZE + 1, STRIDE):
        for c in range(c0, c1 - PATCH_SIZE + 1, STRIDE):
            if valid[r:r+PATCH_SIZE, c:c+PATCH_SIZE].mean() > 0.10:
                coords.append((r, c))
    return np.array(coords, dtype=np.int32)

train_c = make_coords(0,          train_end)
val_c   = make_coords(val_start,  val_end)
test_c  = make_coords(test_start, W)
print(f"Patches -> Train:{len(train_c)}  Val:{len(val_c)}  Test:{len(test_c)} "
      f"(gutter={GUTTER}px between splits)")
assert min(len(train_c), len(val_c), len(test_c)) > 0, \
    "A split is empty — widen the AOI or reduce STRIDE/GUTTER."

train_region = X_full[:, :train_end, :]
train_valid  = valid[:, :train_end]
mu = np.array([train_region[..., b][train_valid].mean() for b in range(N_CH)])
sd = np.array([train_region[..., b][train_valid].std()  for b in range(N_CH)])
sd[sd < 1e-6] = 1.0
X_full = (X_full - mu) / sd
print("Inputs normalised using TRAINING-region statistics only.")

counts = np.array([(y_full[valid] == k).sum() for k in range(NUM_CLASSES)],
                  dtype="float64")
counts[counts == 0] = 1.0
class_w = counts.sum() / (NUM_CLASSES * counts)
class_w = class_w / class_w.mean()
print("Class pixel counts:", counts.astype(int).tolist())
print("Class weights     :", np.round(class_w, 3).tolist())
CLASS_W_TF = tf.constant(class_w, dtype=tf.float32)

X_TF = tf.convert_to_tensor(X_full, tf.float32)
y_TF = tf.convert_to_tensor(y_full, tf.int32)
V_TF = tf.convert_to_tensor(valid.astype("float32"), tf.float32)

def load_patch(coord):
    r, c = coord[0], coord[1]
    x = tf.slice(X_TF, [r, c, 0], [PATCH_SIZE, PATCH_SIZE, N_CH])
    y = tf.slice(y_TF, [r, c],    [PATCH_SIZE, PATCH_SIZE])
    v = tf.slice(V_TF, [r, c],    [PATCH_SIZE, PATCH_SIZE])
    y1 = tf.one_hot(y, NUM_CLASSES) * v[..., None]
    w = tf.reduce_sum(y1 * CLASS_W_TF, axis=-1) * v
    packed = tf.concat([y1, w[..., None]], axis=-1)
    return x, packed

def augment(x, packed):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x); packed = tf.image.flip_left_right(packed)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x);    packed = tf.image.flip_up_down(packed)
    k = tf.random.uniform((), 0, 4, tf.int32)
    x = tf.image.rot90(x, k); packed = tf.image.rot90(packed, k)
    return x, packed

def build_ds(coords, training):
    ds = tf.data.Dataset.from_tensor_slices(coords)
    if training:
        ds = ds.shuffle(min(512, len(coords)), reshuffle_each_iteration=True)
    ds = ds.map(load_patch, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE).repeat()
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(train_c, True)
val_ds   = build_ds(val_c,   False)
test_ds  = build_ds(test_c,  False)
steps_per_epoch  = int(np.ceil(len(train_c) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_c)   / BATCH_SIZE))

def conv_block(x, f):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
    return x

def build_unet(ch, n_cls, base=32):
    inp = layers.Input((PATCH_SIZE, PATCH_SIZE, ch))
    c1 = conv_block(inp, base);   p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base*2);  p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base*4);  p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base*8);  p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, base*16)
    u4 = layers.Conv2DTranspose(base*8, 2, 2, padding="same")(bn)
    c5 = conv_block(layers.concatenate([u4, c4]), base*8)
    u3 = layers.Conv2DTranspose(base*4, 2, 2, padding="same")(c5)
    c6 = conv_block(layers.concatenate([u3, c3]), base*4)
    u2 = layers.Conv2DTranspose(base*2, 2, 2, padding="same")(c6)
    c7 = conv_block(layers.concatenate([u2, c2]), base*2)
    u1 = layers.Conv2DTranspose(base, 2, 2, padding="same")(c7)
    c8 = conv_block(layers.concatenate([u1, c1]), base)
    out = layers.Conv2D(n_cls, 1, activation="softmax")(c8)
    return models.Model(inp, out, name="UNet")

model = build_unet(N_CH, NUM_CLASSES)
model.summary()

def masked_combined_loss(packed_true, y_pred, smooth=1.0):
    y_true = packed_true[..., :NUM_CLASSES]
    w      = packed_true[..., NUM_CLASSES]
    cce = -tf.reduce_sum(y_true * tf.math.log(y_pred + 1e-7), axis=-1)
    cce = tf.reduce_sum(cce * w) / (tf.reduce_sum(w) + 1e-7)
    m = tf.cast(w > 0, tf.float32)[..., None]
    yt, yp = y_true * m, y_pred * m
    inter = tf.reduce_sum(yt * yp, axis=[0, 1, 2])
    union = tf.reduce_sum(yt, axis=[0, 1, 2]) + tf.reduce_sum(yp, axis=[0, 1, 2])
    dice = 1.0 - tf.reduce_mean((2*inter + smooth) / (union + smooth))
    return cce + dice

model.compile(optimizer=optimizers.Adam(LR), loss=masked_combined_loss)

cbs = [
    callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss", save_best_only=True),
    callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=5, min_lr=1e-6),
    callbacks.LambdaCallback(on_epoch_end=lambda *a: gc.collect()),
]

history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps, callbacks=cbs)

yt_all, yp_all = [], []
for xb, packed in test_ds:
    pred = np.argmax(model(xb, training=False).numpy(), axis=-1)
    true = np.argmax(packed[..., :NUM_CLASSES].numpy(), axis=-1)
    w    = packed[..., NUM_CLASSES].numpy()
    keep = w.reshape(-1) > 0
    yp_all.append(pred.reshape(-1)[keep])
    yt_all.append(true.reshape(-1)[keep])
y_pred_lbl = np.concatenate(yp_all)
y_true_lbl = np.concatenate(yt_all)

iou = jaccard_score(y_true_lbl, y_pred_lbl, average=None,
                    labels=range(NUM_CLASSES), zero_division=0)
miou = iou.mean()
f1   = f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)

print("\n========== EVALUATION (vs. WorldCover reference) ==========")
print("NOTE: WorldCover OA ~76.7% -> this is AGREEMENT, not ground truth.")
for i, n in enumerate(CLASS_NAMES):
    print(f"  IoU {n:12s}: {iou[i]:.3f}")
print(f"  mIoU : {miou:.3f}   |   macro-F1 : {f1:.3f}")
print(f"  {'PASS' if miou > 0.57 else 'below'} vs. 0.57 benchmark "
      f"(Tzepkenlis et al. 2023 — different labels, treat as soft target).")
print("Confusion matrix (rows=WorldCover, cols=U-Net):")
print(confusion_matrix(y_true_lbl, y_pred_lbl, labels=range(NUM_CLASSES)))

def cosine_window(size):
    w = np.hanning(size); w = np.clip(w, 1e-3, None)
    return (w[:, None] * w[None, :]).astype("float32")

def predict_smooth(step=PATCH_SIZE // 2):
    win  = cosine_window(PATCH_SIZE)[..., None]
    prob = np.zeros((H, W, NUM_CLASSES), "float32")
    wsum = np.zeros((H, W, 1), "float32")
    rows = list(range(0, H - PATCH_SIZE + 1, step)) + [H - PATCH_SIZE]
    cols = list(range(0, W - PATCH_SIZE + 1, step)) + [W - PATCH_SIZE]
    for r in sorted(set(rows)):
        for c in sorted(set(cols)):
            patch = X_full[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :][None]
            p = model(patch, training=False).numpy()[0]
            prob[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += p * win
            wsum[r:r+PATCH_SIZE, c:c+PATCH_SIZE] += win
    pred = np.argmax(prob / (wsum + 1e-7), axis=-1).astype("uint8")
    pred[~valid] = 255
    prof = profile.copy(); prof.update(count=1, dtype="uint8", nodata=255)
    with rasterio.open(PREDMAP_OUT, "w", **prof) as dst:
        dst.write(pred, 1)
    print(f"\nSaved seam-free classification map -> {PREDMAP_OUT}")

    plt.figure(figsize=(9, 9))
    plt.imshow(np.where(valid, pred, np.nan), cmap="tab10",
               vmin=0, vmax=NUM_CLASSES-1)
    plt.title("Full-Scene U-Net Classification (smooth overlap-blended)")
    plt.axis("off"); plt.show()
    return pred

predicted_map = predict_smooth()

plt.figure(figsize=(6, 4))
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend(); plt.tight_layout(); plt.show()

print("\nPhase 2 complete. Next: Phase 2b (independent point validation).")
