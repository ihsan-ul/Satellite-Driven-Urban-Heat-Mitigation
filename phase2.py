!pip install rasterio scikit-learn tensorflow -q

import os
import gc
import numpy as np
import rasterio
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix


print("TensorFlow:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))


TIF_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"

PATCH_SIZE = 256
STRIDE     = 128
NUM_CLASSES = 4

INPUT_BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12",
                    "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL"]
LABEL_BAND_NAME  = "LABEL"
ALL_BANDS = INPUT_BAND_NAMES + [LABEL_BAND_NAME]

BATCH_SIZE = 8
EPOCHS     = 60
LR         = 1e-3
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.15

MODEL_OUT   = "/content/unet_dubai.keras"
PREDMAP_OUT = "/content/dubai_prediction_map.tif"

CLASS_NAMES = ["Other/Bare", "Vegetation", "Built-up", "Water"]


with rasterio.open(TIF_PATH) as src:
    stack = src.read()
    profile = src.profile
    print(f"Loaded {src.count} bands, size {src.width}x{src.height}, CRS {src.crs}")

stack = np.transpose(stack, (1, 2, 0))
n_bands = stack.shape[-1]
assert n_bands == len(ALL_BANDS), \
    f"Expected {len(ALL_BANDS)} bands, got {n_bands}. Check Phase-1 band order."

X_full = stack[:, :, :len(INPUT_BAND_NAMES)].astype("float32")
X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

y_raw = stack[:, :, -1].astype("float32")
y_raw = np.nan_to_num(y_raw, nan=0.0, posinf=0.0, neginf=0.0)
y_full = y_raw.astype("int32")

invalid = (y_full < 0) | (y_full >= NUM_CLASSES)
if invalid.any():
    print(f"Cleaned {invalid.sum()} invalid/NoData label pixels -> class 0.")
y_full[invalid] = 0

del stack, y_raw
gc.collect()

print("Input array:", X_full.shape, " Label array:", y_full.shape)
print("Label classes present:", np.unique(y_full))


def normalise_per_band(x):
    for b in range(x.shape[-1]):
        band = x[:, :, b]
        mu, sd = band.mean(), band.std()
        if sd > 1e-6:
            x[:, :, b] = (band - mu) / sd
    return x

X_full = normalise_per_band(X_full)
print("Inputs normalised (per-band mean 0 / std 1).")


def make_coords(H, W, size, stride):
    coords = []
    for r in range(0, H - size + 1, stride):
        for c in range(0, W - size + 1, stride):
            coords.append((r, c))
    return np.array(coords, dtype=np.int32)

H, W = y_full.shape
coords = make_coords(H, W, PATCH_SIZE, STRIDE)
print(f"Extracted {len(coords)} patch coordinates of {PATCH_SIZE}x{PATCH_SIZE}.")

if len(coords) < 4:
    print("\n WARNING: very few patches. Use STRIDE=128 or the full Dubai AOI.")

c_vals = coords[:, 1]
c_max  = c_vals.max() if len(c_vals) else 1
train_thr = c_max * (1 - VAL_FRACTION - TEST_FRACTION)
val_thr   = c_max * (1 - TEST_FRACTION)

train_mask = c_vals <  train_thr
val_mask   = (c_vals >= train_thr) & (c_vals < val_thr)
test_mask  = c_vals >= val_thr

if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
    print(" Spatial split too small -> random split.")
    idx = np.random.permutation(len(coords))
    n_test = max(1, int(len(coords) * TEST_FRACTION))
    n_val  = max(1, int(len(coords) * VAL_FRACTION))
    test_c  = coords[idx[:n_test]]
    val_c   = coords[idx[n_test:n_test + n_val]]
    train_c = coords[idx[n_test + n_val:]]
else:
    train_c = coords[train_mask]
    val_c   = coords[val_mask]
    test_c  = coords[test_mask]

print(f"Train:{len(train_c)}  Val:{len(val_c)}  Test:{len(test_c)}")


X_TF = tf.convert_to_tensor(X_full, dtype=tf.float32)
y_TF = tf.convert_to_tensor(y_full, dtype=tf.int32)

def load_patch(coord):
    r = coord[0]
    c = coord[1]
    x = tf.slice(X_TF, [r, c, 0], [PATCH_SIZE, PATCH_SIZE, len(INPUT_BAND_NAMES)])
    y = tf.slice(y_TF, [r, c],    [PATCH_SIZE, PATCH_SIZE])
    y = tf.one_hot(y, NUM_CLASSES)
    return x, y

def augment(x, y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x); y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x);    y = tf.image.flip_up_down(y)
    k = tf.random.uniform((), 0, 4, dtype=tf.int32)
    x = tf.image.rot90(x, k); y = tf.image.rot90(y, k)
    return x, y

def build_ds(coord_array, training):
    ds = tf.data.Dataset.from_tensor_slices(coord_array)
    if training:
        ds = ds.shuffle(min(512, len(coord_array)), reshuffle_each_iteration=True)
    ds = ds.map(load_patch, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    if training:
        ds = ds.repeat()
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = build_ds(train_c, training=True)
val_ds   = build_ds(val_c,   training=False)
test_ds  = build_ds(test_c,  training=False)

steps_per_epoch  = int(np.ceil(len(train_c) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_c)   / BATCH_SIZE))
print(f"steps_per_epoch={steps_per_epoch}  validation_steps={validation_steps}")


def conv_block(x, filters):
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def build_unet(input_channels, num_classes, base=32):
    inputs = layers.Input((PATCH_SIZE, PATCH_SIZE, input_channels))
    c1 = conv_block(inputs, base);        p1 = layers.MaxPooling2D()(c1)
    c2 = conv_block(p1, base*2);          p2 = layers.MaxPooling2D()(c2)
    c3 = conv_block(p2, base*4);          p3 = layers.MaxPooling2D()(c3)
    c4 = conv_block(p3, base*8);          p4 = layers.MaxPooling2D()(c4)
    bn = conv_block(p4, base*16)
    u4 = layers.Conv2DTranspose(base*8, 2, strides=2, padding="same")(bn)
    u4 = layers.concatenate([u4, c4]); c5 = conv_block(u4, base*8)
    u3 = layers.Conv2DTranspose(base*4, 2, strides=2, padding="same")(c5)
    u3 = layers.concatenate([u3, c3]); c6 = conv_block(u3, base*4)
    u2 = layers.Conv2DTranspose(base*2, 2, strides=2, padding="same")(c6)
    u2 = layers.concatenate([u2, c2]); c7 = conv_block(u2, base*2)
    u1 = layers.Conv2DTranspose(base, 2, strides=2, padding="same")(c7)
    u1 = layers.concatenate([u1, c1]); c8 = conv_block(u1, base)
    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(c8)
    return models.Model(inputs, outputs, name="UNet")

model = build_unet(len(INPUT_BAND_NAMES), NUM_CLASSES)
model.summary()


def dice_loss(y_true, y_pred, smooth=1.0):
    y_true_f = tf.reshape(y_true, [-1, NUM_CLASSES])
    y_pred_f = tf.reshape(y_pred, [-1, NUM_CLASSES])
    inter = tf.reduce_sum(y_true_f * y_pred_f, axis=0)
    union = tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0)
    dice = (2. * inter + smooth) / (union + smooth)
    return 1. - tf.reduce_mean(dice)

def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return tf.reduce_mean(cce) + dice_loss(y_true, y_pred)

model.compile(optimizer=optimizers.Adam(LR),
              loss=combined_loss, metrics=["accuracy"])

class GCCallback(callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()

cbs = [
    callbacks.EarlyStopping(monitor="val_loss", patience=10,
                            restore_best_weights=True),
    callbacks.ModelCheckpoint(MODEL_OUT, monitor="val_loss",
                              save_best_only=True),
    callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                patience=5, min_lr=1e-6),
    GCCallback(),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=cbs,
)


y_true_parts, y_pred_parts = [], []
for xb, yb in test_ds.take(int(np.ceil(len(test_c) / BATCH_SIZE))):
    p = np.argmax(model(xb, training=False).numpy(), axis=-1)
    y_pred_parts.append(p.reshape(-1))
    y_true_parts.append(np.argmax(yb.numpy(), axis=-1).reshape(-1))

y_pred_lbl = np.concatenate(y_pred_parts)
y_true_lbl = np.concatenate(y_true_parts)

iou_per_class = jaccard_score(y_true_lbl, y_pred_lbl,
                              average=None, labels=range(NUM_CLASSES),
                              zero_division=0)
miou = iou_per_class.mean()
f1_macro = f1_score(y_true_lbl, y_pred_lbl, average="macro", zero_division=0)

print("\n================ EVALUATION ================")
for i, name in enumerate(CLASS_NAMES):
    print(f"  IoU {name:12s}: {iou_per_class[i]:.3f}")
print(f"  mIoU (mean)   : {miou:.3f}")
print(f"  F1  (macro)   : {f1_macro:.3f}")
print("--------------------------------------------")
if miou > 0.57:
    print(f"  PASS: mIoU {miou:.3f} > 0.57 thesis benchmark")
else:
    print(f"  Below benchmark (0.57). Try more epochs / overlap. mIoU={miou:.3f}")
print("============================================\n")

cm = confusion_matrix(y_true_lbl, y_pred_lbl, labels=range(NUM_CLASSES))
print("Confusion matrix (rows=true, cols=pred):\n", cm)


def show_predictions(n=3):
    n = min(n, len(test_c))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]
    cmap = plt.get_cmap("tab10")
    for i in range(n):
        r, c = test_c[i]
        x_patch = X_full[r:r + PATCH_SIZE, c:c + PATCH_SIZE, :]
        y_patch = y_full[r:r + PATCH_SIZE, c:c + PATCH_SIZE]

        rgb = x_patch[:, :, [2, 1, 0]]
        rgb = (rgb - rgb.min()) / (np.ptp(rgb) + 1e-6)
        pred = np.argmax(model(x_patch[np.newaxis, ...], training=False).numpy()[0],
                         axis=-1)

        axes[i, 0].imshow(rgb);                 axes[i, 0].set_title("Input RGB")
        axes[i, 1].imshow(y_patch, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=NUM_CLASSES-1)
        axes[i, 2].set_title("U-Net Prediction")
        for j in range(3):
            axes[i, j].axis("off")
    plt.tight_layout(); plt.show()

show_predictions(3)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("Loss"); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("Accuracy"); plt.legend()
plt.tight_layout(); plt.show()


def predict_full_scene():
    H, W = y_full.shape
    pred_map = np.zeros((H, W), dtype="uint8")
    for r in range(0, H - PATCH_SIZE + 1, PATCH_SIZE):
        for c in range(0, W - PATCH_SIZE + 1, PATCH_SIZE):
            patch = X_full[r:r+PATCH_SIZE, c:c+PATCH_SIZE, :][np.newaxis, ...]
            p = np.argmax(model(patch, training=False).numpy()[0], axis=-1)
            pred_map[r:r+PATCH_SIZE, c:c+PATCH_SIZE] = p

    out_profile = profile.copy()
    out_profile.update(count=1, dtype="uint8")
    with rasterio.open(PREDMAP_OUT, "w", **out_profile) as dst:
        dst.write(pred_map, 1)
    print(f"Saved full-scene classification map -> {PREDMAP_OUT}")

    plt.figure(figsize=(8, 8))
    plt.imshow(pred_map, cmap="tab10", vmin=0, vmax=NUM_CLASSES-1)
    plt.title("Full-Scene U-Net Classification")
    plt.axis("off"); plt.show()
    return pred_map

predicted_map = predict_full_scene()

print("\nPhase 2 complete.")
