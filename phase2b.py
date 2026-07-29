!pip install rasterio pandas ipywidgets -q

import os, io, time
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
import matplotlib.pyplot as plt

PRED_PATH  = "/content/drive/MyDrive/thesis_phase2_exports/dubai_prediction_map.tif"
STACK_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
OUT_DIR    = "/content/drive/MyDrive/thesis_phase2_exports/phase2b_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLE_CSV   = os.path.join(OUT_DIR, "reference_sample.csv")
NUM_CLASSES  = 4
CLASS_NAMES  = ["Bare/Sand", "Vegetation", "Built-up", "Water"]
POINTS_PER_CLASS = 75
SEED = 42

BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12",
              "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL", "LABEL"]

def stage_A_sample():
    rng = np.random.default_rng(SEED)

    with rasterio.open(PRED_PATH) as src:
        pred = src.read(1)
        transform, crs, nodata = src.transform, src.crs, src.nodata

    with rasterio.open(STACK_PATH) as s:
        wc = s.read(BAND_ORDER.index("LABEL") + 1)
    wc = np.round(wc).astype("int32")

    h = min(pred.shape[0], wc.shape[0]); w = min(pred.shape[1], wc.shape[1])
    pred, wc = pred[:h, :w], wc[:h, :w]
    valid = (pred != (nodata if nodata is not None else 255)) & \
            (pred >= 0) & (pred < NUM_CLASSES)

    rows = []
    for k in range(NUM_CLASSES):
        rr, cc = np.where(valid & (pred == k))
        if len(rr) == 0:
            print(f"  ! No pixels predicted as {CLASS_NAMES[k]} — skipping.")
            continue
        n = min(POINTS_PER_CLASS, len(rr))
        pick = rng.choice(len(rr), size=n, replace=False)
        for i in pick:
            rows.append({"row": int(rr[i]), "col": int(cc[i]),
                         "unet_pred": k, "worldcover_ref": int(wc[rr[i], cc[i]])})

    df = pd.DataFrame(rows)

    xs, ys = rasterio.transform.xy(transform, df["row"].values, df["col"].values)
    lon, lat = warp_transform(crs, "EPSG:4326", xs, ys)
    df["lon"], df["lat"] = np.round(lon, 6), np.round(lat, 6)

    df.insert(0, "point_id", range(len(df)))
    df["true_label"] = -1
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(SAMPLE_CSV, index=False)

    print(f"Stage A complete: {len(df)} reference points -> {SAMPLE_CSV}")
    print(df["unet_pred"].value_counts().rename(
        index=dict(enumerate(CLASS_NAMES))).to_string())
    print("\nNow run Stage B to label them against satellite imagery.")
    return df

def stage_B_label(chip_deg=0.0025):
    import requests
    from PIL import Image
    import ipywidgets as w
    from IPython.display import display, clear_output

    df = pd.read_csv(SAMPLE_CSV)
    todo = df.index[df["true_label"] < 0].tolist()
    if not todo:
        print("All points already labelled ✔  Run Stage C for metrics.")
        return
    print(f"{len(todo)} of {len(df)} points still to label. "
          f"Chip size ≈ {chip_deg*111:.2f} km across.\n")

    state = {"pos": 0}
    out_img  = w.Output()
    out_info = w.Output()

    def esri_chip(lon, lat):
        d = chip_deg
        url = ("https://services.arcgisonline.com/arcgis/rest/services/"
               "World_Imagery/MapServer/export"
               f"?bbox={lon-d},{lat-d},{lon+d},{lat+d}"
               "&bboxSR=4326&imageSR=4326&size=400,400&format=png&f=image")
        r = requests.get(url, timeout=20)
        return Image.open(io.BytesIO(r.content))

    def render():
        idx = todo[state["pos"]]
        row = df.loc[idx]
        with out_img:
            clear_output(wait=True)
            try:
                img = esri_chip(row["lon"], row["lat"])
                plt.figure(figsize=(5, 5)); plt.imshow(img)
                plt.axhline(200, color="red", lw=0.7); plt.axvline(200, color="red", lw=0.7)
                plt.title("Reference point (red cross)"); plt.axis("off"); plt.show()
            except Exception as e:
                print("Imagery load failed:", e, "\nlon,lat =", row["lon"], row["lat"])
        with out_info:
            clear_output(wait=True)
            print(f"Point {state['pos']+1}/{len(todo)}  (id={row['point_id']})")
            print(f"U-Net says   : {CLASS_NAMES[int(row['unet_pred'])]}")
            print(f"WorldCover   : {CLASS_NAMES[int(row['worldcover_ref'])]}")
            print("Pick the TRUE class from the imagery (unbiased):")

    def label(k):
        idx = todo[state["pos"]]
        df.at[idx, "true_label"] = k
        df.to_csv(SAMPLE_CSV, index=False)
        nxt()

    def nxt(_=None):
        if state["pos"] < len(todo) - 1:
            state["pos"] += 1; render()
        else:
            with out_img: clear_output(); print("Done — all points labelled ✔")
            with out_info: clear_output(); print("Run Stage C for accuracy metrics.")
            stage_C_metrics()

    def prev(_=None):
        if state["pos"] > 0:
            state["pos"] -= 1; render()

    btns = [w.Button(description=n, button_style="info") for n in CLASS_NAMES]
    for b, k in zip(btns, range(NUM_CLASSES)):
        b.on_click(lambda _b, kk=k: label(kk))
    skip = w.Button(description="Unsure/Skip", button_style="warning"); skip.on_click(nxt)
    back = w.Button(description="◀ Back");                              back.on_click(prev)

    display(w.HBox([out_img, out_info]))
    display(w.HBox(btns + [skip, back]))
    render()

def stage_C_metrics():
    from sklearn.metrics import (confusion_matrix, cohen_kappa_score,
                                 accuracy_score)
    df = pd.read_csv(SAMPLE_CSV)
    lab = df[df["true_label"] >= 0].copy()
    if len(lab) < 20:
        print(f"Only {len(lab)} labelled points — label more in Stage B first.")
        return
    print(f"Accuracy assessment on {len(lab)} human-labelled reference points.\n")

    y_true = lab["true_label"].astype(int).values

    def report(pred, name):
        acc   = accuracy_score(y_true, pred)
        kappa = cohen_kappa_score(y_true, pred)
        cm = confusion_matrix(y_true, pred, labels=range(NUM_CLASSES))
        with np.errstate(divide="ignore", invalid="ignore"):
            ua = np.diag(cm) / cm.sum(axis=0)
            pa = np.diag(cm) / cm.sum(axis=1)
        print(f"==================  {name}  ==================")
        print(f"  Overall Accuracy : {acc:.3f}    Kappa : {kappa:.3f}")
        for i, n in enumerate(CLASS_NAMES):
            print(f"  {n:12s}  User's(prec)={ua[i]:.3f}  Producer's(rec)={pa[i]:.3f}")
        print("  Confusion matrix (rows=reference truth, cols=map):")
        print("   ", cm.tolist())
        print()
        return acc, kappa, cm

    u_acc, u_k, u_cm = report(lab["unet_pred"].astype(int).values,       "U-NET MAP")
    w_acc, w_k, w_cm = report(lab["worldcover_ref"].astype(int).values,  "WORLDCOVER")

    print("==================  VERDICT  ==================")
    print(f"  U-Net OA {u_acc:.3f} (κ={u_k:.3f})  vs  WorldCover OA {w_acc:.3f} (κ={w_k:.3f})")
    if u_acc >= w_acc:
        print("U-Net matches/exceeds the WorldCover reference against human truth.")
    else:
        print("U-Net below WorldCover — inspect the confusion matrix for the weak class.")

    lab.to_csv(os.path.join(OUT_DIR, "reference_sample_labelled.csv"), index=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for a, (cm, t) in zip(ax, [(u_cm, "U-Net"), (w_cm, "WorldCover")]):
        im = a.imshow(cm, cmap="Blues")
        a.set_xticks(range(NUM_CLASSES)); a.set_yticks(range(NUM_CLASSES))
        a.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        a.set_yticklabels(CLASS_NAMES)
        a.set_xlabel("Map"); a.set_ylabel("Reference truth")
        a.set_title(f"{t} confusion")
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                a.text(j, i, cm[i, j], ha="center",
                       color="white" if cm[i, j] > cm.max()/2 else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150)
    plt.show()
    print(f"\nSaved -> {OUT_DIR}/reference_sample_labelled.csv + confusion_matrices.png")

stage_A_sample()
stage_B_label()
