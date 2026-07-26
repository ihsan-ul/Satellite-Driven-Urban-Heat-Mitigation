!pip install rasterio geemap pandas -q

import os
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


STACK_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
CLASS_PATH = "/content/dubai_prediction_map.tif"

OUT_DIR = "/content/phase3_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

INPUT_BAND_NAMES = ["B2", "B3", "B4", "B8", "B11", "B12",
                    "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL", "LABEL"]
LST_BAND_INDEX = INPUT_BAND_NAMES.index("LST") + 1

CLASS_OTHER, CLASS_VEG, CLASS_BUILT, CLASS_WATER = 0, 1, 2, 3
CLASS_NAMES = ["Other/Bare", "Vegetation", "Built-up", "Water"]


COOLING_COEFFICIENTS = {
    "green_roof":    {"lst_reduction_C": 1.45, "source": "Sanchez-Cordero et al. 2025"},
    "cool_pavement": {"lst_reduction_C": 2.20, "source": "Fork et al. 2025 (albedo, adapted)"},
    "veg_buffer":    {"lst_reduction_C": 1.00, "source": "Bowler et al. 2010 (greening)"},
}
MRT_CROSSREF = {"green_roof_MRT_pedestrian_C": 1.83,
                "green_roof_MRT_canopy_C": 3.50,
                "source": "Alaa et al. 2025 (ENVI-met, hot-arid)"}

with open(os.path.join(OUT_DIR, "cooling_coefficients.json"), "w") as f:
    json.dump({"coefficients": COOLING_COEFFICIENTS,
               "mrt_crossref": MRT_CROSSREF}, f, indent=2)
print("Saved cooling-coefficient config -> cooling_coefficients.json")


with rasterio.open(STACK_PATH) as src:
    lst_baseline = src.read(LST_BAND_INDEX).astype("float32")
    geo_profile = src.profile

with rasterio.open(CLASS_PATH) as src:
    classification = src.read(1).astype("int32")

lst_baseline = np.nan_to_num(lst_baseline, nan=np.nan)
h = min(lst_baseline.shape[0], classification.shape[0])
w = min(lst_baseline.shape[1], classification.shape[1])
lst_baseline   = lst_baseline[:h, :w]
classification = classification[:h, :w]

valid = np.isfinite(lst_baseline)
print(f"Loaded LST baseline + classification. Size: {h}x{w}")
print(f"Valid LST pixels: {valid.sum():,} "
      f"({100*valid.sum()/valid.size:.1f}% of scene)")
print("Baseline LST (C) -> mean {:.1f}, min {:.1f}, max {:.1f}".format(
    np.nanmean(lst_baseline), np.nanmin(lst_baseline), np.nanmax(lst_baseline)))


def apply_intervention(lst, class_map, interventions, deploy_fraction=1.0,
                       zone_mask=None, new_class_for_report=None, seed=42):

    rng = np.random.default_rng(seed)
    sim_lst = lst.copy()
    sim_class = class_map.copy()
    affected_total = 0

    for iv in interventions:
        tgt = iv["target_class"]
        coeff = COOLING_COEFFICIENTS[iv["coefficient_key"]]["lst_reduction_C"]

        eligible = (class_map == tgt) & np.isfinite(lst)
        if zone_mask is not None:
            eligible &= zone_mask

        if deploy_fraction < 1.0:
            rnd = rng.random(eligible.shape)
            eligible &= (rnd < deploy_fraction)

        sim_lst[eligible] -= coeff
        if iv.get("reclass_to") is not None:
            sim_class[eligible] = iv["reclass_to"]

        affected_total += int(eligible.sum())

    return sim_lst, sim_class, affected_total


def summarise(baseline, simulated, affected, scenario_name, mask):
    delta = simulated - baseline
    mean_drop = np.nanmean(delta[mask])
    max_drop  = np.nanmin(delta[mask])
    mean_before = np.nanmean(baseline[mask])
    mean_after  = np.nanmean(simulated[mask])
    return {
        "scenario": scenario_name,
        "pixels_retrofitted": affected,
        "area_km2": affected * (10 * 10) / 1e6,
        "mean_LST_before_C": round(float(mean_before), 3),
        "mean_LST_after_C": round(float(mean_after), 3),
        "mean_scene_cooling_C": round(float(mean_drop), 3),
        "max_local_cooling_C": round(float(max_drop), 3),
    }


DEPLOY_FRACTION = 1.0

S1 = [{"target_class": CLASS_BUILT,
       "coefficient_key": "green_roof",
       "reclass_to": CLASS_VEG}]

S2 = [{"target_class": CLASS_BUILT,
       "coefficient_key": "cool_pavement",
       "reclass_to": CLASS_BUILT}]


S3 = [{"target_class": CLASS_BUILT, "coefficient_key": "green_roof",
       "reclass_to": CLASS_VEG},
      {"target_class": CLASS_OTHER, "coefficient_key": "veg_buffer",
       "reclass_to": CLASS_VEG}]

scenarios = {
    "S1_green_roofs":   S1,
    "S2_cool_pavement": S2,
    "S3_mixed":         S3,
}

results = []
sim_outputs = {}

for name, ivs in scenarios.items():

    frac = DEPLOY_FRACTION
    sim_lst, sim_class, affected = apply_intervention(
        lst_baseline, classification, ivs,
        deploy_fraction=frac, zone_mask=None)
    stats = summarise(lst_baseline, sim_lst, affected, name, valid)
    results.append(stats)
    sim_outputs[name] = (sim_lst, sim_class)
    print(f"\n[{name}]  retrofitted {affected:,} px "
          f"({stats['area_km2']:.2f} km2)  "
          f"mean scene cooling {stats['mean_scene_cooling_C']:+.3f} C")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "scenario_results.csv"), index=False)
print("\n================ SCENARIO SUMMARY ================")
print(results_df.to_string(index=False))
print("Saved -> scenario_results.csv")


PERTURBATIONS = [-0.30, -0.15, 0.0, 0.15, 0.30]
sens_rows = []

for pct in PERTURBATIONS:
    scaled = {k: v["lst_reduction_C"] * (1 + pct)
              for k, v in COOLING_COEFFICIENTS.items()}

    sim = lst_baseline.copy()
    eligible = (classification == CLASS_BUILT) & valid
    sim[eligible] -= scaled["green_roof"]
    mean_drop = np.nanmean((sim - lst_baseline)[valid])

    sens_rows.append({"coeff_change_pct": int(pct * 100),
                      "green_roof_coeff_C": round(scaled["green_roof"], 3),
                      "mean_scene_cooling_C": round(float(mean_drop), 4)})

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(OUT_DIR, "sensitivity_analysis.csv"), index=False)
print("\n=============== SENSITIVITY (S1 green roofs) ===============")
print(sens_df.to_string(index=False))
print("Saved -> sensitivity_analysis.csv")


def plot_scenario(name, sim_lst):
    delta = sim_lst - lst_baseline
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(np.where(valid, lst_baseline, np.nan),
                         cmap="inferno")
    axes[0].set_title("Baseline LST (C)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(np.where(valid, sim_lst, np.nan), cmap="inferno")
    axes[1].set_title(f"Simulated LST (C) — {name}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(np.where(valid, delta, np.nan),
                         cmap="Blues_r", vmin=-3, vmax=0)
    axes[2].set_title("Cooling (delta C, negative = cooler)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_maps.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

for name, (sim_lst, _) in sim_outputs.items():
    plot_scenario(name, sim_lst)

plt.figure(figsize=(7, 5))
plt.plot(sens_df["coeff_change_pct"], sens_df["mean_scene_cooling_C"],
         marker="o")
plt.axvline(0, color="grey", ls="--", lw=1)
plt.xlabel("Coefficient change (%)")
plt.ylabel("Mean scene cooling (C)")
plt.title("Sensitivity of predicted cooling to coefficient uncertainty (S1)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sensitivity_plot.png"), dpi=150)
plt.show()


def save_geotiff(array, filename, dtype="float32"):
    prof = geo_profile.copy()
    prof.update(count=1, dtype=dtype, height=array.shape[0],
                width=array.shape[1])
    with rasterio.open(os.path.join(OUT_DIR, filename), "w", **prof) as dst:
        dst.write(array.astype(dtype), 1)

save_geotiff(lst_baseline, "lst_baseline.tif")
for name, (sim_lst, _) in sim_outputs.items():
    save_geotiff(sim_lst, f"{name}_lst.tif")
    save_geotiff(sim_lst - lst_baseline, f"{name}_delta.tif")
print("\nSaved georeferenced GeoTIFFs (baseline + simulated + delta) to",
      OUT_DIR)


def interactive_map(scenario="S1_green_roofs"):

    import geemap
    sim_lst, _ = sim_outputs[scenario]
    base_tif = os.path.join(OUT_DIR, "lst_baseline.tif")
    sim_tif  = os.path.join(OUT_DIR, f"{scenario}_lst.tif")

    m = geemap.Map()
    m.add_raster(base_tif, colormap="inferno", layer_name="Baseline LST")
    m.add_raster(sim_tif,  colormap="inferno",
                 layer_name=f"Simulated LST ({scenario})")
    return m


DISCLAIMER = (
    "THERMODYNAMIC DISCLAIMER: These maps are first-order estimates produced "
    "by applying published empirical cooling coefficients to U-Net-classified "
    "surfaces. They indicate the RELATIVE priority and approximate magnitude "
    "of cooling benefits to support planning decisions. They are NOT guaranteed "
    "temperature outcomes and do not replace site-specific physical simulation "
    "(e.g., ENVI-met) or field validation."
)
with open(os.path.join(OUT_DIR, "DISCLAIMER.txt"), "w") as f:
    f.write(DISCLAIMER)
print("\n" + "=" * 70)
print(DISCLAIMER)
print("=" * 70)
print("\nPhase 3 complete. Outputs in:", OUT_DIR)
