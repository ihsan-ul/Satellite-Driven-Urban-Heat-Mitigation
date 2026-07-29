!pip install rasterio pandas scipy -q

import os, json
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

LST_PATH   = "/content/drive/MyDrive/thesis_phase1_exports/dubai_LST_downscaled.tif"
CLASS_PATH = "/content/drive/MyDrive/thesis_phase2_exports/dubai_prediction_map.tif"
STACK_PATH = "/content/drive/MyDrive/thesis_phase1_exports/dubai_full_fused_stack.tif"
OUT_DIR    = "/content/drive/MyDrive/thesis_phase3_exports"
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_BARE, CLASS_VEG, CLASS_BUILT, CLASS_WATER = 0, 1, 2, 3
CLASS_NAMES = ["Bare/Sand", "Vegetation", "Built-up", "Water"]
BAND_ORDER  = ["B2","B3","B4","B8","B11","B12","NDVI","NDBI","NDWI","LST","DEM","GHSL","LABEL"]

PIXEL_M          = 10.0
SPILLOVER_SIGMA  = 1.5
PHYSICAL_FLOOR_C = 28.0
SEED = 42


COOLING = {
    "green_roof": {"mean_C": 1.45, "range_C": [0.8, 2.1],
        "applies_to": CLASS_BUILT, "reclass_to": CLASS_VEG, "roofable_only": True,
        "source": "Sanchez-Cordero et al. 2025 — LST reduction, retrofitted green roofs"},
    "cool_roof":  {"mean_C": 1.20, "range_C": [0.5, 2.0],
        "applies_to": CLASS_BUILT, "reclass_to": CLASS_BUILT, "roofable_only": True,
        "source": "Santamouris 2014 / Akbari et al. 2016 — high-albedo roof LST reduction"},
    "veg_buffer": {"mean_C": 1.00, "range_C": [0.5, 1.6],
        "applies_to": CLASS_BARE, "reclass_to": CLASS_VEG, "roofable_only": False,
        "source": "Bowler et al. 2010 — urban greening cooling (empirical review)"},
}
MRT_CROSSREF = {"green_roof_MRT_pedestrian_C": 1.83, "green_roof_MRT_canopy_C": 3.50,
                "source": "Alaa et al. 2025 (ENVI-met, hot-arid) — MRT, discussion only"}

with open(os.path.join(OUT_DIR, "cooling_coefficients.json"), "w") as f:
    json.dump({"cooling": COOLING, "mrt_crossref": MRT_CROSSREF}, f, indent=2, default=str)
print("Saved traceable coefficient config -> cooling_coefficients.json")

def load_aligned():
    with rasterio.open(LST_PATH) as s:
        lst = s.read(1).astype("float32"); prof = s.profile
        t_lst, crs_lst, nd = s.transform, s.crs, s.nodata
    with rasterio.open(CLASS_PATH) as s:
        cls = s.read(1).astype("int32"); t_cls, crs_cls = s.transform, s.crs
    with rasterio.open(STACK_PATH) as s:
        ghsl = s.read(BAND_ORDER.index("GHSL") + 1).astype("float32")
        t_gh, crs_gh = s.transform, s.crs

    for name, tr, cr in [("classification", t_cls, crs_cls), ("GHSL", t_gh, crs_gh)]:
        assert crs_lst == cr, f"{name} CRS {cr} != LST CRS {crs_lst}"
        assert np.allclose(np.array(t_lst)[:6], np.array(tr)[:6], atol=1e-6), \
            f"{name} transform differs from LST — reproject before Phase 3."

    h = min(lst.shape[0], cls.shape[0], ghsl.shape[0])
    w = min(lst.shape[1], cls.shape[1], ghsl.shape[1])
    lst, cls, ghsl = lst[:h, :w], cls[:h, :w], ghsl[:h, :w]
    if nd is not None: lst[lst == nd] = np.nan
    lst[lst < 0] = np.nan
    return lst, cls, ghsl, prof

lst0, classification, ghsl, geo_profile = load_aligned()
valid = np.isfinite(lst0)
N_valid = int(valid.sum())

roof_frac = np.clip(ghsl / 10000.0, 0.0, 1.0)
roof_frac = np.where((classification == CLASS_BUILT) & valid, roof_frac, 0.0)

print(f"Aligned grids OK. Scene {lst0.shape}. Valid LST: {N_valid:,} "
      f"({100*N_valid/valid.size:.1f}%).")
print(f"Baseline LST (C): mean {np.nanmean(lst0):.1f}, "
      f"min {np.nanmin(lst0):.1f}, max {np.nanmax(lst0):.1f}")
print(f"Mean roof-able fraction of built-up: "
      f"{roof_frac[classification==CLASS_BUILT].mean():.2f}")


lo, hi = np.nanpercentile(lst0[valid], [5, 95])
def anomaly_weight(lst):
    return np.clip((lst - lo) / (hi - lo + 1e-6), 0.0, 1.0)
AW = anomaly_weight(lst0)

def apply_intervention(lst, cls, interventions, deploy_fraction=1.0,
                       hotspot_only=False, hotspot_pct=90, seed=SEED):
    rng = np.random.default_rng(seed)
    sim_cls = cls.copy()
    delta = np.zeros_like(lst)
    affected = 0
    hot_thresh = np.nanpercentile(lst[valid], hotspot_pct) if hotspot_only else -np.inf
    for iv in interventions:
        cfg = COOLING[iv]
        elig = (cls == cfg["applies_to"]) & valid
        if hotspot_only:
            elig &= (lst >= hot_thresh)
        p_deploy = np.where(elig, deploy_fraction, 0.0)
        if cfg["roofable_only"]:
            p_deploy = p_deploy * roof_frac
        deployed = elig & (rng.random(lst.shape) < p_deploy)
        d = -cfg["mean_C"] * AW
        delta[deployed] += d[deployed]
        if cfg["reclass_to"] is not None:
            sim_cls[deployed] = cfg["reclass_to"]
        affected += int(deployed.sum())
    delta = gaussian_filter(np.where(valid, delta, 0.0), sigma=SPILLOVER_SIGMA)
    sim_lst = np.maximum(lst + delta, PHYSICAL_FLOOR_C)
    sim_lst = np.where(valid, sim_lst, np.nan)
    return sim_lst, sim_cls, affected

def summarise(base, sim, affected, name):
    d = (sim - base)[valid]
    return {"scenario": name, "pixels_retrofitted": affected,
            "area_km2": round(affected * PIXEL_M**2 / 1e6, 3),
            "mean_LST_before_C": round(float(np.nanmean(base[valid])), 3),
            "mean_LST_after_C":  round(float(np.nanmean(sim[valid])), 3),
            "mean_scene_cooling_C": round(float(np.nanmean(d)), 4),
            "max_local_cooling_C":  round(float(np.nanmin(d)), 3)}


scenarios = {
    "S1_green_roofs":   {"ivs": ["green_roof"],               "deploy": 0.30},
    "S2_cool_roofs":    {"ivs": ["cool_roof"],                "deploy": 0.30},
    "S3_mixed":         {"ivs": ["green_roof", "veg_buffer"], "deploy": 0.30},
    "S4_hotspot_first": {"ivs": ["green_roof", "cool_roof"],  "deploy": 0.50,
                         "hotspot": True},
}

results, sim_outputs = [], {}
for name, sc in scenarios.items():
    sim_lst, sim_cls, aff = apply_intervention(
        lst0, classification, sc["ivs"],
        deploy_fraction=sc["deploy"], hotspot_only=sc.get("hotspot", False))
    stats = summarise(lst0, sim_lst, aff, name)
    results.append(stats); sim_outputs[name] = sim_lst
    print(f"[{name:16s}] retrofit {aff:,} px ({stats['area_km2']} km²)  "
          f"mean cooling {stats['mean_scene_cooling_C']:+.3f} °C  "
          f"peak {stats['max_local_cooling_C']:+.2f} °C")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUT_DIR, "scenario_results.csv"), index=False)
print("\n", results_df.to_string(index=False))


print("\n=========== COOLING WITHIN FOOTPRINT / BUILT-UP ===========")
built = (classification == CLASS_BUILT) & valid
foot_rows = []
for name, sim_lst in sim_outputs.items():
    d = sim_lst - lst0
    aff = d < -0.001
    m_aff   = float(np.nanmean(d[aff]))   if aff.any() else 0.0
    p5_aff  = float(np.nanpercentile(d[aff], 5)) if aff.any() else 0.0
    m_built = float(np.nanmean(d[built]))
    foot_rows.append({"scenario": name,
                      "mean_cooling_retrofitted_C": round(m_aff, 3),
                      "p05_cooling_retrofitted_C":  round(p5_aff, 3),
                      "mean_cooling_all_builtup_C":  round(m_built, 4)})
    print(f"[{name:16s}] retrofitted px: {m_aff:+.2f} °C "
          f"(p05 {p5_aff:+.2f})  | all built-up: {m_built:+.3f} °C")
foot_df = pd.DataFrame(foot_rows)
foot_df.to_csv(os.path.join(OUT_DIR, "footprint_cooling.csv"), index=False)


def _weight_sum(iv_key, hotspot):
    cfg = COOLING[iv_key]
    elig = (classification == cfg["applies_to"]) & valid
    if hotspot:
        elig = elig & (lst0 >= np.nanpercentile(lst0[valid], 90))
    base = elig.astype("float64")
    if cfg["roofable_only"]:
        base = base * np.nan_to_num(roof_frac, nan=0.0)
    aw_safe = np.nan_to_num(AW, nan=0.0)
    return float(np.sum(base * aw_safe))

def monte_carlo_fast(name, ivs, deploy, hotspot=False, n=2000):
    rng = np.random.default_rng(SEED)
    wsum = {k: _weight_sum(k, hotspot) for k in ivs}
    means = np.empty(n)
    for i in range(n):
        drawn = {k: rng.uniform(*COOLING[k]["range_C"]) for k in ivs}
        dep   = np.clip(rng.normal(deploy, 0.05), 0.05, 1.0)
        means[i] = -dep * sum(drawn[k] * wsum[k] for k in ivs) / N_valid
    return {"scenario": name, "mc_runs": n,
            "cooling_mean_C": round(float(means.mean()), 4),
            "cooling_p05_C":  round(float(np.percentile(means, 5)), 4),
            "cooling_p95_C":  round(float(np.percentile(means, 95)), 4)}

mc_rows = [monte_carlo_fast(nm, sc["ivs"], sc["deploy"], sc.get("hotspot", False))
           for nm, sc in scenarios.items()]
mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv(os.path.join(OUT_DIR, "monte_carlo_uncertainty.csv"), index=False)
print("\n=============== MONTE-CARLO UNCERTAINTY (95% CI) ===============")
print(mc_df.to_string(index=False))


def plot_scenario(name, sim_lst):
    delta = sim_lst - lst0
    p2, p98 = np.nanpercentile(lst0[valid], [2, 98])

    cooled = delta[valid & (delta < -0.001)]
    dmin = np.nanpercentile(cooled, 2) if cooled.size else -0.5
    vmax_delta = 0.0

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    panels = [
        (np.where(valid, lst0, np.nan),    "Baseline LST",            "inferno", (p2, p98)),
        (np.where(valid, sim_lst, np.nan), f"Simulated — {name}",      "inferno", (p2, p98)),
        (np.where(valid, delta, np.nan),   "Cooling Δ°C (blue=cooler)", "YlGnBu",  (dmin, vmax_delta)),
    ]
    for a, (arr, t, cm, vlim) in zip(ax, panels):
        im = a.imshow(arr, cmap=cm, vmin=vlim[0], vmax=vlim[1])
        a.set_title(t); a.axis("off")
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_maps.png"), dpi=150, bbox_inches="tight")
    plt.show()

def bar_uncertainty():
    plt.figure(figsize=(8, 5))
    x = range(len(mc_df))
    lower = (mc_df["cooling_mean_C"] - mc_df["cooling_p95_C"]).abs()
    upper = (mc_df["cooling_p05_C"]  - mc_df["cooling_mean_C"]).abs()
    plt.bar(x, -mc_df["cooling_mean_C"], yerr=[lower, upper],
            capsize=6, color="steelblue")
    plt.xticks(x, mc_df["scenario"], rotation=20, ha="right")
    plt.ylabel("Mean scene cooling (°C)  [95% CI]")
    plt.title("Predicted cooling with coefficient + deployment uncertainty")
    plt.grid(axis="y", alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "uncertainty_bars.png"), dpi=150); plt.show()

def save_geotiff(arr, fn):
    prof = geo_profile.copy(); prof.update(count=1, dtype="float32", nodata=-9999)
    with rasterio.open(os.path.join(OUT_DIR, fn), "w", **prof) as dst:
        dst.write(np.nan_to_num(arr, nan=-9999).astype("float32"), 1)

save_geotiff(lst0, "lst_baseline.tif")
for name, sim_lst in sim_outputs.items():
    plot_scenario(name, sim_lst)
    save_geotiff(sim_lst, f"{name}_lst.tif")
    save_geotiff(sim_lst - lst0, f"{name}_delta.tif")
bar_uncertainty()

def plot_cooling_only(name, sim_lst):
    delta = sim_lst - lst0
    d = np.where(valid & (delta < -0.02), delta, np.nan)
    plt.figure(figsize=(7, 9))
    im = plt.imshow(d, cmap="YlGnBu", vmin=np.nanpercentile(d, 2), vmax=0)
    plt.title(f"Where cooling occurs — {name}"); plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Δ°C")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{name}_coolingonly.png"), dpi=150, bbox_inches="tight")
    plt.show()

plot_cooling_only("S3_mixed", sim_outputs["S3_mixed"])
plot_cooling_only("S4_hotspot_first", sim_outputs["S4_hotspot_first"])


with open(os.path.join(OUT_DIR, "DISCLAIMER.txt"), "w") as f: f.write(DISCLAIMER)
print("\n" + "="*70 + "\n" + DISCLAIMER + "\n" + "="*70)
print("\nPhase 3 + 3b complete. Outputs in:", OUT_DIR)
