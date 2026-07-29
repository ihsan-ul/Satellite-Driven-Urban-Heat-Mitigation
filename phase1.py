

!pip install earthengine-api geemap -q

import geemap
import ee
from google.colab import userdata

GEE_PROJECT_ID = userdata.get("GEEID")

try:
    ee.Initialize(project=GEE_PROJECT_ID)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_ID)

print("Earth Engine initialised.")

DUBAI_AOI = ee.Geometry.Rectangle([55.10, 24.80, 55.50, 25.40])
TEST_AOI  = ee.Geometry.Rectangle([55.25, 25.15, 55.32, 25.22])

USE_TEST_AOI = False
AOI = TEST_AOI if USE_TEST_AOI else DUBAI_AOI

START_YEAR = 2016
END_YEAR   = 2025
SUMMER_START_MONTH = 6
SUMMER_END_MONTH   = 9

MAX_CLOUD_PCT   = 40
CLEAR_THRESHOLD = 0.60

EXPORT_SCALE  = 10
EXPORT_CRS    = "EPSG:32640"
EXPORT_FOLDER = "thesis_phase1_exports"
EXPORT_PREFIX = "dubai_test" if USE_TEST_AOI else "dubai_full"

ST_SCALE, ST_OFFSET, KELVIN_TO_C = 0.00341802, 149.0, 273.15

def mask_landsat_c2(image):
    """Mask cloud, cloud-shadow and dilated-cloud using QA_PIXEL bits 1,3,4."""
    qa = image.select("QA_PIXEL")
    mask = (qa.bitwiseAnd(1 << 1).eq(0)
              .And(qa.bitwiseAnd(1 << 3).eq(0))
              .And(qa.bitwiseAnd(1 << 4).eq(0)))
    return image.updateMask(mask)

def add_lst_celsius(image):
    lst_c = (image.select("ST_B10")
                  .multiply(ST_SCALE).add(ST_OFFSET)
                  .subtract(KELVIN_TO_C)
                  .rename("LST"))
    return image.addBands(lst_c)

l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")

landsat_coll = (l8.merge(l9)
                  .filterBounds(AOI)
                  .filter(ee.Filter.calendarRange(START_YEAR, END_YEAR, "year"))
                  .filter(ee.Filter.calendarRange(SUMMER_START_MONTH,
                                                  SUMMER_END_MONTH, "month"))
                  .map(mask_landsat_c2)
                  .map(add_lst_celsius))

lst_composite = landsat_coll.select("LST").median().clip(AOI)
print(f"Landsat LST: 10-yr summer median from "
      f"{landsat_coll.size().getInfo()} scenes ({START_YEAR}-{END_YEAR}).")


CS_PLUS  = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
S2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]

def mask_s2_cs(image):
    """Keep pixels graded 'clear' by Cloud Score+ (cs_cdf)."""
    return image.updateMask(image.select("cs_cdf").gte(CLEAR_THRESHOLD))

def scale_s2(image):
    optical = image.select(S2_BANDS).divide(10000.0)
    return image.addBands(optical, overwrite=True)

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filter(ee.Filter.calendarRange(START_YEAR, END_YEAR, "year"))
        .filter(ee.Filter.calendarRange(SUMMER_START_MONTH,
                                        SUMMER_END_MONTH, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .linkCollection(CS_PLUS, ["cs_cdf"])
        .map(mask_s2_cs)
        .map(scale_s2))

s2_composite = s2.select(S2_BANDS).median().clip(AOI)
print(f"Sentinel-2 optical composite from "
      f"{s2.size().getInfo()} scenes (Cloud Score+ masked).")

ndvi = s2_composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
ndbi = s2_composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
print("Spectral indices computed (NDVI, NDBI, NDWI).")

dem  = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM").clip(AOI)
ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
          .select("built_surface").rename("GHSL").clip(AOI))
print("Auxiliary layers added (DEM, GHSL built-surface).")


worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(AOI)
label = (worldcover
         .remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                [ 1,  1,  1,  1,  2,  0,  0,  3,  1,  1,   0])
         .rename("LABEL"))
print("WorldCover remapped -> 0:Bare/Sand  1:Vegetation  2:Built-up  3:Water")


s2_proj = s2_composite.select("B4").projection()

lst_10m   = lst_composite.reproject(crs=s2_proj, scale=EXPORT_SCALE).rename("LST")
label_10m = label.reproject(crs=s2_proj, scale=EXPORT_SCALE).rename("LABEL")


BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12",
              "NDVI", "NDBI", "NDWI", "LST", "DEM", "GHSL", "LABEL"]

fused_stack = (s2_composite
               .addBands(ndvi).addBands(ndbi).addBands(ndwi)
               .addBands(lst_10m).addBands(dem).addBands(ghsl)
               .addBands(label_10m)
               .select(BAND_ORDER)
               .clip(AOI))

actual_bands = fused_stack.bandNames().getInfo()
assert actual_bands == BAND_ORDER, f"Band order mismatch: {actual_bands}"
print("Fused stack assembled with verified band order:")
print("  ", actual_bands)

stats = lst_10m.reduceRegion(
    reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
    geometry=AOI, scale=EXPORT_SCALE, maxPixels=1e13, bestEffort=True).getInfo()
print("\nLST sanity check (Celsius):", stats)
print(">> Expected: Dubai summer surfaces ~40-60 C. "
      "If min < 20 or max > 75, revisit masking.")

task = ee.batch.Export.image.toDrive(
    image=fused_stack.toFloat(),
    description=f"{EXPORT_PREFIX}_fused_stack",
    folder=EXPORT_FOLDER,
    fileNamePrefix=f"{EXPORT_PREFIX}_fused_stack",
    region=AOI,
    scale=EXPORT_SCALE,
    crs=EXPORT_CRS,
    maxPixels=1e13,
    fileFormat="GeoTIFF")
task.start()
print(f"\nExport STARTED -> Drive/{EXPORT_FOLDER}/{EXPORT_PREFIX}_fused_stack.tif")
print("Monitor progress at https://code.earthengine.google.com/tasks "
      "(or the 'Tasks' tab). Full-AOI export typically takes 10–30 min.")
