
!pip install earthengine-api geemap rasterio -q

import ee
import geemap
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
END_YEAR   = 2026
SUMMER_START_MONTH = 6
SUMMER_END_MONTH   = 9

MAX_CLOUD_PCT = 20

EXPORT_SCALE  = 10
EXPORT_FOLDER = "thesis_phase1_exports"
EXPORT_PREFIX = "dubai_test" if USE_TEST_AOI else "dubai_full"

ST_SCALE  = 0.00341802
ST_OFFSET = 149.0
KELVIN_TO_C = 273.15
S2_SCALE = 10000.0


def mask_landsat_c2(image):
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

clearest = ee.Image(landsat_coll.sort("CLOUD_COVER").first())
print("LST scene:", clearest.get("LANDSAT_PRODUCT_ID").getInfo(),
      "| cloud:", clearest.get("CLOUD_COVER").getInfo(), "%")

lst_composite = clearest.select("LST").clip(AOI)

def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit  = 1 << 10
    cirrus_bit = 1 << 11
    mask = (qa.bitwiseAnd(cloud_bit).eq(0)
              .And(qa.bitwiseAnd(cirrus_bit).eq(0)))
    return image.updateMask(mask)

def scale_s2(image):
    optical = image.select(["B2", "B3", "B4", "B8", "B11", "B12"]) \
                   .divide(S2_SCALE)
    return image.addBands(optical, overwrite=True)

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(AOI)
        .filter(ee.Filter.calendarRange(START_YEAR, END_YEAR, "year"))
        .filter(ee.Filter.calendarRange(SUMMER_START_MONTH,
                                         SUMMER_END_MONTH, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_PCT))
        .map(mask_s2_clouds)
        .map(scale_s2))

s2_bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
s2_composite = s2.select(s2_bands).median().clip(AOI)
print("Sentinel-2 optical composite built.")


ndvi = s2_composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
ndbi = s2_composite.normalizedDifference(["B11", "B8"]).rename("NDBI")
ndwi = s2_composite.normalizedDifference(["B3", "B8"]).rename("NDWI")
print("Spectral indices computed.")


dem = ee.Image("USGS/SRTMGL1_003").select("elevation").rename("DEM").clip(AOI)
ghsl = (ee.Image("JRC/GHSL/P2023A/GHS_BUILT_S/2020")
          .select("built_surface").rename("GHSL").clip(AOI))
print("Auxiliary layers (DEM, GHSL) added.")


worldcover = ee.Image("ESA/WorldCover/v200/2021").select("Map").clip(AOI)
label = (worldcover
         .remap([10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
                [ 1,  1,  1,  1,  2,  0,  0,  3,  1,  1,   0])
         .rename("LABEL"))
print("ESA WorldCover labels added and remapped to thesis classes.")
print("  LABEL codes -> 0:Other/Bare  1:Vegetation  2:Built-up  3:Water")


s2_projection = s2_composite.select("B4").projection()

lst_10m = (lst_composite
           .reproject(crs=s2_projection, scale=EXPORT_SCALE)
           .rename("LST"))

label_10m = (label
             .reproject(crs=s2_projection, scale=EXPORT_SCALE)
             .rename("LABEL"))

fused_stack = (s2_composite
               .addBands(ndvi)
               .addBands(ndbi)
               .addBands(ndwi)
               .addBands(lst_10m)
               .addBands(dem)
               .addBands(ghsl)
               .addBands(label_10m)
               .clip(AOI))

print("Fused multi-band stack assembled:")
print(fused_stack.bandNames().getInfo())


stats = lst_10m.reduceRegion(
    reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
    geometry=AOI, scale=EXPORT_SCALE, maxPixels=1e13, bestEffort=True)
print("LST stats (Celsius):", stats.getInfo())
print(">> Sanity: Dubai summer surfaces are typically ~40-60 C.")




import os, time
from google.colab import drive

if not os.path.isdir("/content/drive/MyDrive"):
    drive.mount("/content/drive")
else:
    print("Drive already mounted.")

task = ee.batch.Export.image.toDrive(
    image=fused_stack.toFloat(),
    description=f"{EXPORT_PREFIX}_fused_stack",
    folder=EXPORT_FOLDER,
    fileNamePrefix=f"{EXPORT_PREFIX}_fused_stack",
    region=AOI,
    scale=EXPORT_SCALE,
    crs="EPSG:32640",
    maxPixels=1e13,
    fileFormat="GeoTIFF")
task.start()
print(f"Export STARTED -> Drive/{EXPORT_FOLDER}/{EXPORT_PREFIX}_fused_stack.tif")

print("Waiting for the export task to complete (polling every 5 min)...")
while task.active():
    print("still running (state:", task.status()["state"], ")")
    time.sleep(300)

final_state = task.status()["state"]
print("Task finished. Final state:", final_state)

if final_state != "COMPLETED":
    print("EXPORT DID NOT COMPLETE. Full status below:")
    print(task.status())
    TIF_PATH = None
else:
    drive_path = (f"/content/drive/MyDrive/{EXPORT_FOLDER}/"
                  f"{EXPORT_PREFIX}_fused_stack.tif")
    if os.path.exists(drive_path):
        size_mb = os.path.getsize(drive_path) / 1e6
        print(f"SUCCESS. File readable by Colab at:\n  {drive_path}")
        print(f"  Size: {size_mb:.1f} MB")
    else:
        print("Task completed but file not visible yet. Give Drive a few")
        print("seconds to sync, then check:", drive_path)
    TIF_PATH = drive_path
