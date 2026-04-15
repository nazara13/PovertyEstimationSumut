"""
================================================================
gee_extractor.py
Ekstraksi Fitur Satelit dari Google Earth Engine
Wilayah: Provinsi Sumatera Utara
================================================================

CARA PAKAI:
  1. Autentikasi GEE: `earthengine authenticate`
  2. Jalankan: `python src/gee_extractor.py`
  3. Output CSV akan tersimpan di: data/processed/features_satellite.csv
"""

import ee
import geemap
import pandas as pd
import geopandas as gpd
import json
import os

# ── Inisialisasi Google Earth Engine ─────────────────────────
ee.Initialize(project="geospatialproject-493419")  # Ganti dengan project GEE Anda

# ── Konfigurasi ───────────────────────────────────────────────
YEAR = 2023
START_DATE = f"{YEAR}-01-01"
END_DATE = f"{YEAR}-12-31"
SCALE = 1000  # resolusi agregasi (meter) — 1km cukup untuk tingkat kabupaten
MAX_PIXELS = 1e13

SPATIAL_DIR = os.path.join("data", "spatial")
PROCESSED_DIR = os.path.join("data", "processed")
GEOJSON_PATH = os.path.join(SPATIAL_DIR, "kabupaten_sumut.geojson")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ── Load Batas Wilayah ke GEE ────────────────────────────────
def load_study_area(geojson_path: str) -> ee.FeatureCollection:
    """Muat batas kabupaten Sumatera Utara dari file GeoJSON lokal ke GEE."""
    gdf = gpd.read_file(geojson_path)
    # Pastikan CRS = WGS84 (EPSG:4326)
    gdf = gdf.to_crs(epsg=4326)
    
    # ⚠️ FIX: Sederhanakan geometri agar tidak melempar "Payload Exceeds Limit" (10MB GEE limit)
    gdf['geometry'] = gdf['geometry'].simplify(0.005, preserve_topology=True)
    
    geojson_dict = json.loads(gdf.to_json())
    fc = ee.FeatureCollection(geojson_dict)
    # Kolom GeoJSON yang tersedia: nmkab, nmprov, idkab, Shape_Leng, Shape_Area, kode_wilayah_dagri, nama_kab_dagri
    print(f"[OK] Sukses load {len(gdf)} kabupaten/kota Sumatera Utara (Simplified Geometry)")
    return fc, gdf


# ── Sentinel-2 – NDVI & NDBI ─────────────────────────────────
def get_sentinel2_indices(aoi: ee.FeatureCollection) -> ee.Image:
    """
    Menghitung rata-rata NDVI & NDBI dari Sentinel-2 (S2_SR_HARMONIZED).
    NDVI = (NIR - Red) / (NIR + Red)  → Vegetasi
    NDBI = (SWIR - NIR) / (SWIR + NIR) → Kepadatan Bangunan
    """
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(aoi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
        .map(lambda img: img.divide(10000).copyProperties(img, img.propertyNames()))
    )

    def compute_indices(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndbi = img.normalizedDifference(["B11", "B8"]).rename("ndbi")
        return img.addBands([ndvi, ndbi])

    composite = s2.map(compute_indices).median().select(["ndvi", "ndbi"])
    print(f"✅ Sentinel-2 composite processing...")
    return composite


# ── VIIRS Nighttime Lights ────────────────────────────────────
def get_nighttime_lights(aoi: ee.FeatureCollection) -> ee.Image:
    """
    Mengambil rata-rata Nighttime Lights (VIIRS/DNB) sebagai proksi aktivitas ekonomi.
    Output band: 'ntl_mean', 'ntl_std'
    """
    ntl = (
        ee.ImageCollection("NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(aoi)
        .select("avg_rad")
    )
    ntl_mean = ntl.mean().rename("ntl_mean")
    ntl_std = ntl.reduce(ee.Reducer.stdDev()).rename("ntl_std")
    return ntl_mean.addBands(ntl_std)


# ── Dynamic World – Land Cover ────────────────────────────────
def get_land_cover(aoi: ee.FeatureCollection) -> ee.Image:
    """
    Menghitung persentase area untuk setiap kelas tutupan lahan (Dynamic World).
    Kelas yang relevan:
      0=water, 1=trees, 2=grass, 3=flooded_vegetation,
      4=crops, 5=shrub, 6=built, 7=bare, 8=snow_ice
    """
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterDate(START_DATE, END_DATE)
        .filterBounds(aoi)
        .select("label")
        .mode()  # Mode: kelas paling dominan per piksel
    )
    built = dw.eq(6).rename("urban_pct")
    crops = dw.eq(4).rename("agri_pct")
    trees = dw.eq(1).rename("tree_pct")
    return built.addBands([crops, trees]).multiply(100)


# ── Agregasi per Kabupaten ────────────────────────────────────
def extract_zonal_stats(
    image: ee.Image, fc: ee.FeatureCollection, scale: int = SCALE
) -> dict:
    """
    Hitung statistik rata-rata per kabupaten menggunakan reduceRegions.
    Menggunakan ee.Reducer.mean() dan ee.Reducer.stdDev().
    """
    stats = image.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean().combine(
            reducer2=ee.Reducer.stdDev(), sharedInputs=True
        ),
        scale=scale,
        crs="EPSG:4326",
    )
    return stats


# ── Main Ekstraksi ────────────────────────────────────────────
def run_extraction():
    print("=" * 60)
    print("🛰️  GEE Feature Extraction — Sumatera Utara Poverty Estimation")
    print("=" * 60)

    fc, gdf = load_study_area(GEOJSON_PATH)

    # Buat stack semua citra
    print("\n📡 Mengambil data Sentinel-2...")
    s2_indices = get_sentinel2_indices(fc)

    print("🌙 Mengambil data VIIRS Nighttime Lights...")
    ntl_image = get_nighttime_lights(fc)

    print("🗺️  Mengambil data Dynamic World (Land Cover)...")
    lc_image = get_land_cover(fc)

    # Gabungkan semua band dalam 1 image stack
    stacked = s2_indices.addBands([ntl_image, lc_image])

    # Ekstraksi zonal statistik per kabupaten
    print("\n📊 Menghitung statistik per kabupaten...")
    stats_fc = extract_zonal_stats(stacked, fc)

    # Konversi ke DataFrame
    print("💾 Mengunduh hasil ke lokal...")
    features_list = stats_fc.getInfo()["features"]

    rows = []
    for feat in features_list:
        props = feat["properties"]
        # Kolom GeoJSON: nmkab (nama kabupaten UPPERCASE), idkab (kode BPS)
        row = {
            "kab_id": props.get("idkab", ""),
            "kabupaten_geo": props.get("nmkab", ""),
            "mean_ndvi": props.get("ndvi_mean"),
            "std_ndvi": props.get("ndvi_stdDev"),
            "mean_ndbi": props.get("ndbi_mean"),
            "mean_ntl": props.get("ntl_mean_mean"),
            "std_ntl": props.get("ntl_mean_stdDev"),
            "urban_pct": props.get("urban_pct_mean"),
            "agri_pct": props.get("agri_pct_mean"),
            "tree_pct": props.get("tree_pct_mean"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = os.path.join(PROCESSED_DIR, "features_satellite.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Selesai! File tersimpan di: {out_path}")
    print(df.describe())
    return df


if __name__ == "__main__":
    run_extraction()
