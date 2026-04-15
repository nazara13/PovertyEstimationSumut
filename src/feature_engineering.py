"""
================================================================
feature_engineering.py
Penggabungan Fitur Satelit + Data Ground Truth BPS
Wilayah: Provinsi Sumatera Utara
================================================================

Input:
  - data/processed/features_satellite.csv   <- hasil GEE
  - data/raw/  (5 file Excel BPS 2025)

Output:
  - data/processed/merged_features.csv      <- dataset siap training
"""

import pandas as pd
import numpy as np
import os

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"

# ── Mapping nama file BPS ke kolom output ─────────────────────
BPS_FILES = {
    "poverty_rate": "Persentase Penduduk Miskin Menurut Kabupaten_Kota di Provinsi Sumatera Utara, 2025.xlsx",
    "poverty_depth": "Indeks Kedalaman Kemiskinan (P1) Menurut Kabupaten_Kota, 2025.xlsx",
    "poverty_severity": "Indeks Keparahan Kemiskinan (P2) Menurut Kabupaten_Kota, 2025.xlsx",
    "jumlah_miskin": "Jumlah Penduduk Miskin Menurut Kabupaten_Kota(000) di Provinsi Sumatera Utara, 2025.xlsx",
    "garis_kemiskinan": "Garis Kemiskinan Menurut Kabupaten_Kota di Provinsi Sumatera Utara, 2025.xlsx",
}

# ── Manual mapping untuk nama yang tidak match sempurna ──────
# Key: nama di BPS, Value: nama di GeoJSON (kolom nmkab, UPPERCASE)
MANUAL_MAPPING = {
    "Toba": "TOBA SAMOSIR",  # BPS pakai "Toba", GeoJSON pakai "TOBA SAMOSIR"
}


# ── 1. Load Semua File BPS ────────────────────────────────────
def load_bps_data() -> pd.DataFrame:
    """
    Membaca 5 file Excel BPS. Setiap file punya format:
      Row 1: header "Kabupaten Kota"
      Row 2: judul panjang
      Row 3: tahun
      Row 4..37: data (Row 4 = Sumatera Utara total, Row 5..37 = per kabupaten)
    """
    dfs = []
    for col_name, fname in BPS_FILES.items():
        path = os.path.join(RAW_DIR, fname)
        if not os.path.exists(path):
            print(f"  [!!] File tidak ditemukan: {fname}")
            continue

        # Baca mulai row ke-4 (index 3), skip 3 baris header BPS
        df = pd.read_excel(path, header=None, skiprows=3, usecols=[0, 1])
        df.columns = ["kabupaten", col_name]

        # Buang baris total provinsi (Sumatera Utara)
        df = df[df["kabupaten"] != "Sumatera Utara"].copy()
        df = df.dropna(subset=["kabupaten"]).reset_index(drop=True)
        df["kabupaten"] = df["kabupaten"].astype(str).str.strip()
        dfs.append(df.set_index("kabupaten"))
        print(f"  [OK] {col_name}: {len(df)} baris dari '{fname}'")

    # Gabungkan semua kolom BPS berdasarkan nama kabupaten
    bps_merged = pd.concat(dfs, axis=1).reset_index()
    bps_merged.rename(columns={"kabupaten": "kabupaten_bps"}, inplace=True)
    print(
        f"\n[OK] Data BPS tergabung: {bps_merged.shape[0]} kabupaten x {bps_merged.shape[1]} kolom"
    )
    return bps_merged


# ── 2. Load Fitur Satelit ─────────────────────────────────────
def load_satellite_features() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DIR, "features_satellite.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"[!!] File fitur satelit belum ada: {path}. Harap jalankan gee_extractor.py terlebih dahulu untuk menarik data asli satelit!")
    df = pd.read_csv(path)
    print(f"[OK] Fitur satelit: {df.shape[0]} kabupaten x {df.shape[1]} kolom")
    return df


# ── 3. Merge Satelit + BPS (berdasarkan nmkab GeoJSON) ───────
def merge_datasets(sat_df: pd.DataFrame, bps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join berdasarkan nama kabupaten:
      - sat_df kolom: 'kabupaten_geo' (dari GeoJSON, UPPERCASE)
      - bps_df kolom: 'kabupaten_bps' (dari BPS, Title Case)

    Pendekatan: normalisasi ke lowercase, + manual mapping untuk 'TOBA SAMOSIR'
    """
    import difflib

    # Tambah kolom kunci untuk join
    sat_df = sat_df.copy()
    bps_df = bps_df.copy()

    sat_df["kunci"] = sat_df["kabupaten_geo"].str.lower().str.strip()

    # Terapkan manual mapping ke BPS
    def bps_to_geo_key(name):
        if name in MANUAL_MAPPING:
            return MANUAL_MAPPING[name].lower()
        return name.lower().strip()

    bps_df["kunci"] = bps_df["kabupaten_bps"].apply(bps_to_geo_key)

    # Fuzzy match untuk sisa nama yang belum exact
    bps_keys = bps_df["kunci"].tolist()
    bps_orig = bps_df["kabupaten_bps"].tolist()

    def find_bps_key(geo_key):
        if geo_key in bps_keys:
            return geo_key
        matches = difflib.get_close_matches(geo_key, bps_keys, n=1, cutoff=0.6)
        return matches[0] if matches else None

    sat_df["bps_kunci"] = sat_df["kunci"].apply(find_bps_key)

    merged = sat_df.merge(
        bps_df.rename(columns={"kunci": "bps_kunci"}), on="bps_kunci", how="left"
    )

    n_matched = merged["poverty_rate"].notna().sum()
    print(f"[OK] Join berhasil: {n_matched}/{len(sat_df)} kabupaten cocok")
    if n_matched < len(sat_df):
        tidak_match = merged[merged["poverty_rate"].isna()]["kabupaten_geo"].tolist()
        print(f"[!!] Tidak match: {tidak_match}")

    return merged


# ── 4. Feature Engineering ────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Buat fitur turunan dari kolom satelit dasar."""
    df = df.copy()

    # Log-transform NTL (distribusi sangat skewed ke kanan)
    df["log_ntl"] = np.log1p(df["mean_ntl"].clip(lower=0))

    # Koefisien variasi NTL — proksi ketimpangan intra-kabupaten
    df["ntl_cv"] = df["std_ntl"] / (df["mean_ntl"].clip(lower=0.01))

    # Rasio area urban terhadap total area terklasifikasi
    total_pct = (df["urban_pct"] + df["agri_pct"] + df["tree_pct"]).replace(0, np.nan)
    df["urban_ratio"] = df["urban_pct"] / total_pct
    df["agri_ratio"] = df["agri_pct"] / total_pct

    # Indeks kesejahteraan proxy:
    #   NTL tinggi + NDBI tinggi + NDVI rendah → cenderung lebih kaya
    df["wealth_proxy"] = (
        df["log_ntl"] * 0.5 + df["mean_ndbi"] * 0.3 - df["mean_ndvi"] * 0.2
    )

    print(f"[OK] Feature engineering selesai — {df.shape[1]} kolom total")
    return df


# ── 5. Clean Missing Values ───────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    missing_before = df[num_cols].isnull().sum().sum()
    if missing_before > 0:
        print(f"[!!] {missing_before} missing values — imputasi median")
        for col in num_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    else:
        print("[OK] Tidak ada missing values")
    return df


# ── Main ──────────────────────────────────────────────────────
def run_feature_engineering():
    print("=" * 60)
    print("FEATURE ENGINEERING — Sumatera Utara Poverty Estimation")
    print("=" * 60)

    print("\n[1/4] Loading data BPS...")
    bps_df = load_bps_data()

    print("\n[2/4] Loading fitur satelit...")
    sat_df = load_satellite_features()

    print("\n[3/4] Merge & feature engineering...")
    merged = merge_datasets(sat_df, bps_df)
    merged = engineer_features(merged)
    merged = clean_data(merged)

    # Kolom final untuk training
    FEATURE_COLS = [
        "mean_ndvi",
        "std_ndvi",
        "mean_ndbi",
        "log_ntl",
        "std_ntl",
        "ntl_cv",
        "urban_pct",
        "agri_pct",
        "tree_pct",
        "urban_ratio",
        "agri_ratio",
        "wealth_proxy",
    ]
    TARGET_COLS = [
        "poverty_rate",
        "poverty_depth",
        "poverty_severity",
        "jumlah_miskin",
        "garis_kemiskinan",
    ]
    META_COLS = ["kabupaten_geo", "kabupaten_bps", "kab_id"]

    keep = [c for c in META_COLS + FEATURE_COLS + TARGET_COLS if c in merged.columns]
    out_df = merged[keep].copy()

    print("\n[4/4] Menyimpan hasil...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out_path = os.path.join(PROCESSED_DIR, "merged_features.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Tersimpan: {out_path}")
    print(f"     Shape: {out_df.shape[0]} baris x {out_df.shape[1]} kolom\n")
    print(
        out_df[["kabupaten_bps", "poverty_rate", "log_ntl", "mean_ndvi"]].to_string(
            index=False
        )
    )
    return out_df


if __name__ == "__main__":
    run_feature_engineering()
