# 🛰️ Poverty Estimation — Sumatera Utara
## Estimasi Kemiskinan Berbasis Citra Satelit & Machine Learning

**Pendekatan:** Multi-source Remote Sensing + Supervised Machine Learning  
**Wilayah Studi:** Provinsi Sumatera Utara (33 Kabupaten/Kota)  
**Target Variabel:** Persentase Penduduk Miskin (P0) per Kabupaten/Kota  

---

## 📁 Struktur Folder

```
PovertyEstimation/
├── data/
│   ├── raw/                  # Data mentah BPS (SUSENAS, PODES) - format CSV/Excel
│   ├── processed/            # Data fitur hasil ekstraksi GEE (sudah bersih)
│   └── spatial/
│       └── kabupaten_sumut.geojson   # Batas wilayah kabupaten Sumatera Utara
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb # EDA - distribusi kemiskinan
│   ├── 02_feature_engineering.ipynb  # Penggabungan fitur satelit & ground truth
│   └── 03_model_training.ipynb       # Training & evaluasi model
├── src/
│   ├── gee_extractor.py      # Script ekstraksi fitur dari Google Earth Engine
│   ├── feature_engineering.py# Kalkulasi NDVI, NDBI, NTL, GLCM
│   ├── model.py              # Training Random Forest & XGBoost
│   └── visualizer.py         # Plotting peta kemiskinan & evaluasi model
├── output/
│   ├── poverty_map.html      # Peta interaktif kemiskinan (Folium)
│   ├── model_rf.pkl          # Model Random Forest tersimpan
│   └── predictions.csv       # Prediksi tingkat kemiskinan per kabupaten
├── requirements.txt
└── README.md
```

---

## 🔬 Metodologi

### Sumber Data Satelit
| Data | Sumber | Resolusi | Kegunaan |
|---|---|---|---|
| Multispektral | Sentinel-2 (ESA, 2023) | 10m | NDVI, NDBI, Tutupan Lahan |
| Cahaya Malam | VIIRS/DNB (NOAA, 2023) | 500m | Proksi Aktivitas Ekonomi |
| Tutupan Lahan | Dynamic World (Google) | 10m | Klasifikasi perkebunan vs pemukiman |

### Ground Truth
- **SUSENAS 2023** (BPS): Persentase Penduduk Miskin per Kabupaten/Kota

### Feature Engineering (per Kabupaten)
1. `mean_ndvi`  — Rata-rata indeks vegetasi
2. `mean_ndbi`  — Rata-rata indeks bangunan
3. `mean_ntl`   — Rata-rata cahaya malam (log-transformed)
4. `std_ntl`    — Simpangan baku cahaya malam (proksi ketimpangan)
5. `urban_pct`  — Persentase area terbangun
6. `agri_pct`   — Persentase area pertanian/perkebunan
7. `glcm_contrast` — Kontras tekstur (proksi kekumuhan)

### Model yang Digunakan
- **Random Forest Regressor** (baseline)
- **XGBoost Regressor** (utama)
- **Evaluasi:** R², RMSE, MAPE (5-fold Cross Validation)

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Autentikasi Google Earth Engine
```bash
earthengine authenticate
```

### 3. Ekstraksi Fitur Satelit (GEE)
```bash
python src/gee_extractor.py
```

### 4. Training Model
```bash
python src/model.py
```

### 5. Eksplorasi via Notebook
```bash
jupyter lab notebooks/
```

---

## 📌 Catatan Khusus (Sumatera Utara)
> ⚠️ **Bias Perkebunan:** Area perkebunan sawit & karet yang luas dapat mengacaukan model.  
> Pastikan fitur `agri_pct` digunakan untuk membedakan kawasan industri perkebunan dari kawasan permukiman.

---

## 👤 Peneliti
- **Institusi:** —
- **Tahun:** 2026
