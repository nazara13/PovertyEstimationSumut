import streamlit as st
import pandas as pd
import numpy as np
import os
import streamlit.components.v1 as components
import xgboost as xgb
import plotly.express as px

# --- Setup Halama ---
st.set_page_config(
    page_title="Poverty Estimation Dashboard",
    page_icon="bar-chart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS Styling Kustom ---
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1E1E2F;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #f39c12;
    }
    .metric-label {
        font-size: 1rem;
        color: #a0a0b0;
    }
    h1 {
        text-align: center;
        color: #ffffff;
    }
    .subtext {
        text-align: center;
        color: #888888;
        font-size: 0.9rem;
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Judul ---
st.markdown("<h1>Poverty Estimation Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtext'>Estimasi Kemiskinan Berbasis Satelit (Sentinel-2, VIIRS NTL) & Machine Learning (Provinsi Sumatera Utara)</div>",
    unsafe_allow_html=True,
)

# --- Fungsi Load Data ---
def load_data():
    preds_path = os.path.join("output", "predictions.csv")
    feats_path = os.path.join("data", "processed", "merged_features.csv")
    
    if os.path.exists(preds_path):
        preds = pd.read_csv(preds_path)
    else:
        preds = pd.DataFrame()

    if os.path.exists(feats_path):
        feats = pd.read_csv(feats_path)
    else:
        feats = pd.DataFrame()
        
    return preds, feats

preds_df, feats_df = load_data()

# --- Sidebar ---
st.sidebar.header("Filter & Navigasi")
menu = st.sidebar.radio("Pilih Tampilan:", ["Pemetaan Interaktif", "Analisis Data", "Performa Model", "Proyeksi Kemiskinan 2030", "Metodologi & Informasi Data"])

st.sidebar.markdown("---")
st.sidebar.header("Pilih Algoritma ML")
algo_dict = {
    "XGBoost": "pred_xgb",
    "LightGBM": "pred_lgbm",
    "Gradient Boosting": "pred_gbr",
    "Random Forest": "pred_rf"
}
selected_algo = st.sidebar.selectbox("Model yang divisualisasikan:", list(algo_dict.keys()))
pred_col = algo_dict[selected_algo]

st.sidebar.markdown("---")
st.sidebar.info(
    "**Info Data**\n\n"
    "Data BPS: Tahun 2025\n"
    "Satelit: 2023 (Live Google Earth Engine)"
)

# --- Footnote Credit ---
st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #8395a7; font-size: 13px; margin-top: 20px;'>
        <i>Dikembangkan oleh:</i><br>
        <b style='color: #c8d6e5; font-size: 15px;'>Krisman Yusuf Nazara</b><br>
        <span style='font-size: 11px;'>Poverty Spatial Estimation © 2025</span>
    </div>
    """, unsafe_allow_html=True
)

# Jika data kosong
if preds_df.empty:
    st.error("Data prediksi belum tersedia. Jalankan pipeline.py terlebih dahulu!")
    st.stop()

# --- Hitung Metrics ---
mean_aktual = preds_df["poverty_rate"].mean()

# Pastikan kolom prediksi baru ada
if pred_col not in preds_df.columns:
    st.error(f"Kolom {pred_col} belum ada. Silakan run ulang model.py!")
    st.stop()

mean_pred = preds_df[pred_col].mean()
total_kab = len(preds_df)

# --- Baris Metrik ---
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Total Kabupaten/Kota</div><div class='metric-value'>{total_kab}</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Rata-rata Kemiskinan Aktual</div><div class='metric-value'>{mean_aktual:.2f}%</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div class='metric-card'><div class='metric-label'>Prediksi Kemiskinan ({selected_algo})</div><div class='metric-value'>{mean_pred:.2f}%</div></div>", unsafe_allow_html=True)

st.write("---")

# ==========================================
# MENU 1: PETA INTERAKTIF
# ==========================================
if menu == "Pemetaan Interaktif":
    st.subheader("Peta Sebaran Tingkat Kemiskinan")
    map_suffix = pred_col.split("_")[1]
    map_path = os.path.join("output", f"poverty_map_{map_suffix}.html")
    
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()
        # Render HTML string di Streamlit
        components.html(map_html, height=600, scrolling=False)
        
        # --- Tabel Data Pendukung di bawah Peta ---
        st.markdown(f"**Tabel Estimasi Kemiskinan ({selected_algo})**")
        
        err_col = f"error_{map_suffix}"
        if err_col not in preds_df.columns:
            preds_df[err_col] = preds_df[pred_col] - preds_df["poverty_rate"]
        
        display_df = preds_df[["kabupaten", "poverty_rate", pred_col, err_col]].copy()
        display_df.rename(columns={
            "kabupaten": "Kabupaten/Kota",
            "poverty_rate": "Aktual BPS (%)",
            pred_col: f"Prediksi {selected_algo} (%)",
            err_col: "Selisih/Error"
        }, inplace=True)
        
        st.dataframe(
            display_df.style.format({
                "Aktual BPS (%)": "{:.2f}%", 
                f"Prediksi {selected_algo} (%)": "{:.2f}%",
                "Selisih/Error": "{:+.2f}"
            }),
            use_container_width=True
        )
    else:
        st.warning("Peta interaktif HTML tidak ditemukan di folder output.")

# ==========================================
# MENU 2: ANALISIS DATA
# ==========================================
elif menu == "Analisis Data":
    st.subheader("📊 Analisis Data Detail")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 5 Wilayah Paling Miskin (Aktual)**")
        top_poor = preds_df.sort_values(by="poverty_rate", ascending=False).head(5)
        st.dataframe(top_poor[["kabupaten", "poverty_rate", pred_col]].style.format({"poverty_rate": "{:.2f}%", pred_col: "{:.2f}%"}), use_container_width=True)

    with c2:
        st.markdown("**Top 5 Wilayah Paling Sejahtera (Aktual)**")
        top_rich = preds_df.sort_values(by="poverty_rate", ascending=True).head(5)
        st.dataframe(top_rich[["kabupaten", "poverty_rate", pred_col]].style.format({"poverty_rate": "{:.2f}%", pred_col: "{:.2f}%"}), use_container_width=True)

    st.markdown("---")
    st.markdown("**🔍 Data Fitur Ekstraksi Satelit & Geo-Spasial Lengkap**")
    with st.expander("📖 Lihat Kamus Metadata (Penjelasan Kolom Satelit)"):
        st.markdown('''
        | Nama Kolom / Fitur | Sumber Satelit | Deskripsi Teknis Metadata |
        | :--- | :--- | :--- |
        | **mean_ndvi** | *Sentinel-2 10m* | Rata-rata *Vegetation Index*. Angka tinggi (=1) menandakan kerapatan hutan/hutan lindung. Angka rendah (<0.2) menandakan aspal/gedung/jalan. |
        | **mean_ndbi** | *Sentinel-2 10m* | Rata-rata *Built-up Index*. Menandakan seberapa luas tutupan semen, gedung, dan aspal kota. |
        | **mean_ntl** | *VIIRS DNB 500m* | Rata-rata Emisi Cahaya Malam (Radiance). Proksi utama dari **Perputaran Roda Ekonomi**. Semakin terang = Semakin makmur. |
        | **ntl_cv** | *Kalkulasi* | *Coefficient of Variation* dari Cahaya Malam (std_ntl dibagi mean_ntl). Proksi **Ketimpangan Kota**. Angka tinggi = ketimpangan ekonomi ekstrem. |
        | **log_ntl** | *Kalkulasi* | Logaritma natural dari cahaya malam agar linear dengan distribusi kemakmuran/populasi. |
        | **urban_pct** | *Dynamic World* | Persentase tutupan lahan murni untuk area urban/perkotaan dalam skala persen. |
        | **agri_pct** | *Dynamic World* | Persentase tutupan lahan murni untuk sektor pertanian/perkebunan. Indikasi kerentanan *seasonal poverty*. |
        | **tree_pct** | *Dynamic World* | Persentase tutupan pohon / kanopi alami. |
        | **urban_ratio** | *Kalkulasi* | Rasio area lahan beton perkotaan dibanding lahan tertebang (*cleared land*). Menandakan kekuatan sentralisasi bisnis. |
        | **wealth_proxy** | *Kalkulasi* | Formula hibrida buatan peneliti: `(Cahaya Malam * 0.5) + (Gedung * 0.3) - (Vegetasi * 0.2)`. Indeks proksi gabungan untuk **Kekayaan Wilayah**. |
        ''')
    st.dataframe(feats_df, use_container_width=True)

# ==========================================
# MENU 3: PERFORMA MODEL
# ==========================================
elif menu == "Performa Model":
    st.subheader("Verifikasi & Performa Algoritma")
    
    st.markdown("### Komparasi Metrik Model (R², MSE, RMSE, MAE)")
    metrics_path = os.path.join("output", "model_metrics.csv")
    if os.path.exists(metrics_path):
        m_df = pd.read_csv(metrics_path)
        # Menyoroti nilai terbaik (max untuk R2, min untuk Error)
        st.dataframe(
            m_df.style.highlight_max(subset=["R2 Score"], color="green")
            .highlight_min(subset=["MSE (Mean Sq Error)", "RMSE (Root Mean Sq Error)", "MAE (Mean Abs Error)"], color="green"),
            use_container_width=True
        )
    else:
        st.info("File metrik komparasi (model_metrics.csv) belum tersedia.")
        
    st.markdown("---")
    
    t1, t2 = st.tabs(["Akurasi (Aktual vs Prediksi)", "Feature Importance & Korelasi"])
    
    with t1:
        scatter_path = os.path.join("output", "scatter_actual_vs_pred.png")
        if os.path.exists(scatter_path):
            st.image(scatter_path, use_container_width=True)
        else:
            st.warning("Gambar Scatter Plot tidak ditemukan.")
            
    with t2:
        colA, colB = st.columns(2)
        with colA:
            fi_path = os.path.join("output", "feature_importance.png")
            if os.path.exists(fi_path):
                st.markdown("**Variable Paling Berpengaruh (XGBoost)**")
                st.image(fi_path, use_container_width=True)
        
        with colB:
            hm_path = os.path.join("output", "heatmap_correlation.png")
            if os.path.exists(hm_path):
                st.markdown("**Korelasi Fitur Multi-Kolineritas**")
                st.image(hm_path, use_container_width=True)

# ==========================================
# MENU 4: PROYEKSI 2030
# ==========================================
elif menu == "Proyeksi Kemiskinan 2030":
    st.subheader("🔮 Proyeksi Kemiskinan 2030 (Skenario Pembangunan)")
    
    st.markdown("Mensimulasikan tingkat kemiskinan di tahun 2030 berdasarkan **skenario laju pertumbuhan spasial** selama 7 tahun (2023 - 2030).")
    
    # --- Input Slider Skenario ---
    c1, c2, c3 = st.columns(3)
    with c1:
        ntl_growth = st.slider("Laju Elektrivikasi & Ekonomi (Nightlight) per tahun", min_value=0.0, max_value=12.0, value=3.5, step=0.5, format="%.1f%%") / 100
    with c2:
        urban_growth = st.slider("Laju Urbanisasi (Bangunan/NDBI) per tahun", min_value=0.0, max_value=8.0, value=2.0, step=0.5, format="%.1f%%") / 100
    with c3:
        green_drop = st.slider("Laju Deforestasi/Lahan Hijau (NDVI) per tahun", min_value=0.0, max_value=5.0, value=1.0, step=0.5, format="%.1f%%") / 100
        
    st.info("**Formula Simulasi (Compound Growth selama 7 tahun):** P_2030 = P_2023 * (1 + laju_pertumbuhan)^7")
    
    # --- Perhitungan Mulitplier 7 Tahun ---
    years = 7
    ntl_mult = (1 + ntl_growth) ** years
    urban_mult = (1 + urban_growth) ** years
    green_mult = (1 - green_drop) ** years
    
    future_X = feats_df.copy()
    
    # Modifikasi Fitur sesuai masa depan
    if "log_ntl" in future_X.columns: future_X["log_ntl"] *= ntl_mult
    if "mean_ndbi" in future_X.columns: future_X["mean_ndbi"] *= urban_mult
    if "urban_pct" in future_X.columns: future_X["urban_pct"] *= urban_mult
    if "urban_ratio" in future_X.columns: future_X["urban_ratio"] *= urban_mult
    if "mean_ndvi" in future_X.columns: future_X["mean_ndvi"] *= green_mult
    if "tree_pct" in future_X.columns: future_X["tree_pct"] *= green_mult
    if "wealth_proxy" in future_X.columns: future_X["wealth_proxy"] *= (ntl_mult * green_mult)
    
    # --- Eksekusi Prediksi Dinamis Mengikuti Algoritma Terpilih ---
    import joblib
    
    feature_cols = ["mean_ndvi","std_ndvi","mean_ndbi","log_ntl","std_ntl","ntl_cv","urban_pct","agri_pct","tree_pct","urban_ratio","agri_ratio","wealth_proxy"]
    available_cols = [c for c in feature_cols if c in future_X.columns]
    
    try:
        if selected_algo == "XGBoost":
            xgb_model_path = os.path.join("output", "model_xgb.json")
            booster = xgb.Booster()
            booster.load_model(xgb_model_path)
            dmatrix = xgb.DMatrix(future_X[available_cols].values, feature_names=available_cols)
            preds_2030 = booster.predict(dmatrix)
        else:
            # Memuat model Scikit-Learn / LightGBM dari format .pkl
            if selected_algo == "LightGBM":
                model_path = os.path.join("output", "model_lgbm.pkl")
            elif selected_algo == "Gradient Boosting":
                model_path = os.path.join("output", "model_gbr.pkl")
            else:
                model_path = os.path.join("output", "model_rf.pkl")
                
            model = joblib.load(model_path)
            preds_2030 = model.predict(future_X[available_cols].values)

        sim_df = pd.DataFrame({
            "Kabupaten/Kota": preds_df["kabupaten"],
            "Kemiskinan 2025 (Aktual)": preds_df["poverty_rate"],
            "Proyeksi 2030 (Skenario)": preds_2030
        })
        sim_df["Delta (%)"] = sim_df["Proyeksi 2030 (Skenario)"] - sim_df["Kemiskinan 2025 (Aktual)"]
        
        m_25 = sim_df["Kemiskinan 2025 (Aktual)"].mean()
        m_30 = sim_df["Proyeksi 2030 (Skenario)"].mean()
        delta_m = m_30 - m_25
        
        st.markdown("---")
        mcol1, mcol2 = st.columns(2)
        with mcol1:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Rata-rata 2025</div><div class='metric-value'>{m_25:.2f}%</div></div>", unsafe_allow_html=True)
        with mcol2:
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Proyeksi 2030</div><div class='metric-value'>{m_30:.2f}%</div><div style='color:#2ecc71; font-weight:bold;'>↓ {abs(delta_m):.2f}% Penurunan Kemiskinan</div></div>", unsafe_allow_html=True)
            
        st.write("")
        
        # Plot Delta horizontal
        sim_df_sorted = sim_df.sort_values(by="Delta (%)")
        fig = px.bar(sim_df_sorted, x="Delta (%)", y="Kabupaten/Kota", orientation='h', height=700,
                     title="Proyeksi Penurunan Kemiskinan per Daerah (2025 ➔ 2030)",
                     color="Delta (%)", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("**Perbandingan Angka Absolut (2025 vs 2030)**")
        st.dataframe(sim_df.style.format({
            "Kemiskinan 2025 (Aktual)": "{:.2f}%", 
            "Proyeksi 2030 (Skenario)": "{:.2f}%",
            "Delta (%)": "{:.2f}%"
        }), use_container_width=True)
    except Exception as e:
        st.error(f"Gagal memuat model algoritma ({selected_algo}). Harap pastikan model .pkl atau .json tersedia di output/. Error: {e}")

# ==========================================
# MENU 5: METODOLOGI & DATA
# ==========================================
elif menu == "Metodologi & Informasi Data":
    st.subheader("Metodologi & Sumber Data")
    
    st.markdown('''
    Dashboard ini dibangun menggunakan pendekatan **Multi-source Remote Sensing dan Supervised Machine Learning**. 
    Tujuannya adalah untuk mengestimasi tingkat kemiskinan dengan menganalisa pola pembangunan, vegetasi, dan aktivitas malam secara spasial dari perspektif luar angkasa.
    ''')

    colA, colB = st.columns(2)
    
    with colA:
        st.info("**1. Data Citra Satelit (Prediktor)**")
        st.markdown('''
        Data spasial aktual diunduh langsung dari *Cloud* menggunakan **Google Earth Engine (GEE)**:
        - **Sentinel-2 (ESA):** Koleksi data `COPERNICUS/S2_SR_HARMONIZED`. Resolusi 10m. Digunakan merumuskan *Normalized Difference Vegetation Index (NDVI)* [Vegetasi/Rural], dan *Normalized Difference Built-up Index (NDBI)* [Pemukiman Bebas/Urban].
        - **VIIRS / DNB (NOAA):** Koleksi data `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG`. Resolusi 500m. Menangkap emisi *Nighttime Lights (NTL)*. Cahaya malam yang lebih terang merupakan proksi kuat dari masifnya perputaran ekonomi yang linier dengan rendahnya kemiskinan.
        - **Dynamic World (Google):** Koleksi data `GOOGLE/DYNAMICWORLD/V1`. Menghitung probabilitas per-piksel untuk memisahkan persentase lahan pertanian (*agri_pct*) dari lahan beton perkotaan (*urban_pct*).
        ''')
        
    with colB:
        st.success("**2. Data Ground Truth (Target Label)**")
        st.markdown('''
        - **BPS Provinsi Sumatera Utara (Survei SUSENAS 2025):** 
          Label aktual yang diukur secara manual/survei di lapangan oleh Badan Pusat Statistik. Terdiri atas target pembelajaran AI:
            - **[P0]** Persentase Penduduk Miskin (Poverty Rate)
            - **[P1]** Indeks Kedalaman Kemiskinan
            - **[P2]** Indeks Keparahan Kemiskinan
            
        *Catatan: Parameter BPS 2025 ini digunakan sebagai acuan validasi absolut (Y-Target) di seluruh arsitektur Machine Learning kami.*
        ''')

    st.markdown("---")
    st.warning("**3. Metode Perhitungan & Machine Learning**")
    st.markdown('''
    1. **Ekstraksi Zonal Statistik (GEE):** Untuk setiap area administratif Kabupaten (Poligon *GeoJSON*), sistem mencari agregasi rata-rata (*mean*) dan kelainan/simpangan baku (*stdDev*) dari ratusan ribu piksel data prediktor (satelit) di atas.
    2. **Algoritma Machine Learning (4 Komparasi):** Data diekstraksi dan divalidasi silang menggunakan *5-Fold Cross Validation* secara *supervised learning*. Untuk mengetahui model yang paling handal, digunakan 4 regresi terkemuka:
        - **Random Forest Regressor:** Base model yang membangun ratusan pohon keputusan *randomized* untuk menangkap tren global.
        - **Gradient Boosting Regressor (GBR):** Membangun pohon keputusan secara berurutan, mengatasi kelemahan error secara iteratif.
        - **eXtreme Gradient Boosting (XGBoost):** Algoritma paling *robust*. Mampu menangani outlier dari data raster beresolusi tinggi. Digunakan sebagai acuan *Colormap Visualizer*.
        - **LightGBM:** Berbasis *histogram-based gradient boosting* yang luar biasa cepat dan efisien terhadap data bergeometri rumit.
    3. **Skenario Proyeksi (Forecasting 2030):** Memanfaatkan metodologi *Scenario-based Policy Making*. Fitur masa depan disintesis menggunakan rumus **Compound Annual Growth Rate (CAGR)**: $V_n = V_0 (1 + r)^t$ di mana laju urbanisasi, pertumbuhan malam (NTL), dan deforestasi dikalkulasi, untuk kemudian '*diinjeksi*' ulang ke dalam model regresi Machine Learning (XGBoost) guna memprediksi potret kemiskinan di 2030.
    ''')
