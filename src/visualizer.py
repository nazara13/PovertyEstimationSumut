"""
================================================================
visualizer.py
Visualisasi Hasil Estimasi Kemiskinan — Peta Interaktif & Grafik
Wilayah: Provinsi Sumatera Utara
================================================================

Output:
  - output/poverty_map.html         ← Peta interaktif Folium
  - output/scatter_actual_vs_pred.png
  - output/feature_importance.png
  - output/heatmap_correlation.png
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import Fullscreen, MiniMap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import os, difflib

SPATIAL_DIR = "data/spatial"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style: dark premium theme
plt.rcParams.update(
    {
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#a0a0b0",
        "ytick.color": "#a0a0b0",
        "axes.edgecolor": "#2a2a4a",
        "grid.color": "#2a2a4a",
        "axes.grid": True,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    }
)

POVERTY_COLORMAP = [
    "#2ecc71",  # < 5%   hijau (sangat rendah)
    "#f1c40f",  # 5–10%  kuning
    "#e67e22",  # 10–15% oranye
    "#e74c3c",  # 15–20% merah
    "#8e44ad",  # > 20%  ungu (sangat tinggi)
]


# ── 1. Load Data ──────────────────────────────────────────────
def load_all_data():
    gdf = gpd.read_file(os.path.join(SPATIAL_DIR, "kabupaten_sumut.geojson"))
    preds = pd.read_csv(os.path.join(OUTPUT_DIR, "predictions.csv"))
    feats = pd.read_csv(os.path.join(PROCESSED_DIR, "merged_features.csv"))
    return gdf, preds, feats


# ── 2. Join GeoDataFrame dengan Prediksi ─────────────────────
def join_geodata(gdf: gpd.GeoDataFrame, preds: pd.DataFrame) -> gpd.GeoDataFrame:
    """Join berdasarkan nama kabupaten dengan fuzzy matching."""

    def best_match(name, choices, cutoff=0.6):
        name_norm = name.lower().strip()
        matches = difflib.get_close_matches(
            name_norm, [c.lower().strip() for c in choices], n=1, cutoff=cutoff
        )
        if matches:
            idx = [c.lower().strip() for c in choices].index(matches[0])
            return choices[idx]
        return None

    # Cari kolom nama kabupaten di GeoJSON
    name_col = next(
        (
            c
            for c in gdf.columns
            if "nama" in c.lower() or "name" in c.lower() or "kab" in c.lower()
        ),
        gdf.columns[0],
    )
    print(f"   Kolom nama kabupaten GeoJSON: '{name_col}'")

    gdf["match_key"] = gdf[name_col].apply(
        lambda x: best_match(str(x), preds["kabupaten"].tolist())
    )
    merged_gdf = gdf.merge(preds, left_on="match_key", right_on="kabupaten", how="left")
    n_matched = merged_gdf["poverty_rate"].notna().sum()
    print(f"✅ {n_matched}/{len(gdf)} kabupaten berhasil di-join dengan prediksi")
    return merged_gdf


# ── 3. Peta Interaktif Folium ────────────────────────────────
def create_interactive_map(gdf_merged: gpd.GeoDataFrame, col_of_interest: str, label_name: str, out_suffix: str):
    print(f"   Membuat peta interaktif untuk {label_name}...")

    gdf_4326 = gdf_merged.to_crs(epsg=4326)
    center = [gdf_4326.geometry.centroid.y.mean(), gdf_4326.geometry.centroid.x.mean()]

    m = folium.Map(
        location=center,
        zoom_start=7,
        tiles=None,
        width="100%",
        height="100%",
    )

    # Tambahkan opsi Base Maps
    folium.TileLayer("CartoDB dark_matter", name="🌃 Dark Mode (Default)").add_to(m)
    
    # NASA Earth at Night (Black Marble / CityLights) max zoom level 8
    folium.TileLayer(
        tiles="https://map1.vis.earthdata.nasa.gov/wmts-webmerc/VIIRS_CityLights_2012/default/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg",
        attr="NASA GIBS",
        name="💡 Citra Lampu Malam (NASA)",
        max_zoom=8
    ).add_to(m)
    
    # ESRI Satellite Imagery (Untuk melihat Vegetasi)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="🌳 Citra Satelit Nyata (Esri)",
    ).add_to(m)

    folium.TileLayer("CartoDB positron", name="🏙️ Light Mode").add_to(m)
    folium.TileLayer("OpenStreetMap", name="🗺️ Peta Standar / Jalan").add_to(m)

    # Gunakan skala absolut kemiskinan yang relevan untuk Indonesia
    vmin = 0
    vmax = 25
    colormap = folium.LinearColormap(
        colors=POVERTY_COLORMAP,
        index=[0, 5, 10, 15, 25],
        vmin=vmin,
        vmax=vmax,
        caption=f"Estimasi % Penduduk Miskin ({label_name})",
    )

    def style_function(feature):
        val = feature["properties"].get(col_of_interest)
        color = colormap(val) if val is not None else "#555555"
        return {
            "fillColor": color,
            "color": "#ffffff",
            "weight": 0.8,
            "fillOpacity": 0.75,
        }

    def highlight_function(feature):
        return {"color": "#ffffff", "weight": 2, "fillOpacity": 0.9}

    def tooltip_fields():
        name_col = next(
            (
                c
                for c in gdf_4326.columns
                if "nama" in c.lower() or "name" in c.lower() or "kab" in c.lower()
            ),
            "kabupaten",
        )
        return [name_col, "poverty_rate", col_of_interest]

    tooltip_cols = tooltip_fields()
    tooltip_aliases = [
        "Kabupaten/Kota",
        "Aktual (%)",
        f"Prediksi {label_name} (%)",
    ]

    folium.GeoJson(
        gdf_4326.__geo_interface__,
        name="Data Spatial Kabupaten",
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=tooltip_cols[: len(tooltip_aliases)],
            aliases=tooltip_aliases[: len(tooltip_cols)],
            localize=True,
            sticky=True,
            labels=True,
            style=(
                "background-color: #1a1a2e; color: #e0e0e0; "
                "font-family: Arial; font-size: 13px; padding: 10px;"
            ),
        ),
    ).add_to(m)

    colormap.add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    
    # 🌟 Aktifkan panel pilihan Base Map / Layer
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    # Judul peta
    title_html = f"""
    <div style="position: fixed; top: 15px; left: 50%; transform: translateX(-50%);
                 z-index: 1000; background: rgba(26,26,46,0.9);
                 padding: 12px 24px; border-radius: 8px;
                 border: 1px solid #4a4a8a; font-family: Arial, sans-serif;">
        <h3 style="margin:0; color:#e0e0e0; font-size:16px; text-align:center;">
            Peta Estimasi Kemiskinan — Sumatera Utara 2025 ({label_name})
        </h3>
        <p style="margin:4px 0 0; color:#a0a0c0; font-size:12px; text-align:center;">
            Sumber: Sentinel-2 · VIIRS NTL · BPS 2025
        </p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    out_path = os.path.join(OUTPUT_DIR, f"poverty_map_{out_suffix}.html")
    m.save(out_path)
    print(f"Peta tersimpan: {out_path}")


# ── 4. Scatter: Aktual vs Prediksi ───────────────────────────
def plot_actual_vs_predicted(preds: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    fig.suptitle(
        "Validasi Model: Aktual vs Prediksi Kemiskinan",
        fontsize=16,
        fontweight="bold",
        color="#e0e0e0",
        y=1.02,
    )

    models_conf = [
        ("pred_xgb", "XGBoost", "#e74c3c"),
        ("pred_rf", "Random Forest", "#3498db"),
        ("pred_lgbm", "LightGBM", "#9b59b6"),
        ("pred_gbr", "Gradient Boosting", "#2ecc71"),
    ]

    for ax, (col, label, color) in zip(axes, models_conf):
        if col not in preds.columns:
            continue
            
        x, y = preds["poverty_rate"], preds[col]
        r2 = 1 - np.sum((y - x) ** 2) / np.sum((x - x.mean()) ** 2)

        # Scatter
        ax.scatter(
            x, y, color=color, alpha=0.8, s=80, edgecolors="white", linewidths=0.5
        )

        # Trendline
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_line,
            p(x_line),
            color="#f39c12",
            linewidth=2,
            linestyle="--",
            label="Trend",
        )

        # Garis sempurna
        ax.plot(
            [x.min(), x.max()],
            [x.min(), x.max()],
            color="#ecf0f1",
            linewidth=1.5,
            linestyle=":",
            label="Ideal (y=x)",
        )

        # Label kabupaten outlier
        errors = np.abs(y - x)
        top_outliers = errors.nlargest(3).index
        for i in top_outliers:
            kab = preds.loc[i, "kabupaten"]
            ax.annotate(
                kab,
                (x[i], y[i]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
                color="#ecf0f1",
                alpha=0.9,
            )

        ax.set_title(f"{label}\n$R^2$ = {r2:.4f}", color="#e0e0e0", fontsize=12)
        ax.set_xlabel("% Penduduk Miskin (Aktual — BPS)", color="#a0a0b0")
        ax.set_ylabel("% Penduduk Miskin (Prediksi)", color="#a0a0b0")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "scatter_actual_vs_pred.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"✅ Scatter plot: {out}")


# ── 5. Feature Importance Bar Chart ──────────────────────────
def plot_feature_importance():
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance_xgboost.csv")
    if not os.path.exists(fi_path):
        print("⚠️  File feature importance tidak ditemukan, lewati plot.")
        return

    fi = pd.read_csv(fi_path).sort_values("importance", ascending=True)
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(fi)))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        fi["feature"], fi["importance"], color=colors, edgecolor="none", height=0.6
    )
    ax.set_xlabel("Importance Score", color="#a0a0b0")
    ax.set_title(
        "Feature Importance — XGBoost\n(Estimasi Kemiskinan Sumatera Utara)",
        color="#e0e0e0",
        fontsize=13,
    )

    for bar, val in zip(bars, fi["importance"]):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#e0e0e0",
        )

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"✅ Feature importance plot: {out}")


# ── 6. Heatmap Korelasi ───────────────────────────────────────
def plot_correlation_heatmap(feats: pd.DataFrame):
    num_df = feats.select_dtypes(include=[np.number])
    corr = num_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 9, "color": "#1a1a2e"},
        ax=ax,
    )
    ax.set_title(
        "Heatmap Korelasi Fitur × Target", color="#e0e0e0", fontsize=14, pad=15
    )
    ax.tick_params(colors="#a0a0b0", labelsize=9)

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "heatmap_correlation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"✅ Heatmap korelasi: {out}")


# ── Main ──────────────────────────────────────────────────────
def run_visualization():
    print("=" * 60)
    print("🎨 Visualisasi — Sumatera Utara Poverty Estimation")
    print("=" * 60)

    gdf, preds, feats = load_all_data()
    gdf_merged = join_geodata(gdf, preds)

    # Buat peta untuk masing-masing model
    create_interactive_map(gdf_merged, "pred_xgb", "XGBoost", "xgb")
    create_interactive_map(gdf_merged, "pred_lgbm", "LightGBM", "lgbm")
    create_interactive_map(gdf_merged, "pred_gbr", "Gradient Boosting", "gbr")
    create_interactive_map(gdf_merged, "pred_rf", "Random Forest", "rf")

    plot_actual_vs_predicted(preds)
    plot_feature_importance()
    plot_correlation_heatmap(feats)

    print("\nSemua visualisasi selesai.")


if __name__ == "__main__":
    run_visualization()
