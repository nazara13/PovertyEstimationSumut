"""
================================================================
pipeline.py
Menjalankan seluruh pipeline dari satu titik
================================================================
Urutan:
  1. Feature Engineering (merge fitur satelit + BPS)
  2. Model Training (RF + XGBoost)
  3. Visualisasi (Peta + Grafik)

Jalankan: python src/pipeline.py
Catatan: Pastikan gee_extractor.py sudah dijalankan terlebih dahulu!
"""

import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from feature_engineering import run_feature_engineering
from model import run_training
from visualizer import run_visualization


def run_full_pipeline():
    print("\n" + "=" * 60)
    print("🚀 FULL PIPELINE — Poverty Estimation Sumatera Utara")
    print("=" * 60 + "\n")

    print("STEP 1/3: Feature Engineering\n" + "-" * 40)
    run_feature_engineering()

    print("\n\nSTEP 2/3: Model Training\n" + "-" * 40)
    run_training()

    print("\n\nSTEP 3/3: Visualization\n" + "-" * 40)
    run_visualization()

    print("\n" + "=" * 60)
    print("🎉 PIPELINE SELESAI!")
    print("   → Peta interaktif: output/poverty_map.html")
    print("   → Prediksi       : output/predictions.csv")
    print("   → Model RF       : output/model_rf.pkl")
    print("   → Model XGBoost  : output/model_xgb.json")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()
