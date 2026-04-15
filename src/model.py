"""
================================================================
model.py
Training, Evaluasi, dan Penyimpanan Model Estimasi Kemiskinan
Wilayah: Provinsi Sumatera Utara
================================================================

Menjalankan 2 model:
  1. Random Forest  (baseline, interpretable)
  2. XGBoost        (utama, lebih akurat)

Evaluasi: R², RMSE, MAE, MAPE — dengan 5-fold Cross Validation
Output  : model tersimpan di output/ , prediksi di output/predictions.csv
"""

import pandas as pd
import numpy as np
import os, joblib, warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
import xgboost as xgb
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
TARGET_COL = "poverty_rate"
RANDOM_STATE = 42


# ── Load Data ─────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    path = os.path.join(PROCESSED_DIR, "merged_features.csv")
    df = pd.read_csv(path)

    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing = set(FEATURE_COLS) - set(available_features)
    if missing:
        print(f"[!!] Fitur tidak ditemukan (akan dilewati): {missing}")

    X = df[available_features].values
    y = df[TARGET_COL].values
    print(f"[OK] Dataset: {X.shape[0]} sampel x {X.shape[1]} fitur")
    return df, X, y


all_metrics = []

# ── Evaluasi Cross Validation ─────────────────────────────────
def evaluate_model(model, X: np.ndarray, y: np.ndarray, model_name: str) -> dict:
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        model,
        X,
        y,
        cv=kf,
        scoring=["r2", "neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error"],
        return_train_score=True,
    )

    metrics = {
        "Model Algorithm": model_name,
        "R2 Score": round(cv_results["test_r2"].mean(), 4),
        "RMSE (Root Mean Sq Error)": round(-cv_results["test_neg_root_mean_squared_error"].mean(), 4),
        "MSE (Mean Sq Error)": round(-cv_results["test_neg_mean_squared_error"].mean(), 4),
        "MAE (Mean Abs Error)": round(-cv_results["test_neg_mean_absolute_error"].mean(), 4),
        "Train R2 (Overfit Check)": round(cv_results["train_r2"].mean(), 4),
    }
    
    all_metrics.append(metrics)

    print(f"\n[EVAL] [{model_name}] 5-Fold CV Results:")
    print(f"   R2   : {metrics['R2 Score']:.4f}")
    print(f"   MSE  : {metrics['MSE (Mean Sq Error)']:.4f}")
    print(f"   RMSE : {metrics['RMSE (Root Mean Sq Error)']:.4f}")
    print(f"   MAE  : {metrics['MAE (Mean Abs Error)']:.4f}")

    return metrics


# ── Model 1: Random Forest ────────────────────────────────────
def train_random_forest(X: np.ndarray, y: np.ndarray) -> RandomForestRegressor:
    print("\n" + "─" * 50)
    print("[RF] Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    evaluate_model(rf, X, y, "Random Forest")
    rf.fit(X, y)  # Final fit on all data
    joblib.dump(rf, os.path.join(OUTPUT_DIR, "model_rf.pkl"))
    print("   -> Model tersimpan: output/model_rf.pkl")
    return rf


# ── Model 2: XGBoost ─────────────────────────────────────────
def train_xgboost(X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
    print("\n" + "-" * 50)
    print("[XGB] Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
    )
    evaluate_model(xgb_model, X, y, "XGBoost")
    xgb_model.fit(X, y)
    xgb_model.save_model(os.path.join(OUTPUT_DIR, "model_xgb.json"))
    return xgb_model

# ── Model 3: LightGBM ────────────────────────────────────────
def train_lightgbm(X: np.ndarray, y: np.ndarray) -> LGBMRegressor:
    print("\n" + "-" * 50)
    print("[LGBM] Training LightGBM...")
    lgbm_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
    )
    evaluate_model(lgbm_model, X, y, "LightGBM")
    lgbm_model.fit(X, y)
    joblib.dump(lgbm_model, os.path.join(OUTPUT_DIR, "model_lgbm.pkl"))
    return lgbm_model

# ── Model 4: Gradient Boosting ───────────────────────────────
def train_gradient_boosting(X: np.ndarray, y: np.ndarray) -> GradientBoostingRegressor:
    print("\n" + "-" * 50)
    print("[GBR] Training Gradient Boosting...")
    gbr_model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        random_state=RANDOM_STATE,
    )
    evaluate_model(gbr_model, X, y, "Gradient Boosting")
    gbr_model.fit(X, y)
    joblib.dump(gbr_model, os.path.join(OUTPUT_DIR, "model_gbr.pkl"))
    return gbr_model


# ── Feature Importance ────────────────────────────────────────
def analyze_feature_importance(
    model, X: np.ndarray, y: np.ndarray, feature_names: list, model_name: str
):
    print(f"\n[FI] Feature Importance -- {model_name}:")

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        # Permutation importance untuk model tanpa built-in
        perm = permutation_importance(
            model, X, y, n_repeats=10, random_state=RANDOM_STATE
        )
        importances = perm.importances_mean

    fi_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    for _, row in fi_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"   {row['feature']:20s} {bar} ({row['importance']:.4f})")

    fi_path = os.path.join(
        OUTPUT_DIR, f'feature_importance_{model_name.lower().replace(" ","_")}.csv'
    )
    fi_df.to_csv(fi_path, index=False)
    return fi_df


# ── Simpan Prediksi ───────────────────────────────────────────
def save_predictions(
    df: pd.DataFrame, rf_model, xgb_model, lgbm_model, gbr_model, X: np.ndarray, feature_names: list
):
    kab_col = "kabupaten_bps" if "kabupaten_bps" in df.columns else "kabupaten_geo"
    preds = df[[kab_col, TARGET_COL]].copy()
    preds.rename(columns={kab_col: "kabupaten"}, inplace=True)
    
    preds["pred_rf"] = rf_model.predict(X)
    preds["pred_xgb"] = xgb_model.predict(X)
    preds["pred_lgbm"] = lgbm_model.predict(X)
    preds["pred_gbr"] = gbr_model.predict(X)
    
    preds["error_rf"] = preds[TARGET_COL] - preds["pred_rf"]
    preds["error_xgb"] = preds[TARGET_COL] - preds["pred_xgb"]
    preds["error_lgbm"] = preds[TARGET_COL] - preds["pred_lgbm"]
    preds["error_gbr"] = preds[TARGET_COL] - preds["pred_gbr"]

    out_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    preds.to_csv(out_path, index=False)
    print(f"\n[OK] Prediksi tersimpan: {out_path}")
    print(
        preds[["kabupaten", TARGET_COL, "pred_xgb", "error_xgb"]].to_string(index=False)
    )
    return preds


# ── Main ──────────────────────────────────────────────────────
def run_training():
    print("=" * 60)
    print("MODEL TRAINING -- Sumatera Utara Poverty Estimation")
    print("=" * 60)

    df, X, y = load_data()
    available_features = [c for c in FEATURE_COLS if c in df.columns]

    rf_model = train_random_forest(X, y)
    xgb_model = train_xgboost(X, y)
    lgbm_model = train_lightgbm(X, y)
    gbr_model = train_gradient_boosting(X, y)

    analyze_feature_importance(rf_model, X, y, available_features, "Random Forest")
    analyze_feature_importance(xgb_model, X, y, available_features, "XGBoost")
    analyze_feature_importance(lgbm_model, X, y, available_features, "LightGBM")
    analyze_feature_importance(gbr_model, X, y, available_features, "Gradient Boosting")

    save_predictions(df, rf_model, xgb_model, lgbm_model, gbr_model, X, available_features)

    print("\n" + "=" * 60)
    print("[OK] Training selesai!")
    
    # Simpan komparasi metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_out = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    metrics_df.to_csv(metrics_out, index=False)
    print(f"[OK] Komparasi metrik disimpan ke {metrics_out}")
    print("=" * 60)


if __name__ == "__main__":
    run_training()
