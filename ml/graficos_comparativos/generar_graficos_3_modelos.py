"""
Generador de Gráficos Comparativos: 3 Modelos + Feature Importance
==================================================================
Compara la Regresión Lineal (log1p), Random Forest y XGBoost.
También genera un gráfico de Feature Importance para mostrar el ML en acción.
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

# ============================================================================
# Rutas
# ============================================================================
BASE = Path(__file__).resolve().parents[1]
OUT  = Path(__file__).resolve().parent / "comparativa_3_modelos"
OUT.mkdir(parents=True, exist_ok=True)

LR_PREDS  = BASE / "modelos/LR/final/predicciones_lr_final.csv"
RF_PREDS  = BASE / "modelos/RF/final/predicciones_rf_final.csv"
XGB_PREDS = BASE / "modelos/XGB/final/predicciones_xgboost_final.csv"
XGB_MODEL = BASE / "modelos/XGB/final/xgboost_final.pkl"
DATA_PATH = BASE / "feature_engineering/processed_v3/model_data_v3_escenario_B_charlson.csv"

# ============================================================================
# Cargar datos
# ============================================================================
lr  = pd.read_csv(LR_PREDS)
rf  = pd.read_csv(RF_PREDS)
xgb = pd.read_csv(XGB_PREDS)

# Asegurar columnas necesarias
for df in [lr, rf, xgb]:
    if "error" not in df.columns:
        df["error"] = df["los_pred"] - df["los_real"]
    if "abs_error" not in df.columns:
        df["abs_error"] = df["error"].abs()
    if "subestima" not in df.columns:
        df["subestima"] = (df["error"] < 0).astype(int)
    if "tramo_los" not in df.columns:
        df["tramo_los"] = pd.cut(
            df["los_real"], bins=[-1, 2, 6, 13, 26, np.inf],
            labels=["0-2", "3-6", "7-13", "14-26", "27+"]
        )

# ============================================================================
# Estilo global
# ============================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLOR_LR  = "#DC2626"  # Red
COLOR_RF  = "#10B981"  # Green
COLOR_XGB = "#2563EB"  # Blue
TRAMOS    = ["0-2", "3-6", "7-13", "14-26", "27+"]

# ============================================================================
# 1. MAE por tramo (3 barras)
# ============================================================================
mae_lr  = [lr[lr["tramo_los"] == t]["abs_error"].mean() for t in TRAMOS]
mae_rf  = [rf[rf["tramo_los"] == t]["abs_error"].mean() for t in TRAMOS]
mae_xgb = [xgb[xgb["tramo_los"] == t]["abs_error"].mean() for t in TRAMOS]

x = np.arange(len(TRAMOS))
w = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
b1 = ax.bar(x - w, mae_lr,  w, label="Regresión Lineal", color=COLOR_LR,  alpha=0.85)
b2 = ax.bar(x,     mae_rf,  w, label="Random Forest",    color=COLOR_RF,  alpha=0.85)
b3 = ax.bar(x + w, mae_xgb, w, label="XGBoost Final",    color=COLOR_XGB, alpha=0.85)

for bar_group in [b1, b2, b3]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8, rotation=45)

ax.set_xticks(x)
ax.set_xticklabels([f"{t} días" for t in TRAMOS])
ax.set_xlabel("Tramo de Estancia Real")
ax.set_ylabel("MAE promedio (días)")
ax.set_title("Comparativa Global: MAE por Tramo (3 Modelos)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "01_mae_por_tramo_3_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 01_mae_por_tramo_3_modelos.png")

# ============================================================================
# 2. Métricas PLOS (3 barras)
# ============================================================================
lr_real  = (lr["los_real"] >= 27).astype(int)
lr_pred  = (lr["los_pred"] >= 27).astype(int)
rf_real  = (rf["los_real"] >= 27).astype(int)
rf_pred  = (rf["los_pred"] >= 27).astype(int)
xgb_real = (xgb["los_real"] >= 27).astype(int)
xgb_pred = (xgb["los_pred"] >= 27).astype(int)

metrics = {
    "Precision": [
        precision_score(lr_real, lr_pred, zero_division=0),
        precision_score(rf_real, rf_pred, zero_division=0),
        precision_score(xgb_real, xgb_pred, zero_division=0)
    ],
    "Recall": [
        recall_score(lr_real, lr_pred, zero_division=0),
        recall_score(rf_real, rf_pred, zero_division=0),
        recall_score(xgb_real, xgb_pred, zero_division=0)
    ],
    "F1-Score": [
        f1_score(lr_real, lr_pred, zero_division=0),
        f1_score(rf_real, rf_pred, zero_division=0),
        f1_score(xgb_real, xgb_pred, zero_division=0)
    ],
}

labels = list(metrics.keys())
lr_vals  = [v[0] for v in metrics.values()]
rf_vals  = [v[1] for v in metrics.values()]
xgb_vals = [v[2] for v in metrics.values()]

x = np.arange(len(labels))
w = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
b1 = ax.bar(x - w, lr_vals,  w, label="Regresión Lineal", color=COLOR_LR,  alpha=0.85)
b2 = ax.bar(x,     rf_vals,  w, label="Random Forest",    color=COLOR_RF,  alpha=0.85)
b3 = ax.bar(x + w, xgb_vals, w, label="XGBoost Final",    color=COLOR_XGB, alpha=0.85)

for bar_group in [b1, b2, b3]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{bar.get_height()*100:.1f}%", ha="center", va="bottom", fontsize=9)

ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Valor")
ax.set_title("Métricas de Detección PLOS (≥ 27 días) — 3 Modelos", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_metricas_plos_3_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_metricas_plos_3_modelos.png")

# ============================================================================
# 3. Porcentaje de Subestimación (3 líneas)
# ============================================================================
pct_sub_lr  = [lr[lr["tramo_los"] == t]["subestima"].mean() * 100 for t in TRAMOS]
pct_sub_rf  = [rf[rf["tramo_los"] == t]["subestima"].mean() * 100 for t in TRAMOS]
pct_sub_xgb = [xgb[xgb["tramo_los"] == t]["subestima"].mean() * 100 for t in TRAMOS]

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(TRAMOS, pct_sub_lr,  "o-", color=COLOR_LR,  lw=2, markersize=8, label="Reg. Lineal")
ax.plot(TRAMOS, pct_sub_rf,  "^-", color=COLOR_RF,  lw=2, markersize=8, label="Random Forest")
ax.plot(TRAMOS, pct_sub_xgb, "s-", color=COLOR_XGB, lw=2, markersize=8, label="XGBoost Final")
ax.axhline(50, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="50% (neutral)")

ax.set_xlabel("Tramo de Estancia Real")
ax.set_ylabel("% de Casos Subestimados")
ax.set_title("Porcentaje de Subestimación por Tramo (3 Modelos)", fontweight="bold")
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_subestimacion_por_tramo_3_modelos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 03_subestimacion_por_tramo_3_modelos.png")

# ============================================================================
# 4. Feature Importance (XGBoost)
# ============================================================================
try:
    with open(XGB_MODEL, "rb") as f:
        xgb_model = pickle.load(f)
    
    # Obtener el dataset original para extraer los nombres de las columnas
    df = pd.read_csv(DATA_PATH, sep=';', nrows=1)
    X_cols = df.drop(columns=["los_dias", "case_id"]).columns
    
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    top_cols = X_cols[indices]
    top_imps = importances[indices]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(top_cols)), top_imps[::-1], color=COLOR_XGB, alpha=0.8)
    ax.set_yticks(range(len(top_cols)))
    ax.set_yticklabels(top_cols[::-1], fontsize=9)
    ax.set_xlabel("Importancia Relativa (Gain)")
    ax.set_title("Feature Importance: Top 15 variables que impulsan a XGBoost", fontweight="bold")
    
    # Añadir valores a las barras
    for i, v in enumerate(top_imps[::-1]):
        ax.text(v + 0.001, i, f"{v:.3f}", va='center', fontsize=8)
        
    plt.tight_layout()
    plt.savefig(OUT / "04_feature_importance_xgb.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ 04_feature_importance_xgb.png")
except Exception as e:
    print(f"Error generando Feature Importance: {e}")

print(f"\n✅ Todos los gráficos comparativos de 3 modelos generados en: {OUT.relative_to(BASE)}")
