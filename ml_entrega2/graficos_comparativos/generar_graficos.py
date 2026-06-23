"""
Gráficos Comparativos: XGBoost Final vs Regresión Lineal Final
==============================================================
Ambos modelos entrenados sobre el MISMO dataset (Escenario B),
con la MISMA partición (random_state=42, 80/20 estratificada).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# Rutas
# ============================================================================
BASE = Path(__file__).resolve().parents[1]
OUT  = Path(__file__).resolve().parent

XGB_PREDS = BASE / "modelos/XGB/final/predicciones_xgboost_final.csv"
LR_PREDS  = BASE / "modelos/LR/final/predicciones_lr_final.csv"

XGB_KFOLD = BASE / "modelos/XGB/final/resumen_kfold_xgboost_final.csv"
LR_KFOLD  = BASE / "modelos/LR/final/resumen_kfold_lr_final.csv"

# ============================================================================
# Cargar datos
# ============================================================================
xgb = pd.read_csv(XGB_PREDS)
lr  = pd.read_csv(LR_PREDS)

# Asegurar columnas necesarias
for df in [xgb, lr]:
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

COLOR_XGB = "#2563EB"
COLOR_LR  = "#DC2626"
TRAMOS    = ["0-2", "3-6", "7-13", "14-26", "27+"]

# ============================================================================
# 1. Scatter: Real vs Predicho (2 paneles)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, df, color, title in [
    (axes[0], lr,  COLOR_LR,  "Regresión Lineal"),
    (axes[1], xgb, COLOR_XGB, "XGBoost Final"),
]:
    ax.scatter(df["los_real"], df["los_pred"], alpha=0.25, s=12, color=color)
    lim = max(df["los_real"].max(), df["los_pred"].max()) + 5
    ax.plot([0, lim], [0, lim], "k--", lw=1.5, label="Predicción perfecta")
    mae = df["abs_error"].mean()
    ax.set_title(f"{title}\nMAE = {mae:.2f} días")
    ax.set_xlabel("LOS Real (días)")
    ax.set_ylabel("LOS Predicho (días)")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(fontsize=9)

fig.suptitle("Gráfico 1: LOS Real vs. Predicho — Regresión Lineal vs. XGBoost\n(Mismo dataset Escenario B, misma partición)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "01_scatter_real_vs_pred.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 01_scatter_real_vs_pred.png")

# ============================================================================
# 2. Histograma de errores superpuesto
# ============================================================================
fig, ax = plt.subplots(figsize=(11, 5))

ax.hist(lr["error"],  bins=60, alpha=0.55, color=COLOR_LR,  label="Regresión Lineal", density=True)
ax.hist(xgb["error"], bins=60, alpha=0.55, color=COLOR_XGB, label="XGBoost Final",    density=True)
ax.axvline(0, color="black", lw=1.5, linestyle="--")
ax.axvline(lr["error"].mean(),  color=COLOR_LR,  lw=2, linestyle=":", label=f"Bias LR = {lr['error'].mean():.2f} d")
ax.axvline(xgb["error"].mean(), color=COLOR_XGB, lw=2, linestyle=":", label=f"Bias XGB = {xgb['error'].mean():.2f} d")

ax.set_xlabel("Error de Predicción (días predichos − días reales)")
ax.set_ylabel("Densidad")
ax.set_title("Gráfico 2: Distribución de Errores de Predicción\n(Mismo dataset, misma partición)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "02_histograma_errores_superpuesto.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 02_histograma_errores_superpuesto.png")

# ============================================================================
# 3. MAE por tramo (barras agrupadas)
# ============================================================================
mae_xgb = [xgb[xgb["tramo_los"] == t]["abs_error"].mean() for t in TRAMOS]
mae_lr  = [lr[lr["tramo_los"]  == t]["abs_error"].mean() for t in TRAMOS]

x = np.arange(len(TRAMOS))
w = 0.35

fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - w/2, mae_lr,  w, label="Regresión Lineal", color=COLOR_LR,  alpha=0.85)
b2 = ax.bar(x + w/2, mae_xgb, w, label="XGBoost Final",    color=COLOR_XGB, alpha=0.85)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([f"{t} días" for t in TRAMOS])
ax.set_xlabel("Tramo de Estancia Real")
ax.set_ylabel("MAE promedio (días)")
ax.set_title("Gráfico 3: MAE por Tramo de LOS — Regresión Lineal vs. XGBoost\n(Mismo dataset, misma partición)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "03_mae_por_tramo.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 03_mae_por_tramo.png")

# ============================================================================
# 4. Boxplot de error absoluto por tramo (2 paneles)
# ============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

for ax, df, color, title in [
    (axes[0], lr,  COLOR_LR,  "Regresión Lineal"),
    (axes[1], xgb, COLOR_XGB, "XGBoost Final"),
]:
    data = [df[df["tramo_los"] == t]["abs_error"].dropna().values for t in TRAMOS]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=TRAMOS,
                    medianprops=dict(color="black", lw=2),
                    whiskerprops=dict(lw=1.2),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"{title}")
    ax.set_xlabel("Tramo LOS Real")
    ax.set_ylabel("Error Absoluto (días)")
    ax.set_yscale("log")

fig.suptitle("Gráfico 4: Distribución de Error Absoluto por Tramo\n(Mismo dataset, misma partición)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "04_boxplot_error_por_tramo.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 04_boxplot_error_por_tramo.png")

# ============================================================================
# 5. Métricas PLOS: Precision / Recall / F1
# ============================================================================
# Calcular directamente desde predicciones
lr_plos_real  = (lr["los_real"] >= 27).astype(int)
lr_plos_pred  = (lr["los_pred"] >= 27).astype(int)
xgb_plos_real = (xgb["los_real"] >= 27).astype(int)
xgb_plos_pred = (xgb["los_pred"] >= 27).astype(int)

from sklearn.metrics import precision_score, recall_score, f1_score

lr_prec  = precision_score(lr_plos_real, lr_plos_pred, zero_division=0)
lr_rec   = recall_score(lr_plos_real, lr_plos_pred, zero_division=0)
lr_f1    = f1_score(lr_plos_real, lr_plos_pred, zero_division=0)

xgb_prec = precision_score(xgb_plos_real, xgb_plos_pred, zero_division=0)
xgb_rec  = recall_score(xgb_plos_real, xgb_plos_pred, zero_division=0)
xgb_f1   = f1_score(xgb_plos_real, xgb_plos_pred, zero_division=0)

metrics = {
    "Precision\nPLOS": [lr_prec, xgb_prec],
    "Recall\nPLOS":    [lr_rec,  xgb_rec],
    "F1-Score\nPLOS":  [lr_f1,   xgb_f1],
}

labels = list(metrics.keys())
lr_vals  = [v[0] for v in metrics.values()]
xgb_vals = [v[1] for v in metrics.values()]

x = np.arange(len(labels))
w = 0.32

fig, ax = plt.subplots(figsize=(9, 6))
b1 = ax.bar(x - w/2, lr_vals,  w, label="Regresión Lineal", color=COLOR_LR,  alpha=0.85)
b2 = ax.bar(x + w/2, xgb_vals, w, label="XGBoost Final",    color=COLOR_XGB, alpha=0.85)

for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
            f"{bar.get_height()*100:.1f}%", ha="center", va="bottom", fontsize=10)

ax.set_ylim(0, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Valor de la Métrica")
ax.set_title("Gráfico 5: Métricas Clínicas PLOS (≥ 27 días)\nRegresión Lineal vs. XGBoost (Mismo dataset)", fontweight="bold")
ax.axhline(0.5, color="gray", linestyle="--", lw=1, alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "05_metricas_plos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 05_metricas_plos.png")

# ============================================================================
# 6. Subestimación por tramo (líneas)
# ============================================================================
pct_sub_xgb = [xgb[xgb["tramo_los"] == t]["subestima"].mean() * 100 for t in TRAMOS]
pct_sub_lr  = [lr[lr["tramo_los"]  == t]["subestima"].mean() * 100 for t in TRAMOS]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(TRAMOS, pct_sub_lr,  "o-", color=COLOR_LR,  lw=2, markersize=8, label="Regresión Lineal")
ax.plot(TRAMOS, pct_sub_xgb, "s-", color=COLOR_XGB, lw=2, markersize=8, label="XGBoost Final")
ax.axhline(50, color="gray", linestyle="--", lw=1.2, alpha=0.7, label="50% (neutral)")

for i, (y_lr, y_xgb) in enumerate(zip(pct_sub_lr, pct_sub_xgb)):
    ax.annotate(f"{y_lr:.0f}%",  (TRAMOS[i], y_lr),  textcoords="offset points", xytext=(-18, 6),  fontsize=9, color=COLOR_LR)
    ax.annotate(f"{y_xgb:.0f}%", (TRAMOS[i], y_xgb), textcoords="offset points", xytext=(5, 6), fontsize=9, color=COLOR_XGB)

ax.set_xlabel("Tramo de Estancia Real")
ax.set_ylabel("% de Casos Subestimados")
ax.set_title("Gráfico 6: Porcentaje de Subestimación por Tramo de LOS\n(Mismo dataset, misma partición)", fontweight="bold")
ax.set_ylim(0, 115)
ax.legend()
plt.tight_layout()
plt.savefig(OUT / "06_subestimacion_por_tramo.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 06_subestimacion_por_tramo.png")

# ============================================================================
# 7. Matriz de confusión PLOS visual (2 paneles)
# ============================================================================
from sklearn.metrics import confusion_matrix

cm_lr  = confusion_matrix(lr_plos_real, lr_plos_pred, labels=[0, 1])
cm_xgb = confusion_matrix(xgb_plos_real, xgb_plos_pred, labels=[0, 1])

matrices = {
    "Regresión Lineal": cm_lr,
    "XGBoost Final":    cm_xgb,
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (title, cm), color in zip(axes, matrices.items(), [COLOR_LR, COLOR_XGB]):
    total = cm.sum()
    im = ax.imshow(cm, cmap="Blues", aspect="auto")

    labels_cm = [["TN", "FP"], ["FN", "TP"]]
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / total * 100
            ax.text(j, i, f"{labels_cm[i][j]}\n{val}\n({pct:.1f}%)",
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: NO PLOS", "Pred: PLOS"])
    ax.set_yticklabels(["Real: NO PLOS", "Real: PLOS"])
    precision = cm[1,1] / (cm[0,1] + cm[1,1]) if (cm[0,1] + cm[1,1]) > 0 else 0
    recall    = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
    ax.set_title(f"{title}\nPrecision={precision*100:.1f}%  Recall={recall*100:.1f}%", fontweight="bold")

fig.suptitle("Gráfico 7: Matriz de Confusión — Detección de PLOS (LOS ≥ 27 días)\n(Mismo dataset, misma partición)", fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "07_matriz_confusion_plos.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 07_matriz_confusion_plos.png")

# ============================================================================
# 8. Tabla visual resumen (calculada desde datos reales)
# ============================================================================
lr_mae   = lr["abs_error"].mean()
lr_rmse  = np.sqrt((lr["error"]**2).mean())
lr_medae = lr["abs_error"].median()
lr_bias  = lr["error"].mean()

xgb_mae   = xgb["abs_error"].mean()
xgb_rmse  = np.sqrt((xgb["error"]**2).mean())
xgb_medae = xgb["abs_error"].median()
xgb_bias  = xgb["error"].mean()

resumen = pd.DataFrame({
    "Métrica": ["MAE (días)", "RMSE (días)", "MedAE (días)", "Precision PLOS", "Recall PLOS", "F1 PLOS", "Sesgo (días)"],
    "Reg. Lineal": [round(lr_mae,3), round(lr_rmse,3), round(lr_medae,3),
                    round(lr_prec*100,1), round(lr_rec*100,1), round(lr_f1*100,1), round(lr_bias,2)],
    "XGBoost":     [round(xgb_mae,3), round(xgb_rmse,3), round(xgb_medae,3),
                    round(xgb_prec*100,1), round(xgb_rec*100,1), round(xgb_f1*100,1), round(xgb_bias,2)],
})

better = []
for _, row in resumen.iterrows():
    m = row["Métrica"]
    lr_v, xgb_v = row["Reg. Lineal"], row["XGBoost"]
    if m in ["MAE (días)", "RMSE (días)", "MedAE (días)"]:
        better.append("XGBoost" if xgb_v < lr_v else "Reg. Lineal")
    elif m == "Recall PLOS":
        better.append("Reg. Lineal" if lr_v > xgb_v else "XGBoost")
    elif m == "Sesgo (días)":
        better.append("XGBoost" if abs(xgb_v) < abs(lr_v) else "Reg. Lineal")
    else:
        better.append("XGBoost" if xgb_v > lr_v else "Reg. Lineal")
resumen["Ganador"] = better

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
table = ax.table(
    cellText=resumen.values,
    colLabels=resumen.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1E3A5F")
        cell.set_text_props(color="white", fontweight="bold")
    elif col == 3 and row > 0:
        winner = resumen.iloc[row - 1]["Ganador"]
        cell.set_facecolor(COLOR_XGB if winner == "XGBoost" else COLOR_LR)
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#F0F4FF")

ax.set_title("Gráfico 8: Resumen Comparativo de Métricas\n(Mismo dataset Escenario B, misma partición 80/20)", fontweight="bold", pad=20, fontsize=13)
plt.tight_layout()
plt.savefig(OUT / "08_tabla_resumen_metricas.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ 08_tabla_resumen_metricas.png")

print("\n✅ Todos los gráficos generados en:", OUT)
