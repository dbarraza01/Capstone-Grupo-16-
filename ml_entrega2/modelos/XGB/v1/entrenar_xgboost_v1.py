import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBRegressor
import joblib

# Rutas — el script vive en ml/modelos/XGB/, la raíz del proyecto es parents[3]
BASE_DIR = Path(__file__).resolve().parents[3]

DATA_PATH  = BASE_DIR / "ml" / "feature_engineering" / "processed_v2" / "model_data_ml_v2.csv"
OUT_DIR    = BASE_DIR / "ml" / "modelos" / "XGB"
REPORT_DIR = BASE_DIR / "ml" / "modelos" / "XGB"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Cargar datos
# ============================================================================
df = pd.read_csv(DATA_PATH, sep=";")

print("Shape dataset:", df.shape)
print("Columnas:", df.columns[:10].tolist())
print("Nulos totales:", df.isna().sum().sum())

ID_COL = "case_id"
TARGET  = "los_dias"

X        = df.drop(columns=[ID_COL, TARGET])
y        = df[TARGET]
case_ids = df[ID_COL]

# Convertir booleanos a int (misma corrección que RF v1)
no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()
if no_numericas:
    print("Columnas no numéricas detectadas:", no_numericas)
    print("Convirtiendo a 0/1...")
    for col in no_numericas:
        X[col] = X[col].astype(int)

print("Todas las columnas de X son numéricas.")

# ============================================================================
# Split estratificado — idéntico al RF v1 para comparación justa
# ============================================================================
los_tramo = pd.cut(
    y,
    bins=[-1, 2, 6, 13, 26, np.inf],
    labels=["0-2", "3-6", "7-13", "14-26", "27+"]
)
print("\nDistribución por tramo (proporción):")
print(los_tramo.value_counts(normalize=True).sort_index())

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X, y, case_ids,
    test_size=0.20,
    random_state=42,           # misma semilla que RF v1
    stratify=los_tramo
)

# Transformación logarítmica del target (misma que RF v1)
y_train_log = np.log1p(y_train)

# ============================================================================
# Definición del modelo XGBoost v1
# Hiperparámetros según plan_implementacion_modelos.md 
# ============================================================================
xgb = XGBRegressor(
    n_estimators=300,        # igual que RF v1 para comparación justa
    max_depth=6,             # más conservador que RF (max_depth=8) por riesgo de overfitting en boosting
    learning_rate=0.1,       # tasa de aprendizaje estándar para primera iteración
    subsample=0.8,           # usa 80% de los datos por árbol para reducir overfitting
    colsample_bytree=0.8,    # usa 80% de las features por árbol
    reg_alpha=0.1,           # regularización L1 — ayuda con la sparsidad del dataset (0.58% densidad)
    reg_lambda=1.0,          # regularización L2 estándar
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    verbosity=0              # silenciar logs internos de XGBoost
)

# ============================================================================
# Entrenamiento
# ============================================================================
print("\nEntrenando XGBoost v1...")
xgb.fit(X_train, y_train_log)
print("Modelo entrenado correctamente.")

# ============================================================================
# Predicción y back-transform
# ============================================================================
y_pred_log = xgb.predict(X_test)
y_pred     = np.expm1(y_pred_log)
y_pred     = np.clip(y_pred, 0, None)  # sin valores negativos

# ============================================================================
# Métricas globales
# ============================================================================
mae   = mean_absolute_error(y_test, y_pred)
rmse  = np.sqrt(mean_squared_error(y_test, y_pred))
medae = median_absolute_error(y_test, y_pred)

print(f"\nMAE:   {mae:.4f}")
print(f"RMSE:  {rmse:.4f}")
print(f"MedAE: {medae:.4f}")

# ============================================================================
# DataFrame de predicciones individuales
# ============================================================================
df_pred = pd.DataFrame({
    "case_id":  ids_test.values,
    "los_real": y_test.values,
    "los_pred": y_pred
})
df_pred["error"]      = df_pred["los_pred"] - df_pred["los_real"]
df_pred["abs_error"]  = df_pred["error"].abs()
df_pred["subestima"]  = (df_pred["error"] < 0).astype(int)

# ============================================================================
# Métricas por tramo
# ============================================================================
df_pred["tramo_los"] = pd.cut(
    df_pred["los_real"],
    bins=[-1, 2, 6, 13, 26, np.inf],
    labels=["0-2", "3-6", "7-13", "14-26", "27+"]
)

metricas_tramo = (
    df_pred
    .groupby("tramo_los", observed=True)
    .agg(
        n                = ("case_id",   "count"),
        los_real_promedio= ("los_real",  "mean"),
        los_pred_promedio= ("los_pred",  "mean"),
        mae              = ("abs_error", "mean"),
        medae            = ("abs_error", "median"),
        error_medio      = ("error",     "mean"),
        pct_subestima    = ("subestima", "mean")
    )
    .reset_index()
)
metricas_tramo["pct_subestima"] = metricas_tramo["pct_subestima"] * 100
print("\nMétricas por tramo:")
print(metricas_tramo.to_string())

# ============================================================================
# Evaluación PLOS (≥27 días)
# ============================================================================
UMBRAL_PLOS = 27
df_pred["plos_real"] = (df_pred["los_real"] >= UMBRAL_PLOS).astype(int)
df_pred["plos_pred"] = (df_pred["los_pred"] >= UMBRAL_PLOS).astype(int)

precision_plos = precision_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)
recall_plos    = recall_score   (df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)
f1_plos        = f1_score       (df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)

print(f"\nPrecision PLOS: {precision_plos:.4f}")
print(f"Recall PLOS:    {recall_plos:.4f}")
print(f"F1 PLOS:        {f1_plos:.4f}")

# Matriz de confusión
cm = confusion_matrix(df_pred["plos_real"], df_pred["plos_pred"])
tn, fp, fn, tp = cm.ravel()

print("\n--- Matriz de Confusión PLOS (LOS >= 27 días) ---")
print(f"{'':20s} {'Pred: Corto':>15} {'Pred: Largo':>15}")
print(f"{'Real: Corto (<27 d)':20s} {'TN = ' + str(tn):>15} {'FP = ' + str(fp):>15}")
print(f"{'Real: Largo (>=27 d)':20s} {'FN = ' + str(fn):>15} {'TP = ' + str(tp):>15}")
print(f"\nDe {tp + fn} pacientes con LOS largo:")
print(f"  - Detectados (TP): {tp} ({100*tp/(tp+fn):.1f}%)")
print(f"  - No detectados (FN): {fn} ({100*fn/(tp+fn):.1f}%) ← casos peligrosos")
print(f"De {tn + fp} pacientes con LOS corto:")
print(f"  - Falsas alarmas (FP): {fp} ({100*fp/(tn+fp):.1f}%)")

# ============================================================================
# Guardar outputs
# ============================================================================
df_pred.to_csv(REPORT_DIR / "predicciones_xgboost_v1.csv", sep=";", index=False)

pd.DataFrame([{
    "modelo":                "XGBRegressor_v1",
    "target_entrenamiento":  "log1p(los_dias)",
    "mae":                   mae,
    "rmse":                  rmse,
    "medae":                 medae,
    "precision_plos_27":     precision_plos,
    "recall_plos_27":        recall_plos,
    "f1_plos_27":            f1_plos
}]).to_csv(REPORT_DIR / "metricas_xgboost_v1.csv", sep=";", index=False)

metricas_tramo.to_csv(REPORT_DIR / "metricas_por_tramo_xgboost_v1.csv", sep=";", index=False)

pd.DataFrame({
    "metrica": ["TN (pred corto, real corto)", "FP (pred largo, real corto)",
                "FN (pred corto, real largo)", "TP (pred largo, real largo)"],
    "valor":   [tn, fp, fn, tp]
}).to_csv(REPORT_DIR / "matriz_confusion_plos_xgb_v1.csv", sep=";", index=False)

joblib.dump(xgb, OUT_DIR / "xgboost_v1.pkl")

print("\n✅ Archivos guardados:")
for f in ["predicciones_xgboost_v1.csv", "metricas_xgboost_v1.csv",
          "metricas_por_tramo_xgboost_v1.csv", "matriz_confusion_plos_xgb_v1.csv",
          "xgboost_v1.pkl"]:
    print(" ", REPORT_DIR / f)
