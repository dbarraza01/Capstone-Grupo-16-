import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import joblib

# ============================================================================
# Definir rutas
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "ml" / "processed" / "model_data_ml_v2.csv"
OUT_DIR = BASE_DIR / "ml" / "models"
REPORT_DIR = BASE_DIR / "ml" / "reports_modelos"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Leer dataset
# ============================================================================

df = pd.read_csv(DATA_PATH, sep=";")

print("Shape dataset:", df.shape)
print("Nulos totales:", df.isna().sum().sum())

# ============================================================================
# Separar variables predictoras y target
# ============================================================================

ID_COL = "case_id"
TARGET = "los_dias"

X = df.drop(columns=[ID_COL, TARGET])
y = df[TARGET]
case_ids = df[ID_COL]

# ============================================================================
# Asegurar que X sea numérico
# ============================================================================

no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()

if len(no_numericas) > 0:
    print("Columnas no numéricas detectadas:", no_numericas)
    print("Convirtiendo a numéricas...")
    for col in no_numericas:
        X[col] = X[col].astype(int)

restantes = X.select_dtypes(exclude=[np.number]).columns.tolist()
if len(restantes) > 0:
    raise ValueError(f"Columnas no numéricas restantes: {restantes}")

print("Todas las columnas de X son numéricas.")

# ============================================================================
# Crear tramos para estratificación
# ============================================================================

los_tramo = pd.cut(
    y,
    bins=[-1, 2, 6, 13, 26, np.inf],
    labels=["0-2", "3-6", "7-13", "14-26", "27+"]
)

# ============================================================================
# Separar train/test
# ============================================================================

X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    case_ids,
    test_size=0.20,
    random_state=42,
    stratify=los_tramo
)

# transformar el target con log1p
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Definición del modelo Gradient Boosting
gb_model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    max_features="sqrt",
    random_state=42
)

# ============================================================================
# Entrenar el modelo
# ============================================================================
print("Entrenando el modelo Gradient Boosting...")
gb_model.fit(X_train, y_train_log)
print("Modelo entrenado correctamente.")

# ============================================================================
# Predecir en test (log-transformed) y transformar de vuelta
# ============================================================================

y_pred_log = gb_model.predict(X_test)

# la escala logaritmica hay que devolverla a días 
y_pred = np.expm1(y_pred_log)
y_pred = np.clip(y_pred, 0, None)

# ============================================================================
# Evaluar modelo
# ============================================================================

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
medae = median_absolute_error(y_test, y_pred)

print(f"\n===== GradientBoostingRegressor_v1 =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("MedAE:", medae)

df_pred = pd.DataFrame({
    "case_id": ids_test.values,
    "los_real": y_test.values,
    "los_pred": y_pred
})

df_pred["error"] = df_pred["los_pred"] - df_pred["los_real"]
df_pred["abs_error"] = df_pred["error"].abs()
df_pred["subestima"] = (df_pred["error"] < 0).astype(int)

df_pred["tramo_los"] = pd.cut(
    df_pred["los_real"],
    bins=[-1, 2, 6, 13, 26, np.inf],
    labels=["0-2", "3-6", "7-13", "14-26", "27+"]
)

metricas_tramo = (
    df_pred
    .groupby("tramo_los", observed=True)
    .agg(
        n=("case_id", "count"),
        los_real_promedio=("los_real", "mean"),
        los_pred_promedio=("los_pred", "mean"),
        mae=("abs_error", "mean"),
        medae=("abs_error", "median"),
        error_medio=("error", "mean"),
        pct_subestima=("subestima", "mean")
    )
    .reset_index()
)
metricas_tramo["pct_subestima"] = metricas_tramo["pct_subestima"] * 100
print("\nMétricas por tramo:")
print(metricas_tramo)

UMBRAL_PLOS = 27
df_pred["plos_real"] = (df_pred["los_real"] >= UMBRAL_PLOS).astype(int)
df_pred["plos_pred"] = (df_pred["los_pred"] >= UMBRAL_PLOS).astype(int)

precision_plos = precision_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)
recall_plos = recall_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)
f1_plos = f1_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0)

print("\nEvaluación PLOS:")
print("Precision PLOS:", precision_plos)
print("Recall PLOS:", recall_plos)
print("F1 PLOS:", f1_plos)

cm = confusion_matrix(df_pred["plos_real"], df_pred["plos_pred"])
tn, fp, fn, tp = cm.ravel()

print("\n--- Matriz de Confusión PLOS (LOS >= 27 días) ---")
print(f"{'':20s} {'Pred: Corto':>15} {'Pred: Largo':>15}")
print(f"{'Real: Corto (<27 d)':20s} {'TN = ' + str(tn):>15} {'FP = ' + str(fp):>15}")
print(f"{'Real: Largo (>=27 d)':20s} {'FN = ' + str(fn):>15} {'TP = ' + str(tp):>15}")

# ============================================================================
# Guardar resultados
# ============================================================================

df_pred.to_csv(REPORT_DIR / "predicciones_gradient_boosting_v1.csv", sep=";", index=False)
metricas_tramo.to_csv(REPORT_DIR / "metricas_por_tramo_gradient_boosting_v1.csv", sep=";", index=False)

metricas_globales = pd.DataFrame([{
    "modelo": "GradientBoostingRegressor_v1",
    "target_entrenamiento": "log1p(los_dias)",
    "mae": mae,
    "rmse": rmse,
    "medae": medae,
    "precision_plos_27": precision_plos,
    "recall_plos_27": recall_plos,
    "f1_plos_27": f1_plos
}])

metricas_globales.to_csv(REPORT_DIR / "metricas_gradient_boosting_v1.csv", sep=";", index=False)
joblib.dump(gb_model, OUT_DIR / "gradient_boosting_v1.pkl")

print("\nArchivos guardados correctamente.")
