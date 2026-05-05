import pandas as pd 
import numpy as np 
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


import joblib

# definir rutas
BASE_DIR = Path(__file__).resolve().parents[3] # Raíz del proyecto

DATA_PATH = BASE_DIR / "ml" / "feature_engineering" / "processed_v2" / "model_data_ml_v2.csv"
OUT_DIR = BASE_DIR / "ml" / "modelos" / "RF"
REPORT_DIR = BASE_DIR / "ml" / "modelos" / "RF"

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# leer el dataset
df = pd.read_csv(DATA_PATH, sep=";")

print("Shape dataset:", df.shape)
print("Columnas:", df.columns[:10])
print("Nulos totales:", df.isna().sum().sum())

# separar features y target
ID_COL = "case_id" # columna de cada paciente
TARGET = "los_dias" # variable objetivo

cols_excluir = [ID_COL, TARGET]

X = df.drop(columns=cols_excluir)
y = df[TARGET]
case_ids = df[ID_COL]

# verificación de que X sea numérico 
no_numericas = X.select_dtypes(exclude=[np.number]).columns.tolist()

if len(no_numericas) > 0:
    print("Columnas no numéricas (booleanos) detectadas:", no_numericas) # columnas tiene_diag_primario es booleana
    print("Convirtiendo a numéricas (0 y 1)...")
    for col in no_numericas:
        X[col] = X[col].astype(int) # convierte booleanos a 0 y 1
        
    # Re-verificar por si alguna columna de texto no se pudo convertir
    restantes = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if len(restantes) > 0:
        raise ValueError(f"No se pudieron convertir a numéricas las siguientes columnas: {restantes}")

print("Todas las columnas de X son numéricas.")

# Creación de tramos para estratificar ---------------------------------------------------------
los_tramo = pd.cut( # Función de pandas para crear tramos (cortes) en los datos
    y, # Variable objetivo
    bins=[-1, 2, 6, 13, 26, np.inf], # Define los límites de los tramos: 0-2, 3-6, 7-13, 14-26, 27+
    labels=["0-2", "3-6", "7-13", "14-26", "27+"] # Etiquetas para cada tramo
)

# Imprime el porcentaje de pacientes en cada tramo (para verificar que la estratificación sea uniforme)
print(los_tramo.value_counts(normalize=True).sort_index())

# ============================================================================
# Separar train/test
# ============================================================================
X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
    X,
    y,
    case_ids,
    test_size=0.20, # tamaño de la muestra de test (20%)
    random_state=42, # semilla para reproducibilidad
    stratify=los_tramo # estratifica los datos para que haya el mismo porcentaje de cada tramo en train y test
)

# transformar el target con log1p
y_train_log = np.log1p(y_train)
y_test_log = np.log1p(y_test)

# Definición del random forest inicial 
# las justificaciones de los hiperparametros recordar colocarlas en el informe
rf = RandomForestRegressor(
    n_estimators=300, # n_estimators es el número de árboles que se van a crear en el modelo. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    max_depth=8, # max_depth es la profundidad máxima de cada árbol. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    min_samples_leaf=10, # min_samples_leaf es el número mínimo de muestras que se requieren para crear una hoja. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    min_samples_split=20, # min_samples_split es el número mínimo de muestras que se requieren para dividir una hoja. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    max_features="sqrt", # max_features es el número máximo de características que se van a utilizar en cada árbol. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    bootstrap=True, # bootstrap es el número máximo de características que se van a utilizar en cada árbol. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
    random_state=42, # random_state es una semilla que se utiliza para garantizar la reproducibilidad de los resultados. Al establecer un valor fijo, se asegura que el modelo genere los mismos resultados cada vez que se ejecute.
    n_jobs=-1 # n_jobs es el número de núcleos del procesador que se van a utilizar para entrenar el modelo. Al ser un modelo de ensamblado, cuantas más instancias se incluyan, más preciso será el modelo, pero también más lento.
)

# ============================================================================
# Entrenar el modelo
# ============================================================================
# aquí el modelo aprende X_train → log1p(LOS)
print("Entrenando el modelo...")
rf.fit(X_train, y_train_log)
print("Modelo entrenado correctamente.")
 
# ============================================================================
# Predecir en test (log-transformed) y transformar de vuelta
# ============================================================================

y_pred_log = rf.predict(X_test) # predicción en log(LOS+1)

# la escala logaritmica hay que devolverla a días 
y_pred = np.expm1(y_pred_log) # exponencial para volver a los días
y_pred = np.clip(y_pred, 0, None) # por seguridad se hace porque no puede haber valores negativos 

# ============================================================================
# calculo de métricas globales 
# ============================================================================
# la media de error absoluto: promedio de cuánto se equivoca el modelo en promedio
mae = mean_absolute_error(y_test, y_pred)
# la raíz del error cuadrático medio: penaliza más los errores grandes
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# la mediana del error absoluto: te da una idea de la "típica" equivocación del modelo
medae = median_absolute_error(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("MedAE:", medae)


# ============================================================================
# Crear dataframe de predicciones 
# se analiza donde el modelo está fallando 
# ============================================================================

df_pred = pd.DataFrame({
    "case_id": ids_test.values,
    "los_real": y_test.values,
    "los_pred": y_pred
})

df_pred["error"] = df_pred["los_pred"] - df_pred["los_real"]
df_pred["abs_error"] = df_pred["error"].abs()
df_pred["subestima"] = (df_pred["error"] < 0).astype(int)

# ============================================================================
# Evaluar por tramo el LOS usando los tramos definidos al inicio
# ¿El modelo falla más en estancias largas?
# ¿Subestima sistemáticamente pacientes de LOS ≥ 27?
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

print(metricas_tramo)


# Evaluar específicamente el Prolonged LOS (≥27 días)

UMBRAL_PLOS = 27

df_pred["plos_real"] = (df_pred["los_real"] >= UMBRAL_PLOS).astype(int)
df_pred["plos_pred"] = (df_pred["los_pred"] >= UMBRAL_PLOS).astype(int)

precision_plos = precision_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0) #probabilidad de detectar un PLOS cuando dice que se va a quedar más de 27 días
recall_plos = recall_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0) # probabilidad de detectar un PLOS real
f1_plos = f1_score(df_pred["plos_real"], df_pred["plos_pred"], zero_division=0) # Balance entre precision y recall, es decir, qué tanto el modelo acierta cuando dice que un paciente se va a quedar más de 27 días y cuántas veces acierta a los pacientes que realmente se quedan más de 27 días.

print("Precision PLOS:", precision_plos)
print("Recall PLOS:", recall_plos)
print("F1 PLOS:", f1_plos)

# Matriz de confusión del PLOS (LOS >= 27 días)
# Filas = LOS real | Columnas = LOS predicho
# TN: predijo corto  y era corto   → sin riesgo, correcto
# FP: predijo largo  y era corto   → falsa alarma
# FN: predijo corto  y era largo   → caso peligroso (no detectado)
# TP: predijo largo  y era largo   → alarma correcta
cm = confusion_matrix(df_pred["plos_real"], df_pred["plos_pred"])
tn, fp, fn, tp = cm.ravel()

print("\n--- Matriz de Confusión PLOS (LOS >= 27 días) ---")
print(f"{'':20s} {'Pred: Corto':>15} {'Pred: Largo':>15}")
print(f"{'Real: Corto (<27 d)':20s} {'TN = ' + str(tn):>15} {'FP = ' + str(fp):>15}")
print(f"{'Real: Largo (>=27 d)':20s} {'FN = ' + str(fn):>15} {'TP = ' + str(tp):>15}")
print(f"\nDe {tp + fn} pacientes que realmente tuvieron LOS largo:")
print(f"  - Detectados correctamente (TP): {tp} ({100*tp/(tp+fn):.1f}%)")
print(f"  - No detectados (FN):           {fn} ({100*fn/(tp+fn):.1f}%) ← casos peligrosos")
print(f"De {tn + fp} pacientes con LOS corto:")
print(f"  - Falsas alarmas (FP):          {fp} ({100*fp/(tn+fp):.1f}%)")

# Guardar matriz de confusión en CSV
cm_df = pd.DataFrame({
    "metrica": ["TN (pred corto, real corto)", "FP (pred largo, real corto)",
                "FN (pred corto, real largo)", "TP (pred largo, real largo)"],
    "valor": [tn, fp, fn, tp]
})
cm_df.to_csv(REPORT_DIR / "matriz_confusion_plos_rf_v1.csv", sep=";", index=False)

# ============================================================================
# Guardar modelo
# ============================================================================

df_pred.to_csv(
    REPORT_DIR / "predicciones_random_forest_v1.csv",
    sep=";",
    index=False
)

metricas_globales = pd.DataFrame([{
    "modelo": "RandomForestRegressor_v1",
    "target_entrenamiento": "log1p(los_dias)",
    "mae": mae,
    "rmse": rmse,
    "medae": medae,
    "precision_plos_27": precision_plos,
    "recall_plos_27": recall_plos,
    "f1_plos_27": f1_plos
}])

metricas_globales.to_csv(
    REPORT_DIR / "metricas_random_forest_v1.csv",
    sep=";",
    index=False
)

metricas_tramo.to_csv(
    REPORT_DIR / "metricas_por_tramo_random_forest_v1.csv",
    sep=";",
    index=False
)

joblib.dump(rf, OUT_DIR / "random_forest_v1.pkl") # guardamos el modelo entrenado, permite reutilizarlo en el futuro sin tener que entrenar de nuevo


print("\nArchivos guardados:")
print(REPORT_DIR / "predicciones_random_forest_v1.csv")
print(REPORT_DIR / "metricas_random_forest_v1.csv")
print(REPORT_DIR / "metricas_por_tramo_random_forest_v1.csv")
print(OUT_DIR / "random_forest_v1.pkl")
