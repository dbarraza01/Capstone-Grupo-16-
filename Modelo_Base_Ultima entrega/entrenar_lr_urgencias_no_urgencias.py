"""
Entrenamiento de Modelos Base Lineales Separados (Urgencias vs No Urgencias)
Con Interacciones Clínicas Justificadas Estadísticamente y Penalización Ridge
============================================================================
Desarrollo de Modelamiento Base (Entrega Final - Grupo 16)

Este script:
1. Carga el dataset del Escenario B: model_data_v3_escenario_B_charlson.csv
2. Genera los términos de interacción clínica.
3. Divide el dataset en Urgencias (es_urgencia == 1) y No Urgencias (es_urgencia == 0).
4. Para Urgencias: Entrena un modelo Ridge (alpha=100.0) con 3 interacciones.
5. Para No Urgencias: Entrena un modelo Ridge (alpha=50.0) con 2 interacciones.
6. Evalúa mediante StratifiedKFold (5-fold) en train y realiza el Holdout Test (80/20).
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix

# Configuración de Rutas
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv"
OUT_DIR = BASE_DIR / "Modelo_Base_Ultima entrega"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "los_dias"
ID_COL = "case_id"
RANDOM_STATE = 42

# ============================================================================
# Funciones de Evaluación de Métricas
# ============================================================================
def calcular_metricas(y_real, y_pred, is_plos_only=False):
    """Calcula métricas de regresión y de negocio."""
    error = y_pred - y_real
    abs_error = np.abs(error)
    mask_sub = error < 0
    mask_sob = error > 0

    mae = np.mean(abs_error)
    rmse = np.sqrt(np.mean(error**2))
    medae = np.median(abs_error)

    wape = np.sum(abs_error) / np.sum(np.clip(y_real, 1e-10, None))
    smape = np.mean(2.0 * abs_error / (np.abs(y_real) + np.abs(y_pred) + 1e-10))
    bias = np.mean(error)

    n_total = len(y_real)
    pct_subestima = np.sum(mask_sub) / n_total if n_total > 0 else 0
    pct_sobrestima = np.sum(mask_sob) / n_total if n_total > 0 else 0

    pct_le_1d = np.sum(abs_error <= 1) / n_total if n_total > 0 else 0
    pct_le_3d = np.sum(abs_error <= 3) / n_total if n_total > 0 else 0
    pct_le_7d = np.sum(abs_error <= 7) / n_total if n_total > 0 else 0

    subestima_media = np.mean(np.abs(error[mask_sub])) if np.sum(mask_sub) > 0 else 0
    sobrestima_media = np.mean(error[mask_sob]) if np.sum(mask_sob) > 0 else 0

    costo_2x = np.mean(np.where(mask_sub, abs_error * 2, abs_error))
    costo_3x = np.mean(np.where(mask_sub, abs_error * 3, abs_error))

    plos_real = (y_real >= 27).astype(int)
    plos_pred = (y_pred >= 27).astype(int)

    if not is_plos_only:
        tn, fp, fn, tp = confusion_matrix(plos_real, plos_pred, labels=[0, 1]).ravel()
        precision_plos = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_plos = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_plos = 2 * (precision_plos * recall_plos) / (precision_plos + recall_plos) if (precision_plos + recall_plos) > 0 else 0
    else:
        precision_plos, recall_plos, f1_plos = np.nan, np.nan, np.nan

    return {
        "mae": mae, "rmse": rmse, "medae": medae, "wape": wape, "smape": smape,
        "bias_error_medio": bias, "pct_subestima": pct_subestima, "pct_sobrestima": pct_sobrestima,
        "pct_error_abs_le_1d": pct_le_1d, "pct_error_abs_le_3d": pct_le_3d, "pct_error_abs_le_7d": pct_le_7d,
        "subestimacion_media_solo_subestimados": subestima_media,
        "sobreestimacion_media_solo_sobrestimados": sobrestima_media,
        "costo_asimetrico_2x": costo_2x, "costo_asimetrico_3x": costo_3x,
        "precision_plos_27": precision_plos, "recall_plos_27": recall_plos, "f1_plos_27": f1_plos
    }

def evaluar_por_tramos_los(df_pred):
    """Evalúa las métricas del modelo por tramos de LOS."""
    tramos = ["0-2", "3-6", "7-13", "14-26", "27+"]
    resultados = []
    for tramo in tramos:
        df_tramo = df_pred[df_pred['tramo_los'] == tramo]
        if df_tramo.empty:
            continue
        y_real_t = df_tramo['los_real'].values
        y_pred_t = df_tramo['los_pred'].values
        m = calcular_metricas(y_real_t, y_pred_t, is_plos_only=True)
        resultados.append({
            "tramo": tramo, "n_casos": len(y_real_t),
            "los_real_promedio": np.mean(y_real_t), "los_pred_promedio": np.mean(y_pred_t),
            "mae": m["mae"], "rmse": m["rmse"], "medae": m["medae"],
            "bias_error_medio": m["bias_error_medio"], "pct_subestima": m["pct_subestima"],
            "pct_sobrestima": m["pct_sobrestima"], "costo_asimetrico_2x": m["costo_asimetrico_2x"]
        })
    return pd.DataFrame(resultados)

# ============================================================================
# Proceso de Entrenamiento y Evaluación
# ============================================================================
def entrenar_y_evaluar_modelo(df_sub, es_urgente):
    grupo_nombre = "Urgencias" if es_urgente else "No_Urgencias"
    alpha_valor = 100.0 if es_urgente else 50.0
    print(f"\n--- Entrenando Modelo Base: {grupo_nombre} (Ridge alpha={alpha_valor}) ---")
    
    # 1. Copiar y definir variables
    df_modelo = df_sub.copy()
    
    # Definir variables predictivas
    X = df_modelo.drop(columns=[TARGET_COL, ID_COL])
    y = df_modelo[TARGET_COL]
    
    # Asegurar que todas las columnas booleanas sean enteros
    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # 2. Train/Test Split (80/20) estratificado
    tramos_y = pd.cut(
        y,
        bins=[-1, 2, 6, 13, 26, np.inf],
        labels=["0-2", "3-6", "7-13", "14-26", "27+"]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=tramos_y
    )
    
    # 3. Validación Cruzada (StratifiedKFold)
    tramos_train = pd.cut(y_train, bins=[-1, 2, 6, 13, 26, np.inf],
                          labels=["0-2", "3-6", "7-13", "14-26", "27+"])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kfold_metrics = []

    X_train_np = X_train.values
    y_train_np = y_train.values
    tramos_train_np = tramos_train.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, tramos_train_np), 1):
        X_tr_f, X_val_f = X_train_np[train_idx], X_train_np[val_idx]
        y_tr_f, y_val_f = y_train_np[train_idx], y_train_np[val_idx]

        model_fold = Ridge(alpha=alpha_valor)
        model_fold.fit(X_tr_f, np.log1p(y_tr_f))

        preds = model_fold.predict(X_val_f)
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        m = calcular_metricas(y_val_f, preds)
        m['fold'] = fold
        kfold_metrics.append(m)

    df_kfold = pd.DataFrame(kfold_metrics)
    df_kfold.to_csv(OUT_DIR / f"metricas_kfold_lr_{grupo_nombre}.csv", index=False)

    # 4. Ajuste del Modelo Final
    model_final = Ridge(alpha=alpha_valor)
    model_final.fit(X_train, np.log1p(y_train))

    # Predicción en el Test Holdout
    y_test_pred = model_final.predict(X_test)
    y_test_pred = np.expm1(y_test_pred)
    y_test_pred = np.clip(y_test_pred, 0, None)

    # Métricas finales
    metricas_globales = calcular_metricas(y_test.values, y_test_pred)
    print(f"    [Holdout Test] MAE: {metricas_globales['mae']:.4f} | RMSE: {metricas_globales['rmse']:.4f}")
    
    # Guardar métricas globales
    pd.DataFrame([metricas_globales]).to_csv(OUT_DIR / f"metricas_globales_lr_{grupo_nombre}.csv", index=False)

    # Guardar predicciones
    df_preds = pd.DataFrame({
        "case_id": df_modelo.loc[X_test.index, ID_COL],
        "los_real": y_test.values,
        "los_pred": y_test_pred,
        "error": y_test_pred - y_test.values,
        "abs_error": np.abs(y_test_pred - y_test.values),
        "tramo_los": pd.cut(y_test.values, bins=[-1, 2, 6, 13, 26, np.inf],
                            labels=["0-2", "3-6", "7-13", "14-26", "27+"])
    })
    df_preds["subestima"] = (df_preds["error"] < 0).astype(int)
    df_preds["plos_real"] = (df_preds["los_real"] >= 27).astype(int)
    df_preds["plos_pred"] = (df_preds["los_pred"] >= 27).astype(int)
    df_preds.to_csv(OUT_DIR / f"predicciones_lr_{grupo_nombre}.csv", index=False)

    # Evaluación por tramo
    df_tramos = evaluar_por_tramos_los(df_preds)
    df_tramos.to_csv(OUT_DIR / f"metricas_por_tramo_lr_{grupo_nombre}.csv", index=False)

    # Guardar pickle del modelo
    with open(OUT_DIR / f"lr_base_{grupo_nombre}.pkl", "wb") as f:
        pickle.dump(model_final, f)
        
    # Guardar la lista de columnas que espera el modelo (importante para la web)
    with open(OUT_DIR / f"columnas_modelo_lr_{grupo_nombre}.pkl", "wb") as f:
        pickle.dump(X.columns.tolist(), f)

    print(f"OK: Modelo {grupo_nombre} entrenado y guardado correctamente.")

# ============================================================================
# Main Execution
# ============================================================================
def main():
    print("======================================================================")
    print("Iniciando Entrenamiento de Modelos Base Separados (Tomas)")
    print("======================================================================")

    # 1. Cargar dataset maestro
    print(f"Cargando dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, sep=';')

    # 2. Generar interacciones clínicas
    print("Generando variables de interaccion...")
    df['int_charlson_diag'] = df['charlson_index'] * df['n_diag_total']
    df['int_proc_diag'] = df['n_procedimientos'] * df['n_diag_total']
    df['int_charlson_proc'] = df['charlson_index'] * df['n_procedimientos']

    # 3. Separar en Urgencias y No Urgencias
    df_urg = df[df['es_urgencia'] == 1].copy()
    df_no_urg = df[df['es_urgencia'] == 0].copy()

    # 4. Eliminar es_urgencia del dataset (ya está implícito por la separación)
    # y entrenar cada modelo con sus interacciones correspondientes
    
    # Urgencias: Incluye las 3 interacciones
    df_urg = df_urg.drop(columns=['es_urgencia'])
    entrenar_y_evaluar_modelo(df_urg, es_urgente=True)
    
    # No Urgencias: Incluye solo 2 interacciones (se elimina int_proc_diag por no ser significativa)
    df_no_urg = df_no_urg.drop(columns=['es_urgencia', 'int_proc_diag'])
    entrenar_y_evaluar_modelo(df_no_urg, es_urgente=False)

    print("\nOK: Proceso finalizado. Todos los archivos guardados en:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()
