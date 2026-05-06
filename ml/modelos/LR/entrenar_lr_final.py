"""
Regresión Lineal Final — Modelo Base (Baseline)
================================================
Usa EXACTAMENTE el mismo dataset, partición y métricas que
entrenar_xgboost_final.py para garantizar comparabilidad al 100%.

Dataset:  model_data_v3_escenario_B_charlson.csv  (Escenario B)
Split:    80/20 estratificado, random_state=42
Target:   los_dias (con transformación log1p)
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix

# ============================================================================
# Configuración y Rutas (IDÉNTICAS al XGBoost Final)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv"
OUT_DIR = BASE_DIR / "ml" / "modelos" / "LR" / "final"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "los_dias"
ID_COL = "case_id"
RANDOM_STATE = 42

# ============================================================================
# Funciones de Evaluación (IDÉNTICAS al XGBoost Final)
# ============================================================================
def calcular_metricas(y_real, y_pred, is_plos_only=False):
    """Calcula todas las métricas de regresión y negocio."""

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
    subestima_mediana = np.median(np.abs(error[mask_sub])) if np.sum(mask_sub) > 0 else 0
    subestima_p90 = np.percentile(np.abs(error[mask_sub]), 90) if np.sum(mask_sub) > 0 else 0

    sobrestima_media = np.mean(error[mask_sob]) if np.sum(mask_sob) > 0 else 0
    sobrestima_mediana = np.median(error[mask_sob]) if np.sum(mask_sob) > 0 else 0
    sobrestima_p90 = np.percentile(error[mask_sob], 90) if np.sum(mask_sob) > 0 else 0

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

    metricas = {
        "mae": mae,
        "rmse": rmse,
        "medae": medae,
        "wape": wape,
        "smape": smape,
        "bias_error_medio": bias,
        "pct_subestima": pct_subestima,
        "pct_sobrestima": pct_sobrestima,
        "pct_error_abs_le_1d": pct_le_1d,
        "pct_error_abs_le_3d": pct_le_3d,
        "pct_error_abs_le_7d": pct_le_7d,
        "subestimacion_media_solo_subestimados": subestima_media,
        "subestimacion_mediana_solo_subestimados": subestima_mediana,
        "subestimacion_p90_solo_subestimados": subestima_p90,
        "sobreestimacion_media_solo_sobrestimados": sobrestima_media,
        "sobreestimacion_mediana_solo_sobrestimados": sobrestima_mediana,
        "sobreestimacion_p90_solo_sobrestimados": sobrestima_p90,
        "costo_asimetrico_2x": costo_2x,
        "costo_asimetrico_3x": costo_3x,
        "precision_plos_27": precision_plos,
        "recall_plos_27": recall_plos,
        "f1_plos_27": f1_plos
    }

    return metricas


def evaluar_por_tramos_los(df_pred):
    """Calcula métricas separadas por cada tramo de LOS."""

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
            "tramo": tramo,
            "n_casos": len(y_real_t),
            "los_real_promedio": np.mean(y_real_t),
            "los_pred_promedio": np.mean(y_pred_t),
            "mae": m["mae"],
            "rmse": m["rmse"],
            "medae": m["medae"],
            "bias_error_medio": m["bias_error_medio"],
            "pct_subestima": m["pct_subestima"],
            "pct_sobrestima": m["pct_sobrestima"],
            "pct_error_abs_le_1d": m["pct_error_abs_le_1d"],
            "pct_error_abs_le_3d": m["pct_error_abs_le_3d"],
            "pct_error_abs_le_7d": m["pct_error_abs_le_7d"],
            "subestimacion_media_solo_subestimados": m["subestimacion_media_solo_subestimados"],
            "sobreestimacion_media_solo_sobrestimados": m["sobreestimacion_media_solo_sobrestimados"],
            "costo_asimetrico_2x": m["costo_asimetrico_2x"],
            "costo_asimetrico_3x": m["costo_asimetrico_3x"]
        })

    return pd.DataFrame(resultados)


# ============================================================================
# Main
# ============================================================================
def main():
    print("="*60)
    print("  Regresión Lineal Final — Modelo Base (Baseline)")
    print("  Dataset: Escenario B (Charlson) — Con transformación log1p")
    print("="*60)

    # 1. Cargar Datos (MISMO dataset que XGBoost)
    print(f"\n[1] Cargando dataset: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH, sep=';')

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    bool_cols = X.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Verificaciones de sanidad
    assert ID_COL not in X.columns
    assert TARGET_COL not in X.columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    assert len(non_numeric) == 0, f"Columnas no numéricas: {non_numeric}"

    print(f"    Total registros: {len(X)}")
    print(f"    Total features: {X.shape[1]}")

    # 2. Partición Estratificada (IDÉNTICA al XGBoost)
    tramos_y = pd.cut(
        y,
        bins=[-1, 2, 6, 13, 26, np.inf],
        labels=["0-2", "3-6", "7-13", "14-26", "27+"]
    )

    print("\n[2] Partición Train/Test (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=tramos_y
    )

    tramos_test = pd.cut(y_test, bins=[-1, 2, 6, 13, 26, np.inf],
                         labels=["0-2", "3-6", "7-13", "14-26", "27+"])

    print(f"    X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"    X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # 3. Validación Cruzada (StratifiedKFold) — misma que XGBoost
    print("\n[3] Ejecutando Validación Cruzada Estratificada (5-Fold) en Train...")

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

        print(f"    Fold {fold}: {len(train_idx)} train, {len(val_idx)} val")

        model_fold = LinearRegression()
        model_fold.fit(X_tr_f, np.log1p(y_tr_f))

        preds = model_fold.predict(X_val_f)
        preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)

        m = calcular_metricas(y_val_f, preds)
        m['fold'] = fold
        kfold_metrics.append(m)

    df_kfold = pd.DataFrame(kfold_metrics)
    df_kfold.to_csv(OUT_DIR / "metricas_kfold_lr_final.csv", index=False)

    # Resumen K-fold
    cols_to_summarize = [
        "mae", "rmse", "medae", "wape", "smape", "bias_error_medio",
        "pct_subestima", "pct_sobrestima", "precision_plos_27",
        "recall_plos_27", "f1_plos_27", "costo_asimetrico_2x", "costo_asimetrico_3x"
    ]
    resumen_kfold = []
    for col in cols_to_summarize:
        resumen_kfold.append({
            "metrica": col,
            "mean": df_kfold[col].mean(),
            "std": df_kfold[col].std()
        })
    df_resumen_kfold = pd.DataFrame(resumen_kfold)
    df_resumen_kfold.to_csv(OUT_DIR / "resumen_kfold_lr_final.csv", index=False)

    print("\n    Resultados promedio StratifiedKFold (Train set):")
    print(f"      MAE:  {df_kfold['mae'].mean():.4f} ± {df_kfold['mae'].std():.4f}")
    print(f"      RMSE: {df_kfold['rmse'].mean():.4f} ± {df_kfold['rmse'].std():.4f}")
    print(f"      Recall PLOS: {df_kfold['recall_plos_27'].mean():.4f} ± {df_kfold['recall_plos_27'].std():.4f}")

    # 4. Entrenamiento del Modelo Final y Evaluación Holdout
    print("\n[4] Entrenando modelo final en todo el Train Set...")
    model_final = LinearRegression()

    start_time = time.time()
    model_final.fit(X_train, np.log1p(y_train))
    print(f"    Entrenamiento completado en {time.time() - start_time:.2f} segundos.")

    # Predicciones Holdout (CON log1p -> expm1)
    y_test_pred = model_final.predict(X_test)
    y_test_pred = np.expm1(y_test_pred)
    y_test_pred = np.clip(y_test_pred, 0, None)

    # 5. Calcular métricas finales
    print("\n[5] Evaluando métricas globales (Holdout Test)...")
    metricas_globales = calcular_metricas(y_test.values, y_test_pred)

    print(f"    MAE Holdout:   {metricas_globales['mae']:.4f}")
    print(f"    RMSE Holdout:  {metricas_globales['rmse']:.4f}")
    print(f"    Recall PLOS:   {metricas_globales['recall_plos_27']:.4f}")
    print(f"    Precision PLOS:{metricas_globales['precision_plos_27']:.4f}")

    # Generar DataFrame de predicciones (formato IDÉNTICO al XGBoost)
    df_preds = pd.DataFrame({
        "case_id": df.loc[X_test.index, ID_COL],
        "los_real": y_test.values,
        "los_pred": y_test_pred,
        "error": y_test_pred - y_test.values,
        "abs_error": np.abs(y_test_pred - y_test.values),
        "tramo_los": pd.cut(y_test.values, bins=[-1, 2, 6, 13, 26, np.inf],
                            labels=["0-2", "3-6", "7-13", "14-26", "27+"])
    })
    df_preds["subestima"] = (df_preds["error"] < 0).astype(int)
    df_preds["sobrestima"] = (df_preds["error"] > 0).astype(int)
    df_preds["plos_real"] = (df_preds["los_real"] >= 27).astype(int)
    df_preds["plos_pred"] = (df_preds["los_pred"] >= 27).astype(int)

    df_preds.to_csv(OUT_DIR / "predicciones_lr_final.csv", index=False)

    # Guardar métricas globales
    df_metricas = pd.DataFrame([metricas_globales])
    df_metricas.to_csv(OUT_DIR / "metricas_lr_final.csv", index=False)

    # Matriz confusión PLOS
    tn, fp, fn, tp = confusion_matrix(df_preds["plos_real"], df_preds["plos_pred"], labels=[0, 1]).ravel()
    df_mc = pd.DataFrame({"TN": [tn], "FP": [fp], "FN": [fn], "TP": [tp]})
    df_mc.to_csv(OUT_DIR / "matriz_confusion_plos_lr_final.csv", index=False)

    # Evaluación por tramo
    df_tramos = evaluar_por_tramos_los(df_preds)
    df_tramos.to_csv(OUT_DIR / "metricas_por_tramo_lr_final.csv", index=False)

    # 6. Guardar Modelo
    print("\n[6] Guardando modelo...")
    with open(OUT_DIR / "lr_final.pkl", "wb") as f:
        pickle.dump(model_final, f)

    print("\n✅ Script finalizado con éxito.")
    print(f"✅ Todos los archivos guardados en: {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
