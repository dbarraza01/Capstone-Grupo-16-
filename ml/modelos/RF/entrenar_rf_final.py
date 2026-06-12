import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor

# ============================================================================
# Configuración y Rutas
# ============================================================================
BASE_DIR = Path(__file__).resolve().parents[3]
DATA_PATH = BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv"
OUT_DIR = BASE_DIR / "ml" / "modelos" / "RF" / "final"

# Crear directorio de salida si no existe
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "los_dias"
ID_COL = "case_id"
RANDOM_STATE = 42
N_JOBS = -1

# Hiperparámetros óptimos (obtenidos en tuning_rf_regularizado.py — Escenario B, target log1p)
# La transformación log1p se usa para reducir el efecto de la asimetría positiva del LOS.
# RandomizedSearchCV con 5-fold CV seleccionó estos parámetros objetivamente.
BEST_PARAMS = {
    "bootstrap": True,
    "max_depth": 20,
    "max_features": 0.5,
    "max_samples": 0.8771414282231924,
    "min_samples_leaf": 11,
    "min_samples_split": 31,
    "n_estimators": 777,
}

# ============================================================================
# Funciones de Evaluación
# ============================================================================
def calcular_metricas(y_real, y_pred, is_plos_only=False):
    """Calcula todas las métricas de regresión y clínicas solicitadas.
    
    IMPORTANTE: Todas las métricas se calculan sobre días reales (después de expm1).
    Las métricas de costo asimétrico son solo evaluación clínica; no penalizan
    el entrenamiento del modelo (que usa MSE simétrico internamente).
    """
    error = y_pred - y_real
    abs_error = np.abs(error)

    mask_sub = error < 0
    mask_sob = error > 0

    mae   = np.mean(abs_error)
    rmse  = np.sqrt(np.mean(error**2))
    medae = np.median(abs_error)

    wape  = np.sum(abs_error) / np.sum(np.clip(y_real, 1e-10, None))
    smape = np.mean(2.0 * abs_error / (np.abs(y_real) + np.abs(y_pred) + 1e-10))

    bias = np.mean(error)

    n_total = len(y_real)
    pct_subestima  = np.sum(mask_sub) / n_total if n_total > 0 else 0
    pct_sobrestima = np.sum(mask_sob) / n_total if n_total > 0 else 0

    pct_le_1d = np.sum(abs_error <= 1) / n_total if n_total > 0 else 0
    pct_le_3d = np.sum(abs_error <= 3) / n_total if n_total > 0 else 0
    pct_le_7d = np.sum(abs_error <= 7) / n_total if n_total > 0 else 0

    subestima_media   = np.mean(np.abs(error[mask_sub]))   if np.sum(mask_sub) > 0 else 0
    subestima_mediana = np.median(np.abs(error[mask_sub])) if np.sum(mask_sub) > 0 else 0
    subestima_p90     = np.percentile(np.abs(error[mask_sub]), 90) if np.sum(mask_sub) > 0 else 0

    sobrestima_media   = np.mean(error[mask_sob])    if np.sum(mask_sob) > 0 else 0
    sobrestima_mediana = np.median(error[mask_sob])  if np.sum(mask_sob) > 0 else 0
    sobrestima_p90     = np.percentile(error[mask_sob], 90) if np.sum(mask_sob) > 0 else 0

    # Costos asimétricos — SOLO evaluación, no afectan el entrenamiento
    costo_2x = np.mean(np.where(mask_sub, abs_error * 2, abs_error))
    costo_3x = np.mean(np.where(mask_sub, abs_error * 3, abs_error))

    # Métricas PLOS (LOS >= 27 días)
    plos_real = (y_real >= 27).astype(int)
    plos_pred = (y_pred >= 27).astype(int)

    if not is_plos_only:
        tn, fp, fn, tp = confusion_matrix(plos_real, plos_pred, labels=[0, 1]).ravel()
        precision_plos = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_plos    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_plos = (2 * precision_plos * recall_plos / (precision_plos + recall_plos)
                   if (precision_plos + recall_plos) > 0 else 0)
    else:
        precision_plos = recall_plos = f1_plos = np.nan

    return {
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
        "f1_plos_27": f1_plos,
    }


def evaluar_por_tramos_los(df_pred):
    """Calcula métricas separadas por tramo de LOS real."""
    tramos = ["0-2", "3-6", "7-13", "14-26", "27+"]
    resultados = []

    for tramo in tramos:
        df_t = df_pred[df_pred["tramo_los"] == tramo]
        if df_t.empty:
            continue

        y_real_t = df_t["los_real"].values
        y_pred_t = df_t["los_pred"].values
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
            "costo_asimetrico_3x": m["costo_asimetrico_3x"],
        })

    return pd.DataFrame(resultados)


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 60)
    print("  Random Forest Final - Escenario B (Consolidación)")
    print("  Modelo con hiperparámetros óptimos y transformación log1p")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # 1. Cargar Datos
    # -----------------------------------------------------------------------
    print(f"\n[1] Cargando dataset: {DATA_PATH.name}")
    df = pd.read_csv(DATA_PATH, sep=";")

    X = df.drop(columns=[TARGET_COL, ID_COL])
    y = df[TARGET_COL]

    # Convertir booleanos a enteros
    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    # Verificaciones de sanidad
    assert ID_COL not in X.columns,       f"Error: {ID_COL} está en las features"
    assert TARGET_COL not in X.columns,   f"Error: {TARGET_COL} está en las features"
    assert "tramo_los" not in X.columns,  "Error: tramo_los está en las features"
    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    assert len(non_numeric) == 0, f"Error: columnas no numéricas: {list(non_numeric)}"

    print(f"    Total registros: {len(X)}")
    print(f"    Total features:  {X.shape[1]}")

    # -----------------------------------------------------------------------
    # 2. Partición Estratificada Holdout (80/20)
    # -----------------------------------------------------------------------
    # La variable auxiliar `tramos_y` se usa SOLO para balancear la partición.
    # No entra como predictor → sin data leakage.
    tramos_y = pd.cut(
        y,
        bins=[-1, 2, 6, 13, 26, np.inf],
        labels=["0-2", "3-6", "7-13", "14-26", "27+"],
    )

    print("\n[2] Partición Train/Test (80/20) estratificada por tramos de LOS")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=tramos_y
    )

    tramos_train = pd.cut(y_train, bins=[-1, 2, 6, 13, 26, np.inf],
                          labels=["0-2", "3-6", "7-13", "14-26", "27+"])
    tramos_test  = pd.cut(y_test,  bins=[-1, 2, 6, 13, 26, np.inf],
                          labels=["0-2", "3-6", "7-13", "14-26", "27+"])

    print(f"    X_train: {X_train.shape},  y_train: {y_train.shape}")
    print(f"    X_test:  {X_test.shape},   y_test:  {y_test.shape}")
    print(f"    Distribución tramos Train:\n{tramos_train.value_counts(normalize=True).sort_index().to_string()}")
    print(f"    Distribución tramos Test:\n{tramos_test.value_counts(normalize=True).sort_index().to_string()}")

    # -----------------------------------------------------------------------
    # 3. Validación Cruzada StratifiedKFold (solo Train — evaluación de estabilidad)
    # -----------------------------------------------------------------------
    # IMPORTANTE: Este K-fold NO busca nuevos hiperparámetros.
    # Usa los mismos parámetros fijos para medir la estabilidad del modelo.
    # El holdout test permanece completamente separado.
    print("\n[3] Ejecutando Validación Cruzada Estratificada (5-Fold) en Train...")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    kfold_metrics = []

    X_train_np    = X_train.values
    y_train_np    = y_train.values
    tramos_train_np = tramos_train.values

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_np, tramos_train_np), 1):
        X_tr_f, X_val_f = X_train_np[train_idx], X_train_np[val_idx]
        y_tr_f, y_val_f = y_train_np[train_idx], y_train_np[val_idx]

        tramos_val_f = pd.cut(
            y_val_f, bins=[-1, 2, 6, 13, 26, np.inf],
            labels=["0-2", "3-6", "7-13", "14-26", "27+"]
        )
        dist_val = list(pd.Series(tramos_val_f).value_counts(normalize=True).sort_index().values.round(2))
        print(f"    Fold {fold}: {len(train_idx)} train / {len(val_idx)} val — "
              f"tramos val: {dist_val}")

        # log1p reduce la asimetría positiva del LOS durante el entrenamiento
        y_tr_f_log = np.log1p(y_tr_f)

        model_fold = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            **BEST_PARAMS,
        )
        model_fold.fit(X_tr_f, y_tr_f_log)

        # Volver a días reales con expm1 antes de calcular métricas
        preds_real = np.clip(np.expm1(model_fold.predict(X_val_f)), 0, None)

        m = calcular_metricas(y_val_f, preds_real)
        m["fold"] = fold
        kfold_metrics.append(m)

    df_kfold = pd.DataFrame(kfold_metrics)
    df_kfold.to_csv(OUT_DIR / "metricas_kfold_rf_final.csv", index=False)

    # Resumen K-fold
    cols_resumen = [
        "mae", "rmse", "medae", "wape", "smape", "bias_error_medio",
        "pct_subestima", "pct_sobrestima",
        "precision_plos_27", "recall_plos_27", "f1_plos_27",
        "costo_asimetrico_2x", "costo_asimetrico_3x",
    ]
    resumen_kfold = [
        {"metrica": c, "mean": df_kfold[c].mean(), "std": df_kfold[c].std()}
        for c in cols_resumen
    ]
    pd.DataFrame(resumen_kfold).to_csv(OUT_DIR / "resumen_kfold_rf_final.csv", index=False)

    print("\n    Resultados promedio StratifiedKFold (Train set):")
    print(f"      MAE:         {df_kfold['mae'].mean():.4f} ± {df_kfold['mae'].std():.4f}")
    print(f"      RMSE:        {df_kfold['rmse'].mean():.4f} ± {df_kfold['rmse'].std():.4f}")
    print(f"      Recall PLOS: {df_kfold['recall_plos_27'].mean():.4f} ± {df_kfold['recall_plos_27'].std():.4f}")

    # -----------------------------------------------------------------------
    # 4. Entrenamiento Final (todo el Train) + Evaluación Holdout
    # -----------------------------------------------------------------------
    print("\n[4] Entrenando modelo final en todo el Train Set...")
    model_final = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        **BEST_PARAMS,
    )

    y_train_log = np.log1p(y_train)

    t0 = time.time()
    model_final.fit(X_train, y_train_log)
    print(f"    Entrenamiento completado en {time.time() - t0:.1f} segundos.")

    # Predicciones en días reales
    y_pred_log  = model_final.predict(X_test)
    y_test_pred = np.clip(np.expm1(y_pred_log), 0, None)

    # -----------------------------------------------------------------------
    # 5. Métricas Holdout
    # -----------------------------------------------------------------------
    print("\n[5] Evaluando métricas globales (Holdout Test)...")
    metricas_globales = calcular_metricas(y_test.values, y_test_pred)

    print(f"    MAE Holdout:    {metricas_globales['mae']:.4f}")
    print(f"    RMSE Holdout:   {metricas_globales['rmse']:.4f}")
    print(f"    Recall PLOS:    {metricas_globales['recall_plos_27']:.4f}")
    print(f"    Precision PLOS: {metricas_globales['precision_plos_27']:.4f}")

    # DataFrame de predicciones
    df_preds = pd.DataFrame({
        "case_id":  df.loc[X_test.index, ID_COL],
        "los_real": y_test.values,
        "los_pred": y_test_pred,
        "error":    y_test_pred - y_test.values,
        "abs_error": np.abs(y_test_pred - y_test.values),
        "tramo_los": pd.cut(
            y_test.values, bins=[-1, 2, 6, 13, 26, np.inf],
            labels=["0-2", "3-6", "7-13", "14-26", "27+"]
        ),
    })
    df_preds["subestima"]  = (df_preds["error"] < 0).astype(int)
    df_preds["sobrestima"] = (df_preds["error"] > 0).astype(int)
    df_preds["plos_real"]  = (df_preds["los_real"] >= 27).astype(int)
    df_preds["plos_pred"]  = (df_preds["los_pred"] >= 27).astype(int)

    # -----------------------------------------------------------------------
    # 6. Guardar Archivos
    # -----------------------------------------------------------------------
    print("\n[6] Guardando archivos...")

    df_preds.to_csv(OUT_DIR / "predicciones_rf_final.csv", index=False)

    pd.DataFrame([metricas_globales]).to_csv(OUT_DIR / "metricas_rf_final.csv", index=False)

    tn, fp, fn, tp = confusion_matrix(
        df_preds["plos_real"], df_preds["plos_pred"], labels=[0, 1]
    ).ravel()
    pd.DataFrame({"TN": [tn], "FP": [fp], "FN": [fn], "TP": [tp]}).to_csv(
        OUT_DIR / "matriz_confusion_plos_rf_final.csv", index=False
    )

    evaluar_por_tramos_los(df_preds).to_csv(
        OUT_DIR / "metricas_por_tramo_rf_final.csv", index=False
    )

    with open(OUT_DIR / "random_forest_final.pkl", "wb") as f:
        pickle.dump(model_final, f)

    # -----------------------------------------------------------------------
    # 7. Gráficos (solo matplotlib)
    # -----------------------------------------------------------------------
    print("\n[7] Generando gráficos...")
    plt.style.use("ggplot")

    # 7.1 Scatter Real vs Pred
    plt.figure(figsize=(8, 8))
    plt.scatter(df_preds["los_real"], df_preds["los_pred"], alpha=0.3, color="steelblue")
    max_val = max(df_preds["los_real"].max(), df_preds["los_pred"].max())
    plt.plot([0, max_val], [0, max_val], "r--", lw=2)
    plt.xlabel("LOS Real (Días)")
    plt.ylabel("LOS Predicho (Días)")
    plt.title("Scatter: LOS Real vs Predicho — RF Final (Holdout)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "scatter_real_vs_pred.png", dpi=150)
    plt.close()

    # 7.2 Histograma de Errores
    plt.figure(figsize=(10, 6))
    plt.hist(df_preds["error"], bins=50, color="mediumseagreen", edgecolor="black")
    plt.axvline(x=0, color="red", linestyle="--", lw=2)
    plt.xlabel("Error (LOS Predicho - LOS Real)")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Errores de Predicción — RF Final")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "histograma_errores.png", dpi=150)
    plt.close()

    # 7.3 Boxplot Error Absoluto por Tramo
    tramos_labels = ["0-2", "3-6", "7-13", "14-26", "27+"]
    data_box = [df_preds[df_preds["tramo_los"] == t]["abs_error"].values for t in tramos_labels]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_box, patch_artist=True,
                boxprops=dict(facecolor="lightsteelblue", color="navy"))
    plt.xticks(range(1, len(tramos_labels) + 1), tramos_labels)
    plt.xlabel("Tramo LOS Real")
    plt.ylabel("Error Absoluto (Días)")
    plt.title("Error Absoluto por Tramo de Estancia — RF Final")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "boxplot_abs_error_por_tramo.png", dpi=150)
    plt.close()

    # 7.4 Barras % Subestimación por Tramo
    df_tramos = evaluar_por_tramos_los(df_preds)
    pct_sub = df_tramos["pct_subestima"] * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_tramos["tramo"], pct_sub, color="salmon", edgecolor="black")
    plt.axhline(y=50, color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Tramo LOS Real")
    plt.ylabel("% Subestimación")
    plt.title("Porcentaje de Subestimación por Tramo — RF Final")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%",
                 ha="center", va="bottom")
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "barras_pct_subestima_por_tramo.png", dpi=150)
    plt.close()

    # 7.5 Barras MAE y RMSE por Tramo
    x = np.arange(len(df_tramos))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, df_tramos["mae"],  width, label="MAE",  color="mediumpurple")
    plt.bar(x + width / 2, df_tramos["rmse"], width, label="RMSE", color="indianred")
    plt.xticks(x, df_tramos["tramo"])
    plt.xlabel("Tramo LOS Real")
    plt.ylabel("Error (Días)")
    plt.title("MAE y RMSE por Tramo de Estancia — RF Final")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "barras_mae_rmse_por_tramo.png", dpi=150)
    plt.close()

    # -----------------------------------------------------------------------
    # 8. Resumen final en consola
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL — Random Forest Escenario B")
    print("=" * 60)
    print(f"  MAE Holdout:              {metricas_globales['mae']:.4f} días")
    print(f"  RMSE Holdout:             {metricas_globales['rmse']:.4f} días")
    print(f"  MedAE Holdout:            {metricas_globales['medae']:.4f} días")
    print(f"  WAPE:                     {metricas_globales['wape']*100:.2f}%")
    print(f"  Sesgo (Bias):             {metricas_globales['bias_error_medio']:.4f} días")
    print(f"  % Subestimación:          {metricas_globales['pct_subestima']*100:.2f}%")
    print(f"  Precision PLOS (>=27d):   {metricas_globales['precision_plos_27']*100:.2f}%")
    print(f"  Recall PLOS (>=27d):      {metricas_globales['recall_plos_27']*100:.2f}%")
    print(f"  F1 PLOS:                  {metricas_globales['f1_plos_27']:.4f}")
    print(f"  Costo Asimétrico 2x:      {metricas_globales['costo_asimetrico_2x']:.4f}")
    print(f"  Costo Asimétrico 3x:      {metricas_globales['costo_asimetrico_3x']:.4f}")
    print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"\n  MAE K-Fold (Train):  {df_kfold['mae'].mean():.4f} ± {df_kfold['mae'].std():.4f}")
    print(f"  RMSE K-Fold (Train): {df_kfold['rmse'].mean():.4f} ± {df_kfold['rmse'].std():.4f}")
    print("=" * 60)

    print(f"\n✅ Script finalizado con éxito.")
    print(f"✅ Archivos guardados en: {OUT_DIR.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
