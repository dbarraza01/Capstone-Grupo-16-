"""
tuning_random_forest_regularizado.py
====================================
Búsqueda regularizada de hiperparámetros para Random Forest.

Cambios vs tuning anterior:
- Separa holdout test ANTES del tuning.
- El tuning se hace solo sobre train.
- Usa RandomizedSearchCV + 5-Fold CV.
- Compara target original vs log1p(LOS).
- Evalúa con MAE, RMSE, MedAE, WAPE, SMAPE, sesgo,
  % subestimación y métricas para LOS prolongado >= 27 días.
- NO usa R2.
- NO guarda modelo final; solo reportes de tuning.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

BASE_DIR = Path(__file__).resolve().parents[3]

RANDOM_STATE = 42
N_FOLDS = 5
N_ITER = 50
N_JOBS = -1
TEST_SIZE = 0.20

TARGET_COL = "los_dias"
ID_COL = "case_id"
UMBRAL_PLOS = 27

# Si quieres reducir tiempo, deja solo ["log1p"]
TARGET_MODES = ["log1p"]

DATASETS = {
    "escenario_A": BASE_DIR / "ml" / "feature_engineering" / "processed_v2" / "model_data_ml_v2.csv",
    "escenario_B": BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv",
    "escenario_C": BASE_DIR / "ml" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_C_elixhauser.csv",
}

OUT_DIR = Path(__file__).resolve().parent


# =============================================================================
# ESPACIO DE BÚSQUEDA REGULARIZADO
# =============================================================================
# La búsqueda anterior permitió árboles demasiado complejos:
# max_depth=None, min_samples_leaf bajo, min_samples_split bajo.
# Esta versión fuerza árboles más generales para reducir overfitting.

PARAM_DISTRIBUTIONS = {
    "regressor__n_estimators": randint(200, 800),
    "regressor__max_depth": [8, 10, 12, 15, 20, 25],
    "regressor__min_samples_split": randint(10, 60),
    "regressor__min_samples_leaf": randint(5, 30),
    "regressor__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "regressor__bootstrap": [True],
    "regressor__max_samples": uniform(0.5, 0.4),  # 0.5 a 0.9
}


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def identity(x):
    return x


def clip_pred(y_pred):
    return np.clip(np.asarray(y_pred, dtype=float), 0, None)


def mae_metric(y_true, y_pred):
    return float(mean_absolute_error(y_true, clip_pred(y_pred)))


def rmse_metric(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, clip_pred(y_pred))))


def medae_metric(y_true, y_pred):
    return float(median_absolute_error(y_true, clip_pred(y_pred)))


def wape_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_pred(y_pred)
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def smape_metric(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_pred(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


SCORING = {
    "mae": make_scorer(mae_metric, greater_is_better=False),
    "rmse": make_scorer(rmse_metric, greater_is_better=False),
    "medae": make_scorer(medae_metric, greater_is_better=False),
}


def convert_types(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(v) for v in obj]
    return obj


def clean_params(params):
    return {k.replace("regressor__", ""): v for k, v in params.items()}


def load_dataset(path):
    df = pd.read_csv(path, sep=";")

    if ID_COL not in df.columns:
        raise ValueError(f"No existe columna {ID_COL} en {path}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe columna {TARGET_COL} en {path}")

    X = df.drop(columns=[ID_COL, TARGET_COL])
    y = df[TARGET_COL].astype(float)
    ids = df[ID_COL]

    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(int)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Columnas no numéricas detectadas: {non_numeric[:20]}")

    return X, y, ids


def make_los_strata(y):
    return pd.cut(
        y,
        bins=[-1, 2, 6, 13, 26, np.inf],
        labels=["0-2", "3-6", "7-13", "14-26", "27+"],
    )


def make_model(target_mode):
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=1,  # evita sobre-paralelización con RandomizedSearchCV
    )

    if target_mode == "original":
        return TransformedTargetRegressor(
            regressor=rf,
            func=identity,
            inverse_func=identity,
            check_inverse=False,
        )

    if target_mode == "log1p":
        return TransformedTargetRegressor(
            regressor=rf,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=False,
        )

    raise ValueError(f"target_mode inválido: {target_mode}")


def evaluate_predictions(y_true, y_pred, ids=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = clip_pred(y_pred)

    error = y_pred - y_true
    abs_error = np.abs(error)

    df_pred = pd.DataFrame({
        "los_real": y_true,
        "los_pred": y_pred,
        "error": error,
        "abs_error": abs_error,
        "subestima": (error < 0).astype(int),
        "plos_real_27": (y_true >= UMBRAL_PLOS).astype(int),
        "plos_pred_27": (y_pred >= UMBRAL_PLOS).astype(int),
    })

    if ids is not None:
        df_pred.insert(0, "case_id", np.asarray(ids))

    metrics = {
        "mae": mae_metric(y_true, y_pred),
        "rmse": rmse_metric(y_true, y_pred),
        "medae": medae_metric(y_true, y_pred),
        "wape": wape_metric(y_true, y_pred),
        "smape": smape_metric(y_true, y_pred),
        "bias_error_medio": float(np.mean(error)),
        "pct_subestima": float(np.mean(error < 0)),
        "pct_error_abs_le_1d": float(np.mean(abs_error <= 1)),
        "pct_error_abs_le_3d": float(np.mean(abs_error <= 3)),
        "pct_error_abs_le_7d": float(np.mean(abs_error <= 7)),
        "precision_plos_27": float(precision_score(df_pred["plos_real_27"], df_pred["plos_pred_27"], zero_division=0)),
        "recall_plos_27": float(recall_score(df_pred["plos_real_27"], df_pred["plos_pred_27"], zero_division=0)),
        "f1_plos_27": float(f1_score(df_pred["plos_real_27"], df_pred["plos_pred_27"], zero_division=0)),
    }

    return metrics, df_pred


def evaluate_by_tramo(df_pred):
    df = df_pred.copy()

    df["tramo_los"] = pd.cut(
        df["los_real"],
        bins=[-1, 2, 6, 13, 26, np.inf],
        labels=["0-2", "3-6", "7-13", "14-26", "27+"],
    )

    rows = []

    for tramo, g in df.groupby("tramo_los", observed=True):
        rows.append({
            "tramo_los": str(tramo),
            "n": int(len(g)),
            "los_real_promedio": float(g["los_real"].mean()),
            "los_pred_promedio": float(g["los_pred"].mean()),
            "mae": float(g["abs_error"].mean()),
            "rmse": float(np.sqrt(np.mean(g["error"] ** 2))),
            "medae": float(g["abs_error"].median()),
            "bias_error_medio": float(g["error"].mean()),
            "pct_subestima": float(g["subestima"].mean()),
        })

    return pd.DataFrame(rows)


def extract_top_results(search, top_n=10):
    results_df = pd.DataFrame(search.cv_results_)
    results_df = results_df.sort_values("rank_test_mae")

    top_results = []

    for i, (_, row) in enumerate(results_df.head(top_n).iterrows(), start=1):
        top_results.append({
            "rank": i,
            "mae_cv": round(-row["mean_test_mae"], 4),
            "rmse_cv": round(-row["mean_test_rmse"], 4),
            "medae_cv": round(-row["mean_test_medae"], 4),
            "std_mae_cv": round(row["std_test_mae"], 4),
            "train_mae": round(-row["mean_train_mae"], 4),
            "params": clean_params(row["params"]),
        })

    return top_results


def run_tuning(X, y, ids, scenario_name, target_mode):
    print("\n" + "=" * 80)
    print(f"RF regularizado | {scenario_name} | target={target_mode}")
    print(f"Muestras: {X.shape[0]} | Features: {X.shape[1]}")
    print("=" * 80)

    strata = make_los_strata(y)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X,
        y,
        ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    model = make_model(target_mode)

    cv = KFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER,
        scoring=SCORING,
        refit="mae",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1,
        return_train_score=True,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    best_model = search.best_estimator_
    best_params = clean_params(search.best_params_)

    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    train_metrics, _ = evaluate_predictions(y_train, y_pred_train, ids_train)
    test_metrics, pred_test_df = evaluate_predictions(y_test, y_pred_test, ids_test)
    tramo_df = evaluate_by_tramo(pred_test_df)

    pred_path = OUT_DIR / f"predicciones_rf_regularizado_{scenario_name}_{target_mode}.csv"
    tramo_path = OUT_DIR / f"metricas_tramos_rf_regularizado_{scenario_name}_{target_mode}.csv"

    pred_test_df.to_csv(pred_path, sep=";", index=False)
    tramo_df.to_csv(tramo_path, sep=";", index=False)

    result = {
        "scenario": scenario_name,
        "target_mode": target_mode,
        "best_params": best_params,
        "best_mae_cv": round(-search.best_score_, 4),
        "elapsed_minutes": round(elapsed / 60, 2),
        "n_features": X.shape[1],
        "n_samples_total": X.shape[0],
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "n_iter": N_ITER,
        "n_folds": N_FOLDS,
        "train_metrics_final_refit": train_metrics,
        "test_metrics_holdout": test_metrics,
        "gap_mae_train_test": round(test_metrics["mae"] - train_metrics["mae"], 4),
        "top10_results_cv": extract_top_results(search, top_n=10),
        "files": {
            "predicciones_test": pred_path.name,
            "metricas_tramos_test": tramo_path.name,
        },
    }

    print("\nResultados:")
    print(f"  Mejor MAE CV:        {result['best_mae_cv']:.4f}")
    print(f"  MAE train final:     {train_metrics['mae']:.4f}")
    print(f"  MAE test holdout:    {test_metrics['mae']:.4f}")
    print(f"  RMSE test holdout:   {test_metrics['rmse']:.4f}")
    print(f"  MedAE test holdout:  {test_metrics['medae']:.4f}")
    print(f"  Gap MAE train-test:  {result['gap_mae_train_test']:.4f}")

    print("\nMejores parámetros:")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")

    return result


def main():
    print("=" * 80)
    print("Random Forest regularizado — tuning con holdout test")
    print("=" * 80)
    print(f"N_ITER={N_ITER} | N_FOLDS={N_FOLDS} | TEST_SIZE={TEST_SIZE}")
    print(f"TARGET_MODES={TARGET_MODES}")

    all_results = {}
    summary_rows = []

    for scenario_name, path in DATASETS.items():
        if not path.exists():
            print(f"\nArchivo no encontrado para {scenario_name}: {path}")
            continue

        X, y, ids = load_dataset(path)
        all_results[scenario_name] = {}

        for target_mode in TARGET_MODES:
            result = run_tuning(X, y, ids, scenario_name, target_mode)
            all_results[scenario_name][target_mode] = result

            train_m = result["train_metrics_final_refit"]
            test_m = result["test_metrics_holdout"]

            summary_rows.append({
                "modelo": "RandomForestRegressor_regularizado",
                "escenario": scenario_name,
                "target_mode": target_mode,
                "mae_cv": result["best_mae_cv"],
                "mae_train": train_m["mae"],
                "mae_test": test_m["mae"],
                "rmse_test": test_m["rmse"],
                "medae_test": test_m["medae"],
                "wape_test": test_m["wape"],
                "smape_test": test_m["smape"],
                "bias_test": test_m["bias_error_medio"],
                "pct_subestima_test": test_m["pct_subestima"],
                "precision_plos_27": test_m["precision_plos_27"],
                "recall_plos_27": test_m["recall_plos_27"],
                "f1_plos_27": test_m["f1_plos_27"],
                "gap_mae_train_test": result["gap_mae_train_test"],
                "n_features": result["n_features"],
                "elapsed_minutes": result["elapsed_minutes"],
            })

    all_results = convert_types(all_results)

    json_path = OUT_DIR / "mejores_hiperparametros_random_forest_regularizado.json"
    csv_path = OUT_DIR / "resumen_tuning_random_forest_regularizado.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(["mae_test", "rmse_test", "medae_test"])
    summary_df.to_csv(csv_path, sep=";", index=False)

    print("\n" + "=" * 80)
    print("RESULTADOS EXPORTADOS")
    print("=" * 80)
    print(f"JSON: {json_path}")
    print(f"CSV:  {csv_path}")
    print("\nResumen ordenado por MAE test:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()