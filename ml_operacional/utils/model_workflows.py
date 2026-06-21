"""Workflows reutilizables de entrenamiento y evaluacion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_operacional.utils.pipeline_operacional import (
    DATA_SPLITS_DIR,
    ID_COL,
    MODELS_DIR,
    REPORTS_DIR,
    SEGMENTS,
    TARGET_COL,
    URGENCY_COL,
    binary_target_los14,
    dataframe_to_markdown,
    default_rf_clf_params,
    default_rf_reg_params,
    default_xgb_clf_params,
    default_xgb_reg_params,
    evaluar_predicciones,
    export_oof_dataset,
    generate_oof_probabilities,
    load_json,
    load_model_bundle,
    load_segment_split,
    make_classifier,
    make_lr_regressor,
    make_regressor,
    prepare_xy,
    save_model_bundle,
)


MODEL_DIR = {
    "xgb": "XGB",
    "rf": "RF",
}


def _model_folder(model_name: str) -> Path:
    return Path(__file__).resolve().parents[1] / MODEL_DIR[model_name]


def _default_params(model_name: str, kind: str, y_binary: np.ndarray | None = None) -> dict:
    if model_name == "xgb" and kind == "clf":
        return default_xgb_clf_params(y_binary)
    if model_name == "xgb" and kind == "reg":
        return default_xgb_reg_params()
    if model_name == "rf" and kind == "clf":
        return default_rf_clf_params()
    if model_name == "rf" and kind == "reg":
        return default_rf_reg_params()
    raise ValueError(f"Parametros default no definidos para {model_name}/{kind}")


def load_best_params(model_name: str, kind: str, segment: str, y_binary: np.ndarray | None = None) -> dict:
    path = _model_folder(model_name) / f"best_params_{kind}_{segment}.json"
    return load_json(path, fallback=_default_params(model_name, kind, y_binary))


def train_two_stage_model(model_name: str) -> None:
    model_label = model_name.upper()
    print(f"Entrenamiento operacional dos etapas: {model_label}")

    for segment in SEGMENTS:
        print(f"\n[{model_label}] Segmento: {segment}")
        train_df = load_segment_split(segment, "train")
        X, y = prepare_xy(train_df)
        y_binary = binary_target_los14(y)
        print(f"  Train shape: {X.shape}; positivos LOS>=14: {int(y_binary.sum())}")

        clf_params = load_best_params(model_name, "clf", segment, y_binary=y_binary)
        reg_params = load_best_params(model_name, "reg", segment)
        print("  Parametros JSON cargados correctamente")

        print("  Generando probabilidades OOF prob_los_14")
        prob_oof = generate_oof_probabilities(model_name, clf_params, X, y_binary)
        train_prob = export_oof_dataset(segment, train_df, prob_oof, model_name)

        print("  Entrenando clasificador final")
        clf_final = make_classifier(model_name, clf_params)
        clf_final.fit(X, y_binary)
        save_model_bundle(
            MODELS_DIR / f"clf_{model_name}_{segment}.joblib",
            clf_final,
            X.columns.tolist(),
            {"model_name": model_name, "segment": segment, "stage": "classifier", "threshold_los": 14},
        )

        print("  Entrenando regresor final con prob_los_14")
        X_reg, y_reg = prepare_xy(train_prob, include_prob=True)
        reg_final = make_regressor(model_name, reg_params)
        reg_final.fit(X_reg, y_reg)
        save_model_bundle(
            MODELS_DIR / f"reg_{model_name}_{segment}.joblib",
            reg_final,
            X_reg.columns.tolist(),
            {"model_name": model_name, "segment": segment, "stage": "regressor", "uses_prob_los_14": True},
        )
        print(f"  Modelos guardados para {segment}")


def evaluate_two_stage_model(model_name: str) -> None:
    model_label = model_name.upper()
    print(f"Evaluacion holdout operacional: {model_label}")
    rows = []

    for segment in SEGMENTS:
        print(f"\n[{model_label}] Segmento: {segment}")
        holdout_df = load_segment_split(segment, "holdout")
        X, y = prepare_xy(holdout_df)

        clf_bundle = load_model_bundle(MODELS_DIR / f"clf_{model_name}_{segment}.joblib")
        reg_bundle = load_model_bundle(MODELS_DIR / f"reg_{model_name}_{segment}.joblib")
        clf = clf_bundle["model"]
        reg = reg_bundle["model"]
        clf_features = clf_bundle["features"]
        reg_features = reg_bundle["features"]

        prob = clf.predict_proba(X[clf_features])[:, 1]
        X_reg = X.copy()
        X_reg["prob_los_14"] = prob
        pred = np.clip(reg.predict(X_reg[reg_features]), 0, None)

        part = pd.DataFrame({
            "case_id": holdout_df[ID_COL].values,
            "los_dias_reales": y.values,
            "prob_riesgo": prob,
            "los_dias_predichos": pred,
            "error_dias": pred - y.values,
            "es_urgencia": holdout_df[URGENCY_COL].values,
            "segmento": segment,
        })
        rows.append(part)
        print(f"  Holdout evaluado: {len(part)} pacientes")

    predicciones = pd.concat(rows, ignore_index=True)
    df_global, _ = evaluar_predicciones(model_label, predicciones, model_name)
    print("\nMetricas globales:")
    print(df_global.to_string(index=False))


def train_lr_model(alpha: float = 1.0) -> None:
    print("Entrenamiento operacional baseline LR/Ridge")
    for segment in SEGMENTS:
        print(f"\n[LR] Segmento: {segment}")
        train_df = load_segment_split(segment, "train")
        X, y = prepare_xy(train_df)
        model = make_lr_regressor(alpha=alpha)
        model.fit(X, y)
        save_model_bundle(
            MODELS_DIR / f"reg_lr_{segment}.joblib",
            model,
            X.columns.tolist(),
            {"model_name": "lr", "segment": segment, "stage": "baseline_ridge", "alpha": alpha},
        )
        print(f"  Modelo LR guardado: reg_lr_{segment}.joblib")


def evaluate_lr_model() -> None:
    print("Evaluacion holdout baseline LR/Ridge")
    rows = []
    for segment in SEGMENTS:
        print(f"\n[LR] Segmento: {segment}")
        holdout_df = load_segment_split(segment, "holdout")
        X, y = prepare_xy(holdout_df)
        bundle = load_model_bundle(MODELS_DIR / f"reg_lr_{segment}.joblib")
        model = bundle["model"]
        features = bundle["features"]
        pred = np.clip(model.predict(X[features]), 0, None)
        rows.append(pd.DataFrame({
            "case_id": holdout_df[ID_COL].values,
            "los_dias_reales": y.values,
            "prob_riesgo": np.nan,
            "los_dias_predichos": pred,
            "error_dias": pred - y.values,
            "es_urgencia": holdout_df[URGENCY_COL].values,
            "segmento": segment,
        }))
        print(f"  Holdout evaluado: {len(holdout_df)} pacientes")

    predicciones = pd.concat(rows, ignore_index=True)
    df_global, _ = evaluar_predicciones("LR", predicciones, "lr")
    print("\nMetricas globales:")
    print(df_global.to_string(index=False))


def build_model_comparison() -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for key in ["xgb", "rf", "lr"]:
        path = REPORTS_DIR / f"metricas_holdout_{key}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No hay metricas holdout para comparar en ml_operacional/reports")

    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison.sort_values("mae").reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "comparacion_final_modelos.csv", index=False)

    markdown = [
        "# Comparacion Final de Modelos Operacionales",
        "",
        dataframe_to_markdown(
            comparison[["modelo", "n_casos", "mae", "rmse", "medae", "me", "pup", "mae_asimetrico_alpha_2"]]
        ),
        "",
    ]
    (REPORTS_DIR / "comparacion_final_modelos.md").write_text("\n".join(markdown), encoding="utf-8")
    return comparison
