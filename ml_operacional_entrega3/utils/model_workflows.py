"""Workflows reutilizables de entrenamiento y evaluacion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_operacional_entrega3.utils.pipeline_operacional import (
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
    guardar_reporte_train_holdout,
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


def _predict_two_stage_split(model_name: str, model_label: str, segment: str, split: str) -> pd.DataFrame:
    df = load_segment_split(segment, split)
    X, y = prepare_xy(df)

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

    print(f"  {split} evaluado: {len(df)} pacientes")
    return pd.DataFrame({
        "case_id": df[ID_COL].values,
        "los_dias_reales": y.values,
        "prob_riesgo": prob,
        "los_dias_predichos": pred,
        "error_dias": pred - y.values,
        "es_urgencia": df[URGENCY_COL].values,
        "segmento": segment,
    })


def evaluate_two_stage_model(model_name: str) -> None:
    model_label = model_name.upper()
    print(f"Evaluacion train/holdout operacional: {model_label}")
    rows_by_split = {"train": [], "holdout": []}

    for segment in SEGMENTS:
        print(f"\n[{model_label}] Segmento: {segment}")
        for split in ["train", "holdout"]:
            rows_by_split[split].append(_predict_two_stage_split(model_name, model_label, segment, split))

    train_predictions = pd.concat(rows_by_split["train"], ignore_index=True)
    holdout_predictions = pd.concat(rows_by_split["holdout"], ignore_index=True)
    train_artifacts = evaluar_predicciones(model_label, train_predictions, model_name, split_name="train")
    holdout_artifacts = evaluar_predicciones(model_label, holdout_predictions, model_name, split_name="holdout")
    gap = guardar_reporte_train_holdout(model_label, model_name, train_artifacts, holdout_artifacts)

    print("\nMetricas holdout por segmento:")
    print(holdout_artifacts["segmentos"].to_string(index=False))
    print("\nGap train vs holdout:")
    print(gap.to_string(index=False))


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


def _predict_lr_split(segment: str, split: str) -> pd.DataFrame:
    df = load_segment_split(segment, split)
    X, y = prepare_xy(df)
    bundle = load_model_bundle(MODELS_DIR / f"reg_lr_{segment}.joblib")
    model = bundle["model"]
    features = bundle["features"]
    pred = np.clip(model.predict(X[features]), 0, None)
    print(f"  {split} evaluado: {len(df)} pacientes")
    return pd.DataFrame({
        "case_id": df[ID_COL].values,
        "los_dias_reales": y.values,
        "prob_riesgo": np.nan,
        "los_dias_predichos": pred,
        "error_dias": pred - y.values,
        "es_urgencia": df[URGENCY_COL].values,
        "segmento": segment,
    })


def evaluate_lr_model() -> None:
    print("Evaluacion train/holdout baseline LR/Ridge")
    rows_by_split = {"train": [], "holdout": []}
    for segment in SEGMENTS:
        print(f"\n[LR] Segmento: {segment}")
        for split in ["train", "holdout"]:
            rows_by_split[split].append(_predict_lr_split(segment, split))

    train_predictions = pd.concat(rows_by_split["train"], ignore_index=True)
    holdout_predictions = pd.concat(rows_by_split["holdout"], ignore_index=True)
    train_artifacts = evaluar_predicciones("LR", train_predictions, "lr", split_name="train")
    holdout_artifacts = evaluar_predicciones("LR", holdout_predictions, "lr", split_name="holdout")
    gap = guardar_reporte_train_holdout("LR", "lr", train_artifacts, holdout_artifacts)

    print("\nMetricas holdout por segmento:")
    print(holdout_artifacts["segmentos"].to_string(index=False))
    print("\nGap train vs holdout:")
    print(gap.to_string(index=False))


def build_model_comparison() -> pd.DataFrame:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for key in ["xgb", "rf", "lr"]:
        path = REPORTS_DIR / f"metricas_holdout_{key}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No hay metricas holdout para comparar en ml_operacional_entrega3/reports")

    comparison = pd.concat(frames, ignore_index=True)
    comparison = comparison.sort_values("mae").reset_index(drop=True)
    comparison.to_csv(REPORTS_DIR / "comparacion_final_modelos.csv", index=False)

    segment_frames = []
    gap_frames = []
    for key in ["xgb", "rf", "lr"]:
        segment_path = REPORTS_DIR / f"metricas_por_segmento_holdout_{key}.csv"
        gap_path = REPORTS_DIR / f"comparacion_train_holdout_{key}.csv"
        if segment_path.exists():
            segment_frames.append(pd.read_csv(segment_path))
        if gap_path.exists():
            gap_frames.append(pd.read_csv(gap_path))
    segment_comparison = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    gap_comparison = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    if not segment_comparison.empty:
        segment_comparison.to_csv(REPORTS_DIR / "comparacion_final_por_segmento.csv", index=False)
    if not gap_comparison.empty:
        gap_comparison.to_csv(REPORTS_DIR / "comparacion_final_train_vs_holdout.csv", index=False)

    markdown = [
        "# Comparacion Final de Modelos Operacionales",
        "",
        "PLOS se define como `LOS >= 14` dias.",
        "",
        "## Holdout Global",
        "",
        dataframe_to_markdown(
            comparison[[
                "modelo",
                "n_casos",
                "mae",
                "rmse",
                "medae",
                "me",
                "pup",
                "mae_asimetrico_alpha_2",
                "precision_plos_14",
                "recall_plos_14",
                "f1_plos_14",
            ]]
        ),
        "",
        "## Holdout por Segmento",
        "",
        dataframe_to_markdown(
            segment_comparison[[
                "modelo",
                "segmento",
                "n_casos",
                "mae",
                "rmse",
                "medae",
                "me",
                "pup",
                "mae_asimetrico_alpha_2",
                "precision_plos_14",
                "recall_plos_14",
                "f1_plos_14",
            ]]
            if not segment_comparison.empty else segment_comparison
        ),
        "",
        "## Gap Train vs Holdout",
        "",
        dataframe_to_markdown(gap_comparison if not gap_comparison.empty else gap_comparison),
        "",
    ]
    (REPORTS_DIR / "comparacion_final_modelos.md").write_text("\n".join(markdown), encoding="utf-8")
    return comparison
