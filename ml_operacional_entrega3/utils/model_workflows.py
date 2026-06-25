"""Workflows reutilizables de entrenamiento y evaluacion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ml_operacional_entrega3.utils.pipeline_operacional import (
    DATA_SPLITS_DIR,
    ID_COL,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SEGMENTS,
    TARGET_COL,
    URGENCY_COL,
    binary_target_los14,
    calcular_metricas_globales,
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

STABILITY_METRICS = [
    "mae",
    "rmse",
    "medae",
    "me",
    "pup",
    "mae_asimetrico_alpha_2",
    "precision_plos_14",
    "recall_plos_14",
    "f1_plos_14",
]


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


def _prepare_lr_baseline_xy(df: pd.DataFrame, segment: str) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara el baseline de regresion lineal usando los splits operacionales actuales."""
    model_df = df.copy()
    drop_cols = [
        ID_COL,
        TARGET_COL,
        URGENCY_COL,
        "prob_los_14",
        "int_charlson_diag",
        "int_charlson_proc",
        "int_proc_diag",
    ]
    X = model_df.drop(columns=[col for col in drop_cols if col in model_df.columns]).copy()

    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Columnas no numericas en features de regresion lineal: {list(non_numeric)}")

    return X, model_df[TARGET_COL].copy()


def train_lr_model() -> None:
    print("Entrenamiento operacional baseline Regresion Lineal basica")
    for segment in SEGMENTS:
        print(f"\n[LR] Segmento: {segment}")
        train_df = load_segment_split(segment, "train")
        X, y = _prepare_lr_baseline_xy(train_df, segment)
        model = make_lr_regressor()
        model.fit(X, y)
        save_model_bundle(
            MODELS_DIR / f"reg_lr_{segment}.joblib",
            model,
            X.columns.tolist(),
            {
                "model_name": "lr",
                "segment": segment,
                "stage": "baseline_linear_regression",
                "regularization": "none",
                "interactions": [],
                "drops_es_urgencia": True,
                "threshold_los": 14,
            },
        )
        print(
            f"  Modelo Regresion Lineal guardado: reg_lr_{segment}.joblib | "
            f"regularizacion=ninguna | features={len(X.columns)}"
        )


def _predict_lr_split(segment: str, split: str) -> pd.DataFrame:
    df = load_segment_split(segment, split)
    X, y = _prepare_lr_baseline_xy(df, segment)
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
    print("Evaluacion train/holdout baseline Regresion Lineal basica")
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


def _safe_stratified_kfold(y_binary: np.ndarray, n_splits: int = 5) -> StratifiedKFold:
    min_class = int(np.min(np.bincount(y_binary))) if len(np.unique(y_binary)) == 2 else 0
    if min_class < 2:
        raise ValueError("No hay suficientes casos PLOS/no PLOS para K-Fold estratificado")
    return StratifiedKFold(
        n_splits=min(n_splits, min_class),
        shuffle=True,
        random_state=RANDOM_STATE,
    )


def _append_fold_metrics(
    rows: list[dict],
    model_label: str,
    fold: int,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    segment: str | None = None,
) -> None:
    row = {"modelo": model_label, "fold": fold}
    if segment is not None:
        row["segmento"] = segment
    row.update(calcular_metricas_globales(y_true, y_pred))
    rows.append(row)


def _stability_two_stage_model(model_name: str) -> tuple[list[dict], list[dict]]:
    model_label = model_name.upper()
    global_fold_predictions: dict[int, list[pd.DataFrame]] = {}
    segment_rows: list[dict] = []

    for segment in SEGMENTS:
        train_df = load_segment_split(segment, "train")
        X, y = prepare_xy(train_df)
        y_binary = binary_target_los14(y)
        clf_params = load_best_params(model_name, "clf", segment, y_binary=y_binary)
        reg_params = load_best_params(model_name, "reg", segment)
        kfold = _safe_stratified_kfold(y_binary)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_binary), 1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            y_binary_train = y_binary[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            clf = make_classifier(model_name, clf_params)
            clf.fit(X_train, y_binary_train)

            X_reg_train = X_train.copy()
            X_reg_train["prob_los_14"] = clf.predict_proba(X_train)[:, 1]
            reg = make_regressor(model_name, reg_params)
            reg.fit(X_reg_train, y_train)

            X_reg_val = X_val.copy()
            X_reg_val["prob_los_14"] = clf.predict_proba(X_val)[:, 1]
            pred = np.clip(reg.predict(X_reg_val), 0, None)

            _append_fold_metrics(segment_rows, model_label, fold, y_val, pred, segment=segment)
            global_fold_predictions.setdefault(fold, []).append(pd.DataFrame({
                "los_dias_reales": y_val.to_numpy(),
                "los_dias_predichos": pred,
            }))

    global_rows: list[dict] = []
    for fold in sorted(global_fold_predictions):
        fold_predictions = pd.concat(global_fold_predictions[fold], ignore_index=True)
        _append_fold_metrics(
            global_rows,
            model_label,
            fold,
            fold_predictions["los_dias_reales"],
            fold_predictions["los_dias_predichos"].to_numpy(),
        )
    return global_rows, segment_rows


def _stability_lr_model() -> tuple[list[dict], list[dict]]:
    global_fold_predictions: dict[int, list[pd.DataFrame]] = {}
    segment_rows: list[dict] = []

    for segment in SEGMENTS:
        train_df = load_segment_split(segment, "train")
        X, y = _prepare_lr_baseline_xy(train_df, segment)
        y_binary = binary_target_los14(y)
        kfold = _safe_stratified_kfold(y_binary)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y_binary), 1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_val = y.iloc[val_idx]

            model = make_lr_regressor()
            model.fit(X_train, y_train)
            pred = np.clip(model.predict(X_val), 0, None)

            _append_fold_metrics(segment_rows, "LR", fold, y_val, pred, segment=segment)
            global_fold_predictions.setdefault(fold, []).append(pd.DataFrame({
                "los_dias_reales": y_val.to_numpy(),
                "los_dias_predichos": pred,
            }))

    global_rows: list[dict] = []
    for fold in sorted(global_fold_predictions):
        fold_predictions = pd.concat(global_fold_predictions[fold], ignore_index=True)
        _append_fold_metrics(
            global_rows,
            "LR",
            fold,
            fold_predictions["los_dias_reales"],
            fold_predictions["los_dias_predichos"].to_numpy(),
        )
    return global_rows, segment_rows


def build_stability_study() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evalua estabilidad K-Fold con la receta final y parametros ya seleccionados."""
    global_rows: list[dict] = []
    segment_rows: list[dict] = []

    for model_name in ["xgb", "rf"]:
        print(f"Calculando estabilidad K-Fold para {model_name.upper()}")
        model_global_rows, model_segment_rows = _stability_two_stage_model(model_name)
        global_rows.extend(model_global_rows)
        segment_rows.extend(model_segment_rows)

    print("Calculando estabilidad K-Fold para LR")
    lr_global_rows, lr_segment_rows = _stability_lr_model()
    global_rows.extend(lr_global_rows)
    segment_rows.extend(lr_segment_rows)

    global_df = pd.DataFrame(global_rows).sort_values(["modelo", "fold"]).reset_index(drop=True)
    segment_df = pd.DataFrame(segment_rows).sort_values(["modelo", "segmento", "fold"]).reset_index(drop=True)
    summary_df = _summarize_stability(global_df)

    global_df.to_csv(REPORTS_DIR / "estabilidad_kfold_global.csv", index=False)
    segment_df.to_csv(REPORTS_DIR / "estabilidad_kfold_por_segmento.csv", index=False)
    summary_df.to_csv(REPORTS_DIR / "resumen_estabilidad_kfold.csv", index=False)
    return global_df, segment_df, summary_df


def _summarize_stability(global_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_label, group in global_df.groupby("modelo", sort=False):
        row = {
            "modelo": model_label,
            "n_folds": int(group["fold"].nunique()),
            "n_casos_promedio_fold": float(group["n_casos"].mean()),
        }
        for metric in STABILITY_METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _format_stability_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary_df.iterrows():
        formatted = {
            "modelo": row["modelo"],
            "n_folds": int(row["n_folds"]),
            "n_casos_promedio_fold": f"{row['n_casos_promedio_fold']:.0f}",
        }
        for metric in STABILITY_METRICS:
            formatted[metric] = f"{row[f'{metric}_mean']:.4f} +/- {row[f'{metric}_std']:.4f}"
        rows.append(formatted)
    return pd.DataFrame(rows)


def _global_gap_summary(gap_comparison: pd.DataFrame) -> pd.DataFrame:
    if gap_comparison.empty:
        return gap_comparison
    metricas = ["mae", "rmse", "mae_asimetrico_alpha_2", "recall_plos_14", "f1_plos_14"]
    cols = ["modelo", "metrica", "train", "holdout", "gap_holdout_minus_train", "ratio_holdout_train"]
    mask = (gap_comparison["alcance"] == "global") & gap_comparison["metrica"].isin(metricas)
    return gap_comparison.loc[mask, cols].reset_index(drop=True)


def _weighted_non_plos_summary(tramo_comparison: pd.DataFrame) -> pd.DataFrame:
    if tramo_comparison.empty:
        return tramo_comparison

    non_plos = tramo_comparison[tramo_comparison["tramo"] != "14+ (PLOS)"].copy()
    rows = []
    for model_label, group in non_plos.groupby("modelo", sort=False):
        weights = group["n_casos"].to_numpy(dtype=float)
        n_cases = int(weights.sum())
        rows.append({
            "modelo": model_label,
            "n_casos_los_lt_14": n_cases,
            "mae_los_lt_14": float(np.average(group["mae"], weights=weights)),
            "rmse_los_lt_14": float(np.sqrt(np.sum(weights * (group["rmse"] ** 2)) / n_cases)),
            "me_los_lt_14": float(np.average(group["me"], weights=weights)),
            "pup_los_lt_14": float(np.average(group["pup"], weights=weights)),
            "mae_asim_los_lt_14": float(np.average(group["mae_asimetrico_alpha_2"], weights=weights)),
            "pct_le_1d_los_lt_14": float(np.average(group["pct_error_abs_le_1d"], weights=weights)),
            "pct_le_3d_los_lt_14": float(np.average(group["pct_error_abs_le_3d"], weights=weights)),
            "pct_le_7d_los_lt_14": float(np.average(group["pct_error_abs_le_7d"], weights=weights)),
        })
    return pd.DataFrame(rows).sort_values("mae_los_lt_14").reset_index(drop=True)


def _clinical_interpretation(comparison: pd.DataFrame, tramo_comparison: pd.DataFrame, non_plos_summary: pd.DataFrame) -> str:
    if comparison.empty or tramo_comparison.empty or non_plos_summary.empty:
        return "_Sin datos suficientes para interpretar._"

    best_global = comparison.sort_values("mae").iloc[0]
    best_non_plos = non_plos_summary.sort_values("mae_los_lt_14").iloc[0]
    best_0_2 = tramo_comparison[tramo_comparison["tramo"] == "0-2"].sort_values("mae").iloc[0]
    best_3_6 = tramo_comparison[tramo_comparison["tramo"] == "3-6"].sort_values("mae").iloc[0]
    best_7_13 = tramo_comparison[tramo_comparison["tramo"] == "7-13"].sort_values("mae").iloc[0]
    best_plos = tramo_comparison[tramo_comparison["tramo"] == "14+ (PLOS)"].sort_values("recall_plos_14", ascending=False).iloc[0]

    return "\n".join([
        f"- El mejor MAE global en holdout es {best_global['modelo']} ({best_global['mae']:.4f}).",
        f"- Para LOS < 14 dias, que concentra la mayoria de los casos, el menor MAE ponderado es {best_non_plos['modelo']} ({best_non_plos['mae_los_lt_14']:.4f}).",
        f"- En el tramo 0-2 dias gana {best_0_2['modelo']} (MAE {best_0_2['mae']:.4f}); en 3-6 gana {best_3_6['modelo']} (MAE {best_3_6['mae']:.4f}); en 7-13 gana {best_7_13['modelo']} (MAE {best_7_13['mae']:.4f}).",
        f"- En PLOS, el mejor recall lo obtiene {best_plos['modelo']} ({best_plos['recall_plos_14']:.4f}), lo que reduce el riesgo de no anticipar estancias prolongadas.",
        "- Lectura clinica: LR corresponde al baseline de regresion lineal basica. Si LR gana en algun tramo corto, eso indica que una regla lineal simple ya captura parte importante del patron local; si XGB gana globalmente o en PLOS, mantiene ventaja operacional por balancear error general y deteccion de estancias prolongadas.",
    ])


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
    tramo_frames = []
    for key in ["xgb", "rf", "lr"]:
        segment_path = REPORTS_DIR / f"metricas_por_segmento_holdout_{key}.csv"
        gap_path = REPORTS_DIR / f"comparacion_train_holdout_{key}.csv"
        tramo_path = REPORTS_DIR / f"metricas_por_tramo_holdout_{key}.csv"
        if segment_path.exists():
            segment_frames.append(pd.read_csv(segment_path))
        if gap_path.exists():
            gap_frames.append(pd.read_csv(gap_path))
        if tramo_path.exists():
            tramo_frames.append(pd.read_csv(tramo_path))
    segment_comparison = pd.concat(segment_frames, ignore_index=True) if segment_frames else pd.DataFrame()
    gap_comparison = pd.concat(gap_frames, ignore_index=True) if gap_frames else pd.DataFrame()
    tramo_comparison = pd.concat(tramo_frames, ignore_index=True) if tramo_frames else pd.DataFrame()
    non_plos_summary = _weighted_non_plos_summary(tramo_comparison)
    if not segment_comparison.empty:
        segment_comparison.to_csv(REPORTS_DIR / "comparacion_final_por_segmento.csv", index=False)
    if not gap_comparison.empty:
        gap_comparison.to_csv(REPORTS_DIR / "comparacion_final_train_vs_holdout.csv", index=False)
    if not tramo_comparison.empty:
        tramo_comparison.to_csv(REPORTS_DIR / "comparacion_final_por_tramo.csv", index=False)
    if not non_plos_summary.empty:
        non_plos_summary.to_csv(REPORTS_DIR / "comparacion_final_los_menor_14.csv", index=False)

    _, _, stability_summary = build_stability_study()

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
        "## Holdout por Tramo LOS",
        "",
        "Los tramos `0-2`, `3-6` y `7-13` corresponden a LOS < 14 dias. Son clinicamente relevantes porque concentran la mayor parte de los casos.",
        "",
        dataframe_to_markdown(
            tramo_comparison[[
                "modelo",
                "tramo",
                "n_casos",
                "mae",
                "rmse",
                "medae",
                "me",
                "pup",
                "mae_asimetrico_alpha_2",
                "pct_error_abs_le_1d",
                "pct_error_abs_le_3d",
                "pct_error_abs_le_7d",
                "los_real_promedio",
                "los_pred_promedio",
                "n_plos_pred",
                "recall_plos_14",
                "f1_plos_14",
            ]]
            if not tramo_comparison.empty else tramo_comparison
        ),
        "",
        "## Sintesis LOS < 14",
        "",
        "Resumen ponderado por cantidad de casos en los tramos no PLOS (`0-2`, `3-6`, `7-13`).",
        "",
        dataframe_to_markdown(non_plos_summary),
        "",
        "## Interpretacion Clinica Final",
        "",
        _clinical_interpretation(comparison, tramo_comparison, non_plos_summary),
        "",
        "## Estabilidad del Modelo (K-Fold sobre Train)",
        "",
        "Este diagnostico reentrena la receta final de cada modelo en 5 folds del train operacional y evalua el fold restante. No vuelve a tunear hiperparametros y no usa el holdout final.",
        "",
        dataframe_to_markdown(_format_stability_summary(stability_summary)),
        "",
        "## Diagnostico de Sobreajuste Train vs Holdout",
        "",
        "El gap compara el rendimiento del modelo final ya entrenado contra el holdout. Gaps grandes en MAE/RMSE o caidas fuertes de recall/F1 PLOS indican mayor riesgo de sobreajuste.",
        "",
        dataframe_to_markdown(_global_gap_summary(gap_comparison)),
        "",
        "## Gap Train vs Holdout Detalle",
        "",
        dataframe_to_markdown(gap_comparison if not gap_comparison.empty else gap_comparison),
        "",
    ]
    (REPORTS_DIR / "comparacion_final_modelos.md").write_text("\n".join(markdown), encoding="utf-8")
    return comparison
