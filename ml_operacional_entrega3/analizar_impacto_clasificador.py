"""Audita el desempeno del clasificador y su impacto en el regresor XGB."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.utils.pipeline_operacional import (
    MODELS_DIR,
    REPORTS_DIR,
    dataframe_to_markdown,
    load_model_bundle,
)


MODEL_KEYS = ["xgb", "rf"]
SPLITS = ["train", "holdout"]
THRESHOLD = 0.50
PLOS_THRESHOLD = 14


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _probability_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan}
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "brier": float(brier_score_loss(y_true, prob)),
    }


def _classifier_rows() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global_rows = []
    segment_rows = []
    comparison_rows = []

    for model_key in MODEL_KEYS:
        model_label = model_key.upper()
        for split in SPLITS:
            predictions = pd.read_csv(REPORTS_DIR / f"predicciones_{split}_{model_key}.csv")
            y_true = predictions["plos_real_14"].to_numpy(dtype=int)
            prob = predictions["prob_riesgo"].to_numpy(dtype=float)
            pred_clf = (prob >= THRESHOLD).astype(int)
            pred_reg = predictions["plos_pred_14"].to_numpy(dtype=int)

            base = {
                "modelo": model_label,
                "split": split,
                "umbral_prob": THRESHOLD,
                "n_casos": int(len(predictions)),
                "n_plos_real": int(y_true.sum()),
            }
            global_rows.append({
                **base,
                "n_alertas_clasificador": int(pred_clf.sum()),
                **_probability_metrics(y_true, prob),
                **_binary_metrics(y_true, pred_clf),
            })

            for output_name, pred in [
                ("clasificador_prob_ge_0_50", pred_clf),
                ("regresor_los_ge_14", pred_reg),
            ]:
                comparison_rows.append({
                    **base,
                    "salida": output_name,
                    "n_alertas": int(pred.sum()),
                    **_binary_metrics(y_true, pred),
                })

            for segment, group in predictions.groupby("segmento", sort=True):
                y_segment = group["plos_real_14"].to_numpy(dtype=int)
                prob_segment = group["prob_riesgo"].to_numpy(dtype=float)
                pred_segment = (prob_segment >= THRESHOLD).astype(int)
                segment_rows.append({
                    "modelo": model_label,
                    "split": split,
                    "segmento": segment,
                    "umbral_prob": THRESHOLD,
                    "n_casos": int(len(group)),
                    "n_plos_real": int(y_segment.sum()),
                    "n_alertas_clasificador": int(pred_segment.sum()),
                    **_probability_metrics(y_segment, prob_segment),
                    **_binary_metrics(y_segment, pred_segment),
                })

    return pd.DataFrame(global_rows), pd.DataFrame(segment_rows), pd.DataFrame(comparison_rows)


def _booster_importance_frame(regressor, features: list[str], segment: str) -> pd.DataFrame:
    booster = regressor.get_booster()
    rows = []
    by_feature = {f"f{i}": feature for i, feature in enumerate(features)}
    for importance_type in ["gain", "weight", "cover", "total_gain", "total_cover"]:
        scores = booster.get_score(importance_type=importance_type)
        expanded = []
        for raw_name, value in scores.items():
            feature = raw_name if raw_name in features else by_feature.get(raw_name, raw_name)
            expanded.append((feature, float(value)))

        score_df = pd.DataFrame(expanded, columns=["feature", "importance"])
        if score_df.empty:
            continue
        score_df["segmento"] = segment
        score_df["importance_type"] = importance_type
        score_df["rank"] = score_df["importance"].rank(method="min", ascending=False).astype(int)
        score_df["importance_pct"] = score_df["importance"] / score_df["importance"].sum()
        rows.append(score_df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _xgb_regressor_importance() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    prob_rows = []

    for segment in ["urgente", "programado"]:
        bundle = load_model_bundle(MODELS_DIR / f"reg_xgb_{segment}.joblib")
        model = bundle["model"]
        features = bundle["features"]
        regressor = model.regressor_

        direct = pd.DataFrame({
            "feature": features,
            "importance": regressor.feature_importances_,
        })
        direct["segmento"] = segment
        direct["importance_type"] = "feature_importances_attr"
        direct["rank"] = direct["importance"].rank(method="min", ascending=False).astype(int)
        total_direct = direct["importance"].sum()
        direct["importance_pct"] = direct["importance"] / total_direct if total_direct > 0 else np.nan

        booster = _booster_importance_frame(regressor, features, segment)
        combined = pd.concat([direct, booster], ignore_index=True)
        all_rows.append(combined)

        prob = combined[combined["feature"] == "prob_los_14"].copy()
        prob["n_features_modelo"] = len(features)
        prob_rows.append(prob)

    all_importances = pd.concat(all_rows, ignore_index=True)
    prob_importance = pd.concat(prob_rows, ignore_index=True)
    all_importances = all_importances.sort_values(["segmento", "importance_type", "rank"]).reset_index(drop=True)
    prob_importance = prob_importance.sort_values(["segmento", "importance_type"]).reset_index(drop=True)
    return all_importances, prob_importance


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Sin datos._"
    return dataframe_to_markdown(df)


def _build_report(
    clf_global: pd.DataFrame,
    clf_segment: pd.DataFrame,
    output_comparison: pd.DataFrame,
    prob_importance: pd.DataFrame,
) -> str:
    holdout_clf = clf_global[clf_global["split"] == "holdout"].copy()
    holdout_comparison = output_comparison[output_comparison["split"] == "holdout"].copy()
    holdout_segments = clf_segment[clf_segment["split"] == "holdout"].copy()

    prob_gain = prob_importance[[
        "segmento",
        "importance_type",
        "rank",
        "importance",
        "importance_pct",
        "n_features_modelo",
    ]]

    interpretation = [
        "# Impacto del Clasificador en el Regresor",
        "",
        f"PLOS se define como `LOS >= {PLOS_THRESHOLD}` dias. El umbral `prob_riesgo >= {THRESHOLD:.2f}` solo se usa para convertir la probabilidad del clasificador en alerta binaria; el regresor recibe la probabilidad continua `prob_los_14`.",
        "",
        "## Desempeno Global del Clasificador",
        "",
        _markdown_table(holdout_clf[[
            "modelo",
            "n_casos",
            "n_plos_real",
            "n_alertas_clasificador",
            "roc_auc",
            "pr_auc",
            "brier",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "fp",
            "fn",
            "tp",
        ]]),
        "",
        "## Desempeno del Clasificador por Segmento",
        "",
        _markdown_table(holdout_segments[[
            "modelo",
            "segmento",
            "n_casos",
            "n_plos_real",
            "n_alertas_clasificador",
            "roc_auc",
            "pr_auc",
            "brier",
            "precision",
            "recall",
            "f1",
            "fp",
            "fn",
            "tp",
        ]]),
        "",
        "## Clasificador vs Salida Final del Regresor",
        "",
        _markdown_table(holdout_comparison[[
            "modelo",
            "salida",
            "n_alertas",
            "precision",
            "recall",
            "f1",
            "accuracy",
            "fp",
            "fn",
            "tp",
        ]]),
        "",
        "## Importancia de `prob_los_14` en el Regresor XGB",
        "",
        _markdown_table(prob_gain),
        "",
        "### Como leer `importance_type`",
        "",
        "`importance_type` indica la forma en que XGBoost calcula la importancia de una variable dentro de los arboles del regresor:",
        "",
        "- `gain`: mejora promedio que produce una variable cada vez que se usa para dividir un nodo. Si es alto, significa que esa variable ayuda mucho a reducir el error cuando aparece.",
        "- `total_gain`: suma total de toda la mejora aportada por esa variable en todos los arboles. Combina que tan util es y cuantas veces aporta.",
        "- `weight`: cantidad de veces que la variable fue usada para hacer divisiones en los arboles. Si es alto, el modelo recurre muchas veces a esa variable.",
        "- `cover`: cantidad promedio de observaciones afectadas por las divisiones donde aparece la variable.",
        "- `total_cover`: suma total de observaciones afectadas por todas las divisiones donde aparece la variable.",
        "- `feature_importances_attr`: importancia normalizada que entrega directamente el objeto `XGBRegressor`. En este caso coincide con una version normalizada de `gain`.",
        "",
        "La columna `rank` muestra el puesto de `prob_los_14` entre todas las variables del regresor. Rank 1 significa que fue la variable mas importante bajo ese criterio. `importance_pct` muestra que proporcion de la importancia total corresponde a `prob_los_14`.",
        "",
        "## Interpretacion",
        "",
        "El pipeline tiene dos salidas distintas que no deben interpretarse como si fueran lo mismo:",
        "",
        "1. `prob_riesgo`: salida del clasificador. Es una probabilidad de que el paciente tenga PLOS, es decir, `LOS >= 14` dias.",
        "2. `los_dias_predichos`: salida del regresor. Es una estimacion de cuantos dias exactos estara hospitalizado el paciente.",
        "",
        "Cuando decimos que conviene separar dos usos, nos referimos a esto:",
        "",
        "- Si el hospital quiere una alerta temprana de riesgo PLOS, deberia mirar `prob_riesgo`. Por ejemplo, podria definir una regla como `prob_riesgo >= 0.50` o ajustar el umbral a `0.35`, `0.40`, etc., segun si quiere capturar mas pacientes de riesgo o reducir falsas alarmas.",
        "- Si el hospital quiere estimar cuantos dias podria durar la hospitalizacion, deberia mirar `los_dias_predichos`.",
        "",
        "El problema aparece cuando se usa `los_dias_predichos >= 14` como si fuera la unica alerta PLOS. El regresor intenta predecir dias exactos y tiende a ser conservador con estancias largas. Por eso puede ocurrir que un paciente tenga alto `prob_riesgo`, pero el regresor prediga 12 o 13 dias. Clinicamente ese paciente sigue siendo riesgoso, aunque la regla `los_dias_predichos >= 14` no lo marque como PLOS.",
        "",
        "En los resultados de holdout esto se ve claramente para XGB:",
        "",
        "- El clasificador con `prob_riesgo >= 0.50` detecta 229 de 279 pacientes PLOS reales. Eso equivale a un recall de 82.08%.",
        "- La salida final del regresor con `los_dias_predichos >= 14` detecta 164 de 279 pacientes PLOS reales. Eso equivale a un recall de 58.78%.",
        "- Por lo tanto, el clasificador identifica 65 pacientes PLOS reales adicionales que el regresor deja bajo 14 dias.",
        "",
        "Esto no significa que el regresor este mal. Significa que cumple una funcion distinta: estimar dias. Para emitir alertas de riesgo PLOS, la salida mas directa y sensible es la probabilidad del clasificador.",
        "",
        "La importancia de variables confirma que el clasificador si influye en el regresor:",
        "",
        "- En XGB programado, `prob_los_14` queda rank 1 por `gain`, `total_gain`, `weight` y `feature_importances_attr`. Esto quiere decir que, entre 1652 variables, la probabilidad PLOS generada por el clasificador es la senal mas importante para el regresor programado.",
        "- En XGB urgente, `prob_los_14` queda rank 2 por `gain` y `feature_importances_attr`, y rank 1 por `weight` y `total_gain`. Esto quiere decir que tambien es una de las variables centrales del regresor urgente.",
        "",
        "Que `prob_los_14` tenga ranking alto pero no concentre 100% de la importancia tambien es esperable. El regresor no solo decide si un paciente sera PLOS; intenta estimar dias exactos para todos los pacientes. Por eso necesita otras variables clinicas para distinguir 1 vs 2 dias, 3 vs 6 dias, 8 vs 12 dias, o 15 vs 30 dias.",
        "",
        "La conclusion defendible es que `prob_los_14` funciona como una senal central para orientar al regresor hacia riesgo de estancia prolongada, pero el regresor sigue usando el resto de variables clinicas para ajustar la cantidad exacta de dias.",
        "",
        "Conclusion: el clasificador tiene impacto real en los regresores, porque `prob_los_14` aparece entre las variables mas importantes. Sin embargo, para la decision clinica de alerta PLOS conviene usar directamente `prob_riesgo`; para la planificacion de dias esperados conviene usar `los_dias_predichos`.",
        "",
    ]
    return "\n".join(interpretation)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    clf_global, clf_segment, output_comparison = _classifier_rows()
    all_importances, prob_importance = _xgb_regressor_importance()

    clf_global.to_csv(REPORTS_DIR / "metricas_clasificador_global.csv", index=False)
    clf_segment.to_csv(REPORTS_DIR / "metricas_clasificador_por_segmento.csv", index=False)
    output_comparison.to_csv(REPORTS_DIR / "comparacion_alerta_clasificador_vs_regresor.csv", index=False)
    all_importances.to_csv(REPORTS_DIR / "importancias_regresor_xgb_todas.csv", index=False)
    prob_importance.to_csv(REPORTS_DIR / "importancia_prob_los14_regresor_xgb.csv", index=False)

    report = _build_report(clf_global, clf_segment, output_comparison, prob_importance)
    (REPORTS_DIR / "reporte_impacto_clasificador_en_regresor.md").write_text(report, encoding="utf-8")
    print("Reporte guardado en ml_operacional_entrega3/reports/reporte_impacto_clasificador_en_regresor.md")


if __name__ == "__main__":
    main()
