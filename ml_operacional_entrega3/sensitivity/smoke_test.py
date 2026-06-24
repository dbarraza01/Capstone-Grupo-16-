"""Prueba de humo liviana para el pipeline de sensibilidad.

No escribe outputs finales. Usa muestras pequenas y pocos arboles para verificar
que las rutas de entrenamiento, prediccion y metricas no fallen en runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.sensitivity.common import (  # noqa: E402
    BASE_PLOS_THRESHOLD,
    fit_predict_direct_regressor,
    fit_predict_two_stage,
    load_holdout_predictions_with_prob,
    load_params_by_segment,
    plos_confusion_metrics,
    pr_curve_dataframe,
    print_dataframe,
    regression_metrics,
    shrink_xgb_params,
)
from ml_operacional_entrega3.sensitivity.escenario_4_hiperparametros import perturb_params  # noqa: E402


SMOKE_TRAIN_ROWS = 120
SMOKE_HOLDOUT_ROWS = 60
SMOKE_ESTIMATORS = 3


def _small_params(kind: str) -> dict[str, dict]:
    return {
        segment: shrink_xgb_params(params, n_estimators=SMOKE_ESTIMATORS)
        for segment, params in load_params_by_segment(kind).items()
    }


def _print_metric_row(label: str, predictions: pd.DataFrame, threshold: int = BASE_PLOS_THRESHOLD) -> None:
    metrics = regression_metrics(
        predictions["los_dias_reales"],
        predictions["los_dias_predichos"],
        threshold=threshold,
    )
    row = {
        "prueba": label,
        "n": metrics["n_casos"],
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "recall_plos": metrics["recall_plos"],
        "f1_plos": metrics["f1_plos"],
    }
    print_dataframe(pd.DataFrame([row]))


def run() -> None:
    print("=== Smoke test del analisis de sensibilidad ===")
    print(
        f"Modo liviano: train_rows={SMOKE_TRAIN_ROWS} por segmento, "
        f"holdout_rows={SMOKE_HOLDOUT_ROWS} por segmento, n_estimators={SMOKE_ESTIMATORS}"
    )

    clf_params = _small_params("clf")
    reg_params = _small_params("reg")

    print("\n[Smoke 1] Dos etapas con umbral PLOS 27")
    pred_threshold = fit_predict_two_stage(
        threshold=27,
        prob_column="prob_los_27",
        clf_params_by_segment=clf_params,
        reg_params_by_segment=reg_params,
        adapt_scale_pos_weight=True,
        max_train_rows=SMOKE_TRAIN_ROWS,
        max_holdout_rows=SMOKE_HOLDOUT_ROWS,
        log_prefix="[SMOKE-E1]",
    )
    _print_metric_row("dos_etapas_umbral_27", pred_threshold, threshold=27)

    print("\n[Smoke 2] Ablation solo demografico-operacional")
    pred_ablation = fit_predict_two_stage(
        threshold=BASE_PLOS_THRESHOLD,
        feature_variant="solo_demografico_operacional",
        prob_column="prob_los_14",
        clf_params_by_segment=clf_params,
        reg_params_by_segment=reg_params,
        max_train_rows=SMOKE_TRAIN_ROWS,
        max_holdout_rows=SMOKE_HOLDOUT_ROWS,
        log_prefix="[SMOKE-E2]",
    )
    _print_metric_row("ablation_solo_demografico", pred_ablation)

    print("\n[Smoke 3] Regresor directo sin clasificador")
    pred_direct = fit_predict_direct_regressor(
        reg_params_by_segment=reg_params,
        max_train_rows=SMOKE_TRAIN_ROWS,
        max_holdout_rows=SMOKE_HOLDOUT_ROWS,
        log_prefix="[SMOKE-E2]",
    )
    _print_metric_row("sin_clasificador_1_etapa", pred_direct)

    print("\n[Smoke 4] Hiperparametros vecinos conservadores")
    conservative_clf = {
        segment: shrink_xgb_params(perturb_params(params, "Conservadora (mas regularizada)"), n_estimators=SMOKE_ESTIMATORS)
        for segment, params in load_params_by_segment("clf").items()
    }
    conservative_reg = {
        segment: shrink_xgb_params(perturb_params(params, "Conservadora (mas regularizada)"), n_estimators=SMOKE_ESTIMATORS)
        for segment, params in load_params_by_segment("reg").items()
    }
    pred_hyper = fit_predict_two_stage(
        threshold=BASE_PLOS_THRESHOLD,
        prob_column="prob_los_14",
        clf_params_by_segment=conservative_clf,
        reg_params_by_segment=conservative_reg,
        max_train_rows=SMOKE_TRAIN_ROWS,
        max_holdout_rows=SMOKE_HOLDOUT_ROWS,
        log_prefix="[SMOKE-E4]",
    )
    _print_metric_row("hiperparametros_conservadores", pred_hyper)

    print("\n[Smoke 5] Curva PR y politicas sobre predicciones holdout existentes")
    holdout = load_holdout_predictions_with_prob()
    curve = pr_curve_dataframe(
        holdout["los_dias_reales"].to_numpy(dtype=float),
        holdout["prob_riesgo"].to_numpy(dtype=float),
    )
    metrics = plos_confusion_metrics(
        holdout["los_dias_reales"],
        holdout["prob_riesgo"],
        0.5,
        score_is_probability=True,
    )
    print_dataframe(
        pd.DataFrame(
            [
                {
                    "prueba": "politica_0_50",
                    "puntos_curva_pr": len(curve),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            ]
        )
    )

    print("\nSmoke test completado sin escribir outputs finales.")


if __name__ == "__main__":
    run()

