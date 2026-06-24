"""Escenario 1: sensibilidad al umbral que define PLOS."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.sensitivity.common import (  # noqa: E402
    BASE_PLOS_THRESHOLD,
    SensitivityOutput,
    delta_mae_pct,
    fit_predict_two_stage,
    metrics_by_tramo,
    print_dataframe,
    regression_metrics,
    write_result,
)


THRESHOLDS = [7, 14, 21, 27]


def run() -> list[SensitivityOutput]:
    print("\n=== Escenario 1: Sensibilidad al umbral PLOS ===")
    rows: list[dict] = []
    tramo_rows = []
    base_mae: float | None = None

    for threshold in THRESHOLDS:
        print(f"\n--- Umbral PLOS: LOS >= {threshold} dias ---")
        predictions = fit_predict_two_stage(
            threshold=threshold,
            prob_column=f"prob_los_{threshold}",
            adapt_scale_pos_weight=True,
            log_prefix="[E1]",
        )
        metrics = regression_metrics(
            predictions["los_dias_reales"],
            predictions["los_dias_predichos"],
            threshold=threshold,
        )
        if threshold == BASE_PLOS_THRESHOLD:
            base_mae = float(metrics["mae"])

        row = {
            "umbral_plos": threshold,
            "n_casos": metrics["n_casos"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "me": metrics["me"],
            "pup": metrics["pup"],
            "mae_asimetrico": metrics["mae_asimetrico"],
            "precision_plos": metrics["precision_plos"],
            "recall_plos": metrics["recall_plos"],
            "f1_plos": metrics["f1_plos"],
            "proporcion_plos": metrics["proporcion_plos"],
            "n_plos_real": metrics["n_plos_real"],
            "n_plos_pred": metrics["n_plos_pred"],
        }
        rows.append(row)

        tramo_df = metrics_by_tramo(predictions, threshold=threshold)
        tramo_df.insert(0, "umbral_plos_analizado", threshold)
        tramo_rows.append(tramo_df)
        print_dataframe(pd.DataFrame([row]))

    if base_mae is not None:
        for row in rows:
            row["delta_mae_pct_vs_umbral_14"] = delta_mae_pct(float(row["mae"]), base_mae)

    resultados = pd.DataFrame(rows)
    por_tramo = pd.concat(tramo_rows, ignore_index=True)
    outputs = [
        write_result(resultados, "escenario_1_resultados.csv"),
        write_result(por_tramo, "escenario_1_resultados_por_tramo.csv"),
    ]
    print("\nResumen Escenario 1:")
    print_dataframe(resultados)
    return outputs


if __name__ == "__main__":
    run()
