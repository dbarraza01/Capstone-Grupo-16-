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
    metrics_by_segment_and_tramo,
    print_dataframe,
    summarize_predictions_by_segment,
    write_result,
)


THRESHOLDS = [7, 14, 21, 27]


def run() -> list[SensitivityOutput]:
    print("\n=== Escenario 1: Sensibilidad al umbral PLOS ===")
    rows: list[dict] = []
    tramo_rows = []

    for threshold in THRESHOLDS:
        print(f"\n--- Umbral PLOS: LOS >= {threshold} dias ---")
        predictions = fit_predict_two_stage(
            threshold=threshold,
            prob_column=f"prob_los_{threshold}",
            adapt_scale_pos_weight=True,
            log_prefix="[E1]",
        )

        metrics_df = summarize_predictions_by_segment(
            predictions,
            threshold=threshold,
            include_mae_ci=True,
        )
        metrics_df["umbral_plos"] = threshold
        keep_cols = [
            "umbral_plos",
            "segmento",
            "n_casos",
            "mae",
            "mae_ci_lower",
            "mae_ci_upper",
            "rmse",
            "me",
            "pup",
            "mae_asimetrico",
            "precision_plos",
            "recall_plos",
            "f1_plos",
            "proporcion_plos",
            "n_plos_real",
            "n_plos_pred",
        ]
        rows.extend(metrics_df[keep_cols].to_dict("records"))

        tramo_df = metrics_by_segment_and_tramo(predictions, threshold=threshold)
        tramo_df.insert(0, "umbral_plos_analizado", threshold)
        tramo_rows.append(tramo_df)
        print_dataframe(metrics_df[keep_cols])

    base_mae_by_segment = {
        row["segmento"]: float(row["mae"])
        for row in rows
        if int(row["umbral_plos"]) == BASE_PLOS_THRESHOLD
    }
    for row in rows:
        base_mae = base_mae_by_segment.get(row["segmento"])
        row["delta_mae_pct_vs_umbral_14"] = (
            delta_mae_pct(float(row["mae"]), base_mae) if base_mae else float("nan")
        )

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
