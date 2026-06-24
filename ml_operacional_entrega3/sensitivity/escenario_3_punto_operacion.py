"""Escenario 3: sensibilidad al punto de operacion del clasificador."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.sensitivity.common import (  # noqa: E402
    SensitivityOutput,
    load_holdout_predictions_with_prob,
    plos_confusion_metrics,
    pr_curve_dataframe,
    print_dataframe,
    write_result,
)


POLICIES = [
    ("Politica B - Alta Seguridad / Alto Recall", 0.35),
    ("Politica A - Base / Equilibrio", 0.50),
    ("Politica C - Eficiencia / Alertas Confiables", 0.65),
]


def run() -> list[SensitivityOutput]:
    print("\n=== Escenario 3: Punto de operacion del clasificador ===")
    predictions = load_holdout_predictions_with_prob()
    y_true = predictions["los_dias_reales"].to_numpy(dtype=float)
    prob = predictions["prob_riesgo"].to_numpy(dtype=float)

    rows: list[dict] = []
    for policy_name, threshold in POLICIES:
        metrics = plos_confusion_metrics(
            y_true,
            prob,
            threshold,
            score_is_probability=True,
        )
        row = {
            "politica_clinica": policy_name,
            "umbral_probabilidad": threshold,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
        }
        rows.append(row)
        print_dataframe(pd.DataFrame([row]))

    policy_df = pd.DataFrame(rows)
    curve_df = pr_curve_dataframe(y_true, prob)
    outputs = [
        write_result(policy_df, "escenario_3_puntos_operacion.csv"),
        write_result(curve_df, "escenario_3_curva_pr.csv"),
    ]
    print("\nResumen Escenario 3:")
    print_dataframe(policy_df)
    print(f"  Curva PR generada con {len(curve_df)} puntos.")
    return outputs


if __name__ == "__main__":
    run()

