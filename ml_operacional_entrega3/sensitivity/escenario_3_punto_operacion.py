"""Escenario 3: sensibilidad al punto de operacion del clasificador."""

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
    classifier_hospital_impact_metrics,
    iter_segment_frames,
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

    rows: list[dict] = []
    curve_frames: list[pd.DataFrame] = []
    for segment, group in iter_segment_frames(predictions):
        y_true = group["los_dias_reales"].to_numpy(dtype=float)
        y_pred_los = group["los_dias_predichos"].to_numpy(dtype=float)
        prob = group["prob_riesgo"].to_numpy(dtype=float)
        for policy_name, threshold in POLICIES:
            metrics = plos_confusion_metrics(
                y_true,
                prob,
                threshold,
                score_is_probability=True,
                true_threshold=BASE_PLOS_THRESHOLD,
            )
            impact = classifier_hospital_impact_metrics(
                y_true,
                y_pred_los,
                prob,
                threshold,
                true_threshold=BASE_PLOS_THRESHOLD,
            )
            row = {
                "segmento": segment,
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
                "camas_bloqueadas_fp": metrics["fp"],
                "pacientes_plos_perdidos_fn": metrics["fn"],
                "fp_por_fn": metrics["fp"] / metrics["fn"] if metrics["fn"] else float("inf"),
                "promedio_dias_subestimados_fn": impact["promedio_dias_subestimados_fn"],
                "promedio_dias_sobrestimados_fp": impact["promedio_dias_sobrestimados_fp"],
            }
            rows.append(row)
        curve = pr_curve_dataframe(y_true, prob, threshold=BASE_PLOS_THRESHOLD)
        curve.insert(0, "segmento", segment)
        curve_frames.append(curve)
        print_dataframe(pd.DataFrame(rows[-len(POLICIES):]))

    policy_df = pd.DataFrame(rows)
    curve_df = pd.concat(curve_frames, ignore_index=True)
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
