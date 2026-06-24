"""Escenario 2: ablation study de features y arquitectura."""

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
    fit_predict_direct_regressor,
    fit_predict_two_stage,
    print_dataframe,
    summarize_predictions_by_segment,
    write_result,
)


VARIANTS = [
    {
        "variante": "Full (linea base)",
        "feature_variant": "full",
        "uses_classifier": True,
        "pregunta": "Referencia completa del pipeline en dos etapas.",
    },
    {
        "variante": "Sin Charlson",
        "feature_variant": "sin_charlson",
        "uses_classifier": True,
        "pregunta": "Mide la dependencia del indice de comorbilidad Charlson.",
    },
    {
        "variante": "Sin capitulos ICD-10",
        "feature_variant": "sin_capitulos_icd10",
        "uses_classifier": True,
        "pregunta": "Mide el aporte marginal de agrupaciones raras por capitulo.",
    },
    {
        "variante": "Sin codigos clinicos",
        "feature_variant": "sin_codigos_clinicos",
        "uses_classifier": True,
        "pregunta": "Mide cuanto se pierde al remover codigos clinicos detallados.",
    },
    {
        "variante": "Sin Clasificador (1 Etapa)",
        "feature_variant": "full",
        "uses_classifier": False,
        "pregunta": "Mide si la probabilidad PLOS de la etapa 1 aporta valor neto.",
    },
]


def run() -> list[SensitivityOutput]:
    print("\n=== Escenario 2: Ablation study de features y componentes ===")
    rows: list[dict] = []

    for variant in VARIANTS:
        print(f"\n--- Variante: {variant['variante']} ---")
        if variant["uses_classifier"]:
            predictions = fit_predict_two_stage(
                threshold=BASE_PLOS_THRESHOLD,
                feature_variant=variant["feature_variant"],
                prob_column="prob_los_14",
                adapt_scale_pos_weight=False,
                log_prefix="[E2]",
            )
        else:
            predictions = fit_predict_direct_regressor(
                feature_variant=variant["feature_variant"],
                log_prefix="[E2]",
            )

        metrics_df = summarize_predictions_by_segment(
            predictions,
            threshold=BASE_PLOS_THRESHOLD,
            include_mae_ci=True,
        )
        for _, metrics in metrics_df.iterrows():
            rows.append(
                {
                    "segmento": metrics["segmento"],
                    "variante": variant["variante"],
                    "pregunta": variant["pregunta"],
                    "n_casos": metrics["n_casos"],
                    "mae": metrics["mae"],
                    "mae_ci_lower": metrics["mae_ci_lower"],
                    "mae_ci_upper": metrics["mae_ci_upper"],
                    "rmse": metrics["rmse"],
                    "recall_plos": metrics["recall_plos"],
                    "f1_plos": metrics["f1_plos"],
                    "precision_plos": metrics["precision_plos"],
                    "mae_asimetrico": metrics["mae_asimetrico"],
                }
            )
        print_dataframe(pd.DataFrame(rows[-len(metrics_df):]))

    resultados = pd.DataFrame(rows)
    baseline = resultados[resultados["variante"] == "Full (linea base)"].set_index("segmento")["mae"].to_dict()
    resultados["delta_mae_pct"] = resultados.apply(
        lambda row: delta_mae_pct(float(row["mae"]), float(baseline[row["segmento"]])),
        axis=1,
    )
    output = write_result(resultados, "escenario_2_resultados.csv")
    print("\nResumen Escenario 2:")
    print_dataframe(resultados)
    return [output]


if __name__ == "__main__":
    run()
