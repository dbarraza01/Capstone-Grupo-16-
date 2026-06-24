"""Escenario 4: robustez a hiperparametros vecinos del optimo."""

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
    load_params_by_segment,
    print_dataframe,
    read_baseline_mae,
    regression_metrics,
    write_result,
)


def _clip_lower(value: float, lower: float) -> float:
    return max(float(value), lower)


def perturb_params(params: dict, variant: str) -> dict:
    out = dict(params)
    if variant == "Conservadora (mas regularizada)":
        if "max_depth" in out:
            out["max_depth"] = max(int(out["max_depth"]) - 1, 2)
        if "learning_rate" in out:
            out["learning_rate"] = float(out["learning_rate"]) * 0.8
        if "min_child_weight" in out:
            out["min_child_weight"] = float(out["min_child_weight"]) + 2
    elif variant == "Compleja (menos regularizada)":
        if "max_depth" in out:
            out["max_depth"] = int(out["max_depth"]) + 1
        if "learning_rate" in out:
            out["learning_rate"] = float(out["learning_rate"]) * 1.2
        if "reg_alpha" in out:
            out["reg_alpha"] = float(out["reg_alpha"]) * 0.5
        if "reg_lambda" in out:
            out["reg_lambda"] = float(out["reg_lambda"]) * 0.5
    elif variant == "Perturbacion estocastica de muestreo":
        if "subsample" in out:
            out["subsample"] = _clip_lower(float(out["subsample"]) - 0.1, 0.5)
        if "colsample_bytree" in out:
            out["colsample_bytree"] = _clip_lower(float(out["colsample_bytree"]) - 0.1, 0.5)
    else:
        raise ValueError(f"Variante de hiperparametros desconocida: {variant}")
    return out


VARIANTS = [
    "Conservadora (mas regularizada)",
    "Compleja (menos regularizada)",
    "Perturbacion estocastica de muestreo",
]


def run() -> list[SensitivityOutput]:
    print("\n=== Escenario 4: Robustez a hiperparametros vecinos ===")
    base_clf_params = load_params_by_segment("clf")
    base_reg_params = load_params_by_segment("reg")
    baseline_mae = read_baseline_mae()
    rows: list[dict] = []

    for variant in VARIANTS:
        print(f"\n--- Variante: {variant} ---")
        clf_params = {segment: perturb_params(params, variant) for segment, params in base_clf_params.items()}
        reg_params = {segment: perturb_params(params, variant) for segment, params in base_reg_params.items()}
        predictions = fit_predict_two_stage(
            threshold=BASE_PLOS_THRESHOLD,
            feature_variant="full",
            prob_column="prob_los_14",
            clf_params_by_segment=clf_params,
            reg_params_by_segment=reg_params,
            adapt_scale_pos_weight=False,
            log_prefix="[E4]",
        )
        metrics = regression_metrics(
            predictions["los_dias_reales"],
            predictions["los_dias_predichos"],
            threshold=BASE_PLOS_THRESHOLD,
        )
        row = {
            "variante_hiperparametros": variant,
            "n_casos": metrics["n_casos"],
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "recall_plos": metrics["recall_plos"],
            "f1_plos": metrics["f1_plos"],
            "precision_plos": metrics["precision_plos"],
            "mae_asimetrico": metrics["mae_asimetrico"],
            "delta_mae_pct": delta_mae_pct(float(metrics["mae"]), baseline_mae),
        }
        rows.append(row)
        print_dataframe(pd.DataFrame([row]))

    resultados = pd.DataFrame(rows)
    output = write_result(resultados, "escenario_4_resultados.csv")
    print("\nResumen Escenario 4:")
    print_dataframe(resultados)
    return [output]


if __name__ == "__main__":
    run()
