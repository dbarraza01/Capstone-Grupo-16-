"""Construccion del informe consolidado de sensibilidad.

Este modulo concentra la redaccion del reporte final. Los escenarios producen
CSVs; este archivo solo los lee e interpreta para regenerar el Markdown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.sensitivity.common import (  # noqa: E402
    BASE_PLOS_THRESHOLD,
    RESULTS_DIR,
    SensitivityOutput,
    delta_mae_pct,
    read_baseline_mae,
    require_all_result_csvs,
    robustness_label,
)
from ml_operacional_entrega3.utils.pipeline_operacional import dataframe_to_markdown  # noqa: E402


def format_markdown_table(df: pd.DataFrame) -> str:
    return dataframe_to_markdown(df)


def build_impact_summary() -> pd.DataFrame:
    """Construye tabla compacta de desviaciones si existen los outputs."""
    rows: list[dict] = []
    path1 = RESULTS_DIR / "escenario_1_resultados.csv"
    if path1.exists():
        df1 = pd.read_csv(path1)
        base = df1.loc[df1["umbral_plos"] == BASE_PLOS_THRESHOLD, "mae"]
        base_mae = float(base.iloc[0]) if not base.empty else read_baseline_mae()
        for _, row in df1.iterrows():
            delta = delta_mae_pct(float(row["mae"]), base_mae)
            rows.append(
                {
                    "escenario": "1 - Umbral PLOS",
                    "variante": f"LOS >= {int(row['umbral_plos'])}",
                    "mae": float(row["mae"]),
                    "delta_mae_pct": delta,
                    "veredicto": robustness_label(delta),
                }
            )

    path2 = RESULTS_DIR / "escenario_2_resultados.csv"
    if path2.exists():
        df2 = pd.read_csv(path2)
        for _, row in df2.iterrows():
            rows.append(
                {
                    "escenario": "2 - Ablation",
                    "variante": row["variante"],
                    "mae": float(row["mae"]),
                    "delta_mae_pct": float(row["delta_mae_pct"]),
                    "veredicto": robustness_label(float(row["delta_mae_pct"])),
                }
            )

    path4 = RESULTS_DIR / "escenario_4_resultados.csv"
    if path4.exists():
        df4 = pd.read_csv(path4)
        for _, row in df4.iterrows():
            rows.append(
                {
                    "escenario": "4 - Hiperparametros",
                    "variante": row["variante_hiperparametros"],
                    "mae": float(row["mae"]),
                    "delta_mae_pct": float(row["delta_mae_pct"]),
                    "veredicto": robustness_label(float(row["delta_mae_pct"])),
                }
            )
    return pd.DataFrame(rows)


def choose_policy_recommendation(policies: pd.DataFrame) -> str:
    if policies.empty:
        return "No hay politicas evaluadas."
    best_recall = policies.sort_values(["recall", "f1"], ascending=False).iloc[0]
    best_f1 = policies.sort_values(["f1", "recall"], ascending=False).iloc[0]
    if float(best_recall["recall"]) - float(best_f1["recall"]) >= 0.05:
        return (
            f"Recomendacion: usar {best_recall['politica_clinica']} "
            f"(umbral={best_recall['umbral_probabilidad']:.2f}) cuando el hospital priorice "
            "no perder pacientes PLOS. Entrega el mayor recall, a costa de mas falsos positivos."
        )
    return (
        f"Recomendacion: usar {best_f1['politica_clinica']} "
        f"(umbral={best_f1['umbral_probabilidad']:.2f}) como punto operativo inicial, "
        "porque maximiza F1 entre las politicas predefinidas."
    )


def _fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float, digits: int = 1) -> str:
    return f"{float(value):.{digits}f}%"


def _scenario1_interpretation(df: pd.DataFrame) -> str:
    base = df.loc[df["umbral_plos"] == BASE_PLOS_THRESHOLD]
    if base.empty:
        base_mae = float(df["mae"].min())
    else:
        base_mae = float(base.iloc[0]["mae"])
    temp = df.copy()
    temp["delta_mae_pct_calc"] = (temp["mae"] - base_mae) / base_mae * 100.0
    best = temp.sort_values("mae").iloc[0]
    worst = temp.reindex(temp["delta_mae_pct_calc"].abs().sort_values(ascending=False).index).iloc[0]
    robust_count = int((temp["delta_mae_pct_calc"].abs() < 5).sum())
    return (
        f"El menor MAE aparece con umbral LOS >= {int(best['umbral_plos'])} "
        f"(MAE={_fmt(best['mae'])}). La mayor desviacion frente al umbral 14 ocurre con "
        f"LOS >= {int(worst['umbral_plos'])}, delta={_fmt_pct(worst['delta_mae_pct_calc'])}. "
        f"{robust_count} de {len(temp)} umbrales quedan dentro del margen de robustez de 5%. "
        "En este escenario se recalcula `scale_pos_weight` de forma deterministica porque cambia "
        "la prevalencia de PLOS; no se realiza re-tuning."
    )


def _scenario2_interpretation(df: pd.DataFrame) -> str:
    temp = df.copy()
    temp["abs_delta"] = temp["delta_mae_pct"].abs()
    most_sensitive = temp.sort_values("abs_delta", ascending=False).iloc[0]
    direct = temp[temp["variante"].astype(str).str.contains("Sin Clasificador", case=False, regex=False)]
    classifier_text = ""
    if not direct.empty:
        row = direct.iloc[0]
        classifier_text = (
            f" La variante sin clasificador cambia el MAE en {_fmt_pct(row['delta_mae_pct'])}; "
            "ese valor cuantifica el aporte neto de usar la probabilidad PLOS como senal para el regresor."
        )
    return (
        f"La ablacion mas sensible es `{most_sensitive['variante']}` "
        f"(MAE={_fmt(most_sensitive['mae'])}, delta={_fmt_pct(most_sensitive['delta_mae_pct'])}). "
        "Si una variante degrada poco el MAE, el bloque removido es complementario o redundante; "
        "si lo degrada mucho, ese bloque sostiene parte importante de la prediccion operacional."
        f"{classifier_text}"
    )


def _scenario3_interpretation(df: pd.DataFrame) -> str:
    high_recall = df.sort_values(["recall", "f1"], ascending=False).iloc[0]
    high_precision = df.sort_values(["precision", "f1"], ascending=False).iloc[0]
    best_f1 = df.sort_values(["f1", "recall"], ascending=False).iloc[0]
    return (
        f"La politica con mayor recall es `{high_recall['politica_clinica']}` "
        f"(recall={_fmt(high_recall['recall'])}, FN={int(high_recall['fn'])}). "
        f"La politica con mayor precision es `{high_precision['politica_clinica']}` "
        f"(precision={_fmt(high_precision['precision'])}, FP={int(high_precision['fp'])}). "
        f"El mejor F1 entre las politicas predefinidas lo obtiene `{best_f1['politica_clinica']}` "
        f"(F1={_fmt(best_f1['f1'])})."
    )


def _scenario4_interpretation(df: pd.DataFrame) -> str:
    temp = df.copy()
    temp["abs_delta"] = temp["delta_mae_pct"].abs()
    most_sensitive = temp.sort_values("abs_delta", ascending=False).iloc[0]
    robust_count = int((temp["abs_delta"] < 5).sum())
    return (
        f"La mayor variacion parametrica observada es `{most_sensitive['variante_hiperparametros']}` "
        f"(MAE={_fmt(most_sensitive['mae'])}, delta={_fmt_pct(most_sensitive['delta_mae_pct'])}). "
        f"{robust_count} de {len(temp)} variantes quedan dentro del margen de robustez de 5%. "
        "Esto permite evaluar si el tuning esta en una region plana o si depende de un punto fragil."
    )


def _direct_answers(scenario2: pd.DataFrame, impact: pd.DataFrame) -> list[str]:
    answers = [
        "- Escenario 1: si las desviaciones frente al umbral 14 son menores a 5%, la conclusion no depende criticamente de definir PLOS como 14 dias.",
        "- Escenario 3: la politica de umbral debe elegirse segun el costo operacional: mayor recall reduce pacientes PLOS no detectados; mayor precision reduce fatiga por alertas.",
        "- Escenario 4: variantes vecinas bajo 5% de delta MAE indican que el tuning es estable; sobre 15% indica fragilidad parametrica.",
    ]
    if not scenario2.empty:
        for _, row in scenario2.iterrows():
            answers.append(
                f"- Escenario 2 / {row['variante']}: MAE={_fmt(row['mae'])}, "
                f"delta={_fmt_pct(row['delta_mae_pct'])}. {row.get('pregunta', '')}"
            )
    if not impact.empty:
        high = impact[impact["delta_mae_pct"].abs() >= 15]
        if high.empty:
            answers.append("- Veredicto cuantitativo: no aparecen variantes con sensibilidad alta segun el limite de 15% de delta MAE.")
        else:
            answers.append(
                "- Veredicto cuantitativo: hay variantes con sensibilidad alta; deben reportarse como limitacion critica."
            )
    return answers


def generate_consolidated_report() -> SensitivityOutput:
    require_all_result_csvs()
    scenario1 = pd.read_csv(RESULTS_DIR / "escenario_1_resultados.csv")
    scenario1_tramos = pd.read_csv(RESULTS_DIR / "escenario_1_resultados_por_tramo.csv")
    scenario2 = pd.read_csv(RESULTS_DIR / "escenario_2_resultados.csv")
    scenario3 = pd.read_csv(RESULTS_DIR / "escenario_3_puntos_operacion.csv")
    scenario4 = pd.read_csv(RESULTS_DIR / "escenario_4_resultados.csv")
    impact = build_impact_summary()

    policy_text = choose_policy_recommendation(scenario3)
    scenario1_text = _scenario1_interpretation(scenario1)
    scenario2_text = _scenario2_interpretation(scenario2)
    scenario3_text = _scenario3_interpretation(scenario3)
    scenario4_text = _scenario4_interpretation(scenario4)
    max_abs_delta = float(impact["delta_mae_pct"].abs().max()) if not impact.empty else np.nan
    global_verdict = (
        "El pipeline es robusto en los escenarios con variaciones menores al 5%."
        if np.isfinite(max_abs_delta) and max_abs_delta < 5
        else "El pipeline muestra sensibilidad en al menos un escenario; se debe discutir como limitacion."
    )

    lines = [
        "# Reporte Consolidado de Analisis de Sensibilidad",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        "Este informe consolida los cuatro escenarios definidos para evaluar la robustez del pipeline XGBoost operacional de dos etapas.",
        "",
        "### Impacto global sobre MAE",
        "",
        format_markdown_table(impact),
        "",
        f"**Veredicto general:** {global_verdict}",
        "",
        "## 2. Justificacion Bibliografica",
        "",
        "- La definicion de estancia prolongada no es universal; se reportan umbrales entre 7, 14, 24/27 dias y percentiles especificos de cohorte.",
        "- Los estudios de dos etapas para LOS justifican separar una decision binaria de riesgo PLOS y una estimacion continua de dias.",
        "- En datos clinicos desbalanceados, la curva Precision-Recall es mas informativa que mirar solo accuracy o MAE.",
        "- La estabilidad frente a hiperparametros vecinos valida que el resultado no depende de una configuracion demasiado fragil.",
        "",
        "Referencias base: Bergstra y Bengio (2012), Chrusciel et al. (2022), Goldstein et al. (2022), Lee et al. (2024), Mahajan et al. (2023), Probst et al. (2019), Saito y Rehmsmeier (2015).",
        "",
        "## 3. Analisis por Escenario",
        "",
        "### Escenario 1 - Variacion del umbral PLOS",
        "",
        "Pregunta: si cambia la definicion clinica de estancia prolongada, se mantiene estable el desempeno del modelo?",
        "",
        format_markdown_table(scenario1),
        "",
        "Desglose por tramo LOS:",
        "",
        format_markdown_table(scenario1_tramos),
        "",
        f"Interpretacion: {scenario1_text}",
        "",
        "### Escenario 2 - Ablation study",
        "",
        "Pregunta: que componentes son indispensables y que se pierde al removerlos?",
        "",
        format_markdown_table(scenario2),
        "",
        f"Interpretacion: {scenario2_text}",
        "",
        "### Escenario 3 - Punto de operacion del clasificador",
        "",
        "Pregunta: que politica de alerta conviene para gestion de camas?",
        "",
        format_markdown_table(scenario3),
        "",
        f"Interpretacion: {scenario3_text}",
        "",
        policy_text,
        "",
        "### Escenario 4 - Hiperparametros vecinos",
        "",
        "Pregunta: el tuning encontrado esta en una region estable o fragil?",
        "",
        format_markdown_table(scenario4),
        "",
        f"Interpretacion: {scenario4_text}",
        "",
        "## 4. Limitaciones y Mejoras Propuestas",
        "",
        "- Los escenarios no re-tunean hiperparametros; esto aisla el efecto de cada perturbacion, pero no mide el mejor rendimiento posible bajo cada nuevo supuesto.",
        "- El holdout permanece fijo para comparabilidad estricta; una validacion externa en otra clinica seria la prueba mas fuerte de generalizacion.",
        "- El punto de operacion del clasificador debe elegirse con gestores de camas, porque el costo de falsos positivos y falsos negativos no es puramente estadistico.",
        "",
        "## 5. Veredicto de Robustez y Toma de Decisiones Clinicas",
        "",
        f"Maxima desviacion absoluta de MAE observada en escenarios 1, 2 y 4: {max_abs_delta:.2f}%.",
        "",
        global_verdict,
        "",
        policy_text,
        "",
        "## 6. Respuestas Directas por Escenario",
        "",
        *_direct_answers(scenario2, impact),
        "",
    ]

    path = RESULTS_DIR / "reporte_sensibilidad_consolidado.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> Guardado {path}")
    return SensitivityOutput(path=path, rows=len(lines))
