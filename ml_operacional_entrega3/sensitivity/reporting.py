"""Construccion del informe consolidado de sensibilidad.

Los escenarios producen CSVs segmentados por cohorte. Este modulo solo lee esos
outputs, calcula resumenes comparables y regenera el Markdown final.
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
    SEGMENT_ORDER,
    SensitivityOutput,
    delta_mae_pct,
    require_all_result_csvs,
    robustness_label,
)
from ml_operacional_entrega3.utils.pipeline_operacional import dataframe_to_markdown  # noqa: E402


def format_markdown_table(df: pd.DataFrame) -> str:
    return dataframe_to_markdown(df)


def _fmt(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}%"


def _segment_title(segment: str) -> str:
    labels = {
        "global": "Global",
        "urgente": "Urgente",
        "programado": "Programado",
    }
    return labels.get(str(segment), str(segment))


def _ordered_segments(df: pd.DataFrame) -> list[str]:
    present = set(df["segmento"].astype(str)) if "segmento" in df.columns else set()
    ordered = [segment for segment in SEGMENT_ORDER if segment in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def format_by_segment(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    if "segmento" not in df.columns:
        return format_markdown_table(df[columns] if columns else df)

    parts: list[str] = []
    for segment in _ordered_segments(df):
        subset = df[df["segmento"].astype(str) == segment].copy()
        if columns is not None:
            subset = subset[[col for col in columns if col in subset.columns]]
        parts.extend(
            [
                f"#### Segmento: {_segment_title(segment)}",
                "",
                format_markdown_table(subset),
                "",
            ]
        )
    return "\n".join(parts).strip()


def build_impact_summary() -> pd.DataFrame:
    """Construye tabla compacta de desviaciones segmentadas."""
    rows: list[dict] = []
    path1 = RESULTS_DIR / "escenario_1_resultados.csv"
    if path1.exists():
        df1 = pd.read_csv(path1)
        base_by_segment = (
            df1[df1["umbral_plos"] == BASE_PLOS_THRESHOLD]
            .set_index("segmento")["mae"]
            .to_dict()
        )
        for _, row in df1.iterrows():
            segment = str(row["segmento"])
            base_mae = float(base_by_segment[segment])
            delta = delta_mae_pct(float(row["mae"]), base_mae)
            rows.append(
                {
                    "escenario": "1 - Umbral PLOS",
                    "segmento": segment,
                    "variante": f"LOS >= {int(row['umbral_plos'])}",
                    "mae": float(row["mae"]),
                    "mae_ci_lower": float(row["mae_ci_lower"]),
                    "mae_ci_upper": float(row["mae_ci_upper"]),
                    "delta_mae_pct": delta,
                    "veredicto": robustness_label(delta),
                }
            )

    path2 = RESULTS_DIR / "escenario_2_resultados.csv"
    if path2.exists():
        df2 = pd.read_csv(path2)
        for _, row in df2.iterrows():
            delta = float(row["delta_mae_pct"])
            rows.append(
                {
                    "escenario": "2 - Ablation",
                    "segmento": row["segmento"],
                    "variante": row["variante"],
                    "mae": float(row["mae"]),
                    "mae_ci_lower": float(row["mae_ci_lower"]),
                    "mae_ci_upper": float(row["mae_ci_upper"]),
                    "delta_mae_pct": delta,
                    "veredicto": robustness_label(delta),
                }
            )

    path4 = RESULTS_DIR / "escenario_4_resultados.csv"
    if path4.exists():
        df4 = pd.read_csv(path4)
        for _, row in df4.iterrows():
            delta = float(row["delta_mae_pct"])
            rows.append(
                {
                    "escenario": "4 - Hiperparametros",
                    "segmento": row["segmento"],
                    "variante": row["variante_hiperparametros"],
                    "mae": float(row["mae"]),
                    "mae_ci_lower": float(row["mae_ci_lower"]),
                    "mae_ci_upper": float(row["mae_ci_upper"]),
                    "delta_mae_pct": delta,
                    "veredicto": robustness_label(delta),
                }
            )
    return pd.DataFrame(rows)


def _ci_overlap(row: pd.Series, baseline: pd.Series) -> bool:
    return not (
        float(row["mae_ci_upper"]) < float(baseline["mae_ci_lower"])
        or float(row["mae_ci_lower"]) > float(baseline["mae_ci_upper"])
    )


def _append_ci_rows(
    rows: list[dict],
    df: pd.DataFrame,
    *,
    scenario_name: str,
    variant_col: str,
    baseline_selector,
) -> None:
    required = {"segmento", "mae", "mae_ci_lower", "mae_ci_upper", variant_col}
    if not required.issubset(df.columns):
        return
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment].copy()
        baseline_candidates = temp[baseline_selector(temp)]
        if baseline_candidates.empty:
            continue
        baseline = baseline_candidates.iloc[0]
        for _, row in temp.iterrows():
            rows.append(
                {
                    "escenario": scenario_name,
                    "segmento": segment,
                    "variante": row[variant_col],
                    "mae": float(row["mae"]),
                    "mae_ci_lower": float(row["mae_ci_lower"]),
                    "mae_ci_upper": float(row["mae_ci_upper"]),
                    "mae_base": float(baseline["mae"]),
                    "mae_base_ci_lower": float(baseline["mae_ci_lower"]),
                    "mae_base_ci_upper": float(baseline["mae_ci_upper"]),
                    "solapa_ic_base": _ci_overlap(row, baseline),
                    "delta_mae_pct": delta_mae_pct(float(row["mae"]), float(baseline["mae"])),
                }
            )


def build_ci_overlap_summary(
    scenario1: pd.DataFrame,
    scenario2: pd.DataFrame,
    scenario4: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict] = []
    _append_ci_rows(
        rows,
        scenario1.assign(variante=scenario1["umbral_plos"].map(lambda value: f"LOS >= {int(value)}")),
        scenario_name="1 - Umbral PLOS",
        variant_col="variante",
        baseline_selector=lambda df: df["umbral_plos"] == BASE_PLOS_THRESHOLD,
    )
    _append_ci_rows(
        rows,
        scenario2,
        scenario_name="2 - Ablation",
        variant_col="variante",
        baseline_selector=lambda df: df["variante"] == "Full (linea base)",
    )
    _append_ci_rows(
        rows,
        scenario4,
        scenario_name="4 - Hiperparametros",
        variant_col="variante_hiperparametros",
        baseline_selector=lambda df: df["variante_hiperparametros"] == "Full (linea base)",
    )
    return pd.DataFrame(rows)


def robustness_verdict_table(impact: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    if impact.empty:
        return pd.DataFrame(rows)

    for segment in ["global", "urgente"]:
        subset = impact[impact["segmento"].astype(str) == segment].copy()
        if subset.empty:
            continue
        subset["abs_delta"] = subset["delta_mae_pct"].abs()
        worst = subset.sort_values("abs_delta", ascending=False).iloc[0]
        rows.append(
            {
                "segmento": segment,
                "max_abs_delta_mae_pct": float(worst["abs_delta"]),
                "escenario_mas_sensible": worst["escenario"],
                "variante_mas_sensible": worst["variante"],
                "veredicto_5pct": "robusto" if float(worst["abs_delta"]) < 5 else "no robusto",
            }
        )
    return pd.DataFrame(rows)


def _scenario1_interpretation(df: pd.DataFrame) -> str:
    bullets: list[str] = []
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment].copy()
        base = temp[temp["umbral_plos"] == BASE_PLOS_THRESHOLD]
        if base.empty:
            continue
        base_mae = float(base.iloc[0]["mae"])
        temp["delta_calc"] = (temp["mae"] - base_mae) / base_mae * 100.0
        best = temp.sort_values("mae").iloc[0]
        worst = temp.reindex(temp["delta_calc"].abs().sort_values(ascending=False).index).iloc[0]
        robust_count = int((temp["delta_calc"].abs() < 5).sum())
        bullets.append(
            f"- {_segment_title(segment)}: mejor MAE con LOS >= {int(best['umbral_plos'])} "
            f"(MAE={_fmt(best['mae'])}); mayor desviacion vs umbral 14 con LOS >= "
            f"{int(worst['umbral_plos'])} (delta={_fmt_pct(worst['delta_calc'])}). "
            f"{robust_count}/{len(temp)} umbrales quedan dentro de +/-5%."
        )
    return "\n".join(bullets)


def _scenario2_interpretation(df: pd.DataFrame) -> str:
    bullets: list[str] = []
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment].copy()
        if temp.empty:
            continue
        temp["abs_delta"] = temp["delta_mae_pct"].abs()
        most_sensitive = temp.sort_values("abs_delta", ascending=False).iloc[0]
        no_classifier = temp[
            temp["variante"].astype(str).str.contains("Sin Clasificador", case=False, regex=False)
        ]
        text = (
            f"- {_segment_title(segment)}: la ablacion mas sensible es "
            f"`{most_sensitive['variante']}` (MAE={_fmt(most_sensitive['mae'])}, "
            f"delta={_fmt_pct(most_sensitive['delta_mae_pct'])})."
        )
        if not no_classifier.empty:
            row = no_classifier.iloc[0]
            text += (
                f" La variante sin clasificador obtiene MAE={_fmt(row['mae'])} "
                f"(delta={_fmt_pct(row['delta_mae_pct'])})."
            )
        bullets.append(text)
    return "\n".join(bullets)


def _compare_full_vs_one_stage(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    metrics = [
        "mae",
        "rmse",
        "mae_asimetrico",
        "precision_plos",
        "recall_plos",
        "f1_plos",
    ]
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment]
        full = temp[temp["variante"] == "Full (linea base)"]
        one_stage = temp[temp["variante"] == "Sin Clasificador (1 Etapa)"]
        if full.empty or one_stage.empty:
            continue
        full_row = full.iloc[0]
        one_row = one_stage.iloc[0]
        out = {"segmento": segment}
        for metric in metrics:
            if metric in full_row.index and metric in one_row.index:
                out[f"{metric}_2_etapas"] = float(full_row[metric])
                out[f"{metric}_1_etapa"] = float(one_row[metric])
                higher_is_better = metric in {"precision_plos", "recall_plos", "f1_plos"}
                if higher_is_better:
                    out[f"ganador_{metric}"] = "2 etapas" if full_row[metric] > one_row[metric] else "1 etapa"
                else:
                    out[f"ganador_{metric}"] = "2 etapas" if full_row[metric] < one_row[metric] else "1 etapa"
        rows.append(out)
    return pd.DataFrame(rows)


def _stacking_paradox_discussion(df: pd.DataFrame) -> str:
    comparison = _compare_full_vs_one_stage(df)
    lines = [
        "La variante `Sin Clasificador (1 Etapa)` obtiene en el holdout global un MAE ligeramente menor que la "
        "`Full (linea base)` de dos etapas. En los resultados actuales, el MAE global pasa de 2.860 dias en dos "
        "etapas a 2.844 dias en una etapa. Esta diferencia es pequena en magnitud, pero metodologicamente relevante "
        "porque evidencia una paradoja de stacking: agregar una prediccion intermedia no garantiza mejorar una metrica "
        "continua como MAE.",
        "",
        "En el pipeline completo, el clasificador estima una probabilidad de PLOS y esa probabilidad entra al regresor. "
        "Como la etapa 1 fue optimizada para una tarea binaria y no mediante joint-tuning con la etapa 2, cualquier "
        "ruido de calibracion, error de ranking o sesgo de probabilidad puede propagarse en cascada al estimador de "
        "dias. En otras palabras, la etapa 1 entrega una senal clinicamente interpretable, pero esa senal tambien "
        "puede introducir error propagation cuando el objetivo final evaluado es una prediccion continua.",
        "",
        "La defensa de la arquitectura de dos etapas no debe basarse solo en MAE. El enfoque de dos etapas provee una "
        "probabilidad explicita de riesgo PLOS, lo que permite ajustar politicas clinicas de alerta como en el "
        "Escenario 3. Un regresor puro de una etapa entrega una estimacion puntual de dias, pero no entrega de forma "
        "natural una perilla operacional de sensibilidad/especificidad para decidir cuantas alertas aceptar, cuantos "
        "falsos negativos tolerar o cuantas camas bloquear preventivamente.",
        "",
        "Al comparar metricas adicionales, la superioridad no es uniforme. La una etapa mejora el MAE global y urgente, "
        "pero la dos etapas mantiene mejor MAE y MAE asimetrico en programados. Por eso el resultado correcto no es "
        "`una etapa siempre gana`, sino que existe un trade-off: la una etapa puede ser mas limpia para error continuo, "
        "mientras la dos etapas entrega una salida probabilistica accionable para gestion hospitalaria.",
        "",
        "Resumen cuantitativo por segmento:",
        "",
    ]
    if not comparison.empty:
        lines.append(format_markdown_table(comparison))
    return "\n".join(lines)


def _ci_significance_discussion(ci_summary: pd.DataFrame) -> str:
    if ci_summary.empty:
        return "No se encontraron columnas de IC 95% para evaluar significancia estadistica."
    no_overlap = ci_summary[~ci_summary["solapa_ic_base"]]
    rows = [
        "El IC 95% del MAE se calculo por bootstrapping percentil con 1000 remuestreos del holdout. "
        "La interpretacion usada es directa: si el intervalo de una variante se solapa con el intervalo de la "
        "linea base del mismo escenario y segmento, la diferencia observada puede explicarse por variabilidad "
        "muestral; si no se solapa, se reporta como evidencia de cambio estadisticamente relevante en MAE.",
        "",
    ]
    if no_overlap.empty:
        rows.append(
            "Resultado: todos los IC 95% de las variantes evaluadas se solapan con su linea base correspondiente. "
            "Por lo tanto, las diferencias puntuales de MAE deben interpretarse como robustas pero no necesariamente "
            "estadisticamente distinguibles bajo este holdout."
        )
    else:
        rows.append(
            f"Resultado: {len(no_overlap)} comparaciones no solapan con la linea base. Esas diferencias deben "
            "reportarse como cambios potencialmente significativos en MAE, especialmente si coinciden con deltas "
            "superiores a la tolerancia operacional de 5%."
        )
        rows.extend(
            [
                "",
                "Comparaciones sin solapamiento:",
                "",
                format_markdown_table(no_overlap),
            ]
        )
    return "\n".join(rows)


def _scenario3_interpretation(df: pd.DataFrame) -> str:
    bullets: list[str] = []
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment].copy()
        if temp.empty:
            continue
        best_f1 = temp.sort_values(["f1", "recall"], ascending=False).iloc[0]
        best_recall = temp.sort_values(["recall", "f1"], ascending=False).iloc[0]
        best_precision = temp.sort_values(["precision", "f1"], ascending=False).iloc[0]
        bullets.append(
            f"- {_segment_title(segment)}: mejor F1 con `{best_f1['politica_clinica']}` "
            f"(umbral={best_f1['umbral_probabilidad']:.2f}, F1={_fmt(best_f1['f1'])}, "
            f"FP={int(best_f1['fp'])}, FN={int(best_f1['fn'])}, "
            f"dias_FN={_fmt(best_f1.get('promedio_dias_subestimados_fn', np.nan))}, "
            f"dias_FP={_fmt(best_f1.get('promedio_dias_sobrestimados_fp', np.nan))}). Mayor recall con "
            f"`{best_recall['politica_clinica']}` (recall={_fmt(best_recall['recall'])}, "
            f"FN={int(best_recall['fn'])}); mayor precision con `{best_precision['politica_clinica']}` "
            f"(precision={_fmt(best_precision['precision'])}, FP={int(best_precision['fp'])})."
        )
    return "\n".join(bullets)


def _policy_recommendation_by_segment(df: pd.DataFrame) -> str:
    urgent = df[df["segmento"].astype(str) == "urgente"].copy()
    scheduled = df[df["segmento"].astype(str) == "programado"].copy()
    lines = [
        "Recomendacion operacional segmentada:",
    ]
    if not urgent.empty:
        urgent_f1 = urgent.sort_values(["f1", "recall"], ascending=False).iloc[0]
        urgent_recall = urgent.sort_values(["recall", "f1"], ascending=False).iloc[0]
        lines.append(
            f"- Urgente: usar como punto inicial `{urgent_f1['politica_clinica']}` "
            f"(umbral={urgent_f1['umbral_probabilidad']:.2f}) porque equilibra FP={int(urgent_f1['fp'])} "
            f"y FN={int(urgent_f1['fn'])}; esos FN tienen un descalce promedio de "
            f"{_fmt(urgent_f1.get('promedio_dias_subestimados_fn', np.nan))} dias. "
            f"Si la clinica prioriza no perder PLOS urgentes, "
            f"`{urgent_recall['politica_clinica']}` reduce FN a {int(urgent_recall['fn'])}, "
            f"aceptando FP={int(urgent_recall['fp'])}."
        )
    if not scheduled.empty:
        scheduled_f1 = scheduled.sort_values(["f1", "precision"], ascending=False).iloc[0]
        lines.append(
            f"- Programado: usar `{scheduled_f1['politica_clinica']}` "
            f"(umbral={scheduled_f1['umbral_probabilidad']:.2f}) porque reduce camas bloqueadas "
            f"por falsos positivos a FP={int(scheduled_f1['fp'])}, con FN={int(scheduled_f1['fn'])}; "
            f"las falsas alertas tienen un descalce promedio de "
            f"{_fmt(scheduled_f1.get('promedio_dias_sobrestimados_fp', np.nan))} dias. "
            "En cirugia o admisiones planificadas, esta politica evita sobrerreservar camas cuando la agenda "
            "puede ajustarse con mas anticipacion."
        )
    return "\n".join(lines)


def _scenario4_interpretation(df: pd.DataFrame) -> str:
    bullets: list[str] = []
    for segment in _ordered_segments(df):
        temp = df[df["segmento"].astype(str) == segment].copy()
        if temp.empty:
            continue
        temp["abs_delta"] = temp["delta_mae_pct"].abs()
        most_sensitive = temp.sort_values("abs_delta", ascending=False).iloc[0]
        robust_count = int((temp["abs_delta"] < 5).sum())
        bullets.append(
            f"- {_segment_title(segment)}: mayor variacion con "
            f"`{most_sensitive['variante_hiperparametros']}` "
            f"(MAE={_fmt(most_sensitive['mae'])}, delta={_fmt_pct(most_sensitive['delta_mae_pct'])}). "
            f"{robust_count}/{len(temp)} variantes quedan dentro de +/-5%."
        )
    return "\n".join(bullets)


def _limitations_and_improvements(verdict: pd.DataFrame) -> list[str]:
    urgent_row = verdict[verdict["segmento"] == "urgente"]
    urgent_status = ""
    if not urgent_row.empty:
        urgent_status = str(urgent_row.iloc[0]["veredicto_5pct"])
    return [
        "- Los escenarios no re-tunean hiperparametros; aislan el efecto de cada perturbacion, pero no miden el mejor rendimiento posible bajo cada nuevo supuesto.",
        "- La sensibilidad debe leerse por cohorte. Un resultado global puede ocultar degradacion en urgentes, que es el segmento operacionalmente mas critico.",
        "- El Escenario 2 remueve bloques completos de informacion; una sensibilidad alta al modo sin codigos clinicos no es una falla del modelo, sino evidencia de que las variables clinicas detalladas sostienen rendimiento.",
        f"- En urgencias, el veredicto 5% queda como `{urgent_status or 'no disponible'}` porque la ablacion sin codigos clinicos supera la tolerancia; la mejora principal es no simplificar el bloque clinico, y en segundo lugar calibrar umbrales del clasificador por segmento.",
        "- El punto de operacion del clasificador debe elegirse con gestion de camas: falsos positivos bloquean capacidad; falsos negativos dejan pacientes PLOS sin alerta temprana.",
        "- Una validacion externa en otra clinica o periodo temporal sigue siendo necesaria para probar generalizacion fuera del holdout actual.",
    ]


def generate_consolidated_report() -> SensitivityOutput:
    require_all_result_csvs()
    scenario1 = pd.read_csv(RESULTS_DIR / "escenario_1_resultados.csv")
    scenario1_tramos = pd.read_csv(RESULTS_DIR / "escenario_1_resultados_por_tramo.csv")
    scenario2 = pd.read_csv(RESULTS_DIR / "escenario_2_resultados.csv")
    scenario3 = pd.read_csv(RESULTS_DIR / "escenario_3_puntos_operacion.csv")
    scenario4 = pd.read_csv(RESULTS_DIR / "escenario_4_resultados.csv")

    impact = build_impact_summary()
    ci_summary = build_ci_overlap_summary(scenario1, scenario2, scenario4)
    verdict = robustness_verdict_table(impact)
    scenario1_text = _scenario1_interpretation(scenario1)
    scenario2_text = _scenario2_interpretation(scenario2)
    scenario3_text = _scenario3_interpretation(scenario3)
    scenario4_text = _scenario4_interpretation(scenario4)
    policy_text = _policy_recommendation_by_segment(scenario3)
    limitations = _limitations_and_improvements(verdict)

    lines = [
        "# Reporte Consolidado de Analisis de Sensibilidad",
        "",
        "## 1. Resumen Ejecutivo",
        "",
        "Este informe consolida cuatro escenarios de sensibilidad del pipeline XGBoost operacional de dos etapas. "
        "Todas las metricas se reportan para tres cohortes: `global`, `urgente` y `programado`, porque el origen "
        "de admision cambia la dificultad clinica y el costo operacional de los errores.",
        "",
        "### Impacto sobre MAE por cohorte",
        "",
        format_by_segment(impact),
        "",
        "### Veredicto cuantitativo 5%",
        "",
        "La tolerancia de robustez se aplica de forma independiente para `global` y `urgente`. "
        "El segmento urgente se reporta aparte porque concentra mayor riesgo operacional en gestion de camas.",
        "",
        format_markdown_table(verdict),
        "",
        "### Analisis de significancia estadistica con bootstrapping",
        "",
        _ci_significance_discussion(ci_summary),
        "",
        "Tabla de solapamiento de IC 95% contra la linea base:",
        "",
        format_by_segment(ci_summary),
        "",
        "## 2. Justificacion Metodologica",
        "",
        "- La definicion de estancia prolongada no es universal; por eso el Escenario 1 prueba umbrales 7, 14, 21 y 27.",
        "- Los tramos del Escenario 1 son adaptativos: el ultimo tramo siempre corresponde al PLOS definido por el umbral evaluado.",
        "- Los modelos de dos etapas separan una decision binaria de riesgo PLOS de una estimacion continua de dias.",
        "- En datos clinicos desbalanceados, precision, recall y F1 son mas informativos que mirar solo accuracy o MAE.",
        "- La estabilidad frente a hiperparametros vecinos valida si el tuning cae en una region estable o demasiado fragil.",
        "",
        "Referencias base: Bergstra y Bengio (2012), Chrusciel et al. (2022), Goldstein et al. (2022), Lee et al. (2024), Mahajan et al. (2023), Probst et al. (2019), Saito y Rehmsmeier (2015).",
        "",
        "## 3. Analisis por Escenario",
        "",
        "### Escenario 1 - Variacion del umbral PLOS",
        "",
        "Pregunta: si cambia la definicion clinica de estancia prolongada, se mantiene estable el desempeno del modelo?",
        "",
        format_by_segment(scenario1),
        "",
        "Desglose por tramos adaptativos de LOS:",
        "",
        format_by_segment(scenario1_tramos),
        "",
        "Interpretacion:",
        "",
        scenario1_text,
        "",
        "### Escenario 2 - Ablation study",
        "",
        "Pregunta: que componentes son indispensables y que se pierde al removerlos?",
        "",
        format_by_segment(scenario2),
        "",
        "Interpretacion:",
        "",
        scenario2_text,
        "",
        "#### Discusion de la Paradoja de Stacking (2 Etapas vs. 1 Etapa)",
        "",
        _stacking_paradox_discussion(scenario2),
        "",
        "#### Significancia estadistica de las diferencias de MAE",
        "",
        _ci_significance_discussion(
            ci_summary[ci_summary["escenario"].astype(str) == "2 - Ablation"]
            if not ci_summary.empty
            else ci_summary
        ),
        "",
        "### Escenario 3 - Punto de operacion del clasificador",
        "",
        "Pregunta: que politica de alerta conviene para gestion de camas en cada origen de admision?",
        "",
        format_by_segment(scenario3),
        "",
        "Interpretacion del balance FP/FN:",
        "",
        scenario3_text,
        "",
        policy_text,
        "",
        "### Escenario 4 - Hiperparametros vecinos",
        "",
        "Pregunta: el tuning encontrado esta en una region estable o fragil?",
        "",
        format_by_segment(scenario4),
        "",
        "Interpretacion:",
        "",
        scenario4_text,
        "",
        "## 4. Limitaciones y Mejoras Propuestas",
        "",
        *limitations,
        "",
        "## 5. Veredicto de Robustez y Uso Clinico",
        "",
        "El modelo debe defenderse como una solucion operacional con dos salidas complementarias: "
        "dias esperados de LOS para planificacion cuantitativa y probabilidad PLOS para alerta temprana. "
        "La evaluacion segmentada evita concluir desde el promedio global cuando urgentes y programados tienen "
        "riesgos, prevalencias y costos de error distintos.",
        "",
        format_markdown_table(verdict),
        "",
        policy_text,
        "",
    ]

    path = RESULTS_DIR / "reporte_sensibilidad_consolidado.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> Guardado {path}")
    return SensitivityOutput(path=path, rows=len(lines))
