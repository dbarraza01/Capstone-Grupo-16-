"""Utilidades compartidas para el analisis de sensibilidad.

Los escenarios entrenan modelos temporales en memoria. No sobrescriben los
modelos productivos guardados en ``modelos_guardados``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_operacional_entrega3.utils.metricas_operacionales import (  # noqa: E402
    error_medio,
    mae,
    mae_asimetrico,
    medae,
    pup,
    rmse,
)
from ml_operacional_entrega3.utils.model_workflows import load_best_params  # noqa: E402
from ml_operacional_entrega3.utils.pipeline_operacional import (  # noqa: E402
    ID_COL,
    ML_OPS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SEGMENTS,
    TARGET_COL,
    URGENCY_COL,
    class_ratio,
    generate_oof_probabilities,
    load_segment_split,
    make_classifier,
    make_regressor,
    prepare_xy,
)


MODEL_NAME = "xgb"
MODEL_LABEL = "XGB"
BASE_PLOS_THRESHOLD = 14
RESULTS_DIR = ML_OPS_DIR / "sensitivity" / "results"
SEGMENT_ORDER = ["global", "urgente", "programado"]


@dataclass(frozen=True)
class SensitivityOutput:
    path: Path
    rows: int


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def binary_target(y: pd.Series | np.ndarray, threshold: int) -> np.ndarray:
    return (np.asarray(y, dtype=float) >= float(threshold)).astype(int)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def plos_confusion_metrics(
    y_true: pd.Series | np.ndarray,
    score_or_pred: pd.Series | np.ndarray,
    threshold: int | float,
    *,
    score_is_probability: bool = False,
    true_threshold: int | float = BASE_PLOS_THRESHOLD,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    score_arr = np.asarray(score_or_pred, dtype=float)
    if score_is_probability:
        plos_real = y_true_arr >= float(true_threshold)
        plos_pred = score_arr >= float(threshold)
    else:
        plos_real = y_true_arr >= float(threshold)
        plos_pred = score_arr >= float(threshold)

    tp = int(np.sum(plos_real & plos_pred))
    tn = int(np.sum(~plos_real & ~plos_pred))
    fp = int(np.sum(~plos_real & plos_pred))
    fn = int(np.sum(plos_real & ~plos_pred))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    accuracy = _safe_div(tp + tn, len(y_true_arr))
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    threshold: int = BASE_PLOS_THRESHOLD,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    abs_error = np.abs(y_pred_arr - y_true_arr)
    plos = plos_confusion_metrics(y_true_arr, y_pred_arr, threshold)
    return {
        "n_casos": int(len(y_true_arr)),
        "mae": mae(y_true_arr, y_pred_arr),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "medae": medae(y_true_arr, y_pred_arr),
        "me": error_medio(y_true_arr, y_pred_arr),
        "pup": pup(y_true_arr, y_pred_arr),
        "mae_asimetrico": mae_asimetrico(y_true_arr, y_pred_arr, alpha=2.0),
        "pct_error_abs_le_1d": float(np.mean(abs_error <= 1)),
        "pct_error_abs_le_3d": float(np.mean(abs_error <= 3)),
        "pct_error_abs_le_7d": float(np.mean(abs_error <= 7)),
        "los_real_promedio": float(np.mean(y_true_arr)),
        "los_pred_promedio": float(np.mean(y_pred_arr)),
        "umbral_plos": int(threshold),
        "proporcion_plos": float(np.mean(y_true_arr >= float(threshold))),
        "n_plos_real": int(np.sum(y_true_arr >= float(threshold))),
        "n_plos_pred": int(np.sum(y_pred_arr >= float(threshold))),
        "precision_plos": plos["precision"],
        "recall_plos": plos["recall"],
        "f1_plos": plos["f1"],
        "accuracy_plos": plos["accuracy"],
        "tp_plos": plos["tp"],
        "fp_plos": plos["fp"],
        "fn_plos": plos["fn"],
        "tn_plos": plos["tn"],
    }


def bootstrap_mae_ci(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Intervalo de confianza percentil para MAE usando bootstrap del holdout."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    n = len(y_true_arr)
    if n == 0:
        return {"mae_ci_lower": np.nan, "mae_ci_upper": np.nan}

    rng = np.random.default_rng(random_state)
    abs_error = np.abs(y_pred_arr - y_true_arr)
    boot_mae = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n, size=n)
        boot_mae[idx] = float(np.mean(abs_error[sample_idx]))

    alpha = (1.0 - float(confidence)) / 2.0
    return {
        "mae_ci_lower": float(np.quantile(boot_mae, alpha)),
        "mae_ci_upper": float(np.quantile(boot_mae, 1.0 - alpha)),
    }


def add_mae_ci(
    metrics: dict[str, float],
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    out = dict(metrics)
    out.update(bootstrap_mae_ci(y_true, y_pred, random_state=random_state))
    return out


def classifier_hospital_impact_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred_los: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    probability_threshold: float,
    *,
    true_threshold: int = BASE_PLOS_THRESHOLD,
) -> dict[str, float]:
    """Impacto en dias de FP/FN generados por la politica del clasificador."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred_los, dtype=float)
    prob_arr = np.asarray(probabilities, dtype=float)
    abs_error = np.abs(y_pred_arr - y_true_arr)

    plos_real = y_true_arr >= float(true_threshold)
    plos_alert = prob_arr >= float(probability_threshold)
    fn_mask = plos_real & ~plos_alert
    fp_mask = ~plos_real & plos_alert

    return {
        "promedio_dias_subestimados_fn": (
            float(np.mean(abs_error[fn_mask])) if np.any(fn_mask) else np.nan
        ),
        "promedio_dias_sobrestimados_fp": (
            float(np.mean(abs_error[fp_mask])) if np.any(fp_mask) else np.nan
        ),
    }


def empty_regression_metrics(threshold: int = BASE_PLOS_THRESHOLD) -> dict[str, float]:
    return {
        "n_casos": 0,
        "mae": np.nan,
        "mae_ci_lower": np.nan,
        "mae_ci_upper": np.nan,
        "rmse": np.nan,
        "medae": np.nan,
        "me": np.nan,
        "pup": np.nan,
        "mae_asimetrico": np.nan,
        "pct_error_abs_le_1d": np.nan,
        "pct_error_abs_le_3d": np.nan,
        "pct_error_abs_le_7d": np.nan,
        "los_real_promedio": np.nan,
        "los_pred_promedio": np.nan,
        "umbral_plos": int(threshold),
        "proporcion_plos": np.nan,
        "n_plos_real": 0,
        "n_plos_pred": 0,
        "precision_plos": np.nan,
        "recall_plos": np.nan,
        "f1_plos": np.nan,
        "accuracy_plos": np.nan,
        "tp_plos": 0,
        "fp_plos": 0,
        "fn_plos": 0,
        "tn_plos": 0,
    }


def tramo_definition(threshold: int) -> tuple[list[float], list[str]]:
    """Retorna tramos adaptativos para el umbral PLOS analizado."""
    definitions = {
        7: ([-1, 2, 6, np.inf], ["0-2", "3-6", "7+ (PLOS)"]),
        14: ([-1, 2, 6, 13, np.inf], ["0-2", "3-6", "7-13", "14+ (PLOS)"]),
        21: ([-1, 5, 12, 20, np.inf], ["0-5", "6-12", "13-20", "21+ (PLOS)"]),
        27: ([-1, 6, 15, 26, np.inf], ["0-6", "7-15", "16-26", "27+ (PLOS)"]),
    }
    if int(threshold) not in definitions:
        raise ValueError(f"No hay tramos adaptativos definidos para umbral PLOS={threshold}")
    return definitions[int(threshold)]


def metrics_by_tramo(
    predictions: pd.DataFrame,
    threshold: int = BASE_PLOS_THRESHOLD,
) -> pd.DataFrame:
    bins, labels = tramo_definition(threshold)
    if predictions.empty:
        rows = []
        for tramo in labels:
            row = {"tramo": tramo}
            row.update(empty_regression_metrics(threshold))
            rows.append(row)
        return pd.DataFrame(rows)

    y_true = predictions["los_dias_reales"].to_numpy(dtype=float)
    y_pred = predictions["los_dias_predichos"].to_numpy(dtype=float)
    tramos = pd.cut(y_true, bins=bins, labels=labels)
    rows: list[dict] = []
    for tramo in labels:
        mask = np.asarray(tramos == tramo)
        if not mask.any():
            row = {"tramo": tramo}
            row.update(empty_regression_metrics(threshold))
            rows.append(row)
            continue
        row = {"tramo": tramo}
        row.update(regression_metrics(y_true[mask], y_pred[mask], threshold=threshold))
        rows.append(row)
    return pd.DataFrame(rows)


def iter_segment_frames(predictions: pd.DataFrame):
    if "segmento" not in predictions.columns:
        raise ValueError("El dataframe de predicciones debe incluir columna 'segmento'")
    yield "global", predictions
    for segment in SEGMENTS:
        yield segment, predictions[predictions["segmento"].astype(str) == segment]


def summarize_predictions_by_segment(
    predictions: pd.DataFrame,
    *,
    threshold: int = BASE_PLOS_THRESHOLD,
    include_mae_ci: bool = False,
) -> pd.DataFrame:
    rows: list[dict] = []
    for segment_idx, (segment, group) in enumerate(iter_segment_frames(predictions)):
        row = {"segmento": segment}
        if group.empty:
            row.update(empty_regression_metrics(threshold))
        else:
            y_true = group["los_dias_reales"].to_numpy()
            y_pred = group["los_dias_predichos"].to_numpy()
            metrics = regression_metrics(y_true, y_pred, threshold=threshold)
            if include_mae_ci:
                metrics = add_mae_ci(
                    metrics,
                    y_true,
                    y_pred,
                    random_state=RANDOM_STATE + int(threshold) * 10 + segment_idx,
                )
            row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_segment_and_tramo(
    predictions: pd.DataFrame,
    threshold: int = BASE_PLOS_THRESHOLD,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for segment, group in iter_segment_frames(predictions):
        tramo_df = metrics_by_tramo(group, threshold=threshold)
        tramo_df.insert(0, "segmento", segment)
        frames.append(tramo_df)
    return pd.concat(frames, ignore_index=True)


def write_result(df: pd.DataFrame, filename: str) -> SensitivityOutput:
    ensure_results_dir()
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"  -> Guardado {path} ({len(df)} filas, {len(df.columns)} columnas)")
    return SensitivityOutput(path=path, rows=len(df))


def read_baseline_mae(segment: str = "global") -> float:
    if segment != "global":
        path = REPORTS_DIR / "metricas_por_segmento_holdout_xgb.csv"
        if not path.exists():
            raise FileNotFoundError(
                "No existe metricas_por_segmento_holdout_xgb.csv; ejecuta primero la evaluacion base."
            )
        df = pd.read_csv(path)
        row = df[
            (df["modelo"].astype(str).str.upper() == MODEL_LABEL)
            & (df["segmento"].astype(str) == segment)
        ]
        if row.empty:
            raise ValueError(f"No se encontro fila XGB para segmento={segment} en {path}")
        return float(row.iloc[0]["mae"])

    path = REPORTS_DIR / "comparacion_final_modelos.csv"
    if not path.exists():
        raise FileNotFoundError(
            "No existe comparacion_final_modelos.csv; ejecuta primero la evaluacion base."
        )
    df = pd.read_csv(path)
    row = df[df["modelo"].astype(str).str.upper() == MODEL_LABEL]
    if "split" in row.columns:
        holdout = row[row["split"].astype(str) == "holdout"]
        if not holdout.empty:
            row = holdout
    if row.empty:
        raise ValueError("No se encontro fila XGB en comparacion_final_modelos.csv")
    return float(row.iloc[0]["mae"])


def read_baseline_mae_by_segment() -> dict[str, float]:
    return {segment: read_baseline_mae(segment) for segment in SEGMENT_ORDER}


def delta_mae_pct(mae_value: float, baseline_mae: float) -> float:
    return float((mae_value - baseline_mae) / baseline_mae * 100.0)


def robustness_label(delta_pct: float) -> str:
    abs_delta = abs(float(delta_pct))
    if abs_delta < 5:
        return "robusto"
    if abs_delta < 15:
        return "sensibilidad moderada"
    return "sensibilidad alta"


def load_params_by_segment(kind: str) -> dict[str, dict]:
    params: dict[str, dict] = {}
    for segment in SEGMENTS:
        train_df = load_segment_split(segment, "train")
        _, y = prepare_xy(train_df)
        y_binary = binary_target(y, BASE_PLOS_THRESHOLD)
        params[segment] = load_best_params(MODEL_NAME, kind, segment, y_binary=y_binary)
    return params


def _adapt_scale_pos_weight(params: dict, y_binary: np.ndarray, enabled: bool) -> dict:
    out = dict(params)
    if enabled and "scale_pos_weight" in out:
        out["scale_pos_weight"] = class_ratio(y_binary)
    return out


def sample_for_smoke(
    df: pd.DataFrame,
    *,
    threshold: int,
    max_rows: int | None,
) -> pd.DataFrame:
    """Submuestra pequena preservando positivos y negativos para pruebas de humo."""
    if max_rows is None or len(df) <= max_rows:
        return df.copy()
    positives = df[df[TARGET_COL] >= threshold]
    negatives = df[df[TARGET_COL] < threshold]
    if positives.empty or negatives.empty:
        return df.sample(n=max_rows, random_state=RANDOM_STATE).copy()

    min_per_class = 10 if max_rows >= 40 else 2
    n_pos = min(len(positives), max(min_per_class, max_rows // 4))
    n_neg = min(len(negatives), max_rows - n_pos)
    if n_neg < min_per_class and len(negatives) >= min_per_class:
        n_neg = min_per_class
        n_pos = min(len(positives), max_rows - n_neg)

    sampled = pd.concat(
        [
            positives.sample(n=n_pos, random_state=RANDOM_STATE),
            negatives.sample(n=n_neg, random_state=RANDOM_STATE),
        ],
        ignore_index=False,
    )
    return sampled.sample(frac=1.0, random_state=RANDOM_STATE).copy()


def shrink_xgb_params(params: dict, *, n_estimators: int = 5) -> dict:
    """Reduce parametros para smoke tests sin cambiar la logica del pipeline."""
    out = dict(params)
    if "n_estimators" in out:
        out["n_estimators"] = min(int(out["n_estimators"]), int(n_estimators))
    else:
        out["n_estimators"] = int(n_estimators)
    if "max_depth" in out:
        out["max_depth"] = min(int(out["max_depth"]), 3)
    return out


def select_feature_columns(
    columns: list[str],
    variant: str = "full",
) -> list[str]:
    if variant == "full":
        return list(columns)
    if variant == "sin_charlson":
        return [col for col in columns if col != "charlson_index"]
    if variant == "sin_capitulos_icd10":
        return [
            col
            for col in columns
            if not (col.startswith("diag_rare_cap_") or col.startswith("proc_rare_sec_"))
        ]
    if variant == "sin_codigos_clinicos":
        keep = {
            "n_diag_total",
            "n_procedimientos",
            "es_urgencia",
            "mes_ingreso",
            "dia_semana_ingreso",
            "tiene_diag_primario",
            "charlson_index",
            "n_diag_primarios",
            "n_diag_secundarios",
        }
        selected = [col for col in columns if col in keep]
        if not selected:
            raise ValueError("La variante sin_codigos_clinicos no encontro columnas validas")
        return selected
    raise ValueError(f"Variante de features desconocida: {variant}")


def fit_predict_two_stage(
    *,
    threshold: int = BASE_PLOS_THRESHOLD,
    feature_variant: str = "full",
    prob_column: str = "prob_los_14",
    clf_params_by_segment: dict[str, dict] | None = None,
    reg_params_by_segment: dict[str, dict] | None = None,
    adapt_scale_pos_weight: bool = False,
    max_train_rows: int | None = None,
    max_holdout_rows: int | None = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    """Entrena un XGB temporal en dos etapas y devuelve predicciones holdout."""
    predictions: list[pd.DataFrame] = []

    for segment in SEGMENTS:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(f"{prefix}[{MODEL_LABEL}] segmento={segment} | umbral={threshold} | features={feature_variant}")
        train_df = load_segment_split(segment, "train")
        holdout_df = load_segment_split(segment, "holdout")
        train_df = sample_for_smoke(train_df, threshold=threshold, max_rows=max_train_rows)
        holdout_df = sample_for_smoke(holdout_df, threshold=threshold, max_rows=max_holdout_rows)

        X_train_all, y_train = prepare_xy(train_df)
        X_holdout_all, y_holdout = prepare_xy(holdout_df)
        feature_cols = select_feature_columns(list(X_train_all.columns), feature_variant)
        X_train = X_train_all[feature_cols].copy()
        X_holdout = X_holdout_all[feature_cols].copy()
        y_binary = binary_target(y_train, threshold)

        clf_params_base = (
            clf_params_by_segment[segment]
            if clf_params_by_segment is not None
            else load_best_params(MODEL_NAME, "clf", segment, y_binary=y_binary)
        )
        reg_params = (
            reg_params_by_segment[segment]
            if reg_params_by_segment is not None
            else load_best_params(MODEL_NAME, "reg", segment)
        )
        clf_params = _adapt_scale_pos_weight(clf_params_base, y_binary, adapt_scale_pos_weight)

        print(
            f"  Train={X_train.shape}; holdout={X_holdout.shape}; "
            f"positivos_train={int(y_binary.sum())}; features={len(feature_cols)}"
        )
        print("  Generando probabilidad OOF para el regresor")
        prob_oof = generate_oof_probabilities(MODEL_NAME, clf_params, X_train, y_binary)

        print("  Entrenando clasificador final temporal")
        clf = make_classifier(MODEL_NAME, clf_params)
        clf.fit(X_train, y_binary)

        print("  Entrenando regresor final temporal")
        X_reg_train = X_train.copy()
        X_reg_train[prob_column] = prob_oof
        reg = make_regressor(MODEL_NAME, reg_params)
        reg.fit(X_reg_train, y_train)

        prob_holdout = clf.predict_proba(X_holdout)[:, 1]
        X_reg_holdout = X_holdout.copy()
        X_reg_holdout[prob_column] = prob_holdout
        pred = np.clip(reg.predict(X_reg_holdout), 0, None)

        predictions.append(
            pd.DataFrame(
                {
                    "case_id": holdout_df[ID_COL].values,
                    "segmento": segment,
                    "es_urgencia": holdout_df[URGENCY_COL].values,
                    "umbral_plos": int(threshold),
                    "los_dias_reales": y_holdout.values,
                    "prob_riesgo": prob_holdout,
                    "los_dias_predichos": pred,
                    "error_dias": pred - y_holdout.values,
                    "feature_variant": feature_variant,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True)


def fit_predict_direct_regressor(
    *,
    feature_variant: str = "full",
    reg_params_by_segment: dict[str, dict] | None = None,
    max_train_rows: int | None = None,
    max_holdout_rows: int | None = None,
    log_prefix: str = "",
) -> pd.DataFrame:
    """Entrena un XGBRegressor temporal de una etapa, sin probabilidad PLOS."""
    predictions: list[pd.DataFrame] = []
    for segment in SEGMENTS:
        prefix = f"{log_prefix} " if log_prefix else ""
        print(f"{prefix}[{MODEL_LABEL}-1 etapa] segmento={segment} | features={feature_variant}")
        train_df = load_segment_split(segment, "train")
        holdout_df = load_segment_split(segment, "holdout")
        train_df = sample_for_smoke(train_df, threshold=BASE_PLOS_THRESHOLD, max_rows=max_train_rows)
        holdout_df = sample_for_smoke(holdout_df, threshold=BASE_PLOS_THRESHOLD, max_rows=max_holdout_rows)
        X_train_all, y_train = prepare_xy(train_df)
        X_holdout_all, y_holdout = prepare_xy(holdout_df)
        feature_cols = select_feature_columns(list(X_train_all.columns), feature_variant)
        X_train = X_train_all[feature_cols].copy()
        X_holdout = X_holdout_all[feature_cols].copy()

        reg_params = (
            reg_params_by_segment[segment]
            if reg_params_by_segment is not None
            else load_best_params(MODEL_NAME, "reg", segment)
        )
        print(f"  Train={X_train.shape}; holdout={X_holdout.shape}; features={len(feature_cols)}")
        reg = make_regressor(MODEL_NAME, reg_params)
        reg.fit(X_train, y_train)
        pred = np.clip(reg.predict(X_holdout), 0, None)
        predictions.append(
            pd.DataFrame(
                {
                    "case_id": holdout_df[ID_COL].values,
                    "segmento": segment,
                    "es_urgencia": holdout_df[URGENCY_COL].values,
                    "umbral_plos": BASE_PLOS_THRESHOLD,
                    "los_dias_reales": y_holdout.values,
                    "prob_riesgo": np.nan,
                    "los_dias_predichos": pred,
                    "error_dias": pred - y_holdout.values,
                    "feature_variant": feature_variant,
                }
            )
        )
    return pd.concat(predictions, ignore_index=True)


def summarize_predictions(
    predictions: pd.DataFrame,
    *,
    threshold: int = BASE_PLOS_THRESHOLD,
) -> dict[str, float]:
    return regression_metrics(
        predictions["los_dias_reales"].to_numpy(),
        predictions["los_dias_predichos"].to_numpy(),
        threshold=threshold,
    )


def pr_curve_dataframe(
    y_true_los: np.ndarray,
    probabilities: np.ndarray,
    threshold: int = BASE_PLOS_THRESHOLD,
) -> pd.DataFrame:
    y_binary = (np.asarray(y_true_los, dtype=float) >= float(threshold)).astype(int)
    precision, recall, thresholds = precision_recall_curve(y_binary, probabilities)
    rows = [{"precision": float(precision[0]), "recall": float(recall[0]), "thresholds": np.nan}]
    for idx, threshold in enumerate(thresholds):
        rows.append(
            {
                "precision": float(precision[idx + 1]),
                "recall": float(recall[idx + 1]),
                "thresholds": float(threshold),
            }
        )
    return pd.DataFrame(rows)


def load_holdout_predictions_with_prob() -> pd.DataFrame:
    path = REPORTS_DIR / "predicciones_holdout_xgb.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta primero ml_operacional_entrega3/XGB/evaluar_xgb.py"
        )
    required = {"los_dias_reales", "los_dias_predichos", "prob_riesgo", "segmento"}
    df = pd.read_csv(path)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")
    return df


def print_dataframe(df: pd.DataFrame, max_rows: int = 12) -> None:
    if len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"  ... {len(df) - max_rows} filas adicionales")
    else:
        print(df.to_string(index=False))


def expected_output_files() -> list[Path]:
    return [
        RESULTS_DIR / "escenario_1_resultados.csv",
        RESULTS_DIR / "escenario_1_resultados_por_tramo.csv",
        RESULTS_DIR / "escenario_2_resultados.csv",
        RESULTS_DIR / "escenario_3_curva_pr.csv",
        RESULTS_DIR / "escenario_3_puntos_operacion.csv",
        RESULTS_DIR / "escenario_4_resultados.csv",
        RESULTS_DIR / "reporte_sensibilidad_consolidado.md",
    ]


def existing_required_csvs() -> list[Path]:
    return [path for path in expected_output_files() if path.suffix == ".csv" and path.exists()]


def require_all_result_csvs() -> list[Path]:
    csv_paths = [path for path in expected_output_files() if path.suffix == ".csv"]
    missing = [path for path in csv_paths if not path.exists()]
    if missing:
        missing_text = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Faltan CSVs para generar el informe:\n{missing_text}")
    return csv_paths


REQUIRED_RESULT_COLUMNS = {
    "escenario_1_resultados.csv": {
        "segmento",
        "umbral_plos",
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
    },
    "escenario_1_resultados_por_tramo.csv": {
        "segmento",
        "umbral_plos_analizado",
        "tramo",
        "n_casos",
        "mae",
        "rmse",
        "medae",
        "me",
        "pup",
        "mae_asimetrico",
    },
    "escenario_2_resultados.csv": {
        "segmento",
        "variante",
        "mae",
        "mae_ci_lower",
        "mae_ci_upper",
        "rmse",
        "recall_plos",
        "f1_plos",
        "delta_mae_pct",
    },
    "escenario_3_puntos_operacion.csv": {
        "segmento",
        "politica_clinica",
        "umbral_probabilidad",
        "tp",
        "fp",
        "fn",
        "tn",
        "precision",
        "recall",
        "f1",
        "promedio_dias_subestimados_fn",
        "promedio_dias_sobrestimados_fp",
    },
    "escenario_3_curva_pr.csv": {
        "segmento",
        "precision",
        "recall",
        "thresholds",
    },
    "escenario_4_resultados.csv": {
        "segmento",
        "variante_hiperparametros",
        "mae",
        "mae_ci_lower",
        "mae_ci_upper",
        "rmse",
        "recall_plos",
        "f1_plos",
        "delta_mae_pct",
    },
}


def _missing_expected_segments(df: pd.DataFrame) -> list[str]:
    if "segmento" not in df.columns:
        return []
    present = set(df["segmento"].dropna().astype(str))
    return [segment for segment in SEGMENT_ORDER if segment not in present]


def validate_result_contracts(require_report: bool = False) -> pd.DataFrame:
    """Valida presencia, columnas y filas de los outputs del plan."""
    rows: list[dict] = []
    for filename, required_columns in REQUIRED_RESULT_COLUMNS.items():
        path = RESULTS_DIR / filename
        row = {
            "archivo": filename,
            "existe": path.exists(),
            "filas": 0,
            "columnas": 0,
            "estado": "missing",
            "faltantes": "",
        }
        if path.exists():
            df = pd.read_csv(path)
            missing = sorted(required_columns - set(df.columns))
            missing_segments = _missing_expected_segments(df) if "segmento" in required_columns else []
            missing_text = ", ".join(missing)
            if missing_segments:
                missing_text = (
                    f"{missing_text}; " if missing_text else ""
                ) + f"segmentos faltantes: {', '.join(missing_segments)}"
            row.update(
                {
                    "filas": len(df),
                    "columnas": len(df.columns),
                    "estado": "ok" if len(df) > 0 and not missing and not missing_segments else "invalid",
                    "faltantes": missing_text,
                }
            )
        rows.append(row)

    if require_report:
        report = RESULTS_DIR / "reporte_sensibilidad_consolidado.md"
        rows.append(
            {
                "archivo": report.name,
                "existe": report.exists(),
                "filas": 1 if report.exists() and report.stat().st_size > 0 else 0,
                "columnas": 1,
                "estado": "ok" if report.exists() and report.stat().st_size > 0 else "missing",
                "faltantes": "",
            }
        )

    validation = pd.DataFrame(rows)
    invalid = validation[validation["estado"] != "ok"]
    if not invalid.empty:
        print("\nValidacion de outputs con problemas:")
        print_dataframe(validation)
    else:
        print("\nValidacion de outputs OK:")
        print_dataframe(validation)
    return validation


def result_files_are_valid(filenames: list[str]) -> tuple[bool, list[str]]:
    """Retorna si un conjunto de outputs existe y cumple contrato minimo."""
    issues: list[str] = []
    for filename in filenames:
        path = RESULTS_DIR / filename
        if not path.exists():
            issues.append(f"{filename}: no existe")
            continue
        if filename not in REQUIRED_RESULT_COLUMNS:
            continue
        df = pd.read_csv(path)
        if df.empty:
            issues.append(f"{filename}: esta vacio")
            continue
        missing = sorted(REQUIRED_RESULT_COLUMNS[filename] - set(df.columns))
        if missing:
            issues.append(f"{filename}: faltan columnas {missing}")
            continue
        if "segmento" in REQUIRED_RESULT_COLUMNS[filename]:
            missing_segments = _missing_expected_segments(df)
            if missing_segments:
                issues.append(f"{filename}: faltan segmentos {missing_segments}")
    return len(issues) == 0, issues
