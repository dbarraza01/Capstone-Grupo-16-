"""Funciones compartidas para el pipeline operacional de LOS."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier, XGBRegressor

from ml_operacional_entrega3.utils.metricas_operacionales import (
    TRAMOS_BINS,
    TRAMOS_LABELS,
    UMBRAL_PLOS,
    calcular_metricas_globales,
    calcular_metricas_por_tramo,
    formatear_metricas_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "ml_entrega2" / "feature_engineering" / "processed_v3" / "model_data_v3_escenario_B_charlson.csv"
ML_OPS_DIR = PROJECT_ROOT / "ml_operacional_entrega3"
DATA_SPLITS_DIR = ML_OPS_DIR / "data_splits"
MODELS_DIR = ML_OPS_DIR / "modelos_guardados"
REPORTS_DIR = ML_OPS_DIR / "reports"

TARGET_COL = "los_dias"
ID_COL = "case_id"
URGENCY_COL = "es_urgencia"
RANDOM_STATE = 42
HOLDOUT_SIZE = 0.20
LOS_RISK_THRESHOLD = 14

SEGMENTS = {
    "urgente": 1,
    "programado": 0,
}


def ensure_dirs() -> None:
    for path in [DATA_SPLITS_DIR, MODELS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, sep=";")
    requeridas = {ID_COL, TARGET_COL, URGENCY_COL}
    faltantes = sorted(requeridas - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en dataset base: {faltantes}")
    return df


def _tramos_para_estratificar(y: pd.Series) -> pd.Series:
    return pd.cut(y, bins=TRAMOS_BINS, labels=TRAMOS_LABELS)


def split_and_export(force: bool = False) -> dict[str, pd.DataFrame]:
    """Crea split 80/20 estratificado por urgencia y tramo LOS, y exporta CSVs."""
    ensure_dirs()
    expected = [
        DATA_SPLITS_DIR / "datos_train_urgente.csv",
        DATA_SPLITS_DIR / "datos_train_programado.csv",
        DATA_SPLITS_DIR / "datos_holdout_urgente.csv",
        DATA_SPLITS_DIR / "datos_holdout_programado.csv",
    ]
    if not force and all(path.exists() for path in expected):
        return {
            "train_urgente": pd.read_csv(DATA_SPLITS_DIR / "datos_train_urgente.csv"),
            "train_programado": pd.read_csv(DATA_SPLITS_DIR / "datos_train_programado.csv"),
            "holdout_urgente": pd.read_csv(DATA_SPLITS_DIR / "datos_holdout_urgente.csv"),
            "holdout_programado": pd.read_csv(DATA_SPLITS_DIR / "datos_holdout_programado.csv"),
        }

    df = load_dataset()
    tramos = _tramos_para_estratificar(df[TARGET_COL]).astype(str)
    strata = df[URGENCY_COL].astype(str) + "_" + tramos
    train_df, holdout_df = train_test_split(
        df,
        test_size=HOLDOUT_SIZE,
        random_state=RANDOM_STATE,
        stratify=strata,
    )

    outputs: dict[str, pd.DataFrame] = {}
    for segment, value in SEGMENTS.items():
        train_segment = train_df[train_df[URGENCY_COL] == value].copy()
        holdout_segment = holdout_df[holdout_df[URGENCY_COL] == value].copy()
        train_segment.to_csv(DATA_SPLITS_DIR / f"datos_train_{segment}.csv", index=False)
        holdout_segment.to_csv(DATA_SPLITS_DIR / f"datos_holdout_{segment}.csv", index=False)
        outputs[f"train_{segment}"] = train_segment
        outputs[f"holdout_{segment}"] = holdout_segment
    return outputs


def load_segment_split(segment: str, split: str) -> pd.DataFrame:
    if segment not in SEGMENTS:
        raise ValueError(f"Segmento desconocido: {segment}")
    if split not in {"train", "holdout"}:
        raise ValueError("split debe ser 'train' o 'holdout'")
    split_and_export(force=False)
    return pd.read_csv(DATA_SPLITS_DIR / f"datos_{split}_{segment}.csv")


def prepare_xy(df: pd.DataFrame, include_prob: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [ID_COL, TARGET_COL]
    if not include_prob and "prob_los_14" in df.columns:
        drop_cols.append("prob_los_14")
    X = df.drop(columns=[col for col in drop_cols if col in df.columns]).copy()
    y = df[TARGET_COL].copy()

    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    non_numeric = X.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        raise ValueError(f"Columnas no numericas en features: {list(non_numeric)}")
    return X, y


def binary_target_los14(y: pd.Series | np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=float) >= LOS_RISK_THRESHOLD).astype(int)


def class_ratio(y_binary: np.ndarray) -> float:
    pos = float(np.sum(y_binary == 1))
    neg = float(np.sum(y_binary == 0))
    return neg / pos if pos > 0 else 1.0


def default_xgb_clf_params(y_binary: np.ndarray | None = None) -> dict:
    return {
        "n_estimators": 30,
        "max_depth": 3,
        "learning_rate": 0.06,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "gamma": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
        "scale_pos_weight": class_ratio(y_binary) if y_binary is not None else 1.0,
    }


def default_xgb_reg_params() -> dict:
    return {
        "n_estimators": 40,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.6,
        "min_child_weight": 5,
        "gamma": 1.0,
        "reg_alpha": 1.0,
        "reg_lambda": 3.0,
    }


def default_rf_clf_params() -> dict:
    return {
        "n_estimators": 30,
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "max_samples": 0.75,
        "bootstrap": True,
    }


def default_rf_reg_params() -> dict:
    return {
        "n_estimators": 30,
        "max_depth": 12,
        "min_samples_split": 20,
        "min_samples_leaf": 8,
        "max_features": "sqrt",
        "max_samples": 0.75,
        "bootstrap": True,
    }


def make_classifier(model_name: str, params: dict) -> object:
    if model_name == "xgb":
        return XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced_subsample",
            **params,
        )
    raise ValueError(f"Modelo clasificador no soportado: {model_name}")


def make_regressor(model_name: str, params: dict) -> object:
    if model_name == "xgb":
        base = XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            **params,
        )
    elif model_name == "rf":
        base = RandomForestRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )
    else:
        raise ValueError(f"Modelo regresor no soportado: {model_name}")
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def make_lr_regressor(alpha: float = 1.0) -> object:
    base = Ridge(alpha=alpha, random_state=RANDOM_STATE)
    return TransformedTargetRegressor(regressor=base, func=np.log1p, inverse_func=np.expm1)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(path: Path, fallback: dict | None = None) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    if fallback is not None:
        save_json(path, fallback)
        return fallback
    raise FileNotFoundError(path)


def clean_regressor_params(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        if key.startswith("regressor__"):
            cleaned[key.replace("regressor__", "", 1)] = value
        else:
            cleaned[key] = value
    return cleaned


def generate_oof_probabilities(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y_binary: np.ndarray,
    n_splits: int = 5,
) -> np.ndarray:
    min_class = int(np.min(np.bincount(y_binary))) if len(np.unique(y_binary)) == 2 else 0
    if min_class < 2:
        raise ValueError("No hay suficientes casos por clase para generar OOF")
    effective_splits = min(n_splits, min_class)
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=RANDOM_STATE)
    probs = np.zeros(len(X), dtype=float)
    X_np = X.values
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_binary), 1):
        model = make_classifier(model_name, params)
        model.fit(X_np[train_idx], y_binary[train_idx])
        probs[val_idx] = model.predict_proba(X_np[val_idx])[:, 1]
        print(f"      OOF fold {fold}/{effective_splits} completado")
    return probs


def export_oof_dataset(
    segment: str,
    train_df: pd.DataFrame,
    probabilities: np.ndarray,
    model_name: str,
) -> pd.DataFrame:
    out = train_df.copy()
    out["prob_los_14"] = probabilities
    # Nombre requerido por el objetivo. El ultimo modelo entrenado puede refrescarlo.
    out.to_csv(DATA_SPLITS_DIR / f"train_con_prob_{segment}.csv", index=False)
    # Nombre model-specific para auditoria sin perdida.
    out.to_csv(DATA_SPLITS_DIR / f"train_con_prob_{model_name}_{segment}.csv", index=False)
    return out


def save_model_bundle(path: Path, model: object, features: list[str], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": features, "metadata": metadata}, path)


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Renderiza una tabla Markdown sin depender de tabulate."""
    if df.empty:
        return "_Sin datos._"
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---" for _ in columns]) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if np.isnan(value) else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def evaluar_predicciones(
    model_name: str,
    predicciones: pd.DataFrame,
    report_prefix: str,
    split_name: str = "holdout",
) -> dict[str, pd.DataFrame]:
    predicciones = predicciones.copy()
    y_true = predicciones["los_dias_reales"].to_numpy()
    y_pred = predicciones["los_dias_predichos"].to_numpy()
    predicciones["plos_real_14"] = (y_true >= UMBRAL_PLOS).astype(int)
    predicciones["plos_pred_14"] = (y_pred >= UMBRAL_PLOS).astype(int)
    global_metrics = calcular_metricas_globales(y_true, y_pred)
    df_global = pd.DataFrame([{ "modelo": model_name, "split": split_name, **global_metrics }])
    df_tramos = calcular_metricas_por_tramo(y_true, y_pred)
    df_tramos.insert(0, "modelo", model_name)
    df_tramos.insert(1, "split", split_name)

    segment_rows = []
    segment_tramo_rows = []
    for (segmento, es_urgencia), group in predicciones.groupby(["segmento", "es_urgencia"], sort=True):
        metrics = calcular_metricas_globales(group["los_dias_reales"], group["los_dias_predichos"])
        segment_rows.append({
            "modelo": model_name,
            "split": split_name,
            "segmento": segmento,
            "es_urgencia": int(es_urgencia),
            **metrics,
        })
        tramo_df = calcular_metricas_por_tramo(group["los_dias_reales"], group["los_dias_predichos"])
        tramo_df.insert(0, "modelo", model_name)
        tramo_df.insert(1, "split", split_name)
        tramo_df.insert(2, "segmento", segmento)
        tramo_df.insert(3, "es_urgencia", int(es_urgencia))
        segment_tramo_rows.append(tramo_df)

    df_segmentos = pd.DataFrame(segment_rows)
    df_segmento_tramos = pd.concat(segment_tramo_rows, ignore_index=True) if segment_tramo_rows else pd.DataFrame()

    df_global.to_csv(REPORTS_DIR / f"metricas_{split_name}_{report_prefix}.csv", index=False)
    df_tramos.to_csv(REPORTS_DIR / f"metricas_por_tramo_{split_name}_{report_prefix}.csv", index=False)
    df_segmentos.to_csv(REPORTS_DIR / f"metricas_por_segmento_{split_name}_{report_prefix}.csv", index=False)
    df_segmento_tramos.to_csv(REPORTS_DIR / f"metricas_por_segmento_tramo_{split_name}_{report_prefix}.csv", index=False)
    predicciones.to_csv(REPORTS_DIR / f"predicciones_{split_name}_{report_prefix}.csv", index=False)

    # Alias historicos para no romper comparaciones ya existentes sobre holdout.
    if split_name == "holdout":
        df_tramos.to_csv(REPORTS_DIR / f"metricas_por_tramo_{report_prefix}.csv", index=False)

    markdown = [
        f"# Reporte de Evaluacion {model_name.upper()} - {split_name}",
        "",
        "## Definicion Operacional",
        "",
        f"PLOS se define como `LOS >= {UMBRAL_PLOS}` dias. Los tramos de evaluacion son: {', '.join(TRAMOS_LABELS)}.",
        "",
        "## Metricas Globales",
        "",
        formatear_metricas_markdown(global_metrics),
        "",
        "## Metricas por Segmento",
        "",
        dataframe_to_markdown(df_segmentos),
        "",
        "## Metricas por Tramo",
        "",
        dataframe_to_markdown(df_tramos.drop(columns=["modelo"])),
        "",
        "## Metricas por Segmento y Tramo",
        "",
        dataframe_to_markdown(df_segmento_tramos.drop(columns=["modelo"]) if not df_segmento_tramos.empty else df_segmento_tramos),
        "",
    ]
    (REPORTS_DIR / f"reporte_evaluacion_{report_prefix}_{split_name}.md").write_text("\n".join(markdown), encoding="utf-8")
    return {
        "global": df_global,
        "tramos": df_tramos,
        "segmentos": df_segmentos,
        "segmento_tramos": df_segmento_tramos,
    }


def guardar_reporte_train_holdout(
    model_name: str,
    report_prefix: str,
    train_artifacts: dict[str, pd.DataFrame],
    holdout_artifacts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    metric_cols = [
        "mae",
        "rmse",
        "medae",
        "me",
        "pup",
        "mae_asimetrico_alpha_2",
        "precision_plos_14",
        "recall_plos_14",
        "f1_plos_14",
    ]
    rows = []

    train_global = train_artifacts["global"].iloc[0]
    holdout_global = holdout_artifacts["global"].iloc[0]
    for metric in metric_cols:
        train_value = float(train_global[metric])
        holdout_value = float(holdout_global[metric])
        rows.append({
            "modelo": model_name,
            "alcance": "global",
            "segmento": "todos",
            "metrica": metric,
            "train": train_value,
            "holdout": holdout_value,
            "gap_holdout_minus_train": holdout_value - train_value,
            "ratio_holdout_train": holdout_value / train_value if train_value != 0 else np.nan,
        })

    train_segments = train_artifacts["segmentos"]
    holdout_segments = holdout_artifacts["segmentos"]
    for segmento in sorted(set(train_segments["segmento"]) | set(holdout_segments["segmento"])):
        train_row = train_segments[train_segments["segmento"] == segmento].iloc[0]
        holdout_row = holdout_segments[holdout_segments["segmento"] == segmento].iloc[0]
        for metric in metric_cols:
            train_value = float(train_row[metric])
            holdout_value = float(holdout_row[metric])
            rows.append({
                "modelo": model_name,
                "alcance": "segmento",
                "segmento": segmento,
                "metrica": metric,
                "train": train_value,
                "holdout": holdout_value,
                "gap_holdout_minus_train": holdout_value - train_value,
                "ratio_holdout_train": holdout_value / train_value if train_value != 0 else np.nan,
            })

    df_gap = pd.DataFrame(rows)
    df_gap.to_csv(REPORTS_DIR / f"comparacion_train_holdout_{report_prefix}.csv", index=False)

    markdown = [
        f"# Reporte Train vs Holdout {model_name.upper()}",
        "",
        "## Definicion Operacional",
        "",
        f"PLOS se define como `LOS >= {UMBRAL_PLOS}` dias. Los tramos de evaluacion son: {', '.join(TRAMOS_LABELS)}.",
        "",
        "## Lectura",
        "",
        "Un gap alto entre holdout y train indica riesgo de sobreajuste. Las filas por segmento permiten verificar si urgentes y programados se comportan distinto.",
        "",
        "## Metricas Train por Segmento",
        "",
        dataframe_to_markdown(train_artifacts["segmentos"]),
        "",
        "## Metricas Holdout por Segmento",
        "",
        dataframe_to_markdown(holdout_artifacts["segmentos"]),
        "",
        "## Gap Train vs Holdout",
        "",
        dataframe_to_markdown(df_gap),
        "",
        "## Holdout por Segmento y Tramo",
        "",
        dataframe_to_markdown(holdout_artifacts["segmento_tramos"].drop(columns=["modelo"])),
        "",
    ]
    (REPORTS_DIR / f"reporte_evaluacion_{report_prefix}.md").write_text("\n".join(markdown), encoding="utf-8")
    return df_gap


def tiempo(label: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start = time.time()
            print(f"    Iniciando {label}...")
            result = func(*args, **kwargs)
            print(f"    {label} completado en {time.time() - start:.2f}s")
            return result
        return wrapper
    return decorator
