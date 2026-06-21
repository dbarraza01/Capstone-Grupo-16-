"""Metricas operacionales para prediccion de estancia hospitalaria.

Las funciones de este modulo trabajan siempre en dias reales. El MAE
asimetrico penaliza la subestimacion con alpha=2 por defecto, siguiendo el
criterio operacional del proyecto.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer


TRAMOS_BINS = [-1, 2, 6, 13, 26, np.inf]
TRAMOS_LABELS = ["0-2", "3-6", "7-13", "14-26", "27+"]


def _to_numpy(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _validar_vectores(y_true: Iterable[float], y_pred: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = _to_numpy(y_true)
    y_pred_arr = _to_numpy(y_pred)
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    if len(y_true_arr) == 0:
        raise ValueError("No se pueden calcular metricas con vectores vacios")
    return y_true_arr, y_pred_arr


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    return float(np.mean(np.abs(y_pred_arr - y_true_arr)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    return float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))


def medae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    return float(np.median(np.abs(y_pred_arr - y_true_arr)))


def error_medio(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """ME: promedio de prediccion - real. Valor negativo implica subestimacion."""
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    return float(np.mean(y_pred_arr - y_true_arr))


def pup(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """PUP: proporcion de pacientes subestimados."""
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    return float(np.mean(y_pred_arr < y_true_arr))


def mae_asimetrico(y_true: Iterable[float], y_pred: Iterable[float], alpha: float = 2.0) -> float:
    """MAE que penaliza alpha veces los casos subestimados."""
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    errores_abs = np.abs(y_pred_arr - y_true_arr)
    pesos = np.where(y_pred_arr < y_true_arr, alpha, 1.0)
    return float(np.mean(pesos * errores_abs))


def scorer_mae_asimetrico(alpha: float = 2.0):
    """Scorer compatible con RandomizedSearchCV."""
    return make_scorer(mae_asimetrico, greater_is_better=False, alpha=alpha)


def serie_tramos_los(y_true: Iterable[float]) -> pd.Series:
    y_true_arr = _to_numpy(y_true)
    return pd.cut(y_true_arr, bins=TRAMOS_BINS, labels=TRAMOS_LABELS)


def calcular_metricas_globales(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    alpha: float = 2.0,
) -> dict[str, float]:
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    abs_error = np.abs(y_pred_arr - y_true_arr)
    return {
        "n_casos": int(len(y_true_arr)),
        "mae": mae(y_true_arr, y_pred_arr),
        "rmse": rmse(y_true_arr, y_pred_arr),
        "medae": medae(y_true_arr, y_pred_arr),
        "me": error_medio(y_true_arr, y_pred_arr),
        "pup": pup(y_true_arr, y_pred_arr),
        "mae_asimetrico_alpha_2": mae_asimetrico(y_true_arr, y_pred_arr, alpha=alpha),
        "pct_error_abs_le_1d": float(np.mean(abs_error <= 1)),
        "pct_error_abs_le_3d": float(np.mean(abs_error <= 3)),
        "pct_error_abs_le_7d": float(np.mean(abs_error <= 7)),
        "los_real_promedio": float(np.mean(y_true_arr)),
        "los_pred_promedio": float(np.mean(y_pred_arr)),
    }


def calcular_metricas_por_tramo(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    alpha: float = 2.0,
) -> pd.DataFrame:
    y_true_arr, y_pred_arr = _validar_vectores(y_true, y_pred)
    tramos = serie_tramos_los(y_true_arr)
    filas = []
    for tramo in TRAMOS_LABELS:
        mask = np.asarray(tramos == tramo)
        if not mask.any():
            filas.append({
                "tramo": tramo,
                "n_casos": 0,
                "mae": np.nan,
                "rmse": np.nan,
                "medae": np.nan,
                "me": np.nan,
                "pup": np.nan,
                "mae_asimetrico_alpha_2": np.nan,
                "los_real_promedio": np.nan,
                "los_pred_promedio": np.nan,
            })
            continue

        m = calcular_metricas_globales(y_true_arr[mask], y_pred_arr[mask], alpha=alpha)
        filas.append({"tramo": tramo, **m})
    return pd.DataFrame(filas)


def formatear_metricas_markdown(metricas: dict[str, float]) -> str:
    orden = [
        "n_casos",
        "mae",
        "rmse",
        "medae",
        "me",
        "pup",
        "mae_asimetrico_alpha_2",
        "pct_error_abs_le_1d",
        "pct_error_abs_le_3d",
        "pct_error_abs_le_7d",
    ]
    filas = ["| Metrica | Valor |", "|---|---:|"]
    for clave in orden:
        valor = metricas.get(clave)
        if valor is None:
            continue
        if clave == "n_casos":
            valor_fmt = f"{int(valor)}"
        elif clave.startswith("pct_") or clave == "pup":
            valor_fmt = f"{100 * valor:.2f}%"
        else:
            valor_fmt = f"{valor:.4f}"
        filas.append(f"| {clave} | {valor_fmt} |")
    return "\n".join(filas)


if __name__ == "__main__":
    y_real = np.array([1, 5, 10, 20, 30], dtype=float)
    y_pred = np.array([2, 3, 12, 18, 25], dtype=float)
    print(calcular_metricas_globales(y_real, y_pred))
    print(calcular_metricas_por_tramo(y_real, y_pred).to_string(index=False))
