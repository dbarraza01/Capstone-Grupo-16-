from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "capstone_plot_cache"
(PLOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(PLOT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))
os.environ.setdefault("FC_CACHEDIR", str(PLOT_CACHE_DIR / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parent
PREDICTIONS_PATH = PROJECT_DIR / "reports" / "predicciones_holdout_lr.csv"
OUTPUT_PATH = CURRENT_DIR / "grafico_regresion_lineal_real_vs_predicho.png"
SUMMARY_PATH = CURRENT_DIR / "metricas_grafico_regresion_lineal.csv"

AXIS_LIMIT = 80
NAVY = "#172554"
BG = "#F4F6FB"
SEGMENT_COLORS = {"programado": "#2563EB", "urgente": "#EF4444"}
SEGMENT_LABELS = {"programado": "No urgencia / programado", "urgente": "Urgencia"}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    abs_error = np.abs(error)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "me": float(np.mean(error)),
        "r2_holdout": float(1 - ss_res / ss_tot) if ss_tot else np.nan,
    }


def main() -> None:
    df = pd.read_csv(PREDICTIONS_PATH)
    df = df.rename(columns={"los_dias_reales": "y_true", "los_dias_predichos": "y_pred"})
    df = df[np.isfinite(df["y_true"]) & np.isfinite(df["y_pred"])].copy()

    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    visible_df = df[(df["y_true"] <= AXIS_LIMIT) & (df["y_pred"] <= AXIS_LIMIT)].copy()
    slope, intercept = np.polyfit(visible_df["y_true"], visible_df["y_pred"], 1)
    metrics = regression_metrics(y_true, y_pred)
    metrics.update({
        "slope_visible_pred_vs_real": float(slope),
        "intercept_visible_pred_vs_real": float(intercept),
        "n_visible_0_80": int(len(visible_df)),
    })
    pd.DataFrame([metrics]).to_csv(SUMMARY_PATH, index=False)

    outside_view = int(((df["y_true"] > AXIS_LIMIT) | (df["y_pred"] > AXIS_LIMIT)).sum())

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#D8DEE9",
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
    })

    fig, ax = plt.subplots(figsize=(11.5, 8))
    for segment, group in df.groupby("segmento", sort=False):
        ax.scatter(
            group["y_true"],
            group["y_pred"],
            s=24,
            alpha=0.50,
            color=SEGMENT_COLORS.get(segment, "#6B7280"),
            edgecolors="none",
            label=SEGMENT_LABELS.get(segment, segment),
        )

    x_line = np.linspace(0, AXIS_LIMIT, 200)
    ax.plot(x_line, x_line, color="#111827", linestyle="--", linewidth=1.6, label="Prediccion perfecta")
    ax.plot(x_line, intercept + slope * x_line, color="#10B981", linewidth=2.4, label="Tendencia visible")

    ax.set_xlim(0, AXIS_LIMIT)
    ax.set_ylim(0, AXIS_LIMIT)
    ax.set_xlabel("LOS real observado (dias)", fontsize=12, fontweight="bold")
    ax.set_ylabel("LOS predicho por regresion lineal (dias)", fontsize=12, fontweight="bold")
    ax.set_title("Regresion Lineal Base: LOS real vs LOS predicho", loc="left", fontsize=19, fontweight="bold", pad=18)
    ax.text(
        0,
        1.015,
        "Holdout actual | diagonal = prediccion perfecta | vista operacional 0-80 dias",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color="#6B7280",
        fontsize=11,
    )

    summary_text = (
        f"n = {metrics['n']:,}\n"
        f"MAE = {metrics['mae']:.2f} dias\n"
        f"RMSE = {metrics['rmse']:.2f} dias\n"
        f"ME = {metrics['me']:.2f} dias\n"
        f"R2 holdout = {metrics['r2_holdout']:.3f}\n"
        f"Tendencia visible: y_pred = {slope:.2f} * y_real + {intercept:.2f}\n"
        f"Fuera de vista 0-80: {outside_view} casos"
    )
    ax.text(
        0.03,
        0.96,
        summary_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )

    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico guardado en: {OUTPUT_PATH}")
    print(f"Metricas guardadas en: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
