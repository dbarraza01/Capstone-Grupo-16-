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


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR = PROJECT_DIR / "graficos_png_presentacion"
CSV_DIR = PROJECT_DIR / "graficos_csv_presentacion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = REPORTS_DIR / "predicciones_holdout_xgb.csv"
OUTPUT_PATH = OUTPUT_DIR / "xgb_error_los_predicho_menos_real_por_tramo_y_segmento.png"
CSV_PATH = CSV_DIR / "xgb_error_los_predicho_menos_real_por_tramo_y_segmento.csv"

TRAMO_BINS = [-1, 2, 6, 13, np.inf]
TRAMO_LABELS = ["0-2", "3-6", "7-13", "14+ (PLOS)"]
SEGMENTS = [("programado", "No urgencia / programado"), ("urgente", "Urgencia")]
NAVY = "#172554"
BG = "#F4F6FB"
UNDER_COLOR = "#EF4444"
OVER_COLOR = "#2563EB"


def build_summary() -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_PATH)
    df["los_real"] = df["los_dias_reales"].astype(float)
    df["los_predicho"] = df["los_dias_predichos"].astype(float)
    df["error_pred_menos_real"] = df["los_predicho"] - df["los_real"]
    df["error_abs"] = df["error_pred_menos_real"].abs()
    df["tramo_los_real"] = pd.cut(df["los_real"], TRAMO_BINS, labels=TRAMO_LABELS)

    summary = (
        df.groupby(["segmento", "tramo_los_real"], observed=False)
        .agg(
            n_casos=("case_id", "count"),
            los_real_promedio=("los_real", "mean"),
            los_predicho_promedio=("los_predicho", "mean"),
            error_promedio_pred_menos_real=("error_pred_menos_real", "mean"),
            mae=("error_abs", "mean"),
        )
        .reset_index()
    )
    summary["tramo_los_real"] = pd.Categorical(summary["tramo_los_real"], categories=TRAMO_LABELS, ordered=True)
    summary = summary.sort_values(["segmento", "tramo_los_real"])
    summary.to_csv(CSV_PATH, index=False)
    return summary


def annotate_bars(ax: plt.Axes, bars, counts: list[int]) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.025
    for bar, n_cases in zip(bars, counts):
        value = bar.get_height()
        y = value + offset if value >= 0 else value - offset
        va = "bottom" if value >= 0 else "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{value:+.1f}d\nn={n_cases}",
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=NAVY,
        )


def main() -> None:
    summary = build_summary()

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

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7), sharey=True)
    y_min = min(-11.5, summary["error_promedio_pred_menos_real"].min() - 1.2)
    y_max = max(3.2, summary["error_promedio_pred_menos_real"].max() + 1.2)

    for ax, (segment_key, segment_label) in zip(axes, SEGMENTS):
        seg = summary[summary["segmento"] == segment_key].set_index("tramo_los_real").loc[TRAMO_LABELS].reset_index()
        values = seg["error_promedio_pred_menos_real"].to_numpy(dtype=float)
        colors = [OVER_COLOR if value >= 0 else UNDER_COLOR for value in values]
        bars = ax.bar(TRAMO_LABELS, values, color=colors, width=0.62)
        ax.axhline(0, color="#111827", linewidth=1.2)
        ax.set_ylim(y_min, y_max)
        ax.set_title(segment_label, loc="left", fontsize=17, fontweight="bold", pad=14)
        ax.set_xlabel("Tramo segun LOS real", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        annotate_bars(ax, bars, seg["n_casos"].astype(int).tolist())

    axes[0].set_ylabel("Diferencia promedio: LOS predicho - LOS real (dias)", fontsize=11, fontweight="bold")

    fig.suptitle(
        "XGBoost: diferencia entre LOS estimado y LOS real por tramo",
        x=0.04,
        y=1.04,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=NAVY,
    )
    fig.text(
        0.04,
        0.965,
        "Holdout actual | barras positivas = sobreestimacion | barras negativas = subestimacion",
        ha="left",
        fontsize=11.5,
        color="#6B7280",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=190, bbox_inches="tight")
    plt.close(fig)

    print(f"Grafico guardado en: {OUTPUT_PATH}")
    print(f"CSV resumen guardado en: {CSV_PATH}")


if __name__ == "__main__":
    main()
