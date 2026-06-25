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
from matplotlib.patches import FancyBboxPatch, Patch


PROJECT_DIR = Path(__file__).resolve().parent
SEGMENT_CSV_ROOT = PROJECT_DIR / "graficos_csv_por_segmento"
OUTPUT_DIR = PROJECT_DIR / "graficos_png_segmentos_juntos"
CSV_DIR = PROJECT_DIR / "graficos_csv_segmentos_juntos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["XGB", "LR"]
MODEL_COLORS = {"XGB": "#2563EB", "LR": "#EF4444"}
METHODS = ["global", "segmento", "pred_tramo", "segmento_pred_tramo"]
METHOD_LABELS = {
    "global": "Global",
    "segmento": "Segmento",
    "pred_tramo": "Tramo pred.",
    "segmento_pred_tramo": "Segmento+tramo",
}
METHOD_COLORS = {
    "global": "#6B7280",
    "segmento": "#14B8A6",
    "pred_tramo": "#6366F1",
    "segmento_pred_tramo": "#2563EB",
}
SEGMENTS = [
    ("no_urgencia", "programado", "No urgencia"),
    ("urgente", "urgente", "Urgencia"),
]
TRAMO_LABELS = ["0-2", "3-6", "7-13", "14+ (PLOS)"]
NAVY = "#172554"
BG = "#F4F6FB"


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": "#D8DEE9",
        "axes.labelcolor": NAVY,
        "xtick.color": NAVY,
        "ytick.color": NAVY,
        "text.color": NAVY,
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.titlesize": 15,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def segment_path(folder: str, filename: str) -> Path:
    path = SEGMENT_CSV_ROOT / folder / filename
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta primero generar_graficos_xgb_vs_lr_por_segmento.py")
    return path


def with_segment_columns(df: pd.DataFrame, folder: str, segment_key: str, segment_label: str) -> pd.DataFrame:
    out = df.copy()
    out["segmento_folder"] = folder
    out["segmento_key"] = segment_key
    out["segmento_label"] = segment_label
    return out


def load_global_ci() -> pd.DataFrame:
    frames = []
    for folder, segment_key, segment_label in SEGMENTS:
        df = pd.read_csv(segment_path(folder, "ic95_bootstrap_metricas_globales_xgb_lr.csv"))
        frames.append(with_segment_columns(df, folder, segment_key, segment_label))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CSV_DIR / "metricas_globales_segmentos_juntos_xgb_lr.csv", index=False)
    return out


def load_tramo_ci() -> pd.DataFrame:
    frames = []
    for folder, segment_key, segment_label in SEGMENTS:
        df = pd.read_csv(segment_path(folder, "ic95_bootstrap_metricas_por_tramo_real_xgb_lr.csv"))
        frames.append(with_segment_columns(df, folder, segment_key, segment_label))
    out = pd.concat(frames, ignore_index=True)
    out["tramo_real"] = pd.Categorical(out["tramo_real"], categories=TRAMO_LABELS, ordered=True)
    out = out.sort_values(["segmento_folder", "tramo_real", "modelo", "metrica"])
    out.to_csv(CSV_DIR / "metricas_por_tramo_segmentos_juntos_xgb_lr.csv", index=False)
    return out


def load_ip90_eval() -> pd.DataFrame:
    frames = []
    for folder, segment_key, segment_label in SEGMENTS:
        df = pd.read_csv(segment_path(folder, "evaluacion_ip90_xgb_lr.csv"))
        frames.append(with_segment_columns(df, folder, segment_key, segment_label))
    out = pd.concat(frames, ignore_index=True)
    out["tramo_real"] = pd.Categorical(out["tramo_real"], categories=TRAMO_LABELS + ["todos"], ordered=True)
    out.to_csv(CSV_DIR / "evaluacion_ip90_segmentos_juntos_xgb_lr.csv", index=False)
    return out


def load_predictions() -> pd.DataFrame:
    frames = []
    for folder, segment_key, segment_label in SEGMENTS:
        df = pd.read_csv(segment_path(folder, "predicciones_holdout_modelos_con_tramos_xgb_lr.csv"))
        frames.append(with_segment_columns(df, folder, segment_key, segment_label))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CSV_DIR / "predicciones_holdout_segmentos_juntos_xgb_lr.csv", index=False)
    return out


def finish_plot(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=190, bbox_inches="tight")
    plt.close(fig)


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", pad=14, fontsize=16, color=NAVY)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", color="#6B7280", fontsize=10.5)


def annotate_bars(ax: plt.Axes, bars, formatter=lambda value: f"{value:.2f}", pad: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * pad
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        va = "bottom" if height >= 0 else "top"
        y = height + offset if height >= 0 else height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            formatter(height),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold",
            color=NAVY,
        )


def row_for_metric(df: pd.DataFrame, segment_label: str, model: str, metric: str, tramo: str | None = None) -> pd.Series:
    mask = (df["segmento_label"] == segment_label) & (df["modelo"] == model) & (df["metrica"] == metric)
    if tramo is not None:
        mask &= df["tramo_real"].astype(str) == tramo
    row = df[mask]
    if row.empty:
        raise ValueError(f"No hay datos para segmento={segment_label}, modelo={model}, metrica={metric}, tramo={tramo}")
    return row.iloc[0]


def plot_segment_metric(global_ci: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(SEGMENTS))
    width = 0.34
    max_high = 0.0
    for i, model in enumerate(MODELS):
        values = []
        lows = []
        highs = []
        for _folder, _segment_key, segment_label in SEGMENTS:
            row = row_for_metric(global_ci, segment_label, model, metric)
            factor = 100 if percent else 1
            values.append(row["valor"] * factor)
            lows.append(row["ic95_inf"] * factor)
            highs.append(row["ic95_sup"] * factor)
        values = np.array(values, dtype=float)
        lows = np.array(lows, dtype=float)
        highs = np.array(highs, dtype=float)
        max_high = max(max_high, float(np.nanmax(highs)))
        pos = x + (i - 0.5) * width
        bars = ax.bar(pos, values, width=width, color=MODEL_COLORS[model], label=model)
        ax.errorbar(pos, values, yerr=np.vstack([values - lows, highs - values]), fmt="none", ecolor="#111827", capsize=5, lw=1.4)
        annotate_bars(ax, bars, formatter=(lambda value: f"{value:.1f}%") if percent else (lambda value: f"{value:.2f}"))
    add_title(ax, title, "Ambos segmentos en la misma figura · IC95 bootstrap · holdout actual")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in SEGMENTS], fontweight="bold")
    if percent:
        ax.set_ylim(0, 105)
    else:
        ax.set_ylim(0, max_high * 1.18 if max_high > 0 else 1)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    finish_plot(fig, filename)


def plot_plos_metrics(global_ci: pd.DataFrame, filename: str, title: str) -> None:
    metrics = ["precision_plos_14", "recall_plos_14", "f1_plos_14"]
    labels = ["Precision", "Recall", "F1"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, (_folder, _segment_key, segment_label) in zip(axes, SEGMENTS):
        x = np.arange(len(metrics))
        width = 0.34
        for i, model in enumerate(MODELS):
            vals = []
            lows = []
            highs = []
            for metric in metrics:
                row = row_for_metric(global_ci, segment_label, model, metric)
                vals.append(row["valor"] * 100)
                lows.append(row["ic95_inf"] * 100)
                highs.append(row["ic95_sup"] * 100)
            vals = np.array(vals)
            lows = np.array(lows)
            highs = np.array(highs)
            pos = x + (i - 0.5) * width
            bars = ax.bar(pos, vals, width=width, color=MODEL_COLORS[model], label=model)
            ax.errorbar(pos, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="#111827", capsize=4, lw=1.2)
            annotate_bars(ax, bars, formatter=lambda value: f"{value:.0f}%", pad=0.015)
        add_title(ax, segment_label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Porcentaje")
    axes[1].legend(frameon=False)
    fig.suptitle(title, x=0.03, y=1.04, ha="left", fontsize=18, fontweight="bold", color=NAVY)
    finish_plot(fig, filename)


def plot_metric_by_tramo(tramo_ci: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.8), sharey=percent)
    panel_lims = []
    for ax, (_folder, _segment_key, segment_label) in zip(axes, SEGMENTS):
        panel_min = 0.0
        panel_max = 0.0
        x = np.arange(len(TRAMO_LABELS))
        width = 0.34
        for i, model in enumerate(MODELS):
            values = []
            lows = []
            highs = []
            for tramo in TRAMO_LABELS:
                row = row_for_metric(tramo_ci, segment_label, model, metric, tramo=tramo)
                factor = 100 if percent else 1
                values.append(row["valor"] * factor)
                lows.append(row["ic95_inf"] * factor)
                highs.append(row["ic95_sup"] * factor)
            values = np.array(values, dtype=float)
            lows = np.array(lows, dtype=float)
            highs = np.array(highs, dtype=float)
            panel_min = min(panel_min, float(np.nanmin(lows)))
            panel_max = max(panel_max, float(np.nanmax(highs)))
            pos = x + (i - 0.5) * width
            bars = ax.bar(pos, values, width=width, color=MODEL_COLORS[model], label=model)
            ax.errorbar(pos, values, yerr=np.vstack([values - lows, highs - values]), fmt="none", ecolor="#111827", capsize=3, lw=1.0)
            annotate_bars(ax, bars, formatter=(lambda value: f"{value:.0f}%") if percent else (lambda value: f"{value:.1f}"), pad=0.015)
        add_title(ax, segment_label)
        ax.set_xticks(x)
        ax.set_xticklabels(TRAMO_LABELS, rotation=0, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "me":
            ax.axhline(0, color="#111827", lw=1.1)
        panel_lims.append((panel_min, panel_max))
        ax.set_ylabel(ylabel)
    axes[0].set_ylabel(ylabel)
    if percent:
        for ax in axes:
            ax.set_ylim(0, 105)
    elif metric == "me":
        for ax, (panel_min, panel_max) in zip(axes, panel_lims):
            pad = (panel_max - panel_min) * 0.12 if panel_max > panel_min else 1
            ax.set_ylim(panel_min - pad, panel_max + pad)
    else:
        for ax, (_panel_min, panel_max) in zip(axes, panel_lims):
            ax.set_ylim(0, panel_max * 1.18 if panel_max > 0 else 1)
    axes[1].legend(frameon=False)
    fig.suptitle(title, x=0.03, y=1.04, ha="left", fontsize=18, fontweight="bold", color=NAVY)
    finish_plot(fig, filename)


def plot_ip90_by_tramo(eval_df: pd.DataFrame, value_col: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharey=percent)
    axes = axes.reshape(2, 2)
    global_max = 0.0
    for row_idx, (_folder, _segment_key, segment_label) in enumerate(SEGMENTS):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            sub = eval_df[
                (eval_df["segmento_label"] == segment_label)
                & (eval_df["modelo"] == model)
                & (eval_df["alcance"] == "tramo_real")
            ].copy()
            x = np.arange(len(TRAMO_LABELS))
            width = 0.18
            for i, method in enumerate(METHODS):
                vals = []
                for tramo in TRAMO_LABELS:
                    row = sub[(sub["tramo_real"].astype(str) == tramo) & (sub["metodo_ip90"] == method)].iloc[0]
                    vals.append(row[value_col] * (100 if percent else 1))
                vals = np.array(vals)
                global_max = max(global_max, float(np.nanmax(vals)))
                pos = x + (i - 1.5) * width
                bars = ax.bar(pos, vals, width=width, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
                annotate_bars(ax, bars, formatter=(lambda value: f"{value:.0f}%") if percent else (lambda value: f"{value:.1f}"), pad=0.01)
            add_title(ax, f"{segment_label} · {model}")
            if percent:
                ax.axhline(90, ls="--", color="#EF4444", lw=1.1)
            ax.set_xticks(x)
            ax.set_xticklabels(TRAMO_LABELS, fontweight="bold")
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    if percent:
        for ax in axes.flatten():
            ax.set_ylim(0, 110)
    else:
        for ax in axes.flatten():
            ax.set_ylim(0, global_max * 1.18 if global_max > 0 else 1)
    handles = [Patch(facecolor=METHOD_COLORS[method], label=METHOD_LABELS[method]) for method in METHODS]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="upper right", bbox_to_anchor=(0.98, 1.02))
    fig.suptitle(title, x=0.03, y=1.03, ha="left", fontsize=18, fontweight="bold", color=NAVY)
    finish_plot(fig, filename)


def plot_ip90_coverage_by_segment(eval_df: pd.DataFrame) -> None:
    method = "segmento_pred_tramo"
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(SEGMENTS))
    width = 0.34
    for i, model in enumerate(MODELS):
        values = []
        for _folder, _segment_key, segment_label in SEGMENTS:
            row = eval_df[
                (eval_df["segmento_label"] == segment_label)
                & (eval_df["modelo"] == model)
                & (eval_df["alcance"] == "segmento")
                & (eval_df["metodo_ip90"] == method)
            ].iloc[0]
            values.append(row["coverage"] * 100)
        pos = x + (i - 0.5) * width
        bars = ax.bar(pos, values, width=width, color=MODEL_COLORS[model], label=model)
        annotate_bars(ax, bars, formatter=lambda value: f"{value:.1f}%")
    add_title(ax, "Cobertura IP90 por segmento", "Metodo: segmento + tramo predicho · calibrado con train")
    ax.axhline(90, ls="--", color="#EF4444", lw=1.2, label="Nominal 90%")
    ax.set_ylabel("Cobertura (%)")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, _, label in SEGMENTS], fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    finish_plot(fig, "grafico_14_ip90_cobertura_modelos_segmento_predtramo.png")


def plot_scatter(pred: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), sharex=True, sharey=True)
    max_val = min(80, float(np.nanpercentile(pred["y_true"], 99.2)))
    for row_idx, (_folder, _segment_key, segment_label) in enumerate(SEGMENTS):
        for col_idx, model in enumerate(MODELS):
            ax = axes[row_idx, col_idx]
            sub = pred[(pred["segmento_label"] == segment_label) & (pred["modelo"] == model)]
            ax.scatter(sub["y_true"], sub["y_pred"], c=MODEL_COLORS[model], alpha=0.45, s=16, edgecolors="none")
            ax.plot([0, max_val], [0, max_val], ls="--", color="#111827", lw=1.1)
            add_title(ax, f"{segment_label} · {model}")
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            if row_idx == 1:
                ax.set_xlabel("LOS real")
            if col_idx == 0:
                ax.set_ylabel("LOS predicho")
    fig.suptitle("Real vs predicho por segmento y modelo", x=0.03, y=1.03, ha="left", fontsize=18, fontweight="bold", color=NAVY)
    finish_plot(fig, "grafico_15_xgb_scatter_real_vs_pred.png")


def metric_value(global_ci: pd.DataFrame, segment_label: str, model: str, metric: str) -> float:
    return float(row_for_metric(global_ci, segment_label, model, metric)["valor"])


def tramo_value(tramo_ci: pd.DataFrame, segment_label: str, model: str, metric: str, tramo: str) -> float:
    return float(row_for_metric(tramo_ci, segment_label, model, metric, tramo=tramo)["valor"])


def plot_dashboard(global_ci: pd.DataFrame, tramo_ci: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(17, 9.5))
    ax.axis("off")
    ax.text(0.02, 0.94, "Resumen por segmento: XGBoost vs Regresion Lineal Base", fontsize=25, fontweight="bold", color=NAVY)
    ax.text(0.02, 0.89, "Holdout actual · PLOS = LOS >= 14 · ambos segmentos en una sola lamina", fontsize=12.5, color="#6B7280")

    cards = []
    for segment_label in ["No urgencia", "Urgencia"]:
        cards.extend([
            (segment_label, "MAE XGB", f"{metric_value(global_ci, segment_label, 'XGB', 'mae'):.2f}", "#2563EB"),
            (segment_label, "MAE LR", f"{metric_value(global_ci, segment_label, 'LR', 'mae'):.2f}", "#EF4444"),
            (segment_label, "F1 PLOS14 XGB", f"{metric_value(global_ci, segment_label, 'XGB', 'f1_plos_14') * 100:.1f}%", "#14B8A6"),
            (segment_label, "MAE 14+ XGB", f"{tramo_value(tramo_ci, segment_label, 'XGB', 'mae', '14+ (PLOS)'):.2f}", "#2563EB"),
            (segment_label, "MAE 14+ LR", f"{tramo_value(tramo_ci, segment_label, 'LR', 'mae', '14+ (PLOS)'):.2f}", "#EF4444"),
        ])

    positions = []
    for col, x0 in enumerate([0.04, 0.53]):
        y_values = [0.68, 0.52, 0.36, 0.20, 0.04]
        positions.extend([(x0, y) for y in y_values])

    for (segment_label, label, value, color), (x0, y0) in zip(cards, positions):
        box = FancyBboxPatch((x0, y0), 0.40, 0.115, boxstyle="round,pad=0.012,rounding_size=0.020", ec=color, fc="white", lw=1.8)
        ax.add_patch(box)
        ax.text(x0 + 0.02, y0 + 0.074, f"{segment_label} · {label}", fontsize=12, fontweight="bold", color=NAVY)
        ax.text(x0 + 0.02, y0 + 0.026, value, fontsize=21, fontweight="bold", color=color)
        if "MAE" in label:
            ax.text(x0 + 0.14, y0 + 0.031, "dias", fontsize=12, color="#6B7280")

    ax.plot([0.50, 0.50], [0.04, 0.80], color="#CBD5E1", lw=1.3)
    ax.text(0.04, 0.82, "No urgencia / programado", fontsize=15, fontweight="bold", color=NAVY)
    ax.text(0.53, 0.82, "Urgencia", fontsize=15, fontweight="bold", color=NAVY)
    finish_plot(fig, "grafico_16_dashboard_resumen_presentacion.png")


def main() -> None:
    setup_style()
    global_ci = load_global_ci()
    tramo_ci = load_tramo_ci()
    ip90_eval = load_ip90_eval()
    pred = load_predictions()

    plot_segment_metric(global_ci, "mae", "MAE por segmento con IC95", "Dias", "grafico_01_mae_global_ic95.png")
    plot_segment_metric(global_ci, "rmse", "RMSE por segmento con IC95", "Dias", "grafico_02_rmse_global_ic95.png")
    plot_segment_metric(global_ci, "mae_asimetrico_alpha_2", "MAE asimetrico por segmento con IC95", "Dias", "grafico_03_mae_asimetrico_global_ic95.png")
    plot_plos_metrics(global_ci, "grafico_04_plos14_precision_recall_f1_ic95.png", "Deteccion PLOS14 por segmento con IC95")

    plot_metric_by_tramo(tramo_ci, "mae", "MAE por tramo LOS real y segmento", "Dias", "grafico_05_mae_por_tramo_ic95.png")
    plot_metric_by_tramo(tramo_ci, "rmse", "RMSE por tramo LOS real y segmento", "Dias", "grafico_06_rmse_por_tramo_ic95.png")
    plot_metric_by_tramo(tramo_ci, "me", "Sesgo por tramo LOS real y segmento", "Predicho - real", "grafico_07_sesgo_por_tramo_ic95.png")
    plot_metric_by_tramo(tramo_ci, "pup", "Subestimacion por tramo LOS real y segmento", "Porcentaje", "grafico_08_subestimacion_por_tramo_ic95.png", percent=True)
    plot_metric_by_tramo(tramo_ci, "mae_asimetrico_alpha_2", "MAE asimetrico por tramo LOS real y segmento", "Dias", "grafico_09_mae_asimetrico_por_tramo_ic95.png")

    plot_segment_metric(global_ci, "mae", "MAE por tipo de admision con IC95", "Dias", "grafico_10_mae_por_segmento_ic95.png")
    plot_plos_metrics(global_ci, "grafico_11_plos14_por_segmento_ic95.png", "PLOS14 por tipo de admision")

    plot_ip90_by_tramo(ip90_eval, "coverage", "Cobertura IP90 por tramo, modelo y segmento", "Cobertura (%)", "grafico_12_xgb_cobertura_ip90_por_tramo_y_metodo.png", percent=True)
    plot_ip90_by_tramo(ip90_eval, "width_mean", "Ancho IP90 por tramo, modelo y segmento", "Ancho medio (dias)", "grafico_13_xgb_ancho_ip90_por_tramo_y_metodo.png")
    plot_ip90_coverage_by_segment(ip90_eval)
    plot_scatter(pred)
    plot_dashboard(global_ci, tramo_ci)

    print(f"Graficos guardados en: {OUTPUT_DIR}")
    print(f"CSVs auxiliares guardados en: {CSV_DIR}")
    for path in sorted(OUTPUT_DIR.glob("grafico_*.png")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
