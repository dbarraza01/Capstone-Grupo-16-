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
from matplotlib.patches import Patch


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
SEGMENT_CI_DIR = PROJECT_DIR / "graficos_csv_por_segmento"
OUTPUT_DIR = PROJECT_DIR / "graficos_png_presentacion"
CSV_DIR = PROJECT_DIR / "graficos_csv_presentacion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["XGB", "LR"]
SEGMENTS = [
    ("programado", "No urgencia"),
    ("urgente", "Urgencia"),
]
MODEL_COLORS = {"XGB": "#2563EB", "LR": "#EF4444"}
NAVY = "#172554"
BG = "#F4F6FB"
TRAMO_ORDER = ["0-2", "3-6", "7-13", "14+ (PLOS)"]


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
        "axes.titlesize": 14,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 10,
    })


def load_segment_metrics() -> pd.DataFrame:
    df = pd.read_csv(REPORTS_DIR / "comparacion_final_por_segmento.csv")
    df = df[(df["split"] == "holdout") & (df["modelo"].isin(MODELS))].copy()
    df["segmento_label"] = df["segmento"].map(dict(SEGMENTS))
    return df


def load_tramo_metrics() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        path = REPORTS_DIR / f"metricas_por_segmento_tramo_holdout_{model.lower()}.csv"
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["split"] == "holdout") & (df["modelo"].isin(MODELS))].copy()
    df["segmento_label"] = df["segmento"].map(dict(SEGMENTS))
    df["tramo"] = pd.Categorical(df["tramo"], categories=TRAMO_ORDER, ordered=True)
    return df.sort_values(["segmento", "tramo", "modelo"])


def load_ci(metric: str, segment_folder: str, model: str, tramo: str | None = None) -> tuple[float, float] | None:
    file_name = "ic95_bootstrap_metricas_por_tramo_real_xgb_lr.csv" if tramo else "ic95_bootstrap_metricas_globales_xgb_lr.csv"
    path = SEGMENT_CI_DIR / segment_folder / file_name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    mask = (df["modelo"] == model) & (df["metrica"] == metric)
    if tramo:
        mask &= df["tramo_real"] == tramo
    row = df[mask]
    if row.empty:
        return None
    return float(row["ic95_inf"].iloc[0]), float(row["ic95_sup"].iloc[0])


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.2f}", offset_frac: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * offset_frac
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=NAVY,
        )


def plot_grouped_bar(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    use_ci: bool = False,
    tramo: str | None = None,
) -> None:
    x = np.arange(len(SEGMENTS))
    width = 0.34
    for i, model in enumerate(MODELS):
        values = []
        lower = []
        upper = []
        for segment_key, _segment_label in SEGMENTS:
            row = df[(df["segmento"] == segment_key) & (df["modelo"] == model)]
            values.append(float(row[metric].iloc[0]))
            folder = "no_urgencia" if segment_key == "programado" else "urgente"
            ci = load_ci(metric, folder, model, tramo=tramo) if use_ci else None
            if ci:
                lower.append(ci[0])
                upper.append(ci[1])
            else:
                lower.append(np.nan)
                upper.append(np.nan)
        values = np.array(values)
        pos = x + (i - 0.5) * width
        bars = ax.bar(pos, values, width=width, color=MODEL_COLORS[model], label=model)
        if use_ci and not np.isnan(lower).all():
            lower_arr = np.array(lower)
            upper_arr = np.array(upper)
            ax.errorbar(pos, values, yerr=np.vstack([values - lower_arr, upper_arr - values]), fmt="none", ecolor="#111827", capsize=4, lw=1.2)
        annotate_bars(ax, bars)
    ax.set_title(title, loc="left", pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in SEGMENTS], fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)


def plot_plos_table(ax: plt.Axes, segment_df: pd.DataFrame) -> None:
    ax.axis("off")
    metrics = [
        ("precision_plos_14", "Precision"),
        ("recall_plos_14", "Recall"),
        ("f1_plos_14", "F1"),
    ]
    rows = []
    colors = []
    for segment_key, segment_label in SEGMENTS:
        for model in MODELS:
            row = segment_df[(segment_df["segmento"] == segment_key) & (segment_df["modelo"] == model)].iloc[0]
            rows.append([segment_label, model, *[f"{row[col] * 100:.1f}%" for col, _ in metrics]])
            colors.append(MODEL_COLORS[model])

    table = ax.table(
        cellText=rows,
        colLabels=["Segmento", "Modelo", *[label for _, label in metrics]],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.00, 0.00, 1.00, 0.86],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.35)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DEE9")
        if row_idx == 0:
            cell.set_facecolor("#E8EEF9")
            cell.set_text_props(fontweight="bold", color=NAVY)
        elif col_idx == 1:
            color = colors[row_idx - 1]
            cell.set_facecolor(color)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("white")
    ax.set_title("Metricas PLOS14 por segmento", loc="left", pad=10, fontweight="bold")


def plot_tramo_heatmap(ax: plt.Axes, tramo_df: pd.DataFrame) -> None:
    rows = []
    labels = []
    for segment_key, segment_label in SEGMENTS:
        for tramo in TRAMO_ORDER:
            labels.append(f"{segment_label} · {tramo}")
            vals = []
            for model in MODELS:
                row = tramo_df[
                    (tramo_df["segmento"] == segment_key)
                    & (tramo_df["tramo"].astype(str) == tramo)
                    & (tramo_df["modelo"] == model)
                ]
                vals.append(float(row["mae"].iloc[0]) if not row.empty else np.nan)
            rows.append(vals)
    data = np.array(rows)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_title("MAE por tramo LOS real: ambos segmentos en una sola vista", loc="left", pad=10)
    ax.set_xticks(np.arange(len(MODELS)))
    ax.set_xticklabels(MODELS, fontweight="bold")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            text_color = "white" if value > np.nanmax(data) * 0.55 else NAVY
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=9, fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("MAE dias")


def build_summary_csv(segment_df: pd.DataFrame, tramo_df: pd.DataFrame) -> None:
    rows = []
    for segment_key, segment_label in SEGMENTS:
        for model in MODELS:
            global_row = segment_df[(segment_df["segmento"] == segment_key) & (segment_df["modelo"] == model)].iloc[0]
            plos_row = tramo_df[
                (tramo_df["segmento"] == segment_key)
                & (tramo_df["modelo"] == model)
                & (tramo_df["tramo"].astype(str) == "14+ (PLOS)")
            ].iloc[0]
            rows.append({
                "segmento": segment_key,
                "segmento_label": segment_label,
                "modelo": model,
                "n_casos": int(global_row["n_casos"]),
                "mae": float(global_row["mae"]),
                "rmse": float(global_row["rmse"]),
                "mae_asimetrico_alpha_2": float(global_row["mae_asimetrico_alpha_2"]),
                "precision_plos_14": float(global_row["precision_plos_14"]),
                "recall_plos_14": float(global_row["recall_plos_14"]),
                "f1_plos_14": float(global_row["f1_plos_14"]),
                "mae_tramo_14_plos": float(plos_row["mae"]),
            })
    pd.DataFrame(rows).to_csv(CSV_DIR / "dashboard_segmentos_compacto_xgb_lr.csv", index=False)


def main() -> None:
    setup_style()
    segment_df = load_segment_metrics()
    tramo_df = load_tramo_metrics()
    build_summary_csv(segment_df, tramo_df)

    plos_df = tramo_df[tramo_df["tramo"].astype(str) == "14+ (PLOS)"].copy()

    fig = plt.figure(figsize=(17.8, 10.0))
    gs = fig.add_gridspec(3, 6, height_ratios=[0.42, 1.0, 1.65], hspace=0.48, wspace=0.48)

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.00, 0.78, "XGBoost vs Regresion Lineal Base por tipo de admision", fontsize=25, fontweight="bold", color=NAVY)
    title_ax.text(
        0.00,
        0.28,
        "Una sola vista para presentacion: no urgencia/programado y urgencia · holdout actual · PLOS = LOS >= 14 dias",
        fontsize=12.5,
        color="#6B7280",
    )
    title_ax.legend(
        handles=[Patch(facecolor=MODEL_COLORS[model], label=model) for model in MODELS],
        frameon=False,
        loc="center right",
        ncol=2,
    )

    ax_mae = fig.add_subplot(gs[1, 0:2])
    plot_grouped_bar(ax_mae, segment_df, "mae", "MAE global", "Dias", use_ci=True)

    ax_plos_mae = fig.add_subplot(gs[1, 2:4])
    plot_grouped_bar(ax_plos_mae, plos_df, "mae", "MAE en pacientes PLOS14", "Dias", use_ci=True, tramo="14+ (PLOS)")

    ax_table = fig.add_subplot(gs[1, 4:6])
    plot_plos_table(ax_table, segment_df)

    ax_heat = fig.add_subplot(gs[2, :])
    plot_tramo_heatmap(ax_heat, tramo_df)

    output_path = OUTPUT_DIR / "dashboard_segmentos_compacto_xgb_vs_lr.png"
    fig.savefig(output_path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print(f"Grafico guardado en: {output_path}")
    print(f"CSV resumen guardado en: {CSV_DIR / 'dashboard_segmentos_compacto_xgb_lr.csv'}")


if __name__ == "__main__":
    main()
