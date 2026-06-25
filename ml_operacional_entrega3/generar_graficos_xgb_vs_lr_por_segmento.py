from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import generar_graficos_xgb_vs_lr as base


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_DIR / "graficos_png_por_segmento"
CSV_ROOT = PROJECT_DIR / "graficos_csv_por_segmento"

SEGMENTS = [
    ("programado", "no_urgencia", "No urgencia / programado"),
    ("urgente", "urgente", "Urgencia"),
]


def _metric_from_ci(global_ci: pd.DataFrame, model: str, metric: str) -> float:
    return float(global_ci[(global_ci["modelo"] == model) & (global_ci["metrica"] == metric)]["valor"].iloc[0])


def _tramo_metric(tramo_ci: pd.DataFrame, model: str, tramo: str, metric: str) -> float:
    row = tramo_ci[
        (tramo_ci["modelo"] == model)
        & (tramo_ci["tramo_real"] == tramo)
        & (tramo_ci["metrica"] == metric)
    ]
    if row.empty:
        return float("nan")
    return float(row["valor"].iloc[0])


def plot_mae_selected_segment(segment_ci: pd.DataFrame, segment_key: str, segment_label: str) -> None:
    metric = "mae"
    rows = [
        segment_ci[
            (segment_ci["modelo"] == model)
            & (segment_ci["segmento"] == segment_key)
            & (segment_ci["metrica"] == metric)
        ].iloc[0]
        for model in base.MODELS
    ]
    df = pd.DataFrame(rows)
    values = df["valor"].to_numpy()
    lows = df["ic95_inf"].to_numpy()
    highs = df["ic95_sup"].to_numpy()
    errors = np.vstack([values - lows, highs - values])

    fig, ax = base.plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["modelo"], values, color=[base.MODEL_COLORS[m] for m in df["modelo"]], width=0.62)
    ax.errorbar(df["modelo"], values, yerr=errors, fmt="none", ecolor="black", elinewidth=1.8, capsize=6)
    base.add_title(ax, f"MAE del segmento con IC95 - {segment_label}", "Holdout filtrado por segmento")
    ax.set_ylabel("Dias")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(highs) * 1.18 if max(highs) > 0 else 1)
    base.annotate_bars(ax, bars, lambda value: f"{value:.2f}")
    base.finish_plot(fig, "grafico_10_mae_por_segmento_ic95.png")


def plot_plos_selected_segment(segment_ci: pd.DataFrame, segment_key: str, segment_label: str) -> None:
    metrics = ["precision_plos_14", "recall_plos_14", "f1_plos_14"]
    labels = ["Precision", "Recall", "F1"]

    fig, ax = base.plt.subplots(figsize=(13, 7))
    x = np.arange(len(labels))
    width = 0.34
    for i, model in enumerate(base.MODELS):
        rows = [
            segment_ci[
                (segment_ci["modelo"] == model)
                & (segment_ci["segmento"] == segment_key)
                & (segment_ci["metrica"] == metric)
            ].iloc[0]
            for metric in metrics
        ]
        df = pd.DataFrame(rows)
        vals = df["valor"].to_numpy() * 100
        lows = df["ic95_inf"].to_numpy() * 100
        highs = df["ic95_sup"].to_numpy() * 100
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=base.MODEL_COLORS[model], label=model)
        ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=5)
        base.annotate_bars(ax, bars, lambda value: f"{value:.1f}%", pad=0.015)

    base.add_title(ax, f"PLOS14 del segmento con IC95 - {segment_label}", "PLOS definido como LOS real/predicho >= 14 dias")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    base.finish_plot(fig, "grafico_11_plos14_por_segmento_ic95.png")


def plot_plos_global_segment(global_ci: pd.DataFrame, segment_label: str) -> None:
    metrics = ["precision_plos_14", "recall_plos_14", "f1_plos_14"]
    labels = ["Precision", "Recall", "F1"]

    fig, ax = base.plt.subplots(figsize=(13, 7))
    x = np.arange(len(labels))
    width = 0.34
    for i, model in enumerate(base.MODELS):
        vals = []
        lows = []
        highs = []
        for metric in metrics:
            row = global_ci[(global_ci["modelo"] == model) & (global_ci["metrica"] == metric)].iloc[0]
            vals.append(row["valor"] * 100)
            lows.append(row["ic95_inf"] * 100)
            highs.append(row["ic95_sup"] * 100)
        vals = np.array(vals)
        lows = np.array(lows)
        highs = np.array(highs)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=base.MODEL_COLORS[model], label=model)
        ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=5)
        base.annotate_bars(ax, bars, lambda value: f"{value:.1f}%", pad=0.015)

    base.add_title(ax, f"Deteccion PLOS14 con IC95 - {segment_label}", "PLOS definido como LOS real/predicho >= 14 dias · holdout filtrado")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    base.finish_plot(fig, "grafico_04_plos14_precision_recall_f1_ic95.png")


def plot_ip90_selected_segment(eval_df: pd.DataFrame, segment_key: str, segment_label: str) -> None:
    method = "segmento_pred_tramo"
    sub = eval_df[(eval_df["alcance"] == "segmento") & (eval_df["metodo_ip90"] == method)]

    fig, ax = base.plt.subplots(figsize=(12, 7))
    values = [
        sub[(sub["modelo"] == model) & (sub["segmento"] == segment_key)]["coverage"].iloc[0] * 100
        for model in base.MODELS
    ]
    bars = ax.bar(base.MODELS, values, color=[base.MODEL_COLORS[model] for model in base.MODELS], width=0.62)
    base.annotate_bars(ax, bars, lambda value: f"{value:.1f}%", pad=0.015)
    base.add_title(ax, f"Cobertura IP90 del segmento - {segment_label}", "Metodo: segmento + tramo predicho · calibrado con train")
    ax.axhline(90, ls="--", color="#EF4444", lw=1.4, label="Nominal 90%")
    ax.set_ylabel("Cobertura")
    ax.set_ylim(0, 110)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    base.finish_plot(fig, "grafico_14_ip90_cobertura_modelos_segmento_predtramo.png")


def plot_scatter_segment(holdout: pd.DataFrame, segment_label: str) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    max_val = min(80, np.nanpercentile(holdout["y_true"], 99.2))
    for ax, model in zip(axes, base.MODELS):
        sub = holdout[holdout["modelo"] == model]
        ax.scatter(sub["y_true"], sub["y_pred"], c=base.MODEL_COLORS[model], alpha=0.45, s=18, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], ls="--", color="#111827", lw=1.2)
        base.add_title(ax, f"{model}: real vs predicho")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("LOS real")
    axes[0].set_ylabel("LOS predicho")
    fig.suptitle(f"Real vs predicho - {segment_label}", x=0.04, y=1.04, ha="left", fontsize=20, fontweight="bold", color=base.NAVY)
    base.finish_plot(fig, "grafico_15_xgb_scatter_real_vs_pred.png")


def plot_dashboard_segment(global_ci: pd.DataFrame, tramo_ci: pd.DataFrame, segment_label: str, n_cases: int) -> None:
    fig, ax = base.plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(0.02, 0.94, f"XGBoost vs Regresion Lineal Base: {segment_label}", fontsize=26, fontweight="bold", color=base.NAVY)
    ax.text(0.02, 0.89, f"Holdout n={n_cases:,} · PLOS = LOS >= 14 · IC95 bootstrap", fontsize=13, color="#6B7280")

    cards = [
        ("MAE XGB", f"{_metric_from_ci(global_ci, 'XGB', 'mae'):.2f} dias", "#2563EB"),
        ("MAE LR", f"{_metric_from_ci(global_ci, 'LR', 'mae'):.2f} dias", "#EF4444"),
        ("F1 PLOS14 XGB", f"{_metric_from_ci(global_ci, 'XGB', 'f1_plos_14') * 100:.1f}%", "#14B8A6"),
        ("Recall PLOS14 XGB", f"{_metric_from_ci(global_ci, 'XGB', 'recall_plos_14') * 100:.1f}%", "#14B8A6"),
        ("MAE 14+ XGB", f"{_tramo_metric(tramo_ci, 'XGB', '14+ (PLOS)', 'mae'):.2f} dias", "#2563EB"),
        ("MAE 14+ LR", f"{_tramo_metric(tramo_ci, 'LR', '14+ (PLOS)', 'mae'):.2f} dias", "#EF4444"),
    ]
    positions = [(0.04, 0.64), (0.37, 0.64), (0.70, 0.64), (0.04, 0.38), (0.37, 0.38), (0.70, 0.38)]
    for (title, value, color), (x_pos, y_pos) in zip(cards, positions):
        box = base.FancyBboxPatch(
            (x_pos, y_pos),
            0.27,
            0.17,
            boxstyle="round,pad=0.015,rounding_size=0.025",
            ec=color,
            fc="white",
            lw=2,
        )
        ax.add_patch(box)
        ax.text(x_pos + 0.03, y_pos + 0.12, title, fontsize=14, fontweight="bold", color=base.NAVY)
        ax.text(x_pos + 0.03, y_pos + 0.055, value, fontsize=24, fontweight="bold", color=color)

    bullets = [
        f"En {segment_label.lower()}, XGB obtiene MAE { _metric_from_ci(global_ci, 'XGB', 'mae'):.2f} frente a { _metric_from_ci(global_ci, 'LR', 'mae'):.2f} en LR.",
        f"El RMSE de XGB es { _metric_from_ci(global_ci, 'XGB', 'rmse'):.2f}; el de LR es { _metric_from_ci(global_ci, 'LR', 'rmse'):.2f}.",
        f"Para PLOS14, XGB alcanza F1 { _metric_from_ci(global_ci, 'XGB', 'f1_plos_14') * 100:.1f}% frente a { _metric_from_ci(global_ci, 'LR', 'f1_plos_14') * 100:.1f}% en LR.",
        "Este panel usa solo pacientes del segmento, por lo que no mezcla urgencia con no urgencia.",
    ]
    y_pos = 0.22
    for bullet in bullets:
        ax.text(0.05, y_pos, "\u2022 " + bullet, fontsize=14, color=base.NAVY)
        y_pos -= 0.07

    base.finish_plot(fig, "grafico_16_dashboard_resumen_presentacion.png")


def generate_for_segment(segment_key: str, folder_name: str, segment_label: str, train_all: pd.DataFrame, holdout_all: pd.DataFrame) -> None:
    base.OUTPUT_DIR = OUTPUT_ROOT / folder_name
    base.CSV_DIR = CSV_ROOT / folder_name
    base.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base.CSV_DIR.mkdir(parents=True, exist_ok=True)

    train = train_all[train_all["segmento"] == segment_key].copy()
    holdout = holdout_all[holdout_all["segmento"] == segment_key].copy()
    holdout.to_csv(base.CSV_DIR / "predicciones_holdout_modelos_con_tramos_xgb_lr.csv", index=False)

    global_ci, segment_ci, tramo_ci, _segment_tramo_ci = base.build_bootstrap_tables(holdout)
    _interval_df, ip90_eval = base.build_ip90(train, holdout)

    title_suffix = f" - {segment_label}"
    base.plot_global_bar(global_ci, "mae", f"MAE global con IC95{title_suffix}", "Dias", "grafico_01_mae_global_ic95.png")
    base.plot_global_bar(global_ci, "rmse", f"RMSE global con IC95{title_suffix}", "Dias", "grafico_02_rmse_global_ic95.png")
    base.plot_global_bar(global_ci, "mae_asimetrico_alpha_2", f"MAE asimetrico global con IC95{title_suffix}", "Dias", "grafico_03_mae_asimetrico_global_ic95.png")
    plot_plos_global_segment(global_ci, segment_label)
    base.plot_by_tramo(tramo_ci, "mae", f"MAE por tramo real con IC95{title_suffix}", "Dias", "grafico_05_mae_por_tramo_ic95.png")
    base.plot_by_tramo(tramo_ci, "rmse", f"RMSE por tramo real con IC95{title_suffix}", "Dias", "grafico_06_rmse_por_tramo_ic95.png")
    base.plot_by_tramo(tramo_ci, "me", f"Sesgo por tramo real con IC95{title_suffix}", "Predicho - real", "grafico_07_sesgo_por_tramo_ic95.png")
    base.plot_by_tramo(tramo_ci, "pup", f"Subestimacion por tramo real con IC95{title_suffix}", "Porcentaje", "grafico_08_subestimacion_por_tramo_ic95.png", percent=True)
    base.plot_by_tramo(tramo_ci, "mae_asimetrico_alpha_2", f"MAE asimetrico por tramo real con IC95{title_suffix}", "Dias", "grafico_09_mae_asimetrico_por_tramo_ic95.png")
    plot_mae_selected_segment(segment_ci, segment_key, segment_label)
    plot_plos_selected_segment(segment_ci, segment_key, segment_label)
    base.plot_ip90_by_tramo(ip90_eval, "coverage", f"Cobertura IP90 por tramo real y metodo{title_suffix}", "Cobertura (%)", "grafico_12_xgb_cobertura_ip90_por_tramo_y_metodo.png", percent=True)
    base.plot_ip90_by_tramo(ip90_eval, "width_mean", f"Ancho IP90 por tramo real y metodo{title_suffix}", "Ancho medio (dias)", "grafico_13_xgb_ancho_ip90_por_tramo_y_metodo.png")
    plot_ip90_selected_segment(ip90_eval, segment_key, segment_label)
    plot_scatter_segment(holdout, segment_label)
    n_cases = int(holdout[holdout["modelo"] == "XGB"]["case_id"].nunique())
    plot_dashboard_segment(global_ci, tramo_ci, segment_label, n_cases)

    print(f"\nSegmento {segment_label}")
    print(f"Graficos guardados en: {base.OUTPUT_DIR}")
    print(f"CSVs auxiliares guardados en: {base.CSV_DIR}")
    for path in sorted(base.OUTPUT_DIR.glob("grafico_*.png")):
        print(f"  - {path.name}")


def main() -> None:
    base.setup_style()
    train_all = base.load_predictions("train")
    holdout_all = base.load_predictions("holdout")

    for segment_key, folder_name, segment_label in SEGMENTS:
        generate_for_segment(segment_key, folder_name, segment_label, train_all, holdout_all)


if __name__ == "__main__":
    main()
