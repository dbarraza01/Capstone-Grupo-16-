from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Callable

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
from matplotlib.patches import FancyBboxPatch


PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"
OUTPUT_DIR = PROJECT_DIR / "graficos_png"
CSV_DIR = PROJECT_DIR / "graficos_csv"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_BOOT = 2000
PLOS_THRESHOLD = 14
MODELS = ["LR", "XGB"]
MODEL_COLORS = {"LR": "#EF4444", "XGB": "#2563EB"}
METHOD_COLORS = {
    "global": "#6B7280",
    "segmento": "#14B8A6",
    "pred_tramo": "#6366F1",
    "segmento_pred_tramo": "#2563EB",
}
TRAMO_BINS = [-1, 2, 6, 13, np.inf]
TRAMO_LABELS = ["0-2", "3-6", "7-13", "14+ (PLOS)"]
SEGMENT_LABELS = {"programado": "Programado", "urgente": "Urgente"}
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
        "axes.titlesize": 20,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    })


def load_predictions(split: str) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        key = model.lower()
        path = REPORTS_DIR / f"predicciones_{split}_{key}.csv"
        df = pd.read_csv(path)
        df["modelo"] = model
        df["y_true"] = df["los_dias_reales"].astype(float)
        df["y_pred"] = df["los_dias_predichos"].astype(float)
        df["error_pred_menos_real"] = df["y_pred"] - df["y_true"]
        df["resid_real_menos_pred"] = df["y_true"] - df["y_pred"]
        df["abs_error"] = np.abs(df["error_pred_menos_real"])
        df["tramo_real"] = pd.cut(df["y_true"], TRAMO_BINS, labels=TRAMO_LABELS).astype(str)
        df["tramo_pred"] = pd.cut(df["y_pred"], TRAMO_BINS, labels=TRAMO_LABELS).astype(str)
        df["plos_real_14"] = (df["y_true"] >= PLOS_THRESHOLD).astype(int)
        df["plos_pred_14"] = (df["y_pred"] >= PLOS_THRESHOLD).astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def confusion_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> tuple[int, int, int, int]:
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_pred_bin = np.asarray(y_pred_bin).astype(int)
    tp = int(((y_true_bin == 1) & (y_pred_bin == 1)).sum())
    tn = int(((y_true_bin == 0) & (y_pred_bin == 0)).sum())
    fp = int(((y_true_bin == 0) & (y_pred_bin == 1)).sum())
    fn = int(((y_true_bin == 1) & (y_pred_bin == 0)).sum())
    return tn, fp, fn, tp


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def metric_value(df: pd.DataFrame, metric: str) -> float:
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)
    err = y_pred - y_true
    abs_err = np.abs(err)
    if metric == "mae":
        return float(abs_err.mean())
    if metric == "rmse":
        return float(np.sqrt(np.mean(err ** 2)))
    if metric == "medae":
        return float(np.median(abs_err))
    if metric == "me":
        return float(err.mean())
    if metric == "pup":
        return float((y_pred < y_true).mean())
    if metric == "mae_asimetrico_alpha_2":
        weights = np.where(y_pred < y_true, 2.0, 1.0)
        return float(np.mean(weights * abs_err))
    if metric in {"precision_plos_14", "recall_plos_14", "f1_plos_14"}:
        tn, fp, fn, tp = confusion_counts(df["plos_real_14"], df["plos_pred_14"])
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        if metric == "precision_plos_14":
            return precision
        if metric == "recall_plos_14":
            return recall
        return f1
    raise ValueError(metric)


def bootstrap_ci(df: pd.DataFrame, metric: str, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple[float, float]:
    if len(df) == 0:
        return np.nan, np.nan
    values = []
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        values.append(metric_value(df.iloc[idx], metric))
    return tuple(np.percentile(values, [2.5, 97.5]))


def build_bootstrap_tables(holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    metrics = [
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

    def rows_for_group(group_df: pd.DataFrame, meta: dict) -> list[dict]:
        rows = []
        for metric in metrics:
            low, high = bootstrap_ci(group_df, metric, rng)
            rows.append({
                **meta,
                "metrica": metric,
                "valor": metric_value(group_df, metric),
                "ic95_inf": low,
                "ic95_sup": high,
                "n_boot": N_BOOT,
                "n": len(group_df),
            })
        return rows

    global_rows = []
    segment_rows = []
    tramo_rows = []
    segment_tramo_rows = []
    for model, model_df in holdout.groupby("modelo", sort=False):
        global_rows.extend(rows_for_group(model_df, {"modelo": model, "alcance": "global", "segmento": "todos", "tramo_real": "todos"}))
        for segment, seg_df in model_df.groupby("segmento", sort=False):
            segment_rows.extend(rows_for_group(seg_df, {"modelo": model, "alcance": "segmento", "segmento": segment, "tramo_real": "todos"}))
        for tramo in TRAMO_LABELS:
            tramo_df = model_df[model_df["tramo_real"] == tramo]
            tramo_rows.extend(rows_for_group(tramo_df, {"modelo": model, "alcance": "tramo_real", "segmento": "todos", "tramo_real": tramo}))
        for (segment, tramo), group_df in model_df.groupby(["segmento", "tramo_real"], sort=False):
            segment_tramo_rows.extend(rows_for_group(group_df, {"modelo": model, "alcance": "segmento_tramo_real", "segmento": segment, "tramo_real": tramo}))

    global_df = pd.DataFrame(global_rows)
    segment_df = pd.DataFrame(segment_rows)
    tramo_df = pd.DataFrame(tramo_rows)
    segment_tramo_df = pd.DataFrame(segment_tramo_rows)
    global_df.to_csv(CSV_DIR / "ic95_bootstrap_metricas_globales_xgb_lr.csv", index=False)
    segment_df.to_csv(CSV_DIR / "ic95_bootstrap_metricas_por_segmento_xgb_lr.csv", index=False)
    tramo_df.to_csv(CSV_DIR / "ic95_bootstrap_metricas_por_tramo_real_xgb_lr.csv", index=False)
    segment_tramo_df.to_csv(CSV_DIR / "ic95_bootstrap_metricas_por_segmento_y_tramo_real_xgb_lr.csv", index=False)
    return global_df, segment_df, tramo_df, segment_tramo_df


def pivot_metric(ci_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    return ci_df[ci_df["metrica"] == metric].copy()


def annotate_bars(ax: plt.Axes, bars, fmt: Callable[[float], str], pad: float = 0.02) -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * pad
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        va = "bottom"
        y = height + offset
        if height < 0:
            va = "top"
            y = height - offset
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt(height),
            ha="center",
            va=va,
            fontsize=10,
            fontweight="bold",
            color=NAVY,
        )


def finish_plot(fig: plt.Figure, path: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def add_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=20, color=NAVY, pad=18)
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", color="#6B7280", fontsize=12)


def plot_global_bar(global_ci: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    df = pivot_metric(global_ci, metric).set_index("modelo").loc[MODELS].reset_index()
    values = df["valor"].to_numpy() * (100 if percent else 1)
    lows = df["ic95_inf"].to_numpy() * (100 if percent else 1)
    highs = df["ic95_sup"].to_numpy() * (100 if percent else 1)
    errors = np.vstack([values - lows, highs - values])

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(df["modelo"], values, color=[MODEL_COLORS[m] for m in df["modelo"]], width=0.62)
    ax.errorbar(df["modelo"], values, yerr=errors, fmt="none", ecolor="black", elinewidth=1.8, capsize=6)
    add_title(ax, title, "Bootstrap sobre holdout actual · XGBoost vs Regresion Lineal Base")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ymax = max(highs) * 1.18 if max(highs) > 0 else 1
    ax.set_ylim(0, ymax)
    annotate_bars(ax, bars, lambda x: f"{x:.1f}%" if percent else f"{x:.2f}")
    finish_plot(fig, filename)


def plot_plos_global(global_ci: pd.DataFrame) -> None:
    metrics = ["precision_plos_14", "recall_plos_14", "f1_plos_14"]
    labels = ["Precision", "Recall", "F1"]
    fig, ax = plt.subplots(figsize=(13, 7))
    x = np.arange(len(labels))
    width = 0.34
    for i, model in enumerate(MODELS):
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
        bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
        ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=5)
        annotate_bars(ax, bars, lambda y: f"{y:.1f}%", pad=0.015)
    add_title(ax, "Deteccion PLOS14 con IC95", "PLOS definido como LOS real/predicho >= 14 dias · holdout")
    ax.set_ylabel("Porcentaje")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish_plot(fig, "grafico_04_plos14_precision_recall_f1_ic95.png")


def plot_by_tramo(tramo_ci: pd.DataFrame, metric: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(TRAMO_LABELS))
    width = 0.34
    for i, model in enumerate(MODELS):
        rows = []
        for tramo in TRAMO_LABELS:
            rows.append(tramo_ci[(tramo_ci["modelo"] == model) & (tramo_ci["tramo_real"] == tramo) & (tramo_ci["metrica"] == metric)].iloc[0])
        df = pd.DataFrame(rows)
        vals = df["valor"].to_numpy() * (100 if percent else 1)
        lows = df["ic95_inf"].to_numpy() * (100 if percent else 1)
        highs = df["ic95_sup"].to_numpy() * (100 if percent else 1)
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
        ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=4)
        annotate_bars(ax, bars, lambda y: f"{y:.0f}%" if percent else f"{y:.2f}", pad=0.015)
    add_title(ax, title, "Tramos definidos por LOS real · IC95 bootstrap")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(TRAMO_LABELS, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish_plot(fig, filename)


def plot_by_segment(segment_ci: pd.DataFrame) -> None:
    metric = "mae"
    fig, ax = plt.subplots(figsize=(12, 7))
    segments = ["programado", "urgente"]
    x = np.arange(len(segments))
    width = 0.34
    for i, model in enumerate(MODELS):
        rows = []
        for segment in segments:
            rows.append(segment_ci[(segment_ci["modelo"] == model) & (segment_ci["segmento"] == segment) & (segment_ci["metrica"] == metric)].iloc[0])
        df = pd.DataFrame(rows)
        vals = df["valor"].to_numpy()
        lows = df["ic95_inf"].to_numpy()
        highs = df["ic95_sup"].to_numpy()
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
        ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=4)
        annotate_bars(ax, bars, lambda y: f"{y:.2f}", pad=0.015)
    add_title(ax, "MAE por segmento con IC95", "Holdout actual · modelos segmentados")
    ax.set_ylabel("Dias")
    ax.set_xticks(x)
    ax.set_xticklabels([SEGMENT_LABELS[s] for s in segments], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish_plot(fig, "grafico_10_mae_por_segmento_ic95.png")


def plot_plos_by_segment(segment_ci: pd.DataFrame) -> None:
    metrics = ["precision_plos_14", "recall_plos_14", "f1_plos_14"]
    segments = ["programado", "urgente"]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, segment in zip(axes, segments):
        x = np.arange(len(metrics))
        width = 0.34
        for i, model in enumerate(MODELS):
            rows = [
                segment_ci[(segment_ci["modelo"] == model) & (segment_ci["segmento"] == segment) & (segment_ci["metrica"] == metric)].iloc[0]
                for metric in metrics
            ]
            df = pd.DataFrame(rows)
            vals = df["valor"].to_numpy() * 100
            lows = df["ic95_inf"].to_numpy() * 100
            highs = df["ic95_sup"].to_numpy() * 100
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
            ax.errorbar(x + offset, vals, yerr=np.vstack([vals - lows, highs - vals]), fmt="none", ecolor="black", capsize=4)
            annotate_bars(ax, bars, lambda y: f"{y:.0f}%", pad=0.015)
        add_title(ax, f"PLOS14 - {SEGMENT_LABELS[segment]}")
        ax.set_xticks(x)
        ax.set_xticklabels(["Precision", "Recall", "F1"], fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Porcentaje")
    axes[0].set_ylim(0, 105)
    axes[1].legend(frameon=False)
    finish_plot(fig, "grafico_11_plos14_por_segmento_ic95.png")


def quantiles_for(train: pd.DataFrame, model: str, method: str, row: pd.Series) -> tuple[float, float]:
    model_train = train[train["modelo"] == model]
    group = model_train
    if method == "segmento":
        group = model_train[model_train["segmento"] == row["segmento"]]
    elif method == "pred_tramo":
        group = model_train[model_train["tramo_pred"] == row["tramo_pred"]]
    elif method == "segmento_pred_tramo":
        group = model_train[(model_train["segmento"] == row["segmento"]) & (model_train["tramo_pred"] == row["tramo_pred"])]
        if len(group) < 30:
            group = model_train[model_train["segmento"] == row["segmento"]]
    if len(group) < 30:
        group = model_train
    return tuple(np.quantile(group["resid_real_menos_pred"], [0.05, 0.95]))


def build_ip90(train: pd.DataFrame, holdout: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    methods = ["global", "segmento", "pred_tramo", "segmento_pred_tramo"]
    out = []
    for _, row in holdout.iterrows():
        base = row.to_dict()
        for method in methods:
            q_low, q_high = quantiles_for(train, row["modelo"], method, row)
            low = max(0.0, row["y_pred"] + q_low)
            high = max(0.0, row["y_pred"] + q_high)
            out.append({
                **base,
                "metodo_ip90": method,
                "pi90_lower": low,
                "pi90_upper": high,
                "pi90_width": high - low,
                "pi90_covered": float(low <= row["y_true"] <= high),
            })
    interval_df = pd.DataFrame(out)

    eval_rows = []
    for keys, group in interval_df.groupby(["modelo", "metodo_ip90"], sort=False):
        model, method = keys
        eval_rows.append({
            "modelo": model,
            "metodo_ip90": method,
            "alcance": "global",
            "segmento": "todos",
            "tramo_real": "todos",
            "n": len(group),
            "coverage": group["pi90_covered"].mean(),
            "width_mean": group["pi90_width"].mean(),
        })
    for keys, group in interval_df.groupby(["modelo", "metodo_ip90", "tramo_real"], sort=False):
        model, method, tramo = keys
        eval_rows.append({
            "modelo": model,
            "metodo_ip90": method,
            "alcance": "tramo_real",
            "segmento": "todos",
            "tramo_real": tramo,
            "n": len(group),
            "coverage": group["pi90_covered"].mean(),
            "width_mean": group["pi90_width"].mean(),
        })
    for keys, group in interval_df.groupby(["modelo", "metodo_ip90", "segmento"], sort=False):
        model, method, segment = keys
        eval_rows.append({
            "modelo": model,
            "metodo_ip90": method,
            "alcance": "segmento",
            "segmento": segment,
            "tramo_real": "todos",
            "n": len(group),
            "coverage": group["pi90_covered"].mean(),
            "width_mean": group["pi90_width"].mean(),
        })
    eval_df = pd.DataFrame(eval_rows)
    interval_df.to_csv(CSV_DIR / "predicciones_holdout_xgb_lr_con_ip90_empirico_train.csv", index=False)
    eval_df.to_csv(CSV_DIR / "evaluacion_ip90_xgb_lr.csv", index=False)
    return interval_df, eval_df


def plot_ip90_by_tramo(eval_df: pd.DataFrame, value_col: str, title: str, ylabel: str, filename: str, percent: bool = False) -> None:
    methods = ["global", "segmento", "pred_tramo", "segmento_pred_tramo"]
    method_labels = ["Global", "Segmento", "Tramo pred.", "Segmento+tramo"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, model in zip(axes, MODELS):
        sub = eval_df[(eval_df["modelo"] == model) & (eval_df["alcance"] == "tramo_real")]
        x = np.arange(len(TRAMO_LABELS))
        width = 0.18
        for i, method in enumerate(methods):
            vals = []
            for tramo in TRAMO_LABELS:
                row = sub[(sub["tramo_real"] == tramo) & (sub["metodo_ip90"] == method)].iloc[0]
                vals.append(row[value_col] * (100 if percent else 1))
            offset = (i - 1.5) * width
            bars = ax.bar(x + offset, vals, width=width, color=METHOD_COLORS[method], label=method_labels[i])
            annotate_bars(ax, bars, lambda y: f"{y:.0f}%" if percent else f"{y:.1f}", pad=0.012)
        add_title(ax, model)
        if percent:
            ax.axhline(90, ls="--", color="#EF4444", lw=1.4)
        ax.set_xticks(x)
        ax.set_xticklabels(TRAMO_LABELS, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylabel)
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle(title, x=0.04, y=1.04, ha="left", fontsize=20, fontweight="bold", color=NAVY)
    finish_plot(fig, filename)


def plot_ip90_segment_model(eval_df: pd.DataFrame) -> None:
    method = "segmento_pred_tramo"
    sub = eval_df[(eval_df["alcance"] == "segmento") & (eval_df["metodo_ip90"] == method)]
    fig, ax = plt.subplots(figsize=(12, 7))
    segments = ["programado", "urgente"]
    x = np.arange(len(segments))
    width = 0.34
    for i, model in enumerate(MODELS):
        vals = [
            sub[(sub["modelo"] == model) & (sub["segmento"] == segment)]["coverage"].iloc[0] * 100
            for segment in segments
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width, color=MODEL_COLORS[model], label=model)
        annotate_bars(ax, bars, lambda y: f"{y:.0f}%", pad=0.015)
    add_title(ax, "Cobertura IP90 por segmento", "Metodo: segmento + tramo predicho · calibrado con train")
    ax.axhline(90, ls="--", color="#EF4444", lw=1.4, label="Nominal 90%")
    ax.set_ylabel("Cobertura")
    ax.set_ylim(0, 110)
    ax.set_xticks(x)
    ax.set_xticklabels([SEGMENT_LABELS[s] for s in segments], fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    finish_plot(fig, "grafico_14_ip90_cobertura_modelos_segmento_predtramo.png")


def plot_scatter(holdout: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=True, sharey=True)
    max_val = min(80, np.nanpercentile(holdout["y_true"], 99.2))
    for ax, model in zip(axes, MODELS):
        sub = holdout[holdout["modelo"] == model]
        colors = sub["segmento"].map({"programado": "#14B8A6", "urgente": "#EF4444"})
        ax.scatter(sub["y_true"], sub["y_pred"], c=colors, alpha=0.45, s=18, edgecolors="none")
        ax.plot([0, max_val], [0, max_val], ls="--", color="#111827", lw=1.2)
        add_title(ax, f"{model}: real vs predicho")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("LOS real")
    axes[0].set_ylabel("LOS predicho")
    finish_plot(fig, "grafico_15_xgb_scatter_real_vs_pred.png")


def metric_from_ci(global_ci: pd.DataFrame, model: str, metric: str) -> float:
    return float(global_ci[(global_ci["modelo"] == model) & (global_ci["metrica"] == metric)]["valor"].iloc[0])


def tramo_metric(tramo_ci: pd.DataFrame, model: str, tramo: str, metric: str) -> float:
    return float(tramo_ci[(tramo_ci["modelo"] == model) & (tramo_ci["tramo_real"] == tramo) & (tramo_ci["metrica"] == metric)]["valor"].iloc[0])


def plot_dashboard(global_ci: pd.DataFrame, tramo_ci: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    ax.text(0.02, 0.94, "XGBoost vs Regresion Lineal Base: resumen ejecutivo", fontsize=26, fontweight="bold", color=NAVY)
    ax.text(0.02, 0.89, "Holdout n=2.391 · PLOS = LOS >= 14 · IC95 bootstrap", fontsize=13, color="#6B7280")

    cards = [
        ("MAE XGB", f"{metric_from_ci(global_ci, 'XGB', 'mae'):.2f} dias", "#2563EB"),
        ("MAE LR", f"{metric_from_ci(global_ci, 'LR', 'mae'):.2f} dias", "#EF4444"),
        ("F1 PLOS14 XGB", f"{metric_from_ci(global_ci, 'XGB', 'f1_plos_14') * 100:.1f}%", "#14B8A6"),
        ("Recall PLOS14 XGB", f"{metric_from_ci(global_ci, 'XGB', 'recall_plos_14') * 100:.1f}%", "#14B8A6"),
        ("MAE 14+ XGB", f"{tramo_metric(tramo_ci, 'XGB', '14+ (PLOS)', 'mae'):.2f} dias", "#2563EB"),
        ("MAE 14+ LR", f"{tramo_metric(tramo_ci, 'LR', '14+ (PLOS)', 'mae'):.2f} dias", "#EF4444"),
    ]
    positions = [(0.04, 0.64), (0.37, 0.64), (0.70, 0.64), (0.04, 0.38), (0.37, 0.38), (0.70, 0.38)]
    for (title, value, color), (x, y) in zip(cards, positions):
        box = FancyBboxPatch((x, y), 0.27, 0.17, boxstyle="round,pad=0.015,rounding_size=0.025", ec=color, fc="white", lw=2)
        ax.add_patch(box)
        ax.text(x + 0.03, y + 0.12, title, fontsize=14, fontweight="bold", color=NAVY)
        ax.text(x + 0.03, y + 0.055, value, fontsize=24, fontweight="bold", color=color)

    bullets = [
        f"XGB reduce el MAE global frente a LR: {metric_from_ci(global_ci, 'XGB', 'mae'):.2f} vs {metric_from_ci(global_ci, 'LR', 'mae'):.2f} dias.",
        f"LR basica muestra fuerte inestabilidad en RMSE: {metric_from_ci(global_ci, 'LR', 'rmse'):.2f} vs {metric_from_ci(global_ci, 'XGB', 'rmse'):.2f} en XGB.",
        f"En PLOS14, XGB logra mayor F1 ({metric_from_ci(global_ci, 'XGB', 'f1_plos_14') * 100:.1f}%) que LR ({metric_from_ci(global_ci, 'LR', 'f1_plos_14') * 100:.1f}%).",
        "La brecha mas critica esta en 14+ dias: LR sobreestima de forma extrema en varios casos, elevando MAE y RMSE.",
    ]
    y = 0.22
    for bullet in bullets:
        ax.text(0.05, y, "\u2022 " + bullet, fontsize=14, color=NAVY)
        y -= 0.07
    finish_plot(fig, "grafico_16_dashboard_resumen_presentacion.png")


def main() -> None:
    setup_style()
    train = load_predictions("train")
    holdout = load_predictions("holdout")
    holdout.to_csv(CSV_DIR / "predicciones_holdout_modelos_con_tramos_xgb_lr.csv", index=False)

    global_ci, segment_ci, tramo_ci, segment_tramo_ci = build_bootstrap_tables(holdout)
    interval_df, ip90_eval = build_ip90(train, holdout)

    plot_global_bar(global_ci, "mae", "MAE global con IC95", "Dias", "grafico_01_mae_global_ic95.png")
    plot_global_bar(global_ci, "rmse", "RMSE global con IC95", "Dias", "grafico_02_rmse_global_ic95.png")
    plot_global_bar(global_ci, "mae_asimetrico_alpha_2", "MAE asimetrico global con IC95", "Dias", "grafico_03_mae_asimetrico_global_ic95.png")
    plot_plos_global(global_ci)
    plot_by_tramo(tramo_ci, "mae", "MAE por tramo real con IC95", "Dias", "grafico_05_mae_por_tramo_ic95.png")
    plot_by_tramo(tramo_ci, "rmse", "RMSE por tramo real con IC95", "Dias", "grafico_06_rmse_por_tramo_ic95.png")
    plot_by_tramo(tramo_ci, "me", "Sesgo por tramo real con IC95", "Predicho - real", "grafico_07_sesgo_por_tramo_ic95.png")
    plot_by_tramo(tramo_ci, "pup", "Subestimacion por tramo real con IC95", "Porcentaje", "grafico_08_subestimacion_por_tramo_ic95.png", percent=True)
    plot_by_tramo(tramo_ci, "mae_asimetrico_alpha_2", "MAE asimetrico por tramo real con IC95", "Dias", "grafico_09_mae_asimetrico_por_tramo_ic95.png")
    plot_by_segment(segment_ci)
    plot_plos_by_segment(segment_ci)
    plot_ip90_by_tramo(ip90_eval, "coverage", "Cobertura IP90 por tramo real y metodo", "Cobertura (%)", "grafico_12_xgb_cobertura_ip90_por_tramo_y_metodo.png", percent=True)
    plot_ip90_by_tramo(ip90_eval, "width_mean", "Ancho IP90 por tramo real y metodo", "Ancho medio (dias)", "grafico_13_xgb_ancho_ip90_por_tramo_y_metodo.png")
    plot_ip90_segment_model(ip90_eval)
    plot_scatter(holdout)
    plot_dashboard(global_ci, tramo_ci)

    print(f"Graficos guardados en: {OUTPUT_DIR}")
    print(f"CSVs auxiliares guardados en: {CSV_DIR}")
    print("Archivos PNG generados:")
    for path in sorted(OUTPUT_DIR.glob("grafico_*.png")):
        print(f"  - {path.name}")


if __name__ == "__main__":
    main()
