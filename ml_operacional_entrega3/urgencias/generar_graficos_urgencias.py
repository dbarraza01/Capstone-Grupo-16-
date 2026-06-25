import os
import tempfile
from pathlib import Path


PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "capstone_plot_cache"
(PLOT_CACHE_DIR / "matplotlib").mkdir(parents=True, exist_ok=True)
(PLOT_CACHE_DIR / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))
os.environ.setdefault("FC_CACHEDIR", str(PLOT_CACHE_DIR / "fontconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOS_THRESHOLD = 14
GROUP_ORDER = ["No urgencia", "Urgencia"]
COLORS = {"No urgencia": "#4C95C8", "Urgencia": "#D84C4C"}
TRAMO_BINS = [-1, 0, 2, 6, 13, np.inf]
TRAMO_LABELS = ["0 dias", "1-2 dias", "3-6 dias", "7-13 dias", "14+ dias\n(PLOS)"]


def load_operational_dataset() -> pd.DataFrame:
    project_dir = Path(__file__).resolve().parents[1]
    split_dir = project_dir / "data_splits"
    frames = []

    for segment in ("programado", "urgente"):
        for split in ("train", "holdout"):
            path = split_dir / f"datos_{split}_{segment}.csv"
            frame = pd.read_csv(path, usecols=["case_id", "los_dias", "es_urgencia"])
            frame["segmento_split"] = segment
            frame["split"] = split
            frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df["grupo_admision"] = df["es_urgencia"].map({0: "No urgencia", 1: "Urgencia"})
    df["tramo_los"] = pd.cut(df["los_dias"], bins=TRAMO_BINS, labels=TRAMO_LABELS)
    df["plos_14"] = df["los_dias"] >= PLOS_THRESHOLD
    return df


def plot_tramos(df: pd.DataFrame, output_dir: Path) -> None:
    counts = (
        df.groupby(["tramo_los", "grupo_admision"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=GROUP_ORDER)
    )
    percentages = counts.div(counts.sum(axis=0), axis=1) * 100

    x = np.arange(len(TRAMO_LABELS))
    width = 0.36

    fig, ax = plt.subplots(figsize=(14, 7))
    for idx, group in enumerate(GROUP_ORDER):
        offset = (idx - 0.5) * width
        values = percentages[group].to_numpy()
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=f"{group} (n={counts[group].sum():,})",
            color=COLORS[group],
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.95,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.9,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax.set_title(
        "Distribucion por Tramos de LOS: Urgencia vs No Urgencia\n"
        "PLOS actualizado: LOS >= 14 dias",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("% del grupo", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(TRAMO_LABELS, fontsize=12)
    ax.set_ylim(0, max(65, percentages.to_numpy().max() + 8))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "05_tramos_los_urgencia_vs_normal.png", dpi=200)
    plt.close(fig)


def plot_tasa_plos(df: pd.DataFrame, output_dir: Path) -> None:
    stats = (
        df.groupby("grupo_admision", observed=False)
        .agg(n_plos=("plos_14", "sum"), total=("plos_14", "size"), tasa=("plos_14", "mean"))
        .reindex(GROUP_ORDER)
    )
    stats["tasa_pct"] = stats["tasa"] * 100
    riesgo_relativo = stats.loc["Urgencia", "tasa"] / stats.loc["No urgencia", "tasa"]

    x = np.arange(len(GROUP_ORDER))
    values = stats["tasa_pct"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(
        x,
        values,
        color=[COLORS[group] for group in GROUP_ORDER],
        edgecolor="#222222",
        linewidth=1.0,
        alpha=0.95,
        width=0.55,
    )

    for bar, group, value in zip(bars, GROUP_ORDER, values):
        row = stats.loc[group]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.2f}%\n({int(row['n_plos']):,}/{int(row['total']):,})",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.annotate(
        f"Riesgo relativo\n{riesgo_relativo:.2f}x",
        xy=(1, values[1]),
        xytext=(0.35, max(values) * 0.58),
        arrowprops={"arrowstyle": "->", "color": "#990000", "lw": 2},
        color="#990000",
        fontsize=14,
        fontweight="bold",
        ha="center",
    )

    ax.set_title(
        "Riesgo de Estancia Prolongada (PLOS) por Grupo de Admision\n"
        "PLOS actualizado: LOS >= 14 dias",
        fontsize=18,
        fontweight="bold",
        pad=18,
    )
    ax.set_ylabel("Tasa de PLOS (LOS >= 14 dias), %", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(GROUP_ORDER, fontsize=12)
    ax.set_ylim(0, max(values) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "06_tasa_plos_urgencia_vs_normal.png", dpi=200)
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    df = load_operational_dataset()
    plot_tramos(df, output_dir)
    plot_tasa_plos(df, output_dir)

    stats = (
        df.groupby("grupo_admision", observed=False)
        .agg(n_plos=("plos_14", "sum"), total=("plos_14", "size"), tasa=("plos_14", "mean"))
        .reindex(GROUP_ORDER)
    )
    riesgo_relativo = stats.loc["Urgencia", "tasa"] / stats.loc["No urgencia", "tasa"]
    print("Graficos actualizados con PLOS = LOS >= 14 dias")
    for group, row in stats.iterrows():
        print(f"{group}: {int(row['n_plos'])}/{int(row['total'])} = {row['tasa'] * 100:.2f}%")
    print(f"Riesgo relativo urgencia/no urgencia: {riesgo_relativo:.2f}x")


if __name__ == "__main__":
    main()
