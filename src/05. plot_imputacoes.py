"""Plota gráficos de comparação dos métodos de imputação (10min), com aparência
de artigo (fonte grande, colorido, nomes descritivos, 300 dpi).

Entrada: ``resultados/imputacao_metrics.csv``.

Saídas (em ``resultados/graficos/``):
  - comparacao_media.png      : ranking médio RMSE/MAE/R2 (1x3) — figura principal.
  - rmse_comparison.png       : RMSE por método, 4 painéis (um por percentual).
  - mae_comparison.png        : MAE por método, 4 painéis.
  - r2_comparison.png         : R2 por método, 4 painéis (com valores anotados).
  - evolucao_percentual.png   : métrica x percentual (linhas por método).
  - r2_heatmap.png            : heatmap R2 (método x percentual).
  - ranking_imputacao.csv     : tabela de ranking.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "resultados"
OUT = RESULTS / "graficos"
CSV = RESULTS / "imputacao_metrics.csv"

PCTS = [0.05, 0.10, 0.20, 0.25]

CORE = ["granufill", "knn_granufill", "knn", "moving_average",
        "moving_median", "cubic", "linear", "quadratic",
        "ffill", "ewma", "seasonal", "pchip"]

# Nomes descritivos para o artigo.
METHOD_LABELS = {
    "granufill": "Granufill",
    "knn_granufill": "KNN + Granufill",
    "knn": "KNN",
    "moving_average": "Moving average",
    "moving_median": "Moving median",
    "cubic": "Cubic spline",
    "linear": "Linear interp.",
    "quadratic": "Quadratic interp.",
    "ffill": "LOCF (carry-forward)",
    "ewma": "EWMA (exponential)",
    "seasonal": "Seasonal naive",
    "pchip": "PCHIP (monotonic)",
}

# Paleta colorida e distinguível (cores consistentes entre figuras).
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860",
           "#DA8BC3", "#CCB974", "#64B5CD", "#8C8C8C", "#FF9F43", "#00B19D"]
METHOD_COLORS = {m: PALETTE[i] for i, m in enumerate(CORE)}

# Fonte maior, adequada para artigo impresso.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})

METRIC_META = {
    "rmse": dict(ylabel="RMSE (bytes)", lower_is_better=True, clip=None,
                 annotate=False),
    "mae": dict(ylabel="MAE (bytes)", lower_is_better=True, clip=None,
                annotate=False),
    "r2": dict(ylabel="R²", lower_is_better=False, clip=(-0.2, 1.0),
               annotate=True),
}


def pct_label(p: float) -> str:
    return f"{int(round(p * 100))}%"


def label(m: str) -> str:
    return METHOD_LABELS.get(m, m)


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["pct"] = pd.to_numeric(df["pct"])
    df = df[df["method"].isin(CORE)].reset_index(drop=True)
    return df


def plot_bars(metric: str, ax, d: pd.DataFrame, annotate: bool = False):
    """Barra horizontal colorida, melhor método no topo."""
    meta = METRIC_META[metric]
    d = d.copy()
    if meta["clip"] is not None:
        d[metric] = d[metric].clip(*meta["clip"])
    d = d.sort_values(metric, ascending=meta["lower_is_better"])

    y = np.arange(len(d))[::-1]  # melhor no topo
    vals = d[metric].to_numpy()
    colors = [METHOD_COLORS[m] for m in d["method"]]

    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([label(m) for m in d["method"]])
    ax.set_xlabel(meta["ylabel"])
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    if metric == "r2":
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(1.0, color="black", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.set_xlim(-0.2, 1.0)
    elif vals.max() > 0:
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    if annotate:
        for yi, v in zip(y, vals):
            off = 0.015 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(v + off, yi, f"{v:.2f}", va="center", ha="left", fontsize=12)


def comparacao_media(df: pd.DataFrame):
    """Figura principal: ranking médio (média dos 4 percentuais) por método."""
    means = df.groupby("method")[["rmse", "mae", "r2"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        plot_bars(metric, ax, means, annotate=METRIC_META[metric]["annotate"])
        ax.set_title(f"({['a','b','c'][list(['rmse','mae','r2']).index(metric)]}) "
                     f"{METRIC_META[metric]['ylabel']}", loc="left", fontsize=16)
    fig.suptitle("Average performance of imputation methods (10 min)",
                 fontsize=18, y=1.02)
    fig.tight_layout()
    return fig


def full_grid(metric: str, df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(17, 12))
    for ax, pct in zip(axes.ravel(), PCTS):
        d = df[df["pct"] == pct]
        plot_bars(metric, ax, d, annotate=METRIC_META[metric]["annotate"])
        ax.set_title(pct_label(pct), loc="center", fontsize=16)
    fig.suptitle(f"{METRIC_META[metric]['ylabel']} by method and fraction of "
                 f"data removed", fontsize=18, y=1.01)
    fig.tight_layout()
    return fig


def evolucao_percentual(df: pd.DataFrame):
    """Linhas: como cada método evolui com o percentual de dados removidos."""
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, metric in zip(axes, ["rmse", "mae", "r2"]):
        for m in CORE:
            sub = df[df["method"] == m].sort_values("pct")
            ax.plot(sub["pct"], sub[metric], marker="o", markersize=5,
                    linewidth=2, color=METHOD_COLORS[m], label=label(m))
        ax.set_xticks(PCTS)
        ax.set_xticklabels([pct_label(p) for p in PCTS])
        ax.set_xlabel("Fraction of data removed")
        ax.set_ylabel(METRIC_META[metric]["ylabel"])
        ax.grid(True, linestyle="--", alpha=0.35)
        if metric == "r2":
            ax.axhline(0.0, color="black", linestyle="--", alpha=0.6)
        elif metric in ("rmse", "mae"):
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="center right", bbox_to_anchor=(1.02, 0.5),
               frameon=False, fontsize=12)
    fig.suptitle("Metric evolution with increasing missing data",
                 fontsize=18)
    fig.tight_layout(rect=[0, 0, 0.94, 1])
    return fig


def r2_heatmap(df: pd.DataFrame):
    pivot = df.pivot(index="method", columns="pct", values="r2")
    pivot["mean"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("mean", ascending=False).drop(columns="mean")
    pivot = pivot.rename(index=label, columns=pct_label)

    fig, ax = plt.subplots(figsize=(9, 9))
    data = pivot.to_numpy()
    im = ax.imshow(data, cmap="RdYlGn", norm=mcolors.Normalize(vmin=-0.2, vmax=1.0),
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=14)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=13)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, color="black")
    fig.colorbar(im, ax=ax, label="R²", fraction=0.046, pad=0.04)
    ax.set_title("R² by method and fraction of data removed", fontsize=16)
    ax.set_xlabel("Fraction of data removed", fontsize=15)
    fig.tight_layout()
    return fig


def ranking(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "method": df["method"].map(label),
        "mean_rmse": df.groupby("method")["rmse"].transform("mean"),
        "mean_mae": df.groupby("method")["mae"].transform("mean"),
        "mean_r2": df.groupby("method")["r2"].transform("mean"),
    }).drop_duplicates().sort_values("mean_r2", ascending=False).reset_index(drop=True)
    return out


def save(fig, name: str):
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(name)


def main():
    df = load()
    OUT.mkdir(parents=True, exist_ok=True)

    save(comparacao_media(df), "comparacao_media.png")
    save(full_grid("rmse", df), "rmse_comparison.png")
    save(full_grid("mae", df), "mae_comparison.png")
    save(full_grid("r2", df), "r2_comparison.png")
    save(evolucao_percentual(df), "evolucao_percentual.png")
    save(r2_heatmap(df), "r2_heatmap.png")

    rank = ranking(df)
    rank.to_csv(OUT / "ranking_imputacao.csv", index=False)
    print("ranking_imputacao.csv")
    print("\nRanking (mean R²):")
    print(rank[["method", "mean_r2"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
