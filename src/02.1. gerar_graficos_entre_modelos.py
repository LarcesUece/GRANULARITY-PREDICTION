"""Consolida metrics.jsonl e gera comparacoes entre tecnicas de imputacao.

Uso:
    python "src/02.1. gerar_graficos_entre_modelos.py"
    python "src/02.1. gerar_graficos_entre_modelos.py" --results-dir RESULTADOS
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(results_dir: Path) -> pd.DataFrame:
    """Le todos os metrics.jsonl abaixo de results_dir em um DataFrame.
    
    Estrutura esperada:
    results_dir/
    ├── granularidade_1/
    │   ├── imputacao_1/
    │   │   └── metrics.jsonl
    │   ├── imputacao_2/
    │   │   └── metrics.jsonl
    │   └── ...
    ├── granularidade_2/
    │   ├── imputacao_1/
    │   │   └── metrics.jsonl
    │   └── ...
    """
    records: list[dict] = []

    for metrics_path in sorted(results_dir.rglob("metrics.jsonl")):
        # Extrai granularidade (primeiro nível) e técnica de imputação (segundo nível)
        relative_path = metrics_path.relative_to(results_dir)
        path_parts = relative_path.parts
        
        # O metrics.jsonl deve estar em: granularidade/imputacao/metrics.jsonl
        if len(path_parts) < 3:
            warnings.warn(
                f"Estrutura inesperada para {metrics_path}: "
                "esperado granularidade/imputacao/metrics.jsonl"
            )
            continue
            
        granularity = path_parts[0]  # Primeiro nível: granularidade
        imputation = path_parts[1]   # Segundo nível: técnica de imputação
        
        with metrics_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    warnings.warn(
                        f"Linha ignorada em {metrics_path}:{line_number}: {error}"
                    )
                    continue

                if not isinstance(record, dict):
                    warnings.warn(
                        f"Registro ignorado em {metrics_path}:{line_number}: "
                        "o JSON nao e um objeto."
                    )
                    continue

                record["source_file"] = str(relative_path)
                record["run_dir"] = str(metrics_path.parent.relative_to(results_dir))
                record["granularity"] = granularity
                record["imputation"] = imputation
                records.append(record)

    if not records:
        return pd.DataFrame()

    dataframe = pd.json_normalize(records, sep=".")
    dataframe.insert(0, "run", range(1, len(dataframe) + 1))
    return dataframe


def metric_columns(dataframe: pd.DataFrame) -> list[str]:
    """Retorna as colunas numericas que representam metricas."""
    ignored = {"run", "batch_size", "best_epoch"}
    candidates = []

    for column in dataframe.columns:
        if column in ignored or column.startswith("spec."):
            continue
        if not any(column.startswith(f"{section}.") for section in ("val", "test", "prediction_sample")):
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            candidates.append(column)

    return candidates


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def plot_metrics(dataframe: pd.DataFrame, output_dir: Path) -> None:
    """Gera um grafico de barras para cada combinacao granularidade x modelo."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    if "model" not in dataframe:
        raise ValueError("Os metrics.jsonl nao possuem a coluna 'model'.")
    
    if "granularity" not in dataframe or "imputation" not in dataframe:
        raise ValueError("Os metrics.jsonl nao possuem as colunas 'granularity' ou 'imputation'.")

    metrics = metric_columns(dataframe)
    if not metrics:
        raise ValueError("Nenhuma metrica numerica foi encontrada nos metrics.jsonl.")

    # Mantem uma linha por execucao e facilita o uso posterior do agregado.
    dataframe.to_csv(output_dir / "metricas_agregadas.csv", index=False)

    # Obtém todas as combinações únicas de granularidade e modelo
    granularities = sorted(dataframe["granularity"].unique())
    models = sorted(dataframe["model"].unique())

    # Para cada métrica, gera gráficos para cada combinação granularidade x modelo
    for metric in metrics:
        plot_data = dataframe[["granularity", "model", "imputation", metric]].dropna(subset=[metric]).copy()
        
        if plot_data.empty:
            continue

        # Cria um grid de subplots: linhas = granularidades, colunas = modelos
        fig, axes = plt.subplots(
            len(granularities),
            len(models),
            figsize=(max(10, 6 * len(models)), max(6, 5 * len(granularities))),
            squeeze=False,
        )

        for i, granularity in enumerate(granularities):
            for j, model in enumerate(models):
                ax = axes[i][j]
                
                # Filtra dados para esta combinação específica
                panel = plot_data[
                    (plot_data["granularity"] == granularity) & 
                    (plot_data["model"] == model)
                ]
                
                if panel.empty:
                    ax.text(0.5, 0.5, "Sem dados", ha="center", va="center", transform=ax.transAxes)
                    ax.set_title(f"{model} - {granularity}")
                    ax.set_xlabel("Técnica de Imputação")
                    ax.set_ylabel(metric)
                    continue
                
                # Calcula média e desvio padrão para cada técnica de imputação
                summary = panel.groupby("imputation")[metric].agg(["mean", "std"]).reset_index()
                
                # Ordena por média para melhor visualização
                summary = summary.sort_values("mean")
                
                # Cria gráfico de barras com barras de erro
                bars = ax.bar(
                    summary["imputation"],
                    summary["mean"],
                    yerr=summary["std"].fillna(0),
                    capsize=5,
                    color="#76a5af",
                    edgecolor="black",
                    alpha=0.8
                )
                
                # Adiciona valores individuais como pontos
                sns.stripplot(
                    data=panel,
                    x="imputation",
                    y=metric,
                    ax=ax,
                    color="darkred",
                    alpha=0.5,
                    size=4,
                    jitter=0.2,
                    zorder=5
                )
                
                # Configurações do subplot
                ax.set_title(f"Modelo: {model}\nGranularidade: {granularity}", fontsize=10)
                ax.set_xlabel("Técnica de Imputação", fontsize=9)
                ax.set_ylabel(metric, fontsize=9)
                ax.tick_params(axis="x", rotation=45, labelsize=8)
                ax.tick_params(axis="y", labelsize=8)
                
                # Adiciona grade horizontal para facilitar leitura
                ax.grid(axis="y", alpha=0.3)
                ax.set_axisbelow(True)

        # Título geral e ajuste do layout
        fig.suptitle(f"Comparação entre Técnicas de Imputação\nMétrica: {metric}", fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])  # Ajusta para não sobrepor o título
        
        # Salva o gráfico
        safe_metric = safe_filename(metric)
        fig.savefig(
            output_dir / f"{safe_metric}_por_granularidade_modelo.png",
            dpi=200,
            bbox_inches="tight"
        )
        plt.close(fig)
        
        # Gera também gráficos individuais para cada combinação (opcional)
        for granularity in granularities:
            for model in models:
                panel = plot_data[
                    (plot_data["granularity"] == granularity) & 
                    (plot_data["model"] == model)
                ]
                
                if panel.empty:
                    continue
                
                fig_individual, ax_individual = plt.subplots(figsize=(10, 6))
                
                summary = panel.groupby("imputation")[metric].agg(["mean", "std"]).reset_index()
                summary = summary.sort_values("mean")
                
                ax_individual.bar(
                    summary["imputation"],
                    summary["mean"],
                    yerr=summary["std"].fillna(0),
                    capsize=5,
                    color="#76a5af",
                    edgecolor="black",
                    alpha=0.8
                )
                
                sns.stripplot(
                    data=panel,
                    x="imputation",
                    y=metric,
                    ax=ax_individual,
                    color="darkred",
                    alpha=0.5,
                    size=5,
                    jitter=0.2,
                    zorder=5
                )
                
                ax_individual.set_title(
                    f"Comparação entre Técnicas de Imputação\n"
                    f"Modelo: {model} | Granularidade: {granularity} | Métrica: {metric}"
                )
                ax_individual.set_xlabel("Técnica de Imputação")
                ax_individual.set_ylabel(metric)
                ax_individual.tick_params(axis="x", rotation=45)
                ax_individual.grid(axis="y", alpha=0.3)
                ax_individual.set_axisbelow(True)
                
                fig_individual.tight_layout()
                fig_individual.savefig(
                    output_dir / 
                    f"{safe_metric}_{safe_filename(granularity)}_{safe_filename(model)}.png",
                    dpi=200,
                    bbox_inches="tight"
                )
                plt.close(fig_individual)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "RESULTADOS")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "RESULTADOS" / "graficos_entre_imputacoes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Diretorio de resultados nao encontrado: {args.results_dir}")

    dataframe = load_metrics(args.results_dir)
    if dataframe.empty:
        print(f"Nenhum registro encontrado em {args.results_dir}")
        return

    plot_metrics(dataframe, args.output_dir)
    print(f"{len(dataframe)} execucoes consolidadas em {args.output_dir}")


if __name__ == "__main__":
    main()