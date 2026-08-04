import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

def load_metrics(metrics_path):
    data = []
    with open(metrics_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    df = pd.json_normalize(data)
    return df

def plot_metrics(df, output_dir):
    """
    Gera gráficos de barras comparando as métricas de teste entre os modelos.
    """
    metrics = ['prediction_sample.RMSE', 'prediction_sample.MAE', 'prediction_sample.SMAPE', 'prediction_sample.NRMSE']
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        # Filtra os melhores resultados (menor RMSE) se houver múltiplas execuções do mesmo modelo
        # ou apenas plota todos divididos por batch_size
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Métricas (Prediction Sample) - Dataset: {dataset.capitalize()}', fontsize=16)
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            sns.barplot(data=dataset_df, x='model', y=metric, hue='batch_size', ax=ax, palette='viridis', errorbar=None)
            ax.set_title(metric.replace('prediction_sample.', ''))
            ax.set_ylabel('Valor')
            ax.set_xlabel('Modelo')
            
        plt.tight_layout()
        plt.savefig(output_dir / f'comparacao_metricas_{dataset}.png', dpi=300)
        plt.close()

def plot_predictions(df, predictions_dir, output_dir):
    """
    Plota as séries temporais reais vs previstas para uma amostra.
    """
    datasets = df['dataset'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        
        # Mantém apenas um registro único por modelo e batch_size para evitar legendas duplicadas
        unique_runs = dataset_df.drop_duplicates(subset=['model', 'batch_size']).sort_values(['model', 'batch_size'])
            
        fig, ax = plt.subplots(figsize=(45, 6))
        
        plotted_true = False
        
        for _, row in unique_runs.iterrows():
            model = row['model']
            batch_size = row['batch_size']
            
            # Reconstrói o caminho pegando apenas o nome do arquivo, 
            # útil caso o caminho absoluto mude.
            file_name = Path(row['prediction_path']).name
            pred_path = predictions_dir / file_name
            
            if not pred_path.exists():
                print(f"Arquivo de predição não encontrado: {pred_path}")
                continue
                
            data = np.load(pred_path)
            
            # Pegar um único registro (ex: amostra 100) para analisar a degradação no horizonte
            sample_idx = min(100, len(data['y_true']) - 1)
            y_true = data['y_true'][sample_idx]
            y_pred = data['y_pred'][sample_idx]
            
            x_vals = np.arange(1, len(y_true) + 1)
            
            if not plotted_true:
                ax.plot(x_vals, y_true, label='Real (y_true)', color='black', linewidth=2, linestyle='--', marker='o')
                plotted_true = True
                
            # Diferencia batch sizes com estilo de linha e marcadores diferentes
            ls = '-' if batch_size == 128 else '-.'
            mk = '.' if batch_size == 128 else '.'
            
            ax.plot(x_vals, y_pred, label=f'{model} (Batch {batch_size})', alpha=0.8, linestyle=ls, marker=mk)
            
        ax.set_title(f'Predição vs Real (Amostra Única no Horizonte) - Dataset: {dataset.capitalize()}')
        ax.set_xlabel('Passos de Tempo (Horizonte)')
        ax.set_ylabel('Valor')
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f'predicoes_amostra_{dataset}.png', dpi=300)
        plt.close()

def plot_degradation(df, predictions_dir, output_dir):
    """
    Gera gráficos de degradação (RMSE ao longo do horizonte de predição) para cada modelo,
    comparando os diferentes batch sizes na legenda.
    """
    datasets = df['dataset'].unique()
    models = df['model'].unique()
    
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        for model in models:
            model_df = dataset_df[dataset_df['model'] == model]
            if model_df.empty:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            plotted_anything = False
            
            # Remove duplicatas de batch_size (mantendo a primeira) e ordena para a legenda
            model_df = model_df.drop_duplicates(subset=['batch_size']).sort_values('batch_size')
            
            for _, row in model_df.iterrows():
                batch_size = row['batch_size']
                file_name = Path(row['prediction_path']).name
                pred_path = predictions_dir / file_name
                
                if not pred_path.exists():
                    continue
                    
                data = np.load(pred_path)
                y_true = data['y_true']
                y_pred = data['y_pred']
                
                # Verifica se a dimensão é 2D (n_amostras, horizonte)
                if y_true.ndim >= 2:
                    # Calcula o RMSE para cada passo no horizonte
                    rmse_per_step = np.sqrt(np.mean((y_true - y_pred)**2, axis=0))
                    steps = np.arange(1, len(rmse_per_step) + 1)
                    
                    ax.plot(steps, rmse_per_step, label=f'Batch {batch_size}', linewidth=2, alpha=0.8)
                    plotted_anything = True
                
            if plotted_anything:
                ax.set_title(f'Degradação do RMSE no Horizonte\nDataset: {dataset.capitalize()} | Modelo: {model}')
                ax.set_xlabel('Passos Futuros (Horizonte)')
                ax.set_ylabel('RMSE (Test)')
                ax.legend(title='Batch Size')
                ax.grid(True, linestyle='--', alpha=0.6)
                
                plt.tight_layout()
                plt.savefig(output_dir / f'degradacao_{dataset}_{model}.png', dpi=300)
            plt.close()

def plot_elapsed_time(elapsed_time_path, output_dir):
    """
    Plota um gráfico comparando o tempo de execução (elapsed_time) de cada método para cada dataset.
    """
    if not elapsed_time_path.exists():
        print(f"Arquivo de tempo de execução não encontrado: {elapsed_time_path}")
        return
        
    with open(elapsed_time_path, 'r') as f:
        data = json.load(f)
        
    # data is like: {"granufill": {"hour": 0.04, "10min": 0.49}, ...}
    records = []
    for method, datasets in data.items():
        for dataset, time in datasets.items():
            records.append({'Método': method, 'Dataset': dataset, 'Tempo (s)': time})
            
    df_time = pd.DataFrame(records)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_time, x='Método', y='Tempo (s)', hue='Dataset', ax=ax, palette='Set2')
    ax.set_title('Comparação de Tempo de Execução por Método', fontsize=14)
    ax.set_ylabel('Tempo (segundos)')
    ax.set_xlabel('Método / Modelo')
    
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparacao_tempo_execucao.png', dpi=300)
    plt.close()

def main():
    # Estrutura de pastas baseada em /home/ismael/Documentos/GRANULARITY-PREDICTION
    base_dir = Path(__file__).resolve().parents[1]
    resultados_dir = base_dir / "RESULTADOS"
    pred_pipeline_dir = resultados_dir / "prediction_pipeline"
    
    metrics_path = pred_pipeline_dir / "metrics.jsonl"
    predictions_dir = pred_pipeline_dir / "predictions"
    output_dir = resultados_dir / "graficos"
    
    elapsed_time_path = base_dir / "data" / "tratados_stored" / "elapsed_time.json"
    
    # Cria o diretório de saída caso não exista
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists():
        print(f"Arquivo de métricas não encontrado: {metrics_path}")
    else:
        print("Carregando métricas...")
        df = load_metrics(metrics_path)
        
        print("Gerando gráficos de comparação de métricas...")
        plot_metrics(df, output_dir)
        
        print("Gerando gráficos de predições (séries temporais)...")
        plot_predictions(df, predictions_dir, output_dir)
        
        print("Gerando gráficos de degradação (horizonte de predição)...")
        plot_degradation(df, predictions_dir, output_dir)
    
    print("Gerando gráfico de tempo de execução...")
    plot_elapsed_time(elapsed_time_path, output_dir)
    
    print(f"Gráficos gerados com sucesso na pasta: {output_dir}")

if __name__ == "__main__":
    main()
