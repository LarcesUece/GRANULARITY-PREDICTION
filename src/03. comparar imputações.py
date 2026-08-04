import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
from os import path
from glob import glob

metrics_path = "/home/ismael/Documentos/GRANULARITY-PREDICTION/RESULTADOS/metrics"
graphs_dir = "/home/ismael/Documentos/GRANULARITY-PREDICTION/RESULTADOS/graficos"
json_files = glob(path.join(metrics_path, "*.jsonl"))

if not json_files:
    raise FileNotFoundError(f"No JSONL files found in {metrics_path}")

print(f"Found {len(json_files)} JSONL files.")

all_records = []

for json_file in json_files:
    filename = str(path.splitext(path.basename(json_file))[0])
    # A primeira palavra antes do underline
    imputation_method = filename.split('_')[0]
    
    df = pd.read_json(json_file, lines=True)
    
    for _, row in df.iterrows():
        dataset = row.get('dataset')
        batch_size = row.get('batch_size')
        model = row.get('model')
        
        # O RMSE do teste está no dicionário 'test'
        test_dict = row.get('test', {})
        test_rmse = test_dict.get('NRMSE', None)
        
        all_records.append({
            'Imputacao': imputation_method,
            'Granularidade': dataset,
            'Modelo': model,
            'Batch_Size': batch_size,
            'Test_NRMSE': test_rmse
        })

consolidated_df = pd.DataFrame(all_records)
consolidated_df = consolidated_df.drop_duplicates(subset=['Granularidade', 'Modelo', 'Batch_Size', 'Imputacao'], keep='last').reset_index(drop=True)
consolidated_df = consolidated_df[consolidated_df['Batch_Size'] == 128]
#print(consolidated_df.head(10))

#print(consolidated_df[(consolidated_df['Granularidade'] == 'day') & (consolidated_df['Modelo'] == 'LSTM') & (consolidated_df['Imputacao'] == 'granufill')])

import seaborn as sns
import os

os.makedirs(graphs_dir, exist_ok=True)

granularities = ['day', 'hour', '10min']

for granularity in granularities:
    granularity_df = consolidated_df[consolidated_df['Granularidade'] == granularity]
    
    if granularity_df.empty:
        continue
        
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Barras agrupadas por Modelo (X), cor da barra é o método de Imputação (hue)
    sns.barplot(data=granularity_df, x='Modelo', y='Test_NRMSE', hue='Imputacao', ax=ax, palette='Set2')
    
    ax.set_title(f'Comparação de Métodos de Imputação - Dataset: {granularity.capitalize()}', fontsize=14)
    ax.set_ylabel('NRMSE (Test)')
    ax.set_xlabel('Modelo')
    
    # Ajusta a legenda para ficar fora do gráfico e não cobrir as barras
    plt.legend(title='Método de Imputação', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(path.join(graphs_dir, f'comparacao_imputacao_{granularity}.png'), dpi=300)
    plt.close()

print(f"Gráficos gerados com sucesso na pasta: {graphs_dir}")
