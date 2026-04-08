import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import joblib
inst = joblib.load("../scalers/instituicoes_validas.joblib")


def complete_id(df):
    df = df.copy()
    
    # 1. Cria um MultiIndex com TODAS as combinações teóricas possíveis 
    # (24 horas * 60 minutos = 1440 linhas por ID)
    ids_unicos = df['ID'].unique()
    horas = range(24)
    minutos = range(6)
    
    # Gera a malha cartesiana (ID x Hora x Minuto)
    multi_idx = pd.MultiIndex.from_product(
        [ids_unicos, horas, minutos], 
        names=['ID', 'hour_id', 'min_id']
    )
    
    # 2. Configura a base atual para usar essas três colunas como índice e remove duplicatas
    # O drop_duplicates evita o erro fatal se a sua base original tiver duas marcações exatas no mesmo minuto
    df = df.drop_duplicates(subset=['ID', 'hour_id', 'min_id'])
    df = df.set_index(['ID', 'hour_id', 'min_id'])
    
    # 3. O REINDEX MÁGICO: Ele vai criar as linhas em branco para as horas/minutos que não existem
    df_full = df.reindex(multi_idx).reset_index()
    
    # 4. Preenchimento (Fill)
    # Primeiro agrupa pelo ID para garantir que o bfill/ffill de uma instituição/dia não vaze para outra
    def preencher_gaps(g):
        # Garante a ordem cronológica
        g = g.sort_values(['hour_id', 'min_id'])
        
        # Preenche a id_institution 
        if 'id_institution' in g.columns:
            g['id_institution'] = g['id_institution'].ffill().bfill()
            
        # Preenche as colunas de bytes (ajuste os nomes conforme sua base)
        if 'n_bytes_hour' in g.columns:
            g['n_bytes_hour'] = g['n_bytes_hour'].ffill().bfill()
        if 'n_bytes_10minutes' in g.columns:
            g['n_bytes_10minutes'] = g['n_bytes_10minutes'].ffill().bfill()
            
        # Preenche o resto (exceto as chaves de índice)
        for col in g.columns:
            if col not in ['ID', 'hour_id', 'min_id', 'id_institution', 'n_bytes_hour', 'n_bytes_10minutes']:
                g[col] = g[col].ffill().bfill()
                
        return g

    # Aplica o preenchimento por grupo (por ID)
    df_out = df_full.groupby('ID', group_keys=False).apply(preencher_gaps)
    
    return df_out.reset_index(drop=True)

def treino_val_teste(df = pd.Series, t_treino = 0, t_teste = 0, t_val = 0):
    return  df[:int(t_treino*len(df))], df[int(t_treino*len(df)):int((t_treino + t_val)*len(df))]  , df[int((t_treino + t_val)*len(df)):]

def scaling(df_series: pd.Series, mode = 0, scaler = None) -> pd.Series:
    if mode == 0:
        scaler = MinMaxScaler()
        dados_reshaped = df_series.values.reshape(-1, 1)
        dados_scaled = scaler.fit_transform(dados_reshaped)
        return pd.Series(dados_scaled.flatten(), index=df_series.index, name=df_series.name),scaler
    elif mode == 1:
        dados_reshaped = df_series.values.reshape(-1, 1)
        dados_scaled = scaler.transform(dados_reshaped)
        return pd.Series(dados_scaled.flatten(), index=df_series.index, name=df_series.name)     

def sliding_window (df_series: pd.Series, inputs: int, outputs: int, step: int = 1) -> pd.DataFrame:

    total_window_size = inputs + outputs
    
    # 1. Validação
    if len(df_series) < total_window_size:
        print(f"Erro: Tamanho dos dados ({len(df_series)}) é menor que a janela total ({total_window_size})")
        return pd.DataFrame() # Retorna um DataFrame vazio

    # 2. Cria as janelas (sliding windows)
    windowed_data = []
    # Itera do primeiro índice inicial possível até o último
    for i in range(0, len(df_series) - total_window_size + 1, step):
        # A fatia vai de 'i' até 'i + tamanho_total'
        window_slice = df_series.iloc[i : i + total_window_size].values
        windowed_data.append(window_slice)

    # 3. Define os nomes das colunas
    x_cols = [f"x_{j}" for j in range(inputs)]
    y_cols = [f"y_{o}" for o in range(outputs)]
    
    # 4. Cria o DataFrame final
    df_windowed = pd.DataFrame(windowed_data, columns=x_cols + y_cols)
    return df_windowed
