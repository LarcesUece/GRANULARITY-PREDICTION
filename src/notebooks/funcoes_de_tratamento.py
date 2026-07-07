import pandas as pd
import numpy as np
pd.set_option('future.no_silent_downcasting', True)
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from os import path,getcwd
from sklearn.neighbors import KNeighborsRegressor

def read_data(folder: str, features: list[str] = ["id_time", "n_bytes"]   ) -> list[pd.DataFrame]:
    print(f"1. Python is running from: {getcwd()}")
    print(f"2. Is the target path valid? {path.exists(folder)}")
    csv_files = glob(path.join(folder, "*.csv"))

    lista_dia = []
    for file in csv_files:
        filename = str(path.splitext(path.basename(file))[0])
        df = pd.read_csv(file)
        for f in features:
            if f not in df.columns:
                print(f"Feature '{f}' not found in file '{filename}'. Adding it with NaN values.")
                df[f] = np.nan
        df = df[features].sort_values(by=features[0])
        df["id_institution"] = filename
        lista_dia.append(df)

    print("\n--- All dataframes loaded successfully! ---")
    return pd.concat(lista_dia, ignore_index=True)


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



def granufill(df_greater:pd.DataFrame,df_less:pd.DataFrame, merging_features: list, target_feature: str, gran_diff: int ) -> pd.DataFrame:
    try:
        df_merged = df_less.merge(df_greater, on=merging_features, how="left", suffixes=(None,"_greater") )
        df_merged[target_feature] = df_merged[target_feature].fillna(df_merged[f"{target_feature}_greater"] / gran_diff)
        return df_merged[df_less.columns]   
    except Exception as e:
        print(f"Erro ao preencher granularidade: {e}")
        return df_less 


def knn_fill_missing(df_series: pd.Series, k: int = 3, weights: str = 'distance') -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando K-Nearest Neighbors (KNN) baseado no índice de tempo.
    """
    series_filled = df_series.copy()
    
    missing_mask = series_filled.isna()
    
    if not missing_mask.any():
        return series_filled
        
    # X (features) será o índice (posição temporal), Y será o valor da série
    X_train = np.where(~missing_mask)[0].reshape(-1, 1)
    y_train = series_filled[~missing_mask].values
    
    X_test = np.where(missing_mask)[0].reshape(-1, 1)
    
    # Ajusta o K caso o número de não-nulos seja menor que K
    n_neighbors = min(k, len(X_train))
    if n_neighbors == 0:
        return series_filled # Não há o que preencher se tudo for NaN
    
    # Treina o modelo KNN
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    
    # Prediz os valores faltantes
    predicted_values = knn.predict(X_test)
    series_filled.iloc[np.where(missing_mask)[0]] = predicted_values
        
    return series_filled


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


def moving_average_fill(df_series: pd.Series, window_size: int = 3, center: bool = False) -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando média móvel (rolling mean) do pandas.
    """
    series_filled = df_series.copy()
    
    if not series_filled.isna().any():
        return series_filled
        
    # Calcula a média móvel usando pandas
    moving_avg = series_filled.rolling(window=window_size, min_periods=1, center=center).mean()
    
    # Preenche os valores nulos com a média móvel
    series_filled = series_filled.fillna(moving_avg)
    
    # Preenche possíveis valores nulos restantes nas bordas (ex: se min_periods não resolver tudo)
    if series_filled.isna().any():
        series_filled = series_filled.bfill().ffill()
        
    return series_filled


def svd_fill_missing(df_series: pd.Series, window_size: int = 24, n_components: int = 2, max_iter: int = 5) -> pd.Series:
    """
    Preenche valores ausentes em uma série temporal univariada usando 
    Singular Spectrum Analysis (SSA) baseado em SVD.
    
    Como o scikit-learn e o pandas não possuem um SVD nativo que suporte 
    diretamente séries 1D com nulos, esta é uma implementação iterativa leve 
    usando o numpy.linalg.svd.
    """
    series_filled = df_series.copy()
    missing_mask = series_filled.isna()
    
    if not missing_mask.any():
        return series_filled
        
    N = len(series_filled)
    L = window_size
    K = N - L + 1
    
    # Se a janela for maior que a série, recai para preenchimento simples
    if K <= 0 or L <= 0:
        return series_filled.fillna(series_filled.mean())
        
    # Chute inicial: preenche temporariamente com a média
    vals = series_filled.fillna(series_filled.mean()).values
    
    for _ in range(max_iter):
        # 1. Constrói a Matriz de Trajetória (Hankel Matrix)
        X = np.column_stack([vals[i:i+L] for i in range(K)])
        
        # 2. Aplica o SVD
        try:
            U, s, Vt = np.linalg.svd(X, full_matrices=False)
        except np.linalg.LinAlgError:
            break # Se o SVD não convergir, interrompe o loop
            
        # 3. Trunca mantendo os componentes principais
        k_comp = min(n_components, len(s))
        X_rec = U[:, :k_comp] @ np.diag(s[:k_comp]) @ Vt[:k_comp, :]
        
        # 4. Reconstrói a série 1D (Média das diagonais invertidas)
        vals_rec = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                vals_rec[i+j] += X_rec[i, j]
                counts[i+j] += 1
        vals_rec /= counts
        
        # 5. Atualiza APENAS os valores nulos originais com a reconstrução
        vals[missing_mask] = vals_rec[missing_mask]
        
    series_filled.iloc[:] = vals
    return series_filled


def moving_median_fill(df_series: pd.Series, window_size: int = 3, center: bool = False) -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando mediana móvel (rolling median) do pandas.
    """
    series_filled = df_series.copy()
    
    if not series_filled.isna().any():
        return series_filled
        
    # Calcula a mediana móvel usando pandas
    moving_median = series_filled.rolling(window=window_size, min_periods=1, center=center).median()
    
    # Preenche os valores nulos com a mediana móvel
    series_filled = series_filled.fillna(moving_median)
    
    # Preenche possíveis valores nulos restantes nas bordas (ex: se min_periods não resolver tudo)
    if series_filled.isna().any():
        series_filled = series_filled.bfill().ffill()
        
    return series_filled
