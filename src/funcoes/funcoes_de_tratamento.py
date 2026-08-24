import pandas as pd
import numpy as np
import torch
pd.set_option('future.no_silent_downcasting', True)
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from os import path,getcwd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from scipy.linalg import svd


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_data(folder: str, features: list[str] = ["id_time", "n_bytes"]   ) -> list[pd.DataFrame]:
    print(f"1. Python is running from: {getcwd()}")
    print(f"2. Is the target path valid? {path.exists(folder)}")
    if not path.exists(folder):
        return False
    csv_files = glob(path.join(folder, "*.csv"))

    lista_dia = []
    for file in csv_files:
        filename = str(path.splitext(path.basename(file))[0])
        df = pd.read_csv(file)
        if features is not None:
            for f in features:
                if f not in df.columns:
                    print(f"Feature '{f}' not found in file '{filename}'. Adding it with NaN values.")
                    df[f] = np.nan
            df = df[features].sort_values(by=features[0])
        df["id_institution"] = filename
        lista_dia.append(df)

    print("\n--- All dataframes loaded successfully! ---")
    df =pd.concat(lista_dia, ignore_index=True)
    df = df.reset_index(drop=True)
    return df


def treino_val_teste(df = pd.Series, t_treino = 0, t_teste = 0, t_val = 0):
    return  df[:int(t_treino*len(df))], df[int(t_treino*len(df)):int((t_treino + t_val)*len(df))]  , df[int((t_treino + t_val)*len(df)):]

def scaling(df_series: pd.Series, mode = 0, scaler = None) -> pd.Series:
    device = get_device()
    dados_np = df_series.to_numpy(dtype=np.float32).reshape(-1, 1)
    if mode == 0:
        scaler = MinMaxScaler()
        scaler.fit(dados_np)
        dados_tensor = torch.as_tensor(dados_np, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        min_offset = torch.as_tensor(scaler.min_, dtype=torch.float32, device=device)
        dados_scaled = dados_tensor * scale + min_offset
        return pd.Series(dados_scaled.detach().cpu().numpy().flatten(), index=df_series.index, name=df_series.name),scaler
    elif mode == 1:
        dados_tensor = torch.as_tensor(dados_np, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device)
        min_offset = torch.as_tensor(scaler.min_, dtype=torch.float32, device=device)
        dados_scaled = dados_tensor * scale + min_offset
        return pd.Series(dados_scaled.detach().cpu().numpy().flatten(), index=df_series.index, name=df_series.name)
    raise ValueError("mode must be 0 (fit/transform) or 1 (transform)")



def granufill(df_greater: pd.DataFrame, df_less: pd.DataFrame, merging_features: list, target_feature: str, gran_diff: int) -> pd.DataFrame:
    try:
        # Cria cópia para evitar que o .dt.floor altere o df_less original permanentemente em memória
        df_less_copy = df_less.copy()
        
        # Guarda o tempo original numa coluna que participará do merge

        df_less_copy["time_original"] = df_less_copy["time"]
        
        if gran_diff == 24:
            df_less_copy["time"] = df_less_copy["time"].dt.floor("d")
        else:
            df_less_copy["time"] = df_less_copy["time"].dt.floor("h")

        # Executa o merge
        df_merged = df_less_copy.merge(df_greater, on=merging_features, how="left", suffixes=(None, "_greater"))

        # Preenche os nulos. 
        # (Se o df_greater não tinha a chave, o _greater será NaN, e o fillna ignorará essa linha)
        col_greater = f"{target_feature}_greater"
        df_merged[target_feature] = df_merged[target_feature].fillna(df_merged[col_greater] / gran_diff)


        # Restaura o tempo utilizando o alinhamento interno do próprio df_merged
        df_merged["time"] = df_merged["time_original"]

        # Retorna apenas as colunas originais
        df_merged = df_merged[df_less.columns]
        return df_merged
        
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

from sklearn.neighbors import KNeighborsRegressor

def knn_with_granufill(df_greater: pd.DataFrame, df_less: pd.DataFrame, merging_features: list, target_feature: str, gran_diff: int, k: int = 3, weights: str = 'distance') -> pd.DataFrame:
    # 1. Extração rápida da série alvo (a que possui os nulos)
    target_series = df_less[target_feature].to_numpy(dtype=np.float32)
    missing_mask = np.isnan(target_series)
    
    # 2. Fuga antecipada (Short-circuit)
    if not missing_mask.any():
        return df_less.copy()
        
    n_samples = len(target_series)
    n_valid = np.count_nonzero(~missing_mask)
    n_neighbors = min(k, n_valid)
    
    if n_neighbors == 0:
        return df_less.copy()

    # 3. Alinhamento dimensional: Extraindo a feature auxiliar do df_greater
    # Pegamos as chaves de merge e o tempo sem sujar o df original
    cols_align = list(set(merging_features + ['time']))
    df_align = df_less[cols_align].copy()
    
    if gran_diff == 24:
        df_align['time'] = df_align['time'].dt.floor('d')
    else:
        df_align['time'] = df_align['time'].dt.floor('h')
        
    # Trazemos apenas a coluna alvo do df_greater para não duplicar dados no merge
    cols_greater = list(set(merging_features + [target_feature]))
    df_merged = df_align.merge(df_greater[cols_greater], on=merging_features, how='left')
    
    # Agora greater_series tem exatamente o mesmo número de linhas que o df_less
    greater_series = df_merged[target_feature].to_numpy(dtype=np.float32)

    # 4. Normalização do Tempo
    time_scaled = np.linspace(0, 1, n_samples, dtype=np.float32)
    
    # 5. Normalização da Feature Auxiliar
    if np.isnan(greater_series).all():
        greater_scaled = np.zeros(n_samples, dtype=np.float32)
    else:
        greater_min = np.nanmin(greater_series)
        greater_max = np.nanmax(greater_series)
        
        if greater_max > greater_min:
            greater_scaled = (greater_series - greater_min) / (greater_max - greater_min)
        else:
            greater_scaled = np.zeros(n_samples, dtype=np.float32)
            
    # O KNN não aceita nulos no 'X'. Se o df_greater não tinha dados para alguma hora, injetamos zero.
    greater_scaled = np.nan_to_num(greater_scaled, nan=0.0)
        
    # 6. Separação de Treino (não-nulos) e Teste (nulos) 
    # X_train e X_test agora possuem o Tempo + a Feature do df_greater
    X = np.column_stack((time_scaled, greater_scaled))
    
    X_train = X[~missing_mask]
    y_train = target_series[~missing_mask]
    
    X_test = X[missing_mask]
    
    # 7. Treinamento e Predição com KNeighborsRegressor
    knn = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    knn.fit(X_train, y_train)
    
    predicted_values = knn.predict(X_test)
    
    # 8. Cópia e atribuição direta no array NumPy
    target_series[missing_mask] = predicted_values
    
    df_result = df_less.copy()
    df_result[target_feature] = target_series
    
    return df_result





def sliding_window (df_series: pd.Series, inputs: int, outputs: int, step: int = 1) -> pd.DataFrame:

    total_window_size = inputs + outputs
    
    # 1. Validação
    if len(df_series) < total_window_size:
        print(f"Erro: Tamanho dos dados ({len(df_series)}) é menor que a janela total ({total_window_size})")
        return pd.DataFrame() # Retorna um DataFrame vazio

    device = get_device()
    series_tensor = torch.as_tensor(
        df_series.to_numpy(dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    windowed_tensor = series_tensor.unfold(0, total_window_size, step).contiguous()

    # 3. Define os nomes das colunas
    x_cols = [f"x_{j}" for j in range(inputs)]
    y_cols = [f"y_{o}" for o in range(outputs)]
    
    # 4. Cria o DataFrame final
    df_windowed = pd.DataFrame(
        windowed_tensor.detach().cpu().numpy(),
        columns=x_cols + y_cols,
    )
    return df_windowed


def cubic_fill_missing(df_series: pd.Series) -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando interpolação cúbica.
    """
    series_filled = pd.to_numeric(df_series.copy(), errors='coerce')
    
    if not series_filled.isna().any():
        return series_filled
        
    series_filled = series_filled.interpolate(method='cubic').bfill().ffill()
        
    return series_filled


def linear_fill_missing(df_series: pd.Series) -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando interpolação linear.
    """
    series_filled = pd.to_numeric(df_series.copy(), errors='coerce')
    
    if not series_filled.isna().any():
        return series_filled
        
    series_filled = series_filled.interpolate(method='linear').bfill().ffill()
        
    return series_filled


def quadratic_fill_missing(df_series: pd.Series) -> pd.Series:
    """
    Preenche valores ausentes (NaN) em uma série temporal univariada 
    usando interpolação quadrática.
    """
    series_filled = pd.to_numeric(df_series.copy(), errors='coerce')
    
    if not series_filled.isna().any():
        return series_filled
        
    series_filled = series_filled.interpolate(method='quadratic').bfill().ffill()
        
    return series_filled



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
