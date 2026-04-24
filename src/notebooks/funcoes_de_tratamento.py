import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import joblib
inst = joblib.load("../scalers/instituicoes_validas.joblib")
from sklearn.preprocessing import MinMaxScaler
import polars as pl
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pyarrow as pa


def complete_id(df):
    # (O .copy() não é necessário no Polars pois os dados são imutáveis)
    ids_unicos = df.select('ID').unique()
    horas = pl.DataFrame({'hour_id': range(24)})
    minutos = pl.DataFrame({'min_id': range(6)})
    
    multi_idx = ids_unicos.join(horas, how="cross").join(minutos, how="cross")
    df = df.unique(subset=['ID', 'hour_id', 'min_id'], keep='first')
    df_full = multi_idx.join(df, on=['ID', 'hour_id', 'min_id'], how="left")

    preencher_gaps = (
        pl.all()
        .exclude(['ID', 'hour_id', 'min_id'])
        .forward_fill()
        .backward_fill()
        .over('ID')
    )

    df_out = (
        df_full
        .sort(['ID', 'hour_id', 'min_id'])
        .with_columns(preencher_gaps)
    )
    return df_out


def treino_val_teste(df, t_treino=0.6, t_teste=0.1, t_val=None):
    """Divide uma série temporal preservando a ordem dos dados.

    Args:
        df: sequência temporal indexada por tempo (Series/array-like).
        t_treino: fração de treino, por exemplo 0.6.
        t_teste: fração de teste, por exemplo 0.1.
        t_val: fração de validação; se None, usa 1 - treino - teste.

    Returns:
        tupla (train, val, test)
    """
    n = len(df)
    if t_val is None:
        t_val = 1.0 - t_treino - t_teste

    total = float(t_treino + t_teste + t_val)
    if not np.isclose(total, 1.0):
        raise ValueError("As frações de treino, validação e teste devem somar 1.0")

    if n == 0:
        return df, df, df

    train_end = int(t_treino * n)
    val_end = train_end + int(t_teste * n)
    return df[:train_end], df[train_end:val_end], df[val_end:]


def sliding_window(df_series: pl.Series, inputs: int, outputs: int, step: int = 1) -> pl.DataFrame:
    total_window_size = inputs + outputs
    if df_series.len() < total_window_size:
        raise ValueError(
            f"Erro: Tamanho dos dados ({df_series.len()}) menor que a janela total ({total_window_size})"
        )

    arr = df_series.to_numpy()
    windows = sliding_window_view(arr, window_shape=total_window_size)
    if step > 1:
        windows = windows[::step]

    x_cols = [f"x_{j}" for j in range(inputs)]
    y_cols = [f"y_{o}" for o in range(outputs)]
    colunas_finais = x_cols + y_cols
    df_windowed = pl.DataFrame(windows, schema=colunas_finais, orient="row")
    return df_windowed


def scaling(df_series: pl.Series, mode: int = 0, scaler=None, feature_range=(0, 1)):
    """Escala uma série usando MinMaxScaler e mantém compatibilidade com o pipeline atual.

    mode=0: fit + transform
    mode=1: transform somente (scaler obrigatório)
    mode=2: inverse_transform (scaler obrigatório)
    """
    dados_reshaped = df_series.to_numpy().reshape(-1, 1)

    if mode == 0:
        scaler = MinMaxScaler(feature_range=feature_range)
        dados_scaled = scaler.fit_transform(dados_reshaped)
        serie_scaled = pl.Series(name=df_series.name, values=dados_scaled.flatten())
        return serie_scaled, scaler

    if scaler is None:
        raise ValueError("Erro: Um scaler treinado deve ser fornecido quando mode != 0.")

    if mode == 1:
        dados_scaled = scaler.transform(dados_reshaped)
        serie_scaled = pl.Series(name=df_series.name, values=dados_scaled.flatten())
        return serie_scaled

    if mode == 2:
        dados_scaled = scaler.inverse_transform(dados_reshaped)
        serie_scaled = pl.Series(name=df_series.name, values=dados_scaled.flatten())
        return serie_scaled

    raise ValueError("Modo inválido para scaling; use 0, 1 ou 2.")
