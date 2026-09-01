from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dados"
INSTITUTIONS_PATH = DATA_PATH / "institutions"
TIMES_PATH = DATA_PATH / "times"
TRATADOS_PATH = DATA_PATH / "tratados"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from funcoes_de_predicao.funcoes_de_tratamento import (
    granufill,
    knn_fill_missing,
    svd_fill_missing,
    cubic_fill_missing,
    linear_fill_missing,
    quadratic_fill_missing,
    moving_average_fill,
    moving_median_fill,
    ffill_fill_missing,
    ewma_fill_missing,
    seasonal_fill_missing,
    pchip_fill_missing,
)



import pandas as pd
import numpy as np
import polars as pl
pl.enable_string_cache()
pd.set_option('future.no_silent_downcasting', True)
import glob
import os
import joblib
from time import perf_counter
import json
from statsmodels.tsa.seasonal import STL
from sklearn.neighbors import KNeighborsRegressor


def read_data(path):
    """Read a treated parquet (time, id_institution, n_bytes) as a pandas DataFrame."""
    return pl.read_parquet(path).to_pandas()


METRIC_GROUPS = {
    "statistical": ["mean", "std", "variance", "minimum", "maximum", "range",
                    "median", "q10", "q25", "q75", "q90", "q95"],
    "variability": ["cv", "fano", "burstiness", "peak_mean"],
    "sparsity": ["zero_ratio", "active_ratio", "missing_ratio", "n_available"],
    "distribution": ["entropy", "skewness", "kurtosis", "iqr"],
    "temporal": ["acf_lag1", "acf_lag2", "local_trend", "previous_hour_mean"],
    "stl_trend": ["stl_trend_mean", "stl_trend_std", "stl_trend_slope"],
    "stl_seasonal": ["stl_seasonal_mean", "stl_seasonal_std",
                     "stl_seasonal_amplitude", "stl_seasonal_ratio"],
    "stl_noise": ["stl_noise_mean", "stl_noise_std",
                  "stl_noise_abs_mean", "stl_noise_ratio"],
    "weekly": ["mean_w", "std_w", "cv_w", "acf_lag1_w"],
    "cross_granularity": ["gm10_mean", "gm10_std", "gm10_max", "gm10_min",
                          "gm10_sum", "gm10_count", "gm10_active_ratio", "gm10_cv"],
}

# Lista única, preservando a ordem dos grupos.
ALL_METRIC_IMPUTATIONS = list(dict.fromkeys(
    metric for metrics in METRIC_GROUPS.values() for metric in metrics
))


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


class timeInputer:
    def __init__(self,pct):
        self.df_day = read_data(TRATADOS_PATH / "df_day.parquet")
        self.df_hour = read_data(TRATADOS_PATH / "df_hour.parquet")
        self.df_10min = read_data(TRATADOS_PATH / "df_10min.parquet")
        self.pct = pct
        self.inst = None
        self.elapsed_time = {}
        self._stl_cache = {}
    
    def filter_inst(self):
        inst = []

        for i in self.df_day["id_institution"].unique():
            instituicao = self.df_day[self.df_day["id_institution"]==i]
            serve = True
            Q1 = instituicao["n_bytes"].quantile(0.20)
            Q3 = instituicao["n_bytes"].quantile(0.80)
            IQR = Q3 - Q1
            L_inferior = Q1 - 1.5*IQR
            L_superior = Q3 + 1.5*IQR

            #verifica se ha instituições com poucos dados e elimina elas se for o caso
            if instituicao.shape[0] < 260:
                serve = False
                
            #verifica se ha instituições com variância zero e elimina elas se for o caso
            elif instituicao["n_bytes"].var() == 0:
                serve = False

            #verifica se ha instituições com valores negativos e elimina elas se for o caso
            elif instituicao[instituicao["n_bytes"]<0].shape[0]>0:
                serve = False

            #verifica se ha instituições com outliers (valores muito altos) e elimina elas se for o caso
            elif instituicao[(instituicao["n_bytes"] > L_superior) | (instituicao["n_bytes"] < L_inferior)].shape[0]>0:
                serve = False


            if serve:
                inst.append(i)


        self.df_day = self.df_day[self.df_day["id_institution"].isin(inst)].reset_index(drop = True)
        self.df_hour = self.df_hour[self.df_hour["id_institution"].isin(inst)].reset_index(drop = True)
        self.df_10min = self.df_10min[self.df_10min["id_institution"].isin(inst)].reset_index(drop = True)
        self.inst = [int(c) for c in inst]
        joblib.dump(inst, "inst.joblib")
    
    def insertTimeRange(self):
        series_day = pd.Series( pd.date_range(start = self.df_day["time"].min(), end = self.df_day["time"].max(), freq = "D"), name = "time")
        series_hour = pd.Series(pd.date_range(start=self.df_hour["time"].min(), end=self.df_hour["time"].max(), freq='h'), name="time")
        series_10min = pd.Series(pd.date_range(start=self.df_10min["time"].min(), end=self.df_10min["time"].max(), freq='10min'), name="time") 
        
        dfs = []
        for id in self.inst:
            df_d = self.df_day[self.df_day["id_institution"] ==  id].copy()
            df_d = df_d.merge(series_day, on = "time", how = "right")
            df_d["id_institution"] = df_d["id_institution"].fillna(id).astype("int")
            df_d["n_bytes"] = df_d["n_bytes"].bfill().ffill()
            dfs.append(df_d)
        self.df_day = pd.concat(dfs).reset_index(drop = True)
        #print("DIA: ",self.df_day["n_bytes"].isna().sum())
        dfs = []
        for id in self.inst:
            df_h = self.df_hour[self.df_hour["id_institution"] ==  id].copy()
            df_h = df_h.merge(series_hour, on = "time", how = "right")              
            df_h["id_institution"] =df_h["id_institution"].fillna(id).astype("int")
            dfs.append(df_h)
        self.df_hour = pd.concat(dfs).reset_index(drop = True)
        dfs = []
        for id in self.inst:
            df_m = self.df_10min[self.df_10min["id_institution"] ==  id].copy()
            df_m = df_m.merge(series_10min, on = "time", how = "right")
            df_m["id_institution"] = df_m["id_institution"].fillna(id).astype("int")
            dfs.append(df_m)
        self.df_10min = pd.concat(dfs).reset_index(drop = True)

    def save_df_day(self):
        FULL_DATASET_PATH = DATA_PATH / 'tratados' / str(self.pct)
        pl.from_pandas(self.df_day).write_parquet(FULL_DATASET_PATH / "df_day.parquet")

    def _random_na(self, df: pd.DataFrame, percentage: float):

        print(len(df))
        print(df["n_bytes"].isnull().sum()/len(df))

        already_nan = df[df["n_bytes"].isnull()].index
        # Calcula a quantidade de registros a serem apagados
        indexes = df.index
        indexes_no_nan = df.drop(already_nan).index
        n_to_replace = int(len(indexes) * percentage) - len(already_nan) if int(len(indexes_no_nan) * percentage) > len(already_nan) else 0
        # Sorteia os índices aleatoriamente (sem repetição)
        idx_to_replace = np.random.choice(indexes_no_nan, size=n_to_replace, replace=False)
        idx_to_replace = [int(i) for i in idx_to_replace]
        idx_to_replace.extend(already_nan)
        idx_to_replace = sorted(idx_to_replace)



        # Atribui NaN na coluna "n_bytes" apenas para os índices sorteados
        df.loc[idx_to_replace, ["n_bytes"]] = np.nan
        
        print(len(df[df["n_bytes"].isnull()])/len(df))
        print()

        return df

    def remove_random_values(self, percentage):
        self.df_hour = self._random_na(self.df_hour, percentage)
        self.df_10min = self._random_na(self.df_10min, percentage)
        


    def _inputeWithMovingAverage(self, df_: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = moving_average_fill(df.loc[filtro, "n_bytes"], window_size = window_size)
        return df
    
    def _inputeWithMovingMedian(self, df_: pd.DataFrame, window_size: int = 24) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = moving_median_fill(df.loc[filtro, "n_bytes"], window_size = window_size)
        return df

    def _inputeWithKNN(self, df_: pd.DataFrame, k: int = 3) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = knn_fill_missing(df.loc[filtro, "n_bytes"], k = k, weights = 'distance')
        return df

    def _inputeWithSVD(self, df_: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = svd_fill_missing(df.loc[filtro, "n_bytes"], n_components = n_components)
        return df

    def _inputeWithGranularity(self, df_greater: pd.DataFrame,df_lesser: pd.DataFrame,gran_diff: int ):
        df_greater = df_greater.reset_index(drop=True)
        df_lesser = df_lesser.reset_index(drop=True)
        return granufill(df_greater, df_lesser, merging_features = ["time", "id_institution"], target_feature = "n_bytes", gran_diff = gran_diff)

    def _inputeWithKNNWithGranufill(self, df_greater: pd.DataFrame, df_lesser: pd.DataFrame, gran_diff: int, k: int = 3):
        df_greater = df_greater.reset_index(drop=True)
        df_lesser = df_lesser.reset_index(drop=True)
        dfs = []
        for id in self.inst:
            filtro = df_lesser["id_institution"] == id
            df_inst = df_lesser[filtro].reset_index(drop=True)
            if not df_inst["n_bytes"].isna().any():
                dfs.append(df_inst)
                continue
            dfs.append(knn_with_granufill(
                df_greater, df_inst,
                merging_features=["time", "id_institution"],
                target_feature="n_bytes",
                gran_diff=gran_diff,
                k=k,
            ))
        return pd.concat(dfs, ignore_index=True)


    def _inputeWithCubic(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = cubic_fill_missing(df.loc[filtro, "n_bytes"])
        return df

    def _inputeWithLinear(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = linear_fill_missing(df.loc[filtro, "n_bytes"])
        return df

    def _inputeWithQuadratic(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = quadratic_fill_missing(df.loc[filtro, "n_bytes"])
        return df

    def _inputeWithFFill(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = ffill_fill_missing(df.loc[filtro, "n_bytes"])
        return df

    def _inputeWithEWMA(self, df_: pd.DataFrame, span: int = 24) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = ewma_fill_missing(df.loc[filtro, "n_bytes"], span=span)
        return df

    def _inputeWithSeasonal(self, df_: pd.DataFrame, lag: int = 144) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = seasonal_fill_missing(df.loc[filtro, "n_bytes"], lag=lag)
        return df

    def _inputeWithPchip(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = pchip_fill_missing(df.loc[filtro, "n_bytes"])
        return df

    @staticmethod
    def _safe_divide(numerator: float, denominator: float) -> float:
        """Divisão protegida para métricas de razão."""
        if denominator is None or not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return float(numerator / denominator)

    @staticmethod
    def _entropy(values: pd.Series) -> float:
        """Entropia de Shannon a partir de um histograma com bins automáticos."""
        x = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        if x.size == 0:
            return np.nan
        if np.all(x == x[0]):
            return 0.0
        counts, _ = np.histogram(x, bins="auto")
        counts = counts[counts > 0]
        if counts.size == 0:
            return np.nan
        probabilities = counts / counts.sum()
        return float(-(probabilities * np.log(probabilities)).sum())

    @staticmethod
    def _linear_slope(values: pd.Series) -> float:
        """Inclinação linear em função da posição temporal da observação."""
        y = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        x = np.arange(len(y), dtype=float)
        valid = np.isfinite(y)
        if valid.sum() < 2:
            return np.nan
        return float(np.polyfit(x[valid], y[valid], 1)[0])

    @staticmethod
    def _acf(values: pd.Series, lag: int) -> float:
        x = pd.to_numeric(values, errors="coerce")
        if x.notna().sum() <= lag:
            return np.nan
        return float(x.autocorr(lag=lag))

    def _basic_metric_value(self, series: pd.Series, metric: str, granularity: str) -> float:
        """Calcula métricas estatísticas, de variabilidade, esparsidade, distribuição e temporal."""
        s = pd.to_numeric(series, errors="coerce")
        available = s.dropna()

        if metric == "missing_ratio":
            return float(s.isna().mean()) if len(s) else np.nan
        if metric == "n_available":
            return float(s.notna().sum())
        if metric == "zero_ratio":
            return float((s.fillna(np.nan) == 0).sum() / len(s)) if len(s) else np.nan
        if metric == "active_ratio":
            return float((s.fillna(np.nan) > 0).sum() / len(s)) if len(s) else np.nan

        if available.empty:
            return np.nan

        if metric == "mean":
            return float(available.mean())
        if metric == "std":
            return float(available.std())
        if metric == "variance":
            return float(available.var())
        if metric == "minimum":
            return float(available.min())
        if metric == "maximum":
            return float(available.max())
        if metric == "range":
            return float(available.max() - available.min())
        if metric == "median":
            return float(available.median())
        if metric == "q10":
            return float(available.quantile(0.10))
        if metric == "q25":
            return float(available.quantile(0.25))
        if metric == "q75":
            return float(available.quantile(0.75))
        if metric == "q90":
            return float(available.quantile(0.90))
        if metric == "q95":
            return float(available.quantile(0.95))

        mean = float(available.mean())
        std = float(available.std())
        variance = float(available.var())

        if metric == "cv":
            return self._safe_divide(std, mean)
        if metric == "fano":
            return self._safe_divide(variance, mean)
        if metric == "burstiness":
            return self._safe_divide(std - mean, std + mean)
        if metric == "peak_mean":
            return self._safe_divide(float(available.max()), mean)

        if metric == "entropy":
            return self._entropy(available)
        if metric == "skewness":
            return float(available.skew())
        if metric == "kurtosis":
            return float(available.kurt())
        if metric == "iqr":
            return float(available.quantile(0.75) - available.quantile(0.25))

        if metric == "acf_lag1":
            return self._acf(s, 1)
        if metric == "acf_lag2":
            return self._acf(s, 2)
        if metric == "local_trend":
            return self._linear_slope(s)
        if metric == "previous_hour_mean":
            lag = 1 if granularity == "hour" else 6
            shifted = s.shift(lag).dropna()
            return float(shifted.mean()) if not shifted.empty else np.nan

        raise ValueError(f"Métrica básica desconhecida: {metric}")

    def _weekly_metric_value(self, df_inst: pd.DataFrame, metric: str) -> float:
        """
        Métricas semanais calculadas sobre a série de médias semanais.
        A coluna time precisa estar disponível e já convertida para datetime.
        """
        tmp = df_inst[["time", "n_bytes"]].copy()
        tmp["time"] = pd.to_datetime(tmp["time"])
        tmp["n_bytes"] = pd.to_numeric(tmp["n_bytes"], errors="coerce")
        tmp = tmp.dropna(subset=["time"]).sort_values("time").set_index("time")

        if tmp.empty:
            return np.nan

        weekly = tmp["n_bytes"].resample("W").mean().dropna()
        if weekly.empty:
            return np.nan

        if metric == "mean_w":
            return float(weekly.mean())
        if metric == "std_w":
            return float(weekly.std())
        if metric == "cv_w":
            return self._safe_divide(float(weekly.std()), float(weekly.mean()))
        if metric == "acf_lag1_w":
            return self._acf(weekly, 1)

        raise ValueError(f"Métrica semanal desconhecida: {metric}")

    def _gm10_metric_value(self, institution_id: int, metric: str) -> float:
        """Métricas cross-granularity calculadas na série original de 10 minutos."""
        source = self.df_10min[self.df_10min["id_institution"] == institution_id]["n_bytes"]
        s = pd.to_numeric(source, errors="coerce")
        available = s.dropna()

        if metric == "gm10_count":
            return float(available.count())
        if metric == "gm10_active_ratio":
            return float((s.fillna(np.nan) > 0).sum() / len(s)) if len(s) else np.nan
        if available.empty:
            return np.nan
        if metric == "gm10_mean":
            return float(available.mean())
        if metric == "gm10_std":
            return float(available.std())
        if metric == "gm10_max":
            return float(available.max())
        if metric == "gm10_min":
            return float(available.min())
        if metric == "gm10_sum":
            return float(available.sum())
        if metric == "gm10_cv":
            return self._safe_divide(float(available.std()), float(available.mean()))

        raise ValueError(f"Métrica cross-granularity desconhecida: {metric}")

    def _stl_metric_values(self, series: pd.Series, period: int) -> dict:
        """
        Decompõe uma série com STL e retorna todas as métricas STL solicitadas.

        Para permitir a decomposição, NaNs são preenchidos SOMENTE em uma cópia
        temporária por interpolação linear. Os dados originais não são alterados.
        """
        s = pd.to_numeric(series, errors="coerce").astype(float)
        if s.notna().sum() < max(2 * period, 10):
            return {}

        temp = s.interpolate(method="linear", limit_direction="both")
        if temp.isna().any():
            median = s.median()
            if not np.isfinite(median):
                return {}
            temp = temp.fillna(median)

        try:
            result = STL(temp.to_numpy(dtype=float), period=period, robust=True).fit()
        except Exception:
            return {}

        trend = pd.Series(result.trend, index=s.index, dtype=float)
        seasonal = pd.Series(result.seasonal, index=s.index, dtype=float)
        noise = pd.Series(result.resid, index=s.index, dtype=float)

        total_var = float(np.var(temp.to_numpy(dtype=float), ddof=1)) if len(temp) > 1 else np.nan
        seasonal_var = float(np.var(seasonal.to_numpy(dtype=float), ddof=1)) if len(seasonal) > 1 else np.nan
        noise_var = float(np.var(noise.to_numpy(dtype=float), ddof=1)) if len(noise) > 1 else np.nan

        return {
            "stl_trend_mean": float(trend.mean()),
            "stl_trend_std": float(trend.std()),
            "stl_trend_slope": self._linear_slope(trend),
            "stl_seasonal_mean": float(seasonal.mean()),
            "stl_seasonal_std": float(seasonal.std()),
            "stl_seasonal_amplitude": float(seasonal.max() - seasonal.min()),
            "stl_seasonal_ratio": self._safe_divide(seasonal_var, total_var),
            "stl_noise_mean": float(noise.mean()),
            "stl_noise_std": float(noise.std()),
            "stl_noise_abs_mean": float(noise.abs().mean()),
            "stl_noise_ratio": self._safe_divide(noise_var, total_var),
        }

    def _metric_value(self, df_inst: pd.DataFrame, metric: str, granularity: str) -> float:
        """Despacha o cálculo da métrica usada como valor de imputação."""
        if metric in METRIC_GROUPS["weekly"]:
            return self._weekly_metric_value(df_inst, metric)

        if metric in METRIC_GROUPS["cross_granularity"]:
            institution_id = int(df_inst["id_institution"].iloc[0])
            return self._gm10_metric_value(institution_id, metric)

        if metric in (METRIC_GROUPS["stl_trend"] +
                      METRIC_GROUPS["stl_seasonal"] +
                      METRIC_GROUPS["stl_noise"]):
            period = 24 if granularity == "hour" else 24 * 6
            institution_id = int(df_inst["id_institution"].iloc[0])
            cache_key = (institution_id, period)
            # ponytail: uma única decomposição STL serve às 11 métricas STL; cache evita 11x o mesmo cálculo
            if cache_key not in self._stl_cache:
                self._stl_cache[cache_key] = self._stl_metric_values(df_inst["n_bytes"], period=period)
            return self._stl_cache[cache_key].get(metric, np.nan)

        return self._basic_metric_value(df_inst["n_bytes"], metric, granularity)

    def _inputeWithMetric(self, df_: pd.DataFrame, metric: str, granularity: str) -> pd.DataFrame:
        """
        Usa uma feature/métrica como método de imputação.

        Para cada instituição, a métrica é calculada com os dados disponíveis e
        seu valor substitui somente as posições originalmente ausentes. Se a
        métrica for indefinida (NaN/inf), usa-se a mediana observada como fallback.
        """
        if metric not in ALL_METRIC_IMPUTATIONS:
            raise ValueError(f"Métrica não registrada para imputação: {metric}")
        if granularity not in {"hour", "10min"}:
            raise ValueError(f"Granularidade inválida: {granularity}")

        df = df_.copy()

        for institution_id in self.inst:
            filtro = df["id_institution"] == institution_id
            if not filtro.any():
                continue

            df_inst = df.loc[filtro].copy()
            missing_mask = df_inst["n_bytes"].isna()
            if not missing_mask.any():
                continue

            value = self._metric_value(df_inst, metric, granularity)

            if value is None or not np.isfinite(value):
                fallback = pd.to_numeric(df_inst["n_bytes"], errors="coerce").median()
                value = float(fallback) if np.isfinite(fallback) else 0.0

            missing_indexes = df_inst.index[missing_mask]
            df.loc[missing_indexes, "n_bytes"] = float(value)

        return df

    
    def countTimeFilling(self, method, granularity, func,*args, **kwargs):

        if method not in self.elapsed_time.keys():
            self.elapsed_time[method] = {}

        start = perf_counter()
        df_ = func(*args, **kwargs)
        end = perf_counter()
        self.elapsed_time[method][granularity] = end - start

        if not (DATA_PATH / "tratados" / str(self.pct)).exists():
            (DATA_PATH / "tratados" / str(self.pct)).mkdir(parents=True, exist_ok=True)
        if not (DATA_PATH / "tratados" / str(self.pct) / method).exists():
            (DATA_PATH / "tratados" / str(self.pct) / method).mkdir(parents=True, exist_ok=True)
        
        pl.from_pandas(df_).write_parquet(DATA_PATH / "tratados" / str(self.pct) / method / f"df_{granularity}.parquet")
        print(60*"="+"\n" +  method + "\n" + 60*"="+"\n")
        print(granularity + ":   ", df_["n_bytes"].isna().sum())
        print()


    def runFilling(self):
        self.countTimeFilling("granufill", "hour", self._inputeWithGranularity, self.df_day, self.df_hour, 24)
        df_hour = self._inputeWithGranularity(self.df_day, self.df_hour, 24)   
        self.countTimeFilling("granufill", "10min", self._inputeWithGranularity, df_hour, self.df_10min, 24*6)
        del df_hour
        self.countTimeFilling("knn_granufill", "hour", self._inputeWithKNNWithGranufill, self.df_day, self.df_hour, 24)
        df_hour_knn = self._inputeWithKNNWithGranufill(self.df_day, self.df_hour, 24)
        self.countTimeFilling("knn_granufill", "10min", self._inputeWithKNNWithGranufill, df_hour_knn, self.df_10min, 24*6)
        del df_hour_knn
        self.countTimeFilling("moving_average", "hour", self._inputeWithMovingAverage, self.df_hour, 24)
        self.countTimeFilling("moving_average", "10min", self._inputeWithMovingAverage, self.df_10min, 24*6)    
        self.countTimeFilling("moving_median", "hour", self._inputeWithMovingMedian, self.df_hour, 24)
        self.countTimeFilling("moving_median", "10min", self._inputeWithMovingMedian, self.df_10min, 24*6)
        self.countTimeFilling("knn", "hour", self._inputeWithKNN, self.df_hour, 24)
        self.countTimeFilling("knn", "10min", self._inputeWithKNN, self.df_10min, 144)
        self.countTimeFilling("cubic", "hour", self._inputeWithCubic, self.df_hour)
        self.countTimeFilling("cubic", "10min", self._inputeWithCubic, self.df_10min)
        self.countTimeFilling("linear", "hour", self._inputeWithLinear, self.df_hour)
        self.countTimeFilling("linear", "10min", self._inputeWithLinear, self.df_10min)
        self.countTimeFilling("quadratic", "hour", self._inputeWithQuadratic, self.df_hour)
        self.countTimeFilling("quadratic", "10min", self._inputeWithQuadratic, self.df_10min)
        self.countTimeFilling("ffill", "hour", self._inputeWithFFill, self.df_hour)
        self.countTimeFilling("ffill", "10min", self._inputeWithFFill, self.df_10min)
        self.countTimeFilling("ewma", "hour", self._inputeWithEWMA, self.df_hour, 24)
        self.countTimeFilling("ewma", "10min", self._inputeWithEWMA, self.df_10min, 144)
        self.countTimeFilling("seasonal", "hour", self._inputeWithSeasonal, self.df_hour, 24)
        self.countTimeFilling("seasonal", "10min", self._inputeWithSeasonal, self.df_10min, 144)
        self.countTimeFilling("pchip", "hour", self._inputeWithPchip, self.df_hour)
        self.countTimeFilling("pchip", "10min", self._inputeWithPchip, self.df_10min)

        # Cada feature solicitada também passa a ser avaliada como um método de imputação.
        # O nome da pasta/método é exatamente o nome da métrica.
        for metric in ALL_METRIC_IMPUTATIONS:
            self.countTimeFilling(metric, "hour", self._inputeWithMetric,
                                  self.df_hour, metric, "hour")
            self.countTimeFilling(metric, "10min", self._inputeWithMetric,
                                  self.df_10min, metric, "10min")
        
        #self.countTimeFilling("svd", "hour", self._inputeWithSVD, self.df_hour, 24)
        #self.countTimeFilling("svd", "10min", self._inputeWithSVD, self.df_10min, 144)

        with open(DATA_PATH / "tratados" / str(self.pct) / "elapsed_time.json", "w") as f:
            json.dump(self.elapsed_time, f, indent=4)
        print("tempo de execução salvo em " + str(DATA_PATH / "tratados" / str(self.pct) / "elapsed_time.json"))





    def runAll(self):
        print("iniciando...")
        
        print("filtrando dados...")
        self.filter_inst()

        print("inserindo lacunas...")
        self.insertTimeRange()
        print("Day", self.df_day["n_bytes"].isna().sum())
        print("Hour", self.df_hour["n_bytes"].isna().sum())
        print("10min", self.df_10min["n_bytes"].isna().sum())
        print()

        print("inserindo nulos aleatórios em horas e minutos ...")
        self.remove_random_values(self.pct)

        print("Day", self.df_day["n_bytes"].isna().sum())
        print("Hour", self.df_hour["n_bytes"].isna().sum())
        print("10min", self.df_10min["n_bytes"].isna().sum())
        print()

        print("rodando os metodos...")
        self.runFilling()
        #self.save_df_day()

        print("feito!")
        print(self.elapsed_time)
        

if __name__ == "__main__":
    processor = timeInputer(0.2)
    processor.runAll()
        