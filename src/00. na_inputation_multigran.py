from funcoes_de_predicao import funcoes_de_tratamento
from sqlalchemy.sql import traversals
from pandas import plotting
from funcoes_de_predicao import funcoes_de_tratamento
from pathlib import Path
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
INSTITUTIONS_PATH = DATA_PATH / "institutions"
TIMES_PATH = DATA_PATH / "times"
TRATADOS_PATH = DATA_PATH / "tratados"

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from funcoes_de_predicao.funcoes_de_tratamento import (
    read_data,
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class timeInputer:
    def __init__(self,pct):
        self.df_day = read_data(INSTITUTIONS_PATH / "agg_1_day", features = None)
        self.df_hour = read_data(INSTITUTIONS_PATH / "agg_1_hour", features = None)
        self.df_10min = read_data(INSTITUTIONS_PATH / "agg_10_minutes", features = None)
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

    def _merge_id_date(self, df: pd.DataFrame, time: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["id_time"] = df["id_time"].astype(int) 
        df = df.merge(time, on="id_time", how="left")
        df["time"] = pd.to_datetime(df["time"], utc=True)

        df["time"] = pd.to_datetime(pd.DataFrame(
                {
                    "year": df["time"].dt.year.astype("int"),
                    "month": df["time"].dt.month.astype("int"),
                    "day": df["time"].dt.day.astype("int"),
                    "hour": df["time"].dt.hour.astype("int"),
                    "minute": (round(df["time"].dt.minute.astype("int") /10))*10,
                    "second": 0
                }
            )
        )
        df = df.sort_values(by = ["time", "id_institution"])
        df.drop(columns=["id_time"], inplace=True)
        df["id_institution"] = df["id_institution"].astype("int")
        return df


            

        
    def insertTimeRange(self):
        self.df_day = self._merge_id_date(self.df_day, pd.read_csv(TIMES_PATH / 'times_1_day.csv'))
        self.df_hour = self._merge_id_date(self.df_hour, pd.read_csv(TIMES_PATH / 'times_1_hour.csv'))
        self.df_10min = self._merge_id_date(self.df_10min, pd.read_csv(TIMES_PATH / 'times_10_minutes.csv'))

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

    def _train_test_split(self,df, train_size):
        train = []
        test = []
        for inst in list(df['id_institution'].unique()):
            train.append(df.iloc[:train_size])
            test.append(df.iloc[train_size:])
        return pd.concat[train], pd.concat[test]

    def _apply_pca_by_groups(
        self,
        df: pd.DataFrame, 
        pca_groups: dict = None, 
        extra_cols_to_drop: list = None,
        drop_original_cols: bool = True,
        n_components: int = 1
    ) -> pd.DataFrame:
        """
        Aplica PCA em grupos customizados de colunas para qualquer DataFrame.
        
        Parâmetros:
        -----------
        df : pd.DataFrame
            DataFrame de entrada (ex: df_day, df_hour, df_10min).
        pca_groups : dict, optional
            Dicionário onde a CHAVE é o nome da nova feature e o VALOR é a lista de colunas.
            Se None, utiliza os grupos padrão de Volume e Topologia.
        extra_cols_to_drop : list, optional
            Lista de colunas adicionais para remover no final (ex: ['tcp_udp_ratio_bytes']).
        drop_original_cols : bool, default True
            Se True, remove as colunas originais que foram resumidas pelo PCA.
        n_components : int, default 1
            Número de componentes a extrair por grupo.
            
        Retorna:
        --------
        pd.DataFrame
            Novo DataFrame com as componentes do PCA e colunas ajustadas.
        """
        df_out = df.copy()
        
        # Grupos padrão caso nenhum seja passado
        if pca_groups is None:
            pca_groups = {
                'pca_volume': ['n_packets', 'n_flows', 'sum_n_dest_ip', 'sum_n_dest_ports', 'sum_n_dest_asn'],
                'pca_topologia': ['average_n_dest_asn', 'std_n_dest_asn', 'average_n_dest_ip', 'std_n_dest_ip']
            }
            
        cols_used_in_pca = []
        
        # Processa cada grupo definido no dicionário
        for new_feature_name, cols in pca_groups.items():
            # Garante que só processa colunas existentes no DataFrame recebido
            existing_cols = [c for c in cols if c in df_out.columns]
            
            if len(existing_cols) > 0:
                scaler = StandardScaler()
                # Trata nulos antes de normalizar
                scaled_data = scaler.fit_transform(df_out[existing_cols].fillna(-1))
                
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(scaled_data)
                
                # Adiciona a(s) nova(s) coluna(s) gerada(s)
                if n_components == 1:
                    df_out[new_feature_name] = pca_result[:, 0]
                else:
                    for i in range(n_components):
                        df_out[f"{new_feature_name}_{i+1}"] = pca_result[:, i]
                        
                cols_used_in_pca.extend(existing_cols)

        # Define quais colunas serão descartadas no final
        cols_to_drop = []
        if drop_original_cols:
            cols_to_drop.extend(cols_used_in_pca)
            
        if extra_cols_to_drop:
            cols_to_drop.extend(extra_cols_to_drop)
            
        # Remove colunas duplicadas da lista de descarte e aplica a remoção
        cols_to_drop = list(set([c for c in cols_to_drop if c in df_out.columns]))
        df_out = df_out.drop(columns=cols_to_drop)
        
        return df_out
    def transform_pca(self):
        self.df_day = self._apply_pca_by_groups(self.df_day)
        self.df_hour = self._apply_pca_by_groups(self.df_hour)
        self.df_10min = self._apply_pca_by_groups(self.df_10min)
    
    def _build_lagged_dataset(self, df: pd.DataFrame, target_col: str, feature_cols: list, lags: list) -> pd.DataFrame:
        df = df.copy()
        
        # Identifica e converte colunas do tipo 'object' para float no DataFrame original
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.fillna(-1)
        
        df_lagged = pd.DataFrame(index=df.index)
        
        # Adiciona a variável alvo atual (t)
        df_lagged[target_col] = df[target_col]
        
        # Cria os lags selecionados para as variáveis pertinentes
        for col in feature_cols:
            if col in df.columns:
                for lag in lags:
                    df_lagged[f"{col}_lag_{lag}"] = df[col].shift(lag)
                    
        # Preenche valores nulos do shift e garante tipo float para todas as colunas
        df_lagged = df_lagged.fillna(-1)
        return df_lagged.astype(float)

    # 3. SELEÇÃO E FILTRAGEM VIA XGBOOST
    def _xgboost_train_pertinent(self, df_lagged: pd.DataFrame, target_col: str = "n_bytes", threshold: float = 0.02):
        np.random.seed(42)
        
        # Verificação de segurança adicional para colunas do tipo 'object'
        object_cols = df_lagged.select_dtypes(include=['object', 'category']).columns
        if len(object_cols) > 0:
            print(f"Convertendo colunas do tipo 'object' encontradas para float: {list(object_cols)}")
            for col in object_cols:
                df_lagged[col] = pd.to_numeric(df_lagged[col], errors='coerce').fillna(-1)
                
        # Assegura que todo o DataFrame esteja em float antes da separação em X e y
        df_lagged = df_lagged.astype(float)
        
        X = df_lagged.drop(columns=[target_col], errors='ignore')
        y = df_lagged[target_col]

        # Modelo XGBoost com parâmetros para evitar overfitting no ruído
        modelo_xgb = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

        modelo_xgb.fit(X, y)

        # Extração e ordenação das importâncias
        importancias = pd.Series(modelo_xgb.feature_importances_, index=X.columns)
        importancias_ordenadas = importancias.sort_values(ascending=False)

        # Aplicação do limiar de corte
        lags_selecionados = importancias_ordenadas[importancias_ordenadas > threshold].index

        print(f"Total de Lags Candidatos Testados: {X.shape[1]}")
        print(f"Lags Retidos (Ganho > {threshold}): {len(lags_selecionados)}")
        print(f"Variáveis/Lags Mantidos: {list(lags_selecionados)}\n")
        
        return lags_selecionados

    def filter_lags(self):
        TARGET = "n_bytes"

        # --- 1. DIA ---
        df_day_clean = self.df_day.select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns]'])
        df_day_clean = df_day_clean.drop(columns=["id_institution"], errors='ignore')
        pertinent_day = list(df_day_clean.columns)
        
        lags_day = [1, 2, 3, 6, 7, 8, 13, 14]
        df_day_lagged = self._build_lagged_dataset(self.df_day, TARGET, pertinent_day, lags_day)
        features_day = self._xgboost_train_pertinent(df_day_lagged, target_col=TARGET, threshold=0.02)
        
        # Mantém colunas de data/tempo e identificadores
        temp_day = list(self.df_day.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns) + ["id_institution"]
        temp_day = [c for c in temp_day if c in self.df_day.columns]
        self.df_day = pd.concat([self.df_day[temp_day], df_day_lagged[features_day]], axis=1)

        # --- 2. HORA ---
        df_hour_clean = self.df_hour.select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns]'])
        df_hour_clean = df_hour_clean.drop(columns=["id_institution"], errors='ignore')
        pertinent_hour = list(df_hour_clean.columns)
        
        lags_hour = [1, 2, 3, 6, 13, 25, 49, 73, 127, 167]
        df_hour_lagged = self._build_lagged_dataset(self.df_hour, TARGET, pertinent_hour, lags_hour)
        features_hour = self._xgboost_train_pertinent(df_hour_lagged, target_col=TARGET, threshold=0.02)
        
        temp_hour = list(self.df_hour.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns) + ["id_institution"]
        temp_hour = [c for c in temp_hour if c in self.df_hour.columns]
        self.df_hour = pd.concat([self.df_hour[temp_hour], df_hour_lagged[features_hour]], axis=1)

        # --- 3. 10 MINUTOS ---
        df_10min_clean = self.df_10min.select_dtypes(exclude=['datetime', 'datetime64', 'datetime64[ns]'])
        df_10min_clean = df_10min_clean.drop(columns=["id_institution"], errors='ignore')
        pertinent_10min = list(df_10min_clean.columns)
        
        lags_10min = [1, 2, 3, 6, 143, 144, 145]
        df_10min_lagged = self._build_lagged_dataset(self.df_10min, TARGET, pertinent_10min, lags_10min)
        features_10min = self._xgboost_train_pertinent(df_10min_lagged, target_col=TARGET, threshold=0.02)
        
        temp_10min = list(self.df_10min.select_dtypes(include=['datetime', 'datetime64', 'datetime64[ns]']).columns) + ["id_institution"]
        temp_10min = [c for c in temp_10min if c in self.df_10min.columns]
        self.df_10min = pd.concat([self.df_10min[temp_10min], df_10min_lagged[features_10min]], axis=1)

        return self.df_day, self.df_hour, self.df_10min

if __name__ == '__main__':
    ti = timeInputer(0.2)
    ti.filter_inst()
    ti.insertTimeRange()
    ti.transform_pca()
    ti.filter_lags()
    print(ti.df_day, ti.df_hour, ti.df_10min)
