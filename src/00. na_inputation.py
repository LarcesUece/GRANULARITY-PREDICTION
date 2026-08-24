
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data"
INSTITUTIONS_PATH = DATA_PATH / "institutions"
TIMES_PATH = DATA_PATH / "times"

import sys
sys.path.append('/home/ismael/Documentos/GRANULARITY-PREDICTION/src/funcoes')



import pandas as pd
import numpy as np
import polars as pl
pl.enable_string_cache()
pd.set_option('future.no_silent_downcasting', True)
import glob
import os
import joblib
from funcoes_de_tratamento import *
from time import perf_counter
import json


class timeInputer:
    def __init__(self,pct):
        self.df_day = read_data(INSTITUTIONS_PATH / "agg_1_day")
        self.df_hour = read_data(INSTITUTIONS_PATH / "agg_1_hour")
        self.df_10min = read_data(INSTITUTIONS_PATH / "agg_10_minutes")
        self.pct = pct
        self.inst = None
        self.elapsed_time = {}
    
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

    def save_df_day(self):
        FULL_DATASET_PATH = DATA_PATH / 'tratados' / str(self.pct)
        self.df_day.to_parquet(FULL_DATASET_PATH / "df_day.parquet", index = False)

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

    def _inputeWithKNNWithGranufill(self, df_greater: pd.DataFrame,df_lesser: pd.DataFrame,gran_diff: int ):
        df_greater = df_greater.reset_index(drop=True)
        df_lesser = df_lesser.reset_index(drop=True)
        return knn_with_granufill(df_greater, df_lesser, merging_features = ["time", "id_institution"], target_feature = "n_bytes", gran_diff = gran_diff)
    
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
        
        df_.to_parquet(DATA_PATH / "tratados" / str(self.pct) / method / f"df_{granularity}.parquet", index = False)
        print(60*"="+"\n" +  method + "\n" + 60*"="+"\n")
        print(granularity + ":   ", df_["n_bytes"].isna().sum())
        print()


    def runFilling(self):
        self.countTimeFilling("granufill", "hour", self._inputeWithGranularity, self.df_day, self.df_hour, 24)
        df_hour = self._inputeWithGranularity(self.df_day, self.df_hour, 24)   
        self.countTimeFilling("granufill", "10min", self._inputeWithGranularity, df_hour, self.df_10min, 24*6)
        del df_hour

        self.countTimeFilling("knn_with_granufill", "hour", self._inputeWithKNNWithGranufill, self.df_day, self.df_hour, 24)
        df_hour = self._inputeWithGranularity(self.df_day, self.df_hour, 24)   
        self.countTimeFilling("knn_with_granufill", "10min", self._inputeWithKNNWithGranufill, df_hour, self.df_10min, 24*6)
        del df_hour
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
        

    
        
