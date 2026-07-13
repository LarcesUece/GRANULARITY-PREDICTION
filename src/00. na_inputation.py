import pandas as pd
import polars as pl
pl.enable_string_cache()
pd.set_option('future.no_silent_downcasting', True)
import glob
import os
import joblib
from funcoes_de_tratamento import *
from time import perf_counter
import json

df_day = read_data('../../data/institutions/agg_1_day')
df_hour = read_data('../../data/institutions/agg_1_hour')
df_10min = read_data('../../data/institutions/agg_10_minutes')

class timeInputer:
    def __init__(self):
        self.df_day = read_data('../../data/institutions/agg_1_day')
        self.df_hour = read_data('../../data/institutions/agg_1_hour')
        self.df_10min = read_data('../../data/institutions/agg_10_minutes')
        self.inst = None
        self.elapsed_time = {}
    
    def filter_inst(self):
        inst = []

        for i in df_day["id_institution"].unique():
            instituicao = df_day[df_day["id_institution"]==i]
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
                out += 1


            if serve:
                inst.append(i)


        self.df_day = self.df_day[self.df_day["id_institution"].isin(inst)]
        self.df_hour = self.df_hour[self.df_hour["id_institution"].isin(inst)]
        self.df_10min = self.df_10min[self.df_10min["id_institution"].isin(inst)]
        self.inst = [int(c) for c in inst]
        joblib.dump(inst, "inst.joblib")
    
    def _merge_id_date(self, df: pd.DataFrame, time: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["id_time"] = df["id_time"].astype(int) 
        df = df.merge(time, on="id_time", how="left")
        df["time"] = pd.to_datetime(df["time"])

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
        self.df_day = self._merge_id_date(self.df_day, pd.read_csv('../data/times/times_1_day.csv'))
        self.df_hour = self._merge_id_date(self.df_hour, pd.read_csv('../data/times/times_1_hour.csv'))
        self.df_10min = self._merge_id_date(self.df_10min, pd.read_csv('../data/times/times_1_10min.csv'))

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
        self.df_day = pd.concat(dfs)
        dfs = []
        for id in self.inst:
            df_h = self.df_hour[self.df_hour["id_institution"] ==  id].copy()
            df_h = df_h.merge(series_hour, on = "time", how = "right")              
            df_h["id_institution"] =df_h["id_institution"].fillna(id).astype("int")
            dfs.append(df_h)
        self.df_hour = pd.concat(dfs)
        dfs = []
        for id in self.inst:
            df_m = self.df_10min[self.df_10min["id_institution"] ==  id].copy()
            df_m = df_m.merge(series_10min, on = "time", how = "right")
            df_m["id_institution"] = df_m["id_institution"].fillna(id).astype("int")
            dfs.append(df_m)
        self.df_10min = pd.concat(dfs)


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
        df_greater = df_greater.copy()
        df_lesser = df_lesser.copy()
        return granufill(df_greater, df_lesser, merging_features = ["time", "id_institution"], target_feature = "n_bytes", gran_diff = gran_diff)


    def _inputeWithCubic(self, df_: pd.DataFrame) -> pd.DataFrame:
        df = df_.copy()
        for id in self.inst:
            filtro = df["id_institution"] == id
            df.loc[filtro, "n_bytes"] = cubic_fill_missing(df.loc[filtro, "n_bytes"])
        return df
    def countTimeFilling(self, method, granularity, func,*args, **kwargs):

        if method not in self.elapsed_time.keys():
            self.elapsed_time[method] = {}

        start = perf_counter()
        df_ = func(*args, **kwargs)
        end = perf_counter()
        self.elapsed_time[method][granularity] = end - start
        
        df_.to_parquet(f"../data/tratados_stored/{method}/df_{granularity}.parquet", index = False)


    def runFilling(self):
        self.countTimeFilling("granufill", "hour", self._inputeWithGranularity, self.df_day, self.df_hour, 24)
        self.countTimeFilling("granufill", "10min", self._inputeWithGranularity, self.df_hour, self.df_10min, 24*6)
        self.countTimeFilling("moving_average", "hour", self._inputeWithMovingAverage, self.df_hour, 24)
        self.countTimeFilling("moving_average", "10min", self._inputeWithMovingAverage, self.df_10min, 24*6)    
        self.countTimeFilling("moving_median", "hour", self._inputeWithMovingMedian, self.df_hour, 24)
        self.countTimeFilling("moving_median", "10min", self._inputeWithMovingMedian, self.df_10min, 24*6)
        self.countTimeFilling("knn", "hour", self._inputeWithKNN, self.df_hour, 24)
        self.countTimeFilling("knn", "10min", self._inputeWithKNN, self.df_10min, 144)
        self.countTimeFilling("cubic", "hour", self._inputeWithCubic, self.df_hour)
        self.countTimeFilling("cubic", "10min", self._inputeWithCubic, self.df_10min)
        
        #self.countTimeFilling("svd", "hour", self._inputeWithSVD, self.df_hour, 24)
        #self.countTimeFilling("svd", "10min", self._inputeWithSVD, self.df_10min, 144)

        with open("../data/tratados/elapsed_time.json", "w") as f:
            json.dump(self.elapsed_time, f, indent=4)
        print("tempo de execução salvo em ../data/tratados_stored/elapsed_time.json")

    
    def runAll(self):
        print("iniciando...")
        
        print("filtrando dados...")
        self.filter_inst()

        print("inserindo lacunas...")
        self.insertTimeRange()

        print("rodando os metodos...")
        self.runFilling()

        print("feito!")
        print(self.elapsed_time)
        

if __name__ == "__main__":
    processor = timeInputer()
    processor.runAll()
        

    
        
