import pandas as pd
import polars as pl
pl.enable_string_cache()
pd.set_option('future.no_silent_downcasting', True)
import glob
import os
import joblib
from funcoes_de_tratamento import *

df_day = read_data('../../data/institutions/agg_1_day')
df_hour = read_data('../../data/institutions/agg_1_hour')
df_10min = read_data('../../data/institutions/agg_10_minutes')

class timeInputer:
    def __init__(self):
        self.df_day = read_data('../../data/institutions/agg_1_day')
        self.df_hour = read_data('../../data/institutions/agg_1_hour')
        self.df_10min = read_data('../../data/institutions/agg_10_minutes')
        self.inst = None
    
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

    def _insertTimeRange(self):
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


                