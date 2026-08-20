"""Rebuild ``dados/tratados/df_*.parquet`` from the raw institution CSVs.

Produces one parquet per granularity with columns ``[time, id_institution, n_bytes]``,
where ``time`` is UTC and floored to the granularity (10min / hour / day). This is the
format consumed by the notebooks and ``01. prediction_pipeline.py``.

Run from anywhere::

    python src/preparar_tratados.py
"""
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "dados"
INSTITUTIONS_PATH = DATA_PATH / "institutions"
TIMES_PATH = DATA_PATH / "times"
TRATADOS_PATH = DATA_PATH / "tratados"

GRANULARITIES = {
    "day": ("agg_1_day", "times_1_day.csv"),
    "hour": ("agg_1_hour", "times_1_hour.csv"),
    "10min": ("agg_10_minutes", "times_10_minutes.csv"),
}


def merge_time(df: pd.DataFrame, times: pd.DataFrame) -> pd.DataFrame:
    """Merge ``id_time`` -> ``time`` (UTC, floored to the granularity).

    Mirrors ``_merge_id_date`` from ``00. na_inputation.py``.
    """
    df = df.copy()
    df["id_time"] = df["id_time"].astype(int)
    df = df.merge(times, on="id_time", how="left")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["time"] = pd.to_datetime(pd.DataFrame({
        "year": df["time"].dt.year.astype(int),
        "month": df["time"].dt.month.astype(int),
        "day": df["time"].dt.day.astype(int),
        "hour": df["time"].dt.hour.astype(int),
        "minute": (round(df["time"].dt.minute.astype(int) / 10)) * 10,
        "second": 0,
    }))
    df = df.sort_values(["time", "id_institution"])
    df.drop(columns=["id_time"], inplace=True)
    df["id_institution"] = df["id_institution"].astype(int)
    return df.reset_index(drop=True)


def build(granularity: str) -> pd.DataFrame:
    inst_dir, times_file = GRANULARITIES[granularity]
    times = pd.read_csv(TIMES_PATH / times_file)

    frames = []
    for f in sorted((INSTITUTIONS_PATH / inst_dir).glob("*.csv")):
        df = pd.read_csv(f)[["id_time", "n_bytes"]]
        df["id_institution"] = int(f.stem)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    df["id_time"] = pd.to_numeric(df["id_time"], errors="coerce").astype(int)
    df["n_bytes"] = pd.to_numeric(df["n_bytes"], errors="coerce")
    df["id_institution"] = df["id_institution"].astype(int)
    return merge_time(df, times)[["time", "id_institution", "n_bytes"]]


def main() -> None:
    TRATADOS_PATH.mkdir(parents=True, exist_ok=True)
    for g in GRANULARITIES:
        out = build(g)
        target = TRATADOS_PATH / f"df_{g}.parquet"
        pl.from_pandas(out).write_parquet(target)
        print(f"{g:5s}: {out.shape[0]:>10,} rows  {list(out.columns)}  -> {target.name}")


if __name__ == "__main__":
    main()
