"""Avalia os 4 novos métodos de imputação (ffill, ewma, seasonal, pchip) no dado
de 10min e mescla os resultados em ``resultados/imputacao_metrics.json`` + ``.csv``.

Reusa o mesmo ground truth determinístico (seed=42) de ``04. avaliar_imputacoes.py``,
então não é necessário re-rodar as imputações STL (caras) já calculadas.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TRATADOS = ROOT / "dados" / "tratados"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("av04", SRC / "04. avaliar_imputacoes.py")
av04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(av04)

timeInputer = av04.timeInputer
create_ground_truth = av04.create_ground_truth
evaluate_method = av04.evaluate_method
PERCENTAGES = av04.PERCENTAGES
LABELS = av04.LABELS

NEW_METHODS = {
    "ffill": {
        "hour": lambda p: p._inputeWithFFill(p.df_hour),
        "10min": lambda p: p._inputeWithFFill(p.df_10min),
    },
    "ewma": {
        "hour": lambda p: p._inputeWithEWMA(p.df_hour, 24),
        "10min": lambda p: p._inputeWithEWMA(p.df_10min, 144),
    },
    "seasonal": {
        "hour": lambda p: p._inputeWithSeasonal(p.df_hour, 24),
        "10min": lambda p: p._inputeWithSeasonal(p.df_10min, 144),
    },
    "pchip": {
        "hour": lambda p: p._inputeWithPchip(p.df_hour),
        "10min": lambda p: p._inputeWithPchip(p.df_10min),
    },
}


def main():
    processor = timeInputer(0.2)
    processor.filter_inst()
    processor.insertTimeRange()
    clean_hour = processor.df_hour
    clean_10min = processor.df_10min

    json_path = ROOT / "resultados" / "imputacao_metrics.json"
    results = json.load(open(json_path, encoding="utf-8"))

    for pct in PERCENTAGES:
        label = LABELS[pct]
        processor.df_hour = clean_hour
        processor.df_10min = clean_10min
        processor.pct = label

        df_10min_nan, gt = create_ground_truth(clean_10min, pct, seed=42)
        processor.df_10min = df_10min_nan
        gt["time"] = pd.to_datetime(gt["time"])

        for name, funcs in NEW_METHODS.items():
            folder = TRATADOS / label / name
            folder.mkdir(parents=True, exist_ok=True)

            hour_out = funcs["hour"](processor)
            pl.from_pandas(hour_out).write_parquet(folder / "df_hour.parquet")

            out = funcs["10min"](processor)
            pl.from_pandas(out).write_parquet(folder / "df_10min.parquet")
            out["time"] = pd.to_datetime(out["time"])

            m = evaluate_method(gt, out)
            results[label]["methods"][name] = m
            print(f"{label} {name:9s} rmse={m['rmse']:.3e} mae={m['mae']:.3e} r2={m['r2']:.4f}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = []
    for label, res in results.items():
        for method, m in res["methods"].items():
            rows.append({"pct": label, "method": method,
                         "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "n": m["n"]})
    pd.DataFrame(rows).to_csv(ROOT / "resultados" / "imputacao_metrics.csv", index=False)
    print(f"Resultados atualizados em {json_path} e .csv")


if __name__ == "__main__":
    main()
