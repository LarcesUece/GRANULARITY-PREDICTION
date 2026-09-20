"""Avalia a eficácia das imputações no dado de 10min usando ground truth.

Para cada percentual (5, 10, 20, 25%):
  1. Encontra o maior intervalo sem falhas (maior trecho contíguo de dados
     válidos) de cada instituição na série de 10min.
  2. Remove o percentual de pontos dentro desse intervalo, guardando os valores
     reais como ground truth.
  3. Roda todas as imputações e salva os parquets em ``dados/tratados/<pct>/``.
  4. Calcula RMSE, MAE e R2 comparando imputado vs ground truth (10min apenas).

Saídas:
  - dados/tratados/<pct>/ground_truth.parquet
  - resultados/imputacao_metrics.json
  - resultados/imputacao_metrics.csv
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
TRATADOS_PATH = ROOT / "dados" / "tratados"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("na_inputation", SRC / "00. na_inputation.py")
na = importlib.util.module_from_spec(spec)
spec.loader.exec_module(na)
timeInputer = na.timeInputer

PERCENTAGES = [0.05, 0.10, 0.20, 0.25]
LABELS = {0.05: "0.05", 0.10: "0.10", 0.20: "0.20", 0.25: "0.25"}


def largest_valid_run(mask: np.ndarray):
    """Retorna (inicio, comprimento) do maior trecho contíguo de True."""
    best_start, best_len = 0, 0
    run_start = None
    for i in range(len(mask) + 1):
        is_valid = i < len(mask) and mask[i]
        if is_valid and run_start is None:
            run_start = i
        elif not is_valid and run_start is not None:
            if i - run_start > best_len:
                best_len = i - run_start
                best_start = run_start
            run_start = None
    return best_start, best_len


def create_ground_truth(df_10min: pd.DataFrame, pct: float, seed: int):
    """Remove pct% dos pontos dentro do maior intervalo sem falhas de cada instituição.

    Retorna (df_com_nulos, ground_truth).
    """
    df = df_10min.copy()
    rng = np.random.default_rng(seed)
    gt_rows = []

    for inst in df["id_institution"].unique():
        m = df["id_institution"] == inst
        block_idx = df.index[m]
        valid = df.loc[m, "n_bytes"].notna().to_numpy()
        if not valid.any():
            continue

        start, length = largest_valid_run(valid)
        if length == 0:
            continue

        seg = block_idx[start:start + length]
        n_remove = max(1, int(round(len(seg) * pct)))
        chosen = rng.choice(seg, size=n_remove, replace=False)

        for c in chosen:
            gt_rows.append({
                "id_institution": int(df.loc[c, "id_institution"]),
                "time": df.loc[c, "time"],
                "n_bytes_true": float(df.loc[c, "n_bytes"]),
            })
        df.loc[chosen, "n_bytes"] = np.nan

    return df, pd.DataFrame(gt_rows)


def evaluate_method(gt: pd.DataFrame, imputed: pd.DataFrame) -> dict:
    """RMSE, MAE e R2 entre valores imputados e o ground truth."""
    merged = gt.merge(imputed, on=["id_institution", "time"], how="left")
    true = merged["n_bytes_true"].to_numpy(dtype=float)
    pred = merged["n_bytes"].to_numpy(dtype=float)
    mask = np.isfinite(true) & np.isfinite(pred)
    true, pred = true[mask], pred[mask]
    if len(true) == 0:
        return {"rmse": None, "mae": None, "r2": None, "n": 0}
    return {
        "rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "mae": float(np.mean(np.abs(true - pred))),
        "r2": float(r2_score(true, pred)),
        "n": int(len(true)),
    }


def main():
    processor = timeInputer(0.2)
    processor.filter_inst()
    processor.insertTimeRange()

    # Snapshot dos dados limpos (com as lacunas originais, sem remoção artificial).
    clean_hour = processor.df_hour
    clean_10min = processor.df_10min

    all_results = {}

    for pct in PERCENTAGES:
        label = LABELS[pct]
        print(f"\n{'=' * 60}\n{label} ({pct:.0%})\n{'=' * 60}")

        processor.df_hour = clean_hour
        processor.df_10min = clean_10min
        processor.pct = label  # pasta de saída em dados/tratados/

        df_10min_nan, gt = create_ground_truth(processor.df_10min, pct, seed=42)
        processor.df_10min = df_10min_nan

        gt_dir = TRATADOS_PATH / label
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_path = gt_dir / "ground_truth.parquet"
        pl.from_pandas(gt).write_parquet(gt_path)
        print(f"ground truth: {len(gt)} pontos removidos -> {gt_path}")

        processor.runFilling()

        gt["time"] = pd.to_datetime(gt["time"])
        method_results = {}
        for method_dir in sorted((TRATADOS_PATH / label).iterdir()):
            if not method_dir.is_dir():
                continue
            f = method_dir / "df_10min.parquet"
            if not f.exists():
                continue
            imp = pl.read_parquet(f).to_pandas()
            imp["time"] = pd.to_datetime(imp["time"])
            method_results[method_dir.name] = evaluate_method(gt, imp)

        all_results[label] = {
            "n_ground_truth": int(len(gt)),
            "methods": method_results,
        }
        for m, r in method_results.items():
            print(f"  {m:24s} rmse={r['rmse']!s:>20} mae={r['mae']!s:>20} r2={r['r2']!s}")

    out_dir = ROOT / "resultados"
    out_dir.mkdir(exist_ok=True)

    json_path = out_dir / "imputacao_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    rows = []
    for label, res in all_results.items():
        for method, m in res["methods"].items():
            rows.append({"pct": label, "method": method,
                         "rmse": m["rmse"], "mae": m["mae"], "r2": m["r2"], "n": m["n"]})
    csv_path = out_dir / "imputacao_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    print(f"\nResultados salvos em {json_path} e {csv_path}")


if __name__ == "__main__":
    main()
