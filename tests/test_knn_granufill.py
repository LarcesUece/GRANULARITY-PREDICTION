"""Testes para a função knn_with_granufill.

Valida:
  1. Preenchimento de NaNs em um cenário sintético conhecido (erro baixo).
  2. Short-circuit: sem nulos -> retorna cópia idêntica.
  3. Short-circuit: só nulos -> retorna cópia idêntica.
  4. Dados reais (uma instituição): injeta nulos, preenche e mede RMSE/NRMSE.

Os resultados são salvos em JSON (resultados/knn_granufill_results.json).
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

spec = importlib.util.spec_from_file_location("na_inputation", SRC / "00. na_inputation.py")
na = importlib.util.module_from_spec(spec)
spec.loader.exec_module(na)

knn_with_granufill = na.knn_with_granufill


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _nrmse(y_true, y_pred):
    rng = y_true.max() - y_true.min()
    return float(_rmse(y_true, y_pred) / rng) if rng > 0 else float("nan")


def test_synthetic():
    np.random.seed(42)
    D = 5
    day_times = pd.date_range("2024-01-01", periods=D, freq="D")
    day_vals = np.arange(1, D + 1) * 100.0
    df_greater = pd.DataFrame({"time": day_times, "id_institution": 1, "n_bytes": day_vals})

    hour_times = pd.date_range("2024-01-01", periods=D * 24, freq="h")
    day_idx = np.repeat(np.arange(D), 24)
    hour_pattern = 1 + 0.1 * np.sin(2 * np.pi * np.arange(D * 24) / 24)
    hour_vals = (day_vals[day_idx] / 24.0) * hour_pattern

    df_less = pd.DataFrame({"time": hour_times, "id_institution": 1, "n_bytes": hour_vals})
    missing_idx = [5, 30, 60, 100]

    df_less_nan = df_less.copy()
    df_less_nan.loc[missing_idx, "n_bytes"] = np.nan

    result = knn_with_granufill(
        df_greater, df_less_nan,
        merging_features=["time", "id_institution"],
        target_feature="n_bytes",
        gran_diff=24,
        k=3,
    )

    assert len(result) == len(df_less), "linhas perdidas no preenchimento"
    assert not result["n_bytes"].isna().any(), "ainda há NaNs após o preenchimento"

    filled = result.loc[missing_idx, "n_bytes"].to_numpy(dtype=float)
    true = df_less.loc[missing_idx, "n_bytes"].to_numpy(dtype=float)
    rmse = _rmse(true, filled)
    nrmse = _nrmse(hour_vals, result["n_bytes"].to_numpy(dtype=float))

    # Padrão suave -> o preenchimento deve ficar perto do valor real
    assert rmse < 2.0, f"RMSE sintético alto: {rmse:.4f}"

    return {
        "test": "synthetic",
        "status": "ok",
        "n_rows": int(len(result)),
        "n_filled": len(missing_idx),
        "rmse_filled": rmse,
        "nrmse_series": nrmse,
    }


def test_no_missing_short_circuit():
    df_greater = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq="D"),
                               "id_institution": 1, "n_bytes": [100.0, 200.0]})
    df_less = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=48, freq="h"),
                            "id_institution": 1, "n_bytes": np.arange(48, dtype=float)})

    result = knn_with_granufill(df_greater, df_less, ["time", "id_institution"], "n_bytes", 24, k=3)
    pd.testing.assert_frame_equal(result, df_less)
    return {"test": "no_missing_short_circuit", "status": "ok", "returned_unchanged": True}


def test_all_missing_short_circuit():
    df_greater = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=2, freq="D"),
                               "id_institution": 1, "n_bytes": [100.0, 200.0]})
    df_less = pd.DataFrame({"time": pd.date_range("2024-01-01", periods=48, freq="h"),
                            "id_institution": 1, "n_bytes": np.nan})

    result = knn_with_granufill(df_greater, df_less, ["time", "id_institution"], "n_bytes", 24, k=3)
    assert result["n_bytes"].isna().all(), "série toda nula deveria permanecer intocada"
    return {"test": "all_missing_short_circuit", "status": "ok", "returned_unchanged": True}


def test_real_data():
    df_day = pl.read_parquet(ROOT / "dados" / "tratados" / "df_day.parquet").to_pandas()
    df_hour = pl.read_parquet(ROOT / "dados" / "tratados" / "df_hour.parquet").to_pandas()

    inst_id = int(df_day["id_institution"].unique()[0])
    df_greater = df_day[df_day["id_institution"] == inst_id].reset_index(drop=True)
    df_less = df_hour[df_hour["id_institution"] == inst_id].reset_index(drop=True)

    rng = np.random.default_rng(0)
    n_missing = int(len(df_less) * 0.2)
    missing_idx = rng.choice(len(df_less), size=n_missing, replace=False)

    true_vals = df_less["n_bytes"].to_numpy(dtype=float).copy()
    df_less_nan = df_less.copy()
    df_less_nan.loc[missing_idx, "n_bytes"] = np.nan

    result = knn_with_granufill(
        df_greater, df_less_nan,
        merging_features=["time", "id_institution"],
        target_feature="n_bytes",
        gran_diff=24,
        k=3,
    )

    assert not result["n_bytes"].isna().any(), "ainda há NaNs no caso real"
    filled = result.loc[missing_idx, "n_bytes"].to_numpy(dtype=float)
    true = true_vals[missing_idx]
    rmse = _rmse(true, filled)
    nrmse = _nrmse(true_vals, result["n_bytes"].to_numpy(dtype=float))

    return {
        "test": "real_data",
        "status": "ok",
        "institution": inst_id,
        "n_rows": int(len(df_less)),
        "n_filled": int(n_missing),
        "rmse_filled": rmse,
        "nrmse_series": nrmse,
    }


def main():
    results = []
    for fn in (test_synthetic, test_no_missing_short_circuit, test_all_missing_short_circuit, test_real_data):
        name = fn.__name__
        try:
            results.append(fn())
            print(f"PASS {name}")
        except AssertionError as e:
            results.append({"test": name, "status": "fail", "error": str(e)})
            print(f"FAIL {name}: {e}")

    out_dir = ROOT / "resultados"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "knn_granufill_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Resultados salvos em {out_path}")
    return results


if __name__ == "__main__":
    main()
