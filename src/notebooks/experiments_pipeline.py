import argparse
import gc
import shutil
import sys
from pathlib import Path
from typing import List

# ensure local notebooks folder imports work when script run from repo
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import funcoes_de_avaliacao as f_eval

# local helper modules (from the notebooks folder)
import funcoes_de_treinamento as f_trein
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using PyTorch device:", device)
if device.type == "cuda":
    print("CUDA GPU:", torch.cuda.get_device_name(device))


BASE = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR = BASE / "experiments_results"


def ensure(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _require_gpu_dataframe_libs():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this pipeline. Refusing to load/process data in CPU RAM."
        )
    try:
        import cudf
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "cuDF and CuPy are required to keep the tabular data on GPU. "
            "Install RAPIDS/cuDF for your CUDA version before running this pipeline."
        ) from exc
    return cudf, cp


def _clear_accelerator_memory(cp=None):
    gc.collect()
    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except RuntimeError:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def _is_cuda_oom(exc: BaseException):
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "cuda",
            "out_of_memory",
            "out of memory",
            "cudaerrormemoryallocation",
            "std::bad_alloc",
        )
    )


def _window_sizes(window: str, windows: List[str]):
    if window == windows[0]:
        return 7, 1, 7 * 24, 1 * 24, 7 * 24 * 6, 1 * 24 * 6
    if window == windows[1]:
        return 7, 2, 7 * 24, 2 * 24, 7 * 24 * 6, 2 * 24 * 6
    if window == windows[2]:
        return 14, 1, 14 * 24, 1 * 24, 14 * 24 * 6, 1 * 24 * 6
    if window == windows[3]:
        return 14, 3, 14 * 24, 3 * 24, 14 * 24 * 6, 3 * 24 * 6
    return 30, 7, 30 * 24, 7 * 24, 30 * 24 * 6, 7 * 24 * 6


def _make_minmax_scaler(data_min: float, data_max: float, n_samples: int):
    data_range = data_max - data_min
    scale = 1.0 if data_range == 0 else 1.0 / data_range

    scaler = MinMaxScaler()
    scaler.feature_range = (0, 1)
    scaler.copy = True
    scaler.clip = False
    scaler.n_features_in_ = 1
    scaler.n_samples_seen_ = n_samples
    scaler.data_min_ = np.array([data_min], dtype=np.float32)
    scaler.data_max_ = np.array([data_max], dtype=np.float32)
    scaler.data_range_ = np.array([data_range], dtype=np.float32)
    scaler.scale_ = np.array([scale], dtype=np.float32)
    scaler.min_ = np.array([-data_min * scale], dtype=np.float32)
    return scaler


def _scaled_windows_to_cudf(values, institution_id, inputs, outputs, columns, cudf, cp):
    total_window_size = inputs + outputs
    if values.size < total_window_size:
        return None

    values = cp.ascontiguousarray(values, dtype=cp.float32)
    rows = values.size - total_window_size + 1
    windows = cp.lib.stride_tricks.as_strided(
        values,
        shape=(rows, total_window_size),
        strides=(values.strides[0], values.strides[0]),
    )
    windows = cp.ascontiguousarray(windows)
    ids = cp.full((rows, 1), institution_id, dtype=cp.float32)

    id_columns = columns[: ids.shape[1]]
    window_columns = columns[ids.shape[1] :]

    ids_df = cudf.DataFrame(ids, columns=id_columns)
    windows_df = cudf.DataFrame(windows, columns=window_columns)

    return cudf.concat([ids_df, windows_df], axis=1)
    # return cudf.DataFrame(cp.concatenate((ids, windows), axis=1), columns=columns)


def _parquet_sources(path: Path):
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet part files found in {path}")
        return files
    return [path]


def _parquet_shape(files):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read parquet metadata without loading the "
            "whole dataset into GPU memory."
        ) from exc

    total_rows = 0
    total_columns = None
    for file in files:
        metadata = pq.ParquetFile(str(file)).metadata
        total_rows += metadata.num_rows
        if total_columns is None:
            total_columns = metadata.num_columns
        elif total_columns != metadata.num_columns:
            raise ValueError(f"Parquet part {file} has a different column count.")
    return total_rows, int(total_columns or 0)


def _reset_parquet_dataset(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _write_empty_parquet_part(output_dir: Path, columns, cudf):
    df = cudf.DataFrame({col: cudf.Series([], dtype="float32") for col in columns})
    df.to_parquet(str(output_dir / "part_00000.parquet"), index=False)
    del df
    _clear_accelerator_memory()


def _write_gpu_parquet_part(df, output_dir: Path, part_number: int):
    if df is None:
        return part_number
    df.to_parquet(str(output_dir / f"part_{part_number:05d}.parquet"), index=False)
    del df
    _clear_accelerator_memory()
    return part_number + 1


def _cupy_to_torch(array, cp):
    array = cp.ascontiguousarray(array, dtype=cp.float32)
    return torch.utils.dlpack.from_dlpack(array)


def _read_parquet_part_gpu(file: Path, cudf, cp):
    df = cudf.read_parquet(str(file))
    data = cp.asarray(df.to_cupy(na_value=0), dtype=cp.float32)
    tensor = _cupy_to_torch(data, cp)
    del df, data
    return tensor


def _read_parquet_part_cpu(file: Path):
    import pyarrow.parquet as pq

    table = pq.read_table(str(file))
    array = table.to_pandas().to_numpy(dtype=np.float32, copy=False)
    np.nan_to_num(array, copy=False)
    tensor = torch.as_tensor(array, dtype=torch.float32)
    del table, array
    return tensor


def _load_feature_tensor(path: Path, inputs: int, target_device: torch.device):
    cudf, cp = _require_gpu_dataframe_libs()
    files = _parquet_sources(path)
    total_rows, total_columns = _parquet_shape(files)
    tensor = torch.empty(
        (total_rows, total_columns),
        dtype=torch.float32,
        device=target_device,
    )

    offset = 0
    for file in files:
        try:
            part = _read_parquet_part_gpu(file, cudf, cp)
        except Exception as exc:
            if target_device.type == "cpu" and _is_cuda_oom(exc):
                _clear_accelerator_memory(cp)
                part = _read_parquet_part_cpu(file)
            else:
                raise

        rows = part.shape[0]
        tensor[offset : offset + rows].copy_(part, non_blocking=target_device.type == "cuda")
        offset += rows
        del part
        _clear_accelerator_memory(cp)

    return tensor[:, : inputs + 1], tensor[:, inputs + 1 :]


def _load_gpu_feature_target(path: Path, inputs: int):
    try:
        return _load_feature_tensor(path, inputs, torch.device("cuda"))
    except Exception as exc:
        if not _is_cuda_oom(exc):
            raise
        print(
            f"GPU memory was not enough to keep {path.name} fully on VRAM; "
            "falling back to CPU storage with GPU batches."
        )
    _clear_accelerator_memory()
    return _load_feature_tensor(path, inputs, torch.device("cpu"))


def _inverse_scale_by_institution(values, real_values, ids, scalers):
    values_out = torch.empty_like(values)
    real_out = torch.empty_like(real_values)
    for institution_tensor in torch.unique(ids):
        institution_id = int(institution_tensor.detach().cpu().item())
        scaler = scalers[institution_id]
        scale = torch.tensor(
            float(scaler.scale_[0]), dtype=torch.float32, device=values.device
        )
        min_offset = torch.tensor(
            float(scaler.min_[0]), dtype=torch.float32, device=values.device
        )
        mask = ids == institution_tensor
        values_out[mask] = (values[mask] - min_offset) / scale
        real_out[mask] = (real_values[mask] - min_offset) / scale
    return values_out, real_out


def data_processing(window: str, windows: List[str]):
    print("Starting data processing for window:", window)
    cudf, cp = _require_gpu_dataframe_libs()
    inst = [
        int(c)
        for c in joblib.load(str(HERE / "../scalers/instituicoes_validas.joblib"))
    ]

    (
        inputs_day,
        outputs_day,
        inputs_hour,
        outputs_hour,
        inputs_10min,
        outputs_10min,
    ) = _window_sizes(window, windows)

    # target directories
    tdir = EXPERIMENTS_DIR / "Tabelas_criadas" / window
    ensure(tdir)
    scalers_dir = EXPERIMENTS_DIR / "scalers"
    ensure(scalers_dir)

    # Helper to process a resolution
    def process_resolution(input_path, inputs, outputs, name, scalers_fname):
        df = cudf.read_parquet(
            str(input_path),
            columns=["id_institution", "n_bytes"],
        )
        df["id_institution"] = df["id_institution"].astype("int64")
        df["n_bytes"] = df["n_bytes"].fillna(0).astype("float32")

        x_cols = [f"x_{j}" for j in range(inputs)]
        y_cols = [f"y_{o}" for o in range(outputs)]
        columns = ["id_institution"] + x_cols + y_cols
        scalers = {}

        train_path = tdir / f"treino_{name}.parquet"
        test_path = tdir / f"teste_{name}.parquet"
        val_path = tdir / f"val_{name}.parquet"
        for output_path in (train_path, test_path, val_path):
            _reset_parquet_dataset(output_path)

        part_numbers = {"train": 0, "test": 0, "val": 0}

        def write_split(raw_values, output_dir, split_name):
            if raw_values.size < inputs + outputs:
                return

            scaled_values = raw_values * scale + min_offset
            part = _scaled_windows_to_cudf(
                scaled_values,
                i,
                inputs,
                outputs,
                columns,
                cudf,
                cp,
            )
            part_numbers[split_name] = _write_gpu_parquet_part(
                part,
                output_dir,
                part_numbers[split_name],
            )
            del scaled_values, part
            _clear_accelerator_memory(cp)

        for i in inst:
            series = df.loc[df["id_institution"] == i, "n_bytes"]
            tamanho = len(series)
            if tamanho <= 10 * (inputs + outputs):
                print(f"  {i}: dataset pequeno ({tamanho})")
                continue

            values = cp.asarray(series.to_cupy(na_value=0), dtype=cp.float32)
            train_end = int(0.6 * tamanho)
            val_end = int(0.7 * tamanho)
            train_values = values[:train_end]
            val_values = values[train_end:val_end]
            test_values = values[val_end:]

            train_min = cp.min(train_values)
            train_max = cp.max(train_values)
            data_range = train_max - train_min
            scale = cp.where(
                data_range == 0, cp.float32(1.0), cp.float32(1.0) / data_range
            )
            min_offset = -train_min * scale

            scalers[i] = _make_minmax_scaler(
                float(train_min.get()),
                float(train_max.get()),
                int(train_values.size),
            )

            write_split(train_values, train_path, "train")
            write_split(val_values, val_path, "val")
            write_split(test_values, test_path, "test")

            del (
                series,
                values,
                train_values,
                val_values,
                test_values,
                train_min,
                train_max,
                data_range,
                scale,
                min_offset,
            )
            _clear_accelerator_memory(cp)

        if part_numbers["train"] == 0:
            _write_empty_parquet_part(train_path, columns, cudf)
        if part_numbers["test"] == 0:
            _write_empty_parquet_part(test_path, columns, cudf)
        if part_numbers["val"] == 0:
            _write_empty_parquet_part(val_path, columns, cudf)

        joblib.dump(scalers, scalers_dir / scalers_fname)

        del df
        _clear_accelerator_memory(cp)

    process_resolution(
        BASE / "data" / "tratados" / "df_day.parquet",
        inputs_day,
        outputs_day,
        "dia",
        "scalers_day.joblib",
    )
    process_resolution(
        BASE / "data" / "tratados" / "df_hour.parquet",
        inputs_hour,
        outputs_hour,
        "hora",
        "scalers_hour.joblib",
    )
    process_resolution(
        BASE / "data" / "tratados" / "df_10min.parquet",
        inputs_10min,
        outputs_10min,
        "10min",
        "scalers_10min.joblib",
    )

    print("Data processing finished. Tables and scalers saved to", tdir)


def train_models(window: str, windows: List[str]):
    print("Starting training for window:", window)
    (
        inputs_day,
        outputs_day,
        inputs_hour,
        outputs_hour,
        inputs_10min,
        outputs_10min,
    ) = _window_sizes(window, windows)

    tdir = EXPERIMENTS_DIR / "Tabelas_criadas" / window
    modelos_dir = EXPERIMENTS_DIR / "MODELOS" / window
    ensure(modelos_dir)

    def train_granularity(train_path, val_path, inputs, outputs, dimensao, jobs):
        print("Loading training tensors for:", dimensao)
        X_train, y_train = _load_gpu_feature_target(train_path, inputs)
        X_val, y_val = _load_gpu_feature_target(val_path, inputs)
        _clear_accelerator_memory()
        try:
            for optimizer, model_path in jobs:
                optimizer(
                    X_train[:, 1:],
                    y_train,
                    X_val[:, 1:],
                    y_val,
                    dimensao,
                    inputs,
                    outputs,
                    path_modelo=str(modelos_dir / model_path),
                )
                _clear_accelerator_memory()
        finally:
            del X_train, y_train, X_val, y_val
            _clear_accelerator_memory()

    # Same optimization calls as before, but each granularity is released before
    # the next one is loaded.
    train_granularity(
        tdir / "treino_10min.parquet",
        tdir / "val_10min.parquet",
        inputs_10min,
        outputs_10min,
        "10minutos",
        (
            (f_trein.otimizar_GRU, "gru_10min_otimizada_sem_id.pt"),
            (f_trein.otimizar_LSTM, "lstm_10min_otimizada_sem_id.pt"),
            (f_trein.otimizar_RNN, "rnn_10min_otimizada_sem_id.pt"),
        ),
    )
    train_granularity(
        tdir / "treino_dia.parquet",
        tdir / "val_dia.parquet",
        inputs_day,
        outputs_day,
        "diária",
        (
            (f_trein.otimizar_GRU, "gru_diaria_otimizada_sem_id.pt"),
            (f_trein.otimizar_RNN, "rnn_diaria_otimizada_sem_id.pt"),
            (f_trein.otimizar_LSTM, "lstm_diaria_otimizada_sem_id.pt"),
        ),
    )
    train_granularity(
        tdir / "treino_hora.parquet",
        tdir / "val_hora.parquet",
        inputs_hour,
        outputs_hour,
        "horária",
        (
            (f_trein.otimizar_GRU, "gru_horaria_otimizada_sem_id.pt"),
            (f_trein.otimizar_LSTM, "lstm_horaria_otimizada_sem_id.pt"),
            (f_trein.otimizar_RNN, "rnn_horaria_otimizada_sem_id.pt"),
        ),
    )

    print("Training finished. Models saved to", modelos_dir)


def evaluate_models(window: str, windows: List[str]):
    print("Starting evaluation for window:", window)
    (
        inputs_day,
        outputs_day,
        inputs_hour,
        outputs_hour,
        inputs_10min,
        outputs_10min,
    ) = _window_sizes(window, windows)

    tdir = EXPERIMENTS_DIR / "Tabelas_criadas" / window
    modelos_dir = EXPERIMENTS_DIR / "MODELOS" / window
    plots_dir = EXPERIMENTS_DIR / "plots"
    ensure(plots_dir)

    # load models
    model_keys = ["GRU", "LSTM", "RNN"]
    modelos = {
        "GRU": {
            "Diario": f_trein.load_torch_model(
                modelos_dir / "gru_diaria_otimizada_sem_id.pt"
            ),
            "Horario": f_trein.load_torch_model(
                modelos_dir / "gru_horaria_otimizada_sem_id.pt"
            ),
            "10minutos": f_trein.load_torch_model(
                modelos_dir / "gru_10min_otimizada_sem_id.pt"
            ),
        },
        "LSTM": {
            "Diario": f_trein.load_torch_model(
                modelos_dir / "lstm_diaria_otimizada_sem_id.pt"
            ),
            "Horario": f_trein.load_torch_model(
                modelos_dir / "lstm_horaria_otimizada_sem_id.pt"
            ),
            "10minutos": f_trein.load_torch_model(
                modelos_dir / "lstm_10min_otimizada_sem_id.pt"
            ),
        },
        "RNN": {
            "Diario": f_trein.load_torch_model(
                modelos_dir / "rnn_diaria_otimizada_sem_id.pt"
            ),
            "Horario": f_trein.load_torch_model(
                modelos_dir / "rnn_horaria_otimizada_sem_id.pt"
            ),
            "10minutos": f_trein.load_torch_model(
                modelos_dir / "rnn_10min_otimizada_sem_id.pt"
            ),
        },
    }

    # The notebooks use treino_*.parquet as test matrices in the evaluation flow.
    X_test_d, y_test_d = _load_gpu_feature_target(
        tdir / "treino_dia.parquet", inputs_day
    )
    X_test_h, y_test_h = _load_gpu_feature_target(
        tdir / "treino_hora.parquet", inputs_hour
    )
    X_test_m, y_test_m = _load_gpu_feature_target(
        tdir / "treino_10min.parquet", inputs_10min
    )

    gc.collect()

    # compute metrics using the helper from funcoes_de_avaliacao
    metricas = {}
    for model in model_keys:
        print("Evaluating model:", model)
        metricas[model] = f_eval.comparar_desempeho_granularidade(
            X_test_d[:, 1:].reshape((X_test_d.shape[0], 1, X_test_d.shape[1] - 1)),
            X_test_h[:, 1:].reshape((X_test_h.shape[0], 1, X_test_h.shape[1] - 1)),
            X_test_m[:, 1:].reshape((X_test_m.shape[0], 1, X_test_m.shape[1] - 1)),
            y_test_d,
            y_test_h,
            y_test_m,
            modelos[model]["Diario"],
            modelos[model]["Horario"],
            modelos[model]["10minutos"],
        )
        _clear_accelerator_memory()

    # Save metrics and plots
    import json

    metrics_file = EXPERIMENTS_DIR / f"metrics_{window}.json"
    with open(metrics_file, "w") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    # Basic bar plot as in notebook
    modelos_list = list(metricas.keys())
    metricas_plot = ["RMSE", "MAE", "NRMSE"]
    cores = {
        "RMSE": {"Diario": "#155fe9", "Horario": "#ff3f0f", "10minutos": "#73ff00"},
        "MAE": {"Diario": "#1f77b4", "Horario": "#ff7f0e", "10minutos": "#2ca02c"},
        "NRMSE": {"Diario": "#aec7e8", "Horario": "#ffbb78", "10minutos": "#98df8a"},
    }

    x = np.arange(len(modelos_list))
    largura_barra = 0.09
    offsets = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(14, 7))
    idx_offset = 0
    for metrica in metricas_plot:
        for granularidade in ["Diario", "Horario", "10minutos"]:
            valores = [
                metricas[modelo][granularidade][metrica] for modelo in modelos_list
            ]
            posicao = x + (offsets[idx_offset] * largura_barra)
            nome_gran = (
                "Diária"
                if granularidade == "Diario"
                else ("Horária" if granularidade == "Horario" else "10 Minutos")
            )
            label = f"{metrica} ({nome_gran})"
            cor = cores[metrica][granularidade]
            barras = ax.bar(posicao, valores, largura_barra, label=label, color=cor)
            ax.bar_label(barras, padding=3, fmt="%.4f", fontsize=8, rotation=90)
            idx_offset += 1

    ax.set_ylabel("Valor do Erro")
    ax.set_title(f"Comparativo de Desempenho - {window}")
    ax.set_xticks(x)
    ax.set_xticklabels(modelos_list)
    ax.legend(
        title="Métrica e Granularidade", bbox_to_anchor=(1.02, 1), loc="upper left"
    )
    plt.tight_layout()
    plt.savefig(plots_dir / f"comparativo_{window}.png")
    plt.close(fig)
    del (
        X_test_d,
        y_test_d,
        X_test_h,
        y_test_h,
        X_test_m,
        y_test_m,
        modelos,
    )
    _clear_accelerator_memory()

    print("Evaluation finished. Metrics saved to", metrics_file, "plots to", plots_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments pipeline: data -> train -> evaluate"
    )
    parser.add_argument("--window", default="14-3", help="Window to run (default 14-3)")
    parser.add_argument(
        "--stages",
        default="all",
        help="stages: data,train,evaluate or comma-separated or all",
    )
    args = parser.parse_args()
    windows = ["7-1", "7-2", "14-1", "14-3", "30-7"]
    w = args.window
    if w not in windows:
        raise SystemExit(f"window must be one of {windows}")

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    stages = (
        args.stages.split(",")
        if args.stages != "all"
        else ["data", "train", "evaluate"]
    )

    ensure(EXPERIMENTS_DIR)

    if "data" in stages:
        data_processing(w, windows)
    if "train" in stages:
        train_models(w, windows)
    if "evaluate" in stages:
        evaluate_models(w, windows)


if __name__ == "__main__":
    main()
