"""GPU-oriented prediction pipeline for CESNET24 time-series parquet files.

The pipeline loads treated parquet files from ``tratados/``, builds chronological
sliding windows per institution, trains MLP/RNN/LSTM/GRU models for batch sizes
128 and 256, and evaluates each run with the project evaluation helpers.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = ROOT / "src" / "notebooks"
if str(NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS_DIR))

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

torch = None
nn = None
DataLoader = None
TensorDataset = None
TorchRegressionModel = None
generate_GRU = None
generate_LSTM = None
generate_MLP_model = None
generate_RNN = None
save_torch_model = None
_avaliar_modelo_em_batches = None
avaliar_modelo = None


class MissingDependencyError(RuntimeError):
    """Raised when the active Python environment cannot run model training."""


class CudaConfigurationError(RuntimeError):
    """Raised when CUDA is required but PyTorch cannot use it."""


MODEL_NAMES = ("GRU",)#("MLP", "RNN", "LSTM", "GRU")
BATCH_SIZES = (128, 256)
'''
WINDOWS = {
    "7-1": {"10min": (7 * 24 * 6, 1 * 24 * 6)},
    "7-2": {"10min": (7 * 24 * 6, 2 * 24 * 6)},
    "14-1": {"10min": (14 * 24 * 6, 1 * 24 * 6)},
    "14-3": {"10min": (14 * 24 * 6, 3 * 24 * 6)},
    "30-7": {"10min": (30 * 24 * 6, 7 * 24 * 6)},
}
'''

WINDOWS = {
    "7-1": {"day": (7, 1), "hour": (7 * 24, 1 * 24), "10min": (7 * 24 * 6, 1 * 24 * 6)},
    "7-2": {"day": (7, 2), "hour": (7 * 24, 2 * 24), "10min": (7 * 24 * 6, 2 * 24 * 6)},
    "14-1": {"day": (14, 1), "hour": (14 * 24, 1 * 24), "10min": (14 * 24 * 6, 1 * 24 * 6)},
    "14-3": {"day": (14, 3), "hour": (14 * 24, 3 * 24), "10min": (14 * 24 * 6, 3 * 24 * 6)},
    "30-7": {"day": (30, 7), "hour": (30 * 24, 7 * 24), "10min": (30 * 24 * 6, 7 * 24 * 6)},
}


def ensure_runtime_dependencies() -> None:
    """Import PyTorch and project helpers lazily so ``--help`` still works."""
    global DataLoader
    global TensorDataset
    global TorchRegressionModel
    global _avaliar_modelo_em_batches
    global avaliar_modelo
    global generate_GRU
    global generate_LSTM
    global generate_MLP_model
    global generate_RNN
    global nn
    global save_torch_model
    global torch

    if torch is not None:
        return

    try:
        import torch as torch_module
        from torch import nn as nn_module
        from torch.utils.data import DataLoader as DataLoaderClass
        from torch.utils.data import TensorDataset as TensorDatasetClass
    except ModuleNotFoundError as exc:
        raise MissingDependencyError(
            "Missing PyTorch in the active Python environment. Install the repo "
            "requirements first, preferably in Python 3.11/3.12 with CUDA support: "
            "python -m pip install -r requirements.txt"
        ) from exc

    from funcoes_de_avaliacao import _avaliar_modelo_em_batches as avaliar_batches_func
    from funcoes_de_avaliacao import avaliar_modelo as avaliar_modelo_func
    from funcoes_de_treinamento import TorchRegressionModel as TorchRegressionModelClass
    from funcoes_de_treinamento import generate_GRU as generate_GRU_func
    from funcoes_de_treinamento import generate_LSTM as generate_LSTM_func
    from funcoes_de_treinamento import generate_MLP_model as generate_MLP_model_func
    from funcoes_de_treinamento import generate_RNN as generate_RNN_func
    from funcoes_de_treinamento import save_torch_model as save_torch_model_func

    torch = torch_module
    nn = nn_module
    DataLoader = DataLoaderClass
    TensorDataset = TensorDatasetClass
    TorchRegressionModel = TorchRegressionModelClass
    _avaliar_modelo_em_batches = avaliar_batches_func
    avaliar_modelo = avaliar_modelo_func
    generate_GRU = generate_GRU_func
    generate_LSTM = generate_LSTM_func
    generate_MLP_model = generate_MLP_model_func
    generate_RNN = generate_RNN_func
    save_torch_model = save_torch_model_func


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    lookback: int
    horizon: int
    train_step: int
    val_step: int
    test_step: int


@dataclass
class WindowedDataset:
    X_train_mlp: np.ndarray
    X_val_mlp: np.ndarray
    X_test_mlp: np.ndarray
    X_train_seq: np.ndarray
    X_val_seq: np.ndarray
    X_test_seq: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    spec: DatasetSpec
    rows_loaded: int
    institutions_used: int


def cuda_install_hint() -> str:
    return (
        "This environment can see NVIDIA only if PyTorch was installed with CUDA support. "
        f"Current interpreter: {sys.executable}. Try reinstalling PyTorch with the official "
        "CUDA wheel, for example: python -m pip install --upgrade --force-reinstall torch "
        "--index-url https://download.pytorch.org/whl/cu128"
    )


def configure_torch(cuda_device: int | None = None, allow_cpu: bool = False) -> torch.device:
    """Enable CUDA knobs that improve throughput without changing data scale."""
    ensure_runtime_dependencies()
    torch.set_float32_matmul_precision("high")

    if torch.cuda.is_available():
        if cuda_device is not None:
            torch.cuda.set_device(cuda_device)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    elif allow_cpu:
        device = torch.device("cpu")
    else:
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_build = getattr(torch.version, "cuda", None)
        raise CudaConfigurationError(
            "CUDA is required for this pipeline, but PyTorch cannot use it. "
            f"torch={torch_version}, torch.version.cuda={cuda_build!r}, "
            f"torch.cuda.is_available()={torch.cuda.is_available()}. {cuda_install_hint()} "
            "Pass --allow-cpu only for debugging."
        )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def nvidia_smi_processes() -> list[dict[str, object]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
    except Exception:
        return []

    processes = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        pid, process_name, used_memory = parts[:3]
        try:
            pid_value: int | str = int(pid)
        except ValueError:
            pid_value = pid
        processes.append({"pid": pid_value, "process": process_name, "used_memory_mb": used_memory})
    return processes


def gpu_report(label: str) -> dict[str, object]:
    """Return and print a compact GPU utilization/memory report."""
    ensure_runtime_dependencies()
    report: dict[str, object] = {
        "label": label,
        "pid": os.getpid(),
        "torch": getattr(torch, "__version__", "unknown"),
        "torch_cuda_build": getattr(torch.version, "cuda", None),
        "cuda_available": torch.cuda.is_available(),
    }
    if not torch.cuda.is_available():
        print(f"[GPU] {label}: CUDA unavailable. {json.dumps(report, ensure_ascii=False)}")
        return report

    device_idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_idx)
    report.update(
        {
            "device": torch.cuda.get_device_name(device_idx),
            "allocated_mb": round(torch.cuda.memory_allocated(device_idx) / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(device_idx) / 1024**2, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated(device_idx) / 1024**2, 1),
            "total_mb": round(props.total_memory / 1024**2, 1),
            "capability": f"{props.major}.{props.minor}",
        }
    )

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        first_gpu = output.splitlines()[device_idx].split(",")
        report.update(
            {
                "gpu_util_pct": int(first_gpu[0].strip()),
                "mem_util_pct": int(first_gpu[1].strip()),
                "nvidia_smi_memory": f"{first_gpu[2].strip()}/{first_gpu[3].strip()} MB",
            }
        )
    except Exception:
        report["nvidia_smi"] = "unavailable"

    print(f"[GPU] {label}: {json.dumps(report, ensure_ascii=False)}")
    return report


def certify_gpu_usage(device: torch.device, label: str = "startup") -> dict[str, object]:
    """Run a real CUDA forward/backward pass and print proof that tensors live on GPU."""
    ensure_runtime_dependencies()
    if device.type != "cuda":
        raise RuntimeError(f"GPU certification requires a CUDA device, got {device}.")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device)

    layer = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Linear(1024, 256)).to(device)
    x = torch.randn(2048, 1024, device=device)
    target = torch.randn(2048, 256, device=device)
    optimizer = torch.optim.SGD(layer.parameters(), lr=1e-4)

    optimizer.zero_grad(set_to_none=True)
    with autocast_context(device, enabled=True):
        prediction = layer(x)
        loss = torch.mean((prediction.float() - target) ** 2)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize(device)

    after_allocated = torch.cuda.memory_allocated(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    current_pid = os.getpid()
    processes = nvidia_smi_processes()
    current_processes = [process for process in processes if process.get("pid") == current_pid]
    pid_seen = bool(current_processes)

    certificate = {
        "label": label,
        "pid": current_pid,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "tensor_device": str(prediction.device),
        "model_device": str(next(layer.parameters()).device),
        "loss": float(loss.detach().cpu()),
        "allocated_delta_mb": round((after_allocated - before_allocated) / 1024**2, 1),
        "peak_allocated_mb": round(peak_allocated / 1024**2, 1),
        "pid_seen_in_nvidia_smi": pid_seen,
        "nvidia_smi_process_count": len(processes),
        "nvidia_smi_current_process": current_processes,
    }
    print(f"[GPU-CERTIFIED] {json.dumps(certificate, ensure_ascii=False)}")

    del optimizer, layer, x, target, prediction, loss
    clear_memory()
    return certificate


def assert_model_on_device(wrapper: TorchRegressionModel) -> None:
    parameter = next(wrapper.model.parameters(), None)
    if parameter is None:
        return
    actual = parameter.device
    expected = wrapper.device
    if actual.type != expected.type or (expected.type == "cuda" and actual.index != expected.index):
        raise RuntimeError(
            f"{wrapper.model_type} parameters are on {actual}, but training device is {expected}."
        )


def clear_memory() -> None:
    ensure_runtime_dependencies()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def discover_parquet_files(data_dir: Path, method: str, window_name: str) -> list[DatasetSpec]:
    """Find one parquet file per granularity unless ``method='all'``."""
    if window_name not in WINDOWS:
        raise ValueError(f"Unknown window {window_name!r}. Choose one of: {', '.join(WINDOWS)}")

    all_files = sorted(data_dir.rglob("*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No parquet files found under {data_dir}")
    by_granularity: dict[str, list[Path]] = {window: [] for window in WINDOWS[window_name]}
    #by_granularity: dict[str, list[Path]] = {"day": [], "hour": [], "10min": []}
    for path in all_files:
        stem = path.stem.lower()
        if "10min" in stem or "10_min" in stem:
            by_granularity["10min"].append(path)
        elif "hour" in stem or "hora" in stem:
            by_granularity["hour"].append(path)
        elif "day" in stem or "dia" in stem:
            by_granularity["day"].append(path)

    specs: list[DatasetSpec] = []
    for granularity, paths in by_granularity.items():
        if not paths:
            continue
        selected = select_paths(paths, method)
        lookback, horizon = WINDOWS[window_name][granularity]
        train_step = 5 if granularity == "10min" else 1
        val_step = 5 if granularity == "10min" else 1
        for path in selected:
            suffix = "new_method" if "new_method" in {part.lower() for part in path.parts} else "original"
            name = granularity if method != "all" else f"{granularity}_{suffix}"
            specs.append(
                DatasetSpec(
                    name=name,
                    path=path,
                    lookback=lookback,
                    horizon=horizon,
                    train_step=train_step,
                    val_step=val_step,
                    test_step=1,
                )
            )
    return specs


def select_paths(paths: list[Path], method: str) -> list[Path]:
    if method == "all":
        return paths
    if method == "new_method":
        chosen = [path for path in paths if "new_method" in {part.lower() for part in path.parts}]
        return chosen or paths[:1]
    if method == "original":
        chosen = [path for path in paths if "new_method" not in {part.lower() for part in path.parts}]
        return chosen[:1] or paths[:1]
    if method == "auto":
        chosen = [path for path in paths if "new_method" in {part.lower() for part in path.parts}]
        return chosen[:1] or paths[:1]
    raise ValueError("method must be auto, new_method, original, or all")


def read_parquet_frame(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load parquet through Arrow first, then Polars, then Pandas."""
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns, memory_map=True, pre_buffer=True)
        return table.combine_chunks().to_pandas(split_blocks=True, self_destruct=True)
    except ModuleNotFoundError:
        pass

    try:
        import polars as pl

        scan = pl.scan_parquet(str(path))
        if columns is not None:
            scan = scan.select([col for col in columns if col in scan.collect_schema().names()])
        return scan.collect(streaming=True).to_pandas()
    except ModuleNotFoundError:
        pass

    try:
        return pd.read_parquet(path, columns=columns)
    except ImportError as exc:
        raise RuntimeError(
            "Parquet loading needs one engine installed. Install pyarrow for the "
            "fastest path, or install polars/fastparquet."
        ) from exc


def resolve_columns(df: pd.DataFrame, id_col: str, time_col: str, target_col: str) -> tuple[str, str | None, str]:
    columns = set(df.columns)
    if target_col not in columns:
        candidates = [col for col in df.columns if col.startswith("n_bytes")]
        if not candidates:
            raise KeyError(f"Target column {target_col!r} not found and no n_bytes* column exists.")
        target_col = candidates[0]
    if id_col not in columns:
        candidates = [col for col in ("id_institution", "institution", "ID", "id") if col in columns]
        if not candidates:
            raise KeyError(f"Institution column {id_col!r} not found.")
        id_col = candidates[0]
    if time_col not in columns:
        time_col = "id_time" if "id_time" in columns else None
    return id_col, time_col, target_col


def prepare_frame(path: Path, id_col: str, time_col: str, target_col: str, limit_rows: int | None) -> pd.DataFrame:
    df = read_parquet_frame(path)
    id_col, time_col, target_col = resolve_columns(df, id_col, time_col, target_col)
    selected = [id_col, target_col]
    if time_col is not None:
        selected.append(time_col)
    df = df[selected].copy()
    df.rename(columns={id_col: "id_institution", target_col: "n_bytes"}, inplace=True)
    if time_col is not None:
        df.rename(columns={time_col: "time"}, inplace=True)
        if df["time"].dtype == object:
            parsed_time = pd.to_datetime(df["time"], errors="coerce", utc=True)
            if parsed_time.notna().all():
                df["time"] = parsed_time
        sort_cols = ["id_institution", "time"]
    else:
        sort_cols = ["id_institution"]

    df["n_bytes"] = pd.to_numeric(df["n_bytes"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.dropna(subset=["id_institution"]).sort_values(sort_cols, kind="mergesort")
    if limit_rows is not None:
        df = df.head(limit_rows)
    return df


def chronological_split(values: np.ndarray, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_end = int(len(values) * train_ratio)
    val_end = int(len(values) * (train_ratio + val_ratio))
    return values[:train_end], values[train_end:val_end], values[val_end:]


def sliding_windows(values: np.ndarray, lookback: int, horizon: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    total = lookback + horizon
    if values.size < total:
        return np.empty((0, lookback), dtype=np.float32), np.empty((0, horizon), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(values, total)[::step]
    windows = np.ascontiguousarray(windows, dtype=np.float32)
    return windows[:, :lookback], windows[:, lookback:]


def append_windows(
    X_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
    values: np.ndarray,
    lookback: int,
    horizon: int,
    step: int,
) -> None:
    X, y = sliding_windows(values, lookback, horizon, step)
    if len(X):
        X_parts.append(X)
        y_parts.append(y)


def build_windowed_dataset(
    spec: DatasetSpec,
    id_col: str,
    time_col: str,
    target_col: str,
    train_ratio: float,
    val_ratio: float,
    min_window_multiplier: int,
    limit_rows: int | None,
) -> WindowedDataset:
    df = prepare_frame(spec.path, id_col, time_col, target_col, limit_rows)
    rows_loaded = len(df)
    X_train_parts: list[np.ndarray] = []
    y_train_parts: list[np.ndarray] = []
    X_val_parts: list[np.ndarray] = []
    y_val_parts: list[np.ndarray] = []
    X_test_parts: list[np.ndarray] = []
    y_test_parts: list[np.ndarray] = []
    institutions_used = 0
    min_len = min_window_multiplier * (spec.lookback + spec.horizon)
    scalers_dict = {}

    for inst_id, group in df.groupby("id_institution", sort=False):
        values = group["n_bytes"].to_numpy(dtype=np.float32, copy=True)
        if values.size <= min_len:
            continue
        institutions_used += 1
        train, val, test = chronological_split(values, train_ratio, val_ratio)
        
        scaler = StandardScaler()
        train = scaler.fit_transform(train.reshape(-1, 1)).flatten()
        val = scaler.transform(val.reshape(-1, 1)).flatten()
        test = scaler.transform(test.reshape(-1, 1)).flatten()
        
        scalers_dict[inst_id] = scaler
        
        append_windows(X_train_parts, y_train_parts, train, spec.lookback, spec.horizon, spec.train_step)
        append_windows(X_val_parts, y_val_parts, val, spec.lookback, spec.horizon, spec.val_step)
        append_windows(X_test_parts, y_test_parts, test, spec.lookback, spec.horizon, spec.test_step)

    scalers_dir = ROOT / "scalers"
    scalers_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers_dict, scalers_dir / f"scalers_{spec.name}.joblib")

    if not X_train_parts or not X_val_parts or not X_test_parts:
        raise ValueError(
            f"{spec.name} did not produce enough windows. Try a smaller --window, "
            "--min-window-multiplier, or different train/validation split."
        )

    X_train = np.ascontiguousarray(np.concatenate(X_train_parts), dtype=np.float32)
    y_train = np.ascontiguousarray(np.concatenate(y_train_parts), dtype=np.float32)
    X_val = np.ascontiguousarray(np.concatenate(X_val_parts), dtype=np.float32)
    y_val = np.ascontiguousarray(np.concatenate(y_val_parts), dtype=np.float32)
    X_test = np.ascontiguousarray(np.concatenate(X_test_parts), dtype=np.float32)
    y_test = np.ascontiguousarray(np.concatenate(y_test_parts), dtype=np.float32)

    return WindowedDataset(
        X_train_mlp=X_train,
        X_val_mlp=X_val,
        X_test_mlp=X_test,
        X_train_seq=X_train[..., None],
        X_val_seq=X_val[..., None],
        X_test_seq=X_test[..., None],
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        spec=spec,
        rows_loaded=rows_loaded,
        institutions_used=institutions_used,
    )


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device, shuffle: bool) -> DataLoader:
    ensure_runtime_dependencies()
    X_tensor = torch.as_tensor(X, dtype=torch.float32)
    y_tensor = torch.as_tensor(y, dtype=torch.float32)
    pin_memory = device.type == "cuda"
    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=0,
        persistent_workers=False,
    )


def move_batch(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)
    try:
        return torch.amp.autocast("cuda", dtype=torch.float16)
    except TypeError:
        return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def batched_metrics(
    wrapper: TorchRegressionModel,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    use_amp: bool,
) -> dict[str, float]:
    loader = make_loader(X, y, batch_size=batch_size, device=wrapper.device, shuffle=False)
    total_sq_error = 0.0
    total_abs_error = 0.0
    total_smape = 0.0
    total_elements = 0
    y_min = None
    y_max = None

    wrapper.model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = move_batch(X_batch, wrapper.device)
            y_batch = move_batch(y_batch, wrapper.device)
            with autocast_context(wrapper.device, use_amp):
                pred = wrapper.model(X_batch)
            diff = y_batch - pred.float()
            total_sq_error += float(torch.sum(diff.square()).detach().cpu())
            total_abs_error += float(torch.sum(torch.abs(diff)).detach().cpu())
            denominator = torch.maximum(
                torch.abs(y_batch) + torch.abs(pred.float()),
                torch.tensor(1e-7, device=wrapper.device),
            )
            total_smape += float(torch.sum(torch.abs(diff) / denominator).detach().cpu())
            total_elements += diff.numel()
            batch_min = torch.min(y_batch).detach()
            batch_max = torch.max(y_batch).detach()
            y_min = batch_min if y_min is None else torch.minimum(y_min, batch_min)
            y_max = batch_max if y_max is None else torch.maximum(y_max, batch_max)

    mse = total_sq_error / max(total_elements, 1)
    rmse = float(np.sqrt(mse))
    data_range = float((y_max - y_min).detach().cpu()) if y_min is not None else 0.0
    return {
        "RMSE": rmse,
        "MAE": total_abs_error / max(total_elements, 1),
        "NRMSE": rmse / max(data_range, 1e-7),
        "SMAPE": 100.0 * total_smape / max(total_elements, 1),
    }


def build_model(model_name: str, dataset: WindowedDataset, device: torch.device) -> tuple[TorchRegressionModel, np.ndarray, np.ndarray, np.ndarray]:
    if model_name == "MLP":
        config = {
            "input_len": dataset.spec.lookback,
            "output": dataset.spec.horizon,
            "dropout_rate": 0.2,
            "dense_units": 512,
            "layers": 3,
        }
        wrapper = TorchRegressionModel(generate_MLP_model(**config), "MLP", config, expects_sequence=False, device=device)
        return wrapper, dataset.X_train_mlp, dataset.X_val_mlp, dataset.X_test_mlp

    units = 128
    config_key = {"RNN": "rnn_units", "LSTM": "lstm_units", "GRU": "gru_units"}[model_name]
    config = {
        "n_timesteps": dataset.spec.lookback,
        "n_features": 1,
        "n_outputs": dataset.spec.horizon,
        "dropout_rate": 0.2,
        config_key: units,
    }
    generator: dict[str, Callable[..., nn.Module]] = {
        "RNN": generate_RNN,
        "LSTM": generate_LSTM,
        "GRU": generate_GRU,
    }
    wrapper = TorchRegressionModel(
        generator[model_name](**config),
        model_name,
        config,
        expects_sequence=True,
        device=device,
    )
    return wrapper, dataset.X_train_seq, dataset.X_val_seq, dataset.X_test_seq


def maybe_compile_model(wrapper: TorchRegressionModel, enabled: bool) -> None:
    if not enabled:
        return
    try:
        wrapper.model = torch.compile(wrapper.model)
        print(f"[compile] Enabled torch.compile for {wrapper.model_type}.")
    except Exception as exc:
        print(f"[compile] torch.compile skipped for {wrapper.model_type}: {exc}")


def train_wrapper(
    wrapper: TorchRegressionModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    use_amp: bool,
) -> tuple[TorchRegressionModel, dict[str, float], int]:
    loader = make_loader(X_train, y_train, batch_size=batch_size, device=wrapper.device, shuffle=True)
    optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    scaler = make_grad_scaler(use_amp and wrapper.device.type == "cuda")
    best_score = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in wrapper.model.state_dict().items()}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        wrapper.model.train()
        train_loss = 0.0
        batches = 0
        for X_batch, y_batch in loader:
            X_batch = move_batch(X_batch, wrapper.device)
            y_batch = move_batch(y_batch, wrapper.device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(wrapper.device, use_amp):
                loss = criterion(wrapper.model(X_batch), y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += float(loss.detach().cpu())
            batches += 1

        val_metrics = batched_metrics(wrapper, X_val, y_val, batch_size=batch_size * 2, use_amp=use_amp)
        val_score = val_metrics["NRMSE"]
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in wrapper.model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"[train] {wrapper.model_type} bs={batch_size} epoch={epoch:03d} "
            f"loss={train_loss / max(batches, 1):.6f} val_nrmse={val_score:.6f} "
            f"val_smape={val_metrics['SMAPE']:.4f} seconds={time.perf_counter() - start:.1f}"
        )
        if stale_epochs >= patience:
            print(f"[train] Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    wrapper.model.load_state_dict(best_state)
    wrapper.model.to(wrapper.device)
    best_metrics = batched_metrics(wrapper, X_val, y_val, batch_size=batch_size * 2, use_amp=use_amp)
    return wrapper, best_metrics, best_epoch


def save_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def save_wrapper_checkpoint(wrapper: TorchRegressionModel, path: Path) -> None:
    compiled_model = wrapper.model
    original_model = getattr(compiled_model, "_orig_mod", None)
    if original_model is None:
        save_torch_model(wrapper, path)
        return

    wrapper.model = original_model
    try:
        save_torch_model(wrapper, path)
    finally:
        wrapper.model = compiled_model


def save_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"y_true": y_true.astype(np.float32), "y_pred": y_pred.astype(np.float32)}
    np.savez_compressed(path, **payload)


def run_training(args: argparse.Namespace) -> None:
    ensure_runtime_dependencies()
    device = configure_torch(cuda_device=args.cuda_device, allow_cpu=args.allow_cpu)
    gpu_report("startup")
    if device.type == "cuda":
        certify_gpu_usage(device)
    elif not args.allow_cpu:
        raise RuntimeError(f"CUDA is required, but selected device is {device}.")

    if args.gpu_certify_only:
        print("[done] GPU certification completed; skipping dataset loading/training.")
        return

    specs = discover_parquet_files(args.data_dir, args.method, args.window)
    if args.granularity:
        requested = set(args.granularity)
        specs = [spec for spec in specs if spec.name.split("_")[0] in requested]
    if not specs:
        raise FileNotFoundError("No matching parquet files for the requested granularities.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"

    for spec in specs:
        print(f"\n[data] Loading {spec.name}: {spec.path}")
        dataset = build_windowed_dataset(
            spec=spec,
            id_col=args.id_col,
            time_col=args.time_col,
            target_col=args.target_col,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            min_window_multiplier=args.min_window_multiplier,
            limit_rows=args.limit_rows,
        )
        print(
            f"[data] {spec.name}: rows={dataset.rows_loaded} institutions={dataset.institutions_used} "
            f"train={len(dataset.y_train)} val={len(dataset.y_val)} test={len(dataset.y_test)} "
            f"lookback={spec.lookback} horizon={spec.horizon}"
        )

        for batch_size in args.batch_sizes:
            for model_name in args.models:
                clear_memory()
                gpu_report(f"before {spec.name}/{model_name}/bs{batch_size}")
                wrapper, X_train, X_val, X_test = build_model(model_name, dataset, device)
                maybe_compile_model(wrapper, args.compile)
                assert_model_on_device(wrapper)

                wrapper, val_metrics, best_epoch = train_wrapper(
                    wrapper=wrapper,
                    X_train=X_train,
                    y_train=dataset.y_train,
                    X_val=X_val,
                    y_val=dataset.y_val,
                    batch_size=batch_size,
                    epochs=args.epochs,
                    patience=args.patience,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay,
                    use_amp=args.amp,
                )

                test_metrics = _avaliar_modelo_em_batches(
                    X_test,
                    dataset.y_test,
                    wrapper,
                    batch_size=batch_size * 2,
                    verbose=True,
                )

                sample_size = min(args.prediction_sample, len(dataset.y_test))
                y_pred = wrapper.predict(X_test[:sample_size], batch_size=batch_size * 2)
                sample_metrics = avaliar_modelo(dataset.y_test[:sample_size], y_pred, verbose=False)

                run_name = f"{spec.name}_{model_name}_bs{batch_size}"
                model_path = args.output_dir / "models" / f"{run_name}.pt"
                pred_path = args.output_dir / "predictions" / f"{run_name}.npz"
                save_wrapper_checkpoint(wrapper, model_path)
                save_predictions(pred_path, dataset.y_test[:sample_size], y_pred)

                row = {
                    "dataset": spec.name,
                    "parquet_path": str(spec.path),
                    "model": model_name,
                    "batch_size": batch_size,
                    "best_epoch": best_epoch,
                    "window": args.window,
                    "spec": {**asdict(spec), "path": str(spec.path)},
                    "val": val_metrics,
                    "test": test_metrics,
                    "prediction_sample": sample_metrics,
                    "model_path": str(model_path),
                    "prediction_path": str(pred_path),
                    "device": str(device),
                    "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "torch": getattr(torch, "__version__", "unknown"),
                    "torch_cuda_build": getattr(torch.version, "cuda", None),
                }
                save_jsonl(metrics_path, row)
                gpu_report(f"after {spec.name}/{model_name}/bs{batch_size}")

                del wrapper, X_train, X_val, X_test, y_pred
                clear_memory()

        del dataset
        clear_memory()

    print(f"\n[done] Metrics written to {metrics_path}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data"/"tratados", help="Directory containing treated parquet files.")
    parser.add_argument(
        "--method",
        choices=("auto", "new_method", "original", "all"),
        default="auto",
        help="Which treated parquet variant to use. auto prefers new_method when available.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "RESULTADOS" / "prediction_pipeline")
    parser.add_argument("--window", choices=tuple(WINDOWS), default="14-3")
    parser.add_argument("--granularity", nargs="*", choices=("day", "hour", "10min"), default=None)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=list(BATCH_SIZES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--min-window-multiplier", type=int, default=10)
    parser.add_argument("--target-col", default="n_bytes")
    parser.add_argument("--id-col", default="id_institution")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--prediction-sample", type=int, default=8192)
    parser.add_argument("--limit-rows", type=int, default=None, help="Debug option to load only the first N rows.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index to use. Defaults to PyTorch's current device.")
    parser.add_argument("--gpu-certify-only", action="store_true", help="Run the CUDA smoke test and exit before loading data.")
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback for debugging when CUDA is unavailable.")
    parser.add_argument("--require-gpu", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision. Disabled by default because raw n_bytes can overflow float16.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help=argparse.SUPPRESS)
    parser.add_argument("--compile", action="store_true", help="Try torch.compile for model execution.")
    parser.set_defaults(amp=False)
    args = parser.parse_args(argv)
    if args.require_gpu:
        args.allow_cpu = False

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("--train-ratio and --val-ratio must be positive and leave room for test data.")
    return args


if __name__ == "__main__":
    try:
        run_training(parse_args())
    except (MissingDependencyError, CudaConfigurationError) as exc:
        raise SystemExit(str(exc)) from exc
