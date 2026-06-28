import gc
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

pd.set_option("future.no_silent_downcasting", True)


EPSILON = 1e-7


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEVICE = get_device()
print(f"Using PyTorch device: {DEVICE}")


def _clear_torch_memory():
    gc.collect()
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


def _loader_pin_memory(tensor, device):
    return device.type == "cuda" and torch.is_tensor(tensor) and tensor.device.type == "cpu"


def _move_batch(batch, device):
    if batch.device == device:
        return batch
    return batch.to(device=device, dtype=torch.float32, non_blocking=True)


class History:
    def __init__(self):
        self.history = {
            "loss": [],
            "val_loss": [],
            "rmse": [],
            "val_rmse": [],
            "mae": [],
            "val_mae": [],
            "nrmse": [],
            "val_nrmse": [],
            "smape": [],
            "val_smape": [],
        }


class MLPRegressor(nn.Module):
    def __init__(self, input_len, output_len, dropout_rate=0.2, dense_units=500, layers=3):
        super().__init__()
        modules = [nn.Dropout(0.1), nn.Flatten()]
        in_features = input_len
        for _ in range(layers):
            modules.append(nn.Linear(in_features, dense_units))
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(dropout_rate))
            in_features = dense_units
        modules.append(nn.Linear(in_features, output_len))
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)


class RecurrentRegressor(nn.Module):
    def __init__(
        self,
        cell_type,
        n_features,
        n_outputs,
        dropout_rate=0.2,
        hidden_units=64,
    ):
        super().__init__()
        cell_type = cell_type.upper()
        cells = {"GRU": nn.GRU, "LSTM": nn.LSTM, "RNN": nn.RNN}
        if cell_type not in cells:
            raise ValueError(f"Unsupported recurrent cell: {cell_type}")
        self.rnn = cells[cell_type](
            input_size=n_features,
            hidden_size=hidden_units,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.output = nn.Linear(hidden_units, n_outputs)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.output(self.dropout(last))


class TorchRegressionModel:
    def __init__(self, model, model_type, config, expects_sequence=False, device=None):
        self.model = model
        self.model_type = model_type
        self.config = config
        self.expects_sequence = expects_sequence
        self.device = device or DEVICE
        self.model.to(self.device)

    def predict(self, X, batch_size=4096):
        self.model.eval()
        return_tensor = torch.is_tensor(X)
        X = _prepare_X(X, expects_sequence=self.expects_sequence)
        tensor = torch.as_tensor(X, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(tensor),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=_loader_pin_memory(tensor, self.device),
        )
        preds = []
        with torch.no_grad():
            for (batch,) in loader:
                batch = _move_batch(batch, self.device)
                preds.append(self.model(batch).detach())
        preds = torch.cat(preds, dim=0)
        if return_tensor:
            return preds
        return preds.cpu().numpy()

    def save(self, path):
        save_torch_model(self, path)


def generate_GRU(n_timesteps, n_features, n_outputs, dropout_rate=0.2, gru_units=64):
    return RecurrentRegressor("GRU", n_features, n_outputs, dropout_rate, gru_units)


def generate_LSTM(
    n_timesteps,
    n_features,
    n_outputs,
    dropout_rate=0.2,
    lstm_units=64,
):
    return RecurrentRegressor("LSTM", n_features, n_outputs, dropout_rate, lstm_units)


def generate_RNN(
    n_timesteps,
    n_features,
    n_outputs,
    dropout_rate=0.2,
    rnn_units=64,
):
    return RecurrentRegressor("RNN", n_features, n_outputs, dropout_rate, rnn_units)


def generate_MLP_model(input_len, output, dropout_rate=0.2, dense_units=500, layers=3):
    return MLPRegressor(input_len, output, dropout_rate, dense_units, layers)


from statsmodels.tsa.statespace.sarimax import SARIMAX


def generate_SARIMA_model(order, seasonal_order, X_train):
    model = SARIMAX(
        X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model


def get_nrmse(global_range):
    global_range = max(float(global_range), EPSILON)

    def nrmse(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(y_true - y_pred))) / global_range)

    nrmse.__name__ = "nrmse"
    return nrmse


def smape(y_true, y_pred):
    if torch.is_tensor(y_true) or torch.is_tensor(y_pred):
        y_true = torch.as_tensor(y_true, dtype=torch.float32)
        y_pred = torch.as_tensor(y_pred, dtype=torch.float32, device=y_true.device)
        diff = torch.abs(y_true - y_pred)
        add = torch.abs(y_true) + torch.abs(y_pred)
        return 100.0 * torch.mean(diff / torch.maximum(add, torch.tensor(EPSILON, device=y_true.device)))

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    diff = np.abs(y_true - y_pred)
    add = np.abs(y_true) + np.abs(y_pred)
    return float(100.0 * np.mean(diff / np.maximum(add, EPSILON)))


def _prepare_X(X, expects_sequence=False):
    if torch.is_tensor(X):
        X = X.to(dtype=torch.float32)
    else:
        X = np.asarray(X, dtype=np.float32)
    if expects_sequence and X.ndim == 2:
        X = X.reshape((X.shape[0], 1, X.shape[1]))
    return X


def _prepare_y(y):
    if torch.is_tensor(y):
        return y.to(dtype=torch.float32)
    return np.asarray(y, dtype=np.float32)


def _metric_values(y_true, y_pred, amplitude_global):
    mse = torch.mean((y_true - y_pred) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(y_true - y_pred))
    nrmse = rmse / max(float(amplitude_global), EPSILON)
    smape_value = smape(y_true, y_pred)
    return {
        "loss": float(mse.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "nrmse": float(nrmse.detach().cpu()),
        "smape": float(smape_value.detach().cpu()),
    }


def _evaluate_model(wrapper, X_tensor, y_tensor, batch_size, amplitude_global):
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=_loader_pin_memory(X_tensor, wrapper.device),
    )
    total_sq_error = 0.0
    total_abs_error = 0.0
    total_smape = 0.0
    total_elements = 0

    wrapper.model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = _move_batch(X_batch, wrapper.device)
            y_batch = _move_batch(y_batch, wrapper.device)
            pred = wrapper.model(X_batch)
            diff = y_batch - pred
            elements = diff.numel()
            total_sq_error += float(torch.sum(diff ** 2).detach().cpu())
            total_abs_error += float(torch.sum(torch.abs(diff)).detach().cpu())
            denominator = torch.maximum(
                torch.abs(y_batch) + torch.abs(pred),
                torch.tensor(EPSILON, device=wrapper.device),
            )
            total_smape += float(torch.sum(torch.abs(diff) / denominator).detach().cpu())
            total_elements += elements

    mse = total_sq_error / max(total_elements, 1)
    rmse = float(np.sqrt(mse))
    mae = total_abs_error / max(total_elements, 1)
    return {
        "loss": mse,
        "rmse": rmse,
        "mae": mae,
        "nrmse": rmse / max(float(amplitude_global), EPSILON),
        "smape": 100.0 * total_smape / max(total_elements, 1),
    }


def _append_metrics(history, train_metrics, val_metrics):
    for key, value in train_metrics.items():
        history.history[key].append(value)
    for key, value in val_metrics.items():
        history.history[f"val_{key}"].append(value)


def _resolve_model_path(path_modelo):
    path = Path(path_modelo)
    if not path.is_absolute():
        path = Path("../../MODELOS") / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_torch_model(wrapper, path):
    path = _resolve_model_path(path)
    torch.save(
        {
            "model_type": wrapper.model_type,
            "config": wrapper.config,
            "expects_sequence": wrapper.expects_sequence,
            "state_dict": wrapper.model.cpu().state_dict(),
        },
        path,
    )
    wrapper.model.to(wrapper.device)


def load_torch_model(path, device=None):
    device = device or DEVICE
    checkpoint = torch.load(path, map_location=device)
    model_type = checkpoint["model_type"]
    config = checkpoint["config"]

    if model_type == "MLP":
        model = generate_MLP_model(**config)
    elif model_type == "GRU":
        model = generate_GRU(**config)
    elif model_type == "LSTM":
        model = generate_LSTM(**config)
    elif model_type == "RNN":
        model = generate_RNN(**config)
    else:
        raise ValueError(f"Unsupported model type in checkpoint: {model_type}")

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return TorchRegressionModel(
        model=model,
        model_type=model_type,
        config=config,
        expects_sequence=checkpoint.get("expects_sequence", model_type != "MLP"),
        device=device,
    )


def _train_model(
    wrapper,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    amplitude_global=None,
):
    X_train = _prepare_X(X_train, wrapper.expects_sequence)
    X_val = _prepare_X(X_val, wrapper.expects_sequence)
    y_train = _prepare_y(y_train)
    y_val = _prepare_y(y_val)

    X_train_tensor = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.as_tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.as_tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.as_tensor(y_val, dtype=torch.float32)
    if amplitude_global is None:
        amplitude_global = float(
            torch.clamp(
                torch.max(y_train_tensor) - torch.min(y_train_tensor),
                min=EPSILON,
            )
            .detach()
            .cpu()
        )

    loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=_loader_pin_memory(X_train_tensor, wrapper.device),
    )
    optimizer = torch.optim.Adam(wrapper.model.parameters())
    criterion = nn.MSELoss()
    history = History()

    for _ in range(epochs):
        wrapper.model.train()
        for X_batch, y_batch in loader:
            X_batch = _move_batch(X_batch, wrapper.device)
            y_batch = _move_batch(y_batch, wrapper.device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(wrapper.model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        wrapper.model.eval()
        train_metrics = _evaluate_model(wrapper, X_train_tensor, y_train_tensor, batch_size, amplitude_global)
        val_metrics = _evaluate_model(wrapper, X_val_tensor, y_val_tensor, batch_size, amplitude_global)
        _append_metrics(history, train_metrics, val_metrics)
        _clear_torch_memory()

    return history


def _print_history(model_name, history):
    print(
        f"modelo treinado!"
        f"\nResultado:"
        f"\nRMSE:\n   teste:{history.history['rmse'][-1]}   val:{history.history['val_rmse'][-1]}"
        f"\nMAE:\n    teste:{history.history['mae'][-1]}   val:{history.history['val_mae'][-1]}"
        f"\nNRMSE:\n  teste:{history.history['nrmse'][-1]}   val:{history.history['val_nrmse'][-1]}"
        f"\nSMAPE:\n  teste:{history.history['smape'][-1]}   val:{history.history['val_smape'][-1]}"
    )


def _plot_history(model_name, history):
    if plt is None:
        print("matplotlib not installed; skipping training metric plots.")
        return
    for metric in ["rmse", "mae", "nrmse", "smape"]:
        plt.title(label=f"{model_name} Val {metric.upper()}")
        plt.plot(history.history[f"val_{metric}"])
        plt.show()


def _save_metrics(path_metricas, model_name, dimensao, history):
    dict_model = {
        "MODELO": [model_name],
        "DIM": [dimensao],
        "RMSE": [history.history["val_rmse"][-1]],
        "MAE": [history.history["val_mae"][-1]],
        "NMRSE": [history.history["val_nrmse"][-1]],
        "SMAPE": [history.history["val_smape"][-1]],
    }
    df = pd.DataFrame(dict_model)
    header_condition = not os.path.exists(path_metricas)
    df.to_csv(path_metricas, index=False, mode="a", header=header_condition)


def _finalize_training(model_name, wrapper, history, dimensao, path_modelo, path_metricas, plot, verbose):
    if verbose:
        _print_history(model_name, history)
    if plot:
        _plot_history(model_name, history)
    if path_modelo is not None:
        wrapper.save(path_modelo)
    if path_metricas is not None:
        _save_metrics(path_metricas, model_name, dimensao, history)
    return history


def criar_e_treinarMLP(
    dimensao,
    input_len,
    output_len,
    dropout_rate,
    dense_units,
    layers,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    path_modelo=None,
    path_metricas=None,
    plot=True,
    verbose=True,
):
    print("criando modelo...")
    config = {
        "input_len": input_len,
        "output": output_len,
        "dropout_rate": dropout_rate,
        "dense_units": dense_units,
        "layers": layers,
    }
    wrapper = TorchRegressionModel(generate_MLP_model(**config), "MLP", config)
    print("treinando modelo...")
    history = _train_model(wrapper, X_train, y_train, X_val, y_val, epochs, batch_size)
    return _finalize_training("MLP", wrapper, history, dimensao, path_modelo, path_metricas, plot, verbose)


def criar_e_treinarGRU(
    dimensao,
    input_len,
    output_len,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    dropout_rate,
    gru_units,
    path_modelo=None,
    path_metricas=None,
    plot=True,
    verbose=True,
):
    print("criando modelo...")
    config = {
        "n_timesteps": 1,
        "n_features": input_len,
        "n_outputs": output_len,
        "dropout_rate": dropout_rate,
        "gru_units": gru_units,
    }
    wrapper = TorchRegressionModel(generate_GRU(**config), "GRU", config, expects_sequence=True)
    print("treinando modelo...")
    history = _train_model(wrapper, X_train, y_train, X_val, y_val, epochs, batch_size)
    return _finalize_training("GRU", wrapper, history, dimensao, path_modelo, path_metricas, plot, verbose)


def criar_e_treinarLSTM(
    dimensao,
    input_len,
    output_len,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    dropout_rate,
    lstm_units,
    path_modelo=None,
    path_metricas=None,
    plot=True,
    verbose=True,
):
    print("criando modelo...")
    config = {
        "n_timesteps": 1,
        "n_features": input_len,
        "n_outputs": output_len,
        "dropout_rate": dropout_rate,
        "lstm_units": lstm_units,
    }
    wrapper = TorchRegressionModel(generate_LSTM(**config), "LSTM", config, expects_sequence=True)
    print("treinando modelo...")
    history = _train_model(wrapper, X_train, y_train, X_val, y_val, epochs, batch_size)
    return _finalize_training("LSTM", wrapper, history, dimensao, path_modelo, path_metricas, plot, verbose)


def criar_e_treinarRNN(
    dimensao,
    input_len,
    output_len,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    dropout_rate,
    rnn_units,
    path_modelo=None,
    path_metricas=None,
    plot=True,
    verbose=True,
):
    print("criando modelo...")
    config = {
        "n_timesteps": 1,
        "n_features": input_len,
        "n_outputs": output_len,
        "dropout_rate": dropout_rate,
        "rnn_units": rnn_units,
    }
    wrapper = TorchRegressionModel(generate_RNN(**config), "RNN", config, expects_sequence=True)
    print("treinando modelo...")
    history = _train_model(wrapper, X_train, y_train, X_val, y_val, epochs, batch_size)
    return _finalize_training("RNN", wrapper, history, dimensao, path_modelo, path_metricas, plot, verbose)


def otimizar_GRU(X_train, y_train, X_val, y_val, dimensao, input_len, output_len, path_modelo=None):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 10, 30, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        gru_units = trial.suggest_int("gru_units", 32, 128)
        try:
            history = criar_e_treinarGRU(
                dimensao=dimensao,
                input_len=input_len,
                output_len=output_len,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                gru_units=gru_units,
                path_modelo=None,
                plot=False,
                verbose=False,
            )
            return history.history["val_nrmse"][-1]
        except Exception as exc:
            raise optuna.exceptions.TrialPruned() from exc
        finally:
            _clear_torch_memory()

    study = optuna.create_study(direction="minimize", study_name="Otimizacao_GRU")
    study.optimize(objective, n_trials=6)
    print(f"\nMelhor NRMSE: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")
    criar_e_treinarGRU(
        dimensao=dimensao,
        input_len=input_len,
        output_len=output_len,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=study.best_params["epochs"],
        batch_size=study.best_params["batch_size"],
        dropout_rate=study.best_params["dropout_rate"],
        gru_units=study.best_params["gru_units"],
        path_modelo=path_modelo,
        plot=True,
    )
    return study


def otimizar_LSTM(X_train, y_train, X_val, y_val, dimensao, input_len, output_len, path_modelo=None):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 10, 30, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        lstm_units = trial.suggest_int("lstm_units", 32, 128)
        try:
            history = criar_e_treinarLSTM(
                dimensao=dimensao,
                input_len=input_len,
                output_len=output_len,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                lstm_units=lstm_units,
                path_modelo=None,
                plot=False,
                verbose=False,
            )
            return history.history["val_nrmse"][-1]
        except Exception as exc:
            raise optuna.exceptions.TrialPruned() from exc
        finally:
            _clear_torch_memory()

    study = optuna.create_study(direction="minimize", study_name="Otimizacao_LSTM")
    study.optimize(objective, n_trials=6)
    print(f"\nMelhor NRMSE: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")
    criar_e_treinarLSTM(
        dimensao=dimensao,
        input_len=input_len,
        output_len=output_len,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=study.best_params["epochs"],
        batch_size=study.best_params["batch_size"],
        dropout_rate=study.best_params["dropout_rate"],
        lstm_units=study.best_params["lstm_units"],
        path_modelo=path_modelo,
        plot=True,
    )
    return study


def otimizar_RNN(X_train, y_train, X_val, y_val, dimensao, input_len, output_len, path_modelo=None):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 10, 30, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        rnn_units = trial.suggest_int("rnn_units", 32, 128)
        try:
            history = criar_e_treinarRNN(
                dimensao=dimensao,
                input_len=input_len,
                output_len=output_len,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                dropout_rate=dropout_rate,
                rnn_units=rnn_units,
                path_modelo=None,
                plot=False,
                verbose=False,
            )
            return history.history["val_nrmse"][-1]
        except Exception as exc:
            raise optuna.exceptions.TrialPruned() from exc
        finally:
            _clear_torch_memory()

    study = optuna.create_study(direction="minimize", study_name="Otimizacao_RNN")
    study.optimize(objective, n_trials=6)
    print(f"\nMelhor NRMSE: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")
    criar_e_treinarRNN(
        dimensao=dimensao,
        input_len=input_len,
        output_len=output_len,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=study.best_params["epochs"],
        batch_size=study.best_params["batch_size"],
        dropout_rate=study.best_params["dropout_rate"],
        rnn_units=study.best_params["rnn_units"],
        path_modelo=path_modelo,
        plot=True,
    )
    return study


def otimizar_MLP(X_train, y_train, X_val, y_val, dimensao, input_len, output_len, path_modelo=None):
    def objective(trial):
        batch_size = trial.suggest_categorical("batch_size", [16, 32])
        epochs = trial.suggest_int("epochs", 10, 30, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        dense_units = trial.suggest_int("dense_units", 100, 500)
        layers = trial.suggest_int("layers", 1, 3)
        try:
            history = criar_e_treinarMLP(
                dimensao=dimensao,
                input_len=input_len,
                output_len=output_len,
                dropout_rate=dropout_rate,
                dense_units=dense_units,
                layers=layers,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=epochs,
                batch_size=batch_size,
                path_modelo=None,
                plot=False,
                verbose=False,
            )
            return history.history["val_nrmse"][-1]
        except Exception as exc:
            raise optuna.exceptions.TrialPruned() from exc
        finally:
            _clear_torch_memory()

    study = optuna.create_study(direction="minimize", study_name="Otimizacao_MLP")
    study.optimize(objective, n_trials=20)
    print(f"\nMelhor NRMSE: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")
    criar_e_treinarMLP(
        dimensao=dimensao,
        input_len=input_len,
        output_len=output_len,
        dropout_rate=study.best_params["dropout_rate"],
        dense_units=study.best_params["dense_units"],
        layers=study.best_params["layers"],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=study.best_params["epochs"],
        batch_size=study.best_params["batch_size"],
        path_modelo=path_modelo,
        plot=True,
    )
    return study
