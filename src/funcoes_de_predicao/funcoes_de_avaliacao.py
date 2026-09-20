import gc

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from funcoes_de_treinamento import smape


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_float_tensor(data, device):
    if torch.is_tensor(data):
        return data.to(device=device, dtype=torch.float32)
    return torch.as_tensor(np.asarray(data), dtype=torch.float32, device=device)


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


def avaliar_modelo(y_real, y_previsto, verbose = False):
    device = get_device()
    y_real_tensor = _as_float_tensor(y_real, device)
    y_previsto_tensor = _as_float_tensor(y_previsto, device)
    diff = y_real_tensor - y_previsto_tensor
    mse = torch.mean(diff ** 2)
    rmse = torch.sqrt(mse)
    data_range = torch.max(y_real_tensor) - torch.min(y_real_tensor)
    nrmse = rmse / torch.clamp(data_range, min=1e-7)
    mae = torch.mean(torch.abs(diff))
    _smape = smape(y_real_tensor, y_previsto_tensor)

    rmse = float(rmse.detach().cpu())
    mae = float(mae.detach().cpu())
    nrmse = float(nrmse.detach().cpu())
    _smape = float(_smape.detach().cpu())
    if verbose:
        print(f"--- Desempenho: ---")
        print(f"RMSE (Erro Médio): {rmse:.4f}")
        print(f"MAE  (Erro Absoluto): {mae:.4f}")
        print(f"SMAPE: {_smape:.4f}")
        print(f"NRMSE: {nrmse:.4f}")
        print("-" * 30)
    return {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "SMAPE": _smape}


def _avaliar_modelo_em_batches(X_test, y_test, modelo, batch_size=4096, verbose=False):
    device = modelo.device
    X_tensor = torch.as_tensor(X_test, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_test, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=_loader_pin_memory(X_tensor, device),
    )

    total_sq_error = 0.0
    total_abs_error = 0.0
    total_smape = 0.0
    total_elements = 0
    data_min = None
    data_max = None

    modelo.model.eval()
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = _move_batch(X_batch, device)
            y_batch = _move_batch(y_batch, device)
            y_pred = modelo.model(X_batch)
            diff = y_batch - y_pred
            total_sq_error += float(torch.sum(diff ** 2).detach().cpu())
            total_abs_error += float(torch.sum(torch.abs(diff)).detach().cpu())
            denominator = torch.maximum(
                torch.abs(y_batch) + torch.abs(y_pred),
                torch.tensor(1e-7, device=device),
            )
            total_smape += float(torch.sum(torch.abs(diff) / denominator).detach().cpu())
            total_elements += diff.numel()

            batch_min = torch.min(y_batch).detach()
            batch_max = torch.max(y_batch).detach()
            data_min = batch_min if data_min is None else torch.minimum(data_min, batch_min)
            data_max = batch_max if data_max is None else torch.maximum(data_max, batch_max)

    mse = total_sq_error / max(total_elements, 1)
    rmse = float(np.sqrt(mse))
    mae = total_abs_error / max(total_elements, 1)
    data_range = float((data_max - data_min).detach().cpu()) if data_min is not None else 0.0
    nrmse = rmse / max(data_range, 1e-7)
    _smape = 100.0 * total_smape / max(total_elements, 1)

    if verbose:
        print(f"--- Desempenho: ---")
        print(f"RMSE (Erro Médio): {rmse:.4f}")
        print(f"MAE  (Erro Absoluto): {mae:.4f}")
        print(f"SMAPE: {_smape:.4f}")
        print(f"NRMSE: {nrmse:.4f}")
        print("-" * 30)
    _clear_torch_memory()
    return {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "SMAPE": _smape}


def comparar_desempeho_granularidade(X_test_d, X_test_h, X_test_10m, y_test_d, y_test_h, y_test_10m, MODELO_d, MODELO_h,MODELO_10MIN):
    print("Carregando modelos...")

    print(f"Desempenho do modelo para granularidade diária:")
    resultado_d = _avaliar_modelo_em_batches(X_test_d, y_test_d, MODELO_d, verbose=True)

    print(f"Desempenho do modelo para granularidade horária:")
    resultado_h = _avaliar_modelo_em_batches(X_test_h, y_test_h, MODELO_h, verbose=True)

    print(f"Desempenho do modelo para granularidade 10minutos:")
    resultado_10m = _avaliar_modelo_em_batches(X_test_10m, y_test_10m, MODELO_10MIN, verbose=True)

    return {"Diario": resultado_d, "Horario": resultado_h, "10minutos": resultado_10m}

def separar_dados_por_instituicao(inst, X_test, y_test =  None):
    idx = np.where(X_test[:,0] == inst)[0]
    X_test_i = X_test[idx]
    if y_test is not None:
        y_test_i = y_test[idx]
        return X_test_i, y_test_i
    else:
        return X_test_i
def avaliar_modelo_inst(inst:list,X_test_d, X_test_h, y_test_d, y_test_h, modelo_d, modelo_h):
    resultados = {
        "instituição": [],
        "granularidade:": [],
        "MAE": [],
        "RMSE": [],
        "NRMSE": [],
        "SMAPE": []
    }
    for i in inst:
        print(f"\n##############################\n \
Avaliando instituição {i}... \
                \n##############################\n")
        X_test_d_i, y_test_d_i = separar_dados_por_instituicao(i, X_test_d, y_test_d)
        X_test_h_i, y_test_h_i = separar_dados_por_instituicao(i, X_test_h, y_test_h)
        resultado = comparar_desempeho_granularidade(X_test_d_i, X_test_h_i, y_test_d_i, y_test_h_i, modelo_d, modelo_h)
        resultados["instituição"].append(i)
        resultados["granularidade:"].append("diária")
        resultados["MAE"].append(resultado["Diario"]["MAE"])
        resultados["RMSE"].append(resultado["Diario"]["RMSE"])
        resultados["NRMSE"].append(resultado["Diario"]["NRMSE"])
        resultados["SMAPE"].append(resultado["Diario"]["SMAPE"])

        resultados["instituição"].append(i)
        resultados["granularidade:"].append("horária")
        resultados["MAE"].append(resultado["Horario"]["MAE"])
        resultados["RMSE"].append(resultado["Horario"]["RMSE"])
        resultados["NRMSE"].append(resultado["Horario"]["NRMSE"])
        resultados["SMAPE"].append(resultado["Horario"]["SMAPE"])
    return pd.DataFrame(resultados)



        
        
