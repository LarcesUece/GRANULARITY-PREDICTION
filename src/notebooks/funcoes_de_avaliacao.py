import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
import numpy as np
import math
from funcoes_de_treinamento import smape



def avaliar_modelo(y_real, y_previsto, verbose = False):
    mse = mean_squared_error(y_real, y_previsto)
    rmse = math.sqrt(mse)
    nrmse = rmse / (max(y_real) - min(y_real))
    mae = mean_absolute_error(y_real, y_previsto)
    _smape = smape(y_real, y_previsto) 
    if verbose:
        print(f"--- Desempenho: ---")
        print(f"RMSE (Erro Médio): {rmse:.4f}")
        print(f"MAE  (Erro Absoluto): {mae:.4f}")
        print(f"SMAPE: {_smape:.4f}")
        print(f"NRMSE: {nrmse:.4f}")
        print("-" * 30)
    return {"RMSE": rmse, "MAE": mae, "NRMSE": nrmse, "SMAPE": _smape}

def comparar_desempeho_granularidade(X_test_d, X_test_h, X_test_10m, y_test_d, y_test_h, y_test_10m, MODELO_d, MODELO_h,MODELO_10MIN):
    print("Carregando modelos...")
    y_pred_d = MODELO_d.predict(X_test_d)
    y_pred_h = MODELO_h.predict(X_test_h)
    y_pred_10m = MODELO_10MIN.predict(X_test_10m)
    
    print(f"Desempenho do modelo para granularidade diária:")
    resultado_d = avaliar_modelo(y_test_d.flatten(), y_pred_d.flatten(), verbose=True)
    
    print(f"Desempenho do modelo para granularidade horária:")
    resultado_h = avaliar_modelo(y_test_h.flatten(), y_pred_h.flatten(), verbose=True)

    print(f"Desempenho do modelo para granularidade 10minutos:")
    resultado_10m = avaliar_modelo(y_test_10m.flatten(), y_pred_10m.flatten(), verbose=True)    
    
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



        
        

