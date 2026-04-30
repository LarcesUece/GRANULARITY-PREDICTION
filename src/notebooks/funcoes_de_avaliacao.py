from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from funcoes_de_treinamento import smape
from time import time, perf_counter
def avaliar_modelo(y_real, y_previsto, tempo = 0.0, verbose = False):
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
        print(f"Tempo de avaliação: {tempo} segundos para {len(y_real)} amostras")
        
        print("-" * 30)
    return { "RMSE":rmse,"MAE": mae, "NRMSE": nrmse, "SMAPE": _smape, "TIME": tempo}

def comparar_desempeho_granularidade(X_test_d, X_test_h, X_test_10m, y_test_d, y_test_h, y_test_10m, MODELO_d, MODELO_h,MODELO_10MIN):
    print("Carregando modelos...")
    
    #---#
    MODELO_h.predict(X_test_h[:1], verbose=0)  # Previsão de teste para "aquecer" o modelo
    MODELO_d.predict(X_test_d[:1], verbose=0)  # Previsão de teste para "aquecer" o modelo
    MODELO_10MIN.predict(X_test_10m[:1], verbose=0)  # Previsão de teste para "aquecer" o modelo
    #---#
    mean_d = np.zeros(5)
    mean_h = np.zeros(5)
    mean_10m = np.zeros(5)
    n = 10
    for _ in range(n):
        start_time = perf_counter() 
        y_pred_d = MODELO_d.predict(X_test_d, verbose=0)
        d_time = perf_counter() - start_time
        start_time = perf_counter()
        y_pred_h = MODELO_h.predict(X_test_h, verbose=0)
        h_time = perf_counter() - start_time
        start_time = perf_counter()
        y_pred_10m = MODELO_10MIN.predict(X_test_10m, verbose=0)
        m_time = perf_counter() - start_time

        dict_d = avaliar_modelo(y_test_d.flatten(), y_pred_d.flatten(),d_time)
        dict_h = avaliar_modelo(y_test_h.flatten(), y_pred_h.flatten(),h_time)
        dict_10m = avaliar_modelo(y_test_10m.flatten(), y_pred_10m.flatten(),m_time)

        mean_d[0] += dict_d["RMSE"]
        mean_d[1] += dict_d["MAE"]
        mean_d[2] += dict_d["NRMSE"]
        mean_d[3] += dict_d["SMAPE"]
        mean_d[4] += dict_d["TIME"]

        mean_h[0] += dict_h["RMSE"]
        mean_h[1] += dict_h["MAE"]
        mean_h[2] += dict_h["NRMSE"]
        mean_h[3] += dict_h["SMAPE"]
        mean_h[4] += dict_h["TIME"]

        mean_10m[0] += dict_10m["RMSE"]
        mean_10m[1] += dict_10m["MAE"]
        mean_10m[2] += dict_10m["NRMSE"]
        mean_10m[3] += dict_10m["SMAPE"]
        mean_10m[4] += dict_10m["TIME"]
        
        


    mean_d /= n
    mean_h /= n
    mean_10m /= n
    

    dict_d = {"RMSE":mean_d[0],"MAE":mean_d[1],"NRMSE":mean_d[2],"SMAPE":mean_d[3],"TIME":mean_d[4]}
    dict_h = {"RMSE":mean_h[0],"MAE":mean_h[1],"NRMSE":mean_h[2],"SMAPE":mean_h[3],"TIME":mean_h[4]}
    dict_10m = {"RMSE":mean_10m[0],"MAE":mean_10m[1],"NRMSE":mean_10m[2],"SMAPE":mean_10m[3],"TIME":mean_10m[4]}    



    print(f"Desempenho do modelo para granularidade diária:")
    print(f"RMSE: {mean_d[0]}")
    print(f"MAE: {mean_d[1]}")
    print(f"NRMSE: {mean_d[2]}")
    print(f"SMAPE: {mean_d[3]}")
    print(f"TIME: {mean_d[4]} para {len(X_test_d)} amostras")
    
    print(f"Desempenho do modelo para granularidade horária:")
    print(f"RMSE: {mean_h[0]}")
    print(f"MAE: {mean_h[1]}")
    print(f"NRMSE: {mean_h[2]}")
    print(f"SMAPE: {mean_h[3]}")
    print(f"TIME: {mean_h[4]} para {len(X_test_h)} amostras")
    
    print(f"Desempenho do modelo para granularidade 10minutos:")
    print(f"RMSE: {mean_10m[0]}")
    print(f"MAE: {mean_10m[1]}")
    print(f"NRMSE: {mean_10m[2]}")
    print(f"SMAPE: {mean_10m[3]}")
    print(f"TIME: {mean_10m[4]} para {len(X_test_10m)} amostras")
    
    return {"Diario": dict_d, "Horario": dict_h, "10minutos:": dict_10m}

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



        
        

