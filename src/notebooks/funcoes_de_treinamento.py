import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from keras.models import Model
from keras.layers import Input, Dense, GRU as GRU_layer, LSTM as LSTM_layer, SimpleRNN as RNN_layer
from keras.layers import Flatten,  Dropout
from tensorflow import cast, float32,reduce_mean,maximum
import tensorflow as tf
from keras.backend import epsilon
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import os
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
import numpy as np
import optuna
import joblib
import gc
inst = joblib.load("../scalers/instituicoes_validas.joblib")


def generate_GRU(n_timesteps, n_features, n_outputs, dropout_rate=0.2, gru_units=64):
    """
    Cria um modelo GRU puro.
    
    n_timesteps: Comprimento da sequência temporal (no seu caso, 1)
    n_features: Quantidade de colunas/variáveis de entrada
    n_outputs: Quantidade de valores a prever
    """
    
    # Camada de Entrada
    inp = Input(shape=(n_timesteps, n_features))
    x = GRU_layer(gru_units, return_sequences=False)(inp) 
    
    # Dropout para evitar overfitting
    x = Dropout(dropout_rate)(x)
    out = Dense(n_outputs, activation='linear')(x)
    
    model = Model(inputs=inp, outputs=out)
    return model



from statsmodels.tsa.statespace.sarimax import SARIMAX

def generate_SARIMA_model(order, seasonal_order, X_train):
    """
    Cria um modelo SARIMA com os parâmetros fornecidos.
    
    order: tupla (p,d,q)
    seasonal_order: tupla (P,D,Q,s)
    X_train: série temporal de treino
    """
    model = SARIMAX(
        X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    return model

def generate_MLP_model(input_len, output, dropout_rate=0.2, dense_units=500, layers=3):
    ip = Input(shape=(input_len,))
    y= Dropout(0.1)(ip)
    y = Flatten()(y)
    for _ in range(layers):
        y = Dense(dense_units, activation='relu')(y)
        y = Dropout(dropout_rate)(y)
    out = Dense(output, activation='linear')(y)
    model = Model(ip, out)

    return model

def generate_LSTM(n_timesteps, 
                  n_features, 
                  n_outputs, 
                  dropout_rate=0.2, 
                  lstm_units=64
                  ):
    inp = Input(shape=(n_timesteps, n_features))
    x = LSTM_layer(lstm_units, return_sequences=False)(inp) 
    x = Dropout(dropout_rate)(x)
    out = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def generate_RNN(n_timesteps, 
                 n_features, 
                 n_outputs, 
                 dropout_rate=0.2, 
                 rnn_units=64
                 ):
    inp = Input(shape=(n_timesteps, n_features))
    x = RNN_layer(rnn_units, return_sequences=False)(inp) 
    x = Dropout(dropout_rate)(x)
    out = Dense(n_outputs, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    return model

def get_nrmse(global_range):
    def nrmse(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # Calcular o RMSE matematicamente
        mse = tf.reduce_mean(tf.square(y_true - y_pred))
        rmse = tf.sqrt(mse)
        
        # Usa a amplitude global fixa em vez de calcular por lote!
        return rmse / global_range
    
    # Renomeia a função interna para ficar bonito no log do Keras
    nrmse.__name__ = 'nrmse'
    return nrmse

def smape(y_true, y_pred):
    y_true = cast(y_true, float32)
    y_pred = cast(y_pred, float32)
    diff = abs(y_true - y_pred)
    add = abs(y_true) + abs(y_pred)
    # epsilon é usado para evitar divisão por zero
    return 100.0 * reduce_mean(diff / maximum(add, epsilon()))


##################################################################
#_____________________FUNÇÕES DE TREINAMENTO_____________________#
##################################################################


def criar_e_treinarMLP(dimensao,
                       input_len, 
                       output_len,
                       dropout_rate,
                        dense_units,
                        layers,
                       X_train,y_train, 
                       X_val, 
                       y_val, 
                       epochs, 
                       batch_size, 
                       path_modelo = None, 
                       path_metricas = None,
                       plot = True,
                       verbose = True
                       ):
    print("criando modelo...")
    mlp = generate_MLP_model(input_len,output_len, dropout_rate=dropout_rate, dense_units=dense_units, layers=layers)
    print("compilando modelo...")
    amplitude_global = np.max(y_train) - np.min(y_train)
    amplitude_global = max(amplitude_global, epsilon())
    mlp.compile(
        optimizer='adam',
        loss='mean_squared_error', # Para regressão, a perda é o MSE
        metrics=[
                RootMeanSquaredError(name='rmse'),
                MeanAbsoluteError(name='mae'),
                get_nrmse(amplitude_global),
                smape
            ]
    )
    print("treinando modelo...")
    history = mlp.fit(
        X_train, 
        y_train, 
        epochs=epochs, # Comece com poucas epochs (ex: 10) e aumente se necessário
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )
    if verbose:
        print(
            f"modelo treinado!\
            \nResultado:\
            \nRMSE:\n   teste:{history.history["rmse"][-1]}   val:{history.history["val_rmse"][-1]}\
            \nMAE:\n    teste:{history.history["mae"][-1]}   val:{history.history["val_mae"][-1]}\
            \nNRMSE:\n  teste:{history.history["nrmse"][-1]}   val:{history.history["val_nrmse"][-1]}\
            \nSMAPE:\n  teste:{history.history["smape"][-1]}   val:{history.history["val_smape"][-1]}"
            )
    
    if plot:
        plt.title(label= 'MLP Val RMSE')
        plt.plot(history.history['val_rmse'])
        plt.show()
        plt.title(label='MLP Val MAE')
        plt.plot(history.history['val_mae'])
        plt.show()
        plt.title(label='MLP Val NRMSE')
        plt.plot(history.history['val_nrmse'])
        plt.show()
        plt.title(label='MLP Val SMAPE')
        plt.plot(history.history['val_smape'])
        plt.show()
    if path_modelo is not None:
        mlp.save(f"../../MODELOS/{path_modelo}")
    if path_metricas is not None:
        dict_model ={
            "idx": 0
            ,"MODELO": "MLP"
            ,"DIM": dimensao
            ,"RMSE": history.history["val_rmse"][-1]
            ,"MAE": history.history["val_mae"][-1]
            ,"NMRSE": history.history['val_nrmse'][-1]
            ,"SMAPE": history.history['val_smape'][-1]
            }

        df = pd.DataFrame(dict_model).set_index("idx")
        header_condition = not os.path.exists(path_metricas)
        df.to_csv(path_metricas,index = False,mode = "a", header=header_condition)
    return history

def criar_e_treinarGRU(dimensao,
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
                        path_modelo = None, 
                        path_metricas = None,
                        plot = True,
                        verbose = True,
                        virada = False):
    print("criando modelo...")
    if virada:
        gru = generate_GRU(input_len,2,output_len, dropout_rate, gru_units)
    else:
        gru = generate_GRU(1,input_len,output_len, dropout_rate, gru_units)

    print("compilando modelo...")
    amplitude_global = np.max(y_train) - np.min(y_train)
    amplitude_global = max(amplitude_global, epsilon())

    gru.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
                RootMeanSquaredError(name='rmse'),
                MeanAbsoluteError(name='mae'),
                get_nrmse(amplitude_global),
                smape
            ]
    )
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    if len(X_val.shape) == 2:
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    print("treinando modelo...")
    history = gru.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    if verbose:
        print(
            f"modelo treinado!\
            \nResultado:\
            \nRMSE:\n   teste:{history.history["rmse"][-1]}   val:{history.history["val_rmse"][-1]}\
            \nMAE:\n    teste:{history.history["mae"][-1]}   val:{history.history["val_mae"][-1]}\
            \nNRMSE:\n  teste:{history.history["nrmse"][-1]}   val:{history.history["val_nrmse"][-1]}\
            \nSMAPE:\n  teste:{history.history["smape"][-1]}   val:{history.history["val_smape"][-1]}"
            )
    
    if plot:
        plt.title(label= 'GRU Val RMSE')
        plt.plot(history.history['val_rmse'])
        plt.show()
        plt.title(label='GRU Val MAE')
        plt.plot(history.history['val_mae'])
        plt.show()
        plt.title(label='GRU Val NRMSE')
        plt.plot(history.history['val_nrmse'])
        plt.show()
        plt.title(label='GRU Val SMAPE')
        plt.plot(history.history['val_smape'])
        plt.show()
    if path_modelo is not None:
        gru.save(f"../../MODELOS/{path_modelo}")
    if path_metricas is not None:
        dict_model ={
            "idx": 0
            ,"MODELO": ["GRU"]
            ,"DIM": [dimensao]
            ,"RMSE": [history.history["val_rmse"][-1]]
            ,"MAE": [history.history["val_mae"][-1]]
            ,"NMRSE": [history.history['val_nrmse'][-1]]
            ,"SMAPE": [history.history['val_smape'][-1]]
            }

        df = pd.DataFrame(dict_model)
        header_condition = not os.path.exists(path_metricas)
        df.to_csv(path_metricas,index = False,mode = "a", header=header_condition)
    return history

def criar_e_treinarLSTM(dimensao,
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
                       path_modelo = None, 
                       path_metricas = None,
                       plot = True,
                       verbose = True):
    print("criando modelo...")
    lstm = generate_LSTM(1,input_len, output_len, dropout_rate, lstm_units)

    print("compilando modelo...")
    amplitude_global = np.max(y_train) - np.min(y_train)
    amplitude_global = max(amplitude_global, epsilon())

    lstm.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=[
                RootMeanSquaredError(name='rmse'),
                MeanAbsoluteError(name='mae'),
                get_nrmse(amplitude_global),
                smape
            ]
    )
    if len(X_train.shape) == 2:
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    if len(X_val.shape) == 2:
        X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    print("treinando modelo...")
    history = lstm.fit(
        X_train, 
        y_train, 
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )
    if verbose:
        print(
            f"modelo treinado!\
            \nResultado:\
            \nRMSE:\n   teste:{history.history["rmse"][-1]}   val:{history.history["val_rmse"][-1]}\
            \nMAE:\n    teste:{history.history["mae"][-1]}   val:{history.history["val_mae"][-1]}\
            \nNRMSE:\n  teste:{history.history["nrmse"][-1]}   val:{history.history["val_nrmse"][-1]}\
            \nSMAPE:\n  teste:{history.history["smape"][-1]}   val:{history.history["val_smape"][-1]}"
            )
    
    if plot:
        plt.title(label= 'LSTM Val RMSE')
        plt.plot(history.history['val_rmse'])
        plt.show()
        plt.title(label='LSTM Val MAE')
        plt.plot(history.history['val_mae'])
        plt.show()
        plt.title(label='LSTM Val NRMSE')
        plt.plot(history.history['val_nrmse'])
        plt.show()
        plt.title(label='LSTM Val SMAPE')
        plt.plot(history.history['val_smape'])
        plt.show()
    if path_modelo is not None:
        lstm.save(f"../../MODELOS/{path_modelo}")
    if path_metricas is not None:
        dict_model ={
            "idx": 0
            ,"MODELO": ["LSTM"]
            ,"DIM": [dimensao]
            ,"RMSE": [history.history["val_rmse"][-1]]
            ,"MAE": [history.history["val_mae"][-1]]
            ,"NMRSE": [history.history['val_nrmse'][-1]]
            ,"SMAPE": [history.history['val_smape'][-1]]
            }

        df = pd.DataFrame(dict_model)
        header_condition = not os.path.exists(path_metricas)
        df.to_csv(path_metricas,index = False,mode = "a", header=header_condition)
    return history

def criar_e_treinarRNN(dimensao,
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
                          path_modelo = None, 
                          path_metricas = None,
                          plot = True,
                          verbose = True):
    print("criando modelo...")
    rnn = generate_RNN(
         n_timesteps = 1,
         n_features=input_len,
         n_outputs = output_len,
         dropout_rate = dropout_rate,
         rnn_units = rnn_units
    )
    
    print("compilando modelo...")
    amplitude_global = np.max(y_train) - np.min(y_train)
    amplitude_global = max(amplitude_global, epsilon())
    
    rnn.compile(
          optimizer='adam',
          loss='mean_squared_error',
          metrics=[
                 RootMeanSquaredError(name='rmse'),
                 MeanAbsoluteError(name='mae'),
                 get_nrmse(amplitude_global),
                 smape
                ]
     )
    if len(X_train.shape) == 2:
          X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    if len(X_val.shape) == 2:
          X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    print("treinando modelo...")
    history = rnn.fit(
          X_train, 
          y_train, 
          epochs=epochs,
          batch_size=batch_size,
          validation_data=(X_val, y_val),
          verbose=0
     )
    if verbose:
          print(
                f"modelo treinado!\
                \nResultado:\
                \nRMSE:\n   teste:{history.history["rmse"][-1]}   val:{history.history["val_rmse"][-1]}\
                \nMAE:\n    teste:{history.history["mae"][-1]}   val:{history.history["val_mae"][-1]}\
                \nNRMSE:\n  teste:{history.history["nrmse"][-1]}   val:{history.history["val_nrmse"][-1]}\
                \nSMAPE:\n  teste:{history.history["smape"][-1]}   val:{history.history["val_smape"][-1]}"
                )
     
    if plot:
          plt.title(label= 'RNN Val RMSE')
          plt.plot(history.history['val_rmse'])
          plt.show()
          plt.title(label='RNN Val MAE')
          plt.plot(history.history['val_mae'])
          plt.show()
          plt.title(label='RNN Val NRMSE')
          plt.plot(history.history['val_nrmse'])
          plt.show()
          plt.title(label='RNN Val SMAPE')
          plt.plot(history.history['val_smape'])
          plt.show()
    if path_modelo is not None:
          rnn.save(f"../../MODELOS/{path_modelo}")
    if path_metricas is not None:
            dict_model ={
                "idx": 0
                ,"MODELO": ["RNN"]
                ,"DIM": [dimensao]
                ,"RMSE": [history.history["val_rmse"][-1]]
                ,"MAE": [history.history["val_mae"][-1]]
                ,"NMRSE": [history.history['val_nrmse'][-1]]
                ,"SMAPE": [history.history['val_smape'][-1]]
                }
    
            df = pd.DataFrame(dict_model)
            header_condition = not os.path.exists(path_metricas)
            df.to_csv(path_metricas,index = False,mode = "a", header=header_condition)
    return history



################################################################
#_____________OTIMIZAÇÃO DE HIPERPARÂMETROS____________________#
################################################################



def otimizar_GRU(X_train, 
                 y_train, 
                 X_val, 
                 y_val, 
                 dimensao, 
                 input_len, 
                 output_len,
                 path_modelo = None):
    
    def objective(trial):
        # 1. Definir o espaço de busca que será passado para a sua função
        # Obs: Você precisará adicionar 'epochs' e 'batch_size' na assinatura da sua função 'criar_e_treinarMLP'
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 30, step=10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        gru_units = trial.suggest_int('gru_units', 32, 128)
        
        try:
            # Chama a SUA função
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
                path_modelo= None,
                plot=False,
                verbose=False
)
            return history.history["val_nrmse"][-1] # O Optuna vai tentar minimizar este valor
            
        except Exception as e:
            # Poda o teste se a rede explodir (NaN) ou der erro
            raise optuna.exceptions.TrialPruned()

    # 2. Criar e rodar o estudo
    study = optuna.create_study(direction='minimize', study_name="Otimizacao_GRU")
    study.optimize(objective, n_trials=20) # Define quantas variações testar

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
        epochs=study.best_params['epochs'],
        batch_size=study.best_params['batch_size'],
        dropout_rate=study.best_params['dropout_rate'],
        gru_units=study.best_params['gru_units'],
        path_modelo=path_modelo,
        plot=True
    )
    return study

def otimizar_LSTM(X_train, 
                 y_train, 
                 X_val, 
                 y_val, 
                 dimensao, 
                 input_len, 
                 output_len,
                 path_modelo = None):
    
    def objective(trial):
        # 1. Definir o espaço de busca que será passado para a sua função
        # Obs: Você precisará adicionar 'epochs' e 'batch_size' na assinatura da sua função 'criar_e_treinarMLP'
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 30, step=10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        lstm_units = trial.suggest_int('lstm_units', 32, 128)
        
        try:
            # Chama a SUA função
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
                path_modelo= None,
                plot=False,
                verbose=False
)
            return history.history["val_nrmse"][-1] # O Optuna vai tentar minimizar este valor
            
        except Exception as e:
            # Poda o teste se a rede explodir (NaN) ou der erro
            raise optuna.exceptions.TrialPruned()

    # 2. Criar e rodar o estudo
    study = optuna.create_study(direction='minimize', study_name="Otimizacao_LSTM")
    study.optimize(objective, n_trials=20) # Define quantas variações testar

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
        epochs=study.best_params['epochs'],
        batch_size=study.best_params['batch_size'],
        dropout_rate=study.best_params['dropout_rate'],
        lstm_units=study.best_params['lstm_units'],
        path_modelo=path_modelo,
        plot=True
    )
    return study

def otimizar_RNN(X_train,
                    y_train, 
                    X_val, 
                    y_val, 
                    dimensao, 
                    input_len, 
                    output_len,
                    path_modelo = None):
        
        def objective(trial):
            # 1. Definir o espaço de busca que será passado para a sua função
            # Obs: Você precisará adicionar 'epochs' e 'batch_size' na assinatura da sua função 'criar_e_treinarMLP'
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            epochs = trial.suggest_int('epochs', 10, 30, step=10)
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            rnn_units = trial.suggest_int('rnn_units', 32, 128)
            
            try:
                # Chama a SUA função
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
                    path_modelo= None,
                    plot=False,
                    verbose=False
    )
                return history.history["val_nrmse"][-1] # O Optuna vai tentar minimizar este valor
                
            except Exception as e:
                # Poda o teste se a rede explodir (NaN) ou der erro
                raise optuna.exceptions.TrialPruned()
    
        # 2. Criar e rodar o estudo
        study = optuna.create_study(direction='minimize', study_name="Otimizacao_RNN")
        study.optimize(objective, n_trials=20) # Define quantas variações testar
    
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
            epochs=study.best_params['epochs'],
            batch_size=study.best_params['batch_size'],
            dropout_rate=study.best_params['dropout_rate'],
            rnn_units=study.best_params['rnn_units'],
            path_modelo=path_modelo,
            plot=True
        )
        return study


def otimizar_MLP(X_train, 
                 y_train, 
                 X_val, 
                 y_val, 
                 dimensao, 
                 input_len, 
                 output_len,
                 path_modelo = None):
    
    def objective(trial):
        # 1. Definir o espaço de busca que será passado para a sua função
        # Obs: Você precisará adicionar 'epochs' e 'batch_size' na assinatura da sua função 'criar_e_treinarMLP'
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        epochs = trial.suggest_int('epochs', 10, 30, step=10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
        dense_units = trial.suggest_int('dense_units', 100, 500)
        layers = trial.suggest_int('layers', 1, 3)
        
        try:
            # Chama a SUA função
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
                path_modelo= None,
                plot=False,
                verbose=False
)
            return history.history["val_nrmse"][-1] # O Optuna vai tentar minimizar este valor
            
        except Exception as e:
            # Poda o teste se a rede explodir (NaN) ou der erro
            raise optuna.exceptions.TrialPruned()

    # 2. Criar e rodar o estudo
    study = optuna.create_study(direction='minimize', study_name="Otimizacao_MLP")
    study.optimize(objective, n_trials=20) # Define quantas variações testar

    print(f"\nMelhor NRMSE: {study.best_value}")
    print(f"Melhores parâmetros: {study.best_params}")
    criar_e_treinarMLP(
        dimensao=dimensao,
        input_len=input_len,
        output_len=output_len,
        dropout_rate=study.best_params['dropout_rate'],
        dense_units=study.best_params['dense_units'],
        layers=study.best_params['layers'],
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=study.best_params['epochs'],
        batch_size=study.best_params['batch_size'],
        path_modelo=path_modelo,
        plot=True
    )
    return study