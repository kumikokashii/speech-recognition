import pickle
import numpy as np
import xgboost as xgb

from .graphs import *

def gather_logits(X, log_dir, logs):
    for i in range(len(logs)):
        log_name = logs[i]
        path2log = '/'.join([log_dir, log_name])
        log = pickle.load(open(path2log, 'rb'))
        
        # Band aid
        if log_name.startswith('NoveltyDetectionSpectrogramMultiLSTMRandomInputModify'):
            log.g_cnfg.Y_vector_len = 1
        if log_name in ['SpectrogramMultiLSTMRandomInputModify_graph_03_finer_layers_run_01.log', 
                        'SpectrogramMultiLSTMRandomInputModify_graph_04_kindof_finer_layers_run_01.log']:
            log.graph_name = 'SpectrogramMultiLSTMRandomInputModify2hiddens'
            
        GraphClass = globals()[log.graph_name]  # Pick up graph class used to train model
        graph = GraphClass(log.g_cnfg)  # Load the same graph configuration

        i_L = graph.get_logits(X, ckp_dir=log.ckp_dir, batch_size=log.t_cnfg.batch_size, annotate=False)
        if i == 0:
            L = i_L
        else:
            L = np.concatenate((L, i_L), axis=1)
    return L

def get_values_with_sklearn_models(list_X, model_dict):
    list_values = []
    for X in list_X:
        n_data = X.shape[0]
        n_models = len(model_dict)
        values = np.empty([n_data, n_models])
        
        for i in model_dict:
            values[:, i] = model_dict[i].decision_function(X).reshape(n_data)        
        list_values.append(values)
        
    return list_values

def get_values_with_xgboost(list_X, xgboost):
    list_values = []
    for X in list_X:
        xgb_X = xgb.DMatrix(X)
        values = xgboost.predict(xgb_X, ntree_limit=xgboost.best_ntree_limit)
        list_values.append(values)
    return list_values

def get_concatenated(to_concatenate):
    n_parts = len(to_concatenate[0])
    list_concatenated = []
    for i in range(n_parts):
        to_concatenate_per_part = [to_concatenate[j][i] for j in range(len(to_concatenate))]
        concatenated_per_part = np.concatenate(to_concatenate_per_part, axis=1)
        list_concatenated.append(concatenated_per_part)
    return list_concatenated
