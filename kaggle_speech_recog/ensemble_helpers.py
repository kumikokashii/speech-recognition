import pickle
import numpy as np

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
