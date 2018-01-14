import pickle
import numpy as np

from .graphs import *

def gather_logits(X, log_dir, logs):
    for i in range(len(logs)):
        path2log = '/'.join([log_dir, logs[i]])
        log = pickle.load(open(path2log, 'rb'))
        GraphClass = globals()[log.graph_name]  # Pick up graph class used to train model
        graph = GraphClass(log.g_cnfg)  # Load the same graph configuration

        i_L = graph.get_logits(X, ckp_dir=log.ckp_dir, batch_size=log.t_cnfg.batch_size, annotate=False)
        if i == 0:
            L = i_L
        else:
            L = np.concatenate((L, i_L), axis=1)
    return L
