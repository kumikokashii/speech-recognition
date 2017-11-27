import pickle
from datetime import datetime

class Config():
    def __init__(self, name):
        self.name = name
        
    def __str__(self):
        sorted_attr = sorted(self.__dict__)
        sorted_attr.remove('name')
        sorted_attr.insert(0, 'name')
        
        output = ''
        for key in sorted_attr:
            value = self.__dict__[key]
            if isinstance(value, int):
                output += '{}: {:,}'.format(key, value)
            else:
                output += '{}: {}'.format(key, value)
            output += '\n'
        return output

class Log():
    def __init__(self, log_dir, joined_name, g_cnfg, t_cnfg):
        self.save_as = log_dir + '/' + joined_name + '.log'
        self.g_cnfg = g_cnfg
        self.t_cnfg = t_cnfg
        self.n_ave_ll_valid = t_cnfg.n_ave_ll_valid

        self.index = []
        self.steps = []
        self.ll_train = []
        self.ll_valid = []
        self.ave_ll_valid = []
        self.end_time = []

        self.best_model_step = -1
        self.best_model_ll = float('inf')
        self.best_model_patient_till = None
        
        self.train_start = None
        self.train_end = None
        
        self.save()
        
    def save(self):
        pickle.dump(self, open(self.save_as, 'wb'))     
        
    def record(self, step, ll_train, ll_valid):
        self.index.append(len(self.index))
        self.steps.append(step)
        self.ll_train.append(ll_train)
        self.ll_valid.append(ll_valid)
        
        ave_ll_valid = self.get_ave_ll_valid()
        self.ave_ll_valid.append(ave_ll_valid)
        
        self.end_time.append(datetime.now())

    def get_ave_ll_valid(self):
        if len(self.ll_valid) < self.n_ave_ll_valid:
            start = 0
            count = len(self.ll_valid)
        else:
            start = len(self.ll_valid) - self.n_ave_ll_valid
            count = self.n_ave_ll_valid
        mini_ll_valid = self.ll_valid[start: start+count]
        ave_ll_valid = sum(mini_ll_valid) / float(count)
        
        return ave_ll_valid
    
    def update_best_model(self, patient_till):
        self.best_model_step = self.steps[-1]
        self.best_model_ll = self.ave_ll_valid[-1]
        self.best_model_patient_till = patient_till
        