import pickle
from datetime import datetime

from bokeh.plotting import figure, show
from ..bokeh4github import show
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter, Legend
from bokeh.layouts import column


class Log():
    def __init__(self, log_dir, joined_name, graph_name, ckp_dir, tb_dir, g_cnfg, t_cnfg):
        self.save_as = log_dir + '/' + joined_name + '.log'
        self.joined_name = joined_name
        self.graph_name = graph_name
        self.ckp_dir = ckp_dir
        self.tb_dir = tb_dir
        self.g_cnfg = g_cnfg
        self.t_cnfg = t_cnfg
        self.n_ave_ll_valid = t_cnfg.n_ave_ll_valid

        self.steps = []
        self.accu_train = []
        self.ll_train = []
        self.accu_valid = []
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
        
    def record(self, step, accu_train, ll_train, accu_valid, ll_valid):
        self.steps.append(step)
        self.accu_train.append(accu_train)
        self.ll_train.append(ll_train)
        self.accu_valid.append(accu_valid)
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

    def get_accuracy_graph(self):
        title = 'Accuracy: ' + ' > '.join([self.graph_name, self.g_cnfg.name, self.t_cnfg.name])
        p = figure(title=title, plot_width=1000, plot_height=400)

        p_list = [['valid. accuracy', self.accu_valid, '#e1f7d5'],  # green
                  ['train accuracy', self.accu_train, '#c9c9ff']]  # blue

        for i in range(len(p_list)):
            name, array, color = p_list[i]
            line = p.line(self.steps, array, line_width=3, color=color)
            p_list[i].append(line)

        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'accuracy'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = NumeralTickFormatter(format='0.00%')

        legend = Legend(items=[(name, [line]) for name, array, color, line in p_list],
                        location='top_left',
                        orientation='vertical',
                        click_policy='hide')
        p.add_layout(legend)
        
        return p
        
    def get_logloss_graph(self):
        title = 'Logloss: ' + ' > '.join([self.graph_name, self.g_cnfg.name, self.t_cnfg.name])
        p = figure(title=title, plot_width=1000, plot_height=400)

        p_list = [['average valid. logloss', self.ave_ll_valid, '#ffb3ba'],  # red
                  ['valid. logloss', self.ll_valid, '#e1f7d5'],  # green
                  ['train logloss', self.ll_train, '#c9c9ff']]  # blue

        for i in range(len(p_list)):
            name, array, color = p_list[i]
            line = p.line(self.steps, array, line_width=3, color=color)
            p_list[i].append(line)

        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'logloss'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = PrintfTickFormatter(format='%.3f')

        legend = Legend(items=[(name, [line]) for name, array, color, line in p_list],
                        location='top_right',
                        orientation='vertical',
                        click_policy='hide')
        p.add_layout(legend)
        
        return p
        
    def show_progress(self, accuracy=True, logloss=True):
        if accuracy and logloss:
            accuracy_graph = self.get_accuracy_graph()
            logloss_graph = self.get_logloss_graph()

            p = column(accuracy_graph, logloss_graph)
            show(p)
            return
        
        if accuracy:
            show(self.get_accuracy_graph())
            return
        
        if logloss:
            show(self.get_logloss_graph())
            return
