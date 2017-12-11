import pickle
from datetime import datetime
import numpy as np

from bokeh.plotting import figure, show
from ..bokeh4github import show
from bokeh.models import NumeralTickFormatter, PrintfTickFormatter, Span, Legend
from bokeh.layouts import column

color_ave_valid = '#66b266'
color_valid = '#b2d8b2'
color_train = '#ededed'
color_ver = '#ffdb94'

color_list = ['#FD367E', '#ffc04c', '#3F6699', '#B0DEDB', '#af9880']  # dark pink, orange, blue, light green, light brown
color_tint = ['#fed6e5', '#fff2db', '#d8e0ea', '#eff8f7', '#efeae5']

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
        self.epochs = []
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
        
    def record(self, step, epoch, accu_train, ll_train, accu_valid, ll_valid):
        self.steps.append(step)
        self.epochs.append(epoch)
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
        
    def add_ave_accu_valid(self):
        n = self.t_cnfg.n_ave_ll_valid
        
        self.ave_accu_valid = []
        for i in range(len(self.accu_valid)):
            start = max(i-n+1, 0)
            mini_accu_valid = self.accu_valid[start: i+1]
            ave_accu_valid = sum(mini_accu_valid) / n
            self.ave_accu_valid.append(ave_accu_valid)
        
    def get_accuracy_graph(self):
        title = 'Accuracy: ' + ' > '.join([self.graph_name, self.g_cnfg.name, self.t_cnfg.name])
        p = figure(title=title, plot_width=1000, plot_height=400)

        p_list = [['train accuracy', self.accu_train, 'solid', color_train],
                  ['valid. accuracy', self.accu_valid, 'dotdash', color_valid],
                  ['average valid. accuracy', self.ave_accu_valid, 'solid', color_ave_valid]]

        for i in range(len(p_list)):
            name, array, dash, color = p_list[i]
            line = p.line(self.steps, array, line_dash=dash, line_width=3, color=color)
            p_list[i].append(line)

            if i == 2:
                # Add vertical line at max accuracy
                i_max = np.argmax(array)
                max_line = Span(location=self.steps[i_max], dimension='height',
                                line_dash='dashed', line_width=3, line_color=color_ver)
                p.add_layout(max_line)            
            
        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'accuracy'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = NumeralTickFormatter(format='0.00%')

        legend = Legend(items=[(name, [line]) for name, array, dash, color, line in p_list],
                        location='bottom_right',
                        orientation='vertical',
                        click_policy='hide')
        p.add_layout(legend)
        
        return p
        
    def get_logloss_graph(self):
        title = 'Logloss: ' + ' > '.join([self.graph_name, self.g_cnfg.name, self.t_cnfg.name])
        p = figure(title=title, plot_width=1000, plot_height=400)

        p_list = [['train logloss', self.ll_train, 'solid', color_train],
                  ['valid. logloss', self.ll_valid, 'dotdash', color_valid],
                  ['average valid. logloss', self.ave_ll_valid, 'solid', color_ave_valid]]

        for i in range(len(p_list)):
            name, array, dash, color = p_list[i]
            line = p.line(self.steps, array, line_dash=dash, line_width=3, color=color)
            p_list[i].append(line)
            
            if i == 2:
                # Add vertical line at min logloss
                i_min = np.argmin(array)
                min_line = Span(location=self.steps[i_min], dimension='height',
                                line_dash='dashed', line_width=3, line_color=color_ver)
                p.add_layout(min_line)

        p.xaxis.axis_label = 'steps'
        p.yaxis.axis_label = 'logloss'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = PrintfTickFormatter(format='%.3f')

        legend = Legend(items=[(name, [line]) for name, array, dash, color, line in p_list],
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

    def get_epoch_summary(self):
        self.add_ave_accu_valid()
        
        epochs = []
        ave_accu_valid = []
        ave_ll_valid = []
        
        current_epoch = 1
        for i in range(len(self.epochs)):
            if current_epoch < self.epochs[i]:
                epochs.append(current_epoch)
                ave_accu_valid.append(self.ave_accu_valid[i-1])
                ave_ll_valid.append(self.ave_ll_valid[i-1])
                current_epoch += 1
                
        return epochs, ave_accu_valid, ave_ll_valid
    
    def compare(path2logs, accuracy=True, logloss=True):  # Static
        if accuracy and logloss:
            accuracy_graph = Log.get_accuracy_compare(path2logs)
            logloss_graph = Log.get_logloss_compare(path2logs)

            p = column(accuracy_graph, logloss_graph)
            show(p)
            return

        if accuracy:
            show(Log.get_accuracy_compare(path2logs))
            return

        if logloss:
            show(Log.get_logloss_compare(path2logs))
            return    

    def get_accuracy_compare(path2logs):  # Static
        p = figure(title='Compare Accuracy', plot_width=1000, plot_height=400)
        legend_items = []

        i_color = 0
        for path2log in path2logs:
            log = pickle.load(open(path2log, 'rb'))
            epochs, ave_accu_valid, ave_ll_valid = log.get_epoch_summary()
            name = ' > '.join([log.graph_name, log.g_cnfg.name, log.t_cnfg.name])
            line = p.line(epochs, ave_accu_valid, line_width=3, color=color_list[i_color])
            legend_items.append((name, [line]))
            
            # Add vertical line at max accuracy
            i_max = np.argmax(ave_accu_valid)
            max_line = Span(location=epochs[i_max], dimension='height',
                            line_dash='dashed', line_width=3, line_color=color_tint[i_color])
            p.add_layout(max_line)

            # Next colors            
            i_color = (i_color + 1) % len(color_list)

        p.xaxis.axis_label = 'epochs'
        p.yaxis.axis_label = 'accuracy'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = NumeralTickFormatter(format='0.00%')

        legend = Legend(items=legend_items,
                        location='bottom_right',
                        orientation='vertical',
                        click_policy='hide')
        p.add_layout(legend)

        return p

    def get_logloss_compare(path2logs):  # Static
        p = figure(title='Compare Logloss', plot_width=1000, plot_height=400)
        legend_items = []

        i_color = 0
        for path2log in path2logs:
            log = pickle.load(open(path2log, 'rb'))
            epochs, ave_accu_valid, ave_ll_valid = log.get_epoch_summary()
            name = ' > '.join([log.graph_name, log.g_cnfg.name, log.t_cnfg.name])
            line = p.line(epochs, ave_ll_valid, line_width=3, color=color_list[i_color])
            legend_items.append((name, [line]))
            
            # Add vertical line at min logloss
            i_min = np.argmin(ave_ll_valid)
            min_line = Span(location=epochs[i_min], dimension='height',
                            line_dash='dashed', line_width=3, line_color=color_tint[i_color])
            p.add_layout(min_line)

            # Next colors
            i_color = (i_color + 1) % len(color_list)

        p.xaxis.axis_label = 'epochs'
        p.yaxis.axis_label = 'logloss'
        p.xaxis.formatter = NumeralTickFormatter(format='0,000')
        p.yaxis.formatter = PrintfTickFormatter(format='%.3f')

        legend = Legend(items=legend_items,
                        location='top_right',
                        orientation='vertical',
                        click_policy='hide')
        p.add_layout(legend)

        return p