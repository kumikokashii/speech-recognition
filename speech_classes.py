from scipy.io import wavfile
import IPython.display as ipd
import numpy as np

from bokeh.layouts import row
from bokeh.plotting import figure, show
from bokeh.models import NumeralTickFormatter
from bokeh.io import output_notebook
output_notebook()

from datetime import datetime
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import os

class Speech():
    def __init__(self, file_path, is_test=True, label=None):
        self.file_path = file_path
        self.is_test = is_test
        self.label = label
        self.sample_rate = None
        self.data = None
        self.data_len = None
        
    def __str__(self):
        return '{}, {}, sample rate {}, data length {}'.format(self.label, self.file_path, self.sample_rate, self.data_len)

    def get_wav_data(self):
        self.sample_rate, self.data = wavfile.read(self.file_path)
        self.data_len = len(self.data)
        
    def show_audio(self):
        ipd.display(ipd.Audio(self.file_path))
    
    def show_graph(self):
        p = figure(plot_width=1000, plot_height=400)
        p.line(np.arange(len(self.data)), self.data, line_width=1)
        show(p)
    
    def hear_and_see(self):
        print(str(self))
        self.show_audio()
        self.show_graph()
        
    def get_data_array_of_length(self, vector_len):
        if self.data_len == vector_len:
            return np.copy(self.data)
        
        if self.data_len < vector_len:  # Pad with zeros at the end
            output = np.zeros(vector_len)
            output[: self.data_len] = self.data
            return output
        
        return self.data[: vector_len]  # Trim the end
        
class SpeechList(list):
    def __init__(self, name):
        super().__init__()
        self.name = name
        
    def get_labels(self):
        labels = []
        for speech in self:
            labels.append(speech.label)
        return list(set(labels))
        
    def get_file_count_per_label(self):
        label_dict = {}
        for speech in self:
            label = speech.label
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
        return label_dict
        
    def get_wav_data(self, annotate=True):
        if annotate:
            start_time = datetime.now()
            print('[ {} ]'.format(self.name))
            print('Getting wav data started @ {:%m/%d/%Y %H:%M:%S}'.format(start_time))

        for i in range(len(self)):
            speech = self[i]
            
            try:
                speech.get_wav_data()
            except ValueError as e:
                print('SKIPPED {} {}'.format(speech.file_path, e))
                continue
            
            if not annotate:
                continue
            if (i > 0) and (i % 50000 == 0):  # Print every 50k files
                now = datetime.now()
                passed_min = (now - start_time).seconds // 60
                passed_sec = (now - start_time).seconds % 60
                print('Completed {:,} data @ {:%m/%d/%Y %H:%M:%S} ({} min {} sec passed)'.format(i, now, passed_min, passed_sec))

        if annotate:
            end_time = datetime.now()
            passed_min = (end_time - start_time).seconds // 60
            passed_sec = (end_time - start_time).seconds % 60
            print('Completed all data @ {:%m/%d/%Y %H:%M:%S} ({} min {} sec passed)'.format(end_time, passed_min, passed_sec))
        
    def get_speech_by_file_path(self, file_path):
        for speech in self:
            if speech.file_path == file_path:
                return speech
        return None
    
    def remove_speech_by_file_path(self, file_path):
        speech = self.get_speech_by_file_path(file_path)
        if speech is not None:
            self.remove(speech)
    
    def get_list_of_label(self, label):
        list_ = []
        for speech in self:
            if speech.label == label:
                list_.append(speech)
        return list_
        
    def get_random(self, label=None):
        if label is None:
            list_ = self
        else:
            list_ = self.get_list_of_label(label)
        return random.choice(list_)
    
    def get_stats(self):
        
        # Find most frequently found data size
        list_ = [speech.data_len for speech in self]
        len_ = len(list_)
        hist, edges = np.histogram(list_, range=(0, max(list_)), bins=max(list_)+1)
        most_often_data_size = np.argmax(hist)

        # Done if only one one data size
        if hist[most_often_data_size] == len_:
            print('All {:,} files are data size {:,}'.format(len_, most_often_data_size))
            return

        # Find stats on files of less than or more than the most frequent data size
        less_than_most_often_list = []
        more_than_most_often_list = []
        for data_len in list_:
            if data_len < most_often_data_size:
                less_than_most_often_list.append(data_len)
            elif data_len > most_often_data_size:
                more_than_most_often_list.append(data_len)

        print('Most often data size: {:,} ({:.2f}% of {} set i.e. {:,} out of {:,} files)'.format(most_often_data_size,
              hist[most_often_data_size]/len_*100, self.name, hist[most_often_data_size], len_))

        print('Less than data size {:,}: {:.2f}% of {} set i.e. {:,} out of {:,} files'.format(most_often_data_size,
              len(less_than_most_often_list)/len_*100, self.name, len(less_than_most_often_list), len_))

        print('More than data size {:,}: {:.2f}% of {} set i.e. {:,} out of {:,} files'.format(most_often_data_size,
              len(more_than_most_often_list)/len_*100, self.name, len(more_than_most_often_list), len_)) 

        def draw_histogram(title, hist, edges, color=None):
            p = figure(title=title, width=450, height=250)
            if color is None:
                p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:])
            else:
                p.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:], color=color)
            p.xaxis.axis_label = 'data size'
            p.yaxis.axis_label = '# of files'
            p.xaxis.formatter = NumeralTickFormatter(format='0,000')
            p.yaxis.formatter = NumeralTickFormatter(format='0,000')
            return p

        hist1, edges1 = np.histogram(less_than_most_often_list, range=(0, most_often_data_size-1), bins=50)
        p1 = draw_histogram(title='# of Files with Data Size < {:,}'.format(most_often_data_size),
                            hist=hist1, edges=edges1)

        hist2, edges2 = np.histogram(more_than_most_often_list, bins=50)
        p2 = draw_histogram(title='# of Files with Data Size > {:,}'.format(most_often_data_size),
                            hist=hist2, edges=edges2, color='#9ecae1')

        p = row(p1, p2)
        show(p)
        
    def get_feature_matrix(self, vector_len):
        list_ = []
        for speech in self:
            list_.append(speech.get_data_array_of_length(vector_len))
        return np.matrix(list_)
    
    def get_label_matrix(self):
        # Labels in one dimension
        list_ = []
        for speech in self:
            list_.append(speech.label)

        # Label indexes in one dimension
        le = LabelEncoder()
        i_list = le.fit_transform(list_)

        # One hot encode label indexes
        i_list_reshaped = [[i] for i in i_list]
        enc = OneHotEncoder(sparse=False)
        return enc.fit_transform(i_list_reshaped)
    
    def get_X_and_y_matrices(self, X_vector_len, split=None):
        X = self.get_feature_matrix(X_vector_len)
        y = self.get_label_matrix()
        
        if split is None:
            return X, y
        
        X1, X2, y1, y2 = train_test_split(X, y, test_size=(1-split), random_state=0)
        return X1, y1, X2, y2
    
    def get_train(path2files_dir):  # Static
        train = SpeechList(name='Train')
        for sub_dir in os.listdir(path2files_dir):
            path2sub_dir = path2files_dir + '/' + sub_dir
            if os.path.isfile(path2sub_dir):
                continue
            for file in os.listdir(path2sub_dir):
                speech = Speech(path2sub_dir + '/' + file, is_test=False, label=sub_dir)
                train.append(speech)

        train.get_wav_data(annotate=False)
        non_wav_file_path = 'train/audio/_background_noise_/README.md'
        train.remove_speech_by_file_path(non_wav_file_path)
        return train
    
    def get_test(path2files_dir):  # Static
        test = SpeechList(name='Test')
        for file in os.listdir(path2files_dir):
            speech = Speech(path2files_dir + '/' + file)
            test.append(speech)

        test.get_wav_data(annotate=False)
        return test