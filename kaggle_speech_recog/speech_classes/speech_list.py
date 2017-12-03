import numpy as np

from bokeh.plotting import figure, show
from ..bokeh4github import show
from bokeh.models import NumeralTickFormatter
from bokeh.layouts import row

from datetime import datetime
import random
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from .speech import *
import os
import pandas as pd

        
class SpeechList(list):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.le = None
        
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
        self.le = LabelEncoder()
        i_list = self.le.fit_transform(list_)

        # One hot encode label indexes
        i_list_reshaped = [[i] for i in i_list]
        enc = OneHotEncoder(sparse=False)
        return enc.fit_transform(i_list_reshaped)
    
    def get_X_and_Y_matrices(self, X_vector_len, split=None):
        X = self.get_feature_matrix(X_vector_len)
        Y = self.get_label_matrix()
        
        if split is None:
            return X, Y
        
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=(1-split), random_state=0)
        return X1, Y1, X2, Y2
    
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
    
    def get_test(path2files_dir, first=None):  # Static
        test = SpeechList(name='Test')
        count = 0
        for file in os.listdir(path2files_dir):
            speech = Speech(path2files_dir + '/' + file)
            test.append(speech)
            count += 1
            if (first is not None) and (count == first):
                break

        test.get_wav_data(annotate=False)
        return test

    def add_predicted_label(self, Y, le):
        self.le = le
        i_list = np.argmax(Y, axis=1)
        list_ = list(self.le.inverse_transform(i_list))
        
        for i in range(len(self)):
            self[i].predicted_label = list_[i]

    def save_submission_csv(self, dir_, name):
        files = []
        labels = []
        for speech in self:
            i_last_slash = speech.file_path.rfind('/')
            files.append(speech.file_path[i_last_slash+1:])
            labels.append(speech.predicted_label)            
        df = pd.DataFrame({'fname': files, 'label': labels})
        
        include = ['yes', 'no' , 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        df.loc[[(label not in include) for label in df['label']], 'label'] = 'unknown'
        
        save_as = '/'.join([dir_, name])
        df.to_csv(save_as, index=False)
        