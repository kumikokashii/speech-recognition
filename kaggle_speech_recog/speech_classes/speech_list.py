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
from copy import deepcopy

        
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
    
    def get_list_of_label(self, label_list):
        list_ = SpeechList(name=', '.join(label_list))
        for speech in self:
            if speech.label in label_list:
                list_.append(speech)
        return list_

    def get_list_of_predicted_label(self, label_list):
        list_ = SpeechList(name=', '.join(label_list))
        for speech in self:
            if speech.predicted_label in label_list:
                list_.append(speech)
        return list_  
    
    def get_random(self, label=None):
        if label is None:
            list_ = self
        else:
            list_ = self.get_list_of_label([label])
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
    
    def get_label_matrix(self, group_unknown=False):
        # Labels in one dimension
        list_ = []
        
        if group_unknown:
            in_group = ['yes', 'no' , 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', '_background_noise_']
            i_unknown = []
            
        for i in range(len(self)):  # Loop through speeches
            label = self[i].label
            if group_unknown and (label not in in_group):
                label = 'unknown'
                i_unknown.append(i)
            list_.append(label)

        # Label indexes in one dimension
        self.le = LabelEncoder()
        i_list = self.le.fit_transform(list_)

        # One hot encode label indexes
        i_list_reshaped = [[i] for i in i_list]
        enc = OneHotEncoder(sparse=False)
        
        if group_unknown:
            return enc.fit_transform(i_list_reshaped), i_unknown
        return enc.fit_transform(i_list_reshaped)
    
    def get_novelty_det_label_matrix(self, novelty_det_in_class):
        list_ = []
        for speech in self:
            in_class = 1 if speech.label in novelty_det_in_class else 0
            list_.append([in_class])
        return np.array(list_)
    
    def get_X_and_Y_matrices(self, X_vector_len, split=None):
        X = self.get_feature_matrix(X_vector_len)
        Y = self.get_label_matrix()
        
        if split is None:
            return X, Y
        
        X1, X2, Y1, Y2 = train_test_split(X, Y, test_size=(1-split), random_state=0)
        return X1, Y1, X2, Y2

    def get_spectrogram_feature_ndarray(self, vector_len, spec_v=None, take_log=False):
        list_ = []
        for speech in self:
            speech.set_spectrogram(vector_len, spec_v, take_log)
            list_.append(speech.spec_data)
        return np.array(list_)
    
    def split_noise(self, vector_len):
        orig_list = self.copy()  # Copy of original list
        self.clear()  # Empty list
        for speech in orig_list:
            if speech.label != '_background_noise_':
                self.append(speech)
                continue
                
            offset = 0
            while (offset < speech.data_len):
                new_speech = deepcopy(speech)
                new_speech.data = speech.data[offset: offset+vector_len]
                new_speech.data_len = len(new_speech.data)
                self.append(new_speech)
                offset += vector_len
                
    def fabricate_noise(self, n, vector_len):
        noise_speech_list = self.get_list_of_label(['_background_noise_'])
        n_noise_speech = len(noise_speech_list)       
        
        for i in range(n):
            i1, i2 = np.random.choice(n_noise_speech, 2)
            s1_data = noise_speech_list[i1].get_data_array_of_length(vector_len)
            s2_data = noise_speech_list[i2].get_data_array_of_length(vector_len)
            ratio = np.random.choice(101, 1) / 100

            new_noise_speech = deepcopy(noise_speech_list[0])
            new_noise_speech.file_path = None 
            new_noise_speech.data = (s1_data*ratio) + (s2_data*(1-ratio))
            new_noise_speech.data_len = vector_len
            self.append(new_noise_speech)
            
    def split_known_unknown(self, X, Y, i_unknown):
        i_unknown = np.array(i_unknown)

        X_unknown = X[i_unknown, :]
        Y_unknown = Y[i_unknown, :]

        boolean_known = np.ones(len(X), bool)
        boolean_known[i_unknown] = False
        
        X_known = X[boolean_known]
        Y_known = Y[boolean_known]
        
        return X_known, X_unknown, Y_known, Y_unknown
        
    def get_group_unknown_spectrogram_X_and_Y(self, X_vector_len, split_noise=False, n_fabricate_noise=0, 
                                              spec_v=None, take_log=False, split=None):
        if split_noise:
            self.split_noise(X_vector_len)
        
        if n_fabricate_noise > 0:
            self.fabricate_noise(n_fabricate_noise, X_vector_len)

        X = self.get_spectrogram_feature_ndarray(X_vector_len, spec_v, take_log)
        Y, i_unknown = self.get_label_matrix(group_unknown=True)                   
        
        X_known, X_unknown, Y_known, Y_unknown = self.split_known_unknown(X, Y, i_unknown)
        
        if split is None:
            return X_known, X_unknown, Y_known, Y_unknown

        # 1. Split known into train and test
        X_train_known, X_test_known, Y_train_known, Y_test_known = train_test_split(X_known, Y_known, test_size=(1-split), random_state=0)        
        
        # 2. Pick unknown for test
        n_test_unknown = int(len(X_test_known) / 11)  # 11 known classes total      
        i_test_unknown = np.random.choice(len(X_unknown), size=n_test_unknown, replace=False)
        X_test_unknown = X_unknown[i_test_unknown, :]
        Y_test_unknown = Y_unknown[i_test_unknown, :]
        
        # 3. Rest of unknown is for train
        boolean_train_unknown = np.ones(len(X_unknown), bool)
        boolean_train_unknown[i_test_unknown] = False
        X_train_unknown = X_unknown[boolean_train_unknown]
        Y_train_unknown = Y_unknown[boolean_train_unknown]

        # 4. Return train with separate known and unknown. Together for test.
        X_test = np.concatenate((X_test_known, X_test_unknown))
        Y_test = np.concatenate((Y_test_known, Y_test_unknown))
        
        return X_train_known, X_train_unknown, Y_train_known, Y_train_unknown, X_test, Y_test            

    def get_spectrogram_X_and_Y(self, X_vector_len, split_noise=False, n_fabricate_noise=0,
                                spec_v=None, take_log=False, split=None, 
                                novelty_det=False, novelty_det_in_class=None):
        if split_noise:
            self.split_noise(X_vector_len)
        
        if n_fabricate_noise > 0:
            self.fabricate_noise(n_fabricate_noise, X_vector_len)

        X = self.get_spectrogram_feature_ndarray(X_vector_len, spec_v, take_log)
        if not novelty_det:
            Y = self.get_label_matrix()
        else:
            Y = self.get_novelty_det_label_matrix(novelty_det_in_class)                       
            
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

    def save_submission_csv(self, dir_, name, 
                            use_novelty_det=False, novelty_det_sigmoid_threshold=0.5, 
                            use_svm=False, keep_noise=False):
        files = []
        labels = []
        for speech in self:
            i_last_slash = speech.file_path.rfind('/')
            files.append(speech.file_path[i_last_slash+1:])
            labels.append(speech.predicted_label)            
        df = pd.DataFrame({'fname': files, 'label': labels})
        
        df.loc[df['label'] == '_background_noise_', 'label'] = 'silence'
        include = ['yes', 'no' , 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']
        df.loc[[(label not in include) for label in df['label']], 'label'] = 'unknown'
        col4label = 'label'

        if use_novelty_det:
            name = name[:-4] + '_novelty_det'
                             
            novelty_det_sigmoid = []
            for speech in self:
                novelty_det_sigmoid.append(speech.novelty_det_sigmoid)
            df['novelty det sigmoid'] = novelty_det_sigmoid
            
            df['label with novelty det'] = df['label']
            df.loc[df['novelty det sigmoid'] <= novelty_det_sigmoid_threshold, 'label with novelty det'] = 'unknown'
            df.loc[df['label'] == 'silence', 'label with novelty det'] = 'silence'
            
            name += '.csv'
            col4label = 'label with novelty det'          
        
        if use_svm:
            name = name[:-4] + '_svm'
                             
            n_svm_in_class = []
            for speech in self:
                n_svm_in_class.append(len(speech.svm_in_class))
            df['# svm in class'] = n_svm_in_class
            
            df['label with svm'] = df['label']
            df.loc[df['# svm in class'] == 0, 'label with svm'] = 'unknown'
            if keep_noise:
                name += '_keepnoise'
                df.loc[df['label'] == 'silence', 'label with svm'] = 'silence'
            
            name += '.csv'
            col4label = 'label with svm'        

        save_as = '/'.join([dir_, name])
        df[['fname', col4label]].rename(columns={col4label: 'label'}).to_csv(save_as, index=False)
        
    def add_svm_predictions(self, Y, one_class_svm):
        svm_prediction = {}
        for label in one_class_svm:
            svm_prediction[label] = one_class_svm[label].predict(Y)

        for i in range(len(self)):
            self[i].svm_in_class = []
            for label in svm_prediction:
                if svm_prediction[label][i] == 1:
                    self[i].svm_in_class.append(label)
                    
    def add_novelty_det_sigmoid(self, sigmoid_matrix):
        for i in range(len(self)):
            self[i].novelty_det_sigmoid = sigmoid_matrix.item((i, 0))
            
    def get_list_sigmoid_bw(self, low=0.95, high=0.98):
        list_ = SpeechList(name='filtered by sigmoid')
        for speech in self:
            if (speech.novelty_det_sigmoid >= low) and (speech.novelty_det_sigmoid <= high):
                list_.append(speech)
        return list_        
                
            
        
        
                             