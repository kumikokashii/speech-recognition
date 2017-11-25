from scipy.io import wavfile
import IPython.display as ipd
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()
from datetime import datetime
import random

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
        
class SpeechList(list):
    def __init__(self, name):
        super().__init__(self)
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
            if (i > 0) and (i % 50000 == 0):  # Print ever 50k data
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
        