from scipy.io import wavfile
import IPython.display as ipd
import numpy as np
from scipy.signal import spectrogram

from bokeh.plotting import figure, show
from ..bokeh4github import show


class Speech():
    def __init__(self, file_path, is_test=True, label=None):
        self.file_path = file_path
        self.is_test = is_test
        self.label = label
        self.sample_rate = None
        self.data = None
        self.data_len = None
        self.predicted_label = None
        
    def __str__(self):
        output = '{}, {}, sample rate {}, data length {}'.format(self.label, self.file_path, self.sample_rate, self.data_len)
        if self.predicted_label is not None:
            output += '\n'
            output += 'prediction: {}'.format(self.predicted_label)
        return output

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

    def set_spectrogram(self, vector_len, spec_v=None, take_log=False):
        if spec_v is None:
            self.spec_f, self.spec_t, self.spec_data = spectrogram(self.get_data_array_of_length(vector_len), 
                                                                   fs=self.sample_rate)
        elif spec_v == '2':   
            len_one_seq = 390
            len_overlap = 240
            self.spec_f, self.spec_t, self.spec_data = spectrogram(self.get_data_array_of_length(vector_len), 
                                                                   fs=self.sample_rate,
                                                                   window='hann', nperseg=len_one_seq, noverlap=len_overlap,
                                                                   detrend=False, scaling='spectrum')            
        if take_log:
            self.spec_data = np.log(self.spec_data+1e-10)  # 1e-10 to avoid divide by zero
            
            