from scipy.io import wavfile
import IPython.display as ipd
import numpy as np

from bokeh.plotting import figure, show
from .bokeh4github import show


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
