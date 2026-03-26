import numpy as np
from scipy import signal
from config.settings import Config

class EEGPreprocessor:
    def __init__(self, config: Config):
        self.config = config
        self.fs = self.config.data.fs
        
    def apply_filters(self, data: np.ndarray) -> np.ndarray:
        """Apply notch and bandpass filters"""
        # Notch filter
        b, a = signal.iirnotch(
            self.config.data.notch_freq, 30, self.fs
        )
        filtered = signal.filtfilt(b, a, data)
        
        # Bandpass filter
        nyq = 0.5 * self.fs
        low = self.config.data.lowcut / nyq
        high = self.config.data.highcut / nyq
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, filtered)
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization"""
        return (data - np.mean(data)) / (np.std(data) + 1e-8)
    
    def preprocess(self, eeg_data: np.ndarray) -> np.ndarray:
        """Full preprocessing pipeline"""
        filtered = self.apply_filters(eeg_data)
        return self.normalize(filtered)