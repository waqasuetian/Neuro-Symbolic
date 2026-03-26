import numpy as np
import pandas as pd
from pyedflib import highlevel
from pathlib import Path
from typing import Tuple
from .preprocessor import EEGPreprocessor
from config.settings import Config
from collections import Counter

class SimpleEEGLoader:
    def __init__(self, config: Config):
        self.config = config
        self.preprocessor = EEGPreprocessor(config)
        self.fs = config.data.fs
        self.n_channels = config.data.n_channels
        self.epoch_len = config.data.epoch_seconds

    
    # Load a single EDF + TSV file pair
    
    def load_pair(self, edf_path: Path, tsv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        # 1. Load EEG 
        signals, _, header = highlevel.read_edf(str(edf_path))
        signals = np.array(signals[:self.n_channels], dtype=np.float32)

        # 2. Preprocess -----------------
        for ch in range(self.n_channels):
            signals[ch] = self.preprocessor.preprocess(signals[ch])

        # 3. Load TSV 
        events = pd.read_csv(tsv_path, sep="\t")

        # Extract clean eventType text (before colon)
        events["eventType"] = events["eventType"].str.split(":").str[0].str.lower()

        # 4. Config-Based Mapping 
        # Example config:
        # event_mapping = {"bckg": 0, "sz_foc_ia": 1}

        mapping = self.config.data.event_mapping 

        # Map eventType → label using config
        events["label"] = events["eventType"].apply(
            lambda x: mapping.get(x, 0)   # default: 0 (background)
        )

        # 5. Epoching 
        samples_per_epoch = self.fs * self.epoch_len
        total_epochs = signals.shape[1] // samples_per_epoch

        X = np.zeros((total_epochs, samples_per_epoch, self.n_channels), dtype=np.float32)
        y = np.zeros(total_epochs, dtype=np.int32)

        for i in range(total_epochs):
            start = i * samples_per_epoch
            end = start + samples_per_epoch

            # EEG segment (time × channels)
            X[i] = signals[:, start:end].T

            # Epoch time in seconds
            epoch_start = i * self.epoch_len
            epoch_end = epoch_start + self.epoch_len

            # Find events overlapping this epoch
            overlaps = events[
                (events["onset"] < epoch_end) &
                ((events["onset"] + events["duration"]) > epoch_start)
            ]

            # Epoch label = max overlapping event label (0 or 1)
            y[i] = overlaps["label"].max() if len(overlaps) > 0 else 0

        return X, y

    
    # Load all EDF+TSV in a folder
    
    def load_all(self, folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        folder_path = Path(folder_path)
        edf_files = folder_path.glob("*_eeg.edf")

        all_X, all_y = [], []

        for edf_path in edf_files:
            tsv_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
            if not tsv_path.exists():
                print(f" Missing TSV for {edf_path.name}, skipping...")
                continue

            X, y = self.load_pair(edf_path, tsv_path)
            all_X.append(X)
            all_y.append(y)

        if len(all_X) == 0:
            return np.array([]), np.array([])

        all_X = np.concatenate(all_X, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        # ----------
        # Print unique classes in the dataset
        # ----------
        unique_classes = np.unique(all_y)
        print(f"Unique classes found in {folder_path}: {unique_classes}, total: {len(unique_classes)}")
        # ---------- Print class counts ----------
        class_counts = Counter(all_y)
        print("Class counts:")
        for cls in sorted(class_counts.keys()):
            print(f" Class {cls}: {class_counts[cls]}")
        return all_X, all_y
