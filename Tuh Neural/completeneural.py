

"""
EEG Seizure Detection Pipeline
CNN-LSTM + PCA + SVM
- EDF loaded per batch
- Channel mismatch skipping
- Short EDF skipping
- TUH EDF + CSV compatible
- Logging + tqdm
- Memory-safe feature extraction
"""

# ===================== IMPORTS =====================
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib

from pyedflib import highlevel

# ===================== CONFIG =====================
class Config:
    class Data:
        fs = 256
        n_channels = 20
        epoch_seconds = 4

    class Train:
        batch_size = 128
        epochs = 1
        seed = 42

    data = Data()
    train = Train()

DATA_ROOT = Path(r"F:\FYP\FYP_Project\data\raw\train1")   
OUT_DIR = Path("outputttt")
OUT_DIR.mkdir(exist_ok=True)

# ===================== LOGGING =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.info

# ===================== LABELS =====================
SEIZURE_LABELS = {'gnsz','tcsz','absz','tnsz','mysz','fnsz','cpsz'}

def label_to_binary(label: str) -> int:
    return 1 if str(label).lower().strip() in SEIZURE_LABELS else 0

# ===================== EDF CACHE =====================
@lru_cache(maxsize=8)
def read_edf_cached(path: Path):
    return highlevel.read_edf(str(path))

# ===================== PREPROCESSOR =====================
class EEGPreprocessor:
    def __init__(self, config: Config):
        self.fs = config.data.fs

    def preprocess(self, signal: np.ndarray) -> np.ndarray:
        signal = signal - np.mean(signal)
        std = np.std(signal) + 1e-8
        return signal / std

# ===================== EEG LOADER =====================
class SimpleEEGLoader:
    def __init__(self, config: Config):
        self.cfg = config
        self.fs = config.data.fs
        self.n_channels = config.data.n_channels
        self.epoch_sec = config.data.epoch_seconds
        self.samples_per_epoch = self.fs * self.epoch_sec
        self.preprocessor = EEGPreprocessor(config)

    def load_pair(
        self, edf_path: Path, csv_path: Path
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:

        try:
            signals, _, _ = read_edf_cached(edf_path)
            signals = np.asarray(signals, dtype=np.float32)

            if signals.shape[0] < self.n_channels:
                log(f"Skipping {edf_path.name}: channel mismatch")
                return None, None

            signals = signals[:self.n_channels]

            for ch in range(self.n_channels):
                signals[ch] = self.preprocessor.preprocess(signals[ch])

            events = pd.read_csv(csv_path, skiprows=5)
            start_col, stop_col = (
                ("start_time","stop_time")
                if "start_time" in events.columns
                else ("start","stop")
            )
            events["label_bin"] = events["label"].apply(label_to_binary)

            total_epochs = signals.shape[1] // self.samples_per_epoch
            if total_epochs == 0:
                log(f"Skipping {edf_path.name}: too short for one epoch")
                return None, None

            X = np.zeros(
                (total_epochs, self.samples_per_epoch, self.n_channels),
                dtype=np.float32
            )
            y = np.zeros(total_epochs, dtype=np.int32)

            for i in range(total_epochs):
                s = i * self.samples_per_epoch
                e = s + self.samples_per_epoch
                X[i] = signals[:, s:e].T

                t0, t1 = i*self.epoch_sec, (i+1)*self.epoch_sec
                overlap = events[
                    (events[start_col] < t1) &
                    (events[stop_col] > t0) &
                    (events["label_bin"] == 1)
                ]
                y[i] = 1 if len(overlap) > 0 else 0

            return X, y

        except Exception as e:
            log(f"Skipping {edf_path.name}: {e}")
            return None, None

    def get_file_list(self, root: Path) -> List[Path]:
        files = [
            p for p in root.rglob("*.edf")
            if p.with_suffix(".csv").exists()
        ]
        log(f"Found {len(files)} EDF+CSV pairs")
        return files

# ===================== LAZY KERAS SEQUENCE =====================
class EEGSequence(tf.keras.utils.Sequence):
    def __init__(self, files, loader, batch_size, shuffle=True):
        self.files = files
        self.loader = loader
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.index = []

        self._build_index()
        self.on_epoch_end()

    def _build_index(self):
        log("Building lazy index...")
        for edf in tqdm(self.files, desc="Indexing"):
            X, y = self.loader.load_pair(edf, edf.with_suffix(".csv"))
            if X is None:
                continue
            for i in range(len(y)):
                self.index.append((edf, i))

        if len(self.index) == 0:
            raise RuntimeError("No valid EEG epochs found")

    def __len__(self):
        return max(1, len(self.index) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index)

    def __getitem__(self, idx):
        batch = self.index[idx*self.batch_size:(idx+1)*self.batch_size]
        Xb, yb = [], []

        for edf, ep in batch:
            X, y = self.loader.load_pair(edf, edf.with_suffix(".csv"))
            if X is None:
                continue
            Xb.append(X[ep])
            yb.append(y[ep])

        return np.array(Xb), np.array(yb)

# ===================== MODEL =====================
def build_cnn_lstm(input_shape):
    model = models.Sequential([
        layers.Input(input_shape),
        layers.Conv1D(32, 7, activation="relu", padding="same"),
        layers.MaxPool1D(4),
        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.MaxPool1D(4),
        layers.LSTM(128),
        layers.Dense(128, activation="relu", name="deep_features"),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ===================== MAIN =====================
def main():
    cfg = Config()
    loader = SimpleEEGLoader(cfg)

    files = loader.get_file_list(DATA_ROOT)
    train_f, val_f = train_test_split(
        files, test_size=0.3, random_state=cfg.train.seed
    )

    train_seq = EEGSequence(train_f, loader, cfg.train.batch_size)
    val_seq = EEGSequence(val_f, loader, cfg.train.batch_size, shuffle=False)

    # Build model
    X0, _ = train_seq[0]
    model = build_cnn_lstm(X0.shape[1:])

    log("Training CNN-LSTM...")
    model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=cfg.train.epochs
    )

    # Feature extractor
    feat_model = models.Model(
        inputs=model.inputs,  # use proper Keras symbolic tensor
        outputs=model.get_layer("deep_features").output
    )

    # ================= FEATURE EXTRACTION =================
    Xf_list, yf_list = [], []

    log("Extracting features batch-wise...")
    for i in tqdm(range(len(train_seq)), desc="Feature extraction"):
        X_batch, y_batch = train_seq[i]
        if len(X_batch) == 0:
            continue
        feats = feat_model.predict(X_batch, verbose=0)
        Xf_list.append(feats)
        yf_list.append(y_batch)

    Xf = np.vstack(Xf_list)
    yf = np.hstack(yf_list)

    # ================= PCA + SVM =================
    scaler = StandardScaler()
    Xf = scaler.fit_transform(Xf)

    pca = PCA(n_components=64)
    Xf = pca.fit_transform(Xf)

    svm = SVC(kernel="rbf", probability=True)
    svm.fit(Xf, yf)

    # ================= SAVE MODELS =================
    log("Saving models...")
    model.save(OUT_DIR / "cnn_lstm.keras")
    feat_model.save(OUT_DIR / "feature_extractor.keras")
    joblib.dump(scaler, OUT_DIR / "scaler.joblib")
    joblib.dump(pca, OUT_DIR / "pca.joblib")
    joblib.dump(svm, OUT_DIR / "svm.joblib")

    log(" Pipeline completed successfully")

# ===================== RUN =====================
if __name__ == "__main__":
    main()
