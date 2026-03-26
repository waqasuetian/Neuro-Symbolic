import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
import mne
from pyedflib import highlevel
from scipy.signal import butter, filtfilt

# =====================================================
# ================= CONFIG ===========================
# =====================================================

DATA_ROOT = Path(r"D:\1\Code and Ablation Study\hyb_Cnn-Lstm\dataset\test")
MODEL_DIR = Path("results")
OUT_CSV = "train1_SVM_RULE_COMBINED_continuous_test_bids.csv"

FS = 256
N_CHANNELS = 19
EPOCH_SEC = 4  # fixed segment size
SAMPLES_PER_EPOCH = FS * EPOCH_SEC
MIN_CONFIDENCE = 0.5

# =====================================================
# ================= LOAD MODELS ======================
# =====================================================

cnn_feat = tf.keras.models.load_model(MODEL_DIR / "feature_extractor.keras")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
pca = joblib.load(MODEL_DIR / "pca.joblib")
svm = joblib.load(MODEL_DIR / "svm.joblib")

# =====================================================
# ================= HELPERS ==========================
# =====================================================

def normalize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-8)

def bandpass(signal, low, high):
    nyq = FS / 2
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

def load_annotations(edf_path):
    ann_file = edf_path.with_suffix(".csv")
    anns = []
    if not ann_file.exists():
        return anns
    with open(ann_file, "r") as f:
        reader = csv.reader(f.readlines()[5:])
        for r in reader:
            try:
                _, s, e, lbl, *_ = r
                anns.append({"start": float(s), "stop": float(e), "label": lbl.lower()})
            except:
                pass
    return anns

def true_label(start, end, anns):
    for a in anns:
        if max(start, a["start"]) < min(end, a["stop"]):
            return a["label"]
    return "bckg"

# =====================================================
# =============== CNN + SVM PIPELINE =================
# =====================================================

def cnn_svm_segments(edf_path):
    signals, _, _ = highlevel.read_edf(str(edf_path))
    signals = [np.asarray(ch, dtype=np.float32) for ch in signals]

    if len(signals) < N_CHANNELS:
        return None

    signals = signals[:N_CHANNELS]
    min_len = min(len(ch) for ch in signals)
    signals = np.stack([normalize(ch[:min_len]) for ch in signals])

    n_epochs = min_len // SAMPLES_PER_EPOCH
    if n_epochs == 0:
        return None

    X = np.zeros((n_epochs, SAMPLES_PER_EPOCH, N_CHANNELS), dtype=np.float32)
    for i in range(n_epochs):
        s, e = i * SAMPLES_PER_EPOCH, (i + 1) * SAMPLES_PER_EPOCH
        X[i] = signals[:, s:e].T

    feats = cnn_feat.predict(X, verbose=0)
    feats = scaler.transform(feats)
    feats = pca.transform(feats)

    preds = svm.predict(feats)
    probs = svm.predict_proba(feats)[:, 1]

    anns = load_annotations(edf_path)
    rows = []

    for i in range(n_epochs):
        start = i * EPOCH_SEC
        end = start + EPOCH_SEC
        rows.append({
            "segment_start": start,
            "segment_end": end,
            "svm_pred": "Seizure" if preds[i] == 1 else "Non-Seizure",
            "svm_prob": float(probs[i]),
            "true_label": true_label(start, end, anns)
        })

    return pd.DataFrame(rows)

# =====================================================
# ================= CONTINUOUS RULE PIPELINE ==========
# =====================================================

def detect_spike_score(epoch):
    ptp = np.ptp(epoch)
    return min(ptp / 300e-6, 1.0)

def detect_rhythmic_score(epoch):
    filtered = bandpass(epoch, 2, 10)
    rms = np.sqrt(np.mean(filtered**2))
    return min(rms / 20e-6, 1.0)

def evolution_score(prev_epochs, curr_epoch):
    if len(prev_epochs) < 3:
        return 0.0
    prev_rms = np.mean([np.sqrt(np.mean(e**2)) for e in prev_epochs])
    curr_rms = np.sqrt(np.mean(curr_epoch**2))
    ratio = curr_rms / (prev_rms + 1e-8)
    return min(max((ratio - 1.0)/0.2, 0.0), 1.0)

def synchrony_score(epoch_data):
    count = 0
    for ch in range(epoch_data.shape[0]):
        spike = detect_spike_score(epoch_data[ch]) > 0.4
        rhythmic = detect_rhythmic_score(epoch_data[ch]) > 0.4
        if spike and rhythmic:
            count += 1
    return min(count / epoch_data.shape[0], 1.0)

def rule_pipeline_continuous(edf_path, n_epochs):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()[:N_CHANNELS]
    min_len = min(data.shape[1], n_epochs * SAMPLES_PER_EPOCH)
    data = data[:, :min_len]

    windows = []
    prev_epochs = []

    for i in range(n_epochs):
        start = i * SAMPLES_PER_EPOCH
        end = start + SAMPLES_PER_EPOCH
        epoch = data[:, start:end]

        morphology = np.mean([detect_spike_score(epoch[ch]) for ch in range(epoch.shape[0])])
        rhythmicity = np.mean([detect_rhythmic_score(epoch[ch]) for ch in range(epoch.shape[0])])
        evolution = evolution_score(prev_epochs, epoch.mean(axis=0))
        synchrony = synchrony_score(epoch)

        conf = 0.35 * morphology + 0.25 * rhythmicity + 0.25 * evolution + 0.15 * synchrony
        pred = "SEIZURE" if conf >= MIN_CONFIDENCE else "bckg"

        windows.append({
            "rule_pred": pred,
            "rule_conf": round(conf, 3),
            "morphology": round(morphology, 3),
            "rhythmicity": round(rhythmicity, 3),
            "evolution": round(evolution, 3),
            "synchrony": round(synchrony, 3)
        })

        prev_epochs.append(epoch.mean(axis=0))
        if len(prev_epochs) > 5:
            prev_epochs.pop(0)

    return pd.DataFrame(windows)

# =====================================================
# ===================== RUN ALL ======================
# =====================================================

final_rows = []
edf_files = list(DATA_ROOT.rglob("*.edf"))
print(f"Found {len(edf_files)} EDF files")

for edf in edf_files:
    print("Processing:", edf.name)
    try:
        df_svm = cnn_svm_segments(edf)
        if df_svm is None:
            continue
        n_epochs = len(df_svm)
        df_rule = rule_pipeline_continuous(str(edf), n_epochs)

        for i in range(n_epochs):
            s = df_svm.iloc[i]
            r = df_rule.iloc[i]
            final_rows.append({
                "edf_file": edf.name,
                "segment_start": s.segment_start,
                "segment_end": s.segment_end,
                "true_label": s.true_label,
                "svm_pred": s.svm_pred,
                "svm_prob": s.svm_prob,
                "rule_pred": r.rule_pred,
                "rule_conf": r.rule_conf,
                "morphology": r.morphology,
                "rhythmicity": r.rhythmicity,
                "evolution": r.evolution,
                "synchrony": r.synchrony
            })

    except Exception as e:
        print("Skipped:", edf.name, e)

df = pd.DataFrame(final_rows)
df.to_csv(OUT_CSV, index=False)

print("\n DONE")
print("Saved:", OUT_CSV)
print(df.head())
