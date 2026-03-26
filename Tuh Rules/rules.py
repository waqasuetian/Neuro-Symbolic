import os
import numpy as np
import mne
import pandas as pd
from scipy.signal import butter, filtfilt
import csv

# =============================
# USER PARAMETERS
# =============================
EDF_DIR = r"F:\FYP\FYP_Project\data\raw\test"
OUTPUT_CSV = "seizure_rule_based_EVENT_LEVEL_TEST.csv"

SFREQ = 256
WINDOW_SIZE = 4.0
STEP_SIZE = 2.0

MIN_SEIZURE_DURATION = 15.0     # Increased from 10s to 15s
MAX_GAP = 3.0                    # TUH rule
MIN_CONFIDENCE = 0.5             # Only consider windows with >= 0.5 confidence

# =============================
# HELPERS
# =============================
def load_annotations(edf_path):
    ann_file = edf_path.replace(".edf", ".csv")
    annotations = []
    if not os.path.exists(ann_file):
        return annotations

    with open(ann_file, "r") as f:
        reader = csv.reader(f.readlines()[5:])
        for row in reader:
            try:
                _, start, stop, label, *_ = row
                annotations.append({
                    "start": float(start),
                    "stop": float(stop),
                    "label": label.lower()
                })
            except:
                continue
    return annotations

def bandpass(signal, low, high):
    nyq = SFREQ / 2
    b, a = butter(3, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, signal)

# =============================
# FEATURE RULES (FIXED)
# =============================
def detect_spike(epoch):
    ptp = np.ptp(epoch)
    duration = np.sum(np.abs(epoch) > 100e-6) / SFREQ
    return ptp > 120e-6 and 0.02 <= duration <= 0.07

def detect_rhythmic_slow(epoch):
    filtered = bandpass(epoch, 2, 10)
    rms = np.sqrt(np.mean(filtered ** 2))
    return rms > 8e-6

def evolution_score(prev_epochs, curr_epoch):
    if len(prev_epochs) < 3:
        return 0
    prev_rms = np.mean([np.sqrt(np.mean(e ** 2)) for e in prev_epochs])
    curr_rms = np.sqrt(np.mean(curr_epoch ** 2))
    return 1 if curr_rms > prev_rms * 1.2 else 0

def synchrony_score(epoch_data):
    count = 0
    for ch in range(epoch_data.shape[0]):
        if detect_spike(epoch_data[ch]) and detect_rhythmic_slow(epoch_data[ch]):
            count += 1
    return 1 if count >= 3 else 0  # Require 3 channels instead of 2

# =============================
# WINDOW-LEVEL CLASSIFICATION
# =============================
def classify_window(epoch_data, prev_epochs):
    morphology = 0
    rhythmic = 0

    for ch in range(epoch_data.shape[0]):
        if detect_spike(epoch_data[ch]):
            morphology = 1
        if detect_rhythmic_slow(epoch_data[ch]):
            rhythmic = 1

    evolution = evolution_score(prev_epochs, epoch_data.mean(axis=0))
    synchrony = synchrony_score(epoch_data)

    confidence = (
        0.35 * morphology +
        0.25 * rhythmic +
        0.25 * evolution +
        0.15 * synchrony
    )

    is_candidate = confidence >= MIN_CONFIDENCE

    return is_candidate, round(confidence, 2), {
        "morphology": morphology,
        "rhythmicity": rhythmic,
        "evolution": evolution,
        "synchrony": synchrony
    }

# =============================
# EVENT-LEVEL SEIZURE LOGIC
# =============================
def confirm_seizure_events(windows):
    events = []
    current = []

    for w in windows:
        if w["candidate"]:
            if not current:
                current.append(w)
            else:
                gap = w["start"] - current[-1]["end"]
                if gap <= MAX_GAP:
                    current.append(w)
                else:
                    if len(current) >= 2:  # Require at least 2 consecutive windows
                        events.append(current)
                    current = [w]
        else:
            if current and len(current) >= 2:  # Only keep sequences with 2+ windows
                events.append(current)
            current = []

    if current and len(current) >= 2:
        events.append(current)

    confirmed = []
    for ev in events:
        duration = ev[-1]["end"] - ev[0]["start"]
        if duration >= MIN_SEIZURE_DURATION:
            confirmed.append((ev[0]["start"], ev[-1]["end"]))

    return confirmed

# =============================
# PROCESS EDF FILE
# =============================
def process_edf_file(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    data = raw.get_data()
    annotations = load_annotations(edf_path)

    win_samples = int(WINDOW_SIZE * SFREQ)
    step_samples = int(STEP_SIZE * SFREQ)

    windows = []
    prev_epochs = []

    for start in range(0, data.shape[1] - win_samples, step_samples):
        end = start + win_samples
        epoch = data[:, start:end]

        candidate, conf, rules = classify_window(epoch, prev_epochs)

        windows.append({
            "start": start / SFREQ,
            "end": end / SFREQ,
            "candidate": candidate,
            "confidence": conf,
            **rules
        })

        prev_epochs.append(epoch.mean(axis=0))
        if len(prev_epochs) > 5:
            prev_epochs.pop(0)

    seizure_events = confirm_seizure_events(windows)

    results = []
    for w in windows:
        pred_label = "SEIZURE" if any(
            not (w["end"] < s or w["start"] > e)
            for s, e in seizure_events
        ) else "bckg"

        true_label = "bckg"
        for ann in annotations:
            if not (w["end"] < ann["start"] or w["start"] > ann["stop"]):
                true_label = ann["label"]

        results.append({
            "edf_file": os.path.basename(edf_path),
            "segment_start": w["start"],
            "segment_end": w["end"],
            "true_label": true_label,
            "predicted_label": pred_label,
            "confidence": w["confidence"],
            "morphology": w["morphology"],
            "rhythmicity": w["rhythmicity"],
            "evolution": w["evolution"],
            "synchrony": w["synchrony"]
        })

    return results

# =============================
# RUN ALL FILES
# =============================
all_results = []

for root, _, files in os.walk(EDF_DIR):
    for f in files:
        if f.endswith(".edf"):
            path = os.path.join(root, f)
            print("Processing:", path)
            all_results.extend(process_edf_file(path))

df = pd.DataFrame(all_results)
df.to_csv(OUTPUT_CSV, index=False)
print("Saved:", OUTPUT_CSV)
