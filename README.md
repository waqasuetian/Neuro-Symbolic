#  Trustworthy Neuro-Symbolic Framework for Clinically Aligned EEG Seizure Detection

---

##  Description

This project presents a **multi-task neuro-symbolic AI framework** for EEG-based seizure analysis. It integrates deep learning, machine learning, and rule-based clinical reasoning to improve **accuracy, interpretability, and clinical trustworthiness**.

The system is designed to perform three main tasks:

- Seizure detection (binary classification)
- Seizure type classification (multi-class classification)
- Explainable clinical reasoning using symbolic rules

It is evaluated on two EEG datasets:

- TUH EEG Seizure Corpus (v2.0.3)
- BIDS Siena Scalp EEG Dataset

---

##  Dataset Information

### 🔹 TUH EEG Seizure Corpus (v2.0.3)

- Large-scale clinical EEG dataset
- Includes expert annotations (seizure onset, offset, type)
- Widely used benchmark for seizure detection research

### 🔹 BIDS Siena Scalp EEG Dataset

- Standard BIDS-formatted EEG dataset
- High-quality annotated recordings
- Used for cross-dataset validation

### EEG Configuration

- Sampling Frequency: 256 Hz  
- Channels: 19 (10–20 system)  
- Epoch Length: 4 seconds (1024 samples)

---

### Dataset Information

Experiments are conducted on EEG datasets (e.g., TUH EEG dataset).

⚠️ Due to size and licensing constraints, datasets are not included.

Dataset Access: [https://isip.piconepress.com/projects/tuh and https://doi.org/10.13026/5d4a-j060]

Expected Format:

data/
│── subject_01/
│── subject_02/
│── ...

##  Code Information

The repository is structured into dataset-specific and task-specific modules.

### 🔹 TUH Pipeline

- EEG preprocessing and filtering
- Feature extraction from seizure and non-seizure segments
- Detection of seizure
- Rule-based clinical reasoning system

### 🔹 BIDS Pipeline

- BIDS-format preprocessing
- Dataset-specific feature engineering
- Detection of seizure activity
- Rule-based reasoning aligned with annotations

### 🔹 Shared Components

- CNN–LSTM feature extractor for EEG representation
- PCA-based dimensionality reduction
- SVM classifier for classification tasks
- Hybrid fusion of neural and symbolic outputs
- Utility scripts for evaluation and visualization

---

## Tasks

###  Seizure Detection

Binary classification:

- Seizure
- Non-seizure
  Uses CNN–LSTM + SVM pipeline.

---

###  Explainable Neuro-Symbolic Reasoning

A rule-based system for clinical interpretability:

- Spike-and-wave morphology detection
- Rhythmicity analysis
- Channel synchrony evaluation
- Seizure evolution tracking

---

##  Methodology

### 1. Preprocessing

- Bandpass filtering (0.5–40 Hz)
- Notch filtering (50 Hz)
- Z-score normalization

### 2. Segmentation

- EEG divided into 4-second windows
- Ensures consistent temporal input

### 3. Feature Extraction

- CNN–LSTM extracts deep temporal-spatial features
- Statistical features for classical ML models

### 4. Dimensionality Reduction

- PCA reduces feature space for SVM training

### 5. Classification

- SVM performs seizure detection 

### 6. Rule-Based System

- Clinical rules evaluate EEG morphology and dynamics

### 7. Hybrid Fusion

Final prediction:
$$
Final\ Prediction = \alpha \cdot Neural\ Model + (1 - \alpha) \cdot Rule\ Engine
$$

##  Results

| Model Type   | BIDS Accuracy | TUH Accuracy |
| ------------ | ------------- | ------------ |
| Neural Model | 98.75%        | 93.41%       |
| Rule-Based   | 73.8%         | 65.0%        |
| Hybrid Model | 93.84%        | 86.06%       |

---

##  Usage Instructions

###  Clone Repository

```bash
git clone https://github.com/waqasuetian/Neuro-Symbolic.git
cd Neuro-Symbolic
```

---

##  Requirements

### Environment

- Python ≥ 3.8

###  Key Libraries

- mne
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib

---

##  Experimental Results

Performance is evaluated using the following metrics:

- Accuracy  
- Sensitivity (Recall)  
- Specificity  
- F1-score  

⚠️ Large result files are not included in this repository. They are available externally upon request.

---

### Key Insight

This project combines deep learning + machine learning + clinical rule-based reasoning to build a multi-task neuro-symbolic EEG system that is both accurate and interpretable for clinical applications.

---

###  Citation

If you use this work, please cite:

@article{ali2026neurosymbolic, title={A Neuro-Symbolic Framework for Seizure Detection}, author={Waqas Ali}, journal={PeerJ}, year={2026} }

---

### Contact

Waqas Ali Email: [waqasalizafarali@gmail.com]
