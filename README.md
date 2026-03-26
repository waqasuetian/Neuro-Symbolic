# Trustworthy Neuro-Symbolic Framework for Clinically Aligned Seizure Monitoring

**A Hybrid Support Vector Machine and Rule-Based Engine for EEG Analysis**

---

## 1. Overview

This repository provides a reproducible implementation of a **neuro-symbolic framework** that integrates **machine learning (Support Vector Machines)** with a **rule-based reasoning system** for seizure detection using EEG signals.

The objective is to enhance:

* Predictive performance
* Interpretability
* Clinical relevance

by combining statistical learning with domain-driven symbolic inference.

---

## 2. Repository Structure

```
Neuro-Symbolic/
│── src/                  # Core implementation
│── rules/                # Rule-based engine definitions
│── models/               # Trained/placeholder models
│── scripts/              # Execution scripts
│── data/                 # (Not included - external)
│── results/              # (Not included - external)
│── requirements.txt
│── README.md
```

---

## 3. Dataset Information

Experiments are conducted on EEG datasets (e.g., TUH EEG dataset).

⚠️ Due to size and licensing constraints, datasets are not included.

**Dataset Access:**
[https://isip.piconepress.com/projects/tuh and https://doi.org/10.13026/5d4a-j060]

**Expected Format:**

```
data/
│── subject_01/
│── subject_02/
│── ...
```

---

## 4. Code Description

### Core Modules

* **Feature Extraction**: Extracts temporal and spectral EEG features
* **SVM Model**: Performs seizure classification
* **Rule-Based Engine**: Applies domain knowledge for event-level refinement
* **Evaluation Module**: Computes performance metrics

---

## 5. Requirements

### Environment

* Python ≥ 3.8

### Dependencies

Install via:

```
pip install -r requirements.txt
```

Key libraries:

* numpy
* pandas
* scikit-learn
* scipy
* matplotlib

---
## System Architecture

<p align="center">
  <img src="figures/architecture.png" width="800"/>
</p>

<p align="center">
  <b>Figure 1:</b> Proposed Neuro-Symbolic Seizure Detection Framework combining SVM and rule-based reasoning.
</p>
## 6. Methodology

The proposed system follows a hybrid pipeline:

1. **Preprocessing**

   * Signal normalization
   * Noise/artifact handling

2. **Segmentation**

   * EEG signals divided into temporal windows

3. **Feature Engineering**

   * Time-domain features
   * Frequency-domain features

4. **Machine Learning**

   * SVM classifier trained on extracted features

5. **Rule-Based Reasoning**

   * Expert-defined rules refine predictions

6. **Decision Fusion**

   * Final output combines statistical and symbolic outputs

---

## 7. Reproducibility Instructions

To reproduce the results:

### Step 1 — Clone Repository

```
git clone https://github.com/waqasuetian/Neuro-Symbolic.git
cd Neuro-Symbolic
```

### Step 2 — Install Dependencies

```
pip install -r requirements.txt
```

### Step 3 — Prepare Dataset

* Download the dataset from above provided link. 

### Step 4 — Run Pipeline

```
python main.py
```

### Step 5 — Evaluate Results

```
python evaluate.py
```

---

## 8. Experimental Results

Performance is evaluated using:

* Accuracy
* Sensitivity (Recall)
* Specificity
* F1-score

⚠️ Large result files are not included; they are available externally.

---

## 9. Algorithms

The repository implements:

* Support Vector Machine (SVM) classifier
* Rule-based inference engine
* Hybrid decision fusion mechanism

---

## 10. Reproducibility Statement

All experiments can be reproduced by:

* Using the same dataset
* Following the pipeline described above
* Maintaining identical preprocessing and parameter settings

---

## 11. Limitations

* Dataset not included due to size constraints
* Performance may vary across datasets
* Rule definitions may require domain adaptation

---

## 12. Citation

If you use this work, please cite:

@article{ali2026neurosymbolic,
title={A Neuro-Symbolic Framework for Seizure Detection},
author={Waqas Ali},
journal={PeerJ},
year={2026}
}

---





---

## 15. Contact

Waqas Ali
Email: [waqasalizafarali@gmail.com]
